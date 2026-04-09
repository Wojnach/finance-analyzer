"""Sentiment analysis — multi-model A/B testing with FinGPT, CryptoBERT, and TradingHero.

Runs multiple sentiment models in shadow mode and logs results for comparison.
The primary (voting) model is the legacy model; shadow models are logged to
data/sentiment_ab_log.jsonl for accuracy tracking. Once a shadow model proves
superior (>60% on 200+ samples), it can be promoted.

Models:
  - CryptoBERT: crypto headlines (legacy primary)
  - Trading-Hero-LLM: stock headlines (legacy primary)
  - FinGPT (Finance-Llama-8B / FinGPT-MT): GGUF via llama-cpp-python (shadow)
  - FinBERT: CPU fallback (shadow)

Phase 3B: Cumulative headline clustering — groups related headlines and scores
them as a batch for richer "drumbeat effect" detection.
"""

import atexit
import json
import logging
import platform
import queue
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_append_jsonl
from portfolio.http_retry import fetch_json

logger = logging.getLogger("portfolio.sentiment")

CRYPTO_TICKERS = {"BTC", "ETH"}

if platform.system() == "Windows":
    MODELS_PYTHON = r"Q:\finance-analyzer\.venv\Scripts\python.exe"
    CRYPTOBERT_SCRIPT = r"Q:\models\cryptobert_infer.py"
    TRADING_HERO_SCRIPT = r"Q:\models\trading_hero_infer.py"
    FINBERT_SCRIPT = r"Q:\models\finbert_infer.py"
    FINGPT_PYTHON = r"Q:\models\.venv-llm\Scripts\python.exe"
    FINGPT_SCRIPT = r"Q:\models\fingpt_infer.py"
else:
    MODELS_PYTHON = "/home/deck/models/.venv/bin/python"
    CRYPTOBERT_SCRIPT = "/home/deck/models/cryptobert_infer.py"
    TRADING_HERO_SCRIPT = "/home/deck/models/trading_hero_infer.py"
    FINBERT_SCRIPT = "/home/deck/models/finbert_infer.py"
    FINGPT_PYTHON = "/home/deck/models/.venv-llm/bin/python"
    FINGPT_SCRIPT = "/home/deck/models/fingpt_infer.py"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AB_LOG_FILE = DATA_DIR / "sentiment_ab_log.jsonl"

TICKER_CATEGORIES = {
    "BTC": "BTC",
    "ETH": "ETH",
    "XAU": "GOLD",
    "XAG": "SILVER",
    "PLTR": "TECHNOLOGY",
    "NVDA": "TECHNOLOGY",
    "AMD": "TECHNOLOGY",
    "GOOGL": "TECHNOLOGY",
    "AMZN": "TECHNOLOGY",
    "AAPL": "TECHNOLOGY",
    "AVGO": "TECHNOLOGY",
    "META": "TECHNOLOGY",
    "MU": "TECHNOLOGY",
    "SOUN": "TECHNOLOGY",
    "SMCI": "TECHNOLOGY",
    "TSM": "TECHNOLOGY",
    "TTWO": "TECHNOLOGY",
    "VRT": "TECHNOLOGY",
    "LMT": "TECHNOLOGY",
    "MSTR": "TECHNOLOGY",
}

CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"


def _is_crypto(ticker):
    return ticker.upper().replace("-USD", "") in CRYPTO_TICKERS


def _fetch_crypto_headlines(ticker="BTC", limit=20, *, cryptocompare_api_key=None):
    category = TICKER_CATEGORIES.get(ticker.upper(), ticker.upper())
    url = f"{CRYPTOCOMPARE_URL}&categories={category}"
    headers = {"User-Agent": "Mozilla/5.0"}
    if cryptocompare_api_key:
        headers["Authorization"] = f"Apikey {cryptocompare_api_key}"
    data = fetch_json(
        url,
        headers=headers,
        timeout=15,
        label="crypto_headlines",
    )
    if data is None:
        return _fetch_crypto_headlines_yahoo_fallback(ticker, limit)
    if isinstance(data, dict) and data.get("Response") == "Error":
        logger.warning("[CryptoCompare] API error: %s", data.get("Message", "unknown"))
        return _fetch_crypto_headlines_yahoo_fallback(ticker, limit)
    raw = data.get("Data", [])
    articles = list(raw)[:limit] if isinstance(raw, list) else []
    parsed = [
        {
            "title": a["title"],
            "source": a.get("source", "unknown"),
            "published": datetime.fromtimestamp(
                a["published_on"], tz=UTC
            ).isoformat(),
        }
        for a in articles
    ]
    if not parsed:
        return _fetch_crypto_headlines_yahoo_fallback(ticker, limit)
    return parsed


# Mapping from short crypto ticker to yfinance symbol for fallback
_CRYPTO_YFINANCE_MAP = {"BTC": "BTC-USD", "ETH": "ETH-USD"}


def _fetch_crypto_headlines_yahoo_fallback(ticker, limit=20):
    """Fallback: fetch crypto headlines via yfinance when CryptoCompare fails."""
    yf_symbol = _CRYPTO_YFINANCE_MAP.get(ticker.upper())
    if not yf_symbol:
        return []
    try:
        articles = _fetch_yahoo_headlines(yf_symbol, limit=limit)
        if articles:
            logger.info("[CryptoCompare] fallback to Yahoo Finance for %s: %d articles",
                        ticker, len(articles))
        return articles
    except Exception as e:
        logger.debug("[Yahoo News] crypto fallback error for %s: %s", ticker, e)
        return []


def _fetch_yahoo_headlines(ticker, limit=10):
    import yfinance as yf

    stock = yf.Ticker(ticker)
    news = stock.news or []
    articles = []
    for item in news[:limit]:
        content = item.get("content", item)
        title = content.get("title", "")
        if not title:
            continue
        pub = content.get("pubDate") or content.get("displayTime", "")
        provider = content.get("provider", {})
        source = (
            provider.get("displayName", "Yahoo Finance")
            if isinstance(provider, dict)
            else "Yahoo Finance"
        )
        articles.append(
            {
                "title": title,
                "source": source,
                "published": pub or datetime.now(UTC).isoformat(),
            }
        )
    return articles


def _fetch_newsapi_headlines(ticker, api_key, limit=10, query=None):
    """Fetch headlines from NewsAPI with optional custom search query."""
    search_q = query or ticker
    data = fetch_json(
        "https://newsapi.org/v2/everything",
        params={"q": search_q, "language": "en", "sortBy": "publishedAt",
                "pageSize": limit},
        headers={"User-Agent": "Mozilla/5.0", "X-Api-Key": api_key},
        timeout=15,
        label=f"newsapi:{ticker}",
    )
    if data is None:
        return []
    articles = data.get("articles", [])
    return [
        {
            "title": a.get("title", ""),
            "source": a.get("source", {}).get("name", "unknown"),
            "published": a.get("publishedAt", datetime.now(UTC).isoformat()),
        }
        for a in articles
        if a.get("title")
    ]


def _fetch_newsapi_with_tracking(ticker, api_key, limit=10, query=None):
    """Fetch from NewsAPI and track the call against daily quota.

    H9/DC-R3-2: only count against budget when the fetch actually returned data
    (not on empty responses or errors), preventing spurious budget exhaustion.
    """
    from portfolio.shared_state import newsapi_track_call
    result = _fetch_newsapi_headlines(ticker, api_key, limit=limit, query=query)
    if result:  # only count against budget when we actually got data
        newsapi_track_call()
    return result


def _fetch_stock_headlines(ticker, newsapi_key=None, limit=20):
    """Fetch stock headlines. NewsAPI for priority tickers (metals), Yahoo for the rest."""
    from portfolio.shared_state import (
        _cached,
        newsapi_quota_ok,
        newsapi_search_query,
        newsapi_ttl_for_ticker,
    )

    articles = []

    # NewsAPI: only for priority tickers during active hours (metals get 20-min refresh)
    ttl = newsapi_ttl_for_ticker(ticker) if newsapi_key else None
    if ttl is not None and newsapi_key and newsapi_quota_ok():
        try:
            query = newsapi_search_query(ticker)
            cached_newsapi = _cached(
                f"newsapi_{ticker}",
                ttl,
                _fetch_newsapi_with_tracking,
                ticker,
                newsapi_key,
                limit,
                query,
            )
            if cached_newsapi:
                articles.extend(cached_newsapi)
        except Exception as e:
            logger.debug("[NewsAPI] error for %s: %s", ticker, e)

    # Yahoo Finance: fallback for metals, primary for everything else
    if len(articles) < limit:
        try:
            remaining = max(0, limit - len(articles))
            if remaining > 0:
                yahoo_articles = _fetch_yahoo_headlines(ticker, limit=remaining)
                seen_titles = {a.get("title", "").lower() for a in articles}
                for ya in yahoo_articles:
                    if ya.get("title", "").lower() not in seen_titles:
                        articles.append(ya)
                        seen_titles.add(ya.get("title", "").lower())
        except Exception as e:
            logger.debug("[Yahoo News] error for %s: %s", ticker, e)

    newsapi_count = len([a for a in articles if a.get("source", "") != "Yahoo Finance"])
    yahoo_count = len(articles) - newsapi_count
    if articles:
        logger.debug("[Headlines %s] %d NewsAPI + %d Yahoo = %d total",
                     ticker, newsapi_count, yahoo_count, len(articles))

    return articles[:limit]


def _run_model(script, texts):
    """Run a sentiment model via subprocess (legacy pattern)."""
    proc = subprocess.run(
        [MODELS_PYTHON, script],
        input=json.dumps(texts),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Model failed: {proc.stderr}")
    return json.loads(proc.stdout)


# ── Warm FinGPT daemon client ──────────────────────────────────────────────
# Rather than spawning a fresh subprocess per call (which cold-loads the
# 1.2 GB GGUF model every time and holds the GPU file-lock for 70-90s), we
# talk to a single long-lived daemon process that loads the model once.
# Protocol is NDJSON over stdin/stdout; see scripts/fingpt_daemon.py.
#
# Module-level singleton with a thread-lock because get_sentiment() is called
# from ThreadPoolExecutor workers in main.py. The daemon itself processes one
# request at a time, so the lock is not a throughput regression — it's the
# same serialization the GPU lock already imposed, just with ~1-3s hold
# instead of ~70-90s per call.

_FINGPT_DAEMON_SCRIPT = str(
    Path(__file__).resolve().parent.parent / "scripts" / "fingpt_daemon.py"
)
_fingpt_daemon_proc: subprocess.Popen | None = None
_fingpt_daemon_reader: "_DaemonReader | None" = None
_fingpt_daemon_lock = threading.Lock()
_fingpt_request_id = 0
_FINGPT_READY_TIMEOUT_S = 180  # model warm-load + startup
_FINGPT_REQUEST_TIMEOUT_S = 60  # per-request inference ceiling


class _DaemonReader(threading.Thread):
    """Background reader that pipes daemon stdout lines into a queue.

    Without this, a hung model would leave the main thread blocked inside
    ``stdout.readline()`` while holding ``_fingpt_daemon_lock``, which would
    starve every ThreadPoolExecutor worker that tries to compute sentiment.
    Using a bounded ``queue.get(timeout=...)`` lets the client detect the
    hang and kill the daemon.

    One None sentinel is pushed on EOF/exception; all subsequent reads will
    return None immediately (the sentinel is re-enqueued on each read so the
    queue stays non-empty).
    """

    def __init__(self, proc: subprocess.Popen):
        super().__init__(daemon=True, name="fingpt-daemon-reader")
        self._proc = proc
        self._q: "queue.Queue[str | None]" = queue.Queue()

    def run(self) -> None:
        try:
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    break
                self._q.put(line)
        except Exception as exc:
            logger.debug("Fingpt daemon reader exception: %s", exc)
        finally:
            self._q.put(None)

    def readline(self, timeout: float) -> str | None:
        """Return the next stdout line, or None on timeout / EOF.

        Once EOF is observed, subsequent calls return None immediately
        because we re-enqueue the sentinel for the next reader.
        """
        try:
            line = self._q.get(timeout=timeout)
        except queue.Empty:
            return None
        if line is None:
            # Re-post sentinel so subsequent reads also observe EOF cheaply.
            self._q.put(None)
            return None
        return line


def _spawn_fingpt_daemon() -> tuple[subprocess.Popen, "_DaemonReader"]:
    """Launch the daemon subprocess and wait for its 'ready' handshake.

    Caller must hold ``_fingpt_daemon_lock``. Returns ``(proc, reader)``.
    The reader is a background thread pumping stdout lines into a bounded
    queue, which is how every subsequent read enforces a timeout.
    """
    logger.info("Spawning FinGPT warm daemon: %s", _FINGPT_DAEMON_SCRIPT)
    proc = subprocess.Popen(
        [FINGPT_PYTHON, _FINGPT_DAEMON_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,  # pass through to parent stderr so we see warm-load logs
        text=True,
        bufsize=1,  # line-buffered
    )
    reader = _DaemonReader(proc)
    reader.start()

    # Wait for the "ready" handshake line. The daemon loads the model on
    # startup (~30-60s cold) so this must be generous — bounded by
    # _FINGPT_READY_TIMEOUT_S (180s).
    ready_line = reader.readline(timeout=_FINGPT_READY_TIMEOUT_S)
    if ready_line is None:
        # Timeout or EOF before handshake arrived — the daemon is dead or hung.
        try:
            proc.kill()
        except Exception:
            pass
        raise RuntimeError(
            "FinGPT daemon did not emit ready handshake within "
            f"{_FINGPT_READY_TIMEOUT_S}s"
        )
    try:
        handshake = json.loads(ready_line)
    except json.JSONDecodeError as exc:
        proc.kill()
        raise RuntimeError(
            f"FinGPT daemon emitted non-JSON handshake: {ready_line!r}"
        ) from exc
    if not handshake.get("ready"):
        proc.kill()
        raise RuntimeError(f"FinGPT daemon failed to warm up: {handshake}")
    logger.info("FinGPT warm daemon ready (model=%s, pid=%d)",
                handshake.get("model"), proc.pid)
    return proc, reader


def _ensure_fingpt_daemon() -> tuple[subprocess.Popen, "_DaemonReader"]:
    """Return the live daemon process + reader, spawning it lazily and
    restarting both if the process has died. Caller must hold
    ``_fingpt_daemon_lock``."""
    global _fingpt_daemon_proc, _fingpt_daemon_reader
    if _fingpt_daemon_proc is not None and _fingpt_daemon_proc.poll() is None:
        return _fingpt_daemon_proc, _fingpt_daemon_reader  # type: ignore[return-value]
    if _fingpt_daemon_proc is not None:
        logger.warning("FinGPT daemon exited (code=%s), restarting",
                       _fingpt_daemon_proc.returncode)
        _fingpt_daemon_proc = None
        _fingpt_daemon_reader = None
    _fingpt_daemon_proc, _fingpt_daemon_reader = _spawn_fingpt_daemon()
    return _fingpt_daemon_proc, _fingpt_daemon_reader


def _stop_fingpt_daemon() -> None:
    """Graceful shutdown hook — closes stdin and waits briefly for exit."""
    global _fingpt_daemon_proc, _fingpt_daemon_reader
    with _fingpt_daemon_lock:
        if _fingpt_daemon_proc is None or _fingpt_daemon_proc.poll() is not None:
            _fingpt_daemon_proc = None
            _fingpt_daemon_reader = None
            return
        try:
            _fingpt_daemon_proc.stdin.write(json.dumps({"quit": True}) + "\n")
            _fingpt_daemon_proc.stdin.flush()
            _fingpt_daemon_proc.stdin.close()
        except Exception:
            pass
        try:
            _fingpt_daemon_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _fingpt_daemon_proc.kill()
        _fingpt_daemon_proc = None
        _fingpt_daemon_reader = None


atexit.register(_stop_fingpt_daemon)


def _run_fingpt(texts, model_name=None, cumulative=False, ticker="unknown"):
    """Run FinGPT sentiment inference via the warm daemon.

    Model is loaded once at daemon spawn; subsequent calls incur only the
    inference cost (~1-3s per batch). On daemon crash or timeout, the client
    retries once with a fresh daemon before giving up (the caller has a
    FinBERT fallback path that handles a second failure gracefully).

    Args:
        texts: List of headline strings
        model_name: Ignored in the daemon client — daemon picks the first
            available model at startup. Kept in the signature for backward
            compat with existing call sites.
        cumulative: If True, run cumulative analysis instead of per-headline
        ticker: Ticker for cumulative context

    Returns:
        List of sentiment result dicts (per-headline) or single dict (cumulative)
    """
    global _fingpt_request_id, _fingpt_daemon_proc, _fingpt_daemon_reader
    # model_name is intentionally unused — daemon auto-detects
    _ = model_name

    req_body = {
        "mode": "cumulative" if cumulative else "headlines",
        "texts": texts,
        "ticker": ticker,
    }

    for attempt in (1, 2):
        with _fingpt_daemon_lock:
            try:
                proc, reader = _ensure_fingpt_daemon()
                _fingpt_request_id += 1
                sent_req_id = _fingpt_request_id
                req_body["request_id"] = sent_req_id
                proc.stdin.write(json.dumps(req_body) + "\n")
                proc.stdin.flush()
                resp_line = reader.readline(timeout=_FINGPT_REQUEST_TIMEOUT_S)
                if resp_line is None:
                    raise RuntimeError(
                        f"FinGPT daemon timeout/EOF after "
                        f"{_FINGPT_REQUEST_TIMEOUT_S}s (req_id={sent_req_id})"
                    )
                resp = json.loads(resp_line)
                if "error" in resp:
                    raise RuntimeError(f"FinGPT daemon error: {resp['error']}")
                # Validate the echo matches — guards against any protocol
                # desync (daemon emitting a prior response, a stray log line,
                # etc.). Missing request_id in response is tolerated for
                # backward compat with daemon versions that don't echo it.
                echoed = resp.get("request_id")
                if echoed is not None and echoed != sent_req_id:
                    raise RuntimeError(
                        f"FinGPT daemon response request_id mismatch: "
                        f"sent {sent_req_id}, got {echoed}"
                    )
                return resp["result"]
            except (BrokenPipeError, RuntimeError, json.JSONDecodeError) as exc:
                logger.warning(
                    "FinGPT daemon request failed (attempt %d): %s", attempt, exc
                )
                # Mark the daemon dead so the next attempt respawns it.
                if _fingpt_daemon_proc is not None:
                    try:
                        _fingpt_daemon_proc.kill()
                    except Exception:
                        pass
                _fingpt_daemon_proc = None
                _fingpt_daemon_reader = None
                if attempt == 2:
                    raise RuntimeError(f"FinGPT failed after retry: {exc}") from exc
    raise RuntimeError("FinGPT unreachable")  # unreachable, satisfies type checker


def _run_finbert(texts):
    """Run FinBERT sentiment inference (CPU fallback)."""
    proc = subprocess.run(
        [MODELS_PYTHON, FINBERT_SCRIPT],
        input=json.dumps(texts),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"FinBERT failed: {proc.stderr[:200]}")
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# Headline clustering (Phase 3B)
# ---------------------------------------------------------------------------

def _cluster_headlines(articles):
    """Group headlines by keyword overlap and time proximity.

    Clusters enable cumulative sentiment analysis — many mildly negative
    headlines about the same topic together signal stronger negativity
    than scoring each independently.

    Returns:
        List of clusters, each a list of article dicts
    """
    if not articles or len(articles) < 3:
        return [articles] if articles else []

    from portfolio.news_keywords import score_headline

    # Extract keywords per headline
    headline_keywords = []
    for a in articles:
        _, matched = score_headline(a.get("title", ""))
        # Also extract significant words (>4 chars, not stopwords)
        words = set()
        for word in a.get("title", "").lower().split():
            clean = word.strip(".,!?;:'\"()[]")
            if len(clean) > 4 and clean not in _STOPWORDS:
                words.add(clean)
        words.update(kw.lower() for kw in matched)
        headline_keywords.append(words)

    # Simple greedy clustering by keyword overlap
    clusters = []
    assigned = set()

    for i in range(len(articles)):
        if i in assigned:
            continue
        cluster = [articles[i]]
        assigned.add(i)
        kw_i = headline_keywords[i]

        for j in range(i + 1, len(articles)):
            if j in assigned:
                continue
            kw_j = headline_keywords[j]
            overlap = len(kw_i & kw_j)
            # Cluster if they share 2+ keywords or 1 matched keyword
            if overlap >= 2 or (overlap >= 1 and kw_i & kw_j & _SIGNIFICANT_KEYWORDS):
                cluster.append(articles[j])
                assigned.add(j)

        clusters.append(cluster)

    return clusters


_STOPWORDS = {
    "about", "after", "again", "being", "between", "could", "during",
    "every", "first", "their", "there", "these", "those", "under",
    "which", "while", "would", "other", "still", "where", "before",
    "should", "since", "until", "years", "might", "price", "stock",
    "market", "shares", "today", "report", "quarter",
}

_SIGNIFICANT_KEYWORDS = {
    "tariff", "tariffs", "war", "crash", "sanctions", "hack", "recession",
    "inflation", "rate", "cut", "hike", "layoffs", "earnings", "fomc",
    "bitcoin", "ethereum", "crypto", "nvidia", "semiconductor",
}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_sentiments(sentiments, headlines=None, dissemination_mult=1.0):
    """Aggregate sentiment scores, weighted by headline keywords and dissemination.

    When headlines are provided, each sentiment score is multiplied by the
    keyword weight from news_keywords.score_headline(). High-impact keywords
    (tariff, war, crash) get 2-3x amplification.

    dissemination_mult applies the FinGPT dissemination-aware multiplier
    to all weights (wider news spread = stronger signal).
    """
    if headlines and len(headlines) == len(sentiments):
        from portfolio.news_keywords import score_headline
        weights = []
        for h in headlines:
            title = h.get("title", "") if isinstance(h, dict) else str(h)
            w, _ = score_headline(title)
            weights.append(w * dissemination_mult)
    else:
        weights = [dissemination_mult] * len(sentiments)

    pos_sum = sum(s["scores"]["positive"] * w for s, w in zip(sentiments, weights))
    neg_sum = sum(s["scores"]["negative"] * w for s, w in zip(sentiments, weights))
    neu_sum = sum(s["scores"]["neutral"] * w for s, w in zip(sentiments, weights))
    total_w = sum(weights)
    if total_w == 0:
        return "neutral", {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
    avg = {"positive": pos_sum / total_w, "negative": neg_sum / total_w, "neutral": neu_sum / total_w}
    overall = max(avg, key=avg.get)
    return overall, avg


def _log_ab_result(ticker, primary_result, shadow_results):
    """Log A/B test results to sentiment_ab_log.jsonl for accuracy comparison."""
    try:
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "ticker": ticker,
            "primary": {
                "model": primary_result.get("model", "unknown"),
                "sentiment": primary_result.get("overall_sentiment", "unknown"),
                "confidence": primary_result.get("confidence", 0.0),
            },
            "shadow": shadow_results,
        }
        atomic_append_jsonl(AB_LOG_FILE, entry)
    except Exception:
        logger.debug("Failed to log A/B result", exc_info=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_sentiment(ticker="BTC", newsapi_key=None, social_posts=None,
                   *, cryptocompare_api_key=None) -> dict:
    """Get sentiment for a ticker using primary model + shadow A/B models.

    Primary model (votes): CryptoBERT (crypto) or Trading-Hero-LLM (stocks)
    Shadow models (logged only): FinGPT, FinBERT

    Returns the primary model's result. Shadow results are logged to
    data/sentiment_ab_log.jsonl for accuracy tracking.
    """
    short = ticker.upper().replace("-USD", "")
    is_crypto = _is_crypto(short)

    if is_crypto:
        articles = _fetch_crypto_headlines(
            short, cryptocompare_api_key=cryptocompare_api_key,
        )
        model_script = CRYPTOBERT_SCRIPT
        model_name = "CryptoBERT"
    else:
        articles = _fetch_stock_headlines(short, newsapi_key=newsapi_key)
        model_script = TRADING_HERO_SCRIPT
        model_name = "Trading-Hero-LLM"

    social = social_posts or []
    all_articles = articles + social
    sources = {
        "news": len(articles),
        "reddit": sum(1 for p in social if "reddit" in p.get("source", "")),
    }

    if not all_articles:
        return {
            "overall_sentiment": "unknown",
            "confidence": 0.0,
            "num_articles": 0,
            "model": model_name,
            "sources": sources,
            "details": [],
        }

    titles = [a["title"] for a in all_articles]

    # Compute dissemination score for weight amplification
    diss_mult = 1.0
    try:
        from portfolio.news_keywords import dissemination_score
        diss_mult = dissemination_score(all_articles)
    except Exception:
        logger.debug("Dissemination score failed, using default 1.0", exc_info=True)

    # --- Primary model (votes in consensus) ---
    sentiments = _run_model(model_script, titles)
    overall, avg = _aggregate_sentiments(sentiments, headlines=all_articles,
                                         dissemination_mult=diss_mult)

    details = []
    for article, sent in zip(all_articles, sentiments):
        details.append(
            {
                "title": article["title"],
                "source": article["source"],
                "published": article["published"],
                "sentiment": sent.get("sentiment") or sent.get("label", "unknown"),
                "confidence": sent["confidence"],
                "scores": sent["scores"],
            }
        )

    primary_result = {
        "overall_sentiment": overall,
        "confidence": round(avg[overall], 4),
        "num_articles": len(sentiments),
        "avg_scores": {k: round(v, 4) for k, v in avg.items()},
        "model": model_name,
        "sources": sources,
        "details": details,
        "dissemination_score": diss_mult,
    }

    # --- Shadow models (A/B testing — logged only, don't affect consensus) ---
    shadow_results = []

    # Shadow: FinGPT (per-headline)
    try:
        fingpt_results = _run_fingpt(titles)
        if fingpt_results:
            fg_overall, fg_avg = _aggregate_sentiments(
                fingpt_results, headlines=all_articles, dissemination_mult=diss_mult
            )
            shadow_results.append({
                "model": fingpt_results[0].get("model", "fingpt") if fingpt_results else "fingpt",
                "sentiment": fg_overall,
                "confidence": round(fg_avg[fg_overall], 4),
                "avg_scores": {k: round(v, 4) for k, v in fg_avg.items()},
            })
    except Exception as e:
        logger.debug("FinGPT shadow failed: %s", e)

    # Shadow: FinGPT cumulative (clustered headlines)
    try:
        clusters = _cluster_headlines(all_articles)
        for cluster in clusters:
            if len(cluster) >= 3:
                cluster_titles = [a["title"] for a in cluster]
                cum_result = _run_fingpt(
                    cluster_titles, cumulative=True, ticker=short
                )
                if cum_result and isinstance(cum_result, dict):
                    shadow_results.append({
                        "model": "fingpt:cumulative",
                        "sentiment": cum_result.get("sentiment", "neutral"),
                        "confidence": cum_result.get("confidence", 0.0),
                        "headline_count": len(cluster),
                    })
    except Exception as e:
        logger.debug("FinGPT cumulative shadow failed: %s", e)

    # Shadow: FinBERT (CPU, fast)
    try:
        finbert_results = _run_finbert(titles)
        if finbert_results:
            fb_overall, fb_avg = _aggregate_sentiments(
                finbert_results, headlines=all_articles, dissemination_mult=diss_mult
            )
            shadow_results.append({
                "model": "FinBERT",
                "sentiment": fb_overall,
                "confidence": round(fb_avg[fb_overall], 4),
                "avg_scores": {k: round(v, 4) for k, v in fb_avg.items()},
            })
    except Exception as e:
        logger.debug("FinBERT shadow failed: %s", e)

    # Log A/B comparison
    if shadow_results:
        _log_ab_result(short, primary_result, shadow_results)

    return primary_result


def get_crypto_sentiment(ticker="BTC") -> dict:
    return get_sentiment(ticker)


if __name__ == "__main__":
    for ticker in ["BTC", "ETH"]:
        print(f"\n{'='*60}")
        print(f"  Sentiment for {ticker}")
        print(f"{'='*60}")
        result = get_sentiment(ticker)
        print(
            f"Overall: {result['overall_sentiment']} "
            f"(confidence: {result['confidence']:.2%})"
        )
        print(f"Model: {result['model']}")
        print(f"Articles analyzed: {result['num_articles']}")
        print(f"Dissemination score: {result.get('dissemination_score', 1.0):.2f}")
        if result.get("avg_scores"):
            s = result["avg_scores"]
            print(
                f"Avg scores: pos={s['positive']:.3f} neg={s['negative']:.3f} neu={s['neutral']:.3f}"
            )
        print("\nTop headlines:")
        for d in result["details"][:5]:
            emoji = {"positive": "+", "negative": "-", "neutral": "~"}.get(
                d["sentiment"], "?"
            )
            print(
                f"  [{emoji}] {d['sentiment']:>8} ({d['confidence']:.2%}) {d['title']}"
            )
