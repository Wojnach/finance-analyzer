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

import json
import logging
import platform
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
else:
    MODELS_PYTHON = "/home/deck/models/.venv/bin/python"
    CRYPTOBERT_SCRIPT = "/home/deck/models/cryptobert_infer.py"
    TRADING_HERO_SCRIPT = "/home/deck/models/trading_hero_infer.py"
    FINBERT_SCRIPT = "/home/deck/models/finbert_infer.py"

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


# ── Deferred fingpt A/B buffering ──────────────────────────────────────────
# Fingpt is a SHADOW sentiment signal — it never votes. Its output lands in
# data/sentiment_ab_log.jsonl alongside the primary model's vote (CryptoBERT
# for crypto, Trading-Hero-LLM for stocks) for accuracy comparison.
#
# Historical note (2026-04-09, feat/fingpt-in-llmbatch):
# - v1: inline subprocess per call, cold-loading the GGUF every time (70-90s
#   GPU lock holds, broke the cycle budget)
# - v2: warm NDJSON daemon (scripts/fingpt_daemon.py) on GPU full offload
#   (OOM'd with llama-server also resident)
# - v3: warm NDJSON daemon on CPU (60-150s/cycle inference, forced
#   _FINGPT_REQUEST_TIMEOUT_S 60→180 and _TICKER_POOL_TIMEOUT 120→500 hotfix
#   bumps; worked but ugly)
# - v4 (current): fingpt runs in portfolio.llm_batch as Phase 3 of the
#   shared llama_server rotation on port 8787, with full -ngl 99 GPU
#   offload like ministral3 and qwen3. Retires ~250 LOC of daemon +
#   client code.
#
# Because fingpt now runs post-cycle in a batched phase, its results arrive
# AFTER get_sentiment() has already returned to the signal engine. The
# primary model and FinBERT shadow are still computed inline, but their A/B
# log write is DEFERRED: get_sentiment() stashes the primary + finbert shadow
# + the raw headlines + the enqueued fingpt sub_keys into
# _pending_ab_entries[ab_key] and returns. Once flush_llm_batch() completes
# in main.py, sentiment.flush_ab_log() walks the pending entries, merges the
# batched fingpt results into each, and writes the final A/B log rows.
#
# This preserves the EXACT schema of sentiment_ab_log.jsonl that downstream
# accuracy tracking consumes: one row per get_sentiment() call, with a
# shadow[] array containing fingpt per-headline + cumulative + finbert.
#
# Known open issue: sentiment_ab_log.jsonl shows fingpt returning constant
# "neutral, 0.7 confidence" for every real headline — see the
# project_fingpt_parser_defaulting_neutral memory. That is a parser / prompt
# bug in /mnt/q/models/fingpt_infer.py, NOT a problem with this migration.
# Scheduled as the immediate follow-up after this PR merges.

_pending_ab_entries: dict[str, dict] = {}
_pending_ab_lock = threading.Lock()


def _stash_ab_context(
    ab_key: str,
    ticker: str,
    primary_result: dict,
    all_articles: list[dict],
    diss_mult: float,
) -> None:
    """Store the inline portion of an A/B entry until the batched fingpt
    results arrive in flush_ab_log(). Called from get_sentiment().

    Thread-safe — multiple ThreadPoolExecutor workers call this concurrently.
    """
    with _pending_ab_lock:
        _pending_ab_entries[ab_key] = {
            "ticker": ticker,
            "primary_result": primary_result,
            "finbert_shadow": None,  # filled in below by get_sentiment
            "all_articles": all_articles,
            "diss_mult": diss_mult,
            "fingpt_headlines_raw": None,  # filled in by Phase 3
            "fingpt_cumulatives_raw": {},  # sub_key → raw dict, filled in by Phase 3
        }


def _stash_finbert_shadow(ab_key: str, finbert_shadow: dict | None) -> None:
    """Attach the inline FinBERT shadow result to a pending A/B entry."""
    with _pending_ab_lock:
        entry = _pending_ab_entries.get(ab_key)
        if entry is not None:
            entry["finbert_shadow"] = finbert_shadow


def _stash_fingpt_result(ab_key: str, sub_key: str, result) -> None:
    """Called from portfolio.llm_batch._flush_fingpt_phase with the parsed
    fingpt result for one (ab_key, sub_key) tuple.

    sub_key is either "headlines" (result is a list of per-headline dicts)
    or "cumul:<N>" (result is a single cumulative dict).
    """
    with _pending_ab_lock:
        entry = _pending_ab_entries.get(ab_key)
        if entry is None:
            # get_sentiment was never called for this key this cycle — can
            # happen if enqueue_fingpt ran but the parent get_sentiment
            # raised before _stash_ab_context. Drop silently.
            return
        if sub_key == "headlines":
            entry["fingpt_headlines_raw"] = result
        elif sub_key.startswith("cumul:"):
            entry["fingpt_cumulatives_raw"][sub_key] = result


def flush_ab_log() -> None:
    """Walk _pending_ab_entries, merge batched fingpt results into shadow
    arrays, write one JSONL row per entry, and clear the buffer.

    Called once per cycle by main.py immediately after flush_llm_batch()
    finishes Phase 3. Safe to call even if some fingpt results are missing
    (the server returned None for that prompt) — those slots just get
    dropped from the shadow array, same as the daemon-era behavior of
    logging a fingpt:error entry.

    Thread-safe: acquires _pending_ab_lock for the entry snapshot, then
    clears the buffer under the same lock so no subsequent cycle can see
    leftover state.
    """
    with _pending_ab_lock:
        entries_snapshot = dict(_pending_ab_entries)
        _pending_ab_entries.clear()

    if not entries_snapshot:
        return

    for ab_key, entry in entries_snapshot.items():
        try:
            shadow: list[dict] = []

            # Fingpt per-headline → aggregate via _aggregate_sentiments the
            # same way the old inline path did. If the raw list is missing
            # (server returned nothing), skip the entry silently.
            fingpt_raw = entry.get("fingpt_headlines_raw")
            if fingpt_raw:
                # Filter out None entries (per-prompt failures).
                usable = [r for r in fingpt_raw if r is not None]
                if usable:
                    try:
                        fg_overall, fg_avg = _aggregate_sentiments(
                            usable,
                            headlines=entry["all_articles"],
                            dissemination_mult=entry.get("diss_mult", 1.0),
                        )
                        shadow.append({
                            "model": usable[0].get("model", "fingpt:finance-llama-8b"),
                            "sentiment": fg_overall,
                            "confidence": round(fg_avg[fg_overall], 4),
                            "avg_scores": {k: round(v, 4) for k, v in fg_avg.items()},
                        })
                    except Exception:
                        logger.debug(
                            "fingpt headlines aggregation failed for %s", ab_key,
                            exc_info=True,
                        )

            # Fingpt cumulative clusters → one shadow entry per cluster.
            for _sub_key in sorted(entry.get("fingpt_cumulatives_raw", {})):
                cum = entry["fingpt_cumulatives_raw"][_sub_key]
                if cum is None:
                    continue
                shadow.append({
                    "model": cum.get("model", "fingpt:cumulative"),
                    "sentiment": cum.get("sentiment", "neutral"),
                    "confidence": cum.get("confidence", 0.0),
                    "headline_count": cum.get("headline_count", 0),
                })

            # FinBERT shadow (already aggregated inline during get_sentiment).
            finbert = entry.get("finbert_shadow")
            if finbert is not None:
                shadow.append(finbert)

            if shadow:
                _log_ab_result(entry["ticker"], entry["primary_result"], shadow)
        except Exception:
            logger.debug("flush_ab_log: entry %s failed", ab_key, exc_info=True)


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
    #
    # 2026-04-09: The A/B log write used to happen inline at the bottom of
    # this function. It is now DEFERRED to flush_ab_log() which is called
    # post-cycle from main.py after flush_llm_batch() completes Phase 3
    # (fingpt sentiment). Rationale: fingpt used to run in a bespoke NDJSON
    # daemon (scripts/fingpt_daemon.py, now retired) blocking inside this
    # function; moving fingpt into portfolio.llm_batch's shared llama_server
    # rotation means the fingpt result does not arrive until AFTER
    # get_sentiment() has returned. Rather than duplicate the A/B log entry
    # or block on the batch, we stash the primary + finbert + context here
    # and let flush_ab_log() assemble the final row.
    #
    # The primary model's voting result is still computed and returned
    # SYNCHRONOUSLY — batching only affects the shadow log, not the vote.
    ab_key = f"{short}:{datetime.now(UTC).isoformat()}"
    _stash_ab_context(ab_key, short, primary_result, all_articles, diss_mult)

    # Shadow: FinGPT — enqueue for post-cycle Phase 3 execution. Zero-cost
    # here; the actual inference runs via llama_server finance-llama-8b
    # rotation after the ticker pool completes.
    try:
        from portfolio.llm_batch import enqueue_fingpt
        enqueue_fingpt(
            ab_key, "headlines",
            {"mode": "headlines", "texts": titles, "ticker": short},
        )
        clusters = _cluster_headlines(all_articles)
        for idx, cluster in enumerate(clusters):
            if len(cluster) >= 3:
                cluster_titles = [a["title"] for a in cluster]
                enqueue_fingpt(
                    ab_key, f"cumul:{idx}",
                    {"mode": "cumulative", "texts": cluster_titles, "ticker": short},
                )
    except Exception as e:
        logger.debug("FinGPT enqueue failed: %s", e)

    # Shadow: FinBERT (CPU, fast) — still runs inline because it's cheap
    # and on CPU (no model swap cost) and we'd rather not add a fourth
    # phase to llm_batch for an already-shadow-of-shadow signal. Stash its
    # aggregated entry into the pending A/B buffer so flush_ab_log sees it.
    try:
        finbert_results = _run_finbert(titles)
        if finbert_results:
            fb_overall, fb_avg = _aggregate_sentiments(
                finbert_results, headlines=all_articles, dissemination_mult=diss_mult
            )
            _stash_finbert_shadow(ab_key, {
                "model": "FinBERT",
                "sentiment": fb_overall,
                "confidence": round(fb_avg[fb_overall], 4),
                "avg_scores": {k: round(v, 4) for k, v in fb_avg.items()},
            })
    except Exception as e:
        logger.debug("FinBERT shadow failed: %s", e)

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
