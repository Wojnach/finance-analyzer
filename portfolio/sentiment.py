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


# 2026-04-09 (fix/bert-inproc-gpu): map subprocess script paths to in-process
# model names so _run_model can try the fast in-process path first and fall
# back to the old subprocess path on failure. See portfolio/bert_sentiment.py
# for the full rationale — short version: subprocess cold-load was ~3-10s per
# call, in-process on GPU is ~50-200ms per call, ~20-60x speedup with the
# same output shape.
_INPROC_BERT_MAP = {
    CRYPTOBERT_SCRIPT: "CryptoBERT",
    TRADING_HERO_SCRIPT: "Trading-Hero-LLM",
    FINBERT_SCRIPT: "FinBERT",
}


def _run_model(script, texts):
    """Run a sentiment model.

    Tries the in-process BERT cache first (portfolio.bert_sentiment) because
    it avoids the ~3-10 s subprocess spawn + cold-load cost and runs on GPU
    if available. Falls back to the legacy subprocess pattern on any failure
    so the main loop stays up even if torch/transformers break or a model
    cache dir is missing.
    """
    model_name = _INPROC_BERT_MAP.get(script)
    if model_name is not None:
        try:
            from portfolio.bert_sentiment import predict as _bert_predict
            return _bert_predict(model_name, texts)
        except Exception as e:
            # Log once per (model, exception class) to keep the log clean if
            # we end up stuck on the subprocess fallback. sentiment.py already
            # has its own logger configured.
            logger.warning(
                "In-process BERT %s failed, falling back to subprocess: %s",
                model_name, e,
            )

    # Legacy subprocess path (also used if script is not one of the three
    # known BERT models, though that doesn't happen today).
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

    2026-04-28: cryptobert_shadow slot added; CryptoBERT was demoted from
    crypto primary to shadow. See get_sentiment docstring for rationale.
    """
    with _pending_ab_lock:
        _pending_ab_entries[ab_key] = {
            "ticker": ticker,
            "primary_result": primary_result,
            "finbert_shadow": None,  # filled in below by get_sentiment
            "cryptobert_shadow": None,  # filled in below for crypto tickers (2026-04-28)
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


def _stash_cryptobert_shadow(ab_key: str, cryptobert_shadow: dict | None) -> None:
    """Attach the inline CryptoBERT shadow result to a pending A/B entry.

    Added 2026-04-28 when CryptoBERT was demoted from crypto primary to
    shadow. Symmetric with _stash_finbert_shadow.
    """
    with _pending_ab_lock:
        entry = _pending_ab_entries.get(ab_key)
        if entry is not None:
            entry["cryptobert_shadow"] = cryptobert_shadow


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

            # CryptoBERT shadow (added 2026-04-28 — was the primary; demoted
            # to shadow due to 99.1% neutral output on press-wire input).
            # Crypto tickers only; entry stays None for stocks.
            cryptobert = entry.get("cryptobert_shadow")
            if cryptobert is not None:
                shadow.append(cryptobert)

            if shadow:
                _log_ab_result(entry["ticker"], entry["primary_result"], shadow)
        except Exception:
            logger.debug("flush_ab_log: entry %s failed", ab_key, exc_info=True)


def _run_finbert(texts):
    """Run FinBERT sentiment inference.

    2026-04-09 (fix/bert-inproc-gpu): routes through _run_model so FinBERT
    also benefits from the in-process GPU cache. _run_model's _INPROC_BERT_MAP
    knows that FINBERT_SCRIPT -> "FinBERT" and will hit bert_sentiment.predict
    first, falling back to the old subprocess path on any exception.
    """
    return _run_model(FINBERT_SCRIPT, texts)


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

# Decisiveness margins added 2026-04-28. See docs/PLAN_sentiment_2026_04_28.md
# for the full rationale; tl;dr: the old aggregator returned the top label by
# 0.001 vs second, so a 0.34/0.33/0.33 split labeled "positive". Now we require
# a real margin before committing to a non-neutral verdict, and we default to
# label-majority over score-averaging so a few decisive headlines are not
# drowned by many tepid-neutral peers.
_DECISIVE_MARGIN_AVG = 0.05      # avg-mode: top-vs-second margin in prob units
_DECISIVE_MARGIN_PER_HEADLINE = 0.10  # majority-mode: per-headline label margin
_DECISIVE_MARGIN_MAJORITY = 1e-9  # majority-mode: top-vs-second weight margin
                                  # (zero-tolerance — exact ties go neutral)


def _compute_weights(sentiments, headlines, dissemination_mult):
    """Return per-sentiment weights from keyword scoring + dissemination."""
    if headlines and len(headlines) == len(sentiments):
        from portfolio.news_keywords import score_headline
        weights = []
        for h in headlines:
            title = h.get("title", "") if isinstance(h, dict) else str(h)
            w, _ = score_headline(title)
            weights.append(w * dissemination_mult)
    else:
        weights = [dissemination_mult] * len(sentiments)
    return weights


def _aggregate_sentiments(sentiments, headlines=None, dissemination_mult=1.0,
                           *, mode="majority"):
    """Aggregate sentiment scores into a single (label, avg_dict) verdict.

    mode="majority" (default, 2026-04-28): label-majority vote.
        Each headline gets its own decisive label (top score must beat second
        by >=_DECISIVE_MARGIN_PER_HEADLINE, else "neutral"), then a weighted
        majority over those labels picks the verdict. Exact ties resolve to
        neutral. The returned avg_dict is still the score-weighted-average
        (kept identical for backward-compat with consumers that read
        avg_scores like sentiment_avg_scores in signal_engine.py:2452).

    mode="average": legacy probability-averaging. Returns the top-scored
        label IF its margin over the second exceeds _DECISIVE_MARGIN_AVG;
        otherwise downgrades to "neutral". The pure-max-without-margin
        behavior was the source of the W16-W17 sentiment regression and is
        no longer reachable.

    When headlines are provided, score weights from news_keywords.score_headline()
    amplify high-impact keywords (tariff/war/crash 3x). dissemination_mult
    multiplies all weights when news is widely cross-referenced.
    """
    if not sentiments:
        return "neutral", {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

    weights = _compute_weights(sentiments, headlines, dissemination_mult)
    total_w = sum(weights)
    if total_w == 0:
        return "neutral", {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

    pos_sum = sum(s["scores"]["positive"] * w for s, w in zip(sentiments, weights))
    neg_sum = sum(s["scores"]["negative"] * w for s, w in zip(sentiments, weights))
    neu_sum = sum(s["scores"]["neutral"] * w for s, w in zip(sentiments, weights))
    avg = {
        "positive": pos_sum / total_w,
        "negative": neg_sum / total_w,
        "neutral":  neu_sum / total_w,
    }

    if mode == "majority":
        verdict = _majority_label(sentiments, weights)
        return verdict, avg

    # mode == "average" — legacy threshold-augmented score-averaging
    overall = max(avg, key=avg.get)
    sorted_scores = sorted(avg.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1]
    if margin < _DECISIVE_MARGIN_AVG and overall != "neutral":
        return "neutral", avg
    return overall, avg


def _majority_label(sentiments, weights):
    """Per-headline decisive label, then weighted majority vote.

    Each headline classified as positive/negative/neutral with its own per-
    headline margin gate. Weights are summed per label; winner returned only
    if it beats the second by more than _DECISIVE_MARGIN_MAJORITY.
    """
    bucket = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for s, w in zip(sentiments, weights):
        scores = s["scores"]
        # decisive per-headline label: top must beat second by margin
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top_score = ordered[0]
        second_score = ordered[1][1]
        if (top_score - second_score) < _DECISIVE_MARGIN_PER_HEADLINE:
            top_label = "neutral"
        bucket[top_label] += w

    ordered_buckets = sorted(bucket.items(), key=lambda kv: kv[1], reverse=True)
    winner_label, winner_w = ordered_buckets[0]
    runner_w = ordered_buckets[1][1]
    if (winner_w - runner_w) <= _DECISIVE_MARGIN_MAJORITY:
        return "neutral"
    return winner_label


def _filter_relevant_headlines(articles, ticker, *, fallback_n=3):
    """Drop wire-noise headlines before model inference.

    Uses news_keywords.is_relevant_headline (keyword OR ticker-synonym match)
    plus a credible-source-with-long-title escape hatch (Reuters/Bloomberg/
    etc. + title >= 25 chars covers in-depth coverage that doesn't happen to
    mention the ticker by name).

    Falls back to most-recent `fallback_n` if the filter would drop
    everything — better to have noisy signal than silent signal on slow
    news days.
    """
    if not articles:
        return []

    from portfolio.news_keywords import is_credible_source, is_relevant_headline

    kept = []
    for a in articles:
        title = a.get("title", "") if isinstance(a, dict) else str(a)
        if is_relevant_headline(title, ticker):
            kept.append(a)
            continue
        # Credible-source escape hatch: long titles from credible outlets are
        # almost always real coverage worth scoring.
        source = a.get("source", "") if isinstance(a, dict) else ""
        if is_credible_source(source) and len(title.strip()) >= 25:
            kept.append(a)

    if kept:
        return kept

    # All-irrelevant fallback: keep the most-recent N articles. Sort by the
    # `published` field (ISO timestamp string sorts chronologically), most
    # recent first. Articles without `published` sort last via empty-string
    # default.
    sorted_articles = sorted(
        articles,
        key=lambda a: a.get("published", "") if isinstance(a, dict) else "",
        reverse=True,
    )
    return sorted_articles[:fallback_n]


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

    2026-04-28 (fix/sentiment-relevance-and-aggregation): two changes here.
      1. Crypto primary model swapped CryptoBERT -> Trading-Hero-LLM.
         CryptoBERT was 99.1% neutral on 2,817 wire-feed samples (it was
         trained on crypto-twitter slang, not press-wire headlines).
         CryptoBERT now runs as a shadow for continuity of the 30d accuracy
         baseline. Trading-Hero is permabull on financial news but at least
         produces variance; the 47% directional accuracy gate still acts as
         a circuit breaker if it underperforms.
      2. Headlines pass through _filter_relevant_headlines() before model
         inference. Bare price-tickers ("Bitcoin: $67,123") and generic
         market-update boilerplate are dropped. Fallback keeps the most-
         recent N when the filter would drop everything.

    Primary model (votes):  Trading-Hero-LLM (both crypto and stocks)
    Shadow models (logged): CryptoBERT (crypto only), FinGPT, FinBERT

    Returns the primary model's result. Shadow results are logged to
    data/sentiment_ab_log.jsonl for accuracy tracking.
    """
    short = ticker.upper().replace("-USD", "")
    is_crypto = _is_crypto(short)

    if is_crypto:
        articles = _fetch_crypto_headlines(
            short, cryptocompare_api_key=cryptocompare_api_key,
        )
    else:
        articles = _fetch_stock_headlines(short, newsapi_key=newsapi_key)
    # 2026-04-28: Trading-Hero-LLM is the primary across all asset classes.
    # See module/function docstring for the CryptoBERT demotion rationale.
    model_script = TRADING_HERO_SCRIPT
    model_name = "Trading-Hero-LLM"

    social = social_posts or []
    raw_all = articles + social
    sources = {
        "news": len(articles),
        "reddit": sum(1 for p in social if "reddit" in p.get("source", "")),
    }

    if not raw_all:
        return {
            "overall_sentiment": "unknown",
            "confidence": 0.0,
            "num_articles": 0,
            "model": model_name,
            "sources": sources,
            "details": [],
        }

    # Drop wire-noise before inference. The filter has a most-recent-N
    # fallback so we never go silent on slow-news days.
    all_articles = _filter_relevant_headlines(raw_all, short)
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
    #
    # 2026-04-10 (perf/llama-swap-reduction): gated by is_llm_on_cycle. Unlike
    # ministral/qwen3 (which go through _cached_or_enqueue's should_enqueue_fn),
    # fingpt enqueues directly because it doesn't use the signal cache — it
    # only writes to the A/B shadow log. When off-cycle, skip the enqueue
    # entirely so the llama_server phase 3 skips the fingpt model swap +
    # inference cost. Fingpt is a shadow signal, so skipping 2 of 3 cycles
    # just reduces A/B sample density from every cycle to every 3rd cycle,
    # which is fine for long-running statistical comparison.
    try:
        from portfolio.llm_batch import enqueue_fingpt, is_llm_on_cycle
        if is_llm_on_cycle("fingpt"):
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

    # Shadow: CryptoBERT — demoted from primary 2026-04-28. Kept as shadow
    # for crypto tickers only so we (a) preserve the 30d accuracy baseline
    # for comparison and (b) still notice if the model ever recovers from
    # its 99.1% neutral-output collapse. Stashed into the pending A/B
    # buffer; picked up by flush_ab_log alongside FinGPT and FinBERT.
    if is_crypto:
        try:
            crypto_results = _run_model(CRYPTOBERT_SCRIPT, titles)
            if crypto_results:
                cb_overall, cb_avg = _aggregate_sentiments(
                    crypto_results, headlines=all_articles, dissemination_mult=diss_mult,
                )
                _stash_cryptobert_shadow(ab_key, {
                    "model": "CryptoBERT",
                    "sentiment": cb_overall,
                    "confidence": round(cb_avg[cb_overall], 4),
                    "avg_scores": {k: round(v, 4) for k, v in cb_avg.items()},
                })
        except Exception as e:
            logger.debug("CryptoBERT shadow failed: %s", e)

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
