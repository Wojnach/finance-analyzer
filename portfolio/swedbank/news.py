"""Headlines for the Swedbank universe — Yahoo only, no shared state.

Deliberately does NOT go through `sentiment.get_sentiment()` or the
`news_event` / `sentiment` vote path. Both end in
`signal_engine.flush_sentiment_state()`, which rewrites
`data/sentiment_state.json` as a whole-dict overwrite from one process's
in-memory copy — a second writer clobbers whatever Layer 1 stored for its own
tickers (signals.py docstring, point 2). This module only reads.

It also does not vote. Scores are attached to the snapshot for display; wiring
them into consensus needs calibration against realized outcomes first, and an
uncalibrated news vote on 22 instruments would move real decisions on nothing
but keyword counts.

NewsAPI is untouched: its 90/day budget is fully committed to XAU/XAG at 20-min
TTL plus MSTR (`shared_state._NEWSAPI_PRIORITY`). Yahoo has no quota.

Coverage is 22 of 26. The four Stockholm equities (INVE-B, SAAB-B, SEB-C,
BEAMMW-B) return nothing from Yahoo — verified live, SAAB-B yields 0 articles —
so they report `available: False` rather than a fabricated neutral score.
"""

from __future__ import annotations

import datetime
import logging
import threading

from portfolio.swedbank.instruments import INSTRUMENTS
from portfolio.swedbank.signals import UNDERLYING

logger = logging.getLogger("portfolio.swedbank.news")

CACHE_TTL_SECONDS = 1800

# Stockholm listings with no Yahoo coverage. Kept explicit rather than derived
# from venue: MINI-TSMC and the two XBT trackers are also STO-listed but map to
# underlyings (TSM, BTC-USD, ETH-USD) that Yahoo covers well.
NO_COVERAGE = {"INVE-B", "SAAB-B", "SEB-C", "BEAMMW-B"}

# Mirrors news_keywords.keyword_severity's return values. Unknown labels rank 0
# rather than raising, so a new severity tier upstream degrades to "lowest"
# instead of taking down the whole news block.
_SEVERITY_RANK = {"normal": 0, "moderate": 1, "high": 2, "critical": 3}

_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def news_ticker_for(key):
    """Symbol to query Yahoo with. Leveraged products use their underlying."""
    return UNDERLYING.get(key, key)


def _aggregate(articles, ticker):
    from portfolio.news_keywords import (
        dissemination_score,
        is_credible_source,
        is_relevant_headline,
        keyword_severity,
        score_headline,
    )

    scored = []
    for a in articles:
        title = (a.get("title") or "").strip()
        if not title:
            continue
        severity_score, keywords = score_headline(title)
        scored.append(
            {
                "title": title,
                "source": a.get("source") or "",
                "published": a.get("published") or "",
                "score": round(float(severity_score), 4),
                "severity": keyword_severity(title),
                "keywords": keywords,
                "credible": is_credible_source(a.get("source") or ""),
                "relevant": is_relevant_headline(title, ticker),
            }
        )

    if not scored:
        return {"available": True, "n": 0, "headlines": [], "max_score": 0.0}

    relevant = [s for s in scored if s["relevant"]]
    # Rank on relevant headlines when any matched the ticker, else fall back to
    # the full set — a sector-wide story with no ticker mention still matters,
    # but a ticker-specific one outranks it.
    basis = relevant or scored
    scores = [s["score"] for s in basis]
    return {
        "available": True,
        "n": len(scored),
        "n_relevant": len(relevant),
        "max_score": round(max(scores), 4),
        "mean_score": round(sum(scores) / len(scores), 4),
        "top_severity": max(
            (s["severity"] for s in basis), key=lambda x: _SEVERITY_RANK.get(x, 0)
        ),
        "dissemination": round(float(dissemination_score(articles)), 4),
        "headlines": sorted(basis, key=lambda s: -s["score"])[:5],
    }


def fetch_for(key, limit=10, fetch_fn=None, use_cache=True):
    """Headlines + keyword scores for one instrument key.

    Never raises — a news outage must degrade this block, not the whole
    signal snapshot the caller is building.
    """
    inst = INSTRUMENTS.get(key)
    base = {"key": key, "checked_at": _now().isoformat()}
    if inst is None:
        return {**base, "available": False, "reason": "unknown instrument"}
    if key in NO_COVERAGE:
        return {
            **base,
            "available": False,
            "reason": "no Yahoo coverage for this Stockholm listing",
        }

    ticker = news_ticker_for(key)
    base["news_ticker"] = ticker
    if ticker != key:
        base["note"] = f"headlines for underlying {ticker}"

    if use_cache:
        with _lock:
            hit = _cache.get(ticker)
        if hit and (_now().timestamp() - hit[0]) < CACHE_TTL_SECONDS:
            return {**base, **hit[1], "cached": True}

    try:
        fetch = fetch_fn
        if fetch is None:
            from portfolio.sentiment import _fetch_yahoo_headlines

            fetch = _fetch_yahoo_headlines
        articles = fetch(ticker, limit=limit) or []
    except Exception as exc:
        logger.warning("swedbank news: fetch failed for %s: %s", ticker, exc)
        return {**base, "available": False, "reason": f"{type(exc).__name__}: {exc}"}

    payload = _aggregate(articles, ticker)
    with _lock:
        _cache[ticker] = (_now().timestamp(), payload)
    return {**base, **payload, "cached": False}


def fetch_universe(keys=None, limit=10, fetch_fn=None, use_cache=True):
    """Sequential, same reason as pricing.sweep — one shared Avanza-era session
    discipline and no reason to hammer Yahoo in parallel for 22 symbols."""
    keys = list(INSTRUMENTS if keys is None else keys)
    return {
        k: fetch_for(k, limit=limit, fetch_fn=fetch_fn, use_cache=use_cache)
        for k in keys
    }
