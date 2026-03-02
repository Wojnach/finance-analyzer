"""News/event detection signal — headline velocity, keyword severity, sentiment shift.

Combines five sub-indicators into a majority-vote composite:
  1. headline_velocity  — article count spike vs baseline
  2. keyword_severity   — highest severity keyword in recent headlines
  3. sentiment_shift    — rapid positive/negative skew in headline keywords
  4. source_weight      — credible sources amplify signal
  5. sector_impact      — keyword + sector mapping for directional vote

The ``context`` parameter is a dict with keys: ticker, config, macro.
Headlines are fetched using the existing sentiment.py functions with caching.
"""

from __future__ import annotations

import logging

import pandas as pd

from portfolio.news_keywords import (
    score_headline,
    keyword_severity,
    is_credible_source,
    get_sector_impact,
    dissemination_score,
    KEYWORD_SECTOR_IMPACT,
)
from portfolio.signal_utils import majority_vote
from portfolio.shared_state import _cached

logger = logging.getLogger("portfolio.signals.news_event")

# Cache TTL for headline fetches (5 minutes)
_HEADLINE_TTL = 300

# Baseline article count (normal news volume)
_BASELINE_ARTICLES = 5

# Max confidence cap
_MAX_CONFIDENCE = 0.7


def _fetch_headlines(ticker: str, config: dict) -> list[dict]:
    """Fetch headlines for a ticker + sector peers.

    Fetches ticker-specific headlines, then merges sector-wide headlines
    from a representative peer. This ensures that "tariff" news hitting NVDA
    also propagates to MU, TSM, AMD, etc.
    """
    short = ticker.upper().replace("-USD", "")
    from portfolio.sentiment import _is_crypto, _fetch_crypto_headlines, _fetch_stock_headlines

    articles = []
    try:
        if _is_crypto(short):
            articles = _cached(
                f"news_headlines_crypto_{short}",
                _HEADLINE_TTL,
                _fetch_crypto_headlines,
                short,
            ) or []
        else:
            newsapi_key = config.get("newsapi_key", "")
            articles = _cached(
                f"news_headlines_stock_{short}",
                _HEADLINE_TTL,
                _fetch_stock_headlines,
                short,
                newsapi_key or None,
            ) or []
    except Exception:
        logger.debug("Failed to fetch headlines for %s", ticker, exc_info=True)

    # Merge sector-peer headlines: fetch from a representative ticker per sector
    # so "tariff" news appearing in NVDA headlines also reaches MU, TSM, AMD, etc.
    # Each sector has one representative whose headlines are shared with all members.
    _SECTOR_REP_TICKER = {
        "semiconductor": "NVDA",
        "big_tech": "AAPL",
        "ai": "NVDA",
        "defense": "LMT",
        "software": "PLTR",
        "gaming": "TTWO",
        "infrastructure": "VRT",
        "metals": None,   # metals use Binance FAPI, not Yahoo
        "crypto": None,   # crypto uses CryptoCompare categories (already shared)
    }
    try:
        from portfolio.news_keywords import TICKER_SECTORS
        seen_titles = {a.get("title", "").lower() for a in articles}
        ticker_secs = TICKER_SECTORS.get(ticker, set())
        for sec in ticker_secs:
            rep = _SECTOR_REP_TICKER.get(sec)
            if not rep or rep == short:
                continue  # skip if no rep, or if this IS the rep
            newsapi_key = config.get("newsapi_key", "")
            peer_articles = _cached(
                f"news_headlines_stock_{rep}",
                _HEADLINE_TTL,
                _fetch_stock_headlines,
                rep,
                newsapi_key or None,
            ) or []
            for a in peer_articles:
                title_lower = a.get("title", "").lower()
                if title_lower and title_lower not in seen_titles:
                    articles.append(a)
                    seen_titles.add(title_lower)
    except Exception:
        logger.debug("Failed to fetch sector headlines for %s", ticker, exc_info=True)

    return articles


def _headline_velocity(headlines: list[dict]) -> tuple[str, dict]:
    """Article count spike detection.

    If article count is 2x+ baseline AND most have negative keywords, vote SELL.
    If article count is 2x+ baseline AND most have positive keywords, vote BUY.
    Otherwise HOLD.
    """
    count = len(headlines)
    indicators = {"article_count": count, "baseline": _BASELINE_ARTICLES}

    if count < _BASELINE_ARTICLES * 2:
        return "HOLD", indicators

    # Check keyword sentiment direction
    neg_count = 0
    pos_count = 0
    for h in headlines:
        title = h.get("title", "")
        sev = keyword_severity(title)
        if sev in ("critical", "high"):
            neg_count += 1
        elif sev == "moderate":
            # moderate keywords can be positive (earnings beat, upgrade)
            lower = title.lower()
            if any(kw in lower for kw in ("beat", "upgrade", "approval", "approved", "raise", "buyback", "split")):
                pos_count += 1
            else:
                neg_count += 1

    indicators["neg_keyword_count"] = neg_count
    indicators["pos_keyword_count"] = pos_count

    if neg_count > pos_count and neg_count >= 2:
        return "SELL", indicators
    if pos_count > neg_count and pos_count >= 2:
        return "BUY", indicators
    return "HOLD", indicators


def _keyword_severity_vote(headlines: list[dict]) -> tuple[str, dict]:
    """Highest severity keyword detection.

    Critical or high keywords → SELL (negative news dominates markets).
    Multiple moderate positive → BUY.
    """
    max_sev = "normal"
    max_weight = 1.0
    all_keywords = []

    for h in headlines:
        title = h.get("title", "")
        weight, matched = score_headline(title)
        all_keywords.extend(matched)
        if weight > max_weight:
            max_weight = weight
            max_sev = keyword_severity(title)

    indicators = {"max_severity": max_sev, "max_weight": max_weight,
                  "keywords_found": list(set(all_keywords))[:10]}

    if max_sev == "critical":
        return "SELL", indicators
    if max_sev == "high":
        return "SELL", indicators
    return "HOLD", indicators


def _sentiment_shift(headlines: list[dict]) -> tuple[str, dict]:
    """Rapid sentiment skew in headline keywords.

    Counts positive vs negative keyword-bearing headlines.
    Strong skew (>60% one direction with 3+ keyword headlines) = directional vote.
    """
    pos = 0
    neg = 0
    for h in headlines:
        title = h.get("title", "").lower()
        sev = keyword_severity(h.get("title", ""))
        if sev != "normal":
            # Check direction
            if any(kw in title for kw in ("beat", "upgrade", "approval", "approved",
                                           "raise", "buyback", "split", "cut")):
                # "rate cut" is positive for markets
                if "rate cut" in title:
                    pos += 1
                elif "guidance cut" in title:
                    neg += 1
                else:
                    pos += 1
            else:
                neg += 1

    total = pos + neg
    indicators = {"positive_headlines": pos, "negative_headlines": neg}

    if total < 3:
        return "HOLD", indicators

    neg_ratio = neg / total
    pos_ratio = pos / total

    if neg_ratio > 0.6:
        return "SELL", indicators
    if pos_ratio > 0.6:
        return "BUY", indicators
    return "HOLD", indicators


def _source_weight_vote(headlines: list[dict]) -> tuple[str, dict]:
    """Credible source amplification.

    If credible sources carry keyword-bearing headlines, amplify their direction.
    """
    credible_neg = 0
    credible_pos = 0
    credible_count = 0

    for h in headlines:
        source = h.get("source", "")
        if not is_credible_source(source):
            continue
        credible_count += 1
        title = h.get("title", "").lower()
        sev = keyword_severity(h.get("title", ""))
        if sev != "normal":
            if any(kw in title for kw in ("beat", "upgrade", "approval", "approved",
                                           "raise", "buyback", "split")):
                credible_pos += 1
            elif "rate cut" in title:
                credible_pos += 1
            else:
                credible_neg += 1

    indicators = {"credible_sources": credible_count,
                  "credible_negative": credible_neg,
                  "credible_positive": credible_pos}

    if credible_neg >= 2:
        return "SELL", indicators
    if credible_pos >= 2:
        return "BUY", indicators
    return "HOLD", indicators


def _sector_impact_vote(headlines: list[dict], ticker: str) -> tuple[str, dict]:
    """Keyword + sector mapping for ticker-specific directional vote.

    Uses KEYWORD_SECTOR_IMPACT to determine if a keyword has a specific
    directional impact on this ticker's sector.
    """
    buy_impacts = 0
    sell_impacts = 0
    matched_impacts = []

    for h in headlines:
        title = h.get("title", "")
        _, matched_kws = score_headline(title)
        for kw in matched_kws:
            impact = get_sector_impact(kw, ticker)
            if impact == "BUY":
                buy_impacts += 1
                matched_impacts.append(f"{kw}:BUY")
            elif impact == "SELL":
                sell_impacts += 1
                matched_impacts.append(f"{kw}:SELL")

    indicators = {"buy_impacts": buy_impacts, "sell_impacts": sell_impacts,
                  "matched_impacts": matched_impacts[:10]}

    if sell_impacts > buy_impacts and sell_impacts >= 1:
        return "SELL", indicators
    if buy_impacts > sell_impacts and buy_impacts >= 1:
        return "BUY", indicators
    return "HOLD", indicators


def _dissemination_vote(headlines: list[dict]) -> tuple[str, dict]:
    """Dissemination-aware news spread scoring (FinGPT pattern).

    High dissemination (score >= 2.0) amplifies the dominant sentiment direction.
    Uses source diversity, credible source presence, and time clustering.
    """
    score = dissemination_score(headlines)
    indicators = {"dissemination_score": score}

    if score < 1.5:
        return "HOLD", indicators

    # High dissemination — determine direction from headline keywords
    neg = 0
    pos = 0
    for h in headlines:
        sev = keyword_severity(h.get("title", ""))
        if sev in ("critical", "high"):
            neg += 1
        elif sev == "moderate":
            title_lower = h.get("title", "").lower()
            if any(kw in title_lower for kw in ("beat", "upgrade", "approval", "approved",
                                                  "raise", "buyback", "split")):
                pos += 1
            else:
                neg += 1

    indicators["dissemination_neg"] = neg
    indicators["dissemination_pos"] = pos

    if neg > pos and neg >= 2:
        return "SELL", indicators
    if pos > neg and pos >= 2:
        return "BUY", indicators
    return "HOLD", indicators


def _thesis_alignment_vote(headlines: list[dict], ticker: str, config: dict) -> tuple[str, dict]:
    """Thesis alignment with prophecy beliefs (config-gated).

    If news headlines align with an active prophecy belief's direction,
    boost that direction. Only active when config.prophecy.news_alignment is true.

    Parameters
    ----------
    headlines : list[dict]
        Recent headlines for the ticker.
    ticker : str
        Instrument ticker.
    config : dict
        Full config dict.

    Returns
    -------
    tuple[str, dict]
        (action, indicators)
    """
    indicators = {"enabled": False, "belief_id": None, "alignment": "none"}

    # Config gate
    prophecy_cfg = config.get("prophecy", {})
    if not prophecy_cfg.get("news_alignment", False):
        return "HOLD", indicators

    indicators["enabled"] = True

    # Get active beliefs for this ticker
    try:
        from portfolio.prophecy import get_active_beliefs
        beliefs = get_active_beliefs(ticker=ticker)
    except Exception:
        return "HOLD", indicators

    if not beliefs:
        return "HOLD", indicators

    # Use highest-conviction belief
    belief = max(beliefs, key=lambda b: b.get("conviction", 0))
    indicators["belief_id"] = belief.get("id", "")
    indicators["belief_direction"] = belief.get("direction", "neutral")
    indicators["belief_conviction"] = belief.get("conviction", 0)

    direction = belief.get("direction", "neutral")
    if direction == "neutral":
        return "HOLD", indicators

    # Check if headlines support the belief direction
    pos_count = 0
    neg_count = 0
    for h in headlines:
        title = h.get("title", "").lower()
        sev = keyword_severity(h.get("title", ""))
        if sev != "normal":
            if any(kw in title for kw in ("beat", "upgrade", "approval", "bullish",
                                           "raise", "buyback", "rally", "surge")):
                pos_count += 1
            else:
                neg_count += 1

    indicators["pos_headlines"] = pos_count
    indicators["neg_headlines"] = neg_count

    # Alignment: does news direction match belief direction?
    if direction == "bullish" and pos_count > neg_count and pos_count >= 2:
        indicators["alignment"] = "confirmed"
        return "BUY", indicators
    elif direction == "bearish" and neg_count > pos_count and neg_count >= 2:
        indicators["alignment"] = "confirmed"
        return "SELL", indicators
    elif direction == "bullish" and neg_count > pos_count and neg_count >= 2:
        indicators["alignment"] = "contradicted"
        # Don't vote against belief -- just HOLD (news temporarily counter to thesis)
        return "HOLD", indicators
    elif direction == "bearish" and pos_count > neg_count and pos_count >= 2:
        indicators["alignment"] = "contradicted"
        return "HOLD", indicators

    return "HOLD", indicators


def compute_news_event_signal(df: pd.DataFrame, context: dict = None) -> dict:
    """Compute the composite news/event detection signal.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data (used minimally — headlines are the primary input).
    context : dict, optional
        Dict with keys: ticker, config, macro.

    Returns
    -------
    dict
        action, confidence, sub_signals, indicators
    """
    result = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {
            "headline_velocity": "HOLD",
            "keyword_severity": "HOLD",
            "sentiment_shift": "HOLD",
            "source_weight": "HOLD",
            "sector_impact": "HOLD",
            "dissemination": "HOLD",
            "thesis_alignment": "HOLD",
        },
        "indicators": {},
    }

    if context is None:
        return result

    ticker = context.get("ticker", "")
    config = context.get("config", {})

    if not ticker:
        return result

    # Fetch headlines
    headlines = _fetch_headlines(ticker, config)
    if not headlines:
        return result

    # Compute each sub-signal
    try:
        vel_action, vel_ind = _headline_velocity(headlines)
    except Exception:
        vel_action, vel_ind = "HOLD", {}

    try:
        sev_action, sev_ind = _keyword_severity_vote(headlines)
    except Exception:
        sev_action, sev_ind = "HOLD", {}

    try:
        shift_action, shift_ind = _sentiment_shift(headlines)
    except Exception:
        shift_action, shift_ind = "HOLD", {}

    try:
        src_action, src_ind = _source_weight_vote(headlines)
    except Exception:
        src_action, src_ind = "HOLD", {}

    try:
        sec_action, sec_ind = _sector_impact_vote(headlines, ticker)
    except Exception:
        sec_action, sec_ind = "HOLD", {}

    try:
        dis_action, dis_ind = _dissemination_vote(headlines)
    except Exception:
        dis_action, dis_ind = "HOLD", {}

    # Thesis alignment (config-gated -- only votes when prophecy.news_alignment is true)
    try:
        thesis_action, thesis_ind = _thesis_alignment_vote(headlines, ticker, config)
    except Exception:
        thesis_action, thesis_ind = "HOLD", {"enabled": False}

    # Populate result
    result["sub_signals"]["headline_velocity"] = vel_action
    result["sub_signals"]["keyword_severity"] = sev_action
    result["sub_signals"]["sentiment_shift"] = shift_action
    result["sub_signals"]["source_weight"] = src_action
    result["sub_signals"]["sector_impact"] = sec_action
    result["sub_signals"]["dissemination"] = dis_action
    result["sub_signals"]["thesis_alignment"] = thesis_action

    result["indicators"].update({f"velocity_{k}": v for k, v in vel_ind.items()})
    result["indicators"].update({f"severity_{k}": v for k, v in sev_ind.items()})
    result["indicators"].update({f"shift_{k}": v for k, v in shift_ind.items()})
    result["indicators"].update({f"source_{k}": v for k, v in src_ind.items()})
    result["indicators"].update({f"sector_{k}": v for k, v in sec_ind.items()})
    result["indicators"].update({f"dissemination_{k}": v for k, v in dis_ind.items()})
    result["indicators"].update({f"thesis_{k}": v for k, v in thesis_ind.items()})
    result["indicators"]["total_headlines"] = len(headlines)

    # Majority vote (include thesis alignment only if it was enabled and voted)
    votes = [vel_action, sev_action, shift_action, src_action, sec_action, dis_action]
    if thesis_ind.get("enabled") and thesis_action != "HOLD":
        votes.append(thesis_action)
    result["action"], result["confidence"] = majority_vote(votes)

    # Cap confidence
    result["confidence"] = min(result["confidence"], _MAX_CONFIDENCE)

    return result
