"""BTC Spot ETF Net Flow signal module.

Tracks daily net inflows/outflows across US spot Bitcoin ETFs (IBIT, GBTC,
FBTC, ARKB, BITB, etc.). ETF flows represent ~9x daily mining supply and
are the dominant marginal buyer/seller for BTC since Jan 2024.

Sub-indicators:
    1. Daily net flow   — absolute $ amount vs threshold (>$200M = strong BUY)
    2. Flow streak      — consecutive days of inflows/outflows (momentum)
    3. Flow vs price    — divergence detection (flows up + price down = accumulation)

Data sources (in order of preference):
    - CoinGlass API (free tier: 1000 req/day, no auth for public endpoints)
    - SoSoValue (scraping, fragile)
    - Manual CSV upload to data/btc_etf_flows.csv

Applicable: BTC-USD only. MSTR gets indirect exposure via btc_proxy signal.

TODO (D5 from research plan 2026-05-23):
    - Implement _fetch_coinglass_flows() with http_retry
    - Add data persistence to data/btc_etf_flow_history.jsonl
    - Tune thresholds on historical flow data (Jan 2024 - present)
    - Add to signal_registry.py (currently discovered but not registered)
    - Shadow period: 30 days before enabling
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from portfolio.signal_utils import majority_vote, safe_float

logger = logging.getLogger("portfolio.signals.btc_etf_flow")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
FLOW_CACHE_FILE = DATA_DIR / "btc_etf_flow_cache.json"
FLOW_HISTORY_FILE = DATA_DIR / "btc_etf_flow_history.jsonl"

FLOW_CACHE_TTL = 3600
STRONG_INFLOW_USD = 200_000_000
MODERATE_INFLOW_USD = 50_000_000
STRONG_OUTFLOW_USD = -200_000_000
MODERATE_OUTFLOW_USD = -50_000_000
STREAK_THRESHOLD = 3

_cache: dict = {}
_cache_ts: float = 0.0

APPLICABLE_TICKERS = frozenset({"BTC-USD"})


def compute(ticker, indicators, context=None):
    """Compute BTC ETF flow signal.

    Returns:
        dict: {action, confidence, indicators: {...}}
    """
    if ticker not in APPLICABLE_TICKERS:
        return {"action": "HOLD", "confidence": 0, "indicators": {}}

    flow_data = _get_flow_data()
    if not flow_data:
        return {"action": "HOLD", "confidence": 0,
                "indicators": {"error": "no_flow_data"}}

    votes = []
    sub_indicators = {}

    daily_vote, daily_ind = _daily_flow_signal(flow_data)
    votes.append(daily_vote)
    sub_indicators["daily_flow"] = daily_ind

    streak_vote, streak_ind = _streak_signal(flow_data)
    votes.append(streak_vote)
    sub_indicators["flow_streak"] = streak_ind

    if indicators:
        div_vote, div_ind = _divergence_signal(flow_data, indicators)
        votes.append(div_vote)
        sub_indicators["flow_divergence"] = div_ind

    action, confidence = majority_vote(votes)
    return {
        "action": action,
        "confidence": round(confidence, 3),
        "indicators": sub_indicators,
    }


def _get_flow_data():
    """Get latest ETF flow data from cache or fetch."""
    global _cache, _cache_ts
    if time.time() - _cache_ts < FLOW_CACHE_TTL and _cache:
        return _cache

    # TODO: implement _fetch_coinglass_flows()
    # For now, try loading from local cache file
    try:
        from portfolio.file_utils import load_json
        data = load_json(FLOW_CACHE_FILE)
        if data:
            _cache = data
            _cache_ts = time.time()
            return data
    except Exception:
        logger.debug("btc_etf_flow: failed to load flow cache", exc_info=True)
    return None


def _daily_flow_signal(flow_data):
    """Sub-1: Daily net flow vs thresholds."""
    net_flow = safe_float(flow_data.get("net_flow_usd"))
    if net_flow is None:
        return "HOLD", {"net_flow_usd": None}

    indicators = {"net_flow_usd": net_flow}
    if net_flow >= STRONG_INFLOW_USD:
        return "BUY", {**indicators, "strength": "strong_inflow"}
    elif net_flow >= MODERATE_INFLOW_USD:
        return "BUY", {**indicators, "strength": "moderate_inflow"}
    elif net_flow <= STRONG_OUTFLOW_USD:
        return "SELL", {**indicators, "strength": "strong_outflow"}
    elif net_flow <= MODERATE_OUTFLOW_USD:
        return "SELL", {**indicators, "strength": "moderate_outflow"}
    return "HOLD", {**indicators, "strength": "neutral"}


def _streak_signal(flow_data):
    """Sub-2: Consecutive days of inflows/outflows."""
    streak = flow_data.get("consecutive_inflow_days", 0)
    indicators = {"consecutive_days": streak}
    if streak >= STREAK_THRESHOLD:
        return "BUY", {**indicators, "direction": "inflow_streak"}
    elif streak <= -STREAK_THRESHOLD:
        return "SELL", {**indicators, "direction": "outflow_streak"}
    return "HOLD", indicators


def _divergence_signal(flow_data, indicators):
    """Sub-3: Flow-price divergence (accumulation/distribution detection)."""
    net_flow = safe_float(flow_data.get("net_flow_usd"))
    price_change_pct = indicators.get("price_change_1d", 0)
    if net_flow is None:
        return "HOLD", {"divergence": "no_data"}

    ind = {"net_flow_usd": net_flow, "price_change_pct": price_change_pct}
    if net_flow > MODERATE_INFLOW_USD and price_change_pct < -1.0:
        return "BUY", {**ind, "divergence": "accumulation"}
    elif net_flow < MODERATE_OUTFLOW_USD and price_change_pct > 1.0:
        return "SELL", {**ind, "divergence": "distribution"}
    return "HOLD", {**ind, "divergence": "none"}
