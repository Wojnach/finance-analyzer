"""CFTC Commitment of Traders (COT) positioning signal for precious metals.

Signal #33.  Contrarian positioning indicator using CFTC COT reports.
Combines 4 sub-indicators via majority vote:
    1. COT Index: 156-week percentile of speculative net positioning (contrarian)
    2. Commercial Hedger Change: week-over-week smart money direction
    3. Managed Money Intensity: speculator crowding z-score
    4. Real Yield Direction: falling yields = bullish gold (with regime gate)

Applicable to XAU-USD and XAG-USD only.  Data sourced from precomputed
deep context files (metals_precompute.py) with CFTC API fallback.

COT reports are published weekly (Friday for Tuesday data), so this signal
is inherently slower-moving than price-based signals.  Max confidence
capped at 0.7 to reflect data staleness.
"""
from __future__ import annotations

import logging
from typing import Any

from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.cot_positioning")

_METALS_TICKERS = {"XAU-USD", "XAG-USD"}
_COMMODITY_MAP = {"XAU-USD": "gold", "XAG-USD": "silver"}

# COT Index thresholds (0-100 percentile scale)
_COT_EXTREME_HIGH = 80  # Overbought (contrarian SELL)
_COT_EXTREME_LOW = 20   # Oversold (contrarian BUY)

# Commercial hedger change threshold (contracts)
_COMM_CHANGE_THRESHOLD = 5000

# Managed money z-score threshold
_MM_ZSCORE_HIGH = 1.5
_MM_ZSCORE_LOW = -1.5

# CFTC API settings for historical lookback
_COT_HISTORY_WEEKS = 156  # 3 years
_CFTC_LEGACY_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
_CFTC_TIMEOUT = 15


def _load_deep_context(ticker: str) -> dict | None:
    """Load precomputed deep context for the given metal."""
    from portfolio.file_utils import load_json

    metal = _COMMODITY_MAP.get(ticker)
    if not metal:
        return None

    path = f"data/{metal}_deep_context.json"
    ctx = load_json(path, default=None)
    if not ctx or not isinstance(ctx, dict):
        logger.debug("Deep context not available: %s", path)
        return None
    return ctx


def _load_cot_history(metal: str) -> list[dict]:
    """Load COT history from the local JSONL file."""
    from portfolio.file_utils import load_jsonl

    entries = load_jsonl("data/cot_history.jsonl")
    return [e for e in entries if e.get("metal") == metal]


def _fetch_cot_historical(commodity_name: str) -> list[dict]:
    """Fetch historical COT data from CFTC SOCRATA API.

    Returns up to 156 weeks of net positioning data for z-score computation.
    commodity_name should be 'GOLD' or 'SILVER'.
    """
    try:
        import requests

        url = (
            f"{_CFTC_LEGACY_URL}"
            f"?$where=commodity_name='{commodity_name}'"
            f"&$order=report_date_as_yyyy_mm_dd DESC"
            f"&$limit={_COT_HISTORY_WEEKS}"
        )
        resp = requests.get(url, timeout=_CFTC_TIMEOUT)
        resp.raise_for_status()
        rows = resp.json()

        result = []
        for r in rows:
            nc_long = _int_safe(r.get("noncomm_positions_long_all"))
            nc_short = _int_safe(r.get("noncomm_positions_short_all"))
            comm_long = _int_safe(r.get("comm_positions_long_all"))
            comm_short = _int_safe(r.get("comm_positions_short_all"))

            if nc_long is not None and nc_short is not None:
                result.append({
                    "date": r.get("report_date_as_yyyy_mm_dd", ""),
                    "nc_net": nc_long - nc_short,
                    "comm_net": (comm_long - comm_short)
                    if comm_long is not None and comm_short is not None
                    else None,
                    "oi": _int_safe(r.get("open_interest_all")),
                })
        return result
    except Exception as e:
        logger.warning("CFTC historical fetch failed: %s", e)
        return []


def _int_safe(val) -> int | None:
    """Safely convert to int, returning None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _compute_cot_index(nc_net_history: list[int]) -> float | None:
    """Compute COT Index as percentile of current net positioning.

    Formula: (Current - Min_156w) / (Max_156w - Min_156w) * 100
    Returns 0-100 scale, or None if insufficient data.
    """
    if len(nc_net_history) < 10:  # Need minimum history for meaningful percentile
        return None

    current = nc_net_history[0]  # Most recent
    hist_min = min(nc_net_history)
    hist_max = max(nc_net_history)
    hist_range = hist_max - hist_min

    if hist_range == 0:
        return 50.0  # No variation

    return round((current - hist_min) / hist_range * 100, 1)


def _sub_cot_index(cot_data: dict, historical: list[dict]) -> tuple[str, float, dict]:
    """Sub-indicator 1: COT Index percentile (contrarian).

    >80 = speculators extremely long = contrarian SELL.
    <20 = speculators extremely short = contrarian BUY.
    """
    indicators = {"cot_index": None, "nc_net": None}

    nc_net = cot_data.get("noncomm_net")
    if nc_net is None:
        return "HOLD", 0.0, indicators

    indicators["nc_net"] = nc_net

    # Build history of nc_net values
    nc_net_history = [nc_net]
    for h in historical:
        val = h.get("nc_net")
        if val is not None:
            nc_net_history.append(val)

    cot_index = _compute_cot_index(nc_net_history)
    if cot_index is None:
        return "HOLD", 0.0, indicators

    indicators["cot_index"] = cot_index

    if cot_index > _COT_EXTREME_HIGH:
        # Speculators extremely long — contrarian SELL
        intensity = min((cot_index - _COT_EXTREME_HIGH) / 20.0, 1.0)
        return "SELL", round(0.4 + 0.3 * intensity, 2), indicators
    elif cot_index < _COT_EXTREME_LOW:
        # Speculators extremely short — contrarian BUY
        intensity = min((_COT_EXTREME_LOW - cot_index) / 20.0, 1.0)
        return "BUY", round(0.4 + 0.3 * intensity, 2), indicators

    return "HOLD", 0.0, indicators


def _sub_commercial_change(cot_data: dict) -> tuple[str, dict]:
    """Sub-indicator 2: Commercial hedger net change (smart money).

    Commercial traders are hedgers with actual exposure — their positioning
    reflects fundamental supply/demand knowledge.  Increasing net long
    (less short) = bullish signal.
    """
    indicators = {"comm_net": None, "comm_net_change": None}

    comm_net = cot_data.get("comm_net")
    if comm_net is None:
        return "HOLD", indicators

    indicators["comm_net"] = comm_net

    # Use the WoW change from metals_precompute if available
    change = cot_data.get("noncomm_net_change")
    if change is not None:
        # If non-commercial net is INCREASING, commercials are getting more short
        # (they're the counterparty).  Speculators adding longs = bearish contrarian.
        indicators["comm_net_change"] = -change  # Commercial change is inverse
        if change > _COMM_CHANGE_THRESHOLD:
            return "SELL", indicators  # Specs adding longs aggressively
        elif change < -_COMM_CHANGE_THRESHOLD:
            return "BUY", indicators   # Specs liquidating longs

    return "HOLD", indicators


def _sub_managed_money(cot_data: dict, historical: list[dict]) -> tuple[str, dict]:
    """Sub-indicator 3: Managed money (hedge fund) sentiment.

    Managed money net long/short intensity relative to recent history.
    Extreme long = contrarian SELL.  Extreme short = contrarian BUY.
    """
    indicators = {"mm_net": None, "mm_zscore": None}

    mm_net = cot_data.get("managed_money_net")
    if mm_net is None:
        return "HOLD", indicators

    indicators["mm_net"] = mm_net

    # Compute z-score against local history
    mm_history = [mm_net]
    for e in historical:
        val = e.get("mm_net")
        if val is not None:
            mm_history.append(val)

    if len(mm_history) < 5:
        return "HOLD", indicators

    import numpy as np

    mean = np.mean(mm_history)
    std = np.std(mm_history)
    if std < 1:
        return "HOLD", indicators

    zscore = (mm_net - mean) / std
    indicators["mm_zscore"] = round(float(zscore), 2)

    if zscore > _MM_ZSCORE_HIGH:
        return "SELL", indicators  # Managed money extremely long — contrarian SELL
    elif zscore < _MM_ZSCORE_LOW:
        return "BUY", indicators   # Managed money extremely short — contrarian BUY

    return "HOLD", indicators


def _sub_real_yield(deep_ctx: dict, ticker: str) -> tuple[str, dict]:
    """Sub-indicator 4: Real yield direction (gold-specific).

    Falling real yields = lower opportunity cost for gold = BUY.
    Rising real yields = higher opportunity cost = SELL.

    CAVEAT: The gold-real yield inverse correlation has weakened since 2022.
    This sub-indicator carries lower weight and is gated by direction clarity.
    """
    indicators = {"real_yield": None, "real_yield_direction": None}

    # Try to get FRED data from deep context
    fred = None
    if isinstance(deep_ctx, dict):
        # Check in refresh_data section
        refresh = deep_ctx.get("refresh_data", {})
        if isinstance(refresh, dict):
            fred = refresh.get("fred")
        # Also check top-level
        if fred is None:
            fred = deep_ctx.get("fred")

    if not isinstance(fred, dict):
        return "HOLD", indicators

    real_yield = fred.get("real_yield")
    direction = fred.get("real_yield_direction")

    if real_yield is not None:
        indicators["real_yield"] = real_yield
    if direction is not None:
        indicators["real_yield_direction"] = direction

    if direction == "falling":
        return "BUY", indicators  # Falling yields = bullish gold
    elif direction == "rising":
        return "SELL", indicators  # Rising yields = bearish gold

    return "HOLD", indicators


def compute_cot_positioning_signal(
    df: Any,
    context: dict | None = None,
    **kwargs,
) -> dict:
    """Compute COT positioning composite signal for precious metals.

    Args:
        df: Price DataFrame (used minimally — this is a fundamental signal).
        context: Must contain 'ticker' key for asset identification.

    Returns:
        Standard signal dict with action, confidence, sub_signals, indicators.
    """
    empty = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {},
        "indicators": {},
    }

    # Extract ticker from context or kwargs
    ticker = ""
    if context and isinstance(context, dict):
        ticker = context.get("ticker", "")
    ticker = kwargs.get("ticker", ticker)

    if ticker not in _METALS_TICKERS:
        return empty

    metal = _COMMODITY_MAP[ticker]

    # Load precomputed deep context
    deep_ctx = _load_deep_context(ticker)

    # Extract COT data from deep context
    cot_data = None
    if deep_ctx:
        refresh = deep_ctx.get("refresh_data", {})
        if isinstance(refresh, dict):
            cot_key = f"cot_{metal}"
            cot_data = refresh.get(cot_key)

    if not cot_data or not isinstance(cot_data, dict):
        logger.debug("No COT data available for %s", ticker)
        return empty

    # Load local COT history for z-score computation
    local_history = _load_cot_history(metal)

    # If local history is thin (<20 entries), fetch from CFTC API
    historical_data = []
    if len(local_history) < 20:
        cftc_name = "GOLD" if metal == "gold" else "SILVER"
        historical_data = _fetch_cot_historical(cftc_name)
    else:
        historical_data = local_history

    # Compute sub-indicators
    votes = []
    sub_signals = {}
    all_indicators = {}

    # Sub 1: COT Index (contrarian percentile)
    cot_vote, cot_conf, cot_ind = _sub_cot_index(cot_data, historical_data)
    sub_signals["cot_index"] = cot_vote
    all_indicators.update(cot_ind)
    votes.append(cot_vote)

    # Sub 2: Commercial hedger change (smart money)
    comm_vote, comm_ind = _sub_commercial_change(cot_data)
    sub_signals["commercial_change"] = comm_vote
    all_indicators.update(comm_ind)
    votes.append(comm_vote)

    # Sub 3: Managed money intensity
    mm_vote, mm_ind = _sub_managed_money(cot_data, historical_data)
    sub_signals["managed_money"] = mm_vote
    all_indicators.update(mm_ind)
    votes.append(mm_vote)

    # Sub 4: Real yield direction (gold-specific, lower weight)
    ry_vote, ry_ind = _sub_real_yield(deep_ctx, ticker)
    sub_signals["real_yield"] = ry_vote
    all_indicators.update(ry_ind)
    votes.append(ry_vote)

    # Add report date to indicators
    all_indicators["report_date"] = cot_data.get("report_date")
    all_indicators["ticker"] = ticker

    # Majority vote
    action, confidence = majority_vote(votes, count_hold=False)

    # Cap confidence at 0.7 (external data, weekly update cycle)
    confidence = min(confidence, 0.7)

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": sub_signals,
        "indicators": all_indicators,
    }
