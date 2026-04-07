"""Cross-asset signal for metals -- correlated market indicators.

Signal #32.  Combines 5 cross-asset sub-indicators via majority vote:
    1. Copper Momentum: copper up -> industrial demand -> silver bullish
    2. GVZ (Gold VIX): high implied vol signals breakout/reversal
    3. Gold/Silver Ratio: mean-reversion signal (high = silver cheap)
    4. SPY Momentum: risk-on/risk-off gauge
    5. Oil Momentum: inflation expectations proxy

Applicable to XAU-USD and XAG-USD only.
Gold and silver interpret some signals differently (e.g. G/S ratio).
"""
from __future__ import annotations

import logging
from typing import Any

from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.metals_cross_asset")

_METALS_TICKERS = {"XAU-USD", "XAG-USD"}

_COPPER_MOVE_PCT = 1.5
_GVZ_ZSCORE_HIGH = 1.5
_GVZ_ZSCORE_LOW = -1.0
_GS_RATIO_ZSCORE = 1.5
_SPY_MOVE_PCT = 0.8
_OIL_MOVE_PCT = 2.0


def _get_cross_asset_context(ticker: str) -> dict | None:
    """Fetch all cross-asset data. Returns None on failure."""
    try:
        from portfolio.metals_cross_assets import get_all_cross_asset_data
    except ImportError:
        logger.debug("metals_cross_assets module not available")
        return None

    data = get_all_cross_asset_data()
    result = {}

    copper = data.get("copper")
    result["copper_change_5d"] = copper["change_5d_pct"] if copper else 0.0

    gvz = data.get("gvz")
    result["gvz_zscore"] = gvz["zscore"] if gvz else 0.0

    gs = data.get("gold_silver_ratio")
    result["gs_ratio_zscore"] = gs["zscore"] if gs else 0.0

    spy = data.get("spy")
    result["spy_change_1d"] = spy["change_1d_pct"] if spy else 0.0

    oil = data.get("oil")
    result["oil_change_5d"] = oil["change_5d_pct"] if oil else 0.0

    return result


def compute_metals_cross_asset_signal(
    df: Any, *, ticker: str = "", config: dict | None = None,
    macro: dict | None = None, **kwargs,
) -> dict:
    """Compute cross-asset composite signal for metals."""
    empty = {
        "action": "HOLD", "confidence": 0.0,
        "sub_signals": {}, "indicators": {},
    }

    if ticker not in _METALS_TICKERS:
        return empty

    ctx = _get_cross_asset_context(ticker)
    if ctx is None:
        return empty

    # Oil data now fetched via get_all_cross_asset_data(); macro dict as fallback
    if ctx["oil_change_5d"] == 0.0 and macro and isinstance(macro, dict):
        oil_ctx = macro.get("oil", {})
        if isinstance(oil_ctx, dict) and "change_5d_pct" in oil_ctx:
            ctx["oil_change_5d"] = oil_ctx["change_5d_pct"]

    is_silver = ticker == "XAG-USD"
    votes = []
    sub_signals = {}

    # Sub 1: Copper Momentum
    cu = ctx["copper_change_5d"]
    if cu > _COPPER_MOVE_PCT:
        sub_signals["copper"] = "BUY"
    elif cu < -_COPPER_MOVE_PCT:
        sub_signals["copper"] = "SELL"
    else:
        sub_signals["copper"] = "HOLD"
    votes.append(sub_signals["copper"])

    # Sub 2: GVZ (Gold Volatility Index)
    # High GVZ = fear/uncertainty → safe haven demand (BUY gold, SELL silver)
    # Low GVZ = complacency → no safe haven premium (SELL gold, BUY silver)
    gvz = ctx["gvz_zscore"]
    if gvz > _GVZ_ZSCORE_HIGH:
        sub_signals["gvz"] = "BUY" if not is_silver else "SELL"
    elif gvz < _GVZ_ZSCORE_LOW:
        sub_signals["gvz"] = "SELL" if not is_silver else "BUY"
    else:
        sub_signals["gvz"] = "HOLD"
    votes.append(sub_signals["gvz"])

    # Sub 3: Gold/Silver Ratio
    gsr = ctx["gs_ratio_zscore"]
    if is_silver:
        # High G/S ratio = silver undervalued relative to gold -> BUY silver
        if gsr > _GS_RATIO_ZSCORE:
            sub_signals["gs_ratio"] = "BUY"
        elif gsr < -_GS_RATIO_ZSCORE:
            sub_signals["gs_ratio"] = "SELL"
        else:
            sub_signals["gs_ratio"] = "HOLD"
    else:
        # For gold: high G/S ratio = gold already outperforming -> HOLD
        # Low G/S ratio = gold underperforming -> mean-reversion BUY
        if gsr > _GS_RATIO_ZSCORE:
            sub_signals["gs_ratio"] = "HOLD"
        elif gsr < -_GS_RATIO_ZSCORE:
            sub_signals["gs_ratio"] = "BUY"
        else:
            sub_signals["gs_ratio"] = "HOLD"
    votes.append(sub_signals["gs_ratio"])

    # Sub 4: SPY Momentum (risk-on/risk-off)
    spy = ctx["spy_change_1d"]
    if spy > _SPY_MOVE_PCT:
        # Risk-on: silver benefits (industrial), gold neutral
        sub_signals["spy_risk"] = "BUY" if is_silver else "HOLD"
    elif spy < -_SPY_MOVE_PCT:
        # Risk-off: gold benefits (safe haven), silver hurt
        sub_signals["spy_risk"] = "BUY" if not is_silver else "SELL"
    else:
        sub_signals["spy_risk"] = "HOLD"
    votes.append(sub_signals["spy_risk"])

    # Sub 5: Oil Momentum (inflation expectations)
    oil = ctx["oil_change_5d"]
    if oil > _OIL_MOVE_PCT:
        sub_signals["oil"] = "BUY"
    elif oil < -_OIL_MOVE_PCT:
        sub_signals["oil"] = "SELL"
    else:
        sub_signals["oil"] = "HOLD"
    votes.append(sub_signals["oil"])

    action, confidence = majority_vote(votes)

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": sub_signals,
        "indicators": {
            "copper_5d": round(cu, 2),
            "gvz_zscore": round(gvz, 2),
            "gs_ratio_zscore": round(gsr, 2),
            "spy_1d": round(spy, 2),
            "oil_5d": round(oil, 2),
        },
    }
