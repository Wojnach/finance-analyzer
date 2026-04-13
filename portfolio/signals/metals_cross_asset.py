"""Cross-asset signal for metals -- correlated market indicators.

Signal #32.  Combines 6 cross-asset sub-indicators via majority vote:
    1. Copper Momentum: copper up -> industrial demand -> silver bullish
    2. GVZ (Gold VIX): high implied vol signals breakout/reversal
    3. Gold/Silver Ratio: mean-reversion signal (high = silver cheap)
    4. G/S Ratio Velocity: rate of change — falling = silver outperforming
    5. SPY Momentum: risk-on/risk-off gauge
    6. Oil Momentum: inflation expectations proxy

Applicable to XAU-USD and XAG-USD only.
Gold and silver interpret some signals differently (e.g. G/S ratio).

2026-04-13: Horizon realignment after live measurement showed 29.1% on
XAG 3h (178 BUY / 1 SELL bias over 179 samples). Root cause was using
5-day / 1-day lookbacks against 3-hour outcomes — lagged features with
no intraday resolution. Fix: switch primary data to intraday (60m bars
via `get_all_cross_asset_intraday`) and tighten thresholds proportionally.
Daily data retained as fallback when intraday fetch fails (weekend,
yfinance hiccup, etc.). GVZ stays daily — it's a daily-published index.
"""
from __future__ import annotations

import logging
from typing import Any

from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.metals_cross_asset")

_METALS_TICKERS = {"XAU-USD", "XAG-USD"}

# 3h-calibrated thresholds. Rationale:
#   Copper: daily 5d threshold was 1.5% (~0.3%/day). 3h typical range ~0.2%
#     — threshold 0.4% captures a clearly directional 3h move.
#   SPY: daily 1d threshold was 0.8%. Intraday 3h range typically 0.2-0.5%
#     — threshold 0.25% catches risk-on/off shifts at 1-3h scale.
#   Oil: daily 5d threshold was 2.0% (~0.4%/day). 3h typical range 0.3-0.8%
#     — threshold 0.5% captures meaningful oil moves.
#   G/S ratio velocity: daily 5d was 2.0%. 3h ratio moves typically 0.3-0.7%
#     — threshold 0.5% captures silver-vs-gold divergence in the last 3h.
_COPPER_MOVE_INTRADAY_PCT = 0.4
_SPY_MOVE_INTRADAY_PCT = 0.25
_OIL_MOVE_INTRADAY_PCT = 0.5
_GS_VELOCITY_INTRADAY_PCT = 0.5

# Daily thresholds retained for the fallback path (intraday fetch failure).
_COPPER_MOVE_DAILY_PCT = 1.5
_SPY_MOVE_DAILY_PCT = 0.8
_OIL_MOVE_DAILY_PCT = 2.0
_GS_VELOCITY_DAILY_PCT = 2.0

_GVZ_ZSCORE_HIGH = 1.5
_GVZ_ZSCORE_LOW = -1.0
_GS_RATIO_ZSCORE = 1.5


def _get_cross_asset_context(ticker: str) -> dict | None:
    """Fetch cross-asset data. Prefer intraday (60m); fall back to daily.

    Returns a dict with per-source payloads AND a ``_using_intraday`` flag
    so the caller can pick the right thresholds. Returns None only if the
    ``metals_cross_assets`` module is not importable (module-level failure).

    Daily G/S ratio is ALWAYS fetched — its z-score is a stable 20-day
    measure that's informative at every horizon, and pre-fetching it here
    ensures the intraday path never makes an extra conditional call mid-
    routing.
    """
    try:
        from portfolio.metals_cross_assets import (
            get_all_cross_asset_data,
            get_all_cross_asset_intraday,
            get_gold_silver_ratio,
            get_gvz,
        )
    except ImportError:
        logger.debug("metals_cross_assets module not available")
        return None

    # Daily anchors — always fetched, cached with their own TTL elsewhere:
    #   GVZ: no intraday source exists (CBOE daily index).
    #   G/S daily: z-score is a stable 20-day measure used on every path.
    gvz = get_gvz()
    gs_daily = get_gold_silver_ratio()

    intraday = get_all_cross_asset_intraday()
    intraday_ok = sum(
        1 for key in ("copper", "gold_silver_ratio", "spy", "oil")
        if intraday.get(key) is not None
    )
    # Require at least 3 of 4 sources to consider intraday healthy;
    # one-off API hiccups shouldn't downgrade the whole signal to stale
    # daily data.
    use_intraday = intraday_ok >= 3

    result: dict = {"_using_intraday": use_intraday}

    if use_intraday:
        # When exactly 3 of 4 intraday sources are healthy, the missing
        # one silently contributes 0 → HOLD. Log at WARNING so operators
        # can see a degraded source rather than a quiet vote loss.
        degraded = [
            key for key in ("copper", "gold_silver_ratio", "spy", "oil")
            if intraday.get(key) is None
        ]
        if degraded:
            logger.warning(
                "metals_cross_asset: intraday source(s) unavailable %s — "
                "sub-signals for these will vote HOLD this cycle",
                degraded,
            )

        copper = intraday["copper"]
        gs = intraday["gold_silver_ratio"]
        spy = intraday["spy"]
        oil = intraday["oil"]
        result["copper_change_pct"] = copper["change_3h_pct"] if copper else 0.0
        result["gs_velocity_pct"] = gs["change_3h_pct"] if gs else 0.0
        result["spy_change_pct"] = spy["change_3h_pct"] if spy else 0.0
        result["oil_change_pct"] = oil["change_3h_pct"] if oil else 0.0
    else:
        logger.debug(
            "metals_cross_asset: intraday health=%d/4, falling back to daily",
            intraday_ok,
        )
        daily = get_all_cross_asset_data()
        copper = daily.get("copper")
        # Daily G/S ratio already pre-fetched above — reuse for both the
        # velocity field (5d change) and the z-score.
        spy = daily.get("spy")
        oil = daily.get("oil")
        result["copper_change_pct"] = copper["change_5d_pct"] if copper else 0.0
        result["gs_velocity_pct"] = gs_daily["change_5d_pct"] if gs_daily else 0.0
        result["spy_change_pct"] = spy["change_1d_pct"] if spy else 0.0
        result["oil_change_pct"] = oil["change_5d_pct"] if oil else 0.0

    result["gvz_zscore"] = gvz["zscore"] if gvz else 0.0
    result["gs_ratio_zscore"] = gs_daily["zscore"] if gs_daily else 0.0

    return result


def compute_metals_cross_asset_signal(
    df: Any, context: dict | None = None, **kwargs,
) -> dict:
    """Compute cross-asset composite signal for metals.

    Args:
        df: OHLCV DataFrame (unused — cross-asset data fetched separately).
        context: dict with keys {ticker, config, macro, regime}.
    """
    empty = {
        "action": "HOLD", "confidence": 0.0,
        "sub_signals": {}, "indicators": {},
    }

    context = context or {}
    ticker = context.get("ticker", kwargs.get("ticker", ""))
    macro = context.get("macro")

    if ticker not in _METALS_TICKERS:
        return empty

    ctx = _get_cross_asset_context(ticker)
    if ctx is None:
        return empty

    using_intraday = ctx.get("_using_intraday", False)

    # Select threshold set based on data cadence
    if using_intraday:
        copper_thr = _COPPER_MOVE_INTRADAY_PCT
        spy_thr = _SPY_MOVE_INTRADAY_PCT
        oil_thr = _OIL_MOVE_INTRADAY_PCT
        gs_vel_thr = _GS_VELOCITY_INTRADAY_PCT
    else:
        copper_thr = _COPPER_MOVE_DAILY_PCT
        spy_thr = _SPY_MOVE_DAILY_PCT
        oil_thr = _OIL_MOVE_DAILY_PCT
        gs_vel_thr = _GS_VELOCITY_DAILY_PCT

    # Oil fallback: if still zero, try macro_data
    if ctx["oil_change_pct"] == 0.0 and macro and isinstance(macro, dict):
        oil_ctx = macro.get("oil", {})
        if isinstance(oil_ctx, dict):
            # Use 1d when available on intraday path (macro dict is daily),
            # else 5d. We take whatever's there rather than zero.
            fallback = oil_ctx.get("change_1d_pct") or oil_ctx.get("change_5d_pct") or 0.0
            ctx["oil_change_pct"] = fallback

    is_silver = ticker == "XAG-USD"
    votes = []
    sub_signals = {}

    # Sub 1: Copper Momentum
    cu = ctx["copper_change_pct"]
    if cu > copper_thr:
        sub_signals["copper"] = "BUY"
    elif cu < -copper_thr:
        sub_signals["copper"] = "SELL"
    else:
        sub_signals["copper"] = "HOLD"
    votes.append(sub_signals["copper"])

    # Sub 2: GVZ (Gold Volatility Index) — still daily
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

    # Sub 3: Gold/Silver Ratio — z-score is daily-stable, OK to keep
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

    # Sub 4: G/S Ratio Velocity — now intraday when available
    # Falling G/S ratio = silver outperforming gold = bullish silver
    # Rising G/S ratio = gold outperforming silver = bearish silver
    gs_vel = ctx["gs_velocity_pct"]
    if is_silver:
        if gs_vel < -gs_vel_thr:
            sub_signals["gs_velocity"] = "BUY"   # Silver gaining vs gold
        elif gs_vel > gs_vel_thr:
            sub_signals["gs_velocity"] = "SELL"   # Silver losing vs gold
        else:
            sub_signals["gs_velocity"] = "HOLD"
    else:
        # For gold: rising G/S = gold outperforming -> BUY gold
        if gs_vel > gs_vel_thr:
            sub_signals["gs_velocity"] = "BUY"
        elif gs_vel < -gs_vel_thr:
            sub_signals["gs_velocity"] = "SELL"
        else:
            sub_signals["gs_velocity"] = "HOLD"
    votes.append(sub_signals["gs_velocity"])

    # Sub 5: SPY Momentum (risk-on/risk-off) — now intraday when available
    spy = ctx["spy_change_pct"]
    if spy > spy_thr:
        # Risk-on: silver benefits (industrial), gold neutral
        sub_signals["spy_risk"] = "BUY" if is_silver else "HOLD"
    elif spy < -spy_thr:
        # Risk-off: gold benefits (safe haven), silver hurt
        sub_signals["spy_risk"] = "BUY" if not is_silver else "SELL"
    else:
        sub_signals["spy_risk"] = "HOLD"
    votes.append(sub_signals["spy_risk"])

    # Sub 6: Oil Momentum (inflation expectations) — now intraday when available
    oil = ctx["oil_change_pct"]
    if oil > oil_thr:
        sub_signals["oil"] = "BUY"
    elif oil < -oil_thr:
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
            "copper_change": round(cu, 3),
            "gvz_zscore": round(gvz, 2),
            "gs_ratio_zscore": round(gsr, 2),
            "gs_velocity": round(gs_vel, 3),
            "spy_change": round(spy, 3),
            "oil_change": round(oil, 3),
            "using_intraday": using_intraday,
        },
    }
