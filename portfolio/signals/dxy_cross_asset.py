"""DXY cross-asset signal — inverse correlation with metals on intraday.

Why this exists: DXY (US Dollar Index) has R² ~0.6 inverse correlation
with silver/gold on 1-3h horizons — arguably the single strongest
short-term directional driver for metals. The existing macro_regime
signal has DXY as 1 of 6 sub-indicators at daily resolution, diluting
its vote. This standalone signal captures DXY directly at 60m resolution
so intraday dollar moves translate to a real BUY/SELL vote for metals.

Metals-only (XAU-USD, XAG-USD). Crypto + equities get HOLD.

Direction: DXY up → USD strong → metals weak → SELL. DXY down → BUY.
Threshold: 0.15% 1h move. DXY typical 1h range is 0.05-0.10% — a 0.15%
move is a clearly directional hour. Confidence scales linearly with move
magnitude up to a 0.5% cap (a ~5 sigma move saturates confidence).

Complements rather than replaces macro_regime's daily DXY vote. Keep both.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("portfolio.signals.dxy_cross_asset")

_METALS_TICKERS = {"XAU-USD", "XAG-USD"}

# 1h DXY move threshold. Typical 1h DXY range is ~0.05-0.10%; 0.15% is a
# ~2-sigma move. Below this, signal votes HOLD.
_DXY_1H_THRESHOLD_PCT = 0.15

# Above this 1h move magnitude, confidence saturates at 1.0. A 0.5% 1h
# move in DXY is exceptional — think major macro data release.
_DXY_1H_CONFIDENCE_CAP_PCT = 0.5


def compute_dxy_cross_asset_signal(
    df: Any, context: dict | None = None, **kwargs,
) -> dict:
    """Compute DXY-direction vote for a metals ticker.

    Args:
        df: OHLCV DataFrame (unused — DXY data fetched separately).
        context: dict with ``ticker`` key. Other keys ignored.

    Returns:
        dict with action, confidence, sub_signals, indicators.
        HOLD with 0 confidence if ticker is not metals or DXY data is
        unavailable.
    """
    empty = {
        "action": "HOLD", "confidence": 0.0,
        "sub_signals": {}, "indicators": {},
    }

    context = context or {}
    ticker = context.get("ticker", kwargs.get("ticker", ""))
    if ticker not in _METALS_TICKERS:
        return empty

    try:
        from portfolio.macro_context import get_dxy_intraday
    except ImportError:
        logger.debug("macro_context.get_dxy_intraday not available")
        return empty

    data = get_dxy_intraday()
    if data is None:
        return empty

    change_1h = data.get("change_1h_pct")
    if change_1h is None:
        return empty

    if change_1h < -_DXY_1H_THRESHOLD_PCT:
        # Weak USD → strong metals → BUY
        action = "BUY"
        confidence = min(abs(change_1h) / _DXY_1H_CONFIDENCE_CAP_PCT, 1.0)
    elif change_1h > _DXY_1H_THRESHOLD_PCT:
        # Strong USD → weak metals → SELL
        action = "SELL"
        confidence = min(abs(change_1h) / _DXY_1H_CONFIDENCE_CAP_PCT, 1.0)
    else:
        action = "HOLD"
        confidence = 0.0

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": {"dxy_1h": action},
        "indicators": {
            "dxy_value": data.get("value"),
            "dxy_change_1h_pct": round(change_1h, 3),
            "dxy_change_3h_pct": (
                round(data["change_3h_pct"], 3)
                if data.get("change_3h_pct") is not None else None
            ),
            "source": data.get("source", "unknown"),
        },
    }
