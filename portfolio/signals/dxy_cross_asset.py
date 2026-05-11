"""DXY cross-asset signal — inverse correlation with metals and crypto.

DXY (US Dollar Index) has R² ~0.6 inverse correlation with silver/gold
on 1-3h horizons, and -0.72 correlation with BTC. This standalone signal
captures DXY directly at 60m resolution so intraday dollar moves
translate to a real BUY/SELL vote.

Supported: XAU-USD, XAG-USD (metals), BTC-USD, ETH-USD (crypto).
Equities get HOLD.

Direction: DXY up → USD strong → risk-off → SELL. DXY down → BUY.
Thresholds differ by asset class — crypto is noisier so uses wider gate.
Confidence scales linearly with move magnitude up to a cap.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("portfolio.signals.dxy_cross_asset")

_METALS_TICKERS = {"XAU-USD", "XAG-USD"}
_CRYPTO_TICKERS = {"BTC-USD", "ETH-USD"}
_SUPPORTED_TICKERS = _METALS_TICKERS | _CRYPTO_TICKERS

_DXY_1H_THRESHOLD_PCT = 0.15
_DXY_1H_THRESHOLD_CRYPTO_PCT = 0.20

_DXY_1H_CONFIDENCE_CAP_PCT = 0.5
_DXY_1H_CONFIDENCE_CAP_CRYPTO_PCT = 0.6


def compute_dxy_cross_asset_signal(
    df: Any, context: dict | None = None, **kwargs,
) -> dict:
    """Compute DXY-direction vote for metals and crypto tickers.

    Args:
        df: OHLCV DataFrame (unused — DXY data fetched separately).
        context: dict with ``ticker`` key. Other keys ignored.

    Returns:
        dict with action, confidence, sub_signals, indicators.
        HOLD with 0 confidence if ticker is unsupported or DXY data
        unavailable.
    """
    empty = {
        "action": "HOLD", "confidence": 0.0,
        "sub_signals": {}, "indicators": {},
    }

    context = context or {}
    ticker = context.get("ticker", kwargs.get("ticker", ""))
    if ticker not in _SUPPORTED_TICKERS:
        return empty

    is_crypto = ticker in _CRYPTO_TICKERS
    threshold = _DXY_1H_THRESHOLD_CRYPTO_PCT if is_crypto else _DXY_1H_THRESHOLD_PCT
    cap = _DXY_1H_CONFIDENCE_CAP_CRYPTO_PCT if is_crypto else _DXY_1H_CONFIDENCE_CAP_PCT

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

    if change_1h < -threshold:
        action = "BUY"
        confidence = min(abs(change_1h) / cap, 1.0)
    elif change_1h > threshold:
        action = "SELL"
        confidence = min(abs(change_1h) / cap, 1.0)
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
            "asset_class": "crypto" if is_crypto else "metals",
        },
    }
