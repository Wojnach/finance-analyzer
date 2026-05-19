"""ADX Dual Regime Meta-Signal — trend/range regime transition detector.

Uses ADX (Average Directional Index) to classify the current market regime
and detect regime transitions. Emits directional signals at transition
points using +DI/-DI to determine direction.

Sub-indicators:
    1. ADX Regime     — ADX level classifies trending (>25) vs ranging (<=25)
    2. ADX Momentum   — ADX rate of change detects regime transitions
    3. DI Spread      — +DI minus -DI provides directional context

When ADX crosses above 25 (entering trend): follow +DI/-DI direction.
When ADX crosses below 25 (entering range): fade the prior trend direction.
Stable regime = HOLD (no transition edge).

Source: Wilder (1978) "New Concepts in Technical Trading Systems".
        ADX > 25 trending filter is the most widely used regime classifier.

Applicable: all assets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, safe_float, true_range

MIN_ROWS = 30
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25.0
ADX_STRONG_TREND = 40.0
ADX_MOMENTUM_LOOKBACK = 5
ADX_MOMENTUM_THRESHOLD = 3.0
MAX_CONFIDENCE = 0.70


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                 period: int = ADX_PERIOD) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute ADX, +DI, -DI from OHLC data."""
    tr = true_range(high, low, close)

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    return adx, plus_di, minus_di


def _adx_regime(adx_val: float, plus_di: float, minus_di: float) -> tuple[float, str]:
    """Classify regime from current ADX and DI values."""
    if np.isnan(adx_val):
        return float("nan"), "HOLD"

    if adx_val > ADX_TREND_THRESHOLD:
        if plus_di > minus_di:
            return adx_val, "BUY"
        return adx_val, "SELL"
    return adx_val, "HOLD"


def _adx_momentum(adx: pd.Series, plus_di: pd.Series,
                  minus_di: pd.Series) -> tuple[float, str]:
    """Detect regime transitions via ADX rate of change."""
    if len(adx) < ADX_MOMENTUM_LOOKBACK + 1:
        return float("nan"), "HOLD"

    current = safe_float(adx.iloc[-1])
    past = safe_float(adx.iloc[-ADX_MOMENTUM_LOOKBACK - 1])
    if np.isnan(current) or np.isnan(past):
        return float("nan"), "HOLD"

    momentum = current - past

    if momentum > ADX_MOMENTUM_THRESHOLD and current > ADX_TREND_THRESHOLD:
        pdi = safe_float(plus_di.iloc[-1])
        mdi = safe_float(minus_di.iloc[-1])
        if pdi > mdi:
            return momentum, "BUY"
        return momentum, "SELL"

    if momentum < -ADX_MOMENTUM_THRESHOLD and current < ADX_TREND_THRESHOLD:
        pdi = safe_float(plus_di.iloc[-1])
        mdi = safe_float(minus_di.iloc[-1])
        if pdi > mdi:
            return momentum, "SELL"
        return momentum, "BUY"

    return momentum, "HOLD"


def _di_spread(plus_di: pd.Series, minus_di: pd.Series) -> tuple[float, str]:
    """Directional spread: +DI - (-DI)."""
    pdi = safe_float(plus_di.iloc[-1])
    mdi = safe_float(minus_di.iloc[-1])
    if np.isnan(pdi) or np.isnan(mdi):
        return float("nan"), "HOLD"

    spread = pdi - mdi

    if spread > 10:
        return spread, "BUY"
    if spread < -10:
        return spread, "SELL"
    return spread, "HOLD"


def compute_adx_regime_switch_signal(df: pd.DataFrame, **kwargs) -> dict:
    """Compute ADX regime switch signal.

    Returns standard signal dict with action, confidence, sub_signals.
    """
    if df is None or len(df) < MIN_ROWS:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "reason": f"insufficient data ({len(df) if df is not None else 0} rows, need {MIN_ROWS})",
        }

    adx, plus_di, minus_di = _compute_adx(df["high"], df["low"], df["close"])

    adx_val = safe_float(adx.iloc[-1])
    pdi_val = safe_float(plus_di.iloc[-1])
    mdi_val = safe_float(minus_di.iloc[-1])

    regime_val, regime_sig = _adx_regime(adx_val, pdi_val, mdi_val)
    momentum_val, momentum_sig = _adx_momentum(adx, plus_di, minus_di)
    spread_val, spread_sig = _di_spread(plus_di, minus_di)

    votes = [regime_sig, momentum_sig, spread_sig]
    action, conf = majority_vote(votes)

    if adx_val > ADX_STRONG_TREND:
        conf = min(conf * 1.15, 1.0)

    conf = min(conf, MAX_CONFIDENCE)

    regime_label = "trending" if adx_val > ADX_TREND_THRESHOLD else "ranging"

    return {
        "action": action,
        "confidence": round(conf, 4),
        "regime": regime_label,
        "sub_signals": {
            "adx_regime": {"value": round(regime_val, 2) if not np.isnan(regime_val) else None,
                           "signal": regime_sig, "regime": regime_label},
            "adx_momentum": {"value": round(momentum_val, 2) if not np.isnan(momentum_val) else None,
                             "signal": momentum_sig},
            "di_spread": {"value": round(spread_val, 2) if not np.isnan(spread_val) else None,
                          "signal": spread_sig},
        },
    }
