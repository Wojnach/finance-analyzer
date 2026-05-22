"""Choppiness Index regime gate signal.

Uses the Choppiness Index (CHOP) to classify market regime as choppy/ranging
vs trending. CHOP > 61.8 = choppy (suppress directional signals).
CHOP < 38.2 = trending (confirm directional signals).

Requires OHLCV DataFrame with at least 20 rows.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, safe_float, sma, true_range

logger = logging.getLogger(__name__)

MIN_ROWS = 20
CHOP_PERIOD = 14
CHOPPY_THRESHOLD = 61.8
TRENDING_THRESHOLD = 38.2
SMA_PERIOD = 14
ROC_PERIOD = 3


def _compute_choppiness(high: pd.Series, low: pd.Series, close: pd.Series,
                        period: int = CHOP_PERIOD) -> pd.Series:
    tr = true_range(high, low, close)
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    hh = high.rolling(window=period, min_periods=period).max()
    ll = low.rolling(window=period, min_periods=period).min()
    price_range = hh - ll
    price_range = price_range.replace(0, np.nan)
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    return chop


def _choppy_gate(chop_val: float) -> str:
    if np.isnan(chop_val):
        return "HOLD"
    if chop_val > CHOPPY_THRESHOLD:
        return "HOLD"
    if chop_val < TRENDING_THRESHOLD:
        return "TREND"
    return "NEUTRAL"


def _trending_direction(close: pd.Series, period: int = SMA_PERIOD) -> str:
    if len(close) < period:
        return "HOLD"
    ma = sma(close, period)
    ma_val = ma.iloc[-1]
    close_val = close.iloc[-1]
    if np.isnan(ma_val) or np.isnan(close_val):
        return "HOLD"
    if close_val > ma_val:
        return "BUY"
    if close_val < ma_val:
        return "SELL"
    return "HOLD"


def _chop_roc(chop_series: pd.Series, period: int = ROC_PERIOD) -> str:
    if len(chop_series) < period + 1:
        return "HOLD"
    recent = chop_series.dropna()
    if len(recent) < period + 1:
        return "HOLD"
    current = recent.iloc[-1]
    past = recent.iloc[-period - 1]
    if np.isnan(current) or np.isnan(past) or past == 0:
        return "HOLD"
    roc = (current - past) / past
    if roc < -0.03:
        return "BUY"
    if roc > 0.03:
        return "SELL"
    return "HOLD"


def compute_choppiness_regime_gate_signal(df: pd.DataFrame,
                                          context: dict = None) -> dict:
    if df is None or len(df) < MIN_ROWS:
        return {"action": "HOLD", "confidence": 0.0,
                "sub_signals": {}, "indicators": {}}

    try:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
    except (KeyError, ValueError):
        return {"action": "HOLD", "confidence": 0.0,
                "sub_signals": {}, "indicators": {}}

    chop_series = _compute_choppiness(high, low, close)
    chop_val = safe_float(chop_series.iloc[-1])

    regime = _choppy_gate(chop_val)

    if regime == "HOLD":
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {
                "choppy_gate": "HOLD",
                "trending_confirm": "HOLD",
                "chop_roc": "HOLD",
            },
            "indicators": {
                "choppiness_index": chop_val,
                "regime": "choppy",
            },
        }

    direction = _trending_direction(close)
    roc_vote = _chop_roc(chop_series)

    if regime == "TREND":
        votes = [direction, roc_vote]
        action, confidence = majority_vote(votes, count_hold=False)
        if action == "HOLD" and direction != "HOLD":
            action = direction
            confidence = 0.35
    else:
        votes = [direction, roc_vote]
        action, confidence = majority_vote(votes, count_hold=False)
        confidence *= 0.7

    return {
        "action": action,
        "confidence": min(confidence, 0.7),
        "sub_signals": {
            "choppy_gate": regime,
            "trending_confirm": direction,
            "chop_roc": roc_vote,
        },
        "indicators": {
            "choppiness_index": chop_val,
            "regime": "trending" if regime == "TREND" else "neutral",
            "sma_14": safe_float(sma(close, SMA_PERIOD).iloc[-1]),
        },
    }
