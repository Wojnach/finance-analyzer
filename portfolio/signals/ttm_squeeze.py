"""TTM Squeeze breakout signal module.

Detects volatility compression (Bollinger Bands inside Keltner Channels)
then uses a momentum histogram to predict breakout direction. The signal
fires on squeeze RELEASE — when compression ends and momentum is clear.

Sub-indicators:
    1. Squeeze State     — BB inside KC = squeeze ON; release = actionable
    2. Momentum Direction — Linear regression of price deviation from midline
    3. Momentum Acceleration — Rising/falling histogram confirms direction

When squeeze has been ON for 3+ bars and then releases:
    - Momentum > 0 and rising → BUY
    - Momentum < 0 and falling → SELL
    - Otherwise → HOLD

Filtered TTM Squeeze signals show 68-72% directional accuracy on daily charts.

Source: John Carter (Simpler Trading); TrendSpider 2025; Deepvue 2025.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from portfolio.signal_utils import ema, majority_vote, safe_float, sma

logger = logging.getLogger(__name__)

MIN_ROWS = 25


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.rolling(period, min_periods=period).mean()


def _bollinger_bands(close: pd.Series, period: int = 20, mult: float = 2.0):
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    upper = mid + mult * std
    lower = mid - mult * std
    return upper, mid, lower


def _keltner_channels(close: pd.Series, high: pd.Series, low: pd.Series,
                      period: int = 20, mult: float = 1.5):
    mid = close.ewm(span=period, adjust=False).mean()
    atr = _atr(high, low, close, period)
    upper = mid + mult * atr
    lower = mid - mult * atr
    return upper, mid, lower


def _linreg_value(series: pd.Series, period: int = 20) -> float:
    if len(series) < period:
        return 0.0
    y = series.iloc[-period:].values
    if np.any(np.isnan(y)):
        y = pd.Series(y).ffill().bfill().values
    x = np.arange(period, dtype=float)
    n = len(x)
    if n < 2:
        return 0.0
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx == 0:
        return 0.0
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    return float(slope * (n - 1) + intercept)


def compute_ttm_squeeze_signal(df: pd.DataFrame, context: dict = None) -> dict:
    if df is None or len(df) < MIN_ROWS:
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    bb_period = 20
    bb_mult = 2.0
    kc_period = 20
    kc_mult = 1.5
    min_squeeze_bars = 2

    bb_upper, bb_mid, bb_lower = _bollinger_bands(close, bb_period, bb_mult)
    kc_upper, kc_mid, kc_lower = _keltner_channels(close, high, low, kc_period, kc_mult)

    squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)

    midline = (high.rolling(bb_period).max() + low.rolling(bb_period).min()) / 2
    avg_mid = (midline + bb_mid) / 2
    deviation = close - avg_mid

    mom_current = _linreg_value(deviation, bb_period)

    dev_shifted = close.shift(1) - avg_mid.shift(1)
    if len(dev_shifted.dropna()) >= bb_period:
        mom_prev = _linreg_value(dev_shifted.dropna(), bb_period)
    else:
        mom_prev = 0.0

    squeeze_count = 0
    for i in range(len(squeeze_on) - 1, -1, -1):
        if squeeze_on.iloc[i]:
            squeeze_count += 1
        else:
            break

    was_squeezing = squeeze_count == 0 and len(squeeze_on) > 1
    if not was_squeezing:
        lookback = min(20, len(squeeze_on))
        recent = squeeze_on.iloc[-lookback:]
        off_idx = None
        for i in range(len(recent) - 1, -1, -1):
            if not recent.iloc[i]:
                off_idx = i
                break
        if off_idx is not None and off_idx > 0:
            consecutive = 0
            for i in range(off_idx - 1, -1, -1):
                if recent.iloc[i]:
                    consecutive += 1
                else:
                    break
            was_squeezing = consecutive >= min_squeeze_bars and off_idx == len(recent) - 1

    currently_squeezing = squeeze_count >= min_squeeze_bars

    # Check for release on current bar OR 1 bar ago (recent release window)
    just_released = False
    had_enough_squeeze = False

    if len(squeeze_on) >= 2:
        # Release on current bar
        if not squeeze_on.iloc[-1] and squeeze_on.iloc[-2]:
            just_released = True
            count_before = 0
            for i in range(len(squeeze_on) - 2, -1, -1):
                if squeeze_on.iloc[i]:
                    count_before += 1
                else:
                    break
            had_enough_squeeze = count_before >= min_squeeze_bars

        # Release 1 bar ago (still in post-release window)
        elif len(squeeze_on) >= 3 and not squeeze_on.iloc[-1] and not squeeze_on.iloc[-2] and squeeze_on.iloc[-3]:
            just_released = True
            count_before = 0
            for i in range(len(squeeze_on) - 3, -1, -1):
                if squeeze_on.iloc[i]:
                    count_before += 1
                else:
                    break
            had_enough_squeeze = count_before >= min_squeeze_bars

    squeeze_vote = "HOLD"
    if currently_squeezing:
        squeeze_vote = "HOLD"
    elif just_released and had_enough_squeeze:
        squeeze_vote = "BUY" if mom_current > 0 else "SELL"

    mom_vote = "HOLD"
    if mom_current > 0 and mom_current > mom_prev:
        mom_vote = "BUY"
    elif mom_current < 0 and mom_current < mom_prev:
        mom_vote = "SELL"

    accel_vote = "HOLD"
    mom_delta = mom_current - mom_prev
    if mom_delta > 0 and mom_current > 0:
        accel_vote = "BUY"
    elif mom_delta < 0 and mom_current < 0:
        accel_vote = "SELL"

    # Gate: signal ONLY fires on squeeze release. Outside of release events,
    # always HOLD. This is the core edge — selectivity, not frequency.
    if not (just_released and had_enough_squeeze):
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {
                "squeeze_state": "HOLD" if currently_squeezing else "HOLD",
                "momentum_direction": mom_vote,
                "momentum_acceleration": accel_vote,
            },
            "indicators": {
                "squeeze_on": bool(squeeze_on.iloc[-1]) if len(squeeze_on) > 0 else False,
                "squeeze_bars": squeeze_count,
                "just_released": False,
                "momentum": safe_float(mom_current),
                "momentum_prev": safe_float(mom_prev),
                "bb_width": safe_float((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_mid.iloc[-1]) if len(bb_mid) > 0 and bb_mid.iloc[-1] != 0 else 0.0,
            },
        }

    # Squeeze just released — use momentum for direction
    votes = [squeeze_vote, mom_vote, accel_vote]
    action, confidence = majority_vote(votes, count_hold=False)
    confidence = min(confidence, 0.7)

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": {
            "squeeze_state": squeeze_vote,
            "momentum_direction": mom_vote,
            "momentum_acceleration": accel_vote,
        },
        "indicators": {
            "squeeze_on": bool(squeeze_on.iloc[-1]) if len(squeeze_on) > 0 else False,
            "squeeze_bars": squeeze_count,
            "just_released": just_released and had_enough_squeeze,
            "momentum": safe_float(mom_current),
            "momentum_prev": safe_float(mom_prev),
            "bb_width": safe_float((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_mid.iloc[-1]) if len(bb_mid) > 0 and bb_mid.iloc[-1] != 0 else 0.0,
        },
    }
