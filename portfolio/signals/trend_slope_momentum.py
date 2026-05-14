"""Trend Slope Momentum signal module.

EMA-smoothed log-price slope z-score blended with 50-day momentum
confirmation. Produces a continuous probability signal, not a binary
crossover. Based on the Forecast-to-Fill methodology (arxiv 2511.08571).

Sub-signals:
    1. Trend Slope     — z-score of EMA-smoothed log-price slope
    2. Momentum 50d    — binary 50-day price momentum confirmation
    3. Probability     — 0.6 * trend + 0.4 * momentum blend
    4. Slope Direction — slope sign agreement gate

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least 260 rows of data (252 for z-score rolling window + buffer).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, safe_float

logger = logging.getLogger(__name__)

MIN_ROWS = 260
EMA_LAMBDA = 0.94
MOMENTUM_LOOKBACK = 50
Z_LOOKBACK = 252
THRESHOLD = 0.52
Z_CLIP = 3.0
TREND_WEIGHT = 0.6
MOMENTUM_WEIGHT = 0.4


def _ema_smooth(series: pd.Series, lam: float) -> pd.Series:
    """Exponential moving average on log prices with decay factor lambda."""
    result = np.empty(len(series))
    result[0] = series.iloc[0]
    for i in range(1, len(series)):
        result[i] = lam * result[i - 1] + (1 - lam) * series.iloc[i]
    return pd.Series(result, index=series.index)


def compute_trend_slope_momentum_signal(
    df: pd.DataFrame, context: dict = None
) -> dict:
    if df is None or len(df) < MIN_ROWS:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {},
        }

    close = df["close"].dropna()
    if len(close) < MIN_ROWS:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {},
        }

    log_close = np.log(close.astype(float))
    y_smooth = _ema_smooth(log_close, EMA_LAMBDA)

    slope = y_smooth.diff()
    slope_clean = slope.dropna()

    if len(slope_clean) < Z_LOOKBACK + 1:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {},
        }

    rolling_mean = slope_clean.rolling(Z_LOOKBACK).mean()
    rolling_std = slope_clean.rolling(Z_LOOKBACK).std()

    last_std = rolling_std.iloc[-1]
    if np.isnan(last_std) or last_std < 1e-12:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {},
        }

    z = (slope_clean.iloc[-1] - rolling_mean.iloc[-1]) / last_std
    z_clipped = float(np.clip(z, -Z_CLIP, Z_CLIP))
    p_trend = (z_clipped + Z_CLIP) / (2.0 * Z_CLIP)

    current_price = float(close.iloc[-1])
    past_price = float(close.iloc[-MOMENTUM_LOOKBACK])
    momentum = 1.0 if current_price / past_price > 1.0 else 0.0

    p_bull = TREND_WEIGHT * p_trend + MOMENTUM_WEIGHT * momentum
    p_bear = 1.0 - p_bull

    current_slope = float(slope_clean.iloc[-1])

    votes = []

    # Sub-signal 1: Trend slope direction
    if z_clipped > 0.5:
        trend_vote = "BUY"
    elif z_clipped < -0.5:
        trend_vote = "SELL"
    else:
        trend_vote = "HOLD"
    votes.append(trend_vote)

    # Sub-signal 2: 50-day momentum
    if momentum > 0.5:
        mom_vote = "BUY"
    else:
        mom_vote = "SELL"
    votes.append(mom_vote)

    # Sub-signal 3: Probability threshold
    if p_bull >= THRESHOLD and current_slope > 0:
        prob_vote = "BUY"
    elif p_bear >= THRESHOLD and current_slope < 0:
        prob_vote = "SELL"
    else:
        prob_vote = "HOLD"
    votes.append(prob_vote)

    # Sub-signal 4: Slope-momentum agreement
    slope_agrees_with_momentum = (current_slope > 0 and momentum > 0.5) or (
        current_slope < 0 and momentum < 0.5
    )
    if slope_agrees_with_momentum:
        agreement_vote = "BUY" if current_slope > 0 else "SELL"
    else:
        agreement_vote = "HOLD"
    votes.append(agreement_vote)

    action, confidence = majority_vote(votes, count_hold=False)

    # Scale confidence by how far probability is from threshold
    if action == "BUY":
        prob_strength = min((p_bull - 0.5) * 4.0, 1.0)
    elif action == "SELL":
        prob_strength = min((p_bear - 0.5) * 4.0, 1.0)
    else:
        prob_strength = 0.0
    confidence = max(0.0, min(confidence * max(prob_strength, 0.3), 1.0))

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": {
            "trend_slope": trend_vote,
            "momentum_50d": mom_vote,
            "probability": prob_vote,
            "slope_momentum_agreement": agreement_vote,
        },
        "indicators": {
            "z_score": safe_float(z_clipped),
            "p_trend": safe_float(p_trend),
            "p_bull": safe_float(p_bull),
            "slope": safe_float(current_slope),
            "momentum_ratio": safe_float(current_price / past_price),
        },
    }
