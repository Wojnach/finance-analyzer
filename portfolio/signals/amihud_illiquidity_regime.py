"""Amihud illiquidity regime signal.

Detects liquidity regimes using the Amihud ILLIQ ratio (|return|/dollar_volume).
High ILLIQ = thin market where breakouts are fakeouts. Low ILLIQ = thick market
where breakouts have conviction.

Sub-indicators:
    1. ILLIQ Z-Score    (current illiquidity vs 60d rolling distribution)
    2. ILLIQ Trend      (rising = deteriorating liquidity, falling = improving)
    3. Volume Confirm   (volume above/below SMA as confirmation)

Requires: open, high, low, close, volume with at least 70 rows.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, safe_float, sma

logger = logging.getLogger(__name__)

MIN_ROWS = 70


def _amihud_illiq(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute Amihud ILLIQ = |return| / dollar_volume."""
    ret = close.pct_change(fill_method=None).abs()
    dollar_vol = close * volume
    dollar_vol = dollar_vol.replace(0, np.nan)
    return ret / dollar_vol


def compute_amihud_illiquidity_regime_signal(
    df: pd.DataFrame, context: dict = None
) -> dict:
    if df is None or len(df) < MIN_ROWS:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {},
        }

    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    illiq = _amihud_illiq(close, volume)
    if illiq.isna().all():
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {},
        }

    lookback = 60
    roll_mean = illiq.rolling(lookback, min_periods=30).mean()
    roll_std = illiq.rolling(lookback, min_periods=30).std()
    roll_std = roll_std.replace(0, np.nan)
    z = (illiq - roll_mean) / roll_std

    cur_z = safe_float(z.iloc[-1])
    if np.isnan(cur_z):
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {"illiq_z": float("nan")},
        }

    # Sub 1: ILLIQ Z-Score regime
    if cur_z > 2.0:
        z_vote = "SELL"
    elif cur_z < -1.0:
        z_vote = "BUY"
    else:
        z_vote = "HOLD"

    # Sub 2: ILLIQ trend (5-bar slope of z-score)
    z_recent = z.dropna().tail(10)
    if len(z_recent) >= 5:
        slope = (z_recent.iloc[-1] - z_recent.iloc[-5]) / 5.0
        slope_val = safe_float(slope)
        if slope_val > 0.3:
            trend_vote = "SELL"
        elif slope_val < -0.3:
            trend_vote = "BUY"
        else:
            trend_vote = "HOLD"
    else:
        slope_val = 0.0
        trend_vote = "HOLD"

    # Sub 3: Volume confirmation
    vol_sma = sma(volume, 20)
    cur_vol = safe_float(volume.iloc[-1])
    cur_vol_sma = safe_float(vol_sma.iloc[-1])
    if cur_vol_sma > 0:
        rvol = cur_vol / cur_vol_sma
    else:
        rvol = 1.0

    if rvol > 1.3:
        vol_vote = "BUY"
    elif rvol < 0.6:
        vol_vote = "SELL"
    else:
        vol_vote = "HOLD"

    votes = [z_vote, trend_vote, vol_vote]
    action, confidence = majority_vote(votes, count_hold=False)

    confidence = min(confidence, 0.7)

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": {
            "illiq_z": z_vote,
            "illiq_trend": trend_vote,
            "volume_confirm": vol_vote,
        },
        "indicators": {
            "illiq_z_score": cur_z,
            "illiq_trend_slope": slope_val,
            "rvol": round(rvol, 3),
            "illiq_raw": safe_float(illiq.iloc[-1]),
        },
    }
