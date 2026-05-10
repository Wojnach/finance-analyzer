"""VWAP Z-Score mean reversion signal module.

Uses volume-weighted average price with standard deviation bands as dynamic
support/resistance. BUY when price deviates significantly below VWAP
(oversold vs volume-weighted fair value), SELL when significantly above.

Sub-signals:
    1. vwap_z        — z-score distance from rolling VWAP
    2. vwap_slope    — VWAP trend direction (rising = bullish context)
    3. volume_confirm — volume above average confirms signal conviction

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least 30 rows of data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, safe_float, sma

MIN_ROWS = 30


def _rolling_vwap(df: pd.DataFrame, period: int) -> pd.Series:
    """Compute rolling VWAP over a fixed window."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = tp * df["volume"]
    vol_sum = df["volume"].rolling(period).sum().replace(0, np.nan)
    vwap = tp_vol.rolling(period).sum() / vol_sum
    return vwap


def compute_vwap_zscore_mr_signal(df: pd.DataFrame, context: dict = None) -> dict:
    """Compute VWAP Z-Score mean reversion signal."""
    if df is None or len(df) < MIN_ROWS:
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    close = df["close"]
    volume = df["volume"]
    period = 20

    try:
        vwap = _rolling_vwap(df, period)
        if vwap.isna().all():
            return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

        deviation = close - vwap
        dev_std = deviation.rolling(period).std()

        valid_std = dev_std.iloc[-1]
        if np.isnan(valid_std) or valid_std < 1e-10:
            return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

        z = deviation.iloc[-1] / valid_std
        z = safe_float(z)

        # Sub-signal 1: VWAP Z-score
        z_buy_thresh = -2.0
        z_sell_thresh = 2.0
        if z < z_buy_thresh:
            vwap_z_vote = "BUY"
        elif z > z_sell_thresh:
            vwap_z_vote = "SELL"
        else:
            vwap_z_vote = "HOLD"

        # Sub-signal 2: VWAP slope (trend context)
        vwap_slope_window = 5
        if len(vwap.dropna()) >= vwap_slope_window:
            vwap_recent = vwap.dropna().iloc[-vwap_slope_window:]
            slope_pct = (vwap_recent.iloc[-1] / vwap_recent.iloc[0] - 1) * 100
            slope_pct = safe_float(slope_pct)
            if slope_pct > 0.1:
                slope_vote = "BUY"
            elif slope_pct < -0.1:
                slope_vote = "SELL"
            else:
                slope_vote = "HOLD"
        else:
            slope_pct = 0.0
            slope_vote = "HOLD"

        # Sub-signal 3: Volume confirmation
        vol_sma = sma(volume, period)
        vol_sma_last = vol_sma.iloc[-1]
        vol_ratio = safe_float(volume.iloc[-1] / vol_sma_last) if (pd.notna(vol_sma_last) and vol_sma_last > 0) else 1.0
        vol_confirm_mult = 1.2
        if vol_ratio > vol_confirm_mult:
            vol_vote = vwap_z_vote if vwap_z_vote != "HOLD" else "HOLD"
        else:
            vol_vote = "HOLD"

        votes = [vwap_z_vote, slope_vote, vol_vote]
        action, confidence = majority_vote(votes, count_hold=False)

        active_votes = sum(1 for v in votes if v != "HOLD")
        if active_votes < 2:
            action, confidence = "HOLD", 0.0

        # Scale confidence by z-score magnitude (further from VWAP = higher conviction)
        if action != "HOLD":
            z_magnitude = min(abs(z) / 3.0, 1.0)
            confidence = confidence * (0.5 + 0.5 * z_magnitude)
            confidence = min(confidence, 0.85)

        return {
            "action": action,
            "confidence": round(confidence, 4),
            "sub_signals": {
                "vwap_z": vwap_z_vote,
                "vwap_slope": slope_vote,
                "volume_confirm": vol_vote,
            },
            "indicators": {
                "vwap_z_score": z,
                "vwap_value": safe_float(vwap.iloc[-1]),
                "vwap_slope_pct": slope_pct,
                "volume_ratio": vol_ratio,
                "close": safe_float(close.iloc[-1]),
            },
        }

    except Exception:
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}
