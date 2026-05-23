"""AutoTune Adaptive Cycle signal module.

Ehlers autocorrelation periodogram detects dominant cycle period, then
adaptive bandpass filter tuned to that period generates BUY/SELL via
rate-of-change zero-crossing. Adapts to changing market cycles.

Source: Ehlers, J.F., "A Rolling Autocorrelation Function", TASC May 2026.

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least 60 rows of data.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, safe_float

MIN_ROWS = 60


def _highpass_filter(data: np.ndarray, period: float = 48.0) -> np.ndarray:
    """2-pole Butterworth high-pass filter to remove trend."""
    n = len(data)
    hp = np.zeros(n)
    alpha1 = (math.cos(0.707 * 2 * math.pi / period) + math.sin(0.707 * 2 * math.pi / period) - 1) / math.cos(0.707 * 2 * math.pi / period)
    for i in range(2, n):
        hp[i] = (1 - alpha1 / 2) * (1 - alpha1 / 2) * (data[i] - 2 * data[i - 1] + data[i - 2]) + 2 * (1 - alpha1) * hp[i - 1] - (1 - alpha1) * (1 - alpha1) * hp[i - 2]
    return hp


def _supersmoother(data: np.ndarray, period: float = 10.0) -> np.ndarray:
    """Ehlers 2-pole SuperSmoother."""
    n = len(data)
    filt = np.zeros(n)
    a1 = math.exp(-1.414 * math.pi / period)
    b1 = 2 * a1 * math.cos(1.414 * math.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    for i in range(2, n):
        filt[i] = c1 * (data[i] + data[i - 1]) / 2 + c2 * filt[i - 1] + c3 * filt[i - 2]
    return filt


def _autocorrelation_periodogram(hp: np.ndarray, window: int = 50,
                                  min_period: int = 8, max_period: int = 48) -> tuple[float, float]:
    """Detect dominant cycle via autocorrelation periodogram.

    Returns (dominant_cycle_period, min_correlation_value).
    """
    n = len(hp)
    if n < window + max_period:
        return 20.0, 0.0

    filt = _supersmoother(hp, 10.0)
    end = n - 1

    corr = np.zeros(max_period + 1)
    for lag in range(0, max_period + 1):
        if lag == 0:
            corr[lag] = 1.0
            continue
        x = filt[end - window + 1:end + 1]
        y = filt[end - window + 1 - lag:end + 1 - lag]
        if len(x) < 5 or len(y) < 5:
            continue
        mx, my = np.mean(x), np.mean(y)
        sx, sy = np.std(x, ddof=0), np.std(y, ddof=0)
        if sx < 1e-10 or sy < 1e-10:
            continue
        corr[lag] = np.mean((x - mx) * (y - my)) / (sx * sy)

    min_corr = 0.0
    dc = 20.0
    for lag in range(min_period // 2, max_period // 2 + 1):
        if corr[lag] < min_corr:
            min_corr = corr[lag]
            dc = 2.0 * lag

    dc = max(min_period, min(max_period, dc))
    return dc, min_corr


def _adaptive_bandpass(data: np.ndarray, period: float,
                       bandwidth: float = 0.22) -> np.ndarray:
    """2-pole IIR bandpass filter with adaptive center frequency."""
    n = len(data)
    bp = np.zeros(n)

    if period < 4:
        period = 4.0

    l1 = math.cos(2 * math.pi / period)
    g1 = math.cos(bandwidth * 2 * math.pi / period)
    if g1 == 0:
        return bp
    s1 = 1.0 / g1 - math.sqrt(max(0, 1.0 / (g1 * g1) - 1))

    for i in range(2, n):
        bp[i] = 0.5 * (1 - s1) * (data[i] - data[i - 2]) + l1 * (1 + s1) * bp[i - 1] - s1 * bp[i - 2]

    return bp


def compute_autotune_adaptive_cycle_signal(df: pd.DataFrame, context: dict = None) -> dict:
    """Compute AutoTune Adaptive Cycle signal.

    Args:
        df: DataFrame with columns: open, high, low, close, volume
        context: Optional dict with keys: ticker, config, asset_class, regime

    Returns:
        dict with keys: action, confidence, sub_signals, indicators
    """
    if df is None or len(df) < MIN_ROWS:
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    try:
        close = df["close"].values.astype(float)
        close = close[~np.isnan(close)]
        if len(close) < MIN_ROWS:
            return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

        hp = _highpass_filter(close, 48.0)
        dc, min_corr = _autocorrelation_periodogram(hp, window=50, min_period=8, max_period=48)
        bp = _adaptive_bandpass(hp, dc, bandwidth=0.22)

        n = len(bp)
        if n < 5:
            return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

        roc = bp[n - 1] - bp[n - 3]
        roc_prev = bp[n - 2] - bp[n - 4]

        bp_vote = "HOLD"
        if min_corr < -0.22:
            if roc > 0 and roc_prev <= 0:
                bp_vote = "BUY"
            elif roc < 0 and roc_prev >= 0:
                bp_vote = "SELL"

        roc_magnitude = abs(roc)
        bp_std = np.std(bp[max(0, n - 50):n])
        if bp_std > 1e-10:
            roc_zscore = roc_magnitude / bp_std
        else:
            roc_zscore = 0.0

        trend_vote = "HOLD"
        if roc > 0:
            trend_vote = "BUY"
        elif roc < 0:
            trend_vote = "SELL"

        corr_strength_vote = "HOLD"
        if min_corr < -0.5:
            corr_strength_vote = trend_vote

        votes = [bp_vote, trend_vote, corr_strength_vote]
        action, confidence = majority_vote(votes, count_hold=False)

        conf_scale = min(1.0, roc_zscore / 2.0)
        confidence = min(0.7, confidence * conf_scale)

        if min_corr > -0.15:
            action = "HOLD"
            confidence = 0.0

        return {
            "action": action,
            "confidence": round(confidence, 4),
            "sub_signals": {
                "bandpass_crossover": bp_vote,
                "trend_direction": trend_vote,
                "correlation_strength": corr_strength_vote,
            },
            "indicators": {
                "dominant_cycle": safe_float(dc),
                "min_correlation": safe_float(min_corr),
                "bandpass_roc": safe_float(roc),
                "bandpass_roc_zscore": safe_float(roc_zscore),
                "bandpass_value": safe_float(bp[n - 1]),
            },
        }
    except Exception:
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}
