"""Gold-Silver ratio Kalman-filtered z-score with regime gate.

Applies a univariate Kalman filter to the gold/silver price ratio to extract
an adaptive fair-value estimate, then computes a z-score of the residual.
Extreme z-scores signal mean-reversion opportunities (high ratio = silver cheap,
low ratio = gold cheap). A regime gate restricts MR trades to non-trending periods.

Sub-signals:
    1. Kalman Z-Score MR: residual z-score > threshold
    2. Ratio Level: absolute ratio above/below historical bands
    3. Kalman Trend: direction and magnitude of Kalman state drift
    4. Regime Stability: suppress signals during strong trends

Applicable to XAU-USD and XAG-USD only. For XAG the ratio is inverted
in interpretation (high G/S = silver undervalued = BUY silver).

Source: SSRN Gold-Silver Ratio MR with ML Regime Gate (2026).
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, safe_float

logger = logging.getLogger("portfolio.signals.gs_kalman_zscore_regime")

_METALS_TICKERS = {"XAU-USD", "XAG-USD"}
MIN_ROWS = 60

_KALMAN_Q = 0.001
_KALMAN_R = 0.5
_KALMAN_INIT_P = 1.0

_Z_BUY = 1.5
_Z_SELL = -1.5
_Z_STRONG = 2.5

_RATIO_HIGH_BAND = 90.0
_RATIO_LOW_BAND = 65.0

_TREND_WINDOW = 20
_TREND_THRESHOLD = 0.3

_REGIME_ADX_TRENDING = 30


def _kalman_filter_1d(observations: np.ndarray, Q: float, R: float,
                      init_P: float) -> tuple[np.ndarray, np.ndarray]:
    """Univariate Kalman filter. Returns (estimates, uncertainties)."""
    n = len(observations)
    x = np.zeros(n)
    P = np.zeros(n)

    x[0] = observations[0]
    P[0] = init_P

    for t in range(1, n):
        x_pred = x[t - 1]
        P_pred = P[t - 1] + Q

        K = P_pred / (P_pred + R)
        x[t] = x_pred + K * (observations[t] - x_pred)
        P[t] = (1 - K) * P_pred

    return x, P


def _compute_gs_ratio(df: pd.DataFrame, context: dict | None) -> np.ndarray | None:
    """Build gold/silver ratio from df or context."""
    if context and "gs_ratio_series" in context:
        series = context["gs_ratio_series"]
        if hasattr(series, "values"):
            return np.asarray(series.values, dtype=float)
        return np.asarray(series, dtype=float)

    close = df["close"].values if "close" in df.columns else None
    if close is None or len(close) < MIN_ROWS:
        return None

    ticker = (context or {}).get("ticker", "")

    if ticker == "XAU-USD":
        silver_close = (context or {}).get("silver_close")
        if silver_close is not None:
            if hasattr(silver_close, "values"):
                silver_close = silver_close.values
            silver_close = np.asarray(silver_close, dtype=float)
            n = min(len(close), len(silver_close))
            if n >= MIN_ROWS:
                return close[-n:] / silver_close[-n:]
    elif ticker == "XAG-USD":
        gold_close = (context or {}).get("gold_close")
        if gold_close is not None:
            if hasattr(gold_close, "values"):
                gold_close = gold_close.values
            gold_close = np.asarray(gold_close, dtype=float)
            n = min(len(close), len(gold_close))
            if n >= MIN_ROWS:
                return gold_close[-n:] / close[-n:]

    return None


def _adx_from_ohlc(df: pd.DataFrame, period: int = 14) -> float:
    """Compute latest ADX from OHLCV DataFrame."""
    if len(df) < period * 2:
        return 0.0
    try:
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)

        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        alpha = 1.0 / period
        atr = np.zeros(len(tr))
        plus_di_s = np.zeros(len(tr))
        minus_di_s = np.zeros(len(tr))
        atr[0] = tr[0]
        plus_di_s[0] = plus_dm[0]
        minus_di_s[0] = minus_dm[0]

        for i in range(1, len(tr)):
            atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha
            plus_di_s[i] = plus_di_s[i - 1] * (1 - alpha) + plus_dm[i] * alpha
            minus_di_s[i] = minus_di_s[i - 1] * (1 - alpha) + minus_dm[i] * alpha

        with np.errstate(divide="ignore", invalid="ignore"):
            plus_di = 100 * plus_di_s / atr
            minus_di = 100 * minus_di_s / atr
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

        adx = np.zeros(len(dx))
        adx[0] = dx[0]
        for i in range(1, len(dx)):
            adx[i] = adx[i - 1] * (1 - alpha) + dx[i] * alpha

        val = float(adx[-1])
        return val if np.isfinite(val) else 0.0
    except Exception:
        return 0.0


def compute_gs_kalman_zscore_regime_signal(
    df: pd.DataFrame, context: dict | None = None, **kwargs,
) -> dict:
    """Compute Gold-Silver Kalman z-score regime signal."""
    empty = {
        "action": "HOLD", "confidence": 0.0,
        "sub_signals": {}, "indicators": {},
    }

    if df is None or len(df) < MIN_ROWS:
        return empty

    context = context or {}
    ticker = context.get("ticker", kwargs.get("ticker", ""))
    if ticker not in _METALS_TICKERS:
        return empty

    gs_ratio = _compute_gs_ratio(df, context)
    if gs_ratio is None or len(gs_ratio) < MIN_ROWS:
        return empty

    mask = np.isfinite(gs_ratio)
    if mask.sum() < MIN_ROWS:
        return empty
    gs_ratio = gs_ratio[mask]

    estimates, uncertainties = _kalman_filter_1d(gs_ratio, _KALMAN_Q, _KALMAN_R, _KALMAN_INIT_P)

    residuals = gs_ratio - estimates

    lookback = min(len(residuals), 120)
    recent_residuals = residuals[-lookback:]
    std = float(np.std(recent_residuals))
    if std < 1e-6:
        return empty

    current_residual = float(residuals[-1])
    z_score = current_residual / std

    current_ratio = float(gs_ratio[-1])
    kalman_est = float(estimates[-1])
    kalman_unc = float(uncertainties[-1])

    trend_window = min(_TREND_WINDOW, len(estimates) - 1)
    kalman_drift = float(estimates[-1] - estimates[-1 - trend_window])
    drift_per_bar = kalman_drift / trend_window if trend_window > 0 else 0.0

    adx = _adx_from_ohlc(df)
    regime_trending = adx > _REGIME_ADX_TRENDING

    votes = []
    sub_signals = {}
    is_silver = ticker == "XAG-USD"

    if z_score > _Z_STRONG:
        sub_signals["kalman_zscore"] = "BUY" if is_silver else "SELL"
    elif z_score > _Z_BUY:
        sub_signals["kalman_zscore"] = "BUY" if is_silver else "SELL"
    elif z_score < -_Z_STRONG:
        sub_signals["kalman_zscore"] = "SELL" if is_silver else "BUY"
    elif z_score < _Z_SELL:
        sub_signals["kalman_zscore"] = "SELL" if is_silver else "BUY"
    else:
        sub_signals["kalman_zscore"] = "HOLD"
    votes.append(sub_signals["kalman_zscore"])

    if current_ratio > _RATIO_HIGH_BAND:
        sub_signals["ratio_level"] = "BUY" if is_silver else "SELL"
    elif current_ratio < _RATIO_LOW_BAND:
        sub_signals["ratio_level"] = "SELL" if is_silver else "BUY"
    else:
        sub_signals["ratio_level"] = "HOLD"
    votes.append(sub_signals["ratio_level"])

    if abs(drift_per_bar) > _TREND_THRESHOLD:
        if drift_per_bar > 0:
            sub_signals["kalman_trend"] = "BUY" if is_silver else "SELL"
        else:
            sub_signals["kalman_trend"] = "SELL" if is_silver else "BUY"
    else:
        sub_signals["kalman_trend"] = "HOLD"
    votes.append(sub_signals["kalman_trend"])

    if regime_trending:
        sub_signals["regime_gate"] = "HOLD"
    else:
        sub_signals["regime_gate"] = sub_signals["kalman_zscore"]
    votes.append(sub_signals["regime_gate"])

    action, confidence = majority_vote(votes, count_hold=False)

    if abs(z_score) > _Z_STRONG:
        confidence = min(confidence * 1.2, 1.0)
    if regime_trending:
        confidence *= 0.6

    confidence = min(confidence, 0.7)

    indicators = {
        "gs_ratio": safe_float(current_ratio),
        "kalman_estimate": safe_float(kalman_est),
        "kalman_uncertainty": safe_float(kalman_unc),
        "kalman_zscore": safe_float(z_score),
        "kalman_drift_per_bar": safe_float(drift_per_bar),
        "residual_std": safe_float(std),
        "adx": safe_float(adx),
        "regime_trending": regime_trending,
    }

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": sub_signals,
        "indicators": indicators,
    }
