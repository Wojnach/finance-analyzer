"""Cubic trend persistence signal module.

Uses the cubic polynomial trend model from Bouchaud et al. (arXiv:2501.16772).
R(t+1) = b*phi(t) + c*phi(t)^3, where phi is exponentially-weighted normalized
trend strength. Weak trends persist (b>0), strong trends revert (c<0).

Sub-signals:
    1. trend_direction   — sign of phi (positive = uptrend, negative = downtrend)
    2. cubic_expected    — sign of E[R] = b*phi + c*phi^3 (net expected direction)
    3. trend_exhaustion  — contrarian when |phi| large enough that cubic dominates

Requires: open, high, low, close, volume columns, at least 80 rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, safe_float

MIN_ROWS = 80

B_DAILY = 0.0129
C_DAILY = -0.0062
B_HOURLY = 0.00132
C_HOURLY = -0.00039

PHI_CAP = 2.5
VOL_WINDOW = 60
T_LOOKBACK = 60


def _compute_weights(T: int, n_bars: int) -> np.ndarray:
    """Compute exponential weights w(n) = M*n*exp(-2n/T), normalized so sum(w^2)=1."""
    n = np.arange(1, n_bars + 1, dtype=np.float64)
    raw = n * np.exp(-2.0 * n / T)
    norm = np.sqrt(np.sum(raw**2))
    if norm < 1e-12:
        return np.zeros(n_bars)
    return raw / norm


def _compute_phi(returns_norm: np.ndarray, weights: np.ndarray) -> float:
    """Compute trend strength phi(t) = sum(w(n) * R_hat(t-n))."""
    n_bars = len(weights)
    if len(returns_norm) < n_bars:
        return 0.0
    recent = returns_norm[-n_bars:][::-1]
    phi = float(np.nansum(weights * recent))
    return np.clip(phi, -PHI_CAP, PHI_CAP)


def _expected_return(phi: float, b: float, c: float) -> float:
    """E[R(t+1)] = b*phi + c*phi^3."""
    return b * phi + c * phi**3


def _detect_timeframe(df: pd.DataFrame) -> str:
    """Guess timeframe from index spacing. Returns 'daily' or 'hourly'."""
    if len(df) < 3:
        return "daily"
    if isinstance(df.index, pd.DatetimeIndex):
        diffs = df.index.to_series().diff().dropna()
        if len(diffs) == 0:
            return "daily"
        median_hours = diffs.median().total_seconds() / 3600
        if median_hours >= 20:
            return "daily"
        return "hourly"
    return "daily"


def compute_cubic_trend_persistence_signal(
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
    log_ret = np.log(close / close.shift(1)).dropna()

    if len(log_ret) < MIN_ROWS - 1:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {},
        }

    sigma = log_ret.rolling(window=VOL_WINDOW, min_periods=20).std()
    last_sigma = sigma.iloc[-1]
    if np.isnan(last_sigma) or last_sigma < 1e-10:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {},
        }

    returns_norm = (log_ret / sigma).dropna().values
    if len(returns_norm) < T_LOOKBACK:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {},
        }

    weights = _compute_weights(T_LOOKBACK, min(T_LOOKBACK, len(returns_norm)))
    phi = _compute_phi(returns_norm, weights)

    tf = _detect_timeframe(df)
    if tf == "daily":
        b, c = B_DAILY, C_DAILY
    else:
        b, c = B_HOURLY, C_HOURLY

    e_r = _expected_return(phi, b, c)

    # --- sub-signal 1: trend direction ---
    if phi > 0.3:
        trend_dir = "BUY"
    elif phi < -0.3:
        trend_dir = "SELL"
    else:
        trend_dir = "HOLD"

    # --- sub-signal 2: cubic expected return ---
    if e_r > 0.0005:
        cubic_exp = "BUY"
    elif e_r < -0.0005:
        cubic_exp = "SELL"
    else:
        cubic_exp = "HOLD"

    # --- sub-signal 3: trend exhaustion (contrarian) ---
    phi_threshold = (-b / (3 * c)) ** 0.5 if c != 0 and (-b / (3 * c)) > 0 else 2.0
    if abs(phi) > phi_threshold:
        trend_exh = "SELL" if phi > 0 else "BUY"
    else:
        trend_exh = "HOLD"

    votes = [trend_dir, cubic_exp, trend_exh]
    action, confidence = majority_vote(votes, count_hold=False)

    confidence = min(confidence * 0.7, 0.7)

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": {
            "trend_direction": trend_dir,
            "cubic_expected": cubic_exp,
            "trend_exhaustion": trend_exh,
        },
        "indicators": {
            "phi": round(safe_float(phi), 4),
            "expected_return": round(safe_float(e_r), 6),
            "sigma": round(safe_float(last_sigma), 6),
            "phi_threshold": round(safe_float(phi_threshold), 4),
            "timeframe": tf,
            "b": b,
            "c": c,
        },
    }
