"""Shared signal utility functions.

Canonical implementations of common technical-analysis helpers used across
multiple signal modules.  Import from here instead of duplicating locally.

All functions operate on ``pd.Series`` inputs and return ``pd.Series``
(or ``float`` for ``safe_float``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average.

    Returns NaN where insufficient data (min_periods = period).
    """
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average using pandas ewm.

    Uses ``adjust=False`` for recursive EMA (standard in TA).
    """
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI.

    Uses ``clip()`` for separating gains/losses (numerically stable).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def safe_float(val) -> float:
    """Convert *val* to float, returning ``NaN`` for non-finite / missing values."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return float("nan")
    try:
        f = float(val)
        return f if np.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothed moving average (RMA / SMMA).

    Equivalent to EMA with ``alpha = 1 / period``.
    """
    alpha = 1.0 / period
    return series.ewm(alpha=alpha, min_periods=period, adjust=False).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average with linearly increasing weights.

    Weight_i = i + 1 for i in 0..period-1 (most recent bar has highest weight).
    """
    weights = np.arange(1, period + 1, dtype=float)

    def _apply_wma(x: np.ndarray) -> float:
        return np.dot(x, weights) / weights.sum()

    return series.rolling(window=period, min_periods=period).apply(
        _apply_wma, raw=True,
    )


def roc(series: pd.Series, period: int) -> pd.Series:
    """Rate of Change: ``100 * (current - n_periods_ago) / n_periods_ago``."""
    shifted = series.shift(period)
    return 100.0 * (series - shifted) / shifted.replace(0, np.nan)
