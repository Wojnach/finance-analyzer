"""Rolling z-score feature normalizer for signal inputs.

Maintains per-ticker, per-indicator rolling statistics (mean, std) and
provides z-score normalization.  This makes signal thresholds regime-adaptive:
a RSI of 72 in low-vol is more extreme than RSI 72 in high-vol.

Usage:
    from portfolio.feature_normalizer import normalize, update

    # Each signal cycle, update with raw value
    update("XAG-USD", "rsi_14", 72.0)

    # Get z-score for threshold comparison
    z = normalize("XAG-USD", "rsi_14", 72.0)  # returns z-score or raw if insufficient data

Research basis: "Standardization (Z-score): Convert features to zero mean/unit variance"
from quantitative signals research paper on 1-3h metals forecasting.
"""
from __future__ import annotations

import logging
from collections import deque

import numpy as np

logger = logging.getLogger("portfolio.feature_normalizer")

_DEFAULT_WINDOW = 100  # rolling window size for stats
_MIN_SAMPLES = 20      # minimum samples before z-scoring (otherwise return raw)

# In-memory storage: {(ticker, indicator_name): deque of values}
_buffers: dict[tuple[str, str], deque] = {}


def _ensure_buffer(ticker: str, indicator: str) -> deque:
    """Get or create the rolling buffer for a ticker+indicator pair."""
    key = (ticker, indicator)
    if key not in _buffers:
        _buffers[key] = deque(maxlen=_DEFAULT_WINDOW)
    return _buffers[key]


def update(ticker: str, indicator: str, value: float) -> None:
    """Record a new raw value for a ticker+indicator pair.

    Call this each signal cycle to build up the rolling distribution.
    """
    if not np.isfinite(value):
        return
    buf = _ensure_buffer(ticker, indicator)
    buf.append(value)


def normalize(ticker: str, indicator: str, value: float) -> float:
    """Z-score normalize a value against its rolling distribution.

    Returns the z-score if sufficient history exists (>= _MIN_SAMPLES),
    otherwise returns the raw value unchanged.  This ensures cold-start
    safety: signals work with raw thresholds until enough data accumulates.
    """
    if not np.isfinite(value):
        return 0.0
    buf = _ensure_buffer(ticker, indicator)
    if len(buf) < _MIN_SAMPLES:
        return value
    arr = np.array(buf, dtype=float)
    mean = arr.mean()
    std = arr.std()
    if std < 1e-12:
        return 0.0
    return float((value - mean) / std)


def has_sufficient_history(ticker: str, indicator: str) -> bool:
    """Check if enough samples have accumulated for z-scoring."""
    key = (ticker, indicator)
    buf = _buffers.get(key)
    return buf is not None and len(buf) >= _MIN_SAMPLES


def get_stats(ticker: str, indicator: str) -> dict | None:
    """Get rolling statistics for a ticker+indicator pair.

    Returns dict with mean, std, count, or None if no data.
    """
    key = (ticker, indicator)
    buf = _buffers.get(key)
    if not buf:
        return None
    arr = np.array(buf, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "count": len(buf),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def clear(ticker: str | None = None, indicator: str | None = None) -> None:
    """Clear buffers.  If ticker given, clear only that ticker's buffers.
    If both ticker and indicator given, clear only that specific pair.
    """
    if ticker and indicator:
        key = (ticker, indicator)
        if key in _buffers:
            del _buffers[key]
    elif ticker:
        keys = [k for k in _buffers if k[0] == ticker]
        for k in keys:
            del _buffers[k]
    else:
        _buffers.clear()
