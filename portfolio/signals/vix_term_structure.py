"""VIX term structure signal — contango/backwardation regime detection.

Uses VIX/VIX3M ratio to detect risk regime shifts. Backwardation
(VIX > VIX3M) occurs <20% of the time and signals market stress.
Deep contango (ratio < 0.85) signals complacency — contrarian BUY
on recovery.

Backtested strongest on metals (SLV 55%/64%/73% at 1d/3d/5d,
GLD 55%/61% at 1d/3d). Weaker on BTC/equities (~40-50%). Applied
to all assets; per-ticker accuracy gating auto-disables for assets
where it underperforms.

Sub-indicators:
    1. Backwardation flag     — ratio >= 1.0 = stress
    2. Contango depth         — how far below 1.0 (deep = complacent)
    3. VIX ratio z-score      — any deviation from 20d mean
    4. Ratio slope (5d)       — rate of change signals transitions

Data: yfinance ^VIX and ^VIX3M (free, no API key).
"""
from __future__ import annotations

import logging

import numpy as np

from portfolio.shared_state import _cached
from portfolio.signal_utils import majority_vote, safe_float

logger = logging.getLogger("portfolio.signals.vix_term_structure")

_MAX_CONFIDENCE = 0.7

_BACKWARDATION_THRESHOLD = 1.0
_STRONG_BACKWARDATION = 1.05
_DEEP_CONTANGO = 0.85
_Z_THRESHOLD = 0.0  # any deviation votes; backtested: z=0.0 >> z=1.0
_SLOPE_THRESHOLD_PCT = 2.0
_LOOKBACK = 20
_CACHE_TTL = 900


def _fetch_vix_data() -> dict | None:
    """Fetch VIX and VIX3M closing prices via yfinance."""
    try:
        import yfinance as yf

        vix = yf.download("^VIX", period="2mo", progress=False)
        vix3m = yf.download("^VIX3M", period="2mo", progress=False)

        if vix is None or vix3m is None or len(vix) < _LOOKBACK or len(vix3m) < _LOOKBACK:
            return None

        close_col = "Close"
        if hasattr(vix.columns, "levels") and len(vix.columns.levels) > 1:
            vix = vix.droplevel(level=1, axis=1)
        if hasattr(vix3m.columns, "levels") and len(vix3m.columns.levels) > 1:
            vix3m = vix3m.droplevel(level=1, axis=1)

        vix_close = vix[close_col].dropna()
        vix3m_close = vix3m[close_col].dropna()

        if len(vix_close) < _LOOKBACK or len(vix3m_close) < _LOOKBACK:
            return None

        idx = vix_close.index.intersection(vix3m_close.index)
        if len(idx) < _LOOKBACK:
            return None

        vix_vals = vix_close.loc[idx].values.astype(float)
        vix3m_vals = vix3m_close.loc[idx].values.astype(float)

        mask = (vix3m_vals > 0) & np.isfinite(vix_vals) & np.isfinite(vix3m_vals)
        vix_vals = vix_vals[mask]
        vix3m_vals = vix3m_vals[mask]

        if len(vix_vals) < _LOOKBACK:
            return None

        ratio = vix_vals / vix3m_vals

        return {
            "vix_current": float(vix_vals[-1]),
            "vix3m_current": float(vix3m_vals[-1]),
            "ratio_current": float(ratio[-1]),
            "ratio_series": ratio.tolist(),
        }
    except Exception as e:
        logger.debug("VIX term structure fetch failed: %s", e)
        return None


def _backwardation_flag(ratio: float) -> str:
    if ratio >= _STRONG_BACKWARDATION:
        return "SELL"
    if ratio >= _BACKWARDATION_THRESHOLD:
        return "SELL"
    if ratio < _DEEP_CONTANGO:
        return "BUY"
    return "HOLD"


def _contango_depth(ratio: float) -> str:
    depth = 1.0 - ratio
    if depth > 0.15:
        return "BUY"
    if depth > 0.10:
        return "BUY"
    if depth < -0.05:
        return "SELL"
    if depth < 0.0:
        return "SELL"
    return "HOLD"


def _ratio_zscore(ratio_series: list[float]) -> tuple[float, str]:
    arr = np.array(ratio_series[-_LOOKBACK:])
    if len(arr) < _LOOKBACK:
        return 0.0, "HOLD"
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if std < 1e-8:
        return 0.0, "HOLD"
    z = (arr[-1] - mean) / std
    if z > _Z_THRESHOLD:
        return float(z), "SELL"
    if z < -_Z_THRESHOLD:
        return float(z), "BUY"
    return float(z), "HOLD"


def _ratio_slope_5d(ratio_series: list[float]) -> tuple[float, str]:
    if len(ratio_series) < 6:
        return 0.0, "HOLD"
    current = ratio_series[-1]
    prev = ratio_series[-6]
    if prev == 0:
        return 0.0, "HOLD"
    slope_pct = (current - prev) / prev * 100
    if slope_pct > _SLOPE_THRESHOLD_PCT:
        return float(slope_pct), "SELL"
    if slope_pct < -_SLOPE_THRESHOLD_PCT:
        return float(slope_pct), "BUY"
    return float(slope_pct), "HOLD"


def compute_vix_term_structure_signal(df=None, context=None, **kwargs) -> dict:
    """Compute VIX term structure signal for any Tier-1 instrument.

    Args:
        df: OHLCV DataFrame (unused — VIX data fetched separately).
        context: Optional dict with ``ticker`` key.

    Returns:
        dict with action, confidence, sub_signals, indicators.
    """
    empty = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {},
        "indicators": {},
    }

    data = _cached("vix_term_structure", _CACHE_TTL, _fetch_vix_data)
    if data is None:
        return empty

    ratio = data["ratio_current"]
    ratio_series = data["ratio_series"]

    sub1 = _backwardation_flag(ratio)
    sub2 = _contango_depth(ratio)
    z_val, sub3 = _ratio_zscore(ratio_series)
    slope_val, sub4 = _ratio_slope_5d(ratio_series)

    votes = [sub1, sub2, sub3, sub4]
    action, confidence = majority_vote(votes, count_hold=False)

    confidence = min(confidence, _MAX_CONFIDENCE)

    if ratio >= _STRONG_BACKWARDATION:
        confidence = min(max(confidence, 0.6), _MAX_CONFIDENCE)

    indicators = {
        "vix": safe_float(data["vix_current"]),
        "vix3m": safe_float(data["vix3m_current"]),
        "ratio": safe_float(ratio),
        "z_score": safe_float(z_val),
        "slope_5d_pct": safe_float(slope_val),
        "in_backwardation": ratio >= _BACKWARDATION_THRESHOLD,
    }

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": {
            "backwardation_flag": sub1,
            "contango_depth": sub2,
            "ratio_zscore": sub3,
            "ratio_slope_5d": sub4,
        },
        "indicators": indicators,
    }
