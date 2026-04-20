"""OVX Metals Spillover signal — oil implied volatility as metals predictor.

Crude Oil Volatility Index (OVX) at extreme quantiles predicts precious
metals returns through three channels:
    1. Inflation expectations (high oil vol -> uncertain inflation -> gold bid)
    2. Risk-off contagion (energy stress -> broad risk aversion -> metals sell)
    3. Dollar channel (oil disruption -> dollar weakness -> gold strength)

At 1d-3d horizons, the contagion/liquidity effect dominates: high OVX
predicts NEGATIVE metals returns (stress sells everything including gold).
Mean-reversion from extremes creates recovery opportunities.

Distinct from metals_cross_asset which uses oil PRICE direction.
This signal uses oil IMPLIED VOLATILITY (expectation of future moves).

Metals-only (XAU-USD, XAG-USD). Returns HOLD for other tickers.

Sub-signals:
    1. ovx_level     - percentile rank over 252d (>80th=SELL, <20th=BUY)
    2. ovx_momentum  - 5d rate of change (rising=SELL, falling=BUY)
    3. ovx_zscore    - z-score over 20d mean (>1.5=SELL, <-1.5=BUY)
    4. ovx_reversion - contrarian when extreme + reversing direction

Data: yfinance ^OVX (CBOE Oil Volatility Index), free, daily.
Cached 4h via shared_state._cached.
"""
from __future__ import annotations

import logging

import numpy as np

from portfolio.shared_state import _cached
from portfolio.signal_utils import majority_vote, safe_float

logger = logging.getLogger("portfolio.signals.ovx_metals_spillover")

_METALS_TICKERS = {"XAU-USD", "XAG-USD"}
_MAX_CONFIDENCE = 0.7
_CACHE_TTL = 4 * 3600  # 4 hours
_MIN_HISTORY = 60  # Minimum trading days needed
_LOOKBACK_PCTILE = 252  # 1 year for percentile rank
_LOOKBACK_ZSCORE = 20  # 20 days for z-score
_MOMENTUM_WINDOW = 5  # 5-day momentum

# Thresholds
_HIGH_PCTILE = 80
_LOW_PCTILE = 20
_MOMENTUM_UP_PCT = 10.0  # 10% 5d rise = stress building
_MOMENTUM_DOWN_PCT = -10.0  # 10% 5d fall = stress easing
_ZSCORE_HIGH = 1.5
_ZSCORE_LOW = -1.5
_REVERSION_HIGH_PCTILE = 75
_REVERSION_LOW_PCTILE = 25
_REVERSION_MOMENTUM_PCT = 5.0


def _fetch_ovx_data() -> dict | None:
    """Fetch OVX historical data via yfinance."""
    try:
        import yfinance as yf

        ovx = yf.download("^OVX", period="15mo", progress=False)
        if ovx is None or len(ovx) < _MIN_HISTORY:
            return None

        if hasattr(ovx.columns, "levels") and len(ovx.columns.levels) > 1:
            ovx = ovx.droplevel(level=1, axis=1)

        close = ovx["Close"].dropna()
        if len(close) < _MIN_HISTORY:
            return None

        values = close.values.astype(float)
        mask = np.isfinite(values) & (values > 0)
        values = values[mask]

        if len(values) < _MIN_HISTORY:
            return None

        return {
            "current": float(values[-1]),
            "series": values.tolist(),
        }
    except Exception as e:
        logger.debug("OVX fetch failed: %s", e)
        return None


def _percentile_rank(series: list[float], lookback: int) -> float:
    """Compute percentile rank of latest value over lookback window."""
    if len(series) < lookback:
        window = series
    else:
        window = series[-lookback:]
    current = window[-1]
    count_below = sum(1 for v in window[:-1] if v < current)
    total = len(window) - 1
    if total <= 0:
        return 50.0
    return (count_below / total) * 100


def _ovx_level_signal(pctile: float) -> str:
    """High OVX = stress contagion = SELL metals; low = calm = BUY."""
    if pctile > _HIGH_PCTILE:
        return "SELL"
    if pctile < _LOW_PCTILE:
        return "BUY"
    return "HOLD"


def _ovx_momentum_signal(series: list[float]) -> tuple[float, str]:
    """5-day rate of change. Rising = building stress, falling = easing."""
    if len(series) < _MOMENTUM_WINDOW + 1:
        return 0.0, "HOLD"
    current = series[-1]
    prev = series[-(_MOMENTUM_WINDOW + 1)]
    if prev <= 0:
        return 0.0, "HOLD"
    change_pct = (current - prev) / prev * 100
    if change_pct > _MOMENTUM_UP_PCT:
        return change_pct, "SELL"
    if change_pct < _MOMENTUM_DOWN_PCT:
        return change_pct, "BUY"
    return change_pct, "HOLD"


def _ovx_zscore_signal(series: list[float]) -> tuple[float, str]:
    """Z-score of current OVX vs recent mean."""
    lookback = min(len(series), _LOOKBACK_ZSCORE)
    if lookback < 10:
        return 0.0, "HOLD"
    window = np.array(series[-lookback:])
    mean = np.mean(window)
    std = np.std(window, ddof=1)
    if std < 1e-8:
        return 0.0, "HOLD"
    z = (window[-1] - mean) / std
    if z > _ZSCORE_HIGH:
        return float(z), "SELL"
    if z < _ZSCORE_LOW:
        return float(z), "BUY"
    return float(z), "HOLD"


def _ovx_reversion_signal(pctile: float, series: list[float]) -> str:
    """Mean reversion: extreme + reversing = contrarian opportunity."""
    if len(series) < _MOMENTUM_WINDOW + 1:
        return "HOLD"
    current = series[-1]
    prev = series[-(_MOMENTUM_WINDOW + 1)]
    if prev <= 0:
        return "HOLD"
    change_pct = (current - prev) / prev * 100

    # Falling from high = recovery → BUY metals
    if pctile > _REVERSION_HIGH_PCTILE and change_pct < -_REVERSION_MOMENTUM_PCT:
        return "BUY"
    # Rising from low = vol expanding from calm → SELL metals
    if pctile < _REVERSION_LOW_PCTILE and change_pct > _REVERSION_MOMENTUM_PCT:
        return "SELL"
    return "HOLD"


def compute_ovx_metals_spillover_signal(
    df=None, context: dict | None = None, **kwargs,
) -> dict:
    """Compute OVX metals spillover signal.

    Args:
        df: OHLCV DataFrame (unused — OVX data fetched separately).
        context: Optional dict with ``ticker`` key.

    Returns:
        dict with action, confidence, sub_signals, indicators.
        HOLD for non-metals tickers or when OVX data unavailable.
    """
    empty = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {},
        "indicators": {},
    }

    context = context or {}
    ticker = context.get("ticker", kwargs.get("ticker", ""))
    if ticker not in _METALS_TICKERS:
        return empty

    data = _cached("ovx_metals_spillover", _CACHE_TTL, _fetch_ovx_data)
    if data is None:
        return empty

    series = data["series"]
    current = data["current"]

    pctile = _percentile_rank(series, _LOOKBACK_PCTILE)
    sub1 = _ovx_level_signal(pctile)
    mom_val, sub2 = _ovx_momentum_signal(series)
    z_val, sub3 = _ovx_zscore_signal(series)
    sub4 = _ovx_reversion_signal(pctile, series)

    votes = [sub1, sub2, sub3, sub4]
    action, confidence = majority_vote(votes, count_hold=False)

    confidence = min(confidence, _MAX_CONFIDENCE)

    sub_signals = {
        "ovx_level": sub1,
        "ovx_momentum": sub2,
        "ovx_zscore": sub3,
        "ovx_reversion": sub4,
    }

    indicators = {
        "ovx_current": safe_float(current),
        "ovx_pctile": safe_float(pctile),
        "ovx_momentum_5d_pct": safe_float(mom_val),
        "ovx_zscore": safe_float(z_val),
    }

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": sub_signals,
        "indicators": indicators,
    }
