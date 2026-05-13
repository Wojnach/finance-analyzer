"""Breakeven Inflation Momentum signal — macro inflation expectations.

Uses FRED 10Y Breakeven Inflation Rate (T10YIE) as inflation expectations
proxy. Rising breakeven = rising inflation expectations = BUY inflation
hedges (gold, silver, BTC). Three sub-signals via majority vote:
    1. BEI Momentum: z-score of 20d change in breakeven rate
    2. BEI Level: absolute breakeven level (high/low thresholds)
    3. BEI Acceleration: 2nd derivative confirms direction

Applies to XAU-USD, XAG-USD (primary) and BTC-USD (inflation hedge).
Data: FRED T10YIE (10Y Breakeven Inflation Rate), cached 4 hours.
Distinct from DFII10 real yield used in metals_cross_asset/gold_real_yield_paradox.

Source: ScienceDirect 2025 gold-economic-indicator framework; gold has
-0.82 correlation with real yields.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote

logger = logging.getLogger(__name__)

MIN_ROWS = 20
_FRED_SERIES = "T10YIE"
_FRED_TIMEOUT = 15
_CACHE_TTL = 4 * 3600
_HISTORY_LIMIT = 400

_APPLICABLE_TICKERS = frozenset({
    "XAU-USD", "XAUUSD", "XAU/USD",
    "XAG-USD", "XAGUSD", "XAG/USD",
    "BTC-USD", "BTCUSD", "BTC/USD",
})

_BEI_CHANGE_LOOKBACK = 20
_BEI_Z_WINDOW = 60
_BEI_Z_BUY = 1.5
_BEI_Z_SELL = -1.5
_BEI_LEVEL_HIGH = 2.5
_BEI_LEVEL_LOW = 1.5
_BEI_ACCEL_LOOKBACK = 10

_bei_cache: dict = {}


def _get_fred_key(context: dict | None) -> str:
    if not context:
        return ""
    cfg = context.get("config")
    if not cfg:
        return ""
    if isinstance(cfg, dict):
        return cfg.get("golddigger", {}).get("fred_api_key", "") or ""
    return getattr(cfg, "fred_api_key", "") or getattr(
        getattr(cfg, "golddigger", None), "fred_api_key", ""
    ) if hasattr(cfg, "fred_api_key") or hasattr(cfg, "golddigger") else ""


def _fetch_bei_values(fred_api_key: str) -> list[float] | None:
    """Fetch 10Y Breakeven Inflation Rate from FRED. Returns list newest-first."""
    now = time.time()
    if (
        _bei_cache.get("key") == fred_api_key
        and _bei_cache.get("data")
        and now - _bei_cache.get("time", 0) < _CACHE_TTL
    ):
        return _bei_cache["data"]

    if not fred_api_key:
        logger.debug("No FRED API key — cannot fetch %s", _FRED_SERIES)
        return _bei_cache.get("data")

    try:
        from portfolio.http_retry import fetch_with_retry
    except ImportError:
        logger.debug("http_retry not available for FRED fetch")
        return _bei_cache.get("data")

    try:
        resp = fetch_with_retry(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": _FRED_SERIES,
                "api_key": fred_api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": _HISTORY_LIMIT,
            },
            timeout=_FRED_TIMEOUT,
        )
        data = resp.json() if hasattr(resp, "json") else json.loads(resp)
        observations = data.get("observations", [])
        values = []
        for obs in observations:
            val = obs.get("value", ".")
            if val != ".":
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    continue

        if values:
            _bei_cache["key"] = fred_api_key
            _bei_cache["data"] = values
            _bei_cache["time"] = now
            logger.debug(
                "FRED %s fetched: %d values, latest=%.3f",
                _FRED_SERIES, len(values), values[0],
            )
            return values
    except Exception:
        logger.warning("FRED %s fetch failed", _FRED_SERIES, exc_info=True)

    return _bei_cache.get("data")


def _compute_change_zscore(
    values: list[float], change_lookback: int, z_window: int,
) -> float:
    """Z-score of the N-day change relative to rolling history of changes."""
    need = change_lookback + z_window + 1
    if len(values) < need:
        return 0.0

    changes = np.array([
        values[i] - values[i + change_lookback]
        for i in range(z_window + 1)
    ])
    current_change = changes[0]
    history = changes[1:]
    mean = float(np.mean(history))
    std = float(np.std(history))
    if std < 1e-10:
        return 0.0
    return (current_change - mean) / std


def _compute_acceleration(values: list[float], lookback: int) -> float:
    """Second derivative proxy: change in rate of change."""
    need = 2 * lookback + 1
    if len(values) < need:
        return 0.0
    recent_change = values[0] - values[lookback]
    prior_change = values[lookback] - values[2 * lookback]
    return recent_change - prior_change


def compute_breakeven_inflation_momentum_signal(
    df: pd.DataFrame, context: dict = None,
) -> dict:
    """Compute breakeven inflation momentum signal."""
    hold = {
        "action": "HOLD", "confidence": 0.0,
        "sub_signals": {}, "indicators": {},
    }

    if df is None or len(df) < MIN_ROWS:
        return hold

    ticker = (context or {}).get("ticker", "")
    ticker_upper = ticker.upper().replace("/", "-")
    if ticker_upper and ticker_upper not in _APPLICABLE_TICKERS:
        return hold

    fred_key = _get_fred_key(context)
    bei_values = _fetch_bei_values(fred_key)
    if not bei_values or len(bei_values) < _BEI_CHANGE_LOOKBACK + _BEI_Z_WINDOW + 1:
        return hold

    current_bei = bei_values[0]
    bei_z = _compute_change_zscore(
        bei_values, _BEI_CHANGE_LOOKBACK, _BEI_Z_WINDOW,
    )
    bei_accel = _compute_acceleration(bei_values, _BEI_ACCEL_LOOKBACK)
    bei_20d_change = bei_values[0] - bei_values[min(_BEI_CHANGE_LOOKBACK, len(bei_values) - 1)]

    sub_signals: dict[str, str] = {}
    votes: list[str] = []

    if bei_z > _BEI_Z_BUY:
        sub_signals["bei_momentum"] = "BUY"
    elif bei_z < _BEI_Z_SELL:
        sub_signals["bei_momentum"] = "SELL"
    else:
        sub_signals["bei_momentum"] = "HOLD"
    votes.append(sub_signals["bei_momentum"])

    if current_bei > _BEI_LEVEL_HIGH:
        sub_signals["bei_level"] = "BUY"
    elif current_bei < _BEI_LEVEL_LOW:
        sub_signals["bei_level"] = "SELL"
    else:
        sub_signals["bei_level"] = "HOLD"
    votes.append(sub_signals["bei_level"])

    if bei_accel > 0.01 and bei_20d_change > 0:
        sub_signals["bei_acceleration"] = "BUY"
    elif bei_accel < -0.01 and bei_20d_change < 0:
        sub_signals["bei_acceleration"] = "SELL"
    else:
        sub_signals["bei_acceleration"] = "HOLD"
    votes.append(sub_signals["bei_acceleration"])

    action, confidence = majority_vote(votes, count_hold=False)

    return {
        "action": action,
        "confidence": min(confidence, 0.7),
        "sub_signals": sub_signals,
        "indicators": {
            "bei_current": round(current_bei, 3),
            "bei_z": round(bei_z, 2),
            "bei_20d_change": round(bei_20d_change, 3),
            "bei_accel": round(bei_accel, 4),
        },
    }
