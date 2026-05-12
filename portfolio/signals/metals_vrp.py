"""Metals Volatility Risk Premium signal — options-derived contrarian indicator.

Uses GVZ (CBOE Gold Volatility Index) from FRED minus realized vol.
High VRP = fear overpriced = contrarian BUY. Low VRP = complacency = SELL.
Metals-only (XAU-USD, XAG-USD). Requires context dict with config (FRED key).
"""
from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, safe_float, sma

logger = logging.getLogger("portfolio.signals.metals_vrp")

MIN_ROWS = 30
_METALS_TICKERS = {"XAU-USD", "XAG-USD"}

_FRED_SERIES = "GVZCLS"
_FRED_TIMEOUT = 15
_CACHE_TTL = 4 * 3600
_HISTORY_LIMIT = 300

_RV_WINDOW = 20
_Z_LOOKBACK = 60
_Z_BUY = 1.5
_Z_SELL = -1.5
_GVZ_PCT_LOOKBACK = 252

_gvz_cache: dict = {}
_gvz_lock = threading.Lock()


def _fetch_gvz(fred_api_key: str) -> list[float] | None:
    """Fetch GVZ history from FRED. Returns list of floats (newest first)."""
    now = time.time()
    with _gvz_lock:
        if (
            _gvz_cache.get("key") == fred_api_key
            and _gvz_cache.get("data")
            and now - _gvz_cache.get("time", 0) < _CACHE_TTL
        ):
            return _gvz_cache["data"]

    if not fred_api_key:
        logger.debug("No FRED API key — cannot fetch GVZ")
        with _gvz_lock:
            return _gvz_cache.get("data")

    try:
        from portfolio.http_retry import fetch_with_retry
    except ImportError:
        import requests

        class _Shim:
            @staticmethod
            def __call__(url, **kwargs):
                return requests.get(url, **kwargs)

        fetch_with_retry = _Shim()

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
        if hasattr(resp, "json"):
            data = resp.json()
        else:
            import json

            data = json.loads(resp)

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
            with _gvz_lock:
                _gvz_cache["key"] = fred_api_key
                _gvz_cache["data"] = values
                _gvz_cache["time"] = now
            logger.debug("GVZ fetched: %d obs, current=%.2f", len(values), values[0])
            return values

    except Exception:
        logger.warning("FRED GVZ fetch failed", exc_info=True)

    with _gvz_lock:
        return _gvz_cache.get("data")


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
    ) or ""


def _compute_realized_vol(close: pd.Series, window: int = _RV_WINDOW) -> pd.Series:
    """Annualized realized volatility from log returns."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window=window, min_periods=window).std() * math.sqrt(252) * 100


def compute_metals_vrp_signal(df: pd.DataFrame, context: dict | None = None) -> dict:
    """Compute metals volatility risk premium signal."""
    hold = {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    if df is None or len(df) < MIN_ROWS:
        return hold

    ticker = (context or {}).get("ticker", "")
    if ticker and ticker not in _METALS_TICKERS:
        return hold

    fred_key = _get_fred_key(context)
    gvz_data = _fetch_gvz(fred_key)

    if not gvz_data or len(gvz_data) < _Z_LOOKBACK:
        return hold

    close = df["close"].dropna()
    if len(close) < MIN_ROWS:
        return hold

    rv_series = _compute_realized_vol(close, _RV_WINDOW)
    current_rv = safe_float(rv_series.iloc[-1])
    if np.isnan(current_rv) or current_rv <= 0:
        return hold

    current_gvz = gvz_data[0]
    gvz_history = np.array(gvz_data[: _HISTORY_LIMIT], dtype=float)

    vrp = current_gvz - current_rv

    gvz_hist_for_z = gvz_history[: _Z_LOOKBACK]
    if len(gvz_hist_for_z) < 20:
        return hold

    rv_values = rv_series.dropna().values
    if len(rv_values) < 20:
        return hold
    rv_mean = float(np.mean(rv_values[-_Z_LOOKBACK:]))
    rv_for_vrp = gvz_hist_for_z - rv_mean
    vrp_mean = float(np.mean(rv_for_vrp))
    vrp_std = float(np.std(rv_for_vrp))
    if vrp_std < 0.01:
        return hold

    vrp_z = (vrp - vrp_mean) / vrp_std

    votes = []
    sub_signals = {}

    # Sub-signal 1: VRP Z-Score
    if vrp_z > _Z_BUY:
        sub_signals["vrp_z"] = "BUY"
        votes.append("BUY")
    elif vrp_z < _Z_SELL:
        sub_signals["vrp_z"] = "SELL"
        votes.append("SELL")
    else:
        sub_signals["vrp_z"] = "HOLD"
        votes.append("HOLD")

    # Sub-signal 2: VRP level vs median
    vrp_median = float(np.median(rv_for_vrp))
    vrp_level_std = vrp_std
    if vrp > vrp_median + vrp_level_std:
        sub_signals["vrp_level"] = "BUY"
        votes.append("BUY")
    elif vrp < vrp_median - vrp_level_std:
        sub_signals["vrp_level"] = "SELL"
        votes.append("SELL")
    else:
        sub_signals["vrp_level"] = "HOLD"
        votes.append("HOLD")

    # Sub-signal 3: VRP momentum (5-bar GVZ change direction)
    if len(gvz_history) >= 6:
        gvz_5d_ago = gvz_history[5]
        gvz_change = current_gvz - gvz_5d_ago
        if gvz_change > 2.0 and vrp_z > 0:
            sub_signals["vrp_momentum"] = "BUY"
            votes.append("BUY")
        elif gvz_change < -2.0 and vrp_z < 0:
            sub_signals["vrp_momentum"] = "SELL"
            votes.append("SELL")
        else:
            sub_signals["vrp_momentum"] = "HOLD"
            votes.append("HOLD")
    else:
        sub_signals["vrp_momentum"] = "HOLD"
        votes.append("HOLD")

    # Sub-signal 4: GVZ percentile rank (contrarian)
    gvz_pct_data = gvz_history[: _GVZ_PCT_LOOKBACK]
    pct_rank = None
    if len(gvz_pct_data) >= 60:
        pct_rank = float(np.sum(gvz_pct_data <= current_gvz)) / len(gvz_pct_data)
        if pct_rank > 0.80:
            sub_signals["gvz_percentile"] = "BUY"
            votes.append("BUY")
        elif pct_rank < 0.20:
            sub_signals["gvz_percentile"] = "SELL"
            votes.append("SELL")
        else:
            sub_signals["gvz_percentile"] = "HOLD"
            votes.append("HOLD")
    else:
        sub_signals["gvz_percentile"] = "HOLD"
        votes.append("HOLD")

    action, confidence = majority_vote(votes, count_hold=False)

    indicators = {
        "gvz_current": safe_float(current_gvz),
        "realized_vol": safe_float(current_rv),
        "vrp": safe_float(vrp),
        "vrp_z": safe_float(vrp_z),
        "vrp_mean": safe_float(vrp_mean),
        "vrp_std": safe_float(vrp_std),
    }
    if pct_rank is not None:
        indicators["gvz_percentile_rank"] = safe_float(pct_rank)

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": sub_signals,
        "indicators": indicators,
    }
