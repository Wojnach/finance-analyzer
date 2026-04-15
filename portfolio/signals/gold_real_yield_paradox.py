"""Gold-Yield Divergence Index (GYDI) signal — macro regime detector.

Detects when gold rises WITH real yields (historically inverse). Three
sub-indicators via majority vote:
    1. Paradox Spread:  gold 30d return > 0 AND real yield 30d change > 0
    2. Correlation Break: 30d gold-yield correlation deviates from 3yr baseline
    3. Momentum Split:  gold trend vs yield trend divergence/convergence

Applies to XAU-USD (primary) and XAG-USD (secondary, via gold proxy).
Data: FRED DFII10 (10Y TIPS real yield), cached 4 hours.

Source: AHA Signals GYDI tracker; Valadkhani 2024 MSI-VAR.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote

logger = logging.getLogger(__name__)

MIN_ROWS = 50
_FRED_SERIES = "DFII10"
_FRED_TIMEOUT = 15
_CACHE_TTL = 4 * 3600
_HISTORY_LIMIT = 800

_APPLICABLE_TICKERS = frozenset({
    "XAU-USD", "XAUUSD", "XAU/USD",
    "XAG-USD", "XAGUSD", "XAG/USD",
})

_yield_cache: dict = {}
_yield_cache_lock = threading.Lock()


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


def _fetch_real_yield(fred_api_key: str) -> list[float] | None:
    """Fetch 10Y TIPS real yield from FRED. Returns list newest-first."""
    now = time.time()
    with _yield_cache_lock:
        if (
            _yield_cache.get("key") == fred_api_key
            and _yield_cache.get("data")
            and now - _yield_cache.get("time", 0) < _CACHE_TTL
        ):
            return _yield_cache["data"]

    if not fred_api_key:
        logger.debug("No FRED API key for real yield fetch")
        return _yield_cache.get("data")

    try:
        from portfolio.http_retry import fetch_with_retry
    except ImportError:
        logger.warning("http_retry not available")
        return _yield_cache.get("data")

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
            data = json.loads(resp.text if hasattr(resp, "text") else resp)

        observations = data.get("observations", [])
        values = []
        for obs in observations:
            val = obs.get("value", ".")
            if val == ".":
                continue
            try:
                values.append(float(val))
            except (ValueError, TypeError):
                continue

        if values:
            with _yield_cache_lock:
                _yield_cache["key"] = fred_api_key
                _yield_cache["data"] = values
                _yield_cache["time"] = now
            logger.debug("Real yield fetched: %d obs, current=%.3f", len(values), values[0])
            return values

    except Exception:
        logger.warning("FRED real yield fetch failed", exc_info=True)

    return _yield_cache.get("data")


def _paradox_spread(gold_returns_30d: float, yield_change_30d: float) -> tuple[str, dict]:
    """Sub-indicator 1: gold and yield both rising (paradox regime = BUY)."""
    both_positive = gold_returns_30d > 0 and yield_change_30d > 0

    if both_positive:
        magnitude = (
            min(abs(gold_returns_30d) / 0.10, 1.0) * 50
            + min(abs(yield_change_30d) / 0.75, 1.0) * 50
        )
        action = "BUY"
    elif gold_returns_30d < 0 and yield_change_30d < 0:
        magnitude = (
            min(abs(gold_returns_30d) / 0.10, 1.0) * 50
            + min(abs(yield_change_30d) / 0.75, 1.0) * 50
        ) * 0.5
        action = "HOLD"
    else:
        magnitude = 0.0
        action = "HOLD"

    return action, {
        "paradox_score": round(magnitude, 1),
        "gold_30d_ret": round(gold_returns_30d, 4),
        "yield_30d_chg": round(yield_change_30d, 4),
    }


def _correlation_break(
    gold_daily_returns: np.ndarray, yield_daily_changes: np.ndarray,
    baseline_corr: float,
) -> tuple[str, dict]:
    """Sub-indicator 2: 30d correlation deviates from 3yr baseline."""
    if len(gold_daily_returns) < 30 or len(yield_daily_changes) < 30:
        return "HOLD", {"corr_break_score": 0.0}

    recent_gold = gold_daily_returns[-30:]
    recent_yield = yield_daily_changes[-30:]

    if np.std(recent_gold) < 1e-10 or np.std(recent_yield) < 1e-10:
        return "HOLD", {"corr_break_score": 0.0}

    corr_30d = float(np.corrcoef(recent_gold, recent_yield)[0, 1])
    if np.isnan(corr_30d):
        return "HOLD", {"corr_break_score": 0.0}

    deviation = abs(corr_30d - baseline_corr)
    break_score = min(deviation / 0.5 * 100, 100)

    if corr_30d > baseline_corr + 0.3:
        action = "BUY"
    elif corr_30d < baseline_corr - 0.3:
        action = "SELL"
    else:
        action = "HOLD"

    return action, {
        "corr_break_score": round(break_score, 1),
        "corr_30d": round(corr_30d, 3),
        "baseline_corr": round(baseline_corr, 3),
    }


def _momentum_split(
    gold_close: pd.Series, yield_values: list[float],
) -> tuple[str, dict]:
    """Sub-indicator 3: gold trend vs yield trend divergence."""
    if len(gold_close) < 200 or len(yield_values) < 50:
        return "HOLD", {"momentum_split_score": 0.0}

    gold_sma50 = float(gold_close.iloc[-50:].mean())
    gold_sma200 = float(gold_close.iloc[-200:].mean())
    gold_mom = gold_sma50 / gold_sma200 - 1 if gold_sma200 > 0 else 0.0

    yield_current = yield_values[0]
    prior_yields = yield_values[1:51]
    yield_sma50 = sum(prior_yields) / len(prior_yields) if prior_yields else yield_current
    yield_mom = yield_current - yield_sma50

    gold_up = gold_mom > 0.005
    yield_up = yield_mom > 0.05

    if gold_up and yield_up:
        score = 75.0
        action = "BUY"
    elif not gold_up and not yield_up:
        score = 25.0
        action = "SELL"
    else:
        score = 50.0
        action = "HOLD"

    return action, {
        "momentum_split_score": round(score, 1),
        "gold_mom": round(gold_mom, 4),
        "yield_mom": round(yield_mom, 4),
    }


def _compute_baseline_correlation(
    gold_daily_returns: np.ndarray, yield_daily_changes: np.ndarray,
    window: int = 756,
) -> float:
    """3-year rolling baseline of 90d gold-yield correlation."""
    n = min(len(gold_daily_returns), len(yield_daily_changes), window)
    if n < 90:
        return -0.45

    correlations = []
    for start in range(0, n - 90, 30):
        g = gold_daily_returns[start : start + 90]
        y = yield_daily_changes[start : start + 90]
        if np.std(g) < 1e-10 or np.std(y) < 1e-10:
            continue
        c = np.corrcoef(g, y)[0, 1]
        if not np.isnan(c):
            correlations.append(c)

    return float(np.mean(correlations)) if correlations else -0.45


def compute_gold_real_yield_paradox_signal(
    df: pd.DataFrame, context: dict | None = None, **kwargs: Any,
) -> dict:
    """Compute GYDI signal for gold (and silver via gold proxy).

    Args:
        df: OHLCV DataFrame with at least MIN_ROWS rows.
        context: dict with keys {ticker, config, asset_class}.

    Returns:
        dict with keys: action, confidence, sub_signals, indicators.
    """
    empty = {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    if df is None or len(df) < MIN_ROWS:
        return empty

    ticker = (context or {}).get("ticker", "")
    if ticker and ticker not in _APPLICABLE_TICKERS:
        return empty

    fred_key = _get_fred_key(context)
    if not fred_key:
        try:
            from portfolio.file_utils import load_json
            cfg = load_json("config.json")
            if cfg:
                fred_key = cfg.get("golddigger", {}).get("fred_api_key", "") or ""
        except Exception:
            pass

    if not fred_key:
        logger.debug("No FRED API key — cannot compute GYDI")
        return empty

    yield_values = _fetch_real_yield(fred_key)
    if not yield_values or len(yield_values) < 30:
        logger.debug("Insufficient real yield data (%d obs)", len(yield_values) if yield_values else 0)
        return empty

    close = df["close"].astype(float)

    gold_30d_return = float(close.iloc[-1] / close.iloc[-min(30, len(close))] - 1)
    yield_30d_change = yield_values[0] - yield_values[min(29, len(yield_values) - 1)]

    gold_daily_returns = close.pct_change(fill_method=None).dropna().values
    min_len = min(len(gold_daily_returns), len(yield_values) - 1)
    yield_daily_changes = np.array([
        yield_values[i] - yield_values[i + 1]
        for i in range(min_len)
    ])[::-1]  # reverse: FRED is newest-first, gold is oldest-first
    gold_daily_returns = gold_daily_returns[-min_len:]

    baseline_corr = _compute_baseline_correlation(gold_daily_returns, yield_daily_changes)

    act1, ind1 = _paradox_spread(gold_30d_return, yield_30d_change)
    act2, ind2 = _correlation_break(gold_daily_returns, yield_daily_changes, baseline_corr)
    act3, ind3 = _momentum_split(close, yield_values)

    votes = [act1, act2, act3]
    action, confidence = majority_vote(votes, count_hold=False)

    gydi = (
        ind1.get("paradox_score", 0) * 0.40
        + ind2.get("corr_break_score", 0) * 0.35
        + ind3.get("momentum_split_score", 0) * 0.25
    )

    regime = "LOW"
    if gydi >= 75:
        regime = "CRITICAL"
    elif gydi >= 50:
        regime = "HIGH"
    elif gydi >= 30:
        regime = "ELEVATED"

    indicators = {**ind1, **ind2, **ind3, "gydi": round(gydi, 1), "gydi_regime": regime}

    return {
        "action": action,
        "confidence": min(confidence, 0.7),
        "sub_signals": {"paradox_spread": act1, "correlation_break": act2, "momentum_split": act3},
        "indicators": indicators,
    }
