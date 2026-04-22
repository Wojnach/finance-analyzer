"""Credit spread risk appetite signal — cross-asset macro indicator.

Signal #35.  Uses ICE BofA High Yield Option-Adjusted Spread (HY OAS)
from FRED API as a measure of institutional credit risk appetite.
Combines 4 sub-indicators via majority vote:
    1. OAS Level Z-Score:    where current spread sits vs 252d history
    2. OAS 5d Momentum:      rate of spread widening/tightening
    3. OAS Momentum Accel:   acceleration of momentum (early warning)
    4. Crisis Level Gate:    hard threshold at 500bp (every bear market)

Direction depends on asset class:
- Gold/Silver (safe haven):  risk-off → BUY, risk-on → SELL
- Crypto/Stocks (risk-on):   risk-off → SELL, risk-on → BUY

Data: FRED API series BAMLH0A0HYM2 (daily, free with API key).
Cached for 4 hours since data updates daily.

Requires context dict with keys: ticker, config (for FRED API key).
"""
from __future__ import annotations

import logging
import time
from typing import Any

from portfolio.file_utils import load_json
from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.credit_spread")

# ---------------------------------------------------------------------------
# Asset-class classification for directional interpretation
# ---------------------------------------------------------------------------
_SAFE_HAVEN = {"XAU-USD", "XAG-USD"}
_RISK_ASSETS = {"BTC-USD", "ETH-USD", "MSTR"}

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
_ZSCORE_RISK_OFF = 1.5      # OAS z-score above this = risk-off
_ZSCORE_RISK_ON = -1.0      # OAS z-score below this = risk-on (complacent)
_MOM_5D_THRESHOLD = 0.20    # 20bp 5d change = meaningful momentum
_ACCEL_THRESHOLD = 0.15     # 15bp acceleration = early warning
_CRISIS_LEVEL = 5.0         # 500bp = crisis threshold
_HISTORY_LIMIT = 300        # FRED observations to fetch (covers ~252 trading days)
_FRED_SERIES = "BAMLH0A0HYM2"
_FRED_TIMEOUT = 15
_CACHE_TTL = 4 * 3600       # 4 hours

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_oas_cache: dict = {}


def _fetch_hy_oas(fred_api_key: str) -> list[float] | None:
    """Fetch HY OAS history from FRED.  Returns list of floats (newest first).

    Cached for 4 hours since data is daily.
    """
    now = time.time()
    if (
        _oas_cache.get("key") == fred_api_key
        and _oas_cache.get("data")
        and now - _oas_cache.get("time", 0) < _CACHE_TTL
    ):
        return _oas_cache["data"]

    if not fred_api_key:
        logger.debug("No FRED API key — cannot fetch HY OAS")
        return _oas_cache.get("data")

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
            _oas_cache["key"] = fred_api_key
            _oas_cache["data"] = values
            _oas_cache["time"] = now
            logger.debug("HY OAS fetched: %d observations, current=%.2f", len(values), values[0])
            return values

    except Exception:
        logger.warning("FRED HY OAS fetch failed", exc_info=True)

    return _oas_cache.get("data")


def _get_fred_key(context: dict | None) -> str:
    """Extract FRED API key from context -> config."""
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


def _is_safe_haven(ticker: str) -> bool:
    """Return True if ticker is a safe-haven asset (gold/silver)."""
    return ticker in _SAFE_HAVEN


# ---------------------------------------------------------------------------
# Sub-indicator 1: OAS Level Z-Score
# ---------------------------------------------------------------------------

def _oas_zscore_signal(values: list[float], safe_haven: bool) -> tuple[str, dict]:
    """Z-score of current OAS vs 252d history."""
    if len(values) < 50:
        return "HOLD", {"oas_zscore": 0.0}

    lookback = min(252, len(values))
    history = values[:lookback]
    current = values[0]
    mean = sum(history) / len(history)
    variance = sum((v - mean) ** 2 for v in history) / len(history)
    std = variance ** 0.5

    if std < 0.01:
        return "HOLD", {"oas_zscore": 0.0}

    zscore = (current - mean) / std

    action = "HOLD"
    if zscore > _ZSCORE_RISK_OFF:
        # Risk-off: spreads unusually wide
        action = "BUY" if safe_haven else "SELL"
    elif zscore < _ZSCORE_RISK_ON:
        # Risk-on: spreads unusually tight (complacent)
        action = "SELL" if safe_haven else "BUY"

    return action, {"oas_zscore": round(zscore, 3), "oas_current": round(current, 2),
                     "oas_mean_252d": round(mean, 2)}


# ---------------------------------------------------------------------------
# Sub-indicator 2: OAS 5d Momentum
# ---------------------------------------------------------------------------

def _oas_momentum_signal(values: list[float], safe_haven: bool) -> tuple[str, dict]:
    """5-day change in OAS (widening vs tightening)."""
    if len(values) < 6:
        return "HOLD", {"oas_mom_5d": 0.0}

    mom = values[0] - values[5]

    action = "HOLD"
    if mom > _MOM_5D_THRESHOLD:
        # Spreads widening rapidly → risk-off
        action = "BUY" if safe_haven else "SELL"
    elif mom < -_MOM_5D_THRESHOLD:
        # Spreads tightening → risk-on
        action = "SELL" if safe_haven else "BUY"

    return action, {"oas_mom_5d": round(mom, 3)}


# ---------------------------------------------------------------------------
# Sub-indicator 3: Momentum Acceleration
# ---------------------------------------------------------------------------

def _oas_acceleration_signal(values: list[float], safe_haven: bool) -> tuple[str, dict]:
    """Acceleration of OAS momentum (early warning of regime shift)."""
    if len(values) < 11:
        return "HOLD", {"oas_accel": 0.0}

    mom_current = values[0] - values[5]
    mom_prev = values[5] - values[10]
    accel = mom_current - mom_prev

    action = "HOLD"
    if accel > _ACCEL_THRESHOLD:
        # Accelerating widening → risk-off intensifying
        action = "BUY" if safe_haven else "SELL"
    elif accel < -_ACCEL_THRESHOLD:
        # Accelerating tightening → risk-on intensifying
        action = "SELL" if safe_haven else "BUY"

    return action, {"oas_accel": round(accel, 3)}


# ---------------------------------------------------------------------------
# Sub-indicator 4: Crisis Level Gate
# ---------------------------------------------------------------------------

def _crisis_level_signal(values: list[float], safe_haven: bool) -> tuple[str, dict]:
    """Hard threshold: OAS above 500bp = crisis mode."""
    if not values:
        return "HOLD", {"oas_crisis": False}

    current = values[0]
    crisis = current >= _CRISIS_LEVEL

    action = "HOLD"
    if crisis:
        # Full crisis → strong safe-haven bid
        action = "BUY" if safe_haven else "SELL"
    elif current <= 2.5:
        # Extreme complacency → contrarian warning
        action = "SELL" if safe_haven else "BUY"

    return action, {"oas_crisis": crisis, "oas_level": round(current, 2)}


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute_credit_spread_signal(
    df: Any, context: dict | None = None, **kwargs,
) -> dict:
    """Compute credit spread risk appetite signal.

    Args:
        df: OHLCV DataFrame (unused — credit spread data fetched from FRED).
        context: dict with keys {ticker, config, asset_class, regime}.

    Returns:
        dict with keys: action, confidence, sub_signals, indicators.
    """
    empty = {
        "action": "HOLD", "confidence": 0.0,
        "sub_signals": {}, "indicators": {},
    }

    context = context or {}
    ticker = context.get("ticker", kwargs.get("ticker", ""))

    if not ticker:
        return empty

    # All tickers can use this signal
    if ticker not in _SAFE_HAVEN and ticker not in _RISK_ASSETS:
        return empty

    safe_haven = _is_safe_haven(ticker)

    # Get FRED API key from config
    fred_key = _get_fred_key(context)

    # Also try loading from config.json directly as fallback
    if not fred_key:
        try:
            cfg = load_json("config.json", default={}) or {}
            fred_key = cfg.get("golddigger", {}).get("fred_api_key", "") or ""
        except Exception:
            pass

    if not fred_key:
        logger.debug("No FRED API key available for credit spread signal")
        return empty

    # Fetch HY OAS data
    values = _fetch_hy_oas(fred_key)
    if not values or len(values) < 20:
        return empty

    # Compute sub-indicators
    zscore_action, zscore_ind = _oas_zscore_signal(values, safe_haven)
    mom_action, mom_ind = _oas_momentum_signal(values, safe_haven)
    accel_action, accel_ind = _oas_acceleration_signal(values, safe_haven)
    crisis_action, crisis_ind = _crisis_level_signal(values, safe_haven)

    votes = [zscore_action, mom_action, accel_action, crisis_action]
    action, confidence = majority_vote(votes, count_hold=False)

    # Merge all indicators
    indicators = {}
    indicators.update(zscore_ind)
    indicators.update(mom_ind)
    indicators.update(accel_ind)
    indicators.update(crisis_ind)
    indicators["safe_haven_mode"] = safe_haven

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": {
            "oas_zscore": zscore_action,
            "oas_momentum": mom_action,
            "oas_acceleration": accel_action,
            "crisis_level": crisis_action,
        },
        "indicators": indicators,
    }
