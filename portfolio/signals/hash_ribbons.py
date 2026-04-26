"""Hash Ribbons BTC miner capitulation signal.

Detects Bitcoin miner capitulation and recovery using hashrate SMA crossover.
When the 30-day SMA of network hashrate drops below the 60-day SMA, miners
are capitulating (unprofitable, shutting down).  When the 30-day crosses back
above the 60-day, capitulation has ended -- historically an 89% win-rate
bottom signal (9 signals since 2011, Capriole Investments).

Sub-indicators:
    1. Hash Ribbon Crossover   (30d vs 60d hashrate SMA)
    2. Price Momentum Filter   (10d vs 20d price SMA confirmation)
    3. Recovery Recency        (days since last recovery crossover)

This is a BUY-only signal (outputs BUY or HOLD, never SELL).
BTC-only -- returns HOLD for non-BTC tickers.

Data: blockchain.info free API (no auth, no rate limit).
Cached for 24h since hashrate is a daily metric.

Source: Charles Edwards, Capriole Investments. Endorsed by VanEck (Dec 2025).
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, safe_float, sma

logger = logging.getLogger("portfolio.signals.hash_ribbons")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_ROWS = 20  # Minimum OHLCV rows for price SMA computation
HASH_FAST = 30  # 30-day SMA period for hashrate
HASH_SLOW = 60  # 60-day SMA period for hashrate
PRICE_FAST = 10  # 10-day SMA for price confirmation
PRICE_SLOW = 20  # 20-day SMA for price confirmation
_CACHE_TTL = 24 * 3600  # 24 hours
_API_TIMEOUT = 20
_HASHRATE_DAYS = 120  # Fetch 120 days to have enough for 60-day SMA

# BTC tickers this signal applies to
_BTC_TICKERS = {"BTC-USD", "BTC/USD", "BTCUSD", "BTC-USDT", "BTCUSDT"}

# Module-level cache
_hash_cache: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Hashrate data fetcher
# ---------------------------------------------------------------------------

def _fetch_hashrate() -> pd.Series | None:
    """Fetch daily BTC network hashrate from blockchain.info.

    Returns a pandas Series indexed by date with hashrate in TH/s.
    Cached for 24 hours.
    """
    now = time.time()
    if _hash_cache.get("data") is not None and now - _hash_cache.get("ts", 0) < _CACHE_TTL:
        return _hash_cache["data"]

    try:
        from portfolio.http_retry import fetch_json
    except ImportError:
        logger.warning("http_retry not available, falling back to requests")
        import requests

        def fetch_json(url, **kw):
            r = requests.get(url, timeout=kw.get("timeout", 20))
            r.raise_for_status()
            return r.json()

    url = (
        f"https://api.blockchain.info/charts/hash-rate"
        f"?timespan={_HASHRATE_DAYS}days&format=json&rollingAverage=1days"
    )
    try:
        data = fetch_json(url, timeout=_API_TIMEOUT, retries=1,
                          label="blockchain_info:hashrate")
    except Exception:
        logger.warning("Failed to fetch hashrate from blockchain.info", exc_info=True)
        return _hash_cache.get("data")

    if not data or not isinstance(data, dict):
        logger.warning("Unexpected hashrate response format")
        return _hash_cache.get("data")

    values = data.get("values", [])
    if not values:
        logger.warning("Empty hashrate values")
        return _hash_cache.get("data")

    try:
        dates = [pd.Timestamp.fromtimestamp(v["x"], tz="UTC").normalize() for v in values]
        rates = [float(v["y"]) for v in values]
        series = pd.Series(rates, index=dates, name="hashrate")
        series = series[~series.index.duplicated(keep="last")].sort_index()
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Failed to parse hashrate data: %s", exc)
        return _hash_cache.get("data")

    _hash_cache["data"] = series
    _hash_cache["ts"] = now
    return series


# ---------------------------------------------------------------------------
# Sub-indicator 1: Hash Ribbon Crossover
# ---------------------------------------------------------------------------

def _hash_ribbon_crossover(hashrate: pd.Series) -> tuple[str, dict]:
    """Detect miner capitulation and recovery via SMA crossover.

    Returns (vote, indicators_dict).
    - Capitulation: 30DMA < 60DMA (miners shutting down)
    - Recovery:     30DMA crosses above 60DMA after capitulation
    """
    if hashrate is None or len(hashrate) < HASH_SLOW + 2:
        return "HOLD", {"hash_sma30": None, "hash_sma60": None, "capitulating": None}

    sma30 = hashrate.rolling(HASH_FAST).mean()
    sma60 = hashrate.rolling(HASH_SLOW).mean()

    curr_fast = sma30.iloc[-1]
    curr_slow = sma60.iloc[-1]
    prev_fast = sma30.iloc[-2]
    prev_slow = sma60.iloc[-2]

    if np.isnan(curr_fast) or np.isnan(curr_slow):
        return "HOLD", {"hash_sma30": None, "hash_sma60": None, "capitulating": None}

    capitulating = curr_fast < curr_slow
    was_capitulating = prev_fast < prev_slow
    recovery_crossover = was_capitulating and not capitulating

    indicators = {
        "hash_sma30": float(curr_fast),
        "hash_sma60": float(curr_slow),
        "capitulating": capitulating,
        "recovery_crossover": recovery_crossover,
        "hash_ratio": float(curr_fast / curr_slow) if curr_slow > 0 else None,
    }

    if recovery_crossover:
        return "BUY", indicators
    return "HOLD", indicators


# ---------------------------------------------------------------------------
# Sub-indicator 2: Price Momentum Filter
# ---------------------------------------------------------------------------

def _price_momentum_filter(close: pd.Series) -> tuple[str, dict]:
    """Confirm upward price momentum via SMA crossover.

    BUY confirmation when SMA(10) > SMA(20).
    """
    if close is None or len(close) < PRICE_SLOW + 1:
        return "HOLD", {"price_sma10": None, "price_sma20": None}

    sma10 = sma(close, PRICE_FAST)
    sma20 = sma(close, PRICE_SLOW)

    val10 = safe_float(sma10.iloc[-1])
    val20 = safe_float(sma20.iloc[-1])

    if val10 is None or val20 is None:
        return "HOLD", {"price_sma10": val10, "price_sma20": val20}

    indicators = {"price_sma10": val10, "price_sma20": val20}

    if val10 > val20:
        return "BUY", indicators
    return "HOLD", indicators


# ---------------------------------------------------------------------------
# Sub-indicator 3: Recovery Recency
# ---------------------------------------------------------------------------

def _recovery_recency(hashrate: pd.Series) -> tuple[str, dict]:
    """How recently did a recovery crossover occur?

    BUY if recovery happened within the last 14 days.
    This catches the signal even if we missed the exact crossover day.
    """
    if hashrate is None or len(hashrate) < HASH_SLOW + 15:
        return "HOLD", {"days_since_recovery": None}

    sma30 = hashrate.rolling(HASH_FAST).mean()
    sma60 = hashrate.rolling(HASH_SLOW).mean()

    # Look back 14 days for any recovery crossover
    window = min(14, len(sma30) - HASH_SLOW - 1)
    days_since = None

    for i in range(1, window + 1):
        idx = -i
        prev_idx = idx - 1
        if abs(prev_idx) >= len(sma30):
            break
        curr_above = sma30.iloc[idx] >= sma60.iloc[idx]
        prev_below = sma30.iloc[prev_idx] < sma60.iloc[prev_idx]
        if curr_above and prev_below:
            days_since = i - 1
            break

    indicators = {"days_since_recovery": days_since}

    if days_since is not None and days_since <= 14:
        return "BUY", indicators
    return "HOLD", indicators


# ---------------------------------------------------------------------------
# Main signal function
# ---------------------------------------------------------------------------

def compute_hash_ribbons_signal(
    df: pd.DataFrame, context: dict | None = None
) -> dict:
    """Compute Hash Ribbons BTC miner capitulation signal.

    Args:
        df: DataFrame with columns: open, high, low, close, volume
        context: Optional dict with keys: ticker, config, asset_class

    Returns:
        dict with keys: action, confidence, sub_signals, indicators
    """
    hold = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {},
        "indicators": {},
    }

    if df is None or len(df) < MIN_ROWS:
        return hold

    # BTC-only signal
    ticker = (context or {}).get("ticker", "")
    if ticker and ticker.upper() not in _BTC_TICKERS:
        return hold

    # Fetch hashrate data
    hashrate = _fetch_hashrate()
    if hashrate is None or len(hashrate) < HASH_SLOW + 2:
        return hold

    close = df["close"].dropna()
    if len(close) < PRICE_SLOW + 1:
        return hold

    # Compute sub-indicators
    ribbon_vote, ribbon_ind = _hash_ribbon_crossover(hashrate)
    price_vote, price_ind = _price_momentum_filter(close)
    recency_vote, recency_ind = _recovery_recency(hashrate)

    sub_signals = {
        "hash_ribbon_crossover": ribbon_vote,
        "price_momentum_filter": price_vote,
        "recovery_recency": recency_vote,
    }

    indicators = {}
    indicators.update(ribbon_ind)
    indicators.update(price_ind)
    indicators.update(recency_ind)

    # Gate: hash ribbon must fire (crossover or recency) for any BUY.
    # Price filter is confirmation only — it cannot trigger BUY by itself.
    hash_fires = ribbon_vote == "BUY" or recency_vote == "BUY"

    if hash_fires and price_vote == "BUY":
        action = "BUY"
        confidence = 0.7  # High conviction when all conditions align
    elif hash_fires:
        # Hash ribbon fires but no price confirmation — still note it
        action = "HOLD"
        confidence = 0.0
        indicators["note"] = "hash_recovery_without_price_confirmation"
    else:
        action = "HOLD"
        confidence = 0.0

    # If we're currently in capitulation (but no recovery yet), note it.
    # Only set if no other note already present (avoid overwriting recovery note).
    if ribbon_ind.get("capitulating") and "note" not in indicators:
        indicators["note"] = "miner_capitulation_active"

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": sub_signals,
        "indicators": indicators,
    }
