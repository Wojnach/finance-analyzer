"""Shared mutable state for the portfolio system.

All modules that need caching or cross-cycle state import from here.
This avoids circular imports and provides a single source of truth.
"""

import logging
import time
import threading

logger = logging.getLogger("portfolio.shared_state")

# Tool cache — avoid re-running expensive tools every cycle
_tool_cache = {}

# Retry cooldown for _cached() errors
_RETRY_COOLDOWN = 60


_CACHE_MAX_SIZE = 256  # evict expired entries when cache exceeds this size
_cache_lock = threading.Lock()


_MAX_STALE_FACTOR = 3  # return None if cached data is older than TTL * this factor


def _cached(key, ttl, func, *args):
    """Cache-through helper: returns cached data if fresh, else calls func.

    On error, returns stale data if it's less than TTL * _MAX_STALE_FACTOR old.
    Beyond that, returns None to prevent trading on dangerously old data.
    """
    now = time.time()
    with _cache_lock:
        if key in _tool_cache and now - _tool_cache[key]["time"] < ttl:
            return _tool_cache[key]["data"]
        # Evict expired entries when cache grows too large
        # Use TTL-aware eviction: entries expire after ttl * _MAX_STALE_FACTOR
        if len(_tool_cache) > _CACHE_MAX_SIZE:
            expired = [k for k, v in _tool_cache.items()
                       if now - v["time"] > v.get("ttl", 3600) * _MAX_STALE_FACTOR]
            for k in expired:
                del _tool_cache[k]
            # LRU fallback: if still over limit (all entries fresh), evict oldest 25%
            if len(_tool_cache) > _CACHE_MAX_SIZE:
                sorted_keys = sorted(
                    _tool_cache, key=lambda k: _tool_cache[k]["time"]
                )
                evict_count = len(sorted_keys) // 4 or 1
                for k in sorted_keys[:evict_count]:
                    del _tool_cache[k]
    try:
        data = func(*args)
        with _cache_lock:
            _tool_cache[key] = {"data": data, "time": now, "ttl": ttl}
        return data
    except Exception as e:
        logger.warning("[%s] error: %s", key, e)
        with _cache_lock:
            if key in _tool_cache:
                age = now - _tool_cache[key]["time"]
                max_stale = ttl * _MAX_STALE_FACTOR
                if age > max_stale:
                    logger.warning(
                        "[%s] stale data too old (%.0fs > %.0fs max), returning None",
                        key, age, max_stale,
                    )
                    return None
                _tool_cache[key]["time"] = now - ttl + _RETRY_COOLDOWN
                return _tool_cache[key]["data"]
        return None


# Cycle counter — incremented at the start of each run() to invalidate per-cycle caches
_run_cycle_id = 0

# Current market state — updated each run() cycle, used by data_collector for yfinance fallback
_current_market_state = "open"

# Regime detection cache (invalidated each cycle)
_regime_cache = {}
_regime_cache_cycle = 0


# --- Rate limiters ---

class _RateLimiter:
    """Token-bucket rate limiter. Sleeps when calls exceed rate."""
    def __init__(self, max_per_minute, name=""):
        self.interval = 60.0 / max_per_minute
        self.last_call = 0.0
        self.name = name
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                wait_time = self.interval - elapsed
                time.sleep(wait_time)
            self.last_call = time.time()


# Alpaca IEX: 200 req/min → target 150/min to leave headroom
_alpaca_limiter = _RateLimiter(150, "alpaca")
# Binance: 1200 weight/min → very generous, but space out slightly
_binance_limiter = _RateLimiter(600, "binance")
# Yahoo Finance (yfinance): no official limit, but be polite — 30/min
_yfinance_limiter = _RateLimiter(30, "yfinance")


# Alpha Vantage: 5 req/min free tier
_alpha_vantage_limiter = _RateLimiter(5, "alpha_vantage")


# NewsAPI: 100 req/day free tier — tiered priority system
# Budget: metals (XAU, XAG) get 20-min refresh during active hours (~84/day)
# All other tickers: Yahoo-only (0 NewsAPI calls)
# BTC/ETH: already served by CryptoCompare, not NewsAPI
_newsapi_daily_count = 0
_newsapi_daily_reset = 0.0  # timestamp of last reset
_NEWSAPI_DAILY_BUDGET = 90  # leave 10-call margin
_newsapi_lock = threading.Lock()

# Tier 1 = 20-min TTL during active hours; Tier 2 = 3h; rest = Yahoo-only
_NEWSAPI_PRIORITY = {"XAU": 1, "XAG": 1, "MSTR": 2}

# Better search queries — raw ticker symbols return sparse results on NewsAPI
_NEWSAPI_SEARCH_QUERIES = {
    "XAU": "gold AND (price OR market OR ounce OR bullion OR futures OR commodity)",
    "XAG": "silver AND (price OR market OR ounce OR bullion OR futures OR commodity)",
    "MSTR": "MicroStrategy OR MSTR",
}

# Active monitoring: 08:00-22:00 CET = 07:00-21:00 UTC
_NEWSAPI_ACTIVE_START_UTC = 7
_NEWSAPI_ACTIVE_END_UTC = 21


def newsapi_quota_ok() -> bool:
    """Check if we still have NewsAPI quota today. Thread-safe."""
    global _newsapi_daily_count, _newsapi_daily_reset
    now = time.time()
    with _newsapi_lock:
        # Reset counter at midnight UTC
        from datetime import datetime, timezone
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()
        if _newsapi_daily_reset < today_start:
            _newsapi_daily_count = 0
            _newsapi_daily_reset = now
        return _newsapi_daily_count < _NEWSAPI_DAILY_BUDGET


def newsapi_track_call():
    """Increment NewsAPI daily counter. Call after each successful API request."""
    global _newsapi_daily_count
    with _newsapi_lock:
        _newsapi_daily_count += 1
        if _newsapi_daily_count == _NEWSAPI_DAILY_BUDGET:
            logger.warning("NewsAPI daily budget exhausted (%d/%d), falling back to Yahoo",
                          _newsapi_daily_count, _NEWSAPI_DAILY_BUDGET)


def newsapi_ttl_for_ticker(ticker: str):
    """Dynamic TTL based on ticker priority and time of day.

    Returns TTL in seconds, or None to skip NewsAPI for this ticker.
    Tier 1 (metals): 20-min during active hours (08:00-22:00 CET).
    Other tickers: None (Yahoo-only, saves budget for metals).
    """
    short = ticker.upper().replace("-USD", "")
    priority = _NEWSAPI_PRIORITY.get(short)
    if priority is None:
        return None

    from datetime import datetime, timezone
    hour_utc = datetime.now(timezone.utc).hour
    is_active = _NEWSAPI_ACTIVE_START_UTC <= hour_utc < _NEWSAPI_ACTIVE_END_UTC

    if is_active:
        if priority == 1:
            return 1200   # 20 min — metals
        return 10800      # 3h — secondary (MSTR etc.)
    return None  # off-hours: Yahoo-only


def newsapi_search_query(ticker: str) -> str:
    """Optimized search query for NewsAPI. Falls back to ticker symbol."""
    short = ticker.upper().replace("-USD", "")
    return _NEWSAPI_SEARCH_QUERIES.get(short, short)


# TTL constants for tool caching
FUNDAMENTALS_TTL = 86400  # 24 hours
ONCHAIN_TTL = 43200      # 12 hours (on-chain data updates slowly)
FEAR_GREED_TTL = 300     # 5 min
SENTIMENT_TTL = 900      # 15 min
MINISTRAL_TTL = 900      # 15 min
ML_SIGNAL_TTL = 900      # 15 min
FUNDING_RATE_TTL = 900   # 15 min
VOLUME_TTL = 300         # 5 min
NEWSAPI_TTL = 1800       # 30 min fallback — overridden by newsapi_ttl_for_ticker()
