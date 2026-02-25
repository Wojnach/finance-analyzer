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


_MAX_STALE_FACTOR = 5  # return None if cached data is older than TTL * this factor


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
        if len(_tool_cache) > _CACHE_MAX_SIZE:
            expired = [k for k, v in _tool_cache.items() if now - v["time"] > 3600]
            for k in expired:
                del _tool_cache[k]
    try:
        data = func(*args)
        with _cache_lock:
            _tool_cache[key] = {"data": data, "time": now}
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


# TTL constants for tool caching
FUNDAMENTALS_TTL = 86400  # 24 hours
FEAR_GREED_TTL = 300     # 5 min
SENTIMENT_TTL = 900      # 15 min
MINISTRAL_TTL = 900      # 15 min
ML_SIGNAL_TTL = 900      # 15 min
FUNDING_RATE_TTL = 900   # 15 min
VOLUME_TTL = 300         # 5 min
