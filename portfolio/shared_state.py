"""Shared mutable state for the portfolio system.

All modules that need caching or cross-cycle state import from here.
This avoids circular imports and provides a single source of truth.
"""

import logging
import threading
import time
from datetime import UTC

logger = logging.getLogger("portfolio.shared_state")

# Tool cache — avoid re-running expensive tools every cycle
_tool_cache = {}

# Retry cooldown for _cached() errors
_RETRY_COOLDOWN = 60


_CACHE_MAX_SIZE = 256  # evict expired entries when cache exceeds this size
_cache_lock = threading.Lock()

# BUG-166: Dogpile/thundering-herd prevention.
# Tracks which keys are currently being refreshed. When a thread sees a cache
# miss and the key is already loading, it returns stale data (if available)
# instead of calling the function redundantly.
_loading_keys: set[str] = set()
_LOADING_TIMEOUT = 120  # seconds to wait for a loading thread before giving up
# C11/SS1: Track when each key was added to _loading_keys for eviction of
# permanently stuck keys (batch flush crash before updating cache).
_loading_timestamps: dict[str, float] = {}

_MAX_STALE_FACTOR = 3  # return None if cached data is older than TTL * this factor


def _cached(key, ttl, func, *args):
    """Cache-through helper: returns cached data if fresh, else calls func.

    Dogpile prevention (BUG-166): when multiple threads detect a cache miss
    simultaneously, only one thread fetches the data. Others return stale
    data if available, preventing redundant expensive calls (LLM inference,
    API requests) and model swap contention.

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

        # C11/SS1: Evict stuck loading keys older than _LOADING_TIMEOUT seconds.
        _now_evict = time.time()
        stuck = [k for k, ts in _loading_timestamps.items()
                 if _now_evict - ts > _LOADING_TIMEOUT]
        for k in stuck:
            _loading_keys.discard(k)
            _loading_timestamps.pop(k, None)
            logger.debug("[%s] evicted stuck loading key (timeout %ds)", k, _LOADING_TIMEOUT)

        # BUG-166: Dogpile prevention — if another thread is already loading
        # this key, return stale data instead of calling func redundantly.
        if key in _loading_keys:
            if key in _tool_cache:
                age = now - _tool_cache[key]["time"]
                max_stale = ttl * _MAX_STALE_FACTOR
                if age <= max_stale:
                    logger.debug("[%s] stale-while-revalidate (another thread loading)", key)
                    return _tool_cache[key]["data"]
            # No stale data available — return None rather than pile on
            logger.debug("[%s] no stale data, another thread loading — returning None", key)
            return None
        _loading_keys.add(key)
        _loading_timestamps[key] = time.time()

    try:
        data = func(*args)
        with _cache_lock:
            _tool_cache[key] = {"data": data, "time": now, "ttl": ttl}
            _loading_keys.discard(key)
        return data
    except KeyboardInterrupt:
        with _cache_lock:
            _loading_keys.discard(key)
        logger.warning("[%s] interrupted (KeyboardInterrupt), returning None", key)
        return None
    except Exception as e:
        logger.warning("[%s] error: %s", key, e)
        with _cache_lock:
            _loading_keys.discard(key)
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


def _cached_or_enqueue(key, ttl, enqueue_fn, context,
                        should_enqueue_fn=None, max_stale_factor=None):
    """Check cache — if fresh return it, if expired enqueue for batch and return stale.

    Unlike _cached(), this never calls the model directly. On miss, it adds
    the request to the batch queue and returns stale data (or None).

    Dogpile prevention (Codex finding #5): uses _loading_keys to avoid
    re-enqueuing the same key every cycle if the batch flush hasn't run yet.

    2026-04-10 (perf/llama-swap-reduction) — two new optional parameters to
    support rotation scheduling of LLM signals:

    - should_enqueue_fn: callable returning bool. If provided and the cache
      is stale-but-present, skip the enqueue when the callback says "no"
      (rotation off-cycle). If stale data is NOT available, force-enqueue
      regardless of the callback — we cannot leave the caller empty-handed
      when no stale fallback exists. Default None means "always enqueue",
      which preserves the pre-rotation behavior for every existing caller.

    - max_stale_factor: integer override for how stale data can be returned,
      in multiples of ttl. Default None means use the module-level
      _MAX_STALE_FACTOR. LLM rotation passes 5 here so each rotated vote
      can stay valid across the full rotation cycle (3 * TTL) plus slippage.
    """
    now = time.time()
    effective_stale_factor = (
        max_stale_factor if max_stale_factor is not None else _MAX_STALE_FACTOR
    )
    with _cache_lock:
        if key in _tool_cache and now - _tool_cache[key]["time"] < ttl:
            return _tool_cache[key]["data"]

        # Check stale availability BEFORE deciding whether to enqueue, because
        # the rotation gate can only safely skip enqueue when we have stale
        # fallback to return. If stale is exhausted we must force-enqueue.
        stale_data = None
        stale_available = False
        if key in _tool_cache:
            age = now - _tool_cache[key]["time"]
            if age <= ttl * effective_stale_factor:
                stale_available = True
                stale_data = _tool_cache[key]["data"]

        # Decide whether to enqueue:
        # - Default (no should_enqueue_fn): always enqueue (legacy behavior)
        # - Callback returns True: enqueue (rotation on-cycle, or force path)
        # - Callback returns False AND stale available: skip (rotation off-cycle,
        #   stale fallback carries us until next on-cycle)
        # - Callback returns False AND stale NOT available: enqueue anyway
        #   (fresh cold path; caller has no fallback, we must refresh)
        if should_enqueue_fn is None:
            should_enq = True
        else:
            try:
                should_enq = bool(should_enqueue_fn()) or not stale_available
            except Exception as e:
                logger.warning(
                    "[%s] should_enqueue_fn raised, defaulting to enqueue: %s",
                    key, e,
                )
                should_enq = True

        if should_enq and enqueue_fn and context is not None and key not in _loading_keys:
            _loading_keys.add(key)
            # C11/SS1: Track enqueue time for stuck-key eviction.
            _loading_timestamps[key] = time.time()
            enqueue_fn(key, context)

        # Return stale if available
        if stale_available:
            return stale_data
    return None


# 2026-04-10 (perf/llama-swap-reduction): monotonic counter of full-LLM
# batch flushes that actually processed work. Drives rotation scheduling in
# portfolio.llm_batch.is_llm_on_cycle — incremented at the end of
# flush_llm_batch() iff at least one phase had queued items. In-memory only,
# resets to 0 on process start; on restart the rotation deterministically
# restarts at ministral with a cold-start warmup cycle that runs all LLMs.
_full_llm_cycle_count = 0


def _update_cache(key, data, ttl=None):
    """Update a cache entry directly (for batch flush results)."""
    with _cache_lock:
        _loading_keys.discard(key)
        # C11/SS1: Clean up timestamp when key is resolved.
        _loading_timestamps.pop(key, None)
        _tool_cache[key] = {
            "data": data,
            "time": time.time(),
            "ttl": ttl or 900,
        }


# Cycle counter — incremented at the start of each run() to invalidate per-cycle caches
_run_cycle_id = 0

# Current market state — updated each run() cycle, used by data_collector for yfinance fallback
_current_market_state = "open"

# Regime detection cache (invalidated each cycle)
# BUG-169: Protected by _regime_lock — accessed from 8 concurrent ThreadPoolExecutor threads
_regime_cache = {}
_regime_cache_cycle = 0
_regime_lock = threading.Lock()


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


# H11/DC-R3-4: yfinance is not thread-safe. This lock is shared across all
# modules (fear_greed, golddigger/data_provider, data_collector) so that
# concurrent calls from the 8-worker ThreadPoolExecutor are serialized.
# data_collector.py imports this lock instead of defining its own.
yfinance_lock = threading.Lock()

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
        from datetime import datetime
        today_start = datetime.now(UTC).replace(
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

    from datetime import datetime
    hour_utc = datetime.now(UTC).hour
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
