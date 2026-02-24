"""Tests for portfolio/shared_state.py — caching, eviction, rate limiting."""

import time
import threading
from unittest.mock import patch, MagicMock

import pytest

from portfolio import shared_state
from portfolio.shared_state import _cached, _RateLimiter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_cache():
    """Reset the module-level cache between tests."""
    shared_state._tool_cache.clear()


@pytest.fixture(autouse=True)
def clean_cache():
    """Ensure every test starts with an empty cache."""
    _clear_cache()
    yield
    _clear_cache()


# ===========================================================================
# _cached() — basic behaviour
# ===========================================================================


class TestCachedFreshData:
    """_cached() calls func and stores result when cache is empty."""

    def test_returns_func_result(self):
        result = _cached("k1", 60, lambda: "fresh")
        assert result == "fresh"

    def test_stores_in_cache(self):
        _cached("k1", 60, lambda: "stored")
        assert "k1" in shared_state._tool_cache
        assert shared_state._tool_cache["k1"]["data"] == "stored"

    def test_func_called_once(self):
        fn = MagicMock(return_value=42)
        _cached("k1", 60, fn)
        fn.assert_called_once()

    def test_passes_args_to_func(self):
        fn = MagicMock(return_value="ok")
        _cached("k1", 60, fn, "a", "b")
        fn.assert_called_once_with("a", "b")


class TestCachedWithinTTL:
    """_cached() returns cached data when within TTL, without calling func again."""

    def test_returns_cached_value(self):
        _cached("k1", 60, lambda: "first")
        counter = MagicMock(return_value="second")
        result = _cached("k1", 60, counter)
        assert result == "first"
        counter.assert_not_called()

    @patch("portfolio.shared_state.time")
    def test_within_ttl_boundary(self, mock_time):
        """At exactly TTL - 1 second, data is still fresh."""
        mock_time.time.return_value = 1000.0
        _cached("k1", 60, lambda: "original")

        mock_time.time.return_value = 1059.0  # 59s < 60s TTL
        fn = MagicMock(return_value="new")
        result = _cached("k1", 60, fn)
        assert result == "original"
        fn.assert_not_called()


class TestCachedTTLExpiry:
    """_cached() refreshes data after TTL expires."""

    @patch("portfolio.shared_state.time")
    def test_refreshes_after_ttl(self, mock_time):
        mock_time.time.return_value = 1000.0
        _cached("k1", 60, lambda: "old")

        mock_time.time.return_value = 1061.0  # 61s > 60s TTL
        result = _cached("k1", 60, lambda: "new")
        assert result == "new"

    @patch("portfolio.shared_state.time")
    def test_func_called_on_refresh(self, mock_time):
        mock_time.time.return_value = 1000.0
        _cached("k1", 10, lambda: "v1")

        mock_time.time.return_value = 1011.0
        fn = MagicMock(return_value="v2")
        _cached("k1", 10, fn)
        fn.assert_called_once()

    @patch("portfolio.shared_state.time")
    def test_at_exact_ttl_boundary_refreshes(self, mock_time):
        """At exactly TTL seconds the condition `now - time < ttl` is False."""
        mock_time.time.return_value = 1000.0
        _cached("k1", 60, lambda: "old")

        mock_time.time.return_value = 1060.0  # exactly 60s
        result = _cached("k1", 60, lambda: "new")
        assert result == "new"


# ===========================================================================
# _cached() — error handling / stale data
# ===========================================================================


class TestCachedStaleDataOnError:
    """On error, _cached() returns stale data if within MAX_STALE_FACTOR * TTL."""

    @patch("portfolio.shared_state.time")
    def test_returns_stale_on_error(self, mock_time):
        mock_time.time.return_value = 1000.0
        _cached("k1", 60, lambda: "good")

        # TTL expired, func raises
        mock_time.time.return_value = 1100.0  # 100s old, max stale = 60*5 = 300s
        result = _cached("k1", 60, self._failing_func)
        assert result == "good"

    @patch("portfolio.shared_state.time")
    def test_returns_stale_near_max_boundary(self, mock_time):
        """Just under MAX_STALE_FACTOR * TTL should still return stale data."""
        mock_time.time.return_value = 1000.0
        _cached("k1", 60, lambda: "ok")

        # age = 299s, max_stale = 300s  →  should still return stale
        mock_time.time.return_value = 1299.0
        result = _cached("k1", 60, self._failing_func)
        assert result == "ok"

    @patch("portfolio.shared_state.time")
    def test_returns_none_when_stale_exceeds_max(self, mock_time):
        """Beyond MAX_STALE_FACTOR * TTL, returns None."""
        mock_time.time.return_value = 1000.0
        _cached("k1", 60, lambda: "old")

        # age = 301s, max_stale = 300s  →  None
        mock_time.time.return_value = 1301.0
        result = _cached("k1", 60, self._failing_func)
        assert result is None

    @patch("portfolio.shared_state.time")
    def test_returns_none_at_exact_max_stale(self, mock_time):
        """At exactly MAX_STALE_FACTOR * TTL, condition `age > max_stale` is False (<=)."""
        mock_time.time.return_value = 1000.0
        _cached("k1", 60, lambda: "val")

        # age = 300s, max_stale = 300s  →  age > max_stale is False → returns stale
        mock_time.time.return_value = 1300.0
        result = _cached("k1", 60, self._failing_func)
        assert result == "val"

    def test_returns_none_when_no_cached_data_on_error(self):
        """First call fails with no cache → returns None."""
        result = _cached("k1", 60, self._failing_func)
        assert result is None

    @staticmethod
    def _failing_func():
        raise RuntimeError("data source down")


# ===========================================================================
# _cached() — retry cooldown
# ===========================================================================


class TestCachedRetryCooldown:
    """On error, _cached() adjusts the cache timestamp to enforce a retry cooldown."""

    @patch("portfolio.shared_state.time")
    def test_cooldown_prevents_immediate_retry(self, mock_time):
        """After an error, the next call within 60s should still return stale data
        without calling func again (because the adjusted timestamp keeps it 'fresh')."""
        mock_time.time.return_value = 1000.0
        _cached("k1", 60, lambda: "original")

        # TTL expired; func fails → stale data returned, timestamp adjusted
        mock_time.time.return_value = 1070.0
        call_count = 0

        def failing():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        result = _cached("k1", 60, failing)
        assert result == "original"
        assert call_count == 1

        # Adjusted time = now - ttl + RETRY_COOLDOWN = 1070 - 60 + 60 = 1070
        # So at 1071 (1s later), age = 1071 - 1070 = 1s < 60s TTL → cache hit
        mock_time.time.return_value = 1071.0
        fn2 = MagicMock(return_value="should_not_call")
        result = _cached("k1", 60, fn2)
        assert result == "original"
        fn2.assert_not_called()

    @patch("portfolio.shared_state.time")
    def test_cooldown_expires_after_retry_delay(self, mock_time):
        """After the 60s retry cooldown, func is called again."""
        mock_time.time.return_value = 1000.0
        _cached("k1", 60, lambda: "original")

        # Error at t=1070 → adjusted time = 1070 - 60 + 60 = 1070
        mock_time.time.return_value = 1070.0
        _cached("k1", 60, self._raise)

        # At t=1131 → age = 1131 - 1070 = 61s > 60s TTL → retry
        mock_time.time.return_value = 1131.0
        fn = MagicMock(return_value="recovered")
        result = _cached("k1", 60, fn)
        assert result == "recovered"
        fn.assert_called_once()

    @staticmethod
    def _raise():
        raise RuntimeError("fail")


# ===========================================================================
# _cached() — cache eviction
# ===========================================================================


class TestCacheEviction:
    """When cache size exceeds _CACHE_MAX_SIZE, expired entries are evicted."""

    @patch("portfolio.shared_state.time")
    def test_evicts_expired_entries(self, mock_time):
        mock_time.time.return_value = 1000.0

        # Fill cache with 260 entries (> 256 max)
        for i in range(260):
            shared_state._tool_cache[f"old_{i}"] = {"data": i, "time": 1000.0}

        # Advance time so all entries are >3600s old (expired)
        mock_time.time.return_value = 5000.0

        # Next _cached() call triggers eviction
        result = _cached("new_key", 60, lambda: "fresh")
        assert result == "fresh"

        # Expired entries should be removed; only "new_key" should remain
        assert "new_key" in shared_state._tool_cache
        assert len(shared_state._tool_cache) <= 2  # new_key + possibly a few non-expired

    @patch("portfolio.shared_state.time")
    def test_does_not_evict_non_expired_entries(self, mock_time):
        mock_time.time.return_value = 5000.0

        # Fill with 260 entries, all recent (not expired at 3600s threshold)
        for i in range(260):
            shared_state._tool_cache[f"recent_{i}"] = {"data": i, "time": 4500.0}

        # Call _cached() — eviction runs but nothing is expired
        result = _cached("trigger", 60, lambda: "val")
        assert result == "val"
        # All 260 + 1 new entry remain (nothing expired)
        assert len(shared_state._tool_cache) == 261

    @patch("portfolio.shared_state.time")
    def test_no_eviction_when_under_max_size(self, mock_time):
        mock_time.time.return_value = 1000.0

        # Fill with only 10 entries (well under 256)
        for i in range(10):
            shared_state._tool_cache[f"k_{i}"] = {"data": i, "time": 1.0}  # very old

        mock_time.time.return_value = 5000.0

        # Even though entries are old, no eviction because cache size <= 256
        result = _cached("new", 60, lambda: "x")
        assert result == "x"
        # Old entries survive because size check comes first
        assert len(shared_state._tool_cache) == 11

    @patch("portfolio.shared_state.time")
    def test_partial_eviction_mixed_ages(self, mock_time):
        """Only entries older than 3600s are evicted; fresh entries survive."""
        mock_time.time.return_value = 5000.0

        # 200 old entries (time=1000, age=4000s > 3600s)
        for i in range(200):
            shared_state._tool_cache[f"old_{i}"] = {"data": i, "time": 1000.0}
        # 60 fresh entries (time=4500, age=500s < 3600s)
        for i in range(60):
            shared_state._tool_cache[f"fresh_{i}"] = {"data": i, "time": 4500.0}

        assert len(shared_state._tool_cache) == 260

        result = _cached("trigger", 60, lambda: "val")
        assert result == "val"

        # Old entries evicted, fresh entries + trigger remain
        assert len(shared_state._tool_cache) == 61  # 60 fresh + 1 trigger
        for i in range(60):
            assert f"fresh_{i}" in shared_state._tool_cache


# ===========================================================================
# _cached() — multiple keys
# ===========================================================================


class TestCachedMultipleKeys:
    """Different keys are cached independently."""

    def test_independent_keys(self):
        _cached("a", 60, lambda: "alpha")
        _cached("b", 60, lambda: "beta")
        assert shared_state._tool_cache["a"]["data"] == "alpha"
        assert shared_state._tool_cache["b"]["data"] == "beta"

    @patch("portfolio.shared_state.time")
    def test_different_ttls(self, mock_time):
        mock_time.time.return_value = 1000.0
        _cached("short", 10, lambda: "s1")
        _cached("long", 120, lambda: "l1")

        mock_time.time.return_value = 1015.0  # 15s: short expired, long fresh
        result_short = _cached("short", 10, lambda: "s2")
        fn_long = MagicMock(return_value="l2")
        result_long = _cached("long", 120, fn_long)

        assert result_short == "s2"
        assert result_long == "l1"
        fn_long.assert_not_called()


# ===========================================================================
# _cached() — thread safety
# ===========================================================================


class TestCachedThreadSafety:
    """Verify that concurrent calls don't corrupt the cache."""

    def test_concurrent_writes(self):
        """Multiple threads writing different keys should all succeed."""
        results = {}
        errors = []

        def worker(key, value):
            try:
                r = _cached(key, 60, lambda: value)
                results[key] = r
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"t_{i}", i)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors
        assert len(results) == 20
        for i in range(20):
            assert results[f"t_{i}"] == i


# ===========================================================================
# _RateLimiter
# ===========================================================================


class TestRateLimiterSleeps:
    """_RateLimiter.wait() sleeps when calls are too fast."""

    @patch("portfolio.shared_state.time")
    def test_sleeps_when_calls_too_fast(self, mock_time):
        limiter = _RateLimiter(60, name="test")  # interval = 1.0s
        limiter.last_call = 0.0

        mock_time.time.return_value = 0.3  # 0.3s since last call, need 1.0s
        mock_time.sleep = MagicMock()

        limiter.wait()

        mock_time.sleep.assert_called_once()
        sleep_arg = mock_time.sleep.call_args[0][0]
        assert abs(sleep_arg - 0.7) < 0.01  # should sleep ~0.7s

    @patch("portfolio.shared_state.time")
    def test_sleep_duration_varies(self, mock_time):
        limiter = _RateLimiter(120, name="test")  # interval = 0.5s
        limiter.last_call = 10.0

        mock_time.time.return_value = 10.2  # 0.2s elapsed, need 0.5s
        mock_time.sleep = MagicMock()

        limiter.wait()

        sleep_arg = mock_time.sleep.call_args[0][0]
        assert abs(sleep_arg - 0.3) < 0.01  # should sleep ~0.3s


class TestRateLimiterNoSleep:
    """_RateLimiter.wait() does not sleep when calls are spaced out."""

    @patch("portfolio.shared_state.time")
    def test_no_sleep_when_spaced_out(self, mock_time):
        limiter = _RateLimiter(60, name="test")  # interval = 1.0s
        limiter.last_call = 0.0

        mock_time.time.return_value = 5.0  # 5s since last call >> 1.0s interval
        mock_time.sleep = MagicMock()

        limiter.wait()

        mock_time.sleep.assert_not_called()

    @patch("portfolio.shared_state.time")
    def test_no_sleep_at_exact_interval(self, mock_time):
        """At exactly the interval boundary, elapsed == interval → no sleep."""
        limiter = _RateLimiter(60, name="test")  # interval = 1.0s
        limiter.last_call = 10.0

        mock_time.time.return_value = 11.0  # exactly 1.0s elapsed
        mock_time.sleep = MagicMock()

        limiter.wait()

        mock_time.sleep.assert_not_called()

    @patch("portfolio.shared_state.time")
    def test_first_call_no_sleep(self, mock_time):
        """First call ever (last_call=0.0) with any positive time should not sleep."""
        limiter = _RateLimiter(60, name="test")
        assert limiter.last_call == 0.0

        mock_time.time.return_value = 1000.0
        mock_time.sleep = MagicMock()

        limiter.wait()

        mock_time.sleep.assert_not_called()


class TestRateLimiterUpdatesLastCall:
    """_RateLimiter.wait() updates last_call to current time after waiting."""

    @patch("portfolio.shared_state.time")
    def test_updates_last_call(self, mock_time):
        limiter = _RateLimiter(60, name="test")
        limiter.last_call = 0.0

        # time.time() is called twice: once for elapsed calc, once after sleep for last_call
        mock_time.time.side_effect = [5.0, 5.0]
        mock_time.sleep = MagicMock()

        limiter.wait()

        assert limiter.last_call == 5.0

    @patch("portfolio.shared_state.time")
    def test_updates_last_call_after_sleep(self, mock_time):
        limiter = _RateLimiter(60, name="test")  # interval = 1.0s
        limiter.last_call = 10.0

        # First time.time() = 10.3 (triggers sleep), second time.time() = 11.0 (after sleep)
        mock_time.time.side_effect = [10.3, 11.0]
        mock_time.sleep = MagicMock()

        limiter.wait()

        assert limiter.last_call == 11.0


class TestRateLimiterInterval:
    """_RateLimiter computes the correct interval from max_per_minute."""

    def test_interval_60_per_min(self):
        limiter = _RateLimiter(60, name="test")
        assert limiter.interval == pytest.approx(1.0)

    def test_interval_150_per_min(self):
        limiter = _RateLimiter(150, name="test")
        assert limiter.interval == pytest.approx(0.4)

    def test_interval_30_per_min(self):
        limiter = _RateLimiter(30, name="test")
        assert limiter.interval == pytest.approx(2.0)

    def test_interval_600_per_min(self):
        limiter = _RateLimiter(600, name="test")
        assert limiter.interval == pytest.approx(0.1)


# ===========================================================================
# Module-level limiters exist
# ===========================================================================


class TestModuleLimiters:
    """Verify the pre-configured limiters are accessible."""

    def test_alpaca_limiter_exists(self):
        assert shared_state._alpaca_limiter.interval == pytest.approx(60.0 / 150)

    def test_binance_limiter_exists(self):
        assert shared_state._binance_limiter.interval == pytest.approx(60.0 / 600)

    def test_yfinance_limiter_exists(self):
        assert shared_state._yfinance_limiter.interval == pytest.approx(60.0 / 30)


# ===========================================================================
# TTL constants
# ===========================================================================


class TestTTLConstants:
    def test_fear_greed_ttl(self):
        assert shared_state.FEAR_GREED_TTL == 300

    def test_sentiment_ttl(self):
        assert shared_state.SENTIMENT_TTL == 900

    def test_ministral_ttl(self):
        assert shared_state.MINISTRAL_TTL == 900

    def test_ml_signal_ttl(self):
        assert shared_state.ML_SIGNAL_TTL == 900

    def test_funding_rate_ttl(self):
        assert shared_state.FUNDING_RATE_TTL == 900

    def test_volume_ttl(self):
        assert shared_state.VOLUME_TTL == 300


# ===========================================================================
# Module-level state variables
# ===========================================================================


class TestModuleState:
    def test_cache_max_size(self):
        assert shared_state._CACHE_MAX_SIZE == 256

    def test_retry_cooldown(self):
        assert shared_state._RETRY_COOLDOWN == 60

    def test_max_stale_factor(self):
        assert shared_state._MAX_STALE_FACTOR == 5

    def test_run_cycle_id_initial(self):
        assert isinstance(shared_state._run_cycle_id, int)

    def test_current_market_state_default(self):
        assert shared_state._current_market_state == "open"

    def test_cache_lock_is_lock(self):
        assert isinstance(shared_state._cache_lock, type(threading.Lock()))
