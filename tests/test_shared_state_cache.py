"""Tests for shared_state.py cache eviction and dogpile prevention."""
import threading
import time

import pytest

from portfolio.shared_state import _CACHE_MAX_SIZE, _cache_lock, _cached, _loading_keys, _tool_cache


@pytest.fixture(autouse=True)
def _clean_cache():
    """Clear the global cache and loading flags before and after each test."""
    with _cache_lock:
        _tool_cache.clear()
        _loading_keys.clear()
    yield
    with _cache_lock:
        _tool_cache.clear()
        _loading_keys.clear()


class TestCacheEvictionStaleEntries:
    """Age-based eviction removes entries older than 1 hour."""

    def test_stale_entries_evicted(self):
        now = time.time()
        with _cache_lock:
            for i in range(_CACHE_MAX_SIZE + 10):
                _tool_cache[f"stale_{i}"] = {"data": i, "time": now - 7200}
        assert len(_tool_cache) > _CACHE_MAX_SIZE

        # Trigger eviction by calling _cached
        _cached("trigger_key", 60, lambda: "value")
        with _cache_lock:
            # Stale entries removed; only the new entry remains
            assert len(_tool_cache) <= _CACHE_MAX_SIZE + 1

    def test_fresh_entries_not_evicted_by_age(self):
        now = time.time()
        with _cache_lock:
            for i in range(_CACHE_MAX_SIZE + 10):
                _tool_cache[f"fresh_{i}"] = {"data": i, "time": now - 60}
        # All entries are <1h old, so age-based eviction won't remove them
        # But LRU fallback should kick in
        _cached("trigger_key", 60, lambda: "value")
        with _cache_lock:
            assert len(_tool_cache) <= _CACHE_MAX_SIZE + 1


class TestCacheEvictionLRUFallback:
    """LRU fallback evicts oldest 25% when all entries are fresh (BUG-55 fix)."""

    def test_lru_evicts_oldest_quarter(self):
        now = time.time()
        with _cache_lock:
            count = _CACHE_MAX_SIZE + 20
            for i in range(count):
                # Entries range from 10s old to (10 + count)s old, all < 1h
                _tool_cache[f"lru_{i}"] = {"data": i, "time": now - 10 - i}
        before = len(_tool_cache)
        assert before > _CACHE_MAX_SIZE

        _cached("trigger_key", 60, lambda: "value")
        with _cache_lock:
            after = len(_tool_cache)
            # Should have evicted ~25% of the entries + kept trigger_key
            assert after < before
            assert after <= _CACHE_MAX_SIZE + 1
            # Oldest entries (highest i) should be gone
            assert f"lru_{count - 1}" not in _tool_cache
            # Newest entries should survive
            assert "trigger_key" in _tool_cache

    def test_lru_evicts_at_least_one(self):
        """Even if 25% rounds to 0, at least 1 entry is evicted."""
        now = time.time()
        with _cache_lock:
            # Fill to just above limit
            for i in range(_CACHE_MAX_SIZE + 1):
                _tool_cache[f"min_{i}"] = {"data": i, "time": now - 10 - i}
        _cached("trigger_key", 60, lambda: "value")
        with _cache_lock:
            assert len(_tool_cache) <= _CACHE_MAX_SIZE + 1


class TestCacheEvictionMixed:
    """Mix of stale + fresh entries."""

    def test_stale_removed_first_then_lru_if_needed(self):
        now = time.time()
        with _cache_lock:
            # Half stale (>ttl*stale_factor), half fresh
            for i in range(_CACHE_MAX_SIZE):
                age = 7200 if i % 2 == 0 else 60
                # Stale entries get small ttl so 7200 > 60*3=180 → evicted
                ttl = 60 if i % 2 == 0 else 3600
                _tool_cache[f"mix_{i}"] = {"data": i, "time": now - age, "ttl": ttl}
            # Add extras to go over limit
            for i in range(20):
                _tool_cache[f"extra_{i}"] = {"data": i, "time": now - 30}

        _cached("trigger_key", 60, lambda: "value")
        with _cache_lock:
            # All stale entries should be gone
            stale_remaining = sum(
                1 for k in _tool_cache
                if k.startswith("mix_") and int(k.split("_")[1]) % 2 == 0
            )
            assert stale_remaining == 0


class TestCachedBasicBehavior:
    """Verify _cached still works correctly after eviction changes."""

    def test_cache_hit(self):
        call_count = 0

        def fetch():
            nonlocal call_count
            call_count += 1
            return "data"

        result1 = _cached("test_hit", 300, fetch)
        result2 = _cached("test_hit", 300, fetch)
        assert result1 == "data"
        assert result2 == "data"
        assert call_count == 1  # second call uses cache

    def test_cache_miss_on_expired(self):
        now = time.time()
        with _cache_lock:
            _tool_cache["test_expired"] = {"data": "old", "time": now - 600}

        result = _cached("test_expired", 300, lambda: "new")
        assert result == "new"

    def test_stale_fallback_on_error(self):
        now = time.time()
        with _cache_lock:
            _tool_cache["test_stale"] = {"data": "stale_data", "time": now - 100}

        def failing_func():
            raise RuntimeError("fail")

        result = _cached("test_stale", 300, failing_func)
        assert result == "stale_data"

    def test_none_when_stale_too_old(self):
        now = time.time()
        # _MAX_STALE_FACTOR = 3, TTL = 60 → max stale = 180s
        with _cache_lock:
            _tool_cache["test_too_old"] = {"data": "ancient", "time": now - 300}

        def failing_func():
            raise RuntimeError("fail")

        result = _cached("test_too_old", 60, failing_func)
        assert result is None


class TestDogpilePrevention:
    """BUG-166: Only one thread should refresh a cache key at a time.

    When a cache entry expires and multiple threads detect the miss,
    only one thread should call the underlying function. Others should
    return stale data (stale-while-revalidate pattern).
    """

    def test_only_one_thread_refreshes(self):
        """Verify only 1 out of N threads calls the function on cache miss."""
        call_count = {"n": 0}
        barrier = threading.Barrier(3)
        lock = threading.Lock()

        def slow_fetch():
            with lock:
                call_count["n"] += 1
            time.sleep(0.1)
            return "fresh_data"

        # Pre-seed with stale data so other threads get stale-while-revalidate
        now = time.time()
        with _cache_lock:
            _tool_cache["dogpile_test"] = {"data": "stale", "time": now - 600, "ttl": 300}

        results = [None, None, None]

        def worker(idx):
            barrier.wait()
            results[idx] = _cached("dogpile_test", 300, slow_fetch)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # Only one thread should have called slow_fetch
        assert call_count["n"] == 1, f"Expected 1 call, got {call_count['n']}"
        # All threads should return data (either fresh or stale)
        for r in results:
            assert r is not None

    def test_stale_data_returned_during_reload(self):
        """Threads that find the key loading should get stale data."""
        now = time.time()
        with _cache_lock:
            _tool_cache["stale_test"] = {"data": "stale_value", "time": now - 400, "ttl": 300}

        load_started = threading.Event()
        can_finish = threading.Event()

        def slow_fetch():
            load_started.set()
            can_finish.wait(timeout=5)
            return "fresh_value"

        # Thread 1 starts loading
        t1_result = [None]

        def t1():
            t1_result[0] = _cached("stale_test", 300, slow_fetch)

        thread1 = threading.Thread(target=t1)
        thread1.start()
        load_started.wait(timeout=5)

        # Thread 2 tries to get the same key while Thread 1 is loading
        t2_result = _cached("stale_test", 300, lambda: "should_not_call")
        assert t2_result == "stale_value", "Thread 2 should get stale data"

        # Let Thread 1 finish
        can_finish.set()
        thread1.join(timeout=5)
        assert t1_result[0] == "fresh_value"

    def test_loading_flag_cleared_on_exception(self):
        """Loading flag must be cleared even if the function raises."""

        def failing():
            raise RuntimeError("boom")

        # First call — fails but clears loading flag
        result1 = _cached("fail_test", 300, failing)
        assert result1 is None

        # Second call — should be able to try again (not stuck in loading)
        result2 = _cached("fail_test", 300, lambda: "recovered")
        assert result2 == "recovered"

    def test_no_dogpile_for_different_keys(self):
        """Different cache keys should not block each other."""
        call_counts = {"a": 0, "b": 0}

        def fetch_a():
            call_counts["a"] += 1
            time.sleep(0.05)
            return "data_a"

        def fetch_b():
            call_counts["b"] += 1
            time.sleep(0.05)
            return "data_b"

        barrier = threading.Barrier(2)

        def worker_a():
            barrier.wait()
            _cached("key_a", 300, fetch_a)

        def worker_b():
            barrier.wait()
            _cached("key_b", 300, fetch_b)

        t1 = threading.Thread(target=worker_a)
        t2 = threading.Thread(target=worker_b)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert call_counts["a"] == 1
        assert call_counts["b"] == 1


class TestCachedOrEnqueueExceptionSafety:
    """BUG-191: enqueue_fn exception must release _loading_keys to avoid 120s stale windows."""

    def test_enqueue_exception_releases_loading_key(self):
        from portfolio.shared_state import (
            _cached_or_enqueue,
            _cache_lock,
            _loading_keys,
            _loading_timestamps,
        )
        with _cache_lock:
            _loading_keys.discard("test_bug191")
            _loading_timestamps.pop("test_bug191", None)

        def failing_enqueue(key, ctx):
            raise RuntimeError("GPU OOM simulated")

        result = _cached_or_enqueue("test_bug191", 60, failing_enqueue, {"ticker": "BTC-USD"})
        assert result is None
        with _cache_lock:
            assert "test_bug191" not in _loading_keys
            assert "test_bug191" not in _loading_timestamps

    def test_enqueue_success_keeps_loading_key(self):
        from portfolio.shared_state import (
            _cached_or_enqueue,
            _cache_lock,
            _loading_keys,
            _loading_timestamps,
        )
        with _cache_lock:
            _loading_keys.discard("test_bug191_ok")
            _loading_timestamps.pop("test_bug191_ok", None)

        enqueue_called = []

        def ok_enqueue(key, ctx):
            enqueue_called.append(key)

        result = _cached_or_enqueue("test_bug191_ok", 60, ok_enqueue, {"ticker": "BTC-USD"})
        assert len(enqueue_called) == 1
        with _cache_lock:
            assert "test_bug191_ok" in _loading_keys
