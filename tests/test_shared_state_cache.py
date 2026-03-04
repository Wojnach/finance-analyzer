"""Tests for shared_state.py cache eviction — BUG-55 LRU fallback."""
import time
import pytest
from unittest.mock import patch

from portfolio.shared_state import _cached, _tool_cache, _CACHE_MAX_SIZE, _cache_lock


@pytest.fixture(autouse=True)
def _clean_cache():
    """Clear the global cache before and after each test."""
    with _cache_lock:
        _tool_cache.clear()
    yield
    with _cache_lock:
        _tool_cache.clear()


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
            # Half stale (>1h), half fresh (<1h)
            for i in range(_CACHE_MAX_SIZE):
                age = 7200 if i % 2 == 0 else 60
                _tool_cache[f"mix_{i}"] = {"data": i, "time": now - age}
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
