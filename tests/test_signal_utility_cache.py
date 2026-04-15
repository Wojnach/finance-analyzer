"""Tests for the in-memory signal_utility cache added 2026-04-15 as part
of the BUG-178 mitigation.

The cache stores the result of signal_utility(horizon) keyed by horizon
with a 300s TTL. It exists to prevent the 5 parallel ticker threads from
all paying the ~3.6s cold walk of the signal log every cycle when the
OS file cache is cold.
"""
from __future__ import annotations

from unittest.mock import patch

import portfolio.accuracy_stats as acc_mod
from portfolio.accuracy_stats import (
    _SIGNAL_UTILITY_CACHE_TTL,
    _signal_utility_cache,
    invalidate_signal_utility_cache,
    signal_utility,
)


class TestSignalUtilityCacheHit:
    def setup_method(self):
        invalidate_signal_utility_cache()

    def test_second_call_is_cached(self):
        """The second call for the same horizon should hit the cache
        (no re-entry into _compute_signal_utility)."""
        with patch.object(acc_mod, "_compute_signal_utility", wraps=acc_mod._compute_signal_utility) as spy:
            r1 = signal_utility("1d")
            r2 = signal_utility("1d")
        assert r1 is r2, "cached result must be the same object (no recomputation)"
        assert spy.call_count == 1, "second call should not re-enter _compute_signal_utility"

    def test_different_horizons_are_separate_cache_entries(self):
        with patch.object(acc_mod, "_compute_signal_utility", wraps=acc_mod._compute_signal_utility) as spy:
            signal_utility("1d")
            signal_utility("3d")
            signal_utility("1d")  # cached
            signal_utility("3d")  # cached
        assert spy.call_count == 2, "one miss per unique horizon, then cache hits"

    def test_explicit_entries_bypass_cache(self):
        """Passing an explicit entries list must skip the cache so test
        fixtures with curated datasets don't contaminate production state."""
        with patch.object(acc_mod, "_compute_signal_utility", wraps=acc_mod._compute_signal_utility) as spy:
            signal_utility("1d", entries=[])  # explicit
            signal_utility("1d", entries=[])  # explicit — still recomputes
        assert spy.call_count == 2, "explicit entries must not be cached"


class TestSignalUtilityCacheTTL:
    def setup_method(self):
        invalidate_signal_utility_cache()

    def test_expired_cache_refreshes(self):
        """When the cached timestamp is older than TTL, the next call
        should recompute rather than return stale data."""
        # Seed cache with an artificially old timestamp.
        signal_utility("1d")
        with acc_mod._signal_utility_cache_lock:
            _ts, value = _signal_utility_cache["1d"]
            _signal_utility_cache["1d"] = (_ts - _SIGNAL_UTILITY_CACHE_TTL - 10, value)

        with patch.object(acc_mod, "_compute_signal_utility", wraps=acc_mod._compute_signal_utility) as spy:
            signal_utility("1d")
        assert spy.call_count == 1, "expired cache entry must trigger recompute"

    def test_fresh_cache_is_reused_within_ttl(self):
        signal_utility("1d")
        with acc_mod._signal_utility_cache_lock:
            _ts, value = _signal_utility_cache["1d"]
            # Set timestamp to "5 seconds ago" (well within TTL)
            import time
            _signal_utility_cache["1d"] = (time.time() - 5, value)
        with patch.object(acc_mod, "_compute_signal_utility", wraps=acc_mod._compute_signal_utility) as spy:
            signal_utility("1d")
        assert spy.call_count == 0, "fresh cache entry must not trigger recompute"


class TestInvalidate:
    def test_invalidate_clears_cache(self):
        signal_utility("1d")
        assert "1d" in _signal_utility_cache
        invalidate_signal_utility_cache()
        assert "1d" not in _signal_utility_cache

    def test_invalidate_is_idempotent(self):
        invalidate_signal_utility_cache()
        invalidate_signal_utility_cache()  # must not raise


class TestCorrectness:
    """The cache must not change the RESULT of signal_utility, only its timing."""

    def setup_method(self):
        invalidate_signal_utility_cache()

    def test_cached_result_equals_uncached(self):
        """First call populates the cache; an explicit-entries call should
        return the same structure (same keys, same shape)."""
        cached = signal_utility("1d")
        # Force recompute by clearing cache.
        invalidate_signal_utility_cache()
        uncached = signal_utility("1d")
        # Both are dicts keyed by signal name.
        assert set(cached.keys()) == set(uncached.keys())
        # Shape per-signal
        for sig in cached:
            assert set(cached[sig].keys()) == {"avg_return", "total_return", "samples", "utility_score"}
            assert cached[sig] == uncached[sig]

    def test_empty_entries_returns_zeros(self):
        """signal_utility called with an explicit empty entries list must
        bypass the cache AND return zeros (not touch disk)."""
        invalidate_signal_utility_cache()
        result = signal_utility("1d", entries=[])
        for stats in result.values():
            assert stats["samples"] == 0
            assert stats["total_return"] == 0.0
            assert stats["avg_return"] == 0.0
            assert stats["utility_score"] == 0.0
        # Cache MUST remain unpopulated because explicit entries bypass it.
        assert "1d" not in _signal_utility_cache


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
