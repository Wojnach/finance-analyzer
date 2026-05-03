"""Tests for the in-memory signal_utility cache added 2026-04-15 as part
of the BUG-178 mitigation, plus the disk-backed L2 cache added 2026-05-03.

L1 stores the result of signal_utility(horizon) keyed by horizon with a
300s TTL. It exists to prevent the 5 parallel ticker threads from all
paying the ~3.6s cold walk of the signal log every cycle when the OS file
cache is cold.

L2 mirrors L1 to disk with a 1h TTL, so the first cycle after a process
restart hits the disk cache instead of paying the ~49s parallel-cold-
compute cost we measured under 4-thread contention.
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

import portfolio.accuracy_stats as acc_mod
from portfolio.accuracy_stats import (
    _SIGNAL_UTILITY_CACHE_TTL,
    _signal_utility_cache,
    invalidate_signal_utility_cache,
    signal_utility,
)
from portfolio.file_utils import atomic_write_json


@pytest.fixture(autouse=True)
def _isolate_disk_cache(tmp_path, monkeypatch):
    """Redirect SIGNAL_UTILITY_CACHE_FILE to a per-test tmp path.

    Without this fixture, every test that calls signal_utility(horizon=X)
    would hit-or-write the production data/signal_utility_cache.json file.
    xdist-safe because tmp_path is per-worker AND per-test.
    """
    monkeypatch.setattr(
        acc_mod, "SIGNAL_UTILITY_CACHE_FILE", tmp_path / "signal_utility_cache.json"
    )
    yield


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
        """When the cached timestamp is older than TTL AND the L2 disk
        cache is also missing/stale, the next call should recompute
        rather than return stale data."""
        # Seed L1 with an artificially old timestamp.
        signal_utility("1d")
        with acc_mod._signal_utility_cache_lock:
            _ts, value = _signal_utility_cache["1d"]
            _signal_utility_cache["1d"] = (_ts - _SIGNAL_UTILITY_CACHE_TTL - 10, value)
        # Also remove L2 disk cache so it can't satisfy the lookup.
        # (Without this, L1 expiry now falls through to L2 — by design.)
        if acc_mod.SIGNAL_UTILITY_CACHE_FILE.exists():
            acc_mod.SIGNAL_UTILITY_CACHE_FILE.unlink()

        with patch.object(acc_mod, "_compute_signal_utility", wraps=acc_mod._compute_signal_utility) as spy:
            signal_utility("1d")
        assert spy.call_count == 1, "expired L1 + missing L2 must trigger recompute"

    def test_l1_expired_falls_through_to_l2(self):
        """L1 expired but L2 still fresh: serve from L2 without recomputing.
        This is the cross-restart hit path that motivated the L2 cache."""
        # Populate both L1 and L2 with one call.
        signal_utility("1d")
        # Expire L1 only — L2 remains fresh on disk.
        with acc_mod._signal_utility_cache_lock:
            _ts, value = _signal_utility_cache["1d"]
            _signal_utility_cache["1d"] = (_ts - _SIGNAL_UTILITY_CACHE_TTL - 10, value)

        with patch.object(acc_mod, "_compute_signal_utility") as spy:
            signal_utility("1d")
        assert spy.call_count == 0, "L1 expired but L2 fresh must serve L2 without recompute"

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


class TestSignalUtilityDiskCache:
    """Disk-backed L2 cache (added 2026-05-03)."""

    def setup_method(self):
        invalidate_signal_utility_cache()

    def test_l2_hit_skips_compute(self):
        """Pre-populate the disk cache, clear L1, then call signal_utility:
        it must serve from disk without re-entering _compute_signal_utility."""
        canned = {
            "rsi": {"avg_return": 0.5, "total_return": 5.0, "samples": 10, "utility_score": 1.58},
        }
        atomic_write_json(
            acc_mod.SIGNAL_UTILITY_CACHE_FILE,
            {"time": time.time(), "1d": canned},
        )
        with patch.object(acc_mod, "_compute_signal_utility") as spy:
            result = signal_utility("1d")
        assert spy.call_count == 0, "L2 hit must short-circuit compute"
        assert result == canned
        # L2 hit must also populate L1 so subsequent calls hit memory.
        assert "1d" in _signal_utility_cache

    def test_l2_miss_then_compute_populates_both(self):
        """First call: L1 miss, L2 miss → compute → write both layers."""
        assert not acc_mod.SIGNAL_UTILITY_CACHE_FILE.exists()
        with patch.object(
            acc_mod, "_compute_signal_utility", wraps=acc_mod._compute_signal_utility
        ) as spy:
            signal_utility("1d")
        assert spy.call_count == 1
        assert "1d" in _signal_utility_cache
        assert acc_mod.SIGNAL_UTILITY_CACHE_FILE.exists()
        # File contents include the horizon and a fresh timestamp.
        from portfolio.file_utils import load_json
        on_disk = load_json(acc_mod.SIGNAL_UTILITY_CACHE_FILE)
        assert "1d" in on_disk
        assert "time" in on_disk
        assert time.time() - on_disk["time"] < 5

    def test_l2_stale_triggers_recompute(self):
        """If the file's "time" is older than the L2 TTL, treat as miss."""
        canned = {"rsi": {"avg_return": 0.5, "total_return": 5.0, "samples": 10, "utility_score": 1.58}}
        atomic_write_json(
            acc_mod.SIGNAL_UTILITY_CACHE_FILE,
            {"time": time.time() - acc_mod._SIGNAL_UTILITY_DISK_TTL - 60, "1d": canned},
        )
        with patch.object(
            acc_mod, "_compute_signal_utility", wraps=acc_mod._compute_signal_utility
        ) as spy:
            signal_utility("1d")
        assert spy.call_count == 1, "stale L2 must not be served"

    def test_l2_corrupt_file_falls_back_to_compute(self):
        """A malformed cache file must NOT crash signal_utility — silent
        fallback to compute mirrors the BUG-178 silent-failure rule."""
        acc_mod.SIGNAL_UTILITY_CACHE_FILE.write_text("{ not json")
        with patch.object(
            acc_mod, "_compute_signal_utility", wraps=acc_mod._compute_signal_utility
        ) as spy:
            result = signal_utility("1d")
        assert spy.call_count == 1
        # Result is a real utility dict — function did not raise.
        assert isinstance(result, dict)

    def test_l2_horizons_are_independent(self):
        """Writing horizon X must not invalidate horizon Y on disk —
        load-merge-write semantics matching regime_accuracy_cache."""
        signal_utility("1d")
        signal_utility("3h")
        from portfolio.file_utils import load_json
        on_disk = load_json(acc_mod.SIGNAL_UTILITY_CACHE_FILE)
        assert "1d" in on_disk
        assert "3h" in on_disk

    def test_l2_write_failure_does_not_crash(self):
        """If _atomic_write_json raises, signal_utility must still return
        the freshly-computed value (no propagating exception)."""
        with patch.object(acc_mod, "_atomic_write_json", side_effect=OSError("disk full")):
            result = signal_utility("1d")
        assert isinstance(result, dict)
        assert "1d" in _signal_utility_cache  # L1 still populated

    def test_invalidate_clears_disk_file(self):
        signal_utility("1d")
        assert acc_mod.SIGNAL_UTILITY_CACHE_FILE.exists()
        invalidate_signal_utility_cache()
        assert not acc_mod.SIGNAL_UTILITY_CACHE_FILE.exists()
        assert "1d" not in _signal_utility_cache


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
