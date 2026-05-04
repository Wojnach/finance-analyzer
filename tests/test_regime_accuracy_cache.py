"""L1+L2 cache for regime accuracy (added 2026-05-04).

Mirror tests for `portfolio.accuracy_stats.get_or_compute_regime_accuracy`.
The function provides the same dogpile-resistant pattern as signal_utility:
in-memory L1 → on-disk L2 → cold compute. Hot path on every ticker × horizon
in `signal_engine.generate_signal`, so a regression here drives cycle time
into the multi-minute range and trips the dashboard stale flag.

xdist safety: every test patches REGIME_ACCURACY_CACHE_FILE to tmp_path AND
clears the module-level L1 dict in setup. Without both, ordering between
worker processes leaks state.
"""

from __future__ import annotations

import threading
import time

import portfolio.accuracy_stats as acc_mod


def _isolate(monkeypatch, tmp_path):
    """Point disk cache at tmp_path and zero the in-memory L1."""
    monkeypatch.setattr(
        acc_mod, "REGIME_ACCURACY_CACHE_FILE", tmp_path / "regime.json"
    )
    with acc_mod._regime_accuracy_cache_lock:
        acc_mod._regime_accuracy_cache.clear()


class TestL1Hit:
    def test_l1_hit_skips_disk_and_compute(self, monkeypatch, tmp_path):
        _isolate(monkeypatch, tmp_path)
        cached = {"trending": {"rsi": {"total": 50, "accuracy": 0.6, "correct": 30, "pct": 60.0}}}
        with acc_mod._regime_accuracy_cache_lock:
            acc_mod._regime_accuracy_cache["1d"] = (time.time(), cached)

        def boom(*_a, **_kw):
            raise AssertionError("should not be called when L1 is hot")

        monkeypatch.setattr(acc_mod, "load_cached_regime_accuracy", boom)
        monkeypatch.setattr(acc_mod, "signal_accuracy_by_regime", boom)

        result = acc_mod.get_or_compute_regime_accuracy("1d")
        assert result is cached

    def test_l1_returns_same_ref_across_calls(self, monkeypatch, tmp_path):
        _isolate(monkeypatch, tmp_path)
        compute_calls = []

        def fake_compute(_h):
            compute_calls.append(1)
            return {"trending": {"rsi": {"total": 30, "accuracy": 0.5, "correct": 15, "pct": 50.0}}}

        monkeypatch.setattr(acc_mod, "signal_accuracy_by_regime", fake_compute)

        first = acc_mod.get_or_compute_regime_accuracy("1d")
        second = acc_mod.get_or_compute_regime_accuracy("1d")
        third = acc_mod.get_or_compute_regime_accuracy("1d")
        assert first is second is third
        assert len(compute_calls) == 1


class TestL2Hit:
    def test_l2_populates_l1_then_subsequent_hits_l1(self, monkeypatch, tmp_path):
        _isolate(monkeypatch, tmp_path)
        # Pre-seed L2 with a fresh entry.
        payload = {"trending": {"macd": {"total": 40, "accuracy": 0.55, "correct": 22, "pct": 55.0}}}
        acc_mod.write_regime_accuracy_cache("1d", payload)

        compute_calls = []

        def fake_compute(_h):
            compute_calls.append(1)
            raise AssertionError("L2 hit should skip compute")

        monkeypatch.setattr(acc_mod, "signal_accuracy_by_regime", fake_compute)

        # First call: L1 miss → L2 hit.
        first = acc_mod.get_or_compute_regime_accuracy("1d")
        assert first == payload
        assert len(compute_calls) == 0

        # Second call: L1 hot now.
        with acc_mod._regime_accuracy_cache_lock:
            assert "1d" in acc_mod._regime_accuracy_cache
        second = acc_mod.get_or_compute_regime_accuracy("1d")
        assert second is first  # same ref


class TestColdMiss:
    def test_l1_l2_miss_computes_and_writes_both(self, monkeypatch, tmp_path):
        _isolate(monkeypatch, tmp_path)
        payload = {"ranging": {"bb": {"total": 35, "accuracy": 0.48, "correct": 17, "pct": 48.0}}}
        compute_calls = []

        def fake_compute(_h):
            compute_calls.append(1)
            return payload

        monkeypatch.setattr(acc_mod, "signal_accuracy_by_regime", fake_compute)

        result = acc_mod.get_or_compute_regime_accuracy("3d")
        assert result == payload
        assert len(compute_calls) == 1

        # L1 populated.
        with acc_mod._regime_accuracy_cache_lock:
            assert "3d" in acc_mod._regime_accuracy_cache
            assert acc_mod._regime_accuracy_cache["3d"][1] is payload

        # L2 populated.
        disk = acc_mod.load_cached_regime_accuracy("3d")
        assert disk == payload

    def test_compute_failure_returns_empty_dict_without_caching(self, monkeypatch, tmp_path):
        _isolate(monkeypatch, tmp_path)

        def fail(_h):
            raise RuntimeError("simulated SQLite failure")

        monkeypatch.setattr(acc_mod, "signal_accuracy_by_regime", fail)

        result = acc_mod.get_or_compute_regime_accuracy("1d")
        assert result == {}

        # On compute exception, L1 must NOT be populated — that would mask a
        # transient failure (e.g. SQLite locked) for the next 5 minutes.
        with acc_mod._regime_accuracy_cache_lock:
            assert "1d" not in acc_mod._regime_accuracy_cache

    def test_compute_returns_empty_caches_to_avoid_recompute_thrash(self, monkeypatch, tmp_path):
        _isolate(monkeypatch, tmp_path)
        # Empty result is a legitimate output for sparse horizons (e.g. 12h with
        # no qualifying outcomes yet). Caching it prevents the per-cycle 30s
        # re-walk that this whole change is meant to kill. Mirrors signal_utility
        # behavior at portfolio/accuracy_stats.py:657-660.
        compute_calls = []

        def empty(_h):
            compute_calls.append(1)
            return {}

        monkeypatch.setattr(acc_mod, "signal_accuracy_by_regime", empty)

        first = acc_mod.get_or_compute_regime_accuracy("12h")
        second = acc_mod.get_or_compute_regime_accuracy("12h")
        third = acc_mod.get_or_compute_regime_accuracy("12h")
        assert first == {} and second == {} and third == {}
        # Single compute — second/third served from L1.
        assert len(compute_calls) == 1

        # L1 populated with empty.
        with acc_mod._regime_accuracy_cache_lock:
            assert "12h" in acc_mod._regime_accuracy_cache
            assert acc_mod._regime_accuracy_cache["12h"][1] == {}


class TestCrossHorizonMerge:
    def test_writing_3d_does_not_evict_1d(self, monkeypatch, tmp_path):
        _isolate(monkeypatch, tmp_path)
        p1 = {"trending": {"rsi": {"total": 30, "accuracy": 0.5, "correct": 15, "pct": 50.0}}}
        p3 = {"ranging":  {"rsi": {"total": 30, "accuracy": 0.6, "correct": 18, "pct": 60.0}}}

        compute_for = {"1d": p1, "3d": p3}
        monkeypatch.setattr(
            acc_mod, "signal_accuracy_by_regime", lambda h: compute_for[h]
        )

        first = acc_mod.get_or_compute_regime_accuracy("1d")
        assert first == p1
        second = acc_mod.get_or_compute_regime_accuracy("3d")
        assert second == p3

        # Both horizons readable from disk.
        assert acc_mod.load_cached_regime_accuracy("1d") == p1
        assert acc_mod.load_cached_regime_accuracy("3d") == p3


class TestTTLExpiry:
    def test_expired_l1_falls_back_to_l2(self, monkeypatch, tmp_path):
        _isolate(monkeypatch, tmp_path)
        l2_payload = {"trending": {"rsi": {"total": 40, "accuracy": 0.65, "correct": 26, "pct": 65.0}}}
        acc_mod.write_regime_accuracy_cache("1d", l2_payload)

        # Seed an EXPIRED L1 entry. ttl=300s; ts in the past by 600s.
        with acc_mod._regime_accuracy_cache_lock:
            acc_mod._regime_accuracy_cache["1d"] = (
                time.time() - 600.0,
                {"stale": "should-not-be-returned"},
            )

        # Compute MUST NOT be called — L2 is fresh.
        monkeypatch.setattr(
            acc_mod, "signal_accuracy_by_regime",
            lambda _h: (_ for _ in ()).throw(AssertionError("L2 should serve")),
        )

        result = acc_mod.get_or_compute_regime_accuracy("1d")
        assert result == l2_payload

        # L1 refreshed with the L2 payload.
        with acc_mod._regime_accuracy_cache_lock:
            cached = acc_mod._regime_accuracy_cache["1d"]
            assert cached[1] == l2_payload
            # Refreshed timestamp ≈ now.
            assert time.time() - cached[0] < 5.0


class TestInvalidate:
    def test_invalidate_one_horizon_drops_l1_only_for_that_horizon(self, monkeypatch, tmp_path):
        _isolate(monkeypatch, tmp_path)
        with acc_mod._regime_accuracy_cache_lock:
            acc_mod._regime_accuracy_cache["1d"] = (time.time(), {"a": 1})
            acc_mod._regime_accuracy_cache["3d"] = (time.time(), {"b": 2})

        acc_mod.invalidate_regime_accuracy_cache("1d")

        with acc_mod._regime_accuracy_cache_lock:
            assert "1d" not in acc_mod._regime_accuracy_cache
            assert "3d" in acc_mod._regime_accuracy_cache

    def test_invalidate_all_clears_l1(self, monkeypatch, tmp_path):
        _isolate(monkeypatch, tmp_path)
        with acc_mod._regime_accuracy_cache_lock:
            acc_mod._regime_accuracy_cache["1d"] = (time.time(), {"a": 1})
            acc_mod._regime_accuracy_cache["3d"] = (time.time(), {"b": 2})

        acc_mod.invalidate_regime_accuracy_cache()

        with acc_mod._regime_accuracy_cache_lock:
            assert acc_mod._regime_accuracy_cache == {}

    def test_invalidate_unknown_horizon_is_noop(self, monkeypatch, tmp_path):
        _isolate(monkeypatch, tmp_path)
        # Should not raise.
        acc_mod.invalidate_regime_accuracy_cache("nonexistent-h")

    def test_invalidate_all_also_clears_l2_disk(self, monkeypatch, tmp_path):
        """invalidate(None) must zero L2 too — otherwise the next get_or_compute
        call after invalidation reads stale L2 and returns it within the 1h TTL.
        That defeated the explicit-invalidation contract pre-2026-05-04."""
        _isolate(monkeypatch, tmp_path)
        # Seed L2 with fresh content.
        acc_mod.write_regime_accuracy_cache("1d", {"x": {"sig": {"total": 30}}})
        assert acc_mod.load_cached_regime_accuracy("1d") is not None

        acc_mod.invalidate_regime_accuracy_cache()

        # L2 must be empty/stale-by-time so a subsequent compute happens.
        assert acc_mod.load_cached_regime_accuracy("1d") is None

    def test_invalidate_one_horizon_expires_l2_global_time(self, monkeypatch, tmp_path):
        """Single horizon eviction must also expire L2 (file uses one shared
        time gate). Without this, surviving horizons would still be served fresh
        but the evicted one would re-populate L1 from a now-known-stale disk."""
        _isolate(monkeypatch, tmp_path)
        acc_mod.write_regime_accuracy_cache("1d", {"x": {"sig": {"total": 30}}})
        acc_mod.write_regime_accuracy_cache("3d", {"y": {"sig": {"total": 40}}})

        acc_mod.invalidate_regime_accuracy_cache("1d")

        # After invalidation, L2 time was zeroed — both horizons return None.
        assert acc_mod.load_cached_regime_accuracy("1d") is None
        assert acc_mod.load_cached_regime_accuracy("3d") is None


class TestConcurrency:
    """Five threads racing on a cold cache must NOT serialize.

    The signal_utility benchmark proved the lock-around-compute alternative
    pushed cycles to 595s on a 4-thread cold start. The dogpile pattern
    accepts at most N redundant computes (at most one per thread) but each
    thread runs in parallel — wall time stays ~one compute, not N×.
    """

    def test_five_threads_share_one_cycle(self, monkeypatch, tmp_path):
        _isolate(monkeypatch, tmp_path)
        compute_calls = []
        compute_lock = threading.Lock()
        sleep_s = 0.2

        def slow_compute(_h):
            with compute_lock:
                compute_calls.append(time.monotonic())
            time.sleep(sleep_s)
            return {"trending": {"rsi": {"total": 30, "accuracy": 0.5, "correct": 15, "pct": 50.0}}}

        monkeypatch.setattr(acc_mod, "signal_accuracy_by_regime", slow_compute)

        results: list = []

        def worker():
            results.append(acc_mod.get_or_compute_regime_accuracy("1d"))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        elapsed = time.monotonic() - t0

        assert len(results) == 5
        # Wall-time must be ~one compute (parallel dogpile), NOT 5x serial.
        # Generous bound to absorb GIL + thread spinup on slow CI.
        assert elapsed < sleep_s * 3, (
            f"Expected <{sleep_s * 3:.2f}s, got {elapsed:.2f}s "
            f"(compute_calls={len(compute_calls)})"
        )

    def test_l1_lock_is_not_held_during_compute(self, monkeypatch, tmp_path):
        """Direct invariant check: while one thread is in `signal_accuracy_by_regime`,
        the L1 lock must be acquirable from another thread.

        This catches the lock-around-compute regression that the wall-time
        bound alone can't distinguish from a single-flight pattern. If a future
        refactor moves `signal_accuracy_by_regime(...)` inside the L1
        `with` block, the second thread would block on lock acquisition for
        the duration of the compute — failing this assertion.
        """
        _isolate(monkeypatch, tmp_path)
        compute_started = threading.Event()
        release_compute = threading.Event()
        lock_seen_free = []

        def gated_compute(_h):
            compute_started.set()
            # Hold compute open until the probe thread has tried to take L1.
            release_compute.wait(timeout=2.0)
            return {"trending": {"rsi": {"total": 30, "accuracy": 0.5, "correct": 15, "pct": 50.0}}}

        monkeypatch.setattr(acc_mod, "signal_accuracy_by_regime", gated_compute)

        def call_under_test():
            acc_mod.get_or_compute_regime_accuracy("1d")

        compute_thread = threading.Thread(target=call_under_test)
        compute_thread.start()

        # Wait until compute is in flight.
        assert compute_started.wait(timeout=2.0), "compute didn't start"

        # While compute is in flight, the L1 lock MUST be acquirable.
        got = acc_mod._regime_accuracy_cache_lock.acquire(timeout=0.5)
        try:
            lock_seen_free.append(got)
        finally:
            if got:
                acc_mod._regime_accuracy_cache_lock.release()

        # Let compute finish so the thread joins cleanly.
        release_compute.set()
        compute_thread.join(timeout=3.0)

        assert lock_seen_free == [True], (
            "L1 lock was held during compute — concurrent ticker threads "
            "would serialize through the slow path."
        )
