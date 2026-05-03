"""BUG-178: thundering-herd protection on accuracy cache miss.

When 5 ticker threads simultaneously see an empty (or expired) accuracy cache,
each thread should NOT recompute — exactly one thread computes, the rest read
the freshly-populated cache. Prior to the fix all 5 threads racing through
load_cached_accuracy() → None → signal_accuracy() yielded 5x redundant
50,000-entry DB scans, blowing the per-ticker post-dispatch phase from 7s to
200+ seconds and tripping the BUG-178 ticker pool watchdog.
"""

import threading
import time

import portfolio.accuracy_stats as acc_mod


class _CallCounter:
    def __init__(self, real_fn, sleep_s=0.5):
        self._real = real_fn
        self._sleep = sleep_s
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        with self._lock:
            self.calls += 1
        time.sleep(self._sleep)
        return self._real(*args, **kwargs)


class TestAccuracyComputeLock:
    """Five concurrent ticker threads must trigger at most one compute."""

    def _race(self, fn, n=5, timeout=10.0):
        results = []
        exceptions = []

        def worker():
            try:
                results.append(fn())
            except Exception as e:
                exceptions.append(e)

        threads = [threading.Thread(target=worker) for _ in range(n)]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=timeout)
        elapsed = time.monotonic() - t0
        assert not exceptions, f"Worker exceptions: {exceptions}"
        return results, elapsed

    def test_get_or_compute_accuracy_serializes_on_miss(self, monkeypatch, tmp_path):
        # Force cache to live in tmp_path so we don't pollute real cache file.
        monkeypatch.setattr(acc_mod, "ACCURACY_CACHE_FILE", tmp_path / "acc.json")

        # signal_accuracy returns a non-empty dict so write_accuracy_cache fires.
        fake_result = {"rsi": {"correct": 5, "total": 10, "accuracy": 0.5, "pct": 50.0}}
        counter = _CallCounter(lambda h: fake_result, sleep_s=0.3)
        monkeypatch.setattr(acc_mod, "signal_accuracy", counter)

        results, elapsed = self._race(lambda: acc_mod.get_or_compute_accuracy("1d"))

        # All threads return the same data.
        assert len(results) == 5
        for r in results:
            assert r == fake_result

        # Only ONE compute under the lock; the other 4 read the populated cache.
        assert counter.calls == 1, f"Expected 1 compute, got {counter.calls}"

        # Wall-time must be ~0.3s + tiny overhead, NOT 5x = 1.5s. Strict bound
        # rules out the regression where all five threads serialize through
        # signal_accuracy.
        assert elapsed < 0.9, f"Expected <0.9s wall, got {elapsed:.2f}s"

    def test_get_or_compute_recent_serializes_on_miss(self, monkeypatch, tmp_path):
        monkeypatch.setattr(acc_mod, "ACCURACY_CACHE_FILE", tmp_path / "acc.json")
        fake_result = {"rsi": {"correct": 3, "total": 5, "accuracy": 0.6, "pct": 60.0}}
        counter = _CallCounter(lambda h, days=7: fake_result, sleep_s=0.3)
        monkeypatch.setattr(acc_mod, "signal_accuracy_recent", counter)

        results, elapsed = self._race(
            lambda: acc_mod.get_or_compute_recent_accuracy("1d", days=7)
        )

        assert len(results) == 5
        assert counter.calls == 1
        assert elapsed < 0.9

    def test_get_or_compute_per_ticker_serializes_on_miss(self, monkeypatch, tmp_path):
        monkeypatch.setattr(acc_mod, "ACCURACY_CACHE_FILE", tmp_path / "acc.json")
        fake_result = {"rsi": {"BTC-USD": {"correct": 2, "total": 4, "accuracy": 0.5}}}
        counter = _CallCounter(lambda h: fake_result, sleep_s=0.3)
        monkeypatch.setattr(acc_mod, "per_ticker_accuracy", counter)

        results, elapsed = self._race(
            lambda: acc_mod.get_or_compute_per_ticker_accuracy("1d")
        )

        assert len(results) == 5
        assert counter.calls == 1
        assert elapsed < 0.9

    def test_get_or_compute_consensus_serializes_on_miss(self, monkeypatch, tmp_path):
        """2026-05-03: consensus wrapper added to fix /api/accuracy timeout.
        Same thundering-herd guarantee as the others."""
        monkeypatch.setattr(acc_mod, "ACCURACY_CACHE_FILE", tmp_path / "acc.json")
        fake_result = {"correct": 7, "total": 10, "accuracy": 0.7, "pct": 70.0}
        counter = _CallCounter(lambda h: fake_result, sleep_s=0.3)
        monkeypatch.setattr(acc_mod, "consensus_accuracy", counter)

        results, elapsed = self._race(
            lambda: acc_mod.get_or_compute_consensus_accuracy("1d")
        )

        assert len(results) == 5
        for r in results:
            assert r == fake_result
        assert counter.calls == 1
        assert elapsed < 0.9

    def test_get_or_compute_consensus_returns_cache_on_hit(self, monkeypatch, tmp_path):
        """Hot path: cached result returns without calling consensus_accuracy."""
        monkeypatch.setattr(acc_mod, "ACCURACY_CACHE_FILE", tmp_path / "acc.json")
        # Pre-populate cache.
        cached_result = {"correct": 4, "total": 5, "accuracy": 0.8, "pct": 80.0}
        acc_mod.write_accuracy_cache("consensus_1d", cached_result)

        # If consensus_accuracy gets called we'd see this fake result instead.
        called = {"n": 0}
        def _shouldnt_run(*_a, **_kw):
            called["n"] += 1
            return {"correct": 0, "total": 0, "accuracy": 0.0, "pct": 0.0}
        monkeypatch.setattr(acc_mod, "consensus_accuracy", _shouldnt_run)

        result = acc_mod.get_or_compute_consensus_accuracy("1d")
        assert result == cached_result
        assert called["n"] == 0



    def test_cache_hit_does_not_acquire_compute_lock(self, monkeypatch, tmp_path):
        """Hot path: when cache is populated, the slow compute must NOT run."""
        monkeypatch.setattr(acc_mod, "ACCURACY_CACHE_FILE", tmp_path / "acc.json")
        fake_result = {"rsi": {"correct": 5, "total": 10, "accuracy": 0.5}}

        # Pre-populate cache.
        acc_mod.write_accuracy_cache("1d", fake_result)

        # If signal_accuracy fires we'd notice via the counter.
        counter = _CallCounter(lambda h: {}, sleep_s=0.5)
        monkeypatch.setattr(acc_mod, "signal_accuracy", counter)

        result = acc_mod.get_or_compute_accuracy("1d")
        assert result == fake_result
        assert counter.calls == 0, "Compute fired despite warm cache"

    def test_accuracy_by_ticker_signal_cached_serializes_on_miss(
        self, monkeypatch, tmp_path
    ):
        """The per-ticker cached helper has the same thundering-herd risk."""
        monkeypatch.setattr(
            acc_mod, "TICKER_ACCURACY_CACHE_FILE", tmp_path / "ticker_acc.json"
        )
        fake_result = {
            "BTC-USD": {"rsi": {"correct": 5, "total": 10, "accuracy": 0.5}}
        }
        counter = _CallCounter(
            lambda h, min_samples=0: fake_result, sleep_s=0.3
        )
        monkeypatch.setattr(acc_mod, "accuracy_by_ticker_signal", counter)

        results, elapsed = self._race(
            lambda: acc_mod.accuracy_by_ticker_signal_cached("1d")
        )

        assert len(results) == 5
        assert counter.calls == 1
        assert elapsed < 0.9


class TestDashboardAccuracyPrewarm:
    """2026-05-04: loop-side pre-warm for /api/accuracy's 4 horizons."""

    def _reset_prewarm(self):
        # Reset the module-level gate to "well in the past" so the next
        # call's now=any-value passes the interval check. Setting to 0.0
        # would not work for now<3600 in test (interval is 1h).
        acc_mod._last_dashboard_prewarm_ts = -10000.0
        # 2026-05-04: also bypass the lazy-load from disk, which would
        # otherwise read the real data/dashboard_prewarm_state.json and
        # pin the gate to "recently fired".
        acc_mod._dashboard_prewarm_loaded = True

    def _isolate_lock(self, monkeypatch, tmp_path):
        """Point the cross-process file-lock and persistence file at
        tmp_path so concurrent test runs (or stale lock files in
        data/) don't suppress prewarm. Added with the 2026-05-04 codex
        P2-2 fix that introduced acquire_lock_file."""
        monkeypatch.setattr(acc_mod, "_DASHBOARD_PREWARM_LOCK_FILE",
                             tmp_path / "prewarm.lock")
        monkeypatch.setattr(acc_mod, "_DASHBOARD_PREWARM_STATE_FILE",
                             tmp_path / "prewarm_state.json")

    def test_first_call_fires_and_warms_all_12_keys(self, monkeypatch, tmp_path):
        """Cold start: one call should populate consensus / signal /
        per_ticker for all 4 horizons = 12 cache keys."""
        self._reset_prewarm()
        self._isolate_lock(monkeypatch, tmp_path)
        monkeypatch.setattr(acc_mod, "ACCURACY_CACHE_FILE", tmp_path / "acc.json")

        # Stub the underlying compute fns so we don't hit the real signal log.
        monkeypatch.setattr(
            acc_mod, "signal_accuracy",
            lambda h: {"rsi": {"correct": 1, "total": 2, "accuracy": 0.5}},
        )
        monkeypatch.setattr(
            acc_mod, "per_ticker_accuracy",
            lambda h: {"BTC-USD": {"correct": 1, "total": 2, "accuracy": 0.5}},
        )
        monkeypatch.setattr(
            acc_mod, "consensus_accuracy",
            lambda h: {"correct": 5, "total": 10, "accuracy": 0.5, "pct": 50.0},
        )

        fired = acc_mod.maybe_prewarm_dashboard_accuracy()
        assert fired is True

        # All 12 cache keys present.
        cache = acc_mod.load_json(acc_mod.ACCURACY_CACHE_FILE, default={})
        for h in ("1d", "3d", "5d", "10d"):
            assert h in cache, f"signal cache key missing: {h}"
            assert f"consensus_{h}" in cache, f"consensus cache key missing: {h}"
            assert f"per_ticker_consensus_{h}" in cache, f"per-ticker key missing: {h}"

    def test_within_interval_does_not_fire(self, monkeypatch, tmp_path):
        """Self-gating: calls within 1h do nothing."""
        self._reset_prewarm()
        self._isolate_lock(monkeypatch, tmp_path)
        monkeypatch.setattr(acc_mod, "ACCURACY_CACHE_FILE", tmp_path / "acc.json")

        call_count = {"n": 0}
        def _track_consensus(h):
            call_count["n"] += 1
            return {"correct": 1, "total": 2, "accuracy": 0.5, "pct": 50.0}
        monkeypatch.setattr(acc_mod, "consensus_accuracy", _track_consensus)
        monkeypatch.setattr(acc_mod, "signal_accuracy", lambda h: {})
        monkeypatch.setattr(acc_mod, "per_ticker_accuracy", lambda h: {})

        # First call: fires, 4 consensus_accuracy invocations.
        assert acc_mod.maybe_prewarm_dashboard_accuracy(now=1000.0) is True
        first_compute = call_count["n"]
        assert first_compute == 4

        # Within 1h: gated, no new computes.
        assert acc_mod.maybe_prewarm_dashboard_accuracy(now=1000.0 + 60) is False
        assert acc_mod.maybe_prewarm_dashboard_accuracy(now=1000.0 + 3599) is False
        assert call_count["n"] == first_compute

        # After 1h: fires again — but cache is hot now, so no new computes.
        assert acc_mod.maybe_prewarm_dashboard_accuracy(now=1000.0 + 3601) is True
        assert call_count["n"] == first_compute  # still 4 — cache was warm

    def test_prewarm_is_idempotent_when_cache_warm(self, monkeypatch, tmp_path):
        """When cache file already has all 12 keys, prewarm fires (gate
        passes) but the underlying compute is never called — the
        get_or_compute_* helpers short-circuit on cache hit."""
        self._reset_prewarm()
        self._isolate_lock(monkeypatch, tmp_path)
        monkeypatch.setattr(acc_mod, "ACCURACY_CACHE_FILE", tmp_path / "acc.json")

        # Pre-populate all 12 keys.
        for h in ("1d", "3d", "5d", "10d"):
            acc_mod.write_accuracy_cache(h, {"rsi": {"correct": 1, "total": 2, "accuracy": 0.5}})
            acc_mod.write_accuracy_cache(f"consensus_{h}",
                {"correct": 5, "total": 10, "accuracy": 0.5, "pct": 50.0})
            acc_mod.write_accuracy_cache(f"per_ticker_consensus_{h}",
                {"BTC-USD": {"correct": 1, "total": 2, "accuracy": 0.5}})

        # Stubs that would FAIL the test if invoked.
        def _shouldnt_run(*_a, **_kw):
            raise AssertionError("compute should not run when cache is warm")
        monkeypatch.setattr(acc_mod, "signal_accuracy", _shouldnt_run)
        monkeypatch.setattr(acc_mod, "consensus_accuracy", _shouldnt_run)
        monkeypatch.setattr(acc_mod, "per_ticker_accuracy", _shouldnt_run)

        fired = acc_mod.maybe_prewarm_dashboard_accuracy()
        assert fired is True  # gate opened

    def test_prewarm_swallows_exceptions(self, monkeypatch, tmp_path):
        """Best-effort: any failure in the underlying compute must not
        propagate. The loop calls this every cycle and cannot crash."""
        self._reset_prewarm()
        self._isolate_lock(monkeypatch, tmp_path)
        monkeypatch.setattr(acc_mod, "ACCURACY_CACHE_FILE", tmp_path / "acc.json")

        def _boom(*_a, **_kw):
            raise RuntimeError("simulated compute failure")
        monkeypatch.setattr(acc_mod, "signal_accuracy", _boom)
        monkeypatch.setattr(acc_mod, "consensus_accuracy", _boom)
        monkeypatch.setattr(acc_mod, "per_ticker_accuracy", _boom)

        # Must not raise.
        result = acc_mod.maybe_prewarm_dashboard_accuracy()
        assert result is False  # caught the exception


class TestDashboardPrewarmPersistence:
    """2026-05-04: prewarm ts persists to disk so loop restarts don't
    re-fire the prewarm immediately. Otherwise every loop restart pays
    one extra cold-cache fanout."""

    def _reset_module_state(self):
        acc_mod._last_dashboard_prewarm_ts = 0.0
        acc_mod._dashboard_prewarm_loaded = False

    def _isolate_lock(self, monkeypatch, tmp_path):
        """Point the cross-process file-lock at tmp_path so the real
        production lock file in data/ doesn't suppress prewarm during
        concurrent test runs."""
        monkeypatch.setattr(acc_mod, "_DASHBOARD_PREWARM_LOCK_FILE",
                             tmp_path / "prewarm.lock")

    def _stub_compute(self, monkeypatch):
        monkeypatch.setattr(acc_mod, "signal_accuracy", lambda h: {"x": 1})
        monkeypatch.setattr(acc_mod, "per_ticker_accuracy", lambda h: {"x": 1})
        monkeypatch.setattr(acc_mod, "consensus_accuracy",
                             lambda h: {"correct": 1, "total": 2, "accuracy": 0.5, "pct": 50.0})

    def test_first_fire_writes_persisted_state(self, monkeypatch, tmp_path):
        """After prewarm fires, dashboard_prewarm_state.json must exist."""
        self._reset_module_state()
        self._isolate_lock(monkeypatch, tmp_path)
        monkeypatch.setattr(acc_mod, "ACCURACY_CACHE_FILE", tmp_path / "acc.json")
        state_file = tmp_path / "prewarm.json"
        monkeypatch.setattr(acc_mod, "_DASHBOARD_PREWARM_STATE_FILE", state_file)
        self._stub_compute(monkeypatch)

        assert acc_mod.maybe_prewarm_dashboard_accuracy(now=10000.0) is True
        assert state_file.exists()
        import json as _json
        persisted = _json.loads(state_file.read_text())
        assert persisted["last_prewarm_ts"] == 10000.0

    def test_persisted_state_seeds_in_memory_gate(self, monkeypatch, tmp_path):
        """A 'restart' (reset module state) reads back persisted ts and
        suppresses the prewarm if still within interval."""
        self._reset_module_state()
        self._isolate_lock(monkeypatch, tmp_path)
        monkeypatch.setattr(acc_mod, "ACCURACY_CACHE_FILE", tmp_path / "acc.json")
        state_file = tmp_path / "prewarm.json"
        monkeypatch.setattr(acc_mod, "_DASHBOARD_PREWARM_STATE_FILE", state_file)

        # Pre-existing persisted ts: prewarm fired 30 minutes ago.
        from portfolio.file_utils import atomic_write_json as _aw
        _aw(state_file, {"last_prewarm_ts": 10000.0})

        # Stubs that would fail the test if invoked.
        def _shouldnt_run(*_a, **_kw):
            raise AssertionError("compute should not run inside interval")
        monkeypatch.setattr(acc_mod, "signal_accuracy", _shouldnt_run)
        monkeypatch.setattr(acc_mod, "consensus_accuracy", _shouldnt_run)
        monkeypatch.setattr(acc_mod, "per_ticker_accuracy", _shouldnt_run)

        # 30 min later — within 1h gate, so must NOT fire.
        assert acc_mod.maybe_prewarm_dashboard_accuracy(now=10000.0 + 1800) is False

    def test_persisted_state_allows_fire_after_interval(self, monkeypatch, tmp_path):
        """A 'restart' >1h after the persisted ts should allow prewarm
        to fire (the persisted gate has expired)."""
        self._reset_module_state()
        self._isolate_lock(monkeypatch, tmp_path)
        monkeypatch.setattr(acc_mod, "ACCURACY_CACHE_FILE", tmp_path / "acc.json")
        state_file = tmp_path / "prewarm.json"
        monkeypatch.setattr(acc_mod, "_DASHBOARD_PREWARM_STATE_FILE", state_file)

        # Persisted ts from 2 hours ago.
        from portfolio.file_utils import atomic_write_json as _aw
        _aw(state_file, {"last_prewarm_ts": 10000.0})
        self._stub_compute(monkeypatch)

        assert acc_mod.maybe_prewarm_dashboard_accuracy(now=10000.0 + 7200) is True

    def test_corrupt_state_falls_back_to_zero(self, monkeypatch, tmp_path):
        """A malformed state file shouldn't pin the gate forever — fall
        back to 0 so the next call fires."""
        self._reset_module_state()
        self._isolate_lock(monkeypatch, tmp_path)
        monkeypatch.setattr(acc_mod, "ACCURACY_CACHE_FILE", tmp_path / "acc.json")
        state_file = tmp_path / "prewarm.json"
        monkeypatch.setattr(acc_mod, "_DASHBOARD_PREWARM_STATE_FILE", state_file)
        # Malformed JSON.
        state_file.write_text("not json at all", encoding="utf-8")
        self._stub_compute(monkeypatch)

        # Should treat as no persisted state, fire prewarm.
        assert acc_mod.maybe_prewarm_dashboard_accuracy(now=10000.0) is True

    def test_negative_or_invalid_ts_falls_back_to_zero(self, monkeypatch, tmp_path):
        """Hostile / corrupt content (negative, string, missing) -> 0."""
        self._reset_module_state()
        self._isolate_lock(monkeypatch, tmp_path)
        monkeypatch.setattr(acc_mod, "ACCURACY_CACHE_FILE", tmp_path / "acc.json")
        state_file = tmp_path / "prewarm.json"
        monkeypatch.setattr(acc_mod, "_DASHBOARD_PREWARM_STATE_FILE", state_file)
        from portfolio.file_utils import atomic_write_json as _aw
        _aw(state_file, {"last_prewarm_ts": -1})
        self._stub_compute(monkeypatch)

        # Negative ts treated as 0, prewarm fires.
        assert acc_mod.maybe_prewarm_dashboard_accuracy(now=10000.0) is True

    def test_disk_state_only_loaded_once_per_process(self, monkeypatch, tmp_path):
        """The persistence read is lazy — happens once per process, not
        per call. After load, gating uses the in-memory ts."""
        self._reset_module_state()
        self._isolate_lock(monkeypatch, tmp_path)
        monkeypatch.setattr(acc_mod, "ACCURACY_CACHE_FILE", tmp_path / "acc.json")
        state_file = tmp_path / "prewarm.json"
        monkeypatch.setattr(acc_mod, "_DASHBOARD_PREWARM_STATE_FILE", state_file)
        from portfolio.file_utils import atomic_write_json as _aw
        _aw(state_file, {"last_prewarm_ts": 10000.0})

        # Patch _load_prewarm_ts_from_disk to count invocations.
        load_calls = {"n": 0}
        original = acc_mod._load_prewarm_ts_from_disk
        def _counting():
            load_calls["n"] += 1
            return original()
        monkeypatch.setattr(acc_mod, "_load_prewarm_ts_from_disk", _counting)

        # 5 calls within the interval.
        for _ in range(5):
            acc_mod.maybe_prewarm_dashboard_accuracy(now=10001.0)

        # Disk read happened exactly once (the lazy-load).
        assert load_calls["n"] == 1


class TestDashboardPrewarmFileLock:
    """2026-05-04 codex P2-2: cross-process file lock around the prewarm
    gate. A second process trying to fire while the first is mid-fanout
    must skip cleanly, not duplicate the work."""

    def _reset(self):
        acc_mod._last_dashboard_prewarm_ts = -10000.0
        acc_mod._dashboard_prewarm_loaded = True

    def test_concurrent_caller_skips_when_lock_held(self, monkeypatch, tmp_path):
        """Acquire the lock externally first; the prewarm call should
        return False (treats as gated) rather than waiting or
        duplicating the fanout."""
        from portfolio.process_lock import acquire_lock_file, release_lock_file

        self._reset()
        lock_file = tmp_path / "prewarm.lock"
        monkeypatch.setattr(acc_mod, "_DASHBOARD_PREWARM_LOCK_FILE", lock_file)
        monkeypatch.setattr(acc_mod, "_DASHBOARD_PREWARM_STATE_FILE",
                             tmp_path / "prewarm_state.json")

        # Stubs that would FAIL the test if invoked.
        def _shouldnt_run(*_a, **_kw):
            raise AssertionError("compute should not run when lock held by other process")
        monkeypatch.setattr(acc_mod, "signal_accuracy", _shouldnt_run)
        monkeypatch.setattr(acc_mod, "consensus_accuracy", _shouldnt_run)
        monkeypatch.setattr(acc_mod, "per_ticker_accuracy", _shouldnt_run)

        # Simulate "other process holds the lock".
        held = acquire_lock_file(lock_file, owner="external_test")
        assert held is not None, "couldn't acquire lock for setup"

        try:
            # Our caller should see the lock held and skip.
            result = acc_mod.maybe_prewarm_dashboard_accuracy(now=10000.0)
            assert result is False
        finally:
            release_lock_file(held)

    def test_disk_re_read_does_not_clobber_negative_seed(self, monkeypatch, tmp_path):
        """The layer-2 re-read inside the file lock honors a positive
        disk ts (newer racer write), but a missing-file 0 must NOT
        overwrite a deliberately-old in-memory ts."""
        self._reset()  # seeds _last_dashboard_prewarm_ts = -10000.0
        monkeypatch.setattr(acc_mod, "_DASHBOARD_PREWARM_LOCK_FILE",
                             tmp_path / "prewarm.lock")
        # State file does NOT exist — disk_ts will be 0.
        monkeypatch.setattr(acc_mod, "_DASHBOARD_PREWARM_STATE_FILE",
                             tmp_path / "missing_state.json")
        monkeypatch.setattr(acc_mod, "ACCURACY_CACHE_FILE", tmp_path / "acc.json")

        fired = {"n": 0}
        def _fire(*_a, **_kw):
            fired["n"] += 1
            return {"correct": 1, "total": 2, "accuracy": 0.5, "pct": 50.0}
        monkeypatch.setattr(acc_mod, "signal_accuracy", _fire)
        monkeypatch.setattr(acc_mod, "consensus_accuracy", _fire)
        monkeypatch.setattr(acc_mod, "per_ticker_accuracy", _fire)

        # Test invokes with now=1000.0; without the >0 guard, the
        # in-memory -10000 would be replaced with 0, and 1000-0=1000
        # would fail the 3600 gate. With the guard, fires.
        result = acc_mod.maybe_prewarm_dashboard_accuracy(now=1000.0)
        assert result is True
        assert fired["n"] == 12  # 4 horizons × 3 wrappers

    def test_releases_lock_even_on_compute_exception(self, monkeypatch, tmp_path):
        """If get_or_compute_* raises, the file lock must still release
        so the next caller isn't permanently blocked."""
        from portfolio.process_lock import acquire_lock_file, release_lock_file

        self._reset()
        lock_file = tmp_path / "prewarm.lock"
        monkeypatch.setattr(acc_mod, "_DASHBOARD_PREWARM_LOCK_FILE", lock_file)
        monkeypatch.setattr(acc_mod, "_DASHBOARD_PREWARM_STATE_FILE",
                             tmp_path / "prewarm_state.json")
        monkeypatch.setattr(acc_mod, "ACCURACY_CACHE_FILE", tmp_path / "acc.json")

        def _boom(*_a, **_kw):
            raise RuntimeError("forced failure")
        monkeypatch.setattr(acc_mod, "signal_accuracy", _boom)
        monkeypatch.setattr(acc_mod, "consensus_accuracy", _boom)
        monkeypatch.setattr(acc_mod, "per_ticker_accuracy", _boom)

        # First call: compute raises but exception is swallowed.
        assert acc_mod.maybe_prewarm_dashboard_accuracy(now=1000.0) is False

        # Lock should be released — verify by acquiring it from outside.
        fh = acquire_lock_file(lock_file, owner="external_release_check")
        assert fh is not None, "lock not released after compute exception"
        release_lock_file(fh)
