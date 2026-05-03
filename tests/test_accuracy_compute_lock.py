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

    def test_first_call_fires_and_warms_all_12_keys(self, monkeypatch, tmp_path):
        """Cold start: one call should populate consensus / signal /
        per_ticker for all 4 horizons = 12 cache keys."""
        self._reset_prewarm()
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
        monkeypatch.setattr(acc_mod, "ACCURACY_CACHE_FILE", tmp_path / "acc.json")

        def _boom(*_a, **_kw):
            raise RuntimeError("simulated compute failure")
        monkeypatch.setattr(acc_mod, "signal_accuracy", _boom)
        monkeypatch.setattr(acc_mod, "consensus_accuracy", _boom)
        monkeypatch.setattr(acc_mod, "per_ticker_accuracy", _boom)

        # Must not raise.
        result = acc_mod.maybe_prewarm_dashboard_accuracy()
        assert result is False  # caught the exception
