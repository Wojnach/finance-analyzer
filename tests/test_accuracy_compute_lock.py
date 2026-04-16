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
