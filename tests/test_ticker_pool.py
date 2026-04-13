"""Tests for OR-I-001: ThreadPoolExecutor must not block on hung threads.

Verifies the non-blocking shutdown pattern used in main.py ticker processing.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class TestNonBlockingShutdown:
    """OR-I-001: pool.shutdown(wait=False, cancel_futures=True) must not block."""

    def test_shutdown_returns_immediately_on_hung_thread(self):
        """After timeout, shutdown(wait=False) should return within 1 second."""
        hang_started = threading.Event()

        def slow_work():
            hang_started.set()
            time.sleep(30)
            return "done"

        def fast_work():
            return "fast"

        pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test")
        f_slow = pool.submit(slow_work)
        f_fast = pool.submit(fast_work)

        hang_started.wait(timeout=5)
        f_fast.result(timeout=5)

        t0 = time.monotonic()
        pool.shutdown(wait=False, cancel_futures=True)
        elapsed = time.monotonic() - t0

        assert elapsed < 1.0, f"shutdown took {elapsed:.1f}s — should be near-instant"

    def test_timeout_collects_completed_futures(self):
        """Completed futures should be collected even when one thread hangs."""
        results = {}

        def work(name, delay):
            time.sleep(delay)
            return name, f"result_{name}"

        pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="test")
        futures = {
            pool.submit(work, "fast1", 0.01): "fast1",
            pool.submit(work, "fast2", 0.02): "fast2",
            pool.submit(work, "slow", 30): "slow",
        }
        try:
            for future in as_completed(futures, timeout=2):
                name, result = future.result()
                results[name] = result
        except TimeoutError:
            for f in futures:
                f.cancel()
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        assert "fast1" in results
        assert "fast2" in results
        assert "slow" not in results

    def test_cancel_futures_prevents_queued_work(self):
        """cancel_futures=True should prevent queued-but-not-started work."""
        started = []
        gate = threading.Event()

        def blocking_work(name):
            started.append(name)
            gate.wait(timeout=10)
            return name

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test")
        f1 = pool.submit(blocking_work, "first")
        f2 = pool.submit(blocking_work, "second")
        f3 = pool.submit(blocking_work, "third")

        time.sleep(0.1)
        pool.shutdown(wait=False, cancel_futures=True)
        gate.set()

        assert "first" in started
        assert f2.cancelled() or f3.cancelled(), "Queued futures should be cancelled"
