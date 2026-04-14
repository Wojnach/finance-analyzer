"""Tests for portfolio.circuit_breaker module."""

from unittest.mock import patch

from portfolio.circuit_breaker import CircuitBreaker, State


class TestCircuitBreakerStates:
    """Test state transitions."""

    def test_initial_state_is_closed(self):
        cb = CircuitBreaker("test")
        assert cb.state == State.CLOSED

    def test_closed_allows_requests(self):
        cb = CircuitBreaker("test")
        assert cb.allow_request() is True

    def test_closed_to_open_after_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        assert cb.state == State.CLOSED
        cb.record_failure()
        assert cb.state == State.CLOSED
        cb.record_failure()
        assert cb.state == State.OPEN

    def test_open_blocks_requests(self):
        cb = CircuitBreaker("test", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == State.OPEN
        assert cb.allow_request() is False

    def test_open_to_half_open_after_timeout(self):
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=30)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == State.OPEN

        # Simulate time passing beyond recovery timeout
        with patch("portfolio.circuit_breaker.time.monotonic", return_value=cb._last_failure_time + 31):
            assert cb.allow_request() is True
            assert cb.state == State.HALF_OPEN

    def test_half_open_to_closed_on_success(self):
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=10)
        cb.record_failure()
        cb.record_failure()

        # Move to HALF_OPEN
        with patch("portfolio.circuit_breaker.time.monotonic", return_value=cb._last_failure_time + 11):
            cb.allow_request()

        assert cb.state == State.HALF_OPEN
        cb.record_success()
        assert cb.state == State.CLOSED
        assert cb._failure_count == 0

    def test_half_open_to_open_on_failure(self):
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=10)
        cb.record_failure()
        cb.record_failure()

        with patch("portfolio.circuit_breaker.time.monotonic", return_value=cb._last_failure_time + 11):
            cb.allow_request()

        assert cb.state == State.HALF_OPEN
        cb.record_failure()
        assert cb.state == State.OPEN

    def test_half_open_allows_one_probe(self):
        """HALF_OPEN allows exactly one probe request (BUG-93)."""
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=5)
        cb.record_failure()
        assert cb.state == State.OPEN

        with patch("portfolio.circuit_breaker.time.monotonic", return_value=cb._last_failure_time + 6):
            assert cb.allow_request() is True  # Probe — transitions OPEN → HALF_OPEN

        assert cb.state == State.HALF_OPEN
        # Second request in HALF_OPEN is blocked (probe already sent)
        assert cb.allow_request() is False


class TestRecordSuccess:
    """Test record_success behavior."""

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker("test", failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        assert cb._failure_count == 2
        cb.record_success()
        assert cb._failure_count == 0

    def test_success_in_closed_stays_closed(self):
        cb = CircuitBreaker("test")
        cb.record_success()
        assert cb.state == State.CLOSED


class TestRecordFailure:
    """Test record_failure behavior."""

    def test_failure_increments_count(self):
        cb = CircuitBreaker("test", failure_threshold=10)
        cb.record_failure()
        assert cb._failure_count == 1
        cb.record_failure()
        assert cb._failure_count == 2

    def test_failure_updates_last_failure_time(self):
        cb = CircuitBreaker("test", failure_threshold=10)
        assert cb._last_failure_time is None
        cb.record_failure()
        assert cb._last_failure_time is not None

    def test_failures_below_threshold_stay_closed(self):
        cb = CircuitBreaker("test", failure_threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == State.CLOSED

    def test_exact_threshold_opens(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.record_failure()
        assert cb.state == State.OPEN


class TestGetStatus:
    """Test get_status returns correct information."""

    def test_initial_status(self):
        cb = CircuitBreaker("my_api", failure_threshold=3, recovery_timeout=30)
        status = cb.get_status()
        assert status["name"] == "my_api"
        assert status["state"] == "CLOSED"
        assert status["failure_count"] == 0
        assert status["last_failure_time"] is None

    def test_status_after_failures(self):
        cb = CircuitBreaker("my_api", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        status = cb.get_status()
        assert status["state"] == "CLOSED"
        assert status["failure_count"] == 2
        assert status["last_failure_time"] is not None

    def test_status_when_open(self):
        cb = CircuitBreaker("my_api", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        status = cb.get_status()
        assert status["state"] == "OPEN"
        assert status["failure_count"] == 2

    def test_status_when_half_open(self):
        cb = CircuitBreaker("my_api", failure_threshold=1, recovery_timeout=5)
        cb.record_failure()
        with patch("portfolio.circuit_breaker.time.monotonic", return_value=cb._last_failure_time + 6):
            cb.allow_request()
        status = cb.get_status()
        assert status["state"] == "HALF_OPEN"


class TestTimingEdgeCases:
    """Test edge cases around timing."""

    def test_open_blocks_before_timeout(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=60)
        cb.record_failure()
        # Just before timeout
        with patch("portfolio.circuit_breaker.time.monotonic", return_value=cb._last_failure_time + 59):
            assert cb.allow_request() is False
            assert cb.state == State.OPEN

    def test_open_allows_at_exact_timeout(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=60)
        cb.record_failure()
        with patch("portfolio.circuit_breaker.time.monotonic", return_value=cb._last_failure_time + 60):
            assert cb.allow_request() is True
            assert cb.state == State.HALF_OPEN

    def test_success_after_partial_failures_resets(self):
        """Success before reaching threshold resets count."""
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb._failure_count == 0
        assert cb.state == State.CLOSED
        # Need full threshold again to open
        cb.record_failure()
        cb.record_failure()
        assert cb.state == State.CLOSED

    def test_multiple_open_close_cycles(self):
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=10)

        # Cycle 1: close -> open -> half_open -> closed
        cb.record_failure()
        cb.record_failure()
        assert cb.state == State.OPEN

        with patch("portfolio.circuit_breaker.time.monotonic", return_value=cb._last_failure_time + 11):
            cb.allow_request()
        cb.record_success()
        assert cb.state == State.CLOSED

        # Cycle 2: close -> open again
        cb.record_failure()
        cb.record_failure()
        assert cb.state == State.OPEN


class TestThreadSafety:
    """Test thread safety of CircuitBreaker."""

    def test_concurrent_failures_reach_threshold(self):
        """Many threads recording failures should eventually open the circuit."""
        import threading

        cb = CircuitBreaker("test", failure_threshold=10, recovery_timeout=60)
        barrier = threading.Barrier(20)

        def record():
            barrier.wait()
            cb.record_failure()

        threads = [threading.Thread(target=record) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cb.state == State.OPEN
        # failure_count should be exactly 20 (no lost increments)
        assert cb._failure_count == 20

    def test_concurrent_success_and_failure(self):
        """Mixed success/failure from multiple threads shouldn't corrupt state."""
        import threading

        cb = CircuitBreaker("test", failure_threshold=100, recovery_timeout=60)
        barrier = threading.Barrier(10)

        def fail():
            barrier.wait()
            for _ in range(50):
                cb.record_failure()

        def succeed():
            barrier.wait()
            for _ in range(50):
                cb.record_success()

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=fail))
            threads.append(threading.Thread(target=succeed))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # State should be valid (no corruption)
        assert cb.state in (State.CLOSED, State.OPEN)

    def test_concurrent_allow_request(self):
        """Many threads calling allow_request during OPEN should be safe."""
        import threading

        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=1)
        cb.record_failure()
        assert cb.state == State.OPEN

        results = []
        lock = threading.Lock()
        barrier = threading.Barrier(10)

        def check():
            barrier.wait()
            import time
            time.sleep(1.1)
            r = cb.allow_request()
            with lock:
                results.append(r)

        threads = [threading.Thread(target=check) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At least one should have gotten through
        assert any(results)


class TestProbeLifecycle:
    """BUG-187: Verify HALF_OPEN probe is sent exactly once via OPEN transition."""

    def test_probe_sent_on_open_to_half_open(self):
        """First allow_request after recovery timeout returns True (probe)."""
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=5)
        cb.record_failure()
        assert cb.state == State.OPEN

        with patch("portfolio.circuit_breaker.time.monotonic",
                   return_value=cb._last_failure_time + 6):
            assert cb.allow_request() is True
            assert cb.state == State.HALF_OPEN
            assert cb._half_open_probe_sent is True

    def test_second_request_blocked_in_half_open(self):
        """After probe, further requests are blocked until success/failure."""
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=5)
        cb.record_failure()

        t = cb._last_failure_time + 6
        with patch("portfolio.circuit_breaker.time.monotonic", return_value=t):
            cb.allow_request()  # probe — True
            assert cb.allow_request() is False  # blocked
            assert cb.allow_request() is False  # still blocked

    def test_probe_flag_reset_on_success(self):
        """After success in HALF_OPEN, flag is cleared for next cycle."""
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=5)
        cb.record_failure()

        with patch("portfolio.circuit_breaker.time.monotonic",
                   return_value=cb._last_failure_time + 6):
            cb.allow_request()  # probe
        cb.record_success()

        assert cb.state == State.CLOSED
        assert cb._half_open_probe_sent is False

    def test_probe_flag_reset_on_failure(self):
        """After failure in HALF_OPEN, flag is cleared (back to OPEN)."""
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=5)
        cb.record_failure()

        with patch("portfolio.circuit_breaker.time.monotonic",
                   return_value=cb._last_failure_time + 6):
            cb.allow_request()  # probe
        cb.record_failure()  # probe failed

        assert cb.state == State.OPEN
        assert cb._half_open_probe_sent is False

    def test_full_recovery_cycle(self):
        """CLOSED → OPEN → HALF_OPEN → CLOSED, verifying probe at each step."""
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=10)

        # Phase 1: CLOSED → OPEN
        cb.record_failure()
        cb.record_failure()
        assert cb.state == State.OPEN

        # Phase 2: OPEN → HALF_OPEN (probe)
        with patch("portfolio.circuit_breaker.time.monotonic",
                   return_value=cb._last_failure_time + 11):
            probe_result = cb.allow_request()
        assert probe_result is True
        assert cb.state == State.HALF_OPEN

        # Phase 3: HALF_OPEN → CLOSED (probe success)
        cb.record_success()
        assert cb.state == State.CLOSED
        assert cb._failure_count == 0
        assert cb.allow_request() is True  # normal traffic resumes
