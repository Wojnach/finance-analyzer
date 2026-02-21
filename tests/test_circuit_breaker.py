"""Tests for portfolio.circuit_breaker module."""

from unittest.mock import patch

import pytest

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

    def test_half_open_allows_request(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=5)
        cb.record_failure()
        assert cb.state == State.OPEN

        with patch("portfolio.circuit_breaker.time.monotonic", return_value=cb._last_failure_time + 6):
            cb.allow_request()

        assert cb.state == State.HALF_OPEN
        assert cb.allow_request() is True


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
