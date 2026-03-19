"""Tests for forecast circuit breaker auto-reset on success (BUG-56)."""
import time
from unittest.mock import mock_open, patch

import pytest

import portfolio.signals.forecast as forecast


@pytest.fixture(autouse=True)
def _reset_breakers():
    """Reset circuit breakers before and after each test."""
    forecast.reset_circuit_breakers()
    yield
    forecast.reset_circuit_breakers()


class TestCircuitBreakerAutoReset:
    """BUG-56: _log_health should reset breaker on success."""

    def test_kronos_breaker_resets_on_success(self):
        forecast._trip_kronos()
        assert forecast._kronos_circuit_open()

        with patch("builtins.open", mock_open()):
            forecast._log_health("kronos", "BTC-USD", success=True, duration_ms=100)

        assert not forecast._kronos_circuit_open()
        assert forecast._kronos_tripped_until == 0.0

    def test_chronos_breaker_resets_on_success(self):
        forecast._trip_chronos()
        assert forecast._chronos_circuit_open()

        with patch("builtins.open", mock_open()):
            forecast._log_health("chronos", "ETH-USD", success=True, duration_ms=200)

        assert not forecast._chronos_circuit_open()
        assert forecast._chronos_tripped_until == 0.0

    def test_breaker_not_reset_on_failure(self):
        forecast._trip_kronos()
        assert forecast._kronos_circuit_open()

        with patch("builtins.open", mock_open()):
            forecast._log_health("kronos", "BTC-USD", success=False, duration_ms=50, error="timeout")

        assert forecast._kronos_circuit_open()

    def test_kronos_success_does_not_reset_chronos(self):
        forecast._trip_kronos()
        forecast._trip_chronos()

        with patch("builtins.open", mock_open()):
            forecast._log_health("kronos", "BTC-USD", success=True, duration_ms=100)

        assert not forecast._kronos_circuit_open()
        assert forecast._chronos_circuit_open()  # still tripped

    def test_chronos_success_does_not_reset_kronos(self):
        forecast._trip_kronos()
        forecast._trip_chronos()

        with patch("builtins.open", mock_open()):
            forecast._log_health("chronos", "ETH-USD", success=True, duration_ms=100)

        assert forecast._kronos_circuit_open()  # still tripped
        assert not forecast._chronos_circuit_open()

    def test_success_when_not_tripped_is_noop(self):
        """Success when breaker is not tripped should not cause errors."""
        assert not forecast._kronos_circuit_open()

        with patch("builtins.open", mock_open()):
            forecast._log_health("kronos", "BTC-USD", success=True, duration_ms=80)

        assert not forecast._kronos_circuit_open()

    def test_health_file_write_failure_still_resets_breaker(self):
        """Even if the JSONL write fails, the breaker should still reset."""
        forecast._trip_kronos()
        assert forecast._kronos_circuit_open()

        with patch("builtins.open", side_effect=OSError("disk full")):
            forecast._log_health("kronos", "BTC-USD", success=True, duration_ms=100)

        # Breaker reset happens after the try/except, so it should still work
        assert not forecast._kronos_circuit_open()


class TestCircuitBreakerBasicBehavior:
    """Verify existing circuit breaker behavior is preserved."""

    def test_trip_kronos_opens_breaker(self):
        assert not forecast._kronos_circuit_open()
        forecast._trip_kronos()
        assert forecast._kronos_circuit_open()

    def test_trip_chronos_opens_breaker(self):
        assert not forecast._chronos_circuit_open()
        forecast._trip_chronos()
        assert forecast._chronos_circuit_open()

    def test_reset_circuit_breakers(self):
        forecast._trip_kronos()
        forecast._trip_chronos()
        forecast.reset_circuit_breakers()
        assert not forecast._kronos_circuit_open()
        assert not forecast._chronos_circuit_open()

    def test_breaker_ttl(self):
        """Breaker should auto-expire after TTL."""
        forecast._kronos_tripped_until = time.monotonic() - 1  # expired
        assert not forecast._kronos_circuit_open()
