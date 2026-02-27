"""Tests for forecast signal circuit breaker.

Verifies that after one GPU failure, subsequent calls skip instantly
instead of waiting for timeouts on every ticker.
"""

import time
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from portfolio.signals.forecast import (
    _run_kronos,
    _run_chronos,
    _kronos_circuit_open,
    _chronos_circuit_open,
    reset_circuit_breakers,
    compute_forecast_signal,
    _CIRCUIT_BREAKER_TTL,
)


@pytest.fixture(autouse=True)
def _reset_breakers():
    """Reset circuit breakers before and after each test."""
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


# --- Kronos circuit breaker ---

class TestKronosCircuitBreaker:
    def test_initially_closed(self):
        assert not _kronos_circuit_open()

    @patch("portfolio.signals.forecast.subprocess.run")
    def test_trips_on_subprocess_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="CUDA error")
        result = _run_kronos([{"close": 100}] * 50)
        assert result is None
        assert _kronos_circuit_open()

    @patch("portfolio.signals.forecast.subprocess.run")
    def test_trips_on_exception(self, mock_run):
        mock_run.side_effect = TimeoutError("timed out")
        result = _run_kronos([{"close": 100}] * 50)
        assert result is None
        assert _kronos_circuit_open()

    @patch("portfolio.signals.forecast.subprocess.run")
    def test_trips_on_empty_results(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout='{"method": "none", "results": {}}'
        )
        result = _run_kronos([{"close": 100}] * 50)
        assert result is None
        assert _kronos_circuit_open()

    @patch("portfolio.signals.forecast.subprocess.run")
    def test_skips_when_tripped(self, mock_run):
        """After tripping, subsequent calls should NOT invoke subprocess."""
        mock_run.side_effect = TimeoutError("timed out")
        _run_kronos([{"close": 100}] * 50)  # trips breaker
        assert mock_run.call_count == 1

        # Second call should be instant skip
        result = _run_kronos([{"close": 200}] * 50)
        assert result is None
        assert mock_run.call_count == 1  # NOT called again

    @patch("portfolio.signals.forecast.subprocess.run")
    def test_does_not_trip_on_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"method": "kronos", "results": {"1h": {"direction": "up", "pct_move": 0.5, "confidence": 0.6}}}'
        )
        result = _run_kronos([{"close": 100}] * 50)
        assert result is not None
        assert not _kronos_circuit_open()

    def test_resets_after_ttl(self):
        """Breaker should auto-reset after TTL expires."""
        import portfolio.signals.forecast as mod
        mod._kronos_tripped_until = time.monotonic() + 0.01  # trip for 10ms
        assert _kronos_circuit_open()
        time.sleep(0.02)
        assert not _kronos_circuit_open()


# --- Chronos circuit breaker ---

class TestChronosCircuitBreaker:
    def test_initially_closed(self):
        assert not _chronos_circuit_open()

    def test_trips_on_exception(self):
        mock_mod = MagicMock()
        mock_mod.forecast_chronos.side_effect = RuntimeError("CUDA error")
        with patch.dict("sys.modules", {"portfolio.forecast_signal": mock_mod}):
            result = _run_chronos([100.0] * 50)
        assert result is None
        assert _chronos_circuit_open()

    def test_skips_when_tripped(self):
        """After tripping, subsequent calls should NOT attempt import."""
        import portfolio.signals.forecast as mod
        mod._chronos_tripped_until = time.monotonic() + 60
        result = _run_chronos([100.0] * 50)
        assert result is None

    def test_resets_after_ttl(self):
        import portfolio.signals.forecast as mod
        mod._chronos_tripped_until = time.monotonic() + 0.01
        assert _chronos_circuit_open()
        time.sleep(0.02)
        assert not _chronos_circuit_open()


# --- Integration: compute_forecast_signal with breakers ---

class TestComputeForecastWithBreaker:
    @patch("portfolio.signals.forecast._run_chronos", return_value=None)
    @patch("portfolio.signals.forecast._run_kronos", return_value=None)
    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_returns_hold_when_both_tripped(self, mock_candles, mock_kronos, mock_chronos):
        mock_candles.return_value = [{"close": 100.0}] * 60
        df = pd.DataFrame({"close": [100.0] * 60})
        result = compute_forecast_signal(df, context={"ticker": "BTC-USD"})
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.signals.forecast._run_chronos", return_value=None)
    @patch("portfolio.signals.forecast._run_kronos", return_value=None)
    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_circuit_indicators_reported(self, mock_candles, mock_kronos, mock_chronos):
        """Circuit breaker state should be visible in indicators."""
        import portfolio.signals.forecast as mod
        mod._kronos_tripped_until = time.monotonic() + 60
        mod._chronos_tripped_until = time.monotonic() + 60

        mock_candles.return_value = [{"close": 100.0}] * 60
        df = pd.DataFrame({"close": [100.0] * 60})
        result = compute_forecast_signal(df, context={"ticker": "BTC-USD"})
        assert result["indicators"]["kronos_circuit_open"] is True
        assert result["indicators"]["chronos_circuit_open"] is True

    @patch("portfolio.signals.forecast._run_chronos", return_value=None)
    @patch("portfolio.signals.forecast._run_kronos", return_value=None)
    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_circuit_closed_indicators(self, mock_candles, mock_kronos, mock_chronos):
        mock_candles.return_value = [{"close": 100.0}] * 60
        df = pd.DataFrame({"close": [100.0] * 60})
        result = compute_forecast_signal(df, context={"ticker": "BTC-USD"})
        assert result["indicators"]["kronos_circuit_open"] is False
        assert result["indicators"]["chronos_circuit_open"] is False


class TestResetCircuitBreakers:
    def test_reset_clears_both(self):
        import portfolio.signals.forecast as mod
        mod._kronos_tripped_until = time.monotonic() + 999
        mod._chronos_tripped_until = time.monotonic() + 999
        assert _kronos_circuit_open()
        assert _chronos_circuit_open()

        reset_circuit_breakers()
        assert not _kronos_circuit_open()
        assert not _chronos_circuit_open()
