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
    _FORECAST_MODELS_DISABLED,
)


@pytest.fixture(autouse=True)
def _reset_breakers():
    """Reset circuit breakers, enable Kronos, and disable models_disabled before and after each test."""
    import portfolio.signals.forecast as mod
    orig_kronos = mod._KRONOS_ENABLED
    orig_disabled = mod._FORECAST_MODELS_DISABLED
    mod._KRONOS_ENABLED = True
    mod._FORECAST_MODELS_DISABLED = False
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()
    mod._KRONOS_ENABLED = orig_kronos
    mod._FORECAST_MODELS_DISABLED = orig_disabled


# --- Kronos disabled by default ---

class TestKronosDisabled:
    def test_kronos_disabled_returns_none(self):
        """When _KRONOS_ENABLED is False, _run_kronos returns None immediately."""
        import portfolio.signals.forecast as mod
        mod._KRONOS_ENABLED = False
        result = _run_kronos([{"close": 100}] * 50)
        assert result is None
        assert not _kronos_circuit_open()  # should NOT trip breaker


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


# --- Forecast models disabled (top-level kill switch) ---

class TestForecastModelsDisabled:
    def test_returns_hold_immediately_when_disabled(self):
        """When _FORECAST_MODELS_DISABLED is True, returns HOLD with no model work."""
        import portfolio.signals.forecast as mod
        mod._FORECAST_MODELS_DISABLED = True
        df = pd.DataFrame({"close": [100.0] * 60})
        result = compute_forecast_signal(df, context={"ticker": "BTC-USD"})
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0
        assert result["indicators"]["models_disabled"] is True
        # Sub-signals should all be HOLD (default)
        for v in result["sub_signals"].values():
            assert v == "HOLD"

    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_no_candle_fetch_when_disabled(self, mock_candles):
        """When disabled, should not even attempt to load candles."""
        import portfolio.signals.forecast as mod
        mod._FORECAST_MODELS_DISABLED = True
        df = pd.DataFrame({"close": [100.0] * 60})
        result = compute_forecast_signal(df, context={"ticker": "BTC-USD"})
        mock_candles.assert_not_called()
        assert result["indicators"]["models_disabled"] is True


class TestForecastFullPathEnabled:
    """Verify that when models are enabled, the full code path runs (candles + both models)."""

    @staticmethod
    def _bypass_cache(key, ttl, fn, *args):
        """Bypass _cached so mocks are called directly."""
        return fn(*args)

    @patch("portfolio.signals.forecast._cached")
    @patch("portfolio.signals.forecast._run_chronos")
    @patch("portfolio.signals.forecast._run_kronos")
    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_full_path_loads_candles_and_calls_models(self, mock_candles, mock_kronos, mock_chronos, mock_cached):
        """With _FORECAST_MODELS_DISABLED=False, candles are loaded and both models called."""
        mock_cached.side_effect = self._bypass_cache
        mock_candles.return_value = [{"close": float(100 + i)} for i in range(80)]
        mock_kronos.return_value = {
            "method": "kronos",
            "results": {
                "1h": {"direction": "up", "pct_move": 0.3, "confidence": 0.6},
                "24h": {"direction": "down", "pct_move": -0.5, "confidence": 0.5},
            },
        }
        mock_chronos.return_value = {
            "1h": {"action": "BUY", "pct_move": 0.4, "confidence": 0.55},
            "24h": {"action": "SELL", "pct_move": -0.6, "confidence": 0.5},
        }
        df = pd.DataFrame({"close": [100.0] * 80})
        result = compute_forecast_signal(df, context={"ticker": "BTC-USD"})

        # Candles should have been loaded
        mock_candles.assert_called_once_with("BTC-USD")
        # Both models should have been invoked
        mock_kronos.assert_called_once()
        mock_chronos.assert_called_once()
        # models_disabled should NOT be in indicators
        assert "models_disabled" not in result["indicators"]
        # Sub-signals should reflect model outputs
        assert result["sub_signals"]["kronos_1h"] == "BUY"
        assert result["sub_signals"]["kronos_24h"] == "SELL"
        assert result["sub_signals"]["chronos_1h"] == "BUY"
        assert result["sub_signals"]["chronos_24h"] == "SELL"
        # 2 BUY + 2 SELL = HOLD
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.forecast._cached")
    @patch("portfolio.signals.forecast._run_chronos")
    @patch("portfolio.signals.forecast._run_kronos")
    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_full_path_majority_buy(self, mock_candles, mock_kronos, mock_chronos, mock_cached):
        """When 3/4 sub-signals are BUY, composite action should be BUY."""
        mock_cached.side_effect = self._bypass_cache
        mock_candles.return_value = [{"close": float(100 + i)} for i in range(80)]
        mock_kronos.return_value = {
            "method": "kronos",
            "results": {
                "1h": {"direction": "up", "pct_move": 0.3, "confidence": 0.6},
                "24h": {"direction": "up", "pct_move": 0.5, "confidence": 0.5},
            },
        }
        mock_chronos.return_value = {
            "1h": {"action": "BUY", "pct_move": 0.4, "confidence": 0.55},
            "24h": {"action": "SELL", "pct_move": -0.2, "confidence": 0.3},
        }
        df = pd.DataFrame({"close": [100.0] * 80})
        result = compute_forecast_signal(df, context={"ticker": "BTC-USD"})

        assert result["action"] == "BUY"
        assert result["confidence"] <= 0.7  # capped
        assert result["sub_signals"]["kronos_1h"] == "BUY"
        assert result["sub_signals"]["kronos_24h"] == "BUY"
        assert result["sub_signals"]["chronos_1h"] == "BUY"
        assert result["sub_signals"]["chronos_24h"] == "SELL"
