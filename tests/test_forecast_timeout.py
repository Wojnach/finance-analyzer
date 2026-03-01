"""Tests for Chronos timeout, prediction dedup, and accuracy metadata.

Covers:
- Chronos timeout behavior (mock slow Chronos)
- Kronos reduced timeout constant
- Prediction deduplication (same ticker within TTL not re-logged)
- Accuracy metadata in logged predictions
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from portfolio.signals.forecast import (
    _run_chronos,
    _CHRONOS_TIMEOUT,
    _KRONOS_TIMEOUT,
    _PREDICTION_DEDUP_TTL,
    compute_forecast_signal,
    reset_circuit_breakers,
    _chronos_circuit_open,
)


@pytest.fixture(autouse=True)
def _reset(tmp_path):
    import portfolio.signals.forecast as mod
    orig_kronos = mod._KRONOS_ENABLED
    orig_disabled = mod._FORECAST_MODELS_DISABLED
    orig_pred_file = mod._PREDICTIONS_FILE
    mod._KRONOS_ENABLED = False
    mod._FORECAST_MODELS_DISABLED = False
    mod._last_prediction_ts.clear()
    # Isolate predictions file per test (avoids cross-worker conflicts in parallel)
    mod._PREDICTIONS_FILE = tmp_path / "forecast_predictions.jsonl"
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()
    mod._KRONOS_ENABLED = orig_kronos
    mod._FORECAST_MODELS_DISABLED = orig_disabled
    mod._PREDICTIONS_FILE = orig_pred_file
    mod._last_prediction_ts.clear()


# ---------------------------------------------------------------------------
# Timeout constants
# ---------------------------------------------------------------------------

class TestTimeoutConstants:
    def test_chronos_timeout_default(self):
        assert _CHRONOS_TIMEOUT == 60

    def test_kronos_timeout_reduced(self):
        """Kronos timeout should be lower than the old 120s."""
        assert _KRONOS_TIMEOUT == 30
        assert _KRONOS_TIMEOUT < 120


# ---------------------------------------------------------------------------
# Chronos timeout behavior
# ---------------------------------------------------------------------------

class TestChronosTimeout:
    def test_timeout_returns_none(self):
        """When Chronos takes too long, should return None and trip breaker."""
        def slow_chronos(*args, **kwargs):
            time.sleep(5)
            return {"1h": {"action": "BUY", "pct_move": 0.3}}

        mock_mod = MagicMock()
        mock_mod.forecast_chronos = slow_chronos
        with patch.dict("sys.modules", {"portfolio.forecast_signal": mock_mod}):
            result = _run_chronos([100.0] * 50, timeout=1)
        assert result is None
        assert _chronos_circuit_open()

    def test_fast_chronos_succeeds(self):
        """When Chronos completes within timeout, should return result."""
        fast_result = {
            "1h": {"action": "BUY", "pct_move": 0.3, "confidence": 0.6},
            "24h": {"action": "SELL", "pct_move": -0.2, "confidence": 0.4},
        }
        mock_mod = MagicMock()
        mock_mod.forecast_chronos.return_value = fast_result
        with patch.dict("sys.modules", {"portfolio.forecast_signal": mock_mod}):
            result = _run_chronos([100.0] * 50, timeout=10)
        assert result is not None
        assert result == fast_result
        assert not _chronos_circuit_open()

    def test_timeout_parameter_used(self):
        """Custom timeout parameter should be used."""
        def medium_chronos(*args, **kwargs):
            time.sleep(0.5)
            return {"1h": {"action": "BUY", "pct_move": 0.3}}

        mock_mod = MagicMock()
        mock_mod.forecast_chronos = medium_chronos
        with patch.dict("sys.modules", {"portfolio.forecast_signal": mock_mod}):
            # 0.2s timeout should trigger timeout
            result = _run_chronos([100.0] * 50, timeout=0.2)
        assert result is None


# ---------------------------------------------------------------------------
# Prediction deduplication
# ---------------------------------------------------------------------------

class TestPredictionDedup:
    @patch("portfolio.signals.forecast._cached")
    @patch("portfolio.signals.forecast._run_chronos")
    @patch("portfolio.signals.forecast._run_kronos", return_value=None)
    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_dedup_prevents_double_logging(self, mock_candles, mock_kronos,
                                            mock_chronos, mock_cached):
        """Second call within dedup TTL should not append to predictions file."""
        import portfolio.signals.forecast as mod

        def bypass(key, ttl, fn, *args):
            return fn(*args) if args else fn()
        mock_cached.side_effect = bypass

        mock_candles.return_value = [{"close": 100.0}] * 60
        mock_chronos.return_value = {
            "1h": {"action": "BUY", "pct_move": 0.3, "confidence": 0.6},
            "24h": {"action": "BUY", "pct_move": 0.5, "confidence": 0.5},
        }

        pred_file = mod._PREDICTIONS_FILE
        # Clear the file
        with open(pred_file, "w") as f:
            pass

        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value={}):
            df = pd.DataFrame({"close": [100.0] * 60})

            # First call
            compute_forecast_signal(df, context={"ticker": "BTC-USD"})
            lines_after_first = pred_file.read_text(encoding="utf-8").strip().split("\n")
            # Filter out empty lines
            lines_after_first = [l for l in lines_after_first if l.strip()]

            # Second call immediately — should be deduped
            compute_forecast_signal(df, context={"ticker": "BTC-USD"})
            lines_after_second = pred_file.read_text(encoding="utf-8").strip().split("\n")
            lines_after_second = [l for l in lines_after_second if l.strip()]

        assert len(lines_after_first) == 1
        assert len(lines_after_second) == 1  # No duplicate

    @patch("portfolio.signals.forecast._cached")
    @patch("portfolio.signals.forecast._run_chronos")
    @patch("portfolio.signals.forecast._run_kronos", return_value=None)
    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_different_tickers_not_deduped(self, mock_candles, mock_kronos,
                                            mock_chronos, mock_cached):
        """Different tickers should not be deduped against each other."""
        import portfolio.signals.forecast as mod

        def bypass(key, ttl, fn, *args):
            return fn(*args) if args else fn()
        mock_cached.side_effect = bypass

        mock_candles.return_value = [{"close": 100.0}] * 60
        mock_chronos.return_value = {
            "1h": {"action": "BUY", "pct_move": 0.3, "confidence": 0.6},
            "24h": {"action": "BUY", "pct_move": 0.5, "confidence": 0.5},
        }

        pred_file = mod._PREDICTIONS_FILE
        with open(pred_file, "w") as f:
            pass

        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value={}):
            df = pd.DataFrame({"close": [100.0] * 60})
            compute_forecast_signal(df, context={"ticker": "BTC-USD"})
            compute_forecast_signal(df, context={"ticker": "ETH-USD"})

        lines = [l for l in pred_file.read_text(encoding="utf-8").strip().split("\n")
                 if l.strip()]
        assert len(lines) == 2  # Both logged

    def test_dedup_ttl_constant(self):
        """Dedup TTL should be 60 seconds."""
        assert _PREDICTION_DEDUP_TTL == 60


# ---------------------------------------------------------------------------
# Accuracy metadata in predictions
# ---------------------------------------------------------------------------

class TestAccuracyMetadata:
    @patch("portfolio.signals.forecast._cached")
    @patch("portfolio.signals.forecast._run_chronos")
    @patch("portfolio.signals.forecast._run_kronos", return_value=None)
    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_predictions_include_accuracy_metadata(self, mock_candles, mock_kronos,
                                                     mock_chronos, mock_cached):
        """Logged predictions should include per_ticker_accuracy and gating_action."""
        import portfolio.signals.forecast as mod

        def bypass(key, ttl, fn, *args):
            return fn(*args) if args else fn()
        mock_cached.side_effect = bypass

        mock_candles.return_value = [{"close": 100.0}] * 60
        mock_chronos.return_value = {
            "1h": {"action": "BUY", "pct_move": 0.3, "confidence": 0.6},
            "24h": {"action": "BUY", "pct_move": 0.5, "confidence": 0.5},
        }

        pred_file = mod._PREDICTIONS_FILE
        with open(pred_file, "w") as f:
            pass

        acc_data = {"AMZN": {"accuracy": 0.90, "samples": 25}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy",
                    return_value=acc_data):
            df = pd.DataFrame({"close": [100.0] * 60})
            compute_forecast_signal(df, context={"ticker": "AMZN"})

        lines = [l for l in pred_file.read_text(encoding="utf-8").strip().split("\n")
                 if l.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["per_ticker_accuracy"] == 0.90
        assert entry["gating_action"] == "raw"
        assert entry["ticker"] == "AMZN"

    @patch("portfolio.signals.forecast._cached")
    @patch("portfolio.signals.forecast._run_chronos")
    @patch("portfolio.signals.forecast._run_kronos", return_value=None)
    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_predictions_accuracy_null_when_no_data(self, mock_candles, mock_kronos,
                                                      mock_chronos, mock_cached):
        """When no accuracy data, per_ticker_accuracy should be None."""
        import portfolio.signals.forecast as mod

        def bypass(key, ttl, fn, *args):
            return fn(*args) if args else fn()
        mock_cached.side_effect = bypass

        mock_candles.return_value = [{"close": 100.0}] * 60
        mock_chronos.return_value = {
            "1h": {"action": "BUY", "pct_move": 0.3, "confidence": 0.6},
            "24h": {"action": "BUY", "pct_move": 0.5, "confidence": 0.5},
        }

        pred_file = mod._PREDICTIONS_FILE
        with open(pred_file, "w") as f:
            pass

        with patch("portfolio.signals.forecast._load_forecast_accuracy",
                    return_value={}):
            df = pd.DataFrame({"close": [100.0] * 60})
            compute_forecast_signal(df, context={"ticker": "NEW-TICKER"})

        lines = [l for l in pred_file.read_text(encoding="utf-8").strip().split("\n")
                 if l.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["per_ticker_accuracy"] is None
        assert entry["gating_action"] == "insufficient_data"
