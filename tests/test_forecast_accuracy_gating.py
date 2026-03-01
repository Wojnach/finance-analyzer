"""Tests for per-ticker forecast accuracy gating.

Covers:
- _accuracy_weighted_vote: abstain/HOLD (<55%), raw (>=55%)
- _load_forecast_accuracy: caching and error handling
- Insufficient samples (<10): raw vote used
- Confidence scaling by accuracy
- Indicator metadata (forecast_accuracy, forecast_gating)
- Kronos disabled by default
- get_ticker_accuracy / get_all_ticker_accuracies in forecast_accuracy.py
"""

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from portfolio.signals.forecast import (
    _accuracy_weighted_vote,
    _health_weighted_vote,
    _load_forecast_accuracy,
    _compute_atr_pct,
    _regime_discount,
    _HOLD_THRESHOLD,
    _MIN_SAMPLES,
    _MAX_CONFIDENCE,
    _VOL_GATE_CRYPTO,
    _VOL_GATE_DEFAULT,
    _REGIME_DISCOUNT_TRENDING,
    _REGIME_DISCOUNT_HIGH_VOL,
    _REGIME_NEUTRAL,
    compute_forecast_signal,
    reset_circuit_breakers,
)
from portfolio.forecast_accuracy import (
    get_ticker_accuracy,
    get_all_ticker_accuracies,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def _ago(hours=0, days=0):
    return (datetime.now(timezone.utc) - timedelta(hours=hours, days=days)).isoformat()


def _make_predictions(ticker, accuracy_pct, n_samples, horizon="24h"):
    """Create n prediction entries with given accuracy for a ticker."""
    correct = int(n_samples * accuracy_pct)
    incorrect = n_samples - correct
    entries = []
    for i in range(correct):
        entries.append({
            "ticker": ticker,
            "ts": _ago(hours=25 + i),
            "sub_signals": {f"chronos_{horizon}": "BUY"},
            "outcome": {horizon: {"change_pct": 1.0}},  # correct BUY
        })
    for i in range(incorrect):
        entries.append({
            "ticker": ticker,
            "ts": _ago(hours=25 + correct + i),
            "sub_signals": {f"chronos_{horizon}": "BUY"},
            "outcome": {horizon: {"change_pct": -1.0}},  # incorrect BUY
        })
    return entries


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset circuit breakers and module state before each test."""
    import portfolio.signals.forecast as mod
    orig_kronos = mod._KRONOS_ENABLED
    orig_disabled = mod._FORECAST_MODELS_DISABLED
    mod._FORECAST_MODELS_DISABLED = False
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()
    mod._KRONOS_ENABLED = orig_kronos
    mod._FORECAST_MODELS_DISABLED = orig_disabled


# ---------------------------------------------------------------------------
# _accuracy_weighted_vote — core gating logic
# ---------------------------------------------------------------------------

class TestAccuracyWeightedVote:
    """Test the accuracy-weighted voting function."""

    def _sub(self, c1="BUY", c24="BUY"):
        return {
            "kronos_1h": "HOLD", "kronos_24h": "HOLD",
            "chronos_1h": c1, "chronos_24h": c24,
        }

    def test_insufficient_samples_uses_raw(self):
        """When samples < min_samples, use raw health-weighted vote."""
        acc_data = {"BTC-USD": {"accuracy": 0.20, "samples": 5}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                self._sub("BUY", "BUY"),
                kronos_ok=False, chronos_ok=True,
                ticker="BTC-USD",
            )
        assert action == "BUY"
        assert info["forecast_gating"] == "insufficient_data"
        assert info["forecast_inverted"] is False

    def test_no_accuracy_data_uses_raw(self):
        """When no accuracy data exists, use raw vote."""
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value={}):
            action, conf, info = _accuracy_weighted_vote(
                self._sub("SELL", "SELL"),
                kronos_ok=False, chronos_ok=True,
                ticker="XAG-USD",
            )
        assert action == "SELL"
        assert info["forecast_gating"] == "insufficient_data"

    def test_high_accuracy_uses_raw_scaled(self):
        """When accuracy > hold_threshold, use raw vote scaled by accuracy."""
        acc_data = {"AMZN": {"accuracy": 0.90, "samples": 20}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                self._sub("BUY", "BUY"),
                kronos_ok=False, chronos_ok=True,
                ticker="AMZN",
            )
        assert action == "BUY"
        assert info["forecast_gating"] == "raw"
        assert info["forecast_inverted"] is False
        assert info["forecast_accuracy"] == 0.90
        # Confidence should be scaled by accuracy
        assert conf > 0
        assert conf <= _MAX_CONFIDENCE

    def test_coinflip_accuracy_forces_hold(self):
        """When accuracy below hold threshold, force HOLD."""
        acc_data = {"BTC-USD": {"accuracy": 0.48, "samples": 50}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                self._sub("BUY", "BUY"),
                kronos_ok=False, chronos_ok=True,
                ticker="BTC-USD",
            )
        assert action == "HOLD"
        assert conf == 0.0
        assert info["forecast_gating"] == "held"
        assert info["forecast_inverted"] is False

    def test_low_accuracy_abstains(self):
        """When accuracy very low, abstain (HOLD) — don't invert."""
        acc_data = {"SOUN": {"accuracy": 0.10, "samples": 20}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                self._sub("BUY", "BUY"),
                kronos_ok=False, chronos_ok=True,
                ticker="SOUN",
            )
        # Bad accuracy → abstain, not invert
        assert action == "HOLD"
        assert conf == 0.0
        assert info["forecast_gating"] == "held"
        assert info["forecast_inverted"] is False
        assert info["forecast_accuracy"] == 0.10

    def test_very_low_accuracy_also_abstains(self):
        """Even 0% accuracy should abstain, not invert."""
        acc_data = {"GRRR": {"accuracy": 0.0, "samples": 20}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                self._sub("SELL", "SELL"),
                kronos_ok=False, chronos_ok=True,
                ticker="GRRR",
            )
        assert action == "HOLD"
        assert conf == 0.0
        assert info["forecast_gating"] == "held"

    def test_5pct_accuracy_abstains(self):
        """5% accuracy should abstain."""
        acc_data = {"TEM": {"accuracy": 0.05, "samples": 30}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                self._sub("SELL", "SELL"),
                kronos_ok=False, chronos_ok=True,
                ticker="TEM",
            )
        assert action == "HOLD"
        assert info["forecast_gating"] == "held"

    def test_40pct_accuracy_still_abstains(self):
        """40% accuracy is still below 55% threshold — abstain."""
        acc_data = {"BAD": {"accuracy": 0.40, "samples": 25}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                self._sub("BUY", "BUY"),
                kronos_ok=False, chronos_ok=True,
                ticker="BAD",
            )
        assert action == "HOLD"
        assert info["forecast_gating"] == "held"

    def test_no_ticker_uses_raw(self):
        """When no ticker is provided, skip accuracy gating."""
        action, conf, info = _accuracy_weighted_vote(
            self._sub("BUY", "BUY"),
            kronos_ok=False, chronos_ok=True,
            ticker="",
        )
        assert action == "BUY"
        assert info["forecast_gating"] == "raw"

    def test_config_overrides_thresholds(self):
        """Config thresholds should override defaults."""
        # With default thresholds, 0.48 is below 0.55 → held
        # With custom threshold 0.45, 0.48 is above → raw
        acc_data = {"X": {"accuracy": 0.48, "samples": 20}}
        cfg = {"hold_threshold": 0.45}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                self._sub("BUY", "BUY"),
                kronos_ok=False, chronos_ok=True,
                ticker="X",
                config_forecast=cfg,
            )
        # 0.48 > 0.45 hold_threshold → raw
        assert info["forecast_gating"] == "raw"

    def test_config_overrides_min_samples(self):
        """Config min_samples should override default."""
        acc_data = {"X": {"accuracy": 0.10, "samples": 15}}
        # Default min_samples=10, so 15 would trigger held.
        # With custom min_samples=20, 15 is insufficient → raw (no data to judge).
        cfg = {"min_samples": 20}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                self._sub("BUY", "BUY"),
                kronos_ok=False, chronos_ok=True,
                ticker="X",
                config_forecast=cfg,
            )
        assert info["forecast_gating"] == "insufficient_data"

    def test_both_models_dead_returns_hold(self):
        """When both models are dead, return HOLD regardless of accuracy."""
        acc_data = {"X": {"accuracy": 0.95, "samples": 50}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                self._sub("BUY", "BUY"),
                kronos_ok=False, chronos_ok=False,
                ticker="X",
            )
        assert action == "HOLD"
        assert conf == 0.0

    def test_boundary_below_hold_threshold(self):
        """Accuracy below hold_threshold should be held."""
        acc_data = {"X": {"accuracy": 0.40, "samples": 20}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                self._sub("BUY", "BUY"),
                kronos_ok=False, chronos_ok=True,
                ticker="X",
            )
        assert info["forecast_gating"] == "held"

    def test_boundary_at_hold_threshold(self):
        """Accuracy exactly at hold_threshold should NOT be held (>=)."""
        acc_data = {"X": {"accuracy": 0.55, "samples": 20}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                self._sub("BUY", "BUY"),
                kronos_ok=False, chronos_ok=True,
                ticker="X",
            )
        # At exactly 0.55 — at hold_threshold, should be "raw"
        assert info["forecast_gating"] == "raw"

    def test_boundary_at_min_samples(self):
        """Samples exactly at min_samples should NOT be insufficient."""
        acc_data = {"X": {"accuracy": 0.10, "samples": 10}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                self._sub("BUY", "BUY"),
                kronos_ok=False, chronos_ok=True,
                ticker="X",
            )
        # At exactly 10 samples, should trigger gating (not insufficient)
        assert info["forecast_gating"] == "held"


# ---------------------------------------------------------------------------
# _load_forecast_accuracy — caching
# ---------------------------------------------------------------------------

class TestLoadForecastAccuracy:
    def test_returns_dict(self):
        """Should return a dict (possibly empty)."""
        mock_data = {"BTC-USD": {"accuracy": 0.55, "samples": 20}}
        with patch("portfolio.signals.forecast._cached", return_value=mock_data):
            result = _load_forecast_accuracy()
        assert isinstance(result, dict)

    def test_error_returns_empty_dict(self):
        """On import error, should return empty dict via _cached."""
        with patch("portfolio.signals.forecast._cached", return_value={}):
            result = _load_forecast_accuracy()
        assert result == {}


# ---------------------------------------------------------------------------
# Kronos disabled by default
# ---------------------------------------------------------------------------

class TestKronosDisabledDefault:
    def test_kronos_disabled_by_default(self):
        """Kronos should be disabled by default."""
        import portfolio.signals.forecast as mod
        # Note: autouse fixture restores original, check module default
        assert mod._KRONOS_ENABLED is False

    @patch("portfolio.signals.forecast._cached")
    @patch("portfolio.signals.forecast._run_chronos")
    @patch("portfolio.signals.forecast._run_kronos")
    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_kronos_not_called_when_disabled(self, mock_candles, mock_kronos,
                                              mock_chronos, mock_cached):
        """When Kronos is disabled, _run_kronos should not be invoked."""
        import portfolio.signals.forecast as mod
        mod._KRONOS_ENABLED = False

        def bypass(key, ttl, fn, *args):
            return fn(*args)
        mock_cached.side_effect = bypass

        mock_candles.return_value = [{"close": 100.0}] * 60
        mock_chronos.return_value = {
            "1h": {"action": "BUY", "pct_move": 0.3, "confidence": 0.6},
            "24h": {"action": "BUY", "pct_move": 0.5, "confidence": 0.5},
        }
        mock_kronos.return_value = None  # Would be skipped anyway

        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value={}):
            df = pd.DataFrame({"close": [100.0] * 60})
            result = compute_forecast_signal(df, context={"ticker": "BTC-USD"})

        # Kronos returns None immediately when disabled, no subprocess called
        assert result["indicators"]["kronos_ok"] is False


# ---------------------------------------------------------------------------
# compute_forecast_signal integration — gating metadata in indicators
# ---------------------------------------------------------------------------

class TestComputeForecastWithGating:
    @staticmethod
    def _bypass_cache(key, ttl, fn, *args):
        if key == "forecast_ticker_accuracy":
            return fn()
        return fn(*args)

    @patch("portfolio.signals.forecast._cached")
    @patch("portfolio.signals.forecast._run_chronos")
    @patch("portfolio.signals.forecast._run_kronos", return_value=None)
    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_gating_metadata_in_indicators(self, mock_candles, mock_kronos,
                                            mock_chronos, mock_cached):
        """Gating metadata should appear in result indicators."""
        import portfolio.signals.forecast as mod
        mod._KRONOS_ENABLED = False

        def bypass(key, ttl, fn, *args):
            return fn(*args) if args else fn()
        mock_cached.side_effect = bypass

        mock_candles.return_value = [{"close": 100.0}] * 60
        mock_chronos.return_value = {
            "1h": {"action": "BUY", "pct_move": 0.3, "confidence": 0.6},
            "24h": {"action": "BUY", "pct_move": 0.5, "confidence": 0.5},
        }

        acc_data = {"BTC-USD": {"accuracy": 0.30, "samples": 25}}
        with patch(
            "portfolio.signals.forecast._load_forecast_accuracy",
            return_value=acc_data,
        ):
            df = pd.DataFrame({"close": [100.0] * 60})
            result = compute_forecast_signal(df, context={"ticker": "BTC-USD"})

        assert result["indicators"]["forecast_accuracy"] == 0.30
        assert result["indicators"]["forecast_samples"] == 25
        assert result["indicators"]["forecast_gating"] == "held"
        assert result["indicators"]["forecast_inverted"] is False
        # Bad accuracy → abstain with HOLD
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.forecast._cached")
    @patch("portfolio.signals.forecast._run_chronos")
    @patch("portfolio.signals.forecast._run_kronos", return_value=None)
    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_high_accuracy_raw_in_indicators(self, mock_candles, mock_kronos,
                                              mock_chronos, mock_cached):
        """High-accuracy ticker should show raw gating."""
        import portfolio.signals.forecast as mod
        mod._KRONOS_ENABLED = False

        def bypass(key, ttl, fn, *args):
            return fn(*args) if args else fn()
        mock_cached.side_effect = bypass

        mock_candles.return_value = [{"close": 100.0}] * 60
        mock_chronos.return_value = {
            "1h": {"action": "BUY", "pct_move": 0.3, "confidence": 0.6},
            "24h": {"action": "BUY", "pct_move": 0.5, "confidence": 0.5},
        }

        acc_data = {"AMZN": {"accuracy": 0.90, "samples": 30}}
        with patch(
            "portfolio.signals.forecast._load_forecast_accuracy",
            return_value=acc_data,
        ):
            df = pd.DataFrame({"close": [100.0] * 60})
            result = compute_forecast_signal(df, context={"ticker": "AMZN"})

        assert result["indicators"]["forecast_gating"] == "raw"
        assert result["indicators"]["forecast_inverted"] is False
        assert result["action"] == "BUY"


# ---------------------------------------------------------------------------
# get_ticker_accuracy / get_all_ticker_accuracies
# ---------------------------------------------------------------------------

class TestGetTickerAccuracy:
    def test_returns_none_when_no_data(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        path.write_text("", encoding="utf-8")
        result = get_ticker_accuracy("BTC-USD", predictions_file=path)
        assert result is None

    def test_single_ticker_correct(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        entries = _make_predictions("XAG-USD", 0.80, 20)
        _write_jsonl(path, entries)
        result = get_ticker_accuracy("XAG-USD", predictions_file=path, days=30)
        assert result is not None
        assert result["samples"] == 20
        assert abs(result["accuracy"] - 0.80) < 0.01

    def test_filters_to_ticker(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        entries = (_make_predictions("BTC-USD", 0.50, 10) +
                   _make_predictions("XAG-USD", 0.90, 10))
        _write_jsonl(path, entries)
        result = get_ticker_accuracy("XAG-USD", predictions_file=path, days=30)
        assert result is not None
        assert result["samples"] == 10
        assert result["accuracy"] == 0.9

    def test_horizon_1h(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        entries = _make_predictions("BTC-USD", 0.60, 10, horizon="1h")
        _write_jsonl(path, entries)
        result = get_ticker_accuracy("BTC-USD", horizon="1h",
                                     predictions_file=path, days=30)
        assert result is not None
        assert result["accuracy"] == 0.6


class TestGetAllTickerAccuracies:
    def test_empty_returns_empty(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        path.write_text("", encoding="utf-8")
        result = get_all_ticker_accuracies(predictions_file=path)
        assert result == {}

    def test_multiple_tickers(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        entries = (_make_predictions("BTC-USD", 0.50, 20) +
                   _make_predictions("XAG-USD", 0.80, 20) +
                   _make_predictions("SOUN", 0.10, 20))
        _write_jsonl(path, entries)
        result = get_all_ticker_accuracies(predictions_file=path, days=30)
        assert len(result) == 3
        assert abs(result["BTC-USD"]["accuracy"] - 0.50) < 0.01
        assert abs(result["XAG-USD"]["accuracy"] - 0.80) < 0.01
        assert abs(result["SOUN"]["accuracy"] - 0.10) < 0.01

    def test_samples_correct(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        entries = _make_predictions("AMZN", 0.95, 40)
        _write_jsonl(path, entries)
        result = get_all_ticker_accuracies(predictions_file=path, days=30)
        assert result["AMZN"]["samples"] == 40

    def test_days_filter(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        # Old entries (10 days ago) + recent entries (1 day ago)
        old = _make_predictions("BTC-USD", 0.30, 10)
        # Shift timestamps to 10 days ago
        for e in old:
            e["ts"] = _ago(days=10)
        recent = _make_predictions("BTC-USD", 0.80, 10)
        _write_jsonl(path, old + recent)
        result = get_all_ticker_accuracies(days=7, predictions_file=path)
        # Only recent entries should be counted
        assert result["BTC-USD"]["samples"] == 10
        assert abs(result["BTC-USD"]["accuracy"] - 0.80) < 0.01


# ---------------------------------------------------------------------------
# _compute_atr_pct
# ---------------------------------------------------------------------------

class TestComputeAtrPct:
    def test_stable_prices_low_atr(self):
        """Stable prices should produce very low ATR%."""
        prices = [100.0] * 30
        result = _compute_atr_pct(prices)
        assert result is not None
        assert result < 0.001  # near zero

    def test_volatile_prices_high_atr(self):
        """Big swings should produce high ATR%."""
        # Alternating between 95 and 105 → ~10% swings
        prices = [95.0 + 10.0 * (i % 2) for i in range(30)]
        result = _compute_atr_pct(prices)
        assert result is not None
        assert result > 0.05  # >5%

    def test_insufficient_data_returns_none(self):
        """Too few prices should return None."""
        assert _compute_atr_pct([100.0] * 5) is None
        assert _compute_atr_pct([]) is None
        assert _compute_atr_pct(None) is None

    def test_realistic_crypto_atr(self):
        """Simulate BTC-like 3% daily ATR."""
        # Small random-walk-like moves ~1% each
        import random
        random.seed(42)
        prices = [67000.0]
        for _ in range(50):
            move = prices[-1] * random.uniform(-0.015, 0.015)
            prices.append(prices[-1] + move)
        result = _compute_atr_pct(prices)
        assert result is not None
        assert 0.001 < result < 0.05  # should be in reasonable range


# ---------------------------------------------------------------------------
# Volatility gate in _accuracy_weighted_vote
# ---------------------------------------------------------------------------

class TestVolatilityGate:
    def _sub(self, c1="BUY", c24="BUY"):
        return {
            "kronos_1h": "HOLD", "kronos_24h": "HOLD",
            "chronos_1h": c1, "chronos_24h": c24,
        }

    def test_high_vol_crypto_forces_hold(self):
        """BTC with ATR >3% should be vol-gated to HOLD."""
        acc_data = {"BTC-USD": {"accuracy": 0.90, "samples": 50}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                self._sub("BUY", "BUY"),
                kronos_ok=False, chronos_ok=True,
                ticker="BTC-USD",
                atr_pct=0.04,  # 4% — above 3% crypto threshold
            )
        assert action == "HOLD"
        assert conf == 0.0
        assert info["forecast_gating"] == "vol_gated"

    def test_low_vol_crypto_passes(self):
        """BTC with ATR <3% should NOT be vol-gated."""
        acc_data = {"BTC-USD": {"accuracy": 0.90, "samples": 50}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                self._sub("BUY", "BUY"),
                kronos_ok=False, chronos_ok=True,
                ticker="BTC-USD",
                atr_pct=0.02,  # 2% — below 3% crypto threshold
            )
        assert action == "BUY"
        assert info["forecast_gating"] == "raw"

    def test_high_vol_stock_forces_hold(self):
        """AMZN with ATR >2% should be vol-gated."""
        acc_data = {"AMZN": {"accuracy": 0.90, "samples": 50}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                self._sub("BUY", "BUY"),
                kronos_ok=False, chronos_ok=True,
                ticker="AMZN",
                atr_pct=0.025,  # 2.5% — above 2% stock threshold
            )
        assert action == "HOLD"
        assert info["forecast_gating"] == "vol_gated"

    def test_low_vol_stock_passes(self):
        """AMZN with ATR <2% should NOT be vol-gated."""
        acc_data = {"AMZN": {"accuracy": 0.90, "samples": 50}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                self._sub("BUY", "BUY"),
                kronos_ok=False, chronos_ok=True,
                ticker="AMZN",
                atr_pct=0.015,  # 1.5% — below 2% stock threshold
            )
        assert action == "BUY"
        assert info["forecast_gating"] == "raw"

    def test_none_atr_skips_vol_gate(self):
        """When atr_pct is None, vol gate is skipped."""
        acc_data = {"BTC-USD": {"accuracy": 0.90, "samples": 50}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                self._sub("BUY", "BUY"),
                kronos_ok=False, chronos_ok=True,
                ticker="BTC-USD",
                atr_pct=None,
            )
        assert action == "BUY"
        assert info["forecast_gating"] == "raw"

    def test_vol_gate_threshold_constants(self):
        """Vol gate threshold constants should be correct."""
        assert _VOL_GATE_CRYPTO == 0.03
        assert _VOL_GATE_DEFAULT == 0.02

    def test_vol_gate_config_override(self):
        """Config should override vol gate thresholds."""
        acc_data = {"BTC-USD": {"accuracy": 0.90, "samples": 50}}
        cfg = {"vol_gate_crypto": 0.05}  # raise threshold to 5%
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                self._sub("BUY", "BUY"),
                kronos_ok=False, chronos_ok=True,
                ticker="BTC-USD",
                atr_pct=0.04,  # 4% — below custom 5% threshold
                config_forecast=cfg,
            )
        assert action == "BUY"  # not gated with higher threshold
        assert info["forecast_gating"] == "raw"


# ---------------------------------------------------------------------------
# _regime_discount
# ---------------------------------------------------------------------------

class TestRegimeDiscount:
    def test_trending_up_discounted(self):
        assert _regime_discount("trending-up") == _REGIME_DISCOUNT_TRENDING

    def test_trending_down_discounted(self):
        assert _regime_discount("trending-down") == _REGIME_DISCOUNT_TRENDING

    def test_breakout_discounted(self):
        assert _regime_discount("breakout") == _REGIME_DISCOUNT_TRENDING

    def test_high_vol_discounted(self):
        assert _regime_discount("high-vol") == _REGIME_DISCOUNT_HIGH_VOL

    def test_capitulation_discounted(self):
        assert _regime_discount("capitulation") == _REGIME_DISCOUNT_HIGH_VOL

    def test_range_bound_neutral(self):
        assert _regime_discount("range-bound") == _REGIME_NEUTRAL

    def test_empty_string_neutral(self):
        assert _regime_discount("") == _REGIME_NEUTRAL

    def test_none_neutral(self):
        assert _regime_discount(None) == _REGIME_NEUTRAL

    def test_config_override_trending(self):
        result = _regime_discount("trending-up", {"regime_discount_trending": 0.3})
        assert result == 0.3

    def test_config_override_high_vol(self):
        result = _regime_discount("high-vol", {"regime_discount_high_vol": 0.4})
        assert result == 0.4

    def test_constants_correct(self):
        assert _REGIME_DISCOUNT_TRENDING == 0.5
        assert _REGIME_DISCOUNT_HIGH_VOL == 0.6
        assert _REGIME_NEUTRAL == 1.0


# ---------------------------------------------------------------------------
# Regime discount in _accuracy_weighted_vote
# ---------------------------------------------------------------------------

class TestRegimeInVote:
    def _sub(self, c1="BUY", c24="BUY"):
        return {
            "kronos_1h": "HOLD", "kronos_24h": "HOLD",
            "chronos_1h": c1, "chronos_24h": c24,
        }

    def test_trending_regime_reduces_confidence(self):
        """Trending regime should reduce confidence relative to neutral."""
        acc_data = {"AMZN": {"accuracy": 0.90, "samples": 50}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            # Without regime
            _, conf_neutral, _ = _accuracy_weighted_vote(
                self._sub("BUY", "BUY"),
                kronos_ok=False, chronos_ok=True,
                ticker="AMZN",
            )
            # With trending regime
            _, conf_trending, info = _accuracy_weighted_vote(
                self._sub("BUY", "BUY"),
                kronos_ok=False, chronos_ok=True,
                ticker="AMZN",
                regime="trending-up",
            )
        assert conf_trending < conf_neutral
        assert info.get("regime_discount") == _REGIME_DISCOUNT_TRENDING

    def test_ranging_regime_no_discount(self):
        """Range-bound regime should NOT discount."""
        acc_data = {"AMZN": {"accuracy": 0.90, "samples": 50}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            _, conf_neutral, _ = _accuracy_weighted_vote(
                self._sub("BUY", "BUY"),
                kronos_ok=False, chronos_ok=True,
                ticker="AMZN",
            )
            _, conf_ranging, _ = _accuracy_weighted_vote(
                self._sub("BUY", "BUY"),
                kronos_ok=False, chronos_ok=True,
                ticker="AMZN",
                regime="range-bound",
            )
        assert conf_neutral == conf_ranging

    def test_regime_discount_still_returns_action(self):
        """Regime discount reduces confidence but doesn't change action."""
        acc_data = {"AMZN": {"accuracy": 0.90, "samples": 50}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, _, info = _accuracy_weighted_vote(
                self._sub("BUY", "BUY"),
                kronos_ok=False, chronos_ok=True,
                ticker="AMZN",
                regime="trending-up",
            )
        assert action == "BUY"  # action preserved
        assert info.get("regime_discount") == _REGIME_DISCOUNT_TRENDING
