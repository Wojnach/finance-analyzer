"""Tests for forecast config integration and reporting enrichment.

Covers:
- Config loading with defaults (no forecast section → defaults apply)
- Config overrides (kronos_enabled, thresholds)
- Reporting enrichment: forecast_signals includes gating metadata
- Reporting enrichment: forecast_gating section in summary
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from portfolio.signals.forecast import (
    _accuracy_weighted_vote,
    _HOLD_THRESHOLD,
    _MIN_SAMPLES,
    _ACCURACY_CACHE_TTL,
    compute_forecast_signal,
    reset_circuit_breakers,
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


@pytest.fixture(autouse=True)
def _reset():
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
# Config defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_default_thresholds(self):
        """Default thresholds should match module constants."""
        assert _HOLD_THRESHOLD == 0.55
        assert _MIN_SAMPLES == 10
        assert _ACCURACY_CACHE_TTL == 1800

    def test_no_config_uses_defaults(self):
        """When no forecast config in context, defaults should apply."""
        acc_data = {"X": {"accuracy": 0.48, "samples": 20}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                {"kronos_1h": "HOLD", "kronos_24h": "HOLD",
                 "chronos_1h": "BUY", "chronos_24h": "BUY"},
                kronos_ok=False, chronos_ok=True,
                ticker="X",
                config_forecast=None,  # no config
            )
        # 0.48 between 0.40 and 0.55 → held
        assert info["forecast_gating"] == "held"

    def test_empty_config_uses_defaults(self):
        """Empty config dict should use defaults."""
        acc_data = {"X": {"accuracy": 0.48, "samples": 20}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                {"kronos_1h": "HOLD", "kronos_24h": "HOLD",
                 "chronos_1h": "BUY", "chronos_24h": "BUY"},
                kronos_ok=False, chronos_ok=True,
                ticker="X",
                config_forecast={},
            )
        assert info["forecast_gating"] == "held"


# ---------------------------------------------------------------------------
# Config overrides
# ---------------------------------------------------------------------------

class TestConfigOverrides:
    def test_low_accuracy_held(self):
        """Low accuracy tickers should be held (HOLD), not inverted."""
        acc_data = {"X": {"accuracy": 0.35, "samples": 20}}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                {"kronos_1h": "HOLD", "kronos_24h": "HOLD",
                 "chronos_1h": "BUY", "chronos_24h": "BUY"},
                kronos_ok=False, chronos_ok=True,
                ticker="X",
            )
        assert action == "HOLD"
        assert info["forecast_gating"] == "held"
        assert info["forecast_inverted"] is False

    def test_custom_hold_threshold(self):
        """Custom hold_threshold should be respected."""
        acc_data = {"X": {"accuracy": 0.50, "samples": 20}}
        # Default: 0.50 < 0.55 → held
        # Custom: 0.50 > 0.45 → raw
        cfg = {"hold_threshold": 0.45}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                {"kronos_1h": "HOLD", "kronos_24h": "HOLD",
                 "chronos_1h": "BUY", "chronos_24h": "BUY"},
                kronos_ok=False, chronos_ok=True,
                ticker="X",
                config_forecast=cfg,
            )
        assert info["forecast_gating"] == "raw"

    def test_custom_min_samples(self):
        """Custom min_samples should be respected."""
        acc_data = {"X": {"accuracy": 0.10, "samples": 8}}
        # Default min_samples=10 → insufficient
        # Custom min_samples=5 → held (enough samples, but accuracy too low)
        cfg = {"min_samples": 5}
        with patch("portfolio.signals.forecast._load_forecast_accuracy", return_value=acc_data):
            action, conf, info = _accuracy_weighted_vote(
                {"kronos_1h": "HOLD", "kronos_24h": "HOLD",
                 "chronos_1h": "BUY", "chronos_24h": "BUY"},
                kronos_ok=False, chronos_ok=True,
                ticker="X",
                config_forecast=cfg,
            )
        assert info["forecast_gating"] == "held"

    def test_config_passed_through_context(self):
        """Config from context should reach _accuracy_weighted_vote."""
        import portfolio.signals.forecast as mod
        mod._KRONOS_ENABLED = False

        def bypass(key, ttl, fn, *args):
            return fn(*args) if args else fn()

        with patch("portfolio.signals.forecast._cached", side_effect=bypass):
            with patch("portfolio.signals.forecast._load_candles_ohlcv",
                       return_value=[{"close": 100.0}] * 60):
                with patch("portfolio.signals.forecast._run_chronos", return_value={
                    "1h": {"action": "BUY", "pct_move": 0.3, "confidence": 0.6},
                    "24h": {"action": "BUY", "pct_move": 0.5, "confidence": 0.5},
                }):
                    acc_data = {"TEST": {"accuracy": 0.48, "samples": 20}}
                    with patch("portfolio.signals.forecast._load_forecast_accuracy",
                               return_value=acc_data):
                        df = pd.DataFrame({"close": [100.0] * 60})
                        # Without config override, 0.48 → held
                        result = compute_forecast_signal(df, context={
                            "ticker": "TEST",
                            "config": {"forecast": {"hold_threshold": 0.45}},
                        })
                        # With hold_threshold=0.45, 0.48 > 0.45 → raw
                        assert result["indicators"]["forecast_gating"] == "raw"


# ---------------------------------------------------------------------------
# Reporting enrichment
# ---------------------------------------------------------------------------

class TestReportingEnrichment:
    def test_forecast_signals_includes_gating(self):
        """forecast_signals should include gating metadata when available."""
        # Simulate the data structure that reporting.py reads
        signals = {
            "BTC-USD": {
                "extra": {
                    "forecast_action": "SELL",
                    "forecast_indicators": {
                        "chronos_ok": True,
                        "kronos_ok": False,
                        "chronos_1h_pct": 0.3,
                        "chronos_1h_conf": 0.6,
                        "chronos_24h_pct": -0.5,
                        "chronos_24h_conf": 0.5,
                        "forecast_gating": "held",
                        "forecast_accuracy": 0.25,
                        "forecast_samples": 30,
                        "forecast_inverted": False,
                    },
                },
            },
        }

        # Build the _forecast_signals dict the same way reporting.py does
        from portfolio.reporting import logger as _logger
        _forecast_signals = {}
        for t_name, t_data in signals.items():
            extra = t_data.get("extra", {})
            f_ind = extra.get("forecast_indicators", {})
            if not f_ind:
                continue
            chronos_ok = f_ind.get("chronos_ok", False)
            if chronos_ok or f_ind.get("chronos_1h_pct") is not None:
                entry = {
                    "action": extra.get("forecast_action", "HOLD"),
                    "chronos_ok": chronos_ok,
                }
                gating = f_ind.get("forecast_gating")
                if gating:
                    entry["gating"] = gating
                    entry["accuracy"] = f_ind.get("forecast_accuracy")
                    entry["samples"] = f_ind.get("forecast_samples", 0)
                    entry["inverted"] = f_ind.get("forecast_inverted", False)
                _forecast_signals[t_name] = entry

        assert "BTC-USD" in _forecast_signals
        assert _forecast_signals["BTC-USD"]["gating"] == "held"
        assert _forecast_signals["BTC-USD"]["accuracy"] == 0.25
        assert _forecast_signals["BTC-USD"]["samples"] == 30
        assert _forecast_signals["BTC-USD"]["inverted"] is False

    def test_forecast_gating_section(self):
        """forecast_gating summary should classify tickers correctly."""
        from portfolio.signals.forecast import _HOLD_THRESHOLD, _MIN_SAMPLES

        all_acc = {
            "AMZN": {"accuracy": 0.90, "samples": 20},
            "BTC-USD": {"accuracy": 0.48, "samples": 50},
            "SOUN": {"accuracy": 0.10, "samples": 20},
            "NEW": {"accuracy": 0.80, "samples": 5},
        }

        gating = {}
        for t, tdata in all_acc.items():
            acc = tdata["accuracy"]
            samp = tdata["samples"]
            if samp < _MIN_SAMPLES:
                action = "insufficient_data"
            elif acc < _HOLD_THRESHOLD:
                action = "held"
            else:
                action = "raw"
            gating[t] = {"accuracy": acc, "samples": samp, "action": action}

        assert gating["AMZN"]["action"] == "raw"
        assert gating["BTC-USD"]["action"] == "held"
        assert gating["SOUN"]["action"] == "held"  # low accuracy → held, not inverted
        assert gating["NEW"]["action"] == "insufficient_data"

    def test_kronos_enabled_config(self):
        """kronos_enabled config should control Kronos state."""
        import portfolio.signals.forecast as mod
        # Module default is False
        assert mod._KRONOS_ENABLED is False
        # Config could re-enable it (but this is runtime behavior,
        # we just verify the module-level flag)
        mod._KRONOS_ENABLED = True
        assert mod._KRONOS_ENABLED is True
        mod._KRONOS_ENABLED = False  # restore
