"""Tests for _validate_signal_result() in portfolio/signal_engine.py.

Covers BUG-23 (None/NaN enter consensus), ARCH-10 (centralized validation),
and ARCH-11 (max_confidence enforcement from signal registry).
"""

import math
import numpy as np
import pytest

from portfolio.signal_engine import _validate_signal_result


class TestValidAction:
    """Action string normalization and validation."""

    def test_valid_buy(self):
        r = _validate_signal_result({"action": "BUY", "confidence": 0.8})
        assert r["action"] == "BUY"

    def test_valid_sell(self):
        r = _validate_signal_result({"action": "SELL", "confidence": 0.5})
        assert r["action"] == "SELL"

    def test_valid_hold(self):
        r = _validate_signal_result({"action": "HOLD", "confidence": 0.0})
        assert r["action"] == "HOLD"

    def test_none_action_defaults_to_hold(self):
        """BUG-23: {\"action\": None} must not corrupt consensus."""
        r = _validate_signal_result({"action": None, "confidence": 0.5})
        assert r["action"] == "HOLD"

    def test_invalid_string_action(self):
        r = _validate_signal_result({"action": "STRONG_BUY", "confidence": 0.5})
        assert r["action"] == "HOLD"

    def test_numeric_action(self):
        r = _validate_signal_result({"action": 1, "confidence": 0.5})
        assert r["action"] == "HOLD"

    def test_lowercase_action_not_accepted(self):
        """Only uppercase BUY/SELL/HOLD are valid."""
        r = _validate_signal_result({"action": "buy", "confidence": 0.5})
        assert r["action"] == "HOLD"

    def test_missing_action_key(self):
        r = _validate_signal_result({"confidence": 0.5})
        assert r["action"] == "HOLD"


class TestConfidenceValidation:
    """Confidence clamping, NaN handling, type coercion."""

    def test_normal_confidence(self):
        r = _validate_signal_result({"action": "BUY", "confidence": 0.75})
        assert r["confidence"] == 0.75

    def test_nan_confidence_defaults_to_zero(self):
        """BUG-23: NaN confidence must not poison weighted consensus."""
        r = _validate_signal_result({"action": "BUY", "confidence": float("nan")})
        assert r["confidence"] == 0.0

    def test_inf_confidence_defaults_to_zero(self):
        r = _validate_signal_result({"action": "BUY", "confidence": float("inf")})
        assert r["confidence"] == 0.0

    def test_neg_inf_confidence_defaults_to_zero(self):
        r = _validate_signal_result({"action": "BUY", "confidence": float("-inf")})
        assert r["confidence"] == 0.0

    def test_numpy_nan_confidence(self):
        r = _validate_signal_result({"action": "BUY", "confidence": np.nan})
        assert r["confidence"] == 0.0

    def test_negative_confidence_clamped_to_zero(self):
        r = _validate_signal_result({"action": "BUY", "confidence": -0.5})
        assert r["confidence"] == 0.0

    def test_confidence_above_one_clamped(self):
        r = _validate_signal_result({"action": "BUY", "confidence": 1.5})
        assert r["confidence"] == 1.0

    def test_none_confidence(self):
        r = _validate_signal_result({"action": "BUY", "confidence": None})
        assert r["confidence"] == 0.0

    def test_string_confidence(self):
        r = _validate_signal_result({"action": "BUY", "confidence": "high"})
        assert r["confidence"] == 0.0

    def test_string_numeric_confidence_coerced(self):
        r = _validate_signal_result({"action": "BUY", "confidence": "0.6"})
        assert r["confidence"] == pytest.approx(0.6)

    def test_missing_confidence_key(self):
        r = _validate_signal_result({"action": "BUY"})
        assert r["confidence"] == 0.0

    def test_integer_confidence_coerced(self):
        r = _validate_signal_result({"action": "BUY", "confidence": 1})
        assert r["confidence"] == 1.0


class TestMaxConfidence:
    """ARCH-11: max_confidence parameter from signal registry."""

    def test_confidence_capped_at_max(self):
        r = _validate_signal_result(
            {"action": "BUY", "confidence": 0.9}, max_confidence=0.7
        )
        assert r["confidence"] == 0.7

    def test_confidence_below_max_unchanged(self):
        r = _validate_signal_result(
            {"action": "BUY", "confidence": 0.5}, max_confidence=0.7
        )
        assert r["confidence"] == 0.5

    def test_confidence_equal_to_max(self):
        r = _validate_signal_result(
            {"action": "BUY", "confidence": 0.7}, max_confidence=0.7
        )
        assert r["confidence"] == 0.7

    def test_default_max_is_one(self):
        r = _validate_signal_result({"action": "BUY", "confidence": 0.95})
        assert r["confidence"] == 0.95


class TestSubSignals:
    """sub_signals normalization."""

    def test_valid_sub_signals_preserved(self):
        subs = {"rsi_divergence": "BUY", "stochastic": "SELL"}
        r = _validate_signal_result(
            {"action": "BUY", "confidence": 0.5, "sub_signals": subs}
        )
        assert r["sub_signals"] == subs

    def test_none_sub_signals_default_to_empty_dict(self):
        r = _validate_signal_result(
            {"action": "BUY", "confidence": 0.5, "sub_signals": None}
        )
        assert r["sub_signals"] == {}

    def test_list_sub_signals_default_to_empty_dict(self):
        r = _validate_signal_result(
            {"action": "BUY", "confidence": 0.5, "sub_signals": ["BUY"]}
        )
        assert r["sub_signals"] == {}

    def test_missing_sub_signals_default_to_empty_dict(self):
        r = _validate_signal_result({"action": "BUY", "confidence": 0.5})
        assert r["sub_signals"] == {}


class TestIndicators:
    """indicators field passthrough."""

    def test_indicators_preserved(self):
        ind = {"rsi": 45.0, "macd": 1.2}
        r = _validate_signal_result(
            {"action": "BUY", "confidence": 0.5, "indicators": ind}
        )
        assert r["indicators"] == ind

    def test_none_indicators_default_to_empty_dict(self):
        r = _validate_signal_result(
            {"action": "BUY", "confidence": 0.5, "indicators": None}
        )
        assert r["indicators"] == {}

    def test_missing_indicators_default_to_empty_dict(self):
        r = _validate_signal_result({"action": "BUY", "confidence": 0.5})
        assert r["indicators"] == {}


class TestEdgeCases:
    """None input, empty dict, garbage."""

    def test_none_input(self):
        r = _validate_signal_result(None)
        assert r == {"action": "HOLD", "confidence": 0.0, "sub_signals": {}}

    def test_empty_dict(self):
        r = _validate_signal_result({})
        assert r["action"] == "HOLD"
        assert r["confidence"] == 0.0

    def test_non_dict_input(self):
        r = _validate_signal_result("BUY")
        assert r == {"action": "HOLD", "confidence": 0.0, "sub_signals": {}}

    def test_false_input(self):
        r = _validate_signal_result(False)
        assert r == {"action": "HOLD", "confidence": 0.0, "sub_signals": {}}

    def test_zero_input(self):
        r = _validate_signal_result(0)
        assert r == {"action": "HOLD", "confidence": 0.0, "sub_signals": {}}

    def test_sig_name_logged_on_invalid_action(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="portfolio.signal_engine"):
            _validate_signal_result({"action": "GARBAGE"}, sig_name="test_sig")
        assert "test_sig" in caplog.text
        assert "invalid action" in caplog.text

    def test_sig_name_logged_on_nan_confidence(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="portfolio.signal_engine"):
            _validate_signal_result(
                {"action": "BUY", "confidence": float("nan")}, sig_name="test_sig"
            )
        assert "test_sig" in caplog.text
        assert "non-finite" in caplog.text

    def test_no_warning_without_sig_name(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="portfolio.signal_engine"):
            _validate_signal_result({"action": "GARBAGE"})
        # Should not log when sig_name is None
        assert "invalid action" not in caplog.text


class TestRegistryMaxConfidenceIntegration:
    """Verify signal_registry entries have correct max_confidence values."""

    def test_capped_signals_have_0_7(self):
        from portfolio.signal_registry import get_enhanced_signals
        enhanced = get_enhanced_signals()
        capped = ["news_event", "econ_calendar", "forecast",
                  "claude_fundamental", "futures_flow"]
        for name in capped:
            assert name in enhanced, f"{name} not in registry"
            assert enhanced[name]["max_confidence"] == 0.7, \
                f"{name} should have max_confidence=0.7"

    def test_uncapped_signals_have_1_0(self):
        from portfolio.signal_registry import get_enhanced_signals
        enhanced = get_enhanced_signals()
        uncapped = ["trend", "momentum", "volume_flow", "volatility_sig",
                    "candlestick", "structure", "fibonacci", "smart_money",
                    "oscillators", "heikin_ashi", "mean_reversion", "calendar",
                    "momentum_factors"]
        for name in uncapped:
            assert name in enhanced, f"{name} not in registry"
            assert enhanced[name]["max_confidence"] == 1.0, \
                f"{name} should have max_confidence=1.0"

    def test_macro_regime_has_1_0(self):
        from portfolio.signal_registry import get_enhanced_signals
        enhanced = get_enhanced_signals()
        assert enhanced["macro_regime"]["max_confidence"] == 1.0
