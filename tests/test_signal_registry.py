"""Tests for portfolio.signal_registry — plugin-style signal registration."""
import pytest
from unittest.mock import patch, MagicMock

from portfolio.signal_registry import (
    _CORE_SIGNALS,
    _ENHANCED_SIGNALS,
    register_signal,
    register_enhanced,
    get_enhanced_signals,
    get_signal_names,
    load_signal_func,
)


class TestRegisterSignal:
    """Tests for the @register_signal decorator."""

    def test_register_enhanced_via_decorator(self):
        @register_signal("test_enhanced_dec", signal_type="enhanced")
        def compute_test(df, **kw):
            return {"vote": "HOLD"}

        assert "test_enhanced_dec" in _ENHANCED_SIGNALS
        entry = _ENHANCED_SIGNALS["test_enhanced_dec"]
        assert entry["name"] == "test_enhanced_dec"
        assert entry["type"] == "enhanced"
        assert entry["func"] is compute_test
        # Cleanup
        del _ENHANCED_SIGNALS["test_enhanced_dec"]

    def test_register_core_via_decorator(self):
        @register_signal("test_core_dec", signal_type="core")
        def compute_core(df, **kw):
            return "BUY"

        assert "test_core_dec" in _CORE_SIGNALS
        entry = _CORE_SIGNALS["test_core_dec"]
        assert entry["type"] == "core"
        assert entry["func"] is compute_core
        # Cleanup
        del _CORE_SIGNALS["test_core_dec"]

    def test_decorator_returns_original_function(self):
        @register_signal("test_passthrough", signal_type="enhanced")
        def my_func():
            return 42

        assert my_func() == 42
        del _ENHANCED_SIGNALS["test_passthrough"]

    def test_requires_macro_flag(self):
        @register_signal("test_macro", signal_type="enhanced", requires_macro=True)
        def compute_macro(df, **kw):
            return {"vote": "HOLD"}

        assert _ENHANCED_SIGNALS["test_macro"]["requires_macro"] is True
        del _ENHANCED_SIGNALS["test_macro"]


class TestRegisterEnhanced:
    """Tests for programmatic registration via register_enhanced."""

    def test_register_enhanced_programmatic(self):
        register_enhanced("test_prog", "portfolio.signals.test", "compute_test")
        assert "test_prog" in _ENHANCED_SIGNALS
        entry = _ENHANCED_SIGNALS["test_prog"]
        assert entry["module_path"] == "portfolio.signals.test"
        assert entry["func_name"] == "compute_test"
        assert entry["func"] is None  # lazy-loaded
        del _ENHANCED_SIGNALS["test_prog"]

    def test_requires_macro_default_false(self):
        register_enhanced("test_nomacro", "mod.path", "func")
        assert _ENHANCED_SIGNALS["test_nomacro"]["requires_macro"] is False
        del _ENHANCED_SIGNALS["test_nomacro"]


class TestGetEnhancedSignals:
    """Tests for get_enhanced_signals."""

    def test_returns_dict_copy(self):
        signals = get_enhanced_signals()
        assert isinstance(signals, dict)
        # Should be a copy, not the original
        signals["injected"] = "bad"
        assert "injected" not in _ENHANCED_SIGNALS

    def test_contains_default_signals(self):
        signals = get_enhanced_signals()
        expected = [
            "trend", "momentum", "volume_flow", "volatility_sig",
            "candlestick", "structure", "fibonacci", "smart_money",
            "oscillators", "heikin_ashi", "mean_reversion", "calendar",
            "momentum_factors", "macro_regime",
        ]
        for name in expected:
            assert name in signals, f"Missing default signal: {name}"

    def test_default_count(self):
        signals = get_enhanced_signals()
        assert len(signals) >= 14  # 14 default enhanced signals


class TestGetSignalNames:
    """Tests for get_signal_names."""

    def test_returns_list(self):
        names = get_signal_names()
        assert isinstance(names, list)

    def test_enhanced_names_present(self):
        names = get_signal_names()
        assert "trend" in names
        assert "macro_regime" in names


class TestLoadSignalFunc:
    """Tests for lazy-loading signal compute functions."""

    def test_returns_cached_func(self):
        func = lambda: None
        entry = {"name": "test", "func": func, "module_path": "x", "func_name": "y"}
        assert load_signal_func(entry) is func

    def test_lazy_loads_module(self):
        mock_mod = MagicMock()
        mock_func = MagicMock()
        mock_mod.compute_test = mock_func

        entry = {
            "name": "test_lazy",
            "func": None,
            "module_path": "portfolio.signals.test_fake",
            "func_name": "compute_test",
        }
        with patch("importlib.import_module", return_value=mock_mod) as mock_import:
            result = load_signal_func(entry)
            mock_import.assert_called_once_with("portfolio.signals.test_fake")
            assert result is mock_func
            assert entry["func"] is mock_func  # cached

    def test_returns_none_on_import_error(self):
        entry = {
            "name": "bad_signal",
            "func": None,
            "module_path": "nonexistent.module",
            "func_name": "bad_func",
        }
        result = load_signal_func(entry)
        assert result is None

    def test_returns_none_on_missing_attr(self):
        mock_mod = MagicMock(spec=[])
        entry = {
            "name": "no_attr",
            "func": None,
            "module_path": "portfolio.signals.real",
            "func_name": "missing_function",
        }
        with patch("importlib.import_module", return_value=mock_mod):
            result = load_signal_func(entry)
            assert result is None


class TestDefaultRegistration:
    """Tests for _register_defaults running at import time."""

    def test_macro_regime_requires_macro(self):
        assert _ENHANCED_SIGNALS["macro_regime"]["requires_macro"] is True

    def test_non_macro_signals_dont_require_macro(self):
        for name in ["trend", "momentum", "volume_flow"]:
            assert _ENHANCED_SIGNALS[name]["requires_macro"] is False

    def test_all_defaults_have_module_path(self):
        for name, entry in _ENHANCED_SIGNALS.items():
            assert entry["module_path"].startswith("portfolio.signals."), \
                f"{name} has unexpected module_path: {entry['module_path']}"
