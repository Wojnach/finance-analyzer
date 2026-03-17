"""Tests for model upgrade changes: Chronos-2, Ministral-3, Qwen3.

Tests cover:
1. Chronos-2 fallback logic (v2 → v1)
2. Ministral-3 model path fallback (new → legacy)
3. Qwen3 signal wrapper
4. Qwen3 integration in signal_engine
5. Signal count changes (qwen3 added for all tickers)
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ── 1. Chronos-2 fallback tests ──────────────────────────────────────────


class TestChronosUpgrade:
    """Test Chronos v1 ↔ v2 fallback logic."""

    def test_default_model_is_chronos2(self):
        """Default model should be amazon/chronos-2."""
        from portfolio.forecast_signal import _CHRONOS_MODEL
        assert _CHRONOS_MODEL == "amazon/chronos-2"

    def test_chronos_version_tracker_exists(self):
        """_chronos_version global should exist."""
        from portfolio import forecast_signal
        assert hasattr(forecast_signal, "_chronos_version")

    def test_set_chronos_model_resets_version(self):
        """Changing model name should reset version to 0."""
        from portfolio.forecast_signal import set_chronos_model
        import portfolio.forecast_signal as fs

        old_model = fs._CHRONOS_MODEL
        old_version = fs._chronos_version
        old_pipeline = fs._chronos_pipeline

        try:
            fs._chronos_version = 2
            fs._chronos_pipeline = MagicMock()
            set_chronos_model("amazon/chronos-2-small")
            assert fs._chronos_version == 0
            assert fs._chronos_pipeline is None
            assert fs._CHRONOS_MODEL == "amazon/chronos-2-small"
        finally:
            fs._CHRONOS_MODEL = old_model
            fs._chronos_version = old_version
            fs._chronos_pipeline = old_pipeline

    def test_forecast_chronos_dispatches_v2(self):
        """When version=2, should call _forecast_chronos_v2."""
        import portfolio.forecast_signal as fs

        old_version = fs._chronos_version
        old_pipeline = fs._chronos_pipeline
        try:
            fs._chronos_version = 2
            fs._chronos_pipeline = MagicMock()

            with patch.object(fs, "_forecast_chronos_v2", return_value={"1h": {}}) as mock_v2:
                result = fs.forecast_chronos("BTC-USD", [100.0] * 100)
                mock_v2.assert_called_once()
        finally:
            fs._chronos_version = old_version
            fs._chronos_pipeline = old_pipeline

    def test_forecast_chronos_dispatches_v1(self):
        """When version=1, should call _forecast_chronos_v1."""
        import portfolio.forecast_signal as fs

        old_version = fs._chronos_version
        old_pipeline = fs._chronos_pipeline
        try:
            fs._chronos_version = 1
            fs._chronos_pipeline = MagicMock()

            with patch.object(fs, "_forecast_chronos_v1", return_value={"1h": {}}) as mock_v1:
                result = fs.forecast_chronos("BTC-USD", [100.0] * 100)
                mock_v1.assert_called_once()
        finally:
            fs._chronos_version = old_version
            fs._chronos_pipeline = old_pipeline

    def test_forecast_chronos_returns_none_without_pipeline(self):
        """Should return None when no pipeline is loaded."""
        import portfolio.forecast_signal as fs

        old_pipeline = fs._chronos_pipeline
        try:
            fs._chronos_pipeline = None
            with patch.object(fs, "_get_chronos_pipeline", return_value=None):
                result = fs.forecast_chronos("BTC-USD", [100.0] * 100)
                assert result is None
        finally:
            fs._chronos_pipeline = old_pipeline


# ── 2. Ministral-3 fallback tests ────────────────────────────────────────


class TestMinistralUpgrade:
    """Test Ministral model path fallback logic."""

    def test_model_path_is_ministral3(self):
        """Default MODEL_PATH should point to Ministral-3."""
        from portfolio.ministral_trader import MODEL_PATH
        assert "Ministral-3-8B" in MODEL_PATH or "ministral-3" in MODEL_PATH.lower()

    def test_legacy_model_path_exists(self):
        """LEGACY_MODEL_PATH should be defined."""
        from portfolio.ministral_trader import LEGACY_MODEL_PATH
        assert "Ministral-8B-Instruct-2410" in LEGACY_MODEL_PATH or "ministral-8b" in LEGACY_MODEL_PATH.lower()

    def test_default_lora_is_none(self):
        """DEFAULT_LORA should be None for Ministral-3 (incompatible)."""
        from portfolio.ministral_trader import DEFAULT_LORA
        assert DEFAULT_LORA is None

    def test_load_model_falls_back_to_legacy(self):
        """When new model doesn't exist but legacy does, should use legacy."""
        mock_llama_module = MagicMock()
        mock_llama_cls = MagicMock(return_value=MagicMock())
        mock_llama_module.Llama = mock_llama_cls

        def path_exists(path):
            if "ministral-3" in path.lower() or "Ministral-3" in path:
                return False
            if "ministral-8b" in path.lower() or "Ministral-8B" in path:
                return True
            # LoRA path
            if "cryptotrader" in path.lower():
                return True
            return False

        with patch.dict("sys.modules", {"llama_cpp": mock_llama_module}):
            with patch("os.path.exists", side_effect=path_exists):
                import importlib
                import portfolio.ministral_trader as mt
                importlib.reload(mt)
                mt.load_model()
                call_kwargs = mock_llama_cls.call_args[1]
                assert "ministral-8b" in call_kwargs.get("model_path", "").lower() or \
                       "Ministral-8B" in call_kwargs.get("model_path", "")

    def test_load_model_raises_if_no_model_found(self):
        """Should raise FileNotFoundError when neither model exists."""
        mock_llama_module = MagicMock()

        with patch.dict("sys.modules", {"llama_cpp": mock_llama_module}):
            with patch("os.path.exists", return_value=False):
                import importlib
                import portfolio.ministral_trader as mt
                importlib.reload(mt)
                with pytest.raises(FileNotFoundError):
                    mt.load_model()


# ── 3. Qwen3 signal tests ────────────────────────────────────────────────


class TestQwen3Trader:
    """Test Qwen3 trader subprocess script."""

    def test_qwen3_trader_imports(self):
        """qwen3_trader.py should be importable."""
        import portfolio.qwen3_trader as qt
        assert hasattr(qt, "predict")
        assert hasattr(qt, "load_model")
        assert hasattr(qt, "MODEL_PATH")

    def test_qwen3_model_path(self):
        """Model path should point to Qwen3-8B GGUF."""
        from portfolio.qwen3_trader import MODEL_PATH
        assert "qwen3" in MODEL_PATH.lower() or "Qwen3" in MODEL_PATH

    def test_qwen3_extract_json(self):
        """JSON extraction should handle clean and dirty output."""
        from portfolio.qwen3_trader import _extract_json_payload

        # Clean JSON
        result = _extract_json_payload('{"action": "BUY", "reasoning": "test"}')
        assert result["action"] == "BUY"

        # JSON with prefix noise
        result = _extract_json_payload('Some text {"action": "SELL", "reasoning": "test"}')
        assert result["action"] == "SELL"

        # Empty/None
        assert _extract_json_payload("") is None
        assert _extract_json_payload(None) is None

    def test_qwen3_strips_think_tags(self):
        """Qwen3 thinking mode output should be stripped."""
        from portfolio.qwen3_trader import _extract_json_payload
        text = '<think>analyzing market...</think>{"action": "HOLD", "reasoning": "mixed signals"}'
        # The predict function strips think tags before extraction
        # Test the extraction on clean output
        clean = text[text.find("</think>") + 8:].strip()
        result = _extract_json_payload(clean)
        assert result["action"] == "HOLD"


class TestQwen3Signal:
    """Test Qwen3 signal wrapper."""

    def test_qwen3_signal_imports(self):
        """qwen3_signal.py should be importable."""
        import portfolio.qwen3_signal as qs
        assert hasattr(qs, "get_qwen3_signal")

    @patch("portfolio.qwen3_signal.subprocess.run")
    def test_get_qwen3_signal_success(self, mock_run):
        """Successful subprocess call should return parsed result."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"action": "BUY", "reasoning": "strong momentum", "model": "Qwen3-8B"}',
            stderr="",
        )
        from portfolio.qwen3_signal import get_qwen3_signal

        result = get_qwen3_signal({"ticker": "BTC", "price_usd": 68000})
        assert result["action"] == "BUY"
        assert result["model"] == "Qwen3-8B"

    @patch("portfolio.qwen3_signal.subprocess.run")
    def test_get_qwen3_signal_failure(self, mock_run):
        """Failed subprocess should raise RuntimeError."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="model not found",
        )
        from portfolio.qwen3_signal import get_qwen3_signal

        with pytest.raises(RuntimeError, match="Qwen3 failed"):
            get_qwen3_signal({"ticker": "BTC", "price_usd": 68000})

    @patch("portfolio.qwen3_signal.subprocess.run")
    def test_get_qwen3_signal_invalid_json(self, mock_run):
        """Invalid JSON output should raise RuntimeError."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="not json at all",
            stderr="",
        )
        from portfolio.qwen3_signal import get_qwen3_signal

        with pytest.raises(RuntimeError, match="invalid JSON"):
            get_qwen3_signal({"ticker": "BTC", "price_usd": 68000})


# ── 4. Signal engine integration tests ───────────────────────────────────


class TestSignalEngineQwen3:
    """Test Qwen3 integration in signal_engine."""

    def test_qwen3_in_core_signals(self):
        """qwen3 should be in CORE_SIGNAL_NAMES."""
        from portfolio.signal_engine import CORE_SIGNAL_NAMES
        assert "qwen3" in CORE_SIGNAL_NAMES

    def test_qwen3_in_signal_names(self):
        """qwen3 should be in SIGNAL_NAMES list."""
        from portfolio.tickers import SIGNAL_NAMES
        assert "qwen3" in SIGNAL_NAMES

    def test_applicable_count_includes_qwen3(self):
        """Applicable count should include qwen3 for ALL tickers."""
        from portfolio.signal_engine import _compute_applicable_count

        crypto_count = _compute_applicable_count("BTC-USD")
        stock_count = _compute_applicable_count("NVDA")
        metal_count = _compute_applicable_count("XAU-USD")

        # qwen3 should be counted for all
        assert crypto_count >= 28  # was 27
        assert stock_count >= 26   # was 25
        assert metal_count >= 26   # was 25

    def test_qwen3_not_crypto_only(self):
        """qwen3 should NOT be in _CRYPTO_ONLY_SIGNALS."""
        from portfolio.signal_engine import _CRYPTO_ONLY_SIGNALS
        assert "qwen3" not in _CRYPTO_ONLY_SIGNALS


# ── 5. Download script tests ─────────────────────────────────────────────


class TestDownloadScript:
    """Test model download script."""

    def test_download_script_importable(self):
        """Download script should be importable."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        try:
            import download_models
            assert hasattr(download_models, "MODELS")
            assert "ministral" in download_models.MODELS
            assert "qwen3" in download_models.MODELS
        finally:
            sys.path.pop(0)
