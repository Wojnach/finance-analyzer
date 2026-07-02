"""Tests for the master local-LLM pause switch (portfolio/local_llm_gate.py).

2026-07-02 (local-llm-pause): one bool to pause ALL local model inference —
llama-server GGUF models, subprocess fallbacks, in-process BERT sentiment,
Chronos, and the metals-loop legacy paths. These tests cover the gate's own
decision logic plus every choke point that consumes it.
"""

from unittest.mock import patch

import pytest

import portfolio.local_llm_gate as gate


@pytest.fixture(autouse=True)
def _reset_gate_state(tmp_path, monkeypatch):
    """Point the flag file at tmp_path (xdist safety) and reset transition state."""
    monkeypatch.setattr(gate, "DISABLE_FLAG", tmp_path / "local_llm.disabled")
    monkeypatch.setattr(gate, "_last_state", None)
    yield


class TestGateDecision:
    def test_default_enabled(self, monkeypatch):
        monkeypatch.setattr("portfolio.api_utils.load_config", lambda: {})
        assert gate.local_llm_enabled() is True

    def test_flag_file_pauses(self, monkeypatch):
        monkeypatch.setattr("portfolio.api_utils.load_config", lambda: {})
        gate.DISABLE_FLAG.touch()
        assert gate.local_llm_enabled() is False

    def test_flag_file_removal_resumes(self, monkeypatch):
        monkeypatch.setattr("portfolio.api_utils.load_config", lambda: {})
        gate.DISABLE_FLAG.touch()
        assert gate.local_llm_enabled() is False
        gate.DISABLE_FLAG.unlink()
        assert gate.local_llm_enabled() is True

    def test_config_false_pauses(self, monkeypatch):
        monkeypatch.setattr(
            "portfolio.api_utils.load_config",
            lambda: {"local_llm": {"enabled": False}},
        )
        assert gate.local_llm_enabled() is False

    def test_config_true_enabled(self, monkeypatch):
        monkeypatch.setattr(
            "portfolio.api_utils.load_config",
            lambda: {"local_llm": {"enabled": True}},
        )
        assert gate.local_llm_enabled() is True

    def test_config_unreadable_fails_open(self, monkeypatch):
        def _boom():
            raise OSError("config.json missing (worktree)")

        monkeypatch.setattr("portfolio.api_utils.load_config", _boom)
        assert gate.local_llm_enabled() is True

    def test_flag_wins_even_if_config_says_enabled(self, monkeypatch):
        monkeypatch.setattr(
            "portfolio.api_utils.load_config",
            lambda: {"local_llm": {"enabled": True}},
        )
        gate.DISABLE_FLAG.touch()
        assert gate.local_llm_enabled() is False


class TestLlamaServerChokePoints:
    def test_query_returns_none_when_paused(self):
        import portfolio.llama_server as ls

        with patch.object(ls, "local_llm_enabled", return_value=False):
            assert ls.query_llama_server("ministral3", "prompt") is None

    def test_query_batch_returns_nones_when_paused(self):
        import portfolio.llama_server as ls

        with patch.object(ls, "local_llm_enabled", return_value=False):
            out = ls.query_llama_server_batch(
                "qwen3", [{"prompt": "a"}, {"prompt": "b"}]
            )
        assert out == [None, None]

    def test_model_load_safe_false_when_paused(self):
        import portfolio.llama_server as ls

        with patch.object(ls, "local_llm_enabled", return_value=False):
            assert ls.model_load_safe() is False

    def test_query_stops_resident_server_when_paused(self):
        import portfolio.llama_server as ls

        with patch.object(ls, "local_llm_enabled", return_value=False), \
             patch.object(ls, "_local_proc", object()), \
             patch.object(ls, "_stop_server") as stop:
            ls.query_llama_server("ministral3", "prompt")
        stop.assert_called_once()

    def test_query_no_stop_when_no_server(self):
        import portfolio.llama_server as ls

        with patch.object(ls, "local_llm_enabled", return_value=False), \
             patch.object(ls, "_local_proc", None), \
             patch.object(ls, "_stop_server") as stop:
            ls.query_llama_server("ministral3", "prompt")
        stop.assert_not_called()


class TestBertChokePoint:
    def test_predict_returns_neutral_placeholders_when_paused(self):
        import portfolio.bert_sentiment as bs

        with patch.object(gate, "local_llm_enabled", return_value=False):
            out = bs.predict("CryptoBERT", ["gold to the moon", "silver crash"])
        assert len(out) == 2
        for row in out:
            assert row["sentiment"] == "neutral"
            assert row["confidence"] == 0.0
            assert set(row["scores"]) == {"positive", "negative", "neutral"}

    def test_predict_placeholder_shape_matches_legacy(self):
        import portfolio.bert_sentiment as bs

        with patch.object(gate, "local_llm_enabled", return_value=False):
            out = bs.predict("FinBERT", ["x" * 500])
        # legacy subprocess shape truncates text to 100 chars
        assert out[0]["text"] == "x" * 100
        assert set(out[0]) == {"text", "sentiment", "confidence", "scores"}


class TestForecastChokePoint:
    def test_forecast_chronos_none_when_paused(self):
        import portfolio.forecast_signal as fs

        with patch.object(fs, "local_llm_enabled", return_value=False):
            assert fs.forecast_chronos("XAG-USD", [30.0] * 200) is None


class TestMetalsChokePoints:
    def test_run_ministral_metals_none_when_paused(self):
        import data.metals_llm as mlm

        with patch.object(mlm, "local_llm_enabled", return_value=False):
            assert mlm._run_ministral_metals({"ticker": "XAG-USD"}) is None

    def test_run_chronos_metals_none_when_paused(self):
        import data.metals_llm as mlm

        with patch.object(mlm, "local_llm_enabled", return_value=False):
            assert mlm._run_chronos_metals("XAG-USD", [30.0] * 200) is None
