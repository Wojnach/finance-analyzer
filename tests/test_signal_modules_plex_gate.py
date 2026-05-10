"""Tests for the Plex-aware subprocess fallback gate in qwen3_signal /
ministral_signal.

When `query_llama_server` returns None because `_start_server` aborted due to
Plex transcoding, the cold-start subprocess fallback would re-create the
exact VRAM pressure the abort was trying to avoid. These tests confirm
`model_load_safe()` blocks that path and the call returns a HOLD instead.
"""

from unittest.mock import MagicMock, patch


class TestQwen3SignalPlexGate:
    """`portfolio.qwen3_signal._call_qwen3` skips subprocess fallback when unsafe."""

    @patch("portfolio.qwen3_signal.run_safe")
    @patch("portfolio.llama_server.model_load_safe", return_value=False)
    @patch("portfolio.qwen3_signal.query_llama_server", return_value=None)
    def test_unsafe_returns_hold_without_spawning(
        self, mock_qls, mock_safe, mock_run_safe
    ):
        """model_load_safe=False → HOLD, no subprocess spawn."""
        # _build_prompt is imported inside _call_qwen3 from qwen3_trader (lazy),
        # so we patch at the source module.
        from portfolio.qwen3_signal import _call_qwen3
        with patch("portfolio.qwen3_trader._build_prompt", return_value="prompt"):
            result = _call_qwen3({"ticker": "BTC-USD"})
        assert result["action"] == "HOLD"
        assert "Plex" in result["reasoning"]
        mock_run_safe.assert_not_called()

    @patch("portfolio.qwen3_signal.run_safe")
    @patch("portfolio.llama_server.model_load_safe", return_value=True)
    @patch("portfolio.qwen3_signal.query_llama_server", return_value=None)
    def test_safe_proceeds_to_subprocess(self, mock_qls, mock_safe, mock_run_safe):
        """model_load_safe=True → falls back to subprocess as before (regression guard)."""
        from portfolio.qwen3_signal import _call_qwen3
        proc_result = MagicMock()
        proc_result.returncode = 0
        proc_result.stdout = '{"action": "BUY", "reasoning": "ok"}'
        mock_run_safe.return_value = proc_result
        with patch("portfolio.qwen3_trader._build_prompt", return_value="prompt"):
            result = _call_qwen3({"ticker": "BTC-USD"})
        assert result == {"action": "BUY", "reasoning": "ok"}
        mock_run_safe.assert_called_once()


class TestMinistralSignalPlexGate:
    """`portfolio.ministral_signal._call_model` skips subprocess fallback when unsafe."""

    @patch("portfolio.ministral_signal.run_safe")
    @patch("portfolio.llama_server.model_load_safe", return_value=False)
    @patch("portfolio.ministral_signal.query_llama_server", return_value=None)
    def test_unsafe_returns_hold_without_spawning(
        self, mock_qls, mock_safe, mock_run_safe
    ):
        """model_load_safe=False → HOLD, no subprocess spawn."""
        from portfolio.ministral_signal import _call_model
        with patch("portfolio.ministral_trader._build_prompt", return_value="prompt"), \
             patch("portfolio.ministral_trader._parse_response", return_value=("HOLD", "x", None)):
            result = _call_model({"ticker": "BTC-USD"})
        assert result["action"] == "HOLD"
        assert "Plex" in result["reasoning"]
        mock_run_safe.assert_not_called()

    @patch("portfolio.ministral_signal.run_safe")
    @patch("portfolio.llama_server.model_load_safe", return_value=True)
    @patch("portfolio.ministral_signal.query_llama_server", return_value=None)
    def test_safe_proceeds_to_subprocess(self, mock_qls, mock_safe, mock_run_safe):
        """model_load_safe=True → falls back to subprocess (regression guard)."""
        from portfolio.ministral_signal import _call_model
        proc_result = MagicMock()
        proc_result.returncode = 0
        proc_result.stdout = '{"action": "BUY", "reasoning": "ok"}'
        mock_run_safe.return_value = proc_result
        with patch("portfolio.ministral_trader._build_prompt", return_value="prompt"):
            result = _call_model({"ticker": "BTC-USD"})
        assert result == {"action": "BUY", "reasoning": "ok"}
        mock_run_safe.assert_called_once()
