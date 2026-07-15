"""Dynamic LLM voting: LLM signals leave the applicable set when GPU
inference is unavailable (herc2 asleep) — removal, not abstain (2026-07-15)."""

from unittest.mock import patch

from portfolio.signal_engine import _compute_applicable_count
from portfolio.tickers import GPU_SIGNALS


def test_phi4_in_gpu_signals():
    assert "phi4_mini" in GPU_SIGNALS


def test_llm_signals_excluded_when_gpu_skipped():
    # skip_gpu=True must drop every GPU signal from the applicable count.
    for ticker in ("BTC-USD", "XAU-USD"):
        full = _compute_applicable_count(ticker, skip_gpu=False)
        skipped = _compute_applicable_count(ticker, skip_gpu=True)
        # phi4_mini is enabled on these via per-ticker override, so the
        # count must drop by at least the applicable GPU signals present.
        assert skipped < full, f"{ticker}: GPU signals not removed"


class TestRemoteAvailabilityGate:
    def _cfg(self):
        return {
            "url": "http://100.78.196.30:8788",
            "models": ["phi4_mini"],
            "connect_timeout": 3,
        }

    @patch("portfolio.llama_server._remote_health", {"ts": 0.0, "ok": None})
    @patch("portfolio.llama_server._requests.get")
    def test_reachable_remote_available(self, mock_get):
        from portfolio import llama_server as ls

        mock_get.return_value.status_code = 200
        with patch.object(ls, "_remote_config", return_value=self._cfg()):
            assert ls.remote_llm_available() is True

    @patch("portfolio.llama_server._remote_health", {"ts": 0.0, "ok": None})
    @patch(
        "portfolio.llama_server._requests.get",
        side_effect=ConnectionError("herc2 asleep"),
    )
    def test_unreachable_remote_unavailable(self, mock_get):
        from portfolio import llama_server as ls

        with patch.object(ls, "_remote_config", return_value=self._cfg()):
            assert ls.remote_llm_available() is False

    @patch("portfolio.llama_server._remote_health", {"ts": 0.0, "ok": None})
    def test_no_remote_uses_local_gate(self):
        from portfolio import llama_server as ls

        with (
            patch.object(ls, "_remote_config", return_value=None),
            patch.object(ls, "local_llm_enabled", return_value=False),
        ):
            assert ls.remote_llm_available() is False
