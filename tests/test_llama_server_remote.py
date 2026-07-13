"""Remote llama-server routing (config local_llm.remote) — 2026-07-13."""

from unittest.mock import MagicMock, patch

from portfolio import llama_server

REMOTE = {"url": "http://100.78.196.30:8788", "models": ["phi4_mini"]}


def _resp(content="ok"):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {"content": content}
    return r


class TestRemoteRouting:
    @patch("portfolio.llama_server._remote_config", return_value=REMOTE)
    @patch("portfolio.llama_server._requests.post", return_value=_resp("HOLD"))
    def test_whitelisted_model_queries_remote(self, mock_post, _cfg):
        out = llama_server.query_llama_server("phi4_mini", "prompt")
        assert out == "HOLD"
        assert mock_post.call_args[0][0].startswith("http://100.78.196.30:8788")

    @patch("portfolio.llama_server._remote_config", return_value=REMOTE)
    @patch("portfolio.llama_server._requests.post")
    def test_non_whitelisted_model_abstains(self, mock_post, _cfg):
        assert llama_server.query_llama_server("qwen3", "prompt") is None
        mock_post.assert_not_called()

    @patch("portfolio.llama_server._remote_config", return_value=REMOTE)
    @patch("portfolio.llama_server.local_llm_enabled", return_value=False)
    @patch("portfolio.llama_server._requests.post", return_value=_resp("HOLD"))
    def test_remote_bypasses_local_gate(self, mock_post, _gate, _cfg):
        # Gate flag blocks LOCAL inference only; remote inference runs
        # nothing on this machine and must still work.
        assert llama_server.query_llama_server("phi4_mini", "prompt") == "HOLD"

    @patch("portfolio.llama_server._remote_config", return_value=REMOTE)
    @patch(
        "portfolio.llama_server._requests.post",
        side_effect=ConnectionError("refused"),
    )
    def test_unreachable_remote_abstains(self, mock_post, _cfg):
        assert llama_server.query_llama_server("phi4_mini", "prompt") is None

    @patch("portfolio.llama_server._remote_config", return_value=REMOTE)
    @patch("portfolio.llama_server._start_server")
    @patch("portfolio.llama_server._requests.post", return_value=_resp())
    def test_remote_never_spawns_local_server(self, _post, mock_start, _cfg):
        llama_server.query_llama_server("phi4_mini", "prompt")
        llama_server.query_llama_server("qwen3", "prompt")
        mock_start.assert_not_called()

    @patch("portfolio.llama_server._remote_config", return_value=REMOTE)
    @patch("portfolio.llama_server._requests.post", return_value=_resp("A"))
    def test_batch_routes_remote(self, mock_post, _cfg):
        out = llama_server.query_llama_server_batch(
            "phi4_mini", [{"prompt": "a"}, {"prompt": "b"}]
        )
        assert out == ["A", "A"]
        assert mock_post.call_count == 2

    @patch("portfolio.llama_server._remote_config", return_value=None)
    @patch("portfolio.llama_server.local_llm_enabled", return_value=False)
    def test_no_remote_config_keeps_local_gate_semantics(self, _gate, _cfg):
        assert llama_server.query_llama_server("phi4_mini", "prompt") is None
