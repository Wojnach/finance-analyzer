"""Tests for portfolio.llama_server — model management and query serialization."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_llama_state():
    """Reset module-level state between tests."""
    import portfolio.llama_server as ls
    ls._local_proc = None
    ls._local_model = None
    # Clean up lock file if it exists
    import os
    for f in [ls._LOCK_FILE, ls._PID_FILE]:
        try:
            os.remove(f)
        except OSError:
            pass
    yield
    ls._local_proc = None
    ls._local_model = None
    for f in [ls._LOCK_FILE, ls._PID_FILE]:
        try:
            os.remove(f)
        except OSError:
            pass


class TestQueryLlamaServer:
    """Test query_llama_server behavior."""

    def test_unknown_model_returns_none(self):
        from portfolio.llama_server import query_llama_server
        result = query_llama_server("nonexistent_model", "test prompt")
        assert result is None

    @patch("portfolio.llama_server._ensure_model", return_value=True)
    @patch("portfolio.llama_server._acquire_file_lock")
    @patch("portfolio.llama_server._release_file_lock")
    @patch("portfolio.llama_server._requests")
    def test_successful_query(self, mock_req, mock_release, mock_acquire, mock_ensure):
        from portfolio.llama_server import query_llama_server
        mock_fh = MagicMock()
        mock_acquire.return_value = mock_fh
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": "BUY"}
        mock_req.post.return_value = mock_response

        result = query_llama_server("ministral3", "test prompt")
        assert result == "BUY"
        mock_release.assert_called_once_with(mock_fh)

    @patch("portfolio.llama_server._ensure_model", return_value=False)
    @patch("portfolio.llama_server._acquire_file_lock")
    @patch("portfolio.llama_server._release_file_lock")
    def test_model_ensure_failure_returns_none(self, mock_release, mock_acquire, mock_ensure):
        from portfolio.llama_server import query_llama_server
        mock_fh = MagicMock()
        mock_acquire.return_value = mock_fh
        result = query_llama_server("ministral3", "test prompt")
        assert result is None
        mock_release.assert_called_once_with(mock_fh)

    @patch("portfolio.llama_server._acquire_file_lock", return_value=None)
    def test_lock_timeout_returns_none(self, mock_acquire):
        from portfolio.llama_server import query_llama_server
        result = query_llama_server("ministral3", "test prompt")
        assert result is None


class TestBUG165RaceCondition:
    """BUG-165: Verify locks are held during the entire query operation.

    The fix ensures that _thread_lock and the file lock are held from model
    swap through HTTP query completion. This prevents another thread from
    swapping the model while a query is in-flight.
    """

    @patch("portfolio.llama_server._ensure_model", return_value=True)
    @patch("portfolio.llama_server._acquire_file_lock")
    @patch("portfolio.llama_server._release_file_lock")
    @patch("portfolio.llama_server._requests")
    def test_lock_held_during_query(self, mock_req, mock_release, mock_acquire, mock_ensure):
        """Verify the file lock is NOT released before the HTTP query completes."""
        from portfolio.llama_server import query_llama_server

        mock_fh = MagicMock()
        mock_acquire.return_value = mock_fh

        call_order = []

        def track_post(*args, **kwargs):
            call_order.append("http_post")
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"content": "result"}
            return resp

        def track_release(fh):
            call_order.append("release_lock")

        mock_req.post.side_effect = track_post
        mock_release.side_effect = track_release

        query_llama_server("ministral3", "test")

        # Lock release must happen AFTER the HTTP post (BUG-165 fix)
        assert call_order == ["http_post", "release_lock"], (
            f"Lock released before query completed: {call_order}"
        )

    @patch("portfolio.llama_server._ensure_model", return_value=True)
    @patch("portfolio.llama_server._acquire_file_lock")
    @patch("portfolio.llama_server._release_file_lock")
    @patch("portfolio.llama_server._requests")
    def test_lock_released_on_query_exception(self, mock_req, mock_release, mock_acquire, mock_ensure):
        """Verify the file lock is released even if the HTTP query fails."""
        from portfolio.llama_server import query_llama_server

        mock_fh = MagicMock()
        mock_acquire.return_value = mock_fh
        mock_req.post.side_effect = ConnectionError("server killed")

        result = query_llama_server("ministral3", "test")
        assert result is None
        mock_release.assert_called_once_with(mock_fh)

    @patch("portfolio.llama_server._ensure_model", return_value=True)
    @patch("portfolio.llama_server._acquire_file_lock")
    @patch("portfolio.llama_server._release_file_lock")
    @patch("portfolio.llama_server._requests")
    def test_serialized_queries_prevent_race(self, mock_req, mock_release, mock_acquire, mock_ensure):
        """Verify that concurrent queries are serialized (no overlap)."""
        from portfolio.llama_server import query_llama_server

        mock_fh = MagicMock()
        mock_acquire.return_value = mock_fh

        active_queries = {"count": 0, "max": 0}
        lock = threading.Lock()

        def slow_post(*args, **kwargs):
            with lock:
                active_queries["count"] += 1
                active_queries["max"] = max(active_queries["max"], active_queries["count"])
            time.sleep(0.05)  # Simulate query time
            with lock:
                active_queries["count"] -= 1
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"content": "ok"}
            return resp

        mock_req.post.side_effect = slow_post

        threads = []
        for _ in range(3):
            t = threading.Thread(target=query_llama_server, args=("ministral3", "test"))
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=30)

        # BUG-165: With locks held during query, max concurrent should be 1
        assert active_queries["max"] == 1, (
            f"Race condition: {active_queries['max']} concurrent queries detected"
        )


class TestEnsureModel:
    """Test model swap logic."""

    @patch("portfolio.llama_server._read_pid_model", return_value=(1234, "ministral3"))
    @patch("portfolio.llama_server._is_server_alive", return_value=True)
    def test_reuses_already_loaded_model(self, mock_alive, mock_pid):
        from portfolio.llama_server import _ensure_model
        assert _ensure_model("ministral3") is True

    @patch("portfolio.llama_server._read_pid_model", return_value=(None, None))
    @patch("portfolio.llama_server._start_server", return_value=True)
    def test_starts_server_when_none_running(self, mock_start, mock_pid):
        from portfolio.llama_server import _ensure_model
        assert _ensure_model("qwen3") is True
        mock_start.assert_called_once_with("qwen3")


class TestStopServer:
    """Test server stop behavior."""

    @patch("portfolio.llama_server._stop_server")
    def test_stop_all(self, mock_stop):
        from portfolio.llama_server import stop_all_servers
        stop_all_servers()
        mock_stop.assert_called_once()

    @patch("portfolio.llama_server._stop_server")
    def test_stop_specific_model(self, mock_stop):
        import portfolio.llama_server as ls
        ls._local_model = "qwen3"
        ls.stop_server("qwen3")
        mock_stop.assert_called_once()

    @patch("portfolio.llama_server._stop_server")
    def test_stop_wrong_model_no_op(self, mock_stop):
        import portfolio.llama_server as ls
        ls._local_model = "qwen3"
        ls.stop_server("ministral3")
        mock_stop.assert_not_called()


class TestVramReclaimPoll:
    """perf/llama-swap-reduction: VRAM reclaim active poll replaces time.sleep(4).

    The helper saves ~2-3 s per swap when VRAM is already reclaimed, while
    keeping the 4 s ceiling as a hard fallback if nvidia-smi is unavailable
    or VRAM simply hasn't reclaimed yet.
    """

    @patch("portfolio.llama_server._query_free_vram_mb")
    def test_exits_immediately_when_enough_free(self, mock_query):
        """If first probe already shows ≥ min_free_mb, helper returns 0.0 (no sleep)."""
        from portfolio.llama_server import _wait_for_vram_reclaim
        mock_query.return_value = 9000  # 9 GB free, way above 5.5 GB threshold
        waited = _wait_for_vram_reclaim(min_free_mb=5632, max_wait=4.0)
        assert waited == 0.0
        assert mock_query.call_count == 1

    @patch("portfolio.llama_server._query_free_vram_mb")
    def test_falls_back_to_full_sleep_when_nvidia_smi_unavailable(self, mock_query):
        """If nvidia-smi returns None, helper sleeps the full max_wait as before."""
        from portfolio.llama_server import _wait_for_vram_reclaim
        mock_query.return_value = None
        # Use a tiny max_wait so the test doesn't actually sleep 4 s
        waited = _wait_for_vram_reclaim(min_free_mb=5632, max_wait=0.05)
        assert waited >= 0.04  # close to 0.05 s, allowing timing slop

    @patch("portfolio.llama_server._query_free_vram_mb")
    @patch("portfolio.llama_server.time.sleep")
    def test_polls_until_vram_available(self, mock_sleep, mock_query):
        """Helper polls repeatedly while VRAM is below threshold, exits when reclaimed."""
        from portfolio.llama_server import _wait_for_vram_reclaim
        # Sequence: first probe is low, second probe is high — exit after one poll.
        mock_query.side_effect = [1000, 1000, 6000]
        waited = _wait_for_vram_reclaim(min_free_mb=5632, max_wait=4.0)
        # Should have polled at least twice (first probe + at least one loop iter)
        assert mock_query.call_count >= 2
        assert waited >= 0.0
        # sleep was called with 0.1 s increments
        assert any(call.args[0] == 0.1 for call in mock_sleep.call_args_list)

    @patch("portfolio.llama_server._query_free_vram_mb")
    @patch("portfolio.llama_server.time.sleep")
    def test_exits_at_max_wait_if_vram_never_reclaims(self, mock_sleep, mock_query):
        """If VRAM never frees up, helper respects max_wait ceiling and returns."""
        from portfolio.llama_server import _wait_for_vram_reclaim
        # Always report insufficient free VRAM
        mock_query.return_value = 500
        waited = _wait_for_vram_reclaim(min_free_mb=5632, max_wait=0.3)
        # Should have spent close to max_wait, then given up
        assert waited >= 0.0
        assert waited <= 0.5  # allowing some timing slop

    @patch("portfolio.llama_server.subprocess.run")
    def test_query_free_vram_parses_nvidia_smi_output(self, mock_run):
        """_query_free_vram_mb should parse nvidia-smi csv output to int MiB."""
        from portfolio.llama_server import _query_free_vram_mb
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "8192\n"
        mock_run.return_value = mock_result
        assert _query_free_vram_mb() == 8192

    @patch("portfolio.llama_server.subprocess.run")
    def test_query_free_vram_handles_multi_gpu(self, mock_run):
        """Multi-GPU systems return one line per GPU; we only care about GPU 0."""
        from portfolio.llama_server import _query_free_vram_mb
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "8192\n4096\n"  # GPU 0 has 8 GB, GPU 1 has 4 GB
        mock_run.return_value = mock_result
        assert _query_free_vram_mb() == 8192

    @patch("portfolio.llama_server.subprocess.run")
    def test_query_free_vram_returns_none_on_nvidia_smi_failure(self, mock_run):
        """Any nvidia-smi failure (non-zero exit, missing binary, timeout) returns None."""
        from portfolio.llama_server import _query_free_vram_mb
        mock_result = MagicMock()
        mock_result.returncode = 1  # nvidia-smi failed
        mock_result.stdout = ""
        mock_run.return_value = mock_result
        assert _query_free_vram_mb() is None

    @patch("portfolio.llama_server.subprocess.run")
    def test_query_free_vram_returns_none_on_exception(self, mock_run):
        """Missing nvidia-smi binary raises FileNotFoundError; we treat as unavailable."""
        from portfolio.llama_server import _query_free_vram_mb
        mock_run.side_effect = FileNotFoundError("nvidia-smi not on PATH")
        assert _query_free_vram_mb() is None


class TestCachePromptReuse:
    """perf/llama-swap-reduction: cache_prompt: true in HTTP body enables KV
    cache reuse across successive requests sharing a prompt prefix.

    The parameter is passed to llama.cpp server's /completion endpoint. On
    older server builds that don't recognize it, the field is silently
    ignored — no breakage on fallback.
    """

    @patch("portfolio.llama_server._requests")
    def test_query_http_includes_cache_prompt_true(self, mock_req):
        """Every HTTP body sent to /completion must include cache_prompt: True."""
        from portfolio.llama_server import _query_http
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": "ok"}
        mock_req.post.return_value = mock_response

        _query_http("some prompt", n_predict=100)
        call_args = mock_req.post.call_args
        body = call_args.kwargs["json"]
        assert body.get("cache_prompt") is True, (
            f"cache_prompt not enabled in body: {body}"
        )

    @patch("portfolio.llama_server._requests")
    def test_query_http_preserves_cache_prompt_with_stop_tokens(self, mock_req):
        """cache_prompt must still be present when stop tokens are passed."""
        from portfolio.llama_server import _query_http
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": "ok"}
        mock_req.post.return_value = mock_response

        _query_http("prompt", stop=["\n\n", "[INST]"])
        body = mock_req.post.call_args.kwargs["json"]
        assert body.get("cache_prompt") is True
        assert body.get("stop") == ["\n\n", "[INST]"]

    @patch("portfolio.llama_server._ensure_model", return_value=True)
    @patch("portfolio.llama_server._acquire_file_lock")
    @patch("portfolio.llama_server._release_file_lock")
    @patch("portfolio.llama_server._requests")
    def test_query_llama_server_batch_passes_cache_prompt(
        self, mock_req, mock_release, mock_acquire, mock_ensure
    ):
        """Each request in a batch should carry cache_prompt — critical for the
        Ministral phase where 4 per-ticker prompts share ~300 tokens of prefix.
        """
        from portfolio.llama_server import query_llama_server_batch
        mock_fh = MagicMock()
        mock_acquire.return_value = mock_fh
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": "ok"}
        mock_req.post.return_value = mock_response

        prompts = [
            {"prompt": "[INST]prefix BTC data[/INST]"},
            {"prompt": "[INST]prefix ETH data[/INST]"},
            {"prompt": "[INST]prefix XAU data[/INST]"},
        ]
        results = query_llama_server_batch("ministral3", prompts)
        assert len(results) == 3
        # Every post call must have cache_prompt in its body
        for call in mock_req.post.call_args_list:
            assert call.kwargs["json"].get("cache_prompt") is True
