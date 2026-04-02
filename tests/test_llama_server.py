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
