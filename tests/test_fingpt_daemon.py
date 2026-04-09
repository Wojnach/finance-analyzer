"""Unit tests for the warm fingpt daemon client in portfolio.sentiment.

Covers the NDJSON protocol, thread-safety, lazy init, and crash-restart
behavior. Does not spawn a real daemon — mocks subprocess.Popen so the
tests run on any machine without the .venv-llm or GGUF models present.
"""

from __future__ import annotations

import json
import threading
from unittest import mock

import pytest

from portfolio import sentiment


class _FakePopenPipe:
    """In-memory bidirectional pipe that mimics subprocess.Popen.stdin / stdout."""

    def __init__(self, script: list[dict] | None = None):
        # Pre-seed the responses the fake daemon will emit, one line per item.
        # The first entry is the "ready" handshake.
        self._script = list(script or [{"ready": True, "model": "fake-model"}])
        self._read_lines: list[str] = [json.dumps(o) + "\n" for o in self._script]
        self._write_lines: list[str] = []

    # stdin behavior (parent writes, daemon reads)
    def write(self, s: str) -> int:
        self._write_lines.append(s)
        return len(s)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    # stdout behavior (daemon writes, parent reads)
    def readline(self) -> str:
        if not self._read_lines:
            return ""
        return self._read_lines.pop(0)

    # Helper for tests — queue another daemon response
    def enqueue_response(self, obj: dict) -> None:
        self._read_lines.append(json.dumps(obj) + "\n")


class _FakePopen:
    """Stand-in for subprocess.Popen returned by _spawn_fingpt_daemon."""

    def __init__(self, script: list[dict] | None = None, pid: int = 12345):
        self.pid = pid
        self._alive = True
        self._returncode: int | None = None
        self.stdin = _FakePopenPipe()  # parent writes here (daemon reads)
        # Pre-seed stdout with the ready handshake and any additional scripted
        # responses the test will consume.
        self.stdout = _FakePopenPipe(script=script)

    @property
    def returncode(self) -> int | None:
        return self._returncode

    def poll(self) -> int | None:
        return None if self._alive else self._returncode

    def wait(self, timeout: float | None = None) -> int:
        self._alive = False
        self._returncode = self._returncode or 0
        return self._returncode

    def kill(self) -> None:
        self._alive = False
        self._returncode = -9

    # Test-only helpers
    def simulate_crash(self) -> None:
        self._alive = False
        self._returncode = 1


@pytest.fixture(autouse=True)
def _reset_daemon_state():
    """Ensure each test starts with a fresh daemon-client singleton."""
    sentiment._fingpt_daemon_proc = None
    sentiment._fingpt_request_id = 0
    yield
    sentiment._fingpt_daemon_proc = None


def _spawn_returning(fake: _FakePopen) -> mock.MagicMock:
    """Return a patch target for subprocess.Popen that yields `fake`."""
    mock_popen = mock.MagicMock()
    mock_popen.return_value = fake
    return mock_popen


def test_lazy_daemon_spawn_on_first_call():
    """The first _run_fingpt call should spawn the daemon and send one request."""
    fake = _FakePopen(script=[
        {"ready": True, "model": "fake-model"},
        {"request_id": 1, "result": [{"sentiment": "positive", "confidence": 0.9}]},
    ])
    with mock.patch("portfolio.sentiment.subprocess.Popen", _spawn_returning(fake)):
        result = sentiment._run_fingpt(["hello world"])
    assert result == [{"sentiment": "positive", "confidence": 0.9}]
    assert sentiment._fingpt_daemon_proc is fake
    # One request line was written
    assert len(fake.stdin._write_lines) == 1
    req = json.loads(fake.stdin._write_lines[0])
    assert req["mode"] == "headlines"
    assert req["texts"] == ["hello world"]
    assert req["request_id"] == 1


def test_daemon_reused_across_calls_single_spawn():
    """A second call should reuse the existing daemon instead of spawning again."""
    fake = _FakePopen(script=[
        {"ready": True, "model": "fake-model"},
        {"request_id": 1, "result": ["a"]},
        {"request_id": 2, "result": ["b"]},
    ])
    popen_spy = _spawn_returning(fake)
    with mock.patch("portfolio.sentiment.subprocess.Popen", popen_spy):
        r1 = sentiment._run_fingpt(["h1"])
        r2 = sentiment._run_fingpt(["h2"])
    assert r1 == ["a"]
    assert r2 == ["b"]
    # Popen called exactly once — the second request reused the live daemon
    assert popen_spy.call_count == 1
    # Request IDs incremented
    req_ids = [json.loads(ln)["request_id"] for ln in fake.stdin._write_lines]
    assert req_ids == [1, 2]


def test_cumulative_mode_sets_flag():
    """cumulative=True should route to the cumulative mode in the protocol."""
    fake = _FakePopen(script=[
        {"ready": True, "model": "fake-model"},
        {"request_id": 1, "result": {"sentiment": "negative", "confidence": 0.8}},
    ])
    with mock.patch("portfolio.sentiment.subprocess.Popen", _spawn_returning(fake)):
        result = sentiment._run_fingpt(["h1", "h2", "h3"], cumulative=True, ticker="BTC")
    assert result == {"sentiment": "negative", "confidence": 0.8}
    req = json.loads(fake.stdin._write_lines[0])
    assert req["mode"] == "cumulative"
    assert req["ticker"] == "BTC"


def test_daemon_error_response_raises_runtime_error():
    """If the daemon replies with {"error": ...} we raise after retry."""
    # First spawn: immediately errors. Retry spawns a second fake that also errors.
    spawn_count = [0]
    fakes: list[_FakePopen] = []

    def factory(*args, **kwargs):
        fake = _FakePopen(script=[
            {"ready": True, "model": "fake-model"},
            {"request_id": spawn_count[0] + 1, "error": "bad things"},
        ])
        spawn_count[0] += 1
        fakes.append(fake)
        return fake

    with mock.patch("portfolio.sentiment.subprocess.Popen", side_effect=factory):
        with pytest.raises(RuntimeError, match="FinGPT failed after retry"):
            sentiment._run_fingpt(["h1"])
    assert spawn_count[0] == 2  # original + one retry


def test_crashed_daemon_restarts_transparently():
    """If the daemon process dies between calls, the next call should respawn it."""
    # Two separate daemons — first one crashes, second succeeds
    first = _FakePopen(script=[
        {"ready": True, "model": "fake-model"},
        {"request_id": 1, "result": ["ok1"]},
    ])
    second = _FakePopen(script=[
        {"ready": True, "model": "fake-model"},
        {"request_id": 2, "result": ["ok2"]},
    ])
    instances = [first, second]

    def factory(*args, **kwargs):
        return instances.pop(0)

    with mock.patch("portfolio.sentiment.subprocess.Popen", side_effect=factory):
        r1 = sentiment._run_fingpt(["h1"])
        # Simulate the first daemon crashing between calls
        first.simulate_crash()
        r2 = sentiment._run_fingpt(["h2"])
    assert r1 == ["ok1"]
    assert r2 == ["ok2"]
    # Each daemon saw exactly one request
    assert len(first.stdin._write_lines) == 1
    assert len(second.stdin._write_lines) == 1


def test_ready_handshake_failure_raises_on_spawn():
    """If the daemon emits {ready: False}, spawn should fail fast."""
    fake = _FakePopen(script=[{"ready": False, "error": "no model"}])
    with mock.patch("portfolio.sentiment.subprocess.Popen", _spawn_returning(fake)):
        with pytest.raises(RuntimeError, match="FinGPT failed after retry"):
            sentiment._run_fingpt(["h1"])


def test_thread_safe_concurrent_calls_serialize():
    """Multiple threads calling _run_fingpt concurrently should serialize and
    each get a distinct request_id with no crosstalk."""
    # One daemon serving N requests, each thread gets one scripted response
    n_threads = 8
    script = [{"ready": True, "model": "fake-model"}]
    for i in range(1, n_threads + 1):
        script.append({"request_id": i, "result": [f"r{i}"]})
    fake = _FakePopen(script=script)

    results: dict[int, list] = {}

    def worker(idx: int) -> None:
        results[idx] = sentiment._run_fingpt([f"headline{idx}"])

    with mock.patch("portfolio.sentiment.subprocess.Popen", _spawn_returning(fake)):
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # All threads got a non-None result
    assert len(results) == n_threads
    # Request IDs assigned sequentially were 1..n_threads, no duplicates
    req_ids_sent = [json.loads(ln)["request_id"] for ln in fake.stdin._write_lines]
    assert sorted(req_ids_sent) == list(range(1, n_threads + 1))
    assert len(req_ids_sent) == n_threads  # no lost writes


def test_stop_daemon_is_idempotent():
    """_stop_fingpt_daemon should tolerate being called when no daemon is running."""
    assert sentiment._fingpt_daemon_proc is None
    sentiment._stop_fingpt_daemon()  # should not raise
    # And after an actual spawn + stop
    fake = _FakePopen(script=[
        {"ready": True, "model": "fake-model"},
        {"request_id": 1, "result": []},
    ])
    with mock.patch("portfolio.sentiment.subprocess.Popen", _spawn_returning(fake)):
        sentiment._run_fingpt([])
    sentiment._stop_fingpt_daemon()
    assert sentiment._fingpt_daemon_proc is None
    # Second stop is still a no-op
    sentiment._stop_fingpt_daemon()
