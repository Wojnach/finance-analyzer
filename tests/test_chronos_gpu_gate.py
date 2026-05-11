"""Tests for Chronos forecast-server GPU gating (2026-05-11).

Background: data/metals_llm.py runs Chronos as a persistent subprocess
server. Every cycle the metals loop logged ``Chronos: insufficient VRAM``
because Plex / ministral / qwen3 were holding VRAM at the moment of the
startup probe. Routing the start + query paths through ``gpu_gate``
serialises GPU access so other LLMs release first.

These tests mock both ``gpu_gate`` and the subprocess pipeline to verify:
  * the gate is acquired before VRAM is probed (start path)
  * the gate is acquired before stdin write (query path)
  * the gate releases on the success path (context-manager __exit__)
  * the gate releases on the exception path
  * a 30 s timeout returns None (HOLD-equivalent) without spawning the model
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _reset_chronos_globals():
    """Reset metals_llm chronos handles before/after each test (xdist safety)."""
    try:
        import data.metals_llm as mlm
    except ImportError:
        yield
        return
    attrs = ("_chronos_proc", "_chronos_job")
    saved = {a: getattr(mlm, a, None) for a in attrs}
    for a in attrs:
        if hasattr(mlm, a):
            setattr(mlm, a, None)
    try:
        yield
    finally:
        for a, v in saved.items():
            if hasattr(mlm, a):
                setattr(mlm, a, None)


def _make_gate(acquired: bool, calls: list | None = None):
    """Build a fake gpu_gate context manager that records calls.

    `calls` accumulates (model_name, timeout, phase) tuples so tests can
    assert ordering — entry, body, exit — and timeout values.
    """
    if calls is None:
        calls = []

    @contextmanager
    def _gate(model_name: str, timeout: float = 60):
        calls.append((model_name, timeout, "enter"))
        try:
            yield acquired
        finally:
            calls.append((model_name, timeout, "exit"))
    return _gate, calls


# ---------------------------------------------------------------------------
# Start path
# ---------------------------------------------------------------------------


class TestStartPathGating:
    def test_start_acquires_gpu_gate_before_vram_check(self):
        """_start_chronos_server must enter the gate BEFORE probing VRAM,
        otherwise the probe fires while ministral/qwen3 still hold the GPU."""
        from data import metals_llm

        fake_gate, gate_calls = _make_gate(acquired=True)

        # VRAM probe — return plenty of free so VRAM gate passes.
        with patch.object(metals_llm, "_start_chronos_server_inner", return_value=MagicMock()) as inner:
            with patch("portfolio.gpu_gate.gpu_gate", fake_gate):
                metals_llm._start_chronos_server()

        # Inner must have been called once (after entering the gate)
        assert inner.call_count == 1
        # Gate must have entered before inner was called and exited after
        assert gate_calls[0] == ("chronos-startup", 30, "enter")
        assert gate_calls[-1] == ("chronos-startup", 30, "exit")

    def test_start_returns_none_on_gate_timeout(self):
        """If gpu_gate times out, _start returns None and never calls inner."""
        from data import metals_llm

        fake_gate, gate_calls = _make_gate(acquired=False)

        with patch.object(metals_llm, "_start_chronos_server_inner") as inner:
            with patch("portfolio.gpu_gate.gpu_gate", fake_gate):
                result = metals_llm._start_chronos_server()

        assert result is None
        assert inner.call_count == 0
        # Gate enter + exit still recorded (releases on timeout path)
        assert ("chronos-startup", 30, "enter") in gate_calls
        assert ("chronos-startup", 30, "exit") in gate_calls

    def test_start_releases_gate_on_inner_exception(self):
        """Even when the inner launch raises, the gate context exits cleanly."""
        from data import metals_llm

        fake_gate, gate_calls = _make_gate(acquired=True)

        with patch.object(metals_llm, "_start_chronos_server_inner",
                          side_effect=RuntimeError("model load died")) as inner:
            with patch("portfolio.gpu_gate.gpu_gate", fake_gate):
                with pytest.raises(RuntimeError):
                    metals_llm._start_chronos_server()

        # Inner was invoked
        assert inner.call_count == 1
        # The exit half of the gate MUST have run (release-on-exception)
        exits = [c for c in gate_calls if c[2] == "exit"]
        assert len(exits) == 1, f"expected 1 exit, got {gate_calls}"

    def test_start_falls_open_when_gpu_gate_unimportable(self):
        """If portfolio.gpu_gate can't be imported (e.g. test env), the
        legacy path still works — _start_chronos_server_inner is called
        directly."""
        from data import metals_llm

        # Force ImportError on the gpu_gate import inside _start_chronos_server.
        # The function does `from portfolio.gpu_gate import gpu_gate` — we
        # patch builtins.__import__ to raise for that specific module.
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def _bad_import(name, *args, **kwargs):
            if name == "portfolio.gpu_gate":
                raise ImportError("simulated missing portfolio path")
            return real_import(name, *args, **kwargs)

        with patch.object(metals_llm, "_start_chronos_server_inner",
                          return_value=MagicMock()) as inner:
            with patch("builtins.__import__", side_effect=_bad_import):
                metals_llm._start_chronos_server()

        assert inner.call_count == 1


# ---------------------------------------------------------------------------
# Query path
# ---------------------------------------------------------------------------


class TestQueryPathGating:
    def test_query_acquires_gpu_gate_before_stdin_write(self):
        """_query_chronos_server must enter the gate before pushing the request."""
        from data import metals_llm

        fake_gate, gate_calls = _make_gate(acquired=True)

        with patch.object(metals_llm, "_query_chronos_server_inner",
                          return_value={"1h": {"direction": "up"}}) as inner:
            with patch("portfolio.gpu_gate.gpu_gate", fake_gate):
                result = metals_llm._query_chronos_server([100.0] * 50, horizons=(1, 3))

        assert result == {"1h": {"direction": "up"}}
        assert inner.call_count == 1
        # First gate call must be "chronos" enter; last call must be exit
        assert gate_calls[0] == ("chronos", 30, "enter")
        assert gate_calls[-1] == ("chronos", 30, "exit")

    def test_query_returns_sentinel_on_gate_timeout(self):
        """Gate timeout → return the _CHRONOS_GATE_TIMEOUT sentinel
        (distinguishable from bare-None "server unavailable") so the
        caller can refuse to fall back to a GPU-loading subprocess.

        Codex fix A 2026-05-11: previously returned bare None which the
        _run_chronos_metals caller treated as "server unavailable" and
        spawned a subprocess that loaded Chronos on CUDA — defeating
        the entire purpose of the gate.
        """
        from data import metals_llm

        fake_gate, gate_calls = _make_gate(acquired=False)

        with patch.object(metals_llm, "_query_chronos_server_inner") as inner:
            with patch("portfolio.gpu_gate.gpu_gate", fake_gate):
                result = metals_llm._query_chronos_server([100.0] * 50)

        assert result is metals_llm._CHRONOS_GATE_TIMEOUT, (
            f"expected _CHRONOS_GATE_TIMEOUT sentinel, got {result!r}"
        )
        assert result is not None, "must not be bare None — caller would fall back"
        assert inner.call_count == 0
        assert ("chronos", 30, "enter") in gate_calls
        assert ("chronos", 30, "exit") in gate_calls

    def test_query_releases_gate_on_inner_exception(self):
        """Gate context releases even if the inner query raises."""
        from data import metals_llm

        fake_gate, gate_calls = _make_gate(acquired=True)

        with patch.object(metals_llm, "_query_chronos_server_inner",
                          side_effect=ValueError("subprocess pipe broke")) as inner:
            with patch("portfolio.gpu_gate.gpu_gate", fake_gate):
                with pytest.raises(ValueError):
                    metals_llm._query_chronos_server([100.0] * 50)

        assert inner.call_count == 1
        exits = [c for c in gate_calls if c[2] == "exit"]
        assert len(exits) == 1

    def test_query_falls_open_when_gpu_gate_unimportable(self):
        """ImportError on gpu_gate → fall through to inner query directly."""
        from data import metals_llm

        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def _bad_import(name, *args, **kwargs):
            if name == "portfolio.gpu_gate":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        with patch.object(metals_llm, "_query_chronos_server_inner",
                          return_value={"ok": True}) as inner:
            with patch("builtins.__import__", side_effect=_bad_import):
                result = metals_llm._query_chronos_server([1.0, 2.0])

        assert result == {"ok": True}
        assert inner.call_count == 1


# ---------------------------------------------------------------------------
# Re-entrancy guard
# ---------------------------------------------------------------------------


class TestNoFallbackOnGateTimeout:
    """Regression for codex fix A 2026-05-11: when the gate times out,
    _run_chronos_metals MUST return None (HOLD) without spawning the
    one-shot subprocess fallback. The subprocess path imports
    portfolio.forecast_signal.forecast_chronos which loads Chronos on
    CUDA — exactly the race the gate was protecting against.
    """

    def test_run_chronos_metals_no_subprocess_on_gate_timeout(self):
        from data import metals_llm

        # Force the inner query to return the gate-timeout sentinel.
        with patch.object(
            metals_llm,
            "_query_chronos_server",
            return_value=metals_llm._CHRONOS_GATE_TIMEOUT,
        ):
            with patch.object(metals_llm.subprocess, "run") as run_mock:
                with patch.object(metals_llm.subprocess, "Popen", create=True) as popen_mock:
                    result = metals_llm._run_chronos_metals(
                        "XAG-USD", [1.0] * 50, horizons=(1, 3),
                    )

        assert result is None, "gate timeout must produce HOLD (None)"
        assert run_mock.call_count == 0, (
            f"subprocess.run must NOT be called on gate timeout — "
            f"got {run_mock.call_count} calls (would load Chronos on CUDA)"
        )
        assert popen_mock.call_count == 0, (
            "subprocess.Popen must NOT be called on gate timeout"
        )

    def test_run_chronos_metals_falls_through_on_bare_none(self):
        """Sanity: bare-None (server-unavailable) still allows the
        subprocess fallback — the sentinel only blocks the gate-timeout
        case. Without this, every cold-start would HOLD until the
        persistent server came up."""
        from data import metals_llm

        # Bare None = server unavailable, not gate timeout
        with patch.object(metals_llm, "_query_chronos_server", return_value=None):
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = '{"1h": {"direction": "up"}}'
            mock_proc.stderr = ""
            with patch.object(metals_llm.subprocess, "run", return_value=mock_proc) as run_mock:
                result = metals_llm._run_chronos_metals(
                    "XAG-USD", [1.0] * 50, horizons=(1, 3),
                )

        # Fallback subprocess WAS invoked
        assert run_mock.call_count == 1, "bare-None must allow subprocess fallback"
        assert result == {"1h": {"direction": "up"}}


class TestReentrancySafety:
    def test_query_inner_calls_start_inner_not_outer(self):
        """When _query_chronos_server_inner sees a dead/missing proc, it
        must call _start_chronos_server_inner (no gate) rather than
        _start_chronos_server (which would try to re-acquire the same
        non-reentrant lock and self-deadlock for 30 s)."""
        from data import metals_llm

        # No process running → cold-start path will fire
        metals_llm._chronos_proc = None

        with patch.object(metals_llm, "_start_chronos_server_inner") as inner_start:
            with patch.object(metals_llm, "_start_chronos_server") as outer_start:
                # _start_chronos_server_inner leaves _chronos_proc None
                # → query returns None — that's fine, we only assert which
                # cold-start variant was invoked.
                metals_llm._query_chronos_server_inner([1.0, 2.0])

        assert inner_start.call_count == 1, "must call non-gated inner start"
        assert outer_start.call_count == 0, "must NOT call gated outer start (self-deadlock)"
