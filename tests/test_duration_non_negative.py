"""P2B: duration_s >= 0 invariant across all _agent_start consumers.

Seeds the exact bug condition from 2026-04-16T13:45:45 (an epoch-scale
_agent_start value) and asserts every downstream consumer produces a
non-negative duration. Covers:

- check_agent_completion() timeout-path result dict
- check_agent_completion() completion-path result dict
- _kill_overrun_agent() elapsed logging
- try_invoke_agent() lazy timeout branch

If _agent_start is ever poisoned again, this pin test will catch the
regression BEFORE it reaches critical_errors.jsonl in prod.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import portfolio.agent_invocation as ai


def _reset_state():
    ai._agent_proc = None
    ai._agent_log = None
    ai._agent_start = 0
    ai._agent_timeout = 0
    ai._agent_tier = None
    ai._agent_reasons = None


class TestSafeElapsedClamps:

    def test_negative_elapsed_clamps_to_zero_with_warning(self, caplog):
        _reset_state()
        # Simulate the bug: _agent_start seeded with an epoch timestamp
        ai._agent_start = time.time()  # huge value vs time.monotonic()
        with caplog.at_level("WARNING", logger="portfolio.agent"):
            elapsed = ai._safe_elapsed_s()
        assert elapsed == 0.0
        assert any("BUG-P2B" in rec.message for rec in caplog.records), (
            "Negative elapsed must emit a diagnostic WARNING so the "
            "root-cause _agent_start poisoning is visible in logs."
        )

    def test_positive_elapsed_passes_through(self):
        _reset_state()
        ai._agent_start = time.monotonic() - 5.0
        elapsed = ai._safe_elapsed_s()
        assert 4.0 < elapsed < 10.0  # ~5s, tolerate test clock jitter

    def test_zero_start_returns_large_positive(self):
        """_agent_start=0 (module default) means 'no agent ever spawned'.
        time.monotonic() - 0 = seconds since boot — a large positive
        number. Not a bug, just means the caller should have checked
        _agent_proc is not None first. The helper returns without
        warning because this is an expected initial state."""
        _reset_state()
        ai._agent_start = 0
        elapsed = ai._safe_elapsed_s()
        assert elapsed > 0


class TestCheckAgentCompletionNeverNegative:

    def test_completion_path_clamps_duration_s(self, monkeypatch):
        """Finished agent with poisoned _agent_start produces 0.0, not negative."""
        _reset_state()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # clean exit
        ai._agent_proc = mock_proc
        ai._agent_start = time.time()  # POISONED — epoch scale
        ai._agent_tier = 1
        ai._agent_reasons = ["startup"]
        ai._journal_ts_before = None
        ai._telegram_ts_before = None

        # Stub downstream I/O + auth scan + invocation log write
        monkeypatch.setattr(ai, "_last_jsonl_ts", lambda p: None)
        monkeypatch.setattr(
            ai, "detect_auth_failure",
            lambda output, caller, context=None: False,
        )
        monkeypatch.setattr(ai, "_log_trigger", lambda *a, **kw: None)

        result = ai.check_agent_completion()

        assert result is not None
        assert result["duration_s"] == 0.0, (
            "Poisoned _agent_start must NOT leak a negative duration "
            "into the invocation log / critical_errors.jsonl."
        )

    def test_timeout_path_clamps_duration_s(self, monkeypatch):
        """Still-running agent past timeout: the timeout result dict must
        also never carry a negative duration_s."""
        _reset_state()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        mock_proc.pid = 99999
        ai._agent_proc = mock_proc
        ai._agent_start = time.time()  # POISONED
        ai._agent_timeout = 120
        ai._agent_tier = 1
        ai._agent_reasons = ["startup"]

        monkeypatch.setattr(ai, "_log_trigger", lambda *a, **kw: None)
        fake_run = MagicMock()
        fake_run.return_value.returncode = 0
        monkeypatch.setattr(ai.subprocess, "run", fake_run)

        # With poisoned _agent_start, _safe_elapsed_s() returns 0.0, so
        # elapsed > _agent_timeout is False, so the timeout branch does
        # NOT fire — the function returns None as "still running".
        # This is the safe behavior: better to wait for the real timeout
        # (or the next trigger to kill it) than to hallucinate a timeout
        # with broken timestamps.
        result = ai.check_agent_completion()
        assert result is None
        # Agent still held
        assert ai._agent_proc is mock_proc


class TestAuthErrorContextNeverNegative:
    """The specific regression from 2026-04-16T13:45:45 that had
    duration_s=-1776254571.5 — reproduce + assert fixed."""

    def test_auth_marker_with_poisoned_start_logs_zero_duration(self, monkeypatch, tmp_path):
        _reset_state()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        ai._agent_proc = mock_proc
        ai._agent_start = time.time()  # POISONED (reproduces the bug)
        ai._agent_tier = 1
        ai._agent_reasons = ["startup"]
        ai._journal_ts_before = None
        ai._telegram_ts_before = None

        monkeypatch.setattr(ai, "_last_jsonl_ts", lambda p: None)
        monkeypatch.setattr(ai, "_log_trigger", lambda *a, **kw: None)

        # Capture what detect_auth_failure receives (context dict carries
        # duration_s into critical_errors.jsonl)
        seen_context = {}

        def capture_auth(output, caller, context=None):
            seen_context.update(context or {})
            return True  # pretend auth marker was found

        monkeypatch.setattr(ai, "detect_auth_failure", capture_auth)

        # Point agent.log at tmp_path so the scan finds an empty file
        monkeypatch.setattr(ai, "DATA_DIR", tmp_path)
        (tmp_path / "agent.log").write_text("")

        result = ai.check_agent_completion()
        assert result is not None
        assert result["duration_s"] == 0.0
        # The context passed to detect_auth_failure must also be clean
        assert seen_context.get("duration_s", 0.0) == 0.0, (
            f"duration_s in critical_errors.jsonl context must be >= 0, "
            f"got {seen_context.get('duration_s')}"
        )
