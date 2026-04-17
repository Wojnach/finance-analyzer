"""P1B: T1 hard-timeout enforced from check_agent_completion, not just
try_invoke_agent. Also covers the shared _kill_overrun_agent helper so
both callers produce identical kill semantics.

Scenario that motivated the fix: 2026-04-16 T1 was invoked at 16:04:58
with timeout=120s, completed at 16:15:01 (603s). The lazy timeout
check in try_invoke_agent only fires when a NEW trigger arrives, so a
hung agent between triggers runs indefinitely.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import portfolio.agent_invocation as ai


def _reset_state():
    ai._agent_proc = None
    ai._agent_log = None
    ai._agent_start = 0
    ai._agent_timeout = 0
    ai._agent_tier = None
    ai._agent_reasons = None
    ai._journal_ts_before = None
    ai._telegram_ts_before = None


class TestCheckCompletionKillsOverrunAgent:
    """check_agent_completion must kill hung agents past their timeout."""

    def test_hung_agent_past_timeout_is_killed(self, monkeypatch):
        _reset_state()

        # Simulate a live T1 invocation started 200s ago with 120s timeout.
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # still running
        mock_proc.pid = 99999
        ai._agent_proc = mock_proc
        ai._agent_start = time.monotonic() - 200.0
        ai._agent_timeout = 120
        ai._agent_tier = 1
        ai._agent_reasons = ["startup"]
        ai._journal_ts_before = "2026-04-17T10:00:00+00:00"
        ai._telegram_ts_before = "2026-04-17T10:00:00+00:00"

        # Stub out subprocess.run for the taskkill call and _log_trigger
        monkeypatch.setattr(ai, "_log_trigger", MagicMock())
        fake_run = MagicMock()
        fake_run.return_value.returncode = 0
        monkeypatch.setattr(ai.subprocess, "run", fake_run)

        result = ai.check_agent_completion()

        assert result is not None, "should return a timeout dict, not None"
        assert result["status"] == "timeout"
        assert result["tier"] == 1
        assert result["journal_written"] is False
        assert result["telegram_sent"] is False
        assert result["duration_s"] >= 200.0
        # State cleared
        assert ai._agent_proc is None

    def test_agent_under_timeout_keeps_running(self, monkeypatch):
        _reset_state()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 99999
        ai._agent_proc = mock_proc
        ai._agent_start = time.monotonic() - 30.0  # well under 120s
        ai._agent_timeout = 120
        ai._agent_tier = 1
        ai._agent_reasons = ["startup"]

        # If the kill helper were invoked, it would call _log_trigger
        log_trigger = MagicMock()
        monkeypatch.setattr(ai, "_log_trigger", log_trigger)

        result = ai.check_agent_completion()

        assert result is None
        assert log_trigger.call_count == 0
        assert ai._agent_proc is mock_proc  # still alive

    def test_no_timeout_when_agent_timeout_unset(self, monkeypatch):
        """Defensive: if _agent_timeout is 0 (fresh module), don't kill a
        running agent just because elapsed > 0."""
        _reset_state()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 99999
        ai._agent_proc = mock_proc
        ai._agent_start = time.monotonic() - 1000.0
        ai._agent_timeout = 0  # not set
        monkeypatch.setattr(ai, "_log_trigger", MagicMock())

        result = ai.check_agent_completion()
        assert result is None
        assert ai._agent_proc is mock_proc


class TestKillOverrunHelper:
    """Shared helper contract: kill the process, clear state, log trigger."""

    def test_helper_returns_true_on_clean_kill(self, monkeypatch):
        _reset_state()
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        ai._agent_proc = mock_proc
        ai._agent_start = time.monotonic() - 150.0
        ai._agent_tier = 1
        ai._agent_reasons = ["startup"]

        monkeypatch.setattr(ai, "_log_trigger", MagicMock())
        fake_run = MagicMock()
        fake_run.return_value.returncode = 0
        monkeypatch.setattr(ai.subprocess, "run", fake_run)

        assert ai._kill_overrun_agent() is True
        assert ai._agent_proc is None

    def test_helper_returns_false_when_taskkill_fails(self, monkeypatch):
        _reset_state()
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        ai._agent_proc = mock_proc
        ai._agent_start = time.monotonic() - 150.0

        monkeypatch.setattr(ai, "_log_trigger", MagicMock())
        fake_run = MagicMock()
        fake_run.return_value.returncode = 1  # unexpected failure
        fake_run.return_value.stderr = b"Access denied"
        monkeypatch.setattr(ai.subprocess, "run", fake_run)
        # platform.system is called via ai.platform, stub to Windows so
        # we hit the taskkill path
        monkeypatch.setattr(ai.platform, "system", lambda: "Windows")

        assert ai._kill_overrun_agent() is False

    def test_helper_is_noop_when_no_agent(self, monkeypatch):
        _reset_state()
        monkeypatch.setattr(ai, "_log_trigger", MagicMock())
        # Should not raise, should return True (nothing to kill)
        assert ai._kill_overrun_agent() is True


class TestInvokeAgentReusesHelper:
    """try_invoke_agent must still honor the same kill path on timeout."""

    def test_invoke_agent_times_out_and_kills(self, monkeypatch):
        _reset_state()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 77777
        ai._agent_proc = mock_proc
        ai._agent_start = time.monotonic() - 200.0
        ai._agent_timeout = 120
        ai._agent_tier = 1
        ai._agent_reasons = ["old-trigger"]

        monkeypatch.setattr(ai, "_log_trigger", MagicMock())
        fake_run = MagicMock()
        fake_run.return_value.returncode = 0
        monkeypatch.setattr(ai.subprocess, "run", fake_run)
        # Block layer2 enabled check so we don't spawn a new agent
        monkeypatch.setattr(
            ai, "_load_config", lambda: {"layer2": {"enabled": False}}
        )

        with patch.object(ai, "_kill_overrun_agent", return_value=True) as m_kill:
            ai.invoke_agent(["new-trigger"], tier=1)
            # Because layer2 enabled=false gate fires BEFORE the kill path,
            # the helper is NOT called. Swap to enabled=true for the assertion.
            assert m_kill.call_count == 0

        # Now with enabled=true — the kill path runs
        monkeypatch.setattr(
            ai, "_load_config", lambda: {"layer2": {"enabled": True}}
        )
        # Re-seed state because the first call cleared it
        ai._agent_proc = mock_proc
        ai._agent_start = time.monotonic() - 200.0
        ai._agent_timeout = 120
        ai._agent_tier = 1
        ai._agent_reasons = ["old-trigger"]

        with patch.object(ai, "_kill_overrun_agent", return_value=True) as m_kill:
            ai.invoke_agent(["new-trigger"], tier=1)
            assert m_kill.call_count == 1
