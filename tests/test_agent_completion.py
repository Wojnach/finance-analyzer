"""Tests for agent completion tracking and rolling stats.

Covers:
  - check_agent_completion: returns None when no agent running
  - check_agent_completion: returns None when agent still running
  - check_agent_completion: detects successful completion (exit 0 + journal + telegram)
  - check_agent_completion: detects failed completion (non-zero exit)
  - check_agent_completion: detects incomplete (exit 0, missing journal)
  - check_agent_completion: detects incomplete (exit 0, missing telegram)
  - check_agent_completion: cleans up globals after completion
  - check_agent_completion: logs completion to INVOCATIONS_FILE
  - check_agent_completion: handles log write failure gracefully
  - get_completion_stats: empty log returns zeroed stats
  - get_completion_stats: counts success/incomplete/failed correctly
  - get_completion_stats: respects time window
  - get_completion_stats: ignores non-completion statuses (invoked, skipped_gate)
  - get_completion_stats: handles corrupt JSONL entries
  - _last_jsonl_ts: returns None for missing file
  - _last_jsonl_ts: returns last timestamp from multi-entry file
"""

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import portfolio.agent_invocation as ai
from portfolio.agent_invocation import (
    check_agent_completion,
    get_completion_stats,
    _last_jsonl_ts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_agent_globals():
    """Reset module-level agent state before and after each test."""
    ai._agent_proc = None
    ai._agent_log = None
    ai._agent_start = 0
    ai._agent_timeout = 900
    ai._agent_tier = None
    ai._agent_reasons = None
    ai._journal_ts_before = None
    ai._telegram_ts_before = None
    yield
    ai._agent_proc = None
    ai._agent_log = None
    ai._agent_start = 0
    ai._agent_timeout = 900
    ai._agent_tier = None
    ai._agent_reasons = None
    ai._journal_ts_before = None
    ai._telegram_ts_before = None


@pytest.fixture
def tmp_invocations(tmp_path):
    """Provide a temp invocations file and patch INVOCATIONS_FILE."""
    inv_file = tmp_path / "invocations.jsonl"
    with patch.object(ai, "INVOCATIONS_FILE", inv_file):
        yield inv_file


@pytest.fixture
def tmp_journal(tmp_path):
    """Provide a temp journal file and patch JOURNAL_FILE."""
    journal_file = tmp_path / "layer2_journal.jsonl"
    with patch.object(ai, "JOURNAL_FILE", journal_file):
        yield journal_file


@pytest.fixture
def tmp_telegram(tmp_path):
    """Provide a temp telegram file and patch TELEGRAM_FILE."""
    tg_file = tmp_path / "telegram_messages.jsonl"
    with patch.object(ai, "TELEGRAM_FILE", tg_file):
        yield tg_file


# ===========================================================================
# _last_jsonl_ts
# ===========================================================================

class TestLastJsonlTs:

    def test_returns_none_for_missing_file(self, tmp_path):
        """Returns None when file does not exist."""
        result = _last_jsonl_ts(tmp_path / "nonexistent.jsonl")
        assert result is None

    def test_returns_none_for_empty_file(self, tmp_path):
        """Returns None when file is empty."""
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        assert _last_jsonl_ts(f) is None

    def test_returns_last_ts(self, tmp_path):
        """Returns the 'ts' from the last entry."""
        f = tmp_path / "test.jsonl"
        entries = [
            {"ts": "2026-03-10T10:00:00+00:00", "data": "first"},
            {"ts": "2026-03-10T11:00:00+00:00", "data": "second"},
            {"ts": "2026-03-10T12:00:00+00:00", "data": "third"},
        ]
        f.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        assert _last_jsonl_ts(f) == "2026-03-10T12:00:00+00:00"

    def test_skips_corrupt_lines(self, tmp_path):
        """Skips malformed JSON lines and returns the last valid ts."""
        f = tmp_path / "test.jsonl"
        f.write_text(
            '{"ts": "2026-03-10T10:00:00+00:00"}\n'
            'NOT JSON\n'
            '{"ts": "2026-03-10T11:00:00+00:00"}\n'
        )
        assert _last_jsonl_ts(f) == "2026-03-10T11:00:00+00:00"

    def test_returns_none_when_no_ts_field(self, tmp_path):
        """Returns None when entries have no 'ts' field."""
        f = tmp_path / "test.jsonl"
        f.write_text('{"status": "invoked"}\n')
        assert _last_jsonl_ts(f) is None


# ===========================================================================
# check_agent_completion — no agent / still running
# ===========================================================================

class TestCheckAgentNotRunning:

    def test_returns_none_when_no_agent(self):
        """Returns None when _agent_proc is None."""
        assert check_agent_completion() is None

    def test_returns_none_when_agent_still_running(self):
        """Returns None when agent process has not exited."""
        proc = MagicMock()
        proc.poll.return_value = None  # still running
        ai._agent_proc = proc
        ai._agent_start = time.time()

        assert check_agent_completion() is None
        # Process should still be tracked
        assert ai._agent_proc is proc


# ===========================================================================
# check_agent_completion — success
# ===========================================================================

class TestCheckAgentSuccess:

    def test_success_when_exit0_journal_telegram(self, tmp_invocations, tmp_journal, tmp_telegram):
        """Status is 'success' when exit=0, journal written, telegram sent."""
        # Set up pre-invocation state
        ai._journal_ts_before = "2026-03-10T09:00:00+00:00"
        ai._telegram_ts_before = "2026-03-10T09:00:00+00:00"

        # Write new journal and telegram entries (simulating agent output)
        tmp_journal.write_text(
            '{"ts": "2026-03-10T09:00:00+00:00"}\n'
            '{"ts": "2026-03-10T10:00:00+00:00"}\n'
        )
        tmp_telegram.write_text(
            '{"ts": "2026-03-10T09:00:00+00:00"}\n'
            '{"ts": "2026-03-10T10:05:00+00:00"}\n'
        )

        proc = MagicMock()
        proc.poll.return_value = 0
        ai._agent_proc = proc
        ai._agent_start = time.time() - 60
        ai._agent_tier = 2
        ai._agent_reasons = ["consensus BTC BUY"]

        result = check_agent_completion()

        assert result is not None
        assert result["status"] == "success"
        assert result["exit_code"] == 0
        assert result["journal_written"] is True
        assert result["telegram_sent"] is True
        assert result["tier"] == 2

    def test_duration_is_positive(self, tmp_invocations, tmp_journal, tmp_telegram):
        """Duration should reflect time since agent start."""
        tmp_journal.write_text('{"ts": "2026-03-10T10:00:00+00:00"}\n')
        tmp_telegram.write_text('{"ts": "2026-03-10T10:00:00+00:00"}\n')

        proc = MagicMock()
        proc.poll.return_value = 0
        ai._agent_proc = proc
        ai._agent_start = time.time() - 45.7
        ai._agent_tier = 1

        result = check_agent_completion()
        assert result["duration_s"] >= 45.0


# ===========================================================================
# check_agent_completion — failed
# ===========================================================================

class TestCheckAgentFailed:

    def test_failed_on_nonzero_exit(self, tmp_invocations, tmp_journal, tmp_telegram):
        """Status is 'failed' when exit code is non-zero."""
        # Even if journal/telegram exist, non-zero exit = failed
        tmp_journal.write_text('{"ts": "2026-03-10T10:00:00+00:00"}\n')
        tmp_telegram.write_text('{"ts": "2026-03-10T10:00:00+00:00"}\n')

        proc = MagicMock()
        proc.poll.return_value = 1
        ai._agent_proc = proc
        ai._agent_start = time.time() - 30
        ai._agent_tier = 3
        ai._agent_reasons = ["price move"]

        result = check_agent_completion()

        assert result["status"] == "failed"
        assert result["exit_code"] == 1

    def test_failed_with_negative_exit_code(self, tmp_invocations):
        """Status is 'failed' for signal-killed process (negative exit code)."""
        proc = MagicMock()
        proc.poll.return_value = -9  # killed by signal
        ai._agent_proc = proc
        ai._agent_start = time.time() - 10
        ai._agent_tier = 1

        result = check_agent_completion()

        assert result["status"] == "failed"
        assert result["exit_code"] == -9


# ===========================================================================
# check_agent_completion — incomplete
# ===========================================================================

class TestCheckAgentIncomplete:

    def test_incomplete_no_journal(self, tmp_invocations, tmp_journal, tmp_telegram):
        """Status is 'incomplete' when exit=0 but no journal written."""
        ai._journal_ts_before = "2026-03-10T09:00:00+00:00"
        ai._telegram_ts_before = "2026-03-10T09:00:00+00:00"

        # Journal has no new entry (same timestamp as before)
        tmp_journal.write_text('{"ts": "2026-03-10T09:00:00+00:00"}\n')
        # Telegram has a new entry
        tmp_telegram.write_text(
            '{"ts": "2026-03-10T09:00:00+00:00"}\n'
            '{"ts": "2026-03-10T10:00:00+00:00"}\n'
        )

        proc = MagicMock()
        proc.poll.return_value = 0
        ai._agent_proc = proc
        ai._agent_start = time.time() - 20
        ai._agent_tier = 2

        result = check_agent_completion()

        assert result["status"] == "incomplete"
        assert result["journal_written"] is False
        assert result["telegram_sent"] is True

    def test_incomplete_no_telegram(self, tmp_invocations, tmp_journal, tmp_telegram):
        """Status is 'incomplete' when exit=0 but no telegram sent."""
        ai._journal_ts_before = "2026-03-10T09:00:00+00:00"
        ai._telegram_ts_before = "2026-03-10T09:00:00+00:00"

        # Journal has a new entry
        tmp_journal.write_text(
            '{"ts": "2026-03-10T09:00:00+00:00"}\n'
            '{"ts": "2026-03-10T10:00:00+00:00"}\n'
        )
        # Telegram has no new entry
        tmp_telegram.write_text('{"ts": "2026-03-10T09:00:00+00:00"}\n')

        proc = MagicMock()
        proc.poll.return_value = 0
        ai._agent_proc = proc
        ai._agent_start = time.time() - 20
        ai._agent_tier = 1

        result = check_agent_completion()

        assert result["status"] == "incomplete"
        assert result["journal_written"] is True
        assert result["telegram_sent"] is False

    def test_incomplete_both_missing(self, tmp_invocations, tmp_journal, tmp_telegram):
        """Status is 'incomplete' when exit=0 but neither journal nor telegram written."""
        ai._journal_ts_before = None
        ai._telegram_ts_before = None

        # Neither file exists (empty tmp_path)
        # tmp_journal and tmp_telegram point to nonexistent files

        proc = MagicMock()
        proc.poll.return_value = 0
        ai._agent_proc = proc
        ai._agent_start = time.time() - 20
        ai._agent_tier = 3

        result = check_agent_completion()

        assert result["status"] == "incomplete"
        assert result["journal_written"] is False
        assert result["telegram_sent"] is False


# ===========================================================================
# check_agent_completion — cleanup
# ===========================================================================

class TestCheckAgentCleanup:

    def test_clears_globals_after_completion(self, tmp_invocations):
        """All agent globals are reset to None/0 after completion."""
        proc = MagicMock()
        proc.poll.return_value = 0
        log_fh = MagicMock()

        ai._agent_proc = proc
        ai._agent_log = log_fh
        ai._agent_start = time.time() - 10
        ai._agent_tier = 2
        ai._agent_reasons = ["test"]
        ai._journal_ts_before = "2026-03-10T09:00:00+00:00"
        ai._telegram_ts_before = "2026-03-10T09:00:00+00:00"

        check_agent_completion()

        assert ai._agent_proc is None
        assert ai._agent_log is None
        assert ai._agent_start == 0
        assert ai._agent_tier is None
        assert ai._agent_reasons is None
        assert ai._journal_ts_before is None
        assert ai._telegram_ts_before is None

    def test_closes_agent_log(self, tmp_invocations):
        """Agent log file handle is closed on completion."""
        proc = MagicMock()
        proc.poll.return_value = 0
        log_fh = MagicMock()

        ai._agent_proc = proc
        ai._agent_log = log_fh
        ai._agent_start = time.time() - 10

        check_agent_completion()

        log_fh.close.assert_called_once()

    def test_handles_log_close_error(self, tmp_invocations):
        """Completion still succeeds even if log close raises."""
        proc = MagicMock()
        proc.poll.return_value = 0
        log_fh = MagicMock()
        log_fh.close.side_effect = OSError("already closed")

        ai._agent_proc = proc
        ai._agent_log = log_fh
        ai._agent_start = time.time() - 10

        result = check_agent_completion()

        assert result is not None
        assert ai._agent_proc is None  # still cleaned up


# ===========================================================================
# check_agent_completion — logging
# ===========================================================================

class TestCheckAgentLogging:

    def test_logs_completion_to_invocations_file(self, tmp_invocations):
        """Completion is logged to INVOCATIONS_FILE."""
        proc = MagicMock()
        proc.poll.return_value = 0
        ai._agent_proc = proc
        ai._agent_start = time.time() - 30
        ai._agent_tier = 2
        ai._agent_reasons = ["consensus BTC BUY"]

        check_agent_completion()

        content = tmp_invocations.read_text().strip()
        entry = json.loads(content)
        assert entry["status"] == "incomplete"  # no journal/telegram files
        assert entry["exit_code"] == 0
        assert entry["tier"] == 2
        assert entry["reasons"] == ["consensus BTC BUY"]
        assert "duration_s" in entry
        assert "journal_written" in entry
        assert "telegram_sent" in entry

    def test_handles_invocation_log_write_failure(self):
        """Completion still returns result even if log write fails."""
        proc = MagicMock()
        proc.poll.return_value = 0
        ai._agent_proc = proc
        ai._agent_start = time.time() - 10
        ai._agent_tier = 1

        with patch.object(ai, "atomic_append_jsonl", side_effect=OSError("disk full")):
            result = check_agent_completion()

        assert result is not None
        assert result["status"] == "incomplete"


# ===========================================================================
# get_completion_stats — basic behavior
# ===========================================================================

class TestGetCompletionStats:

    def test_empty_log_returns_zeroed_stats(self, tmp_invocations):
        """Returns zeroed stats when invocations file does not exist."""
        stats = get_completion_stats(hours=24)

        assert stats["total"] == 0
        assert stats["success"] == 0
        assert stats["incomplete"] == 0
        assert stats["failed"] == 0
        assert stats["completion_rate"] == 0.0

    def test_counts_statuses_correctly(self, tmp_invocations):
        """Correctly counts success, incomplete, and failed entries."""
        now = datetime.now(timezone.utc)
        entries = [
            {"ts": (now - timedelta(hours=1)).isoformat(), "status": "success"},
            {"ts": (now - timedelta(hours=2)).isoformat(), "status": "success"},
            {"ts": (now - timedelta(hours=3)).isoformat(), "status": "incomplete"},
            {"ts": (now - timedelta(hours=4)).isoformat(), "status": "failed"},
            {"ts": (now - timedelta(hours=5)).isoformat(), "status": "success"},
        ]
        tmp_invocations.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n"
        )

        stats = get_completion_stats(hours=24)

        assert stats["total"] == 5
        assert stats["success"] == 3
        assert stats["incomplete"] == 1
        assert stats["failed"] == 1
        assert stats["completion_rate"] == 60.0

    def test_respects_time_window(self, tmp_invocations):
        """Only counts entries within the specified hour window."""
        now = datetime.now(timezone.utc)
        entries = [
            {"ts": (now - timedelta(hours=1)).isoformat(), "status": "success"},
            {"ts": (now - timedelta(hours=2)).isoformat(), "status": "success"},
            {"ts": (now - timedelta(hours=25)).isoformat(), "status": "failed"},  # outside 24h
            {"ts": (now - timedelta(hours=48)).isoformat(), "status": "failed"},  # outside 24h
        ]
        tmp_invocations.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n"
        )

        stats = get_completion_stats(hours=24)
        assert stats["total"] == 2
        assert stats["success"] == 2
        assert stats["failed"] == 0
        assert stats["completion_rate"] == 100.0

    def test_custom_hours_window(self, tmp_invocations):
        """Custom hours parameter is respected."""
        now = datetime.now(timezone.utc)
        entries = [
            {"ts": (now - timedelta(hours=1)).isoformat(), "status": "success"},
            {"ts": (now - timedelta(hours=3)).isoformat(), "status": "failed"},  # outside 2h
        ]
        tmp_invocations.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n"
        )

        stats = get_completion_stats(hours=2)
        assert stats["total"] == 1
        assert stats["success"] == 1
        assert stats["failed"] == 0

    def test_ignores_non_completion_statuses(self, tmp_invocations):
        """Entries with status 'invoked' or 'skipped_gate' are not counted."""
        now = datetime.now(timezone.utc)
        entries = [
            {"ts": (now - timedelta(hours=1)).isoformat(), "status": "invoked"},
            {"ts": (now - timedelta(hours=1)).isoformat(), "status": "skipped_gate"},
            {"ts": (now - timedelta(hours=1)).isoformat(), "status": "success"},
        ]
        tmp_invocations.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n"
        )

        stats = get_completion_stats(hours=24)
        assert stats["total"] == 1
        assert stats["success"] == 1

    def test_handles_corrupt_entries(self, tmp_invocations):
        """Corrupt JSONL lines are skipped gracefully."""
        now = datetime.now(timezone.utc)
        content = (
            '{"ts": "' + (now - timedelta(hours=1)).isoformat() + '", "status": "success"}\n'
            'NOT VALID JSON\n'
            '{"ts": "' + (now - timedelta(hours=2)).isoformat() + '", "status": "failed"}\n'
        )
        tmp_invocations.write_text(content)

        stats = get_completion_stats(hours=24)
        assert stats["total"] == 2
        assert stats["success"] == 1
        assert stats["failed"] == 1

    def test_handles_entries_without_ts(self, tmp_invocations):
        """Entries missing the 'ts' field are skipped."""
        now = datetime.now(timezone.utc)
        entries = [
            {"status": "success"},  # no ts
            {"ts": (now - timedelta(hours=1)).isoformat(), "status": "failed"},
        ]
        tmp_invocations.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n"
        )

        stats = get_completion_stats(hours=24)
        assert stats["total"] == 1
        assert stats["failed"] == 1

    def test_completion_rate_100_percent(self, tmp_invocations):
        """100% completion rate when all entries are success."""
        now = datetime.now(timezone.utc)
        entries = [
            {"ts": (now - timedelta(hours=i)).isoformat(), "status": "success"}
            for i in range(1, 6)
        ]
        tmp_invocations.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n"
        )

        stats = get_completion_stats(hours=24)
        assert stats["completion_rate"] == 100.0

    def test_completion_rate_0_percent(self, tmp_invocations):
        """0% completion rate when no entries are success."""
        now = datetime.now(timezone.utc)
        entries = [
            {"ts": (now - timedelta(hours=1)).isoformat(), "status": "failed"},
            {"ts": (now - timedelta(hours=2)).isoformat(), "status": "incomplete"},
        ]
        tmp_invocations.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n"
        )

        stats = get_completion_stats(hours=24)
        assert stats["total"] == 2
        assert stats["completion_rate"] == 0.0


# ===========================================================================
# Integration: invoke_agent stores tier and reasons for completion tracking
# ===========================================================================

class TestInvokeStoresTrackingState:

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_invoke_sets_agent_tier(self, mock_popen_cls, mock_which):
        """invoke_agent stores the tier in _agent_tier for completion tracking."""
        mock_popen_cls.return_value = MagicMock(pid=99)

        with patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", MagicMock()):
            invoke_agent = ai.invoke_agent
            invoke_agent(["consensus BTC BUY"], tier=2)

        assert ai._agent_tier == 2

    @patch("portfolio.agent_invocation.shutil.which", return_value="/usr/bin/claude")
    @patch("portfolio.agent_invocation.subprocess.Popen")
    def test_invoke_sets_agent_reasons(self, mock_popen_cls, mock_which):
        """invoke_agent stores the reasons in _agent_reasons for completion tracking."""
        mock_popen_cls.return_value = MagicMock(pid=99)

        with patch("portfolio.agent_invocation._load_config", return_value={}), \
             patch("portfolio.agent_invocation.send_or_store"), \
             patch("portfolio.agent_invocation.escape_markdown_v1", side_effect=lambda x: x), \
             patch("builtins.open", MagicMock()):
            ai.invoke_agent(["price move", "F&G crossed"], tier=1)

        assert ai._agent_reasons == ["price move", "F&G crossed"]
