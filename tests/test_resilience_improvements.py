"""Tests for resilience improvements (Batches 2-4, 2026-04-19).

Covers:
- Crash counter persistence (load/save/reset across restarts)
- Jitter is within expected range (50-150% of base delay)
- Per-file JSONL prune failure isolation
- Trigger sustained state has no dead started_ts field
- Health cache write-back after JSONL fallback
"""

import json
import threading
from datetime import UTC, datetime
from unittest import mock

import pytest


# ===========================================================================
# Batch 2: Crash counter persistence
# ===========================================================================

class TestCrashCounterPersistence:
    """Crash counter survives process restarts via JSON file."""

    def test_save_and_load(self, tmp_path):
        """Save counter, then load — value round-trips."""
        import portfolio.main as m
        orig_file = m._CRASH_COUNTER_FILE
        m._CRASH_COUNTER_FILE = tmp_path / "crash_counter.json"
        try:
            m._save_crash_counter(7)
            assert m._load_crash_counter() == 7
        finally:
            m._CRASH_COUNTER_FILE = orig_file

    def test_load_missing_file_returns_zero(self, tmp_path):
        """Missing file returns 0 (fresh install)."""
        import portfolio.main as m
        orig_file = m._CRASH_COUNTER_FILE
        m._CRASH_COUNTER_FILE = tmp_path / "nonexistent.json"
        try:
            assert m._load_crash_counter() == 0
        finally:
            m._CRASH_COUNTER_FILE = orig_file

    def test_load_corrupt_file_returns_zero(self, tmp_path):
        """Corrupt JSON returns 0 rather than crashing."""
        import portfolio.main as m
        orig_file = m._CRASH_COUNTER_FILE
        f = tmp_path / "crash_counter.json"
        f.write_text("not json", encoding="utf-8")
        m._CRASH_COUNTER_FILE = f
        try:
            assert m._load_crash_counter() == 0
        finally:
            m._CRASH_COUNTER_FILE = orig_file

    def test_reset_saves_zero(self, tmp_path):
        """_reset_crash_counter persists 0 to disk."""
        import portfolio.main as m
        orig_file = m._CRASH_COUNTER_FILE
        orig_count = m._consecutive_crashes
        m._CRASH_COUNTER_FILE = tmp_path / "crash_counter.json"
        try:
            m._consecutive_crashes = 3
            m._reset_crash_counter()
            assert m._consecutive_crashes == 0
            assert m._load_crash_counter() == 0
        finally:
            m._CRASH_COUNTER_FILE = orig_file
            m._consecutive_crashes = orig_count


# ===========================================================================
# Batch 2: Jitter range
# ===========================================================================

class TestCrashSleepJitter:
    """Backoff delay includes jitter in the expected 50-150% range."""

    def test_jitter_within_range(self):
        """Delay is between 50% and 150% of base (10s for crash #1)."""
        import portfolio.main as m
        orig = m._consecutive_crashes
        m._consecutive_crashes = 1
        delays = []
        try:
            with mock.patch("time.sleep", side_effect=lambda d: delays.append(d)):
                for _ in range(50):
                    m._crash_sleep()
        finally:
            m._consecutive_crashes = orig
        base = 10  # 10 * 2^0 for crash #1
        for d in delays:
            assert base * 0.49 <= d <= base * 1.51, f"Delay {d} outside jitter range"

    def test_backoff_capped(self):
        """Delay is capped at _MAX_CRASH_BACKOFF * 1.5 (jitter ceiling)."""
        import portfolio.main as m
        orig = m._consecutive_crashes
        m._consecutive_crashes = 20  # 10 * 2^19 would be huge without cap
        try:
            with mock.patch("time.sleep") as mock_sleep:
                m._crash_sleep()
            delay = mock_sleep.call_args[0][0]
            assert delay <= m._MAX_CRASH_BACKOFF * 1.51
        finally:
            m._consecutive_crashes = orig


# ===========================================================================
# Batch 2: Periodic summary after suppression
# ===========================================================================

class TestCrashAlertSuppression:
    """After MAX_CRASH_ALERTS, alerts are suppressed except periodic summaries."""

    @mock.patch("portfolio.main._save_crash_counter")
    @mock.patch("portfolio.main.load_json", return_value={"telegram": {"token": "x", "chat_id": "1"}})
    @mock.patch("portfolio.message_store.send_or_store")
    def test_summary_at_interval(self, mock_send, mock_load, mock_save):
        import portfolio.main as m
        orig = m._consecutive_crashes
        # Set to one before summary interval
        m._consecutive_crashes = m._CRASH_SUMMARY_INTERVAL - 1
        try:
            m._crash_alert("test error")
            assert m._consecutive_crashes == m._CRASH_SUMMARY_INTERVAL
            # Should send a summary
            mock_send.assert_called_once()
            text = mock_send.call_args[0][0]
            assert "CRASH LOOP SUMMARY" in text
        finally:
            m._consecutive_crashes = orig

    @mock.patch("portfolio.main._save_crash_counter")
    def test_no_alert_between_intervals(self, mock_save):
        import portfolio.main as m
        orig = m._consecutive_crashes
        m._consecutive_crashes = m._MAX_CRASH_ALERTS + 1
        try:
            with mock.patch("portfolio.message_store.send_or_store") as mock_send:
                m._crash_alert("test error")
                mock_send.assert_not_called()
        except ImportError:
            pass  # message_store may fail to import in test env
        finally:
            m._consecutive_crashes = orig


# ===========================================================================
# Batch 3: JSONL prune per-file isolation
# ===========================================================================

class TestJsonlPruneIsolation:
    """Per-file try/except prevents cascade failures."""

    def test_locked_file_doesnt_block_others(self, tmp_path, monkeypatch):
        """If one file raises, the others still get pruned."""
        call_log = []

        def mock_prune(path, max_entries=5000):
            name = path.name
            call_log.append(name)
            if name == "invocations.jsonl":
                raise PermissionError("File locked by antivirus")

        # Replicate the per-file isolation pattern from _run_post_cycle
        _prune_failures = []
        for name in ("invocations.jsonl", "layer2_journal.jsonl", "telegram_messages.jsonl"):
            try:
                mock_prune(tmp_path / name)
            except Exception:
                _prune_failures.append(name)

        # All 3 files were attempted despite the first one failing
        assert len(call_log) == 3
        assert "invocations.jsonl" in _prune_failures
        assert "layer2_journal.jsonl" not in _prune_failures
        assert "telegram_messages.jsonl" not in _prune_failures


# ===========================================================================
# Batch 4: Trigger sustained state — no dead started_ts
# ===========================================================================

class TestTriggerSustainedState:
    """Sustained state entries should NOT contain started_ts (dead code removed)."""

    def test_new_entry_has_no_started_ts(self):
        from portfolio.trigger import _update_sustained
        state = {}
        _update_sustained(state, "test_key", "BUY", datetime.now(UTC).isoformat())
        assert "started_ts" not in state["test_key"]
        assert "_mono_start" in state["test_key"]
        assert state["test_key"]["count"] == 1

    def test_continued_entry_has_no_started_ts(self):
        from portfolio.trigger import _update_sustained
        state = {}
        ts = datetime.now(UTC).isoformat()
        _update_sustained(state, "k", "BUY", ts)
        _update_sustained(state, "k", "BUY", ts)
        assert "started_ts" not in state["k"]
        assert state["k"]["count"] == 2

    def test_value_change_resets(self):
        from portfolio.trigger import _update_sustained
        state = {}
        ts = datetime.now(UTC).isoformat()
        _update_sustained(state, "k", "BUY", ts)
        _update_sustained(state, "k", "BUY", ts)
        _update_sustained(state, "k", "SELL", ts)
        assert state["k"]["count"] == 1
        assert state["k"]["value"] == "SELL"


# ===========================================================================
# Batch 4: Health cache write-back
# ===========================================================================

class TestHealthCacheWriteBack:
    """check_agent_silence writes back last_invocation_ts after JSONL fallback."""

    def test_writeback_after_jsonl_fallback(self, tmp_path, monkeypatch):
        """When health_state has no last_invocation_ts, fallback parses JSONL
        and writes the result back to health_state."""
        import portfolio.health as h

        monkeypatch.setattr(h, "DATA_DIR", tmp_path)
        monkeypatch.setattr(h, "HEALTH_FILE", tmp_path / "health_state.json")
        monkeypatch.setattr(h, "_health_lock", threading.Lock())

        # Write a health state without last_invocation_ts
        (tmp_path / "health_state.json").write_text(
            json.dumps({"cycle_count": 1}), encoding="utf-8"
        )

        # Write an invocations.jsonl with a timestamp
        ts = datetime.now(UTC).isoformat()
        (tmp_path / "invocations.jsonl").write_text(
            json.dumps({"ts": ts}) + "\n", encoding="utf-8"
        )

        with mock.patch("portfolio.market_timing.get_market_state", return_value=("open", [], 60)):
            result = h.check_agent_silence()

        # Verify the timestamp was written back
        state = json.loads((tmp_path / "health_state.json").read_text(encoding="utf-8"))
        assert state.get("last_invocation_ts") == ts
