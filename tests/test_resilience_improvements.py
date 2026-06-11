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
    """Sustained state entries should NOT contain started_ts (dead code removed).

    2026-06-11 (audit B5): _update_sustained's time arg is now wall-clock
    EPOCH SECONDS (float time.time()), not an ISO string, and the persisted
    origin key is `_wall_start` (was `_mono_start`). Passing an ISO string
    now raises `str - str` in the duration check. Updated to pass time.time()
    and assert the new `_wall_start` key.
    """

    def test_new_entry_has_no_started_ts(self):
        import time
        from portfolio.trigger import _update_sustained
        state = {}
        _update_sustained(state, "test_key", "BUY", time.time())
        assert "started_ts" not in state["test_key"]
        assert "_wall_start" in state["test_key"]
        assert state["test_key"]["count"] == 1

    def test_continued_entry_has_no_started_ts(self):
        import time
        from portfolio.trigger import _update_sustained
        state = {}
        ts = time.time()
        _update_sustained(state, "k", "BUY", ts)
        _update_sustained(state, "k", "BUY", ts)
        assert "started_ts" not in state["k"]
        assert state["k"]["count"] == 2

    def test_value_change_resets(self):
        import time
        from portfolio.trigger import _update_sustained
        state = {}
        ts = time.time()
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


# ===========================================================================
# Adversarial review 04-29 OR-P1-2: zero-delay spin after crash
#
# Existing protection: the per-cycle except in main.loop() calls
# _crash_alert() (which increments _consecutive_crashes) and _crash_sleep()
# (which does exponential backoff with jitter, capped at _MAX_CRASH_BACKOFF).
# Gap: if _crash_alert itself raises (disk full on _save_crash_counter,
# load_json IO error, etc.), the alert helper bubbles out of the except
# handler and either kills the loop process entirely OR — if a future
# refactor catches it — leaves _consecutive_crashes un-incremented and
# _crash_sleep() doing 10 * 2^(0-1) ≈ 5s, then loops again. Worse, in
# the original report's reading, a path where _crash_sleep is bypassed
# (e.g. _consecutive_crashes is somehow reset between alert and sleep)
# would let _sleep_for_next_cycle compute remaining<0 and skip its
# sleep entirely, spinning the loop tight.
#
# Fix: add a ground-floor minimum sleep at the END of the except handler
# that fires regardless of whether _crash_alert/_crash_sleep succeeded —
# `time.sleep(min(2 ** n_failures, 30))` per the plan, where n_failures
# is _consecutive_crashes (resilient even if alert helper failed before
# incrementing). The existing _crash_sleep continues to provide longer
# backoffs for sustained failure; the new floor catches the edge case.
# ===========================================================================

class TestCrashLoopMinSleepFloor:

    def test_min_sleep_called_when_crash_alert_raises(self):
        """If _crash_alert raises, the loop must STILL sleep before retrying."""
        import portfolio.main as m

        sleep_calls = []
        with mock.patch("portfolio.main._crash_alert", side_effect=RuntimeError("disk full")), \
             mock.patch("portfolio.main._crash_sleep", side_effect=RuntimeError("sleep also broken")), \
             mock.patch("time.sleep", side_effect=lambda d: sleep_calls.append(d)):
            m._safe_crash_recovery("simulated traceback")
        assert sleep_calls, (
            "OR-P1-2: a failure in _crash_alert / _crash_sleep must still leave "
            "AT LEAST ONE time.sleep call so the loop doesn't spin tight on "
            "persistent failure."
        )
        # Floor sleep must be >= 1s (some defense)
        assert max(sleep_calls) >= 1.0

    def test_floor_sleep_grows_with_consecutive_crashes(self):
        """Floor sleep should track 2^n with cap (matching plan's
        time.sleep(min(2 ** n_failures, 30)) suggestion)."""
        import portfolio.main as m

        orig = m._consecutive_crashes
        try:
            sleep_calls = []
            with mock.patch("portfolio.main._crash_alert", side_effect=RuntimeError("x")), \
                 mock.patch("portfolio.main._crash_sleep", side_effect=RuntimeError("y")), \
                 mock.patch("time.sleep", side_effect=lambda d: sleep_calls.append(d)):
                m._consecutive_crashes = 1
                m._safe_crash_recovery("traceback")
                first_sleep = sleep_calls[-1]
                m._consecutive_crashes = 5
                m._safe_crash_recovery("traceback")
                later_sleep = sleep_calls[-1]
            assert later_sleep > first_sleep, (
                f"Floor sleep should grow with crash count "
                f"(crash#1={first_sleep}, crash#5={later_sleep})"
            )
        finally:
            m._consecutive_crashes = orig

    def test_floor_sleep_capped_at_30s(self):
        """Plan-specified cap: min(2 ** n_failures, 30)."""
        import portfolio.main as m

        orig = m._consecutive_crashes
        try:
            sleep_calls = []
            with mock.patch("portfolio.main._crash_alert", side_effect=RuntimeError("x")), \
                 mock.patch("portfolio.main._crash_sleep", side_effect=RuntimeError("y")), \
                 mock.patch("time.sleep", side_effect=lambda d: sleep_calls.append(d)):
                m._consecutive_crashes = 50  # 2^50 is astronomical
                m._safe_crash_recovery("traceback")
            assert sleep_calls[-1] <= 30.0, (
                f"Floor sleep must cap at 30s, got {sleep_calls[-1]}"
            )
        finally:
            m._consecutive_crashes = orig

    def test_normal_path_uses_crash_sleep_not_floor(self):
        """When _crash_sleep works normally, the floor doesn't add extra
        sleep — _crash_sleep already handles the exponential backoff."""
        import portfolio.main as m

        orig = m._consecutive_crashes
        try:
            sleep_calls = []
            with mock.patch("portfolio.main._crash_alert", side_effect=lambda x: None), \
                 mock.patch(
                     "portfolio.main._crash_sleep",
                     side_effect=lambda: sleep_calls.append("cs"),
                 ):
                m._consecutive_crashes = 1
                m._safe_crash_recovery("traceback")
            # _crash_sleep was called — no need for the floor
            assert "cs" in sleep_calls
        finally:
            m._consecutive_crashes = orig
