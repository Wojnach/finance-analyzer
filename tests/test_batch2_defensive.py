"""Tests for Batch 2 defensive fixes (BUG-24, BUG-25, BUG-26, BUG-27).

BUG-24: news_event _fetch_headlines guard against None ticker
BUG-25: load_json propagates OSError (permission denied, disk full)
BUG-26: heartbeat written after initial run (tested via code inspection)
BUG-27: redundant pass removed from trigger.py (code cleanup, no behavioral test)
"""

from pathlib import Path
from unittest.mock import patch

import pytest


class TestBug24NewsEventNoneTicker:
    """BUG-24: _fetch_headlines must not crash on None/empty ticker."""

    def test_fetch_headlines_none_ticker(self):
        from portfolio.signals.news_event import _fetch_headlines
        result = _fetch_headlines(None, {})
        assert result == []

    def test_fetch_headlines_empty_string(self):
        from portfolio.signals.news_event import _fetch_headlines
        result = _fetch_headlines("", {})
        assert result == []

    def test_compute_signal_none_ticker_in_context(self):
        """Full signal path: context with None ticker returns HOLD."""
        import pandas as pd

        from portfolio.signals.news_event import compute_news_event_signal
        df = pd.DataFrame({"close": [100] * 30, "volume": [1000] * 30,
                          "high": [101] * 30, "low": [99] * 30, "open": [100] * 30})
        result = compute_news_event_signal(df, context={"ticker": None, "config": {}})
        assert result["action"] == "HOLD"

    def test_compute_signal_no_context(self):
        """Signal with no context returns HOLD."""
        import pandas as pd

        from portfolio.signals.news_event import compute_news_event_signal
        df = pd.DataFrame({"close": [100] * 30, "volume": [1000] * 30,
                          "high": [101] * 30, "low": [99] * 30, "open": [100] * 30})
        result = compute_news_event_signal(df)
        assert result["action"] == "HOLD"


class TestBug25LoadJsonOSError:
    """BUG-25 / BUG-139: load_json returns default on OSError (Windows compat).
    require_json is the function that propagates OSError for critical files."""

    def test_load_json_permission_error_returns_default(self, tmp_path):
        """BUG-139: PermissionError (file locked by antivirus) → returns default."""
        from portfolio.file_utils import load_json
        path = tmp_path / "test.json"
        path.write_text('{"key": "value"}', encoding="utf-8")

        with patch.object(Path, "read_text", side_effect=PermissionError("Access denied")):
            result = load_json(path, default={"fallback": True})
        assert result == {"fallback": True}

    def test_load_json_oserror_returns_default(self, tmp_path):
        """BUG-139: OSError → returns default (graceful degradation on Windows)."""
        from portfolio.file_utils import load_json
        path = tmp_path / "test.json"
        path.write_text('{"key": "value"}', encoding="utf-8")

        with patch.object(Path, "read_text", side_effect=OSError("Disk full")):
            result = load_json(path, default={"fallback": True})
        assert result == {"fallback": True}

    def test_load_json_still_handles_missing_file(self, tmp_path):
        from portfolio.file_utils import load_json
        path = tmp_path / "nonexistent.json"
        result = load_json(path, default={"fallback": True})
        assert result == {"fallback": True}

    def test_load_json_still_handles_bad_json(self, tmp_path):
        from portfolio.file_utils import load_json
        path = tmp_path / "bad.json"
        path.write_text("{invalid json", encoding="utf-8")
        result = load_json(path, default={"fallback": True})
        assert result == {"fallback": True}

    def test_load_json_valid_file(self, tmp_path):
        from portfolio.file_utils import load_json
        path = tmp_path / "good.json"
        path.write_text('{"hello": "world"}', encoding="utf-8")
        result = load_json(path)
        assert result == {"hello": "world"}


class TestBug26HeartbeatAfterInitialRun:
    """BUG-26: Verify heartbeat is written after initial run() in main.py."""

    def test_heartbeat_code_present_after_initial_run(self):
        """Verify the fix is in place by checking the source code."""
        import inspect

        from portfolio import main
        source = inspect.getsource(main.loop)
        # Find heartbeat write near initial run
        lines = source.split("\n")
        found_initial_run = False
        found_heartbeat_after = False
        for _i, line in enumerate(lines):
            if "run(force_report=True)" in line:
                found_initial_run = True
            if found_initial_run and "heartbeat" in line and "write_text" in line:
                found_heartbeat_after = True
                break
            if found_initial_run and "while True" in line:
                # Went past initial run block without finding heartbeat
                break
        assert found_heartbeat_after, "heartbeat.txt write must be in the initial run() block"


class TestBug27TriggerPassRemoved:
    """BUG-27: Verify redundant pass is removed from trigger.py."""

    def test_no_redundant_pass(self):
        """Check that there's no 'pass' after logger.warning in _check_recent_trade."""
        import inspect

        from portfolio import trigger
        source = inspect.getsource(trigger._check_recent_trade)
        lines = source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "pass":
                # Check if previous non-empty line is a logger call
                for j in range(i - 1, -1, -1):
                    prev = lines[j].strip()
                    if prev:
                        assert "logger" not in prev, \
                            f"Redundant 'pass' after logger call on line {i}"
                        break
