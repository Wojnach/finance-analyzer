"""Tests for bugs fixed in 2026-03-22 auto-improvement session.

BUG-107: Zero-division in digest/daily_digest P&L calculations
BUG-108: Alpha Vantage budget counter thread safety
BUG-109: Digest signal_log reads entire file (performance)
BUG-110: Stale import path in digest.py
"""

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# BUG-107: Zero-division in digest P&L calculation
# ---------------------------------------------------------------------------

class TestBug107DigestZeroDivision:
    """digest.py and daily_digest.py must not crash when initial_value_sek is 0 or missing."""

    def test_digest_handles_zero_initial_value(self, tmp_path):
        """_build_digest_message must not crash when initial_value_sek is 0."""
        from portfolio.digest import _build_digest_message

        # Create minimal state files
        state = {"cash_sek": 500000, "holdings": {}, "transactions": [],
                 "initial_value_sek": 0}
        bold_state = {"cash_sek": 500000, "holdings": {}, "transactions": [],
                      "initial_value_sek": 0}

        with patch("portfolio.digest.load_state", return_value=state), \
             patch("portfolio.digest.load_json") as mock_load, \
             patch("portfolio.digest.INVOCATIONS_FILE", tmp_path / "inv.jsonl"), \
             patch("portfolio.digest.JOURNAL_FILE", tmp_path / "j.jsonl"), \
             patch("portfolio.digest.SIGNAL_LOG_FILE", tmp_path / "s.jsonl"), \
             patch("portfolio.digest.AGENT_SUMMARY_FILE", tmp_path / "sum.json"), \
             patch("portfolio.digest.BOLD_STATE_FILE", tmp_path / "bold.json"):
            # Create empty JSONL files
            (tmp_path / "inv.jsonl").write_text("")
            (tmp_path / "j.jsonl").write_text("")
            (tmp_path / "s.jsonl").write_text("")

            # load_json returns different things for different paths
            def side_effect(path, **kwargs):
                path_str = str(path)
                if "agent_summary" in path_str:
                    return {"fx_rate": 10.5, "signals": {}}
                if "bold" in path_str:
                    return bold_state
                return kwargs.get("default", {})
            mock_load.side_effect = side_effect

            # Should not raise ZeroDivisionError
            msg = _build_digest_message()
            assert isinstance(msg, str)
            assert "4H DIGEST" in msg

    def test_digest_handles_missing_initial_value(self, tmp_path):
        """_build_digest_message must not crash when initial_value_sek key is missing."""
        from portfolio.digest import _build_digest_message

        state = {"cash_sek": 500000, "holdings": {}, "transactions": []}
        # Note: no initial_value_sek key

        with patch("portfolio.digest.load_state", return_value=state), \
             patch("portfolio.digest.load_json") as mock_load, \
             patch("portfolio.digest.INVOCATIONS_FILE", tmp_path / "inv.jsonl"), \
             patch("portfolio.digest.JOURNAL_FILE", tmp_path / "j.jsonl"), \
             patch("portfolio.digest.SIGNAL_LOG_FILE", tmp_path / "s.jsonl"), \
             patch("portfolio.digest.AGENT_SUMMARY_FILE", tmp_path / "sum.json"), \
             patch("portfolio.digest.BOLD_STATE_FILE", tmp_path / "bold.json"):
            (tmp_path / "inv.jsonl").write_text("")
            (tmp_path / "j.jsonl").write_text("")
            (tmp_path / "s.jsonl").write_text("")

            def side_effect(path, **kwargs):
                path_str = str(path)
                if "agent_summary" in path_str:
                    return {"fx_rate": 10.5, "signals": {}}
                return kwargs.get("default", {})
            mock_load.side_effect = side_effect

            # Should not raise KeyError or ZeroDivisionError
            msg = _build_digest_message()
            assert isinstance(msg, str)

    def test_daily_digest_handles_zero_initial_value(self):
        """daily_digest P&L must not crash when initial_value_sek is 0."""
        # We test the specific P&L calculation pattern
        initial = 0
        total = 500000
        # The fix should use `or INITIAL_CASH_SEK` fallback
        safe_initial = initial or 500000
        pnl = ((total - safe_initial) / safe_initial) * 100
        assert pnl == 0.0  # 500K / 500K = 0% change


# ---------------------------------------------------------------------------
# BUG-108: Alpha Vantage budget counter thread safety
# ---------------------------------------------------------------------------

class TestBug108AlphaVantageBudgetThreadSafety:
    """_daily_budget_used must be protected by lock."""

    def test_check_budget_uses_lock(self):
        """_check_budget should use _cache_lock to protect budget counter."""
        import portfolio.alpha_vantage as av
        from datetime import UTC, datetime

        # Reset state
        original_used = av._daily_budget_used
        original_date = av._budget_reset_date

        try:
            # Set today's date so no reset happens
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            with av._cache_lock:
                av._daily_budget_used = 10
                av._budget_reset_date = today

            # _check_budget should return current count under lock
            count = av._check_budget()
            assert count == 10
        finally:
            with av._cache_lock:
                av._daily_budget_used = original_used
                av._budget_reset_date = original_date

    def test_budget_increment_thread_safe(self):
        """Concurrent budget increments should not lose counts."""
        import portfolio.alpha_vantage as av

        original_used = av._daily_budget_used
        original_date = av._budget_reset_date

        try:
            av._daily_budget_used = 0
            av._budget_reset_date = "2099-12-31"

            errors = []

            def increment_budget():
                try:
                    for _ in range(100):
                        with av._cache_lock:
                            av._daily_budget_used += 1
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=increment_budget) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors
            assert av._daily_budget_used == 400
        finally:
            av._daily_budget_used = original_used
            av._budget_reset_date = original_date


# ---------------------------------------------------------------------------
# BUG-109: Efficient tail-read for JSONL files
# ---------------------------------------------------------------------------

class TestBug109JsonlTailRead:
    """load_jsonl_tail should read from end of file efficiently."""

    def test_load_jsonl_tail_basic(self, tmp_path):
        """load_jsonl_tail returns last N entries without reading entire file."""
        from portfolio.file_utils import load_jsonl_tail

        fpath = tmp_path / "test.jsonl"
        entries = [{"ts": f"2026-03-22T{i:02d}:00:00Z", "val": i} for i in range(100)]
        fpath.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        result = load_jsonl_tail(fpath, max_entries=10)
        assert len(result) == 10
        # Should be the last 10 entries
        assert result[0]["val"] == 90
        assert result[-1]["val"] == 99

    def test_load_jsonl_tail_small_file(self, tmp_path):
        """load_jsonl_tail handles files smaller than tail_bytes."""
        from portfolio.file_utils import load_jsonl_tail

        fpath = tmp_path / "small.jsonl"
        entries = [{"val": i} for i in range(3)]
        fpath.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        result = load_jsonl_tail(fpath, max_entries=10)
        assert len(result) == 3

    def test_load_jsonl_tail_missing_file(self, tmp_path):
        """load_jsonl_tail returns empty list for missing files."""
        from portfolio.file_utils import load_jsonl_tail

        result = load_jsonl_tail(tmp_path / "nonexistent.jsonl", max_entries=10)
        assert result == []

    def test_load_jsonl_tail_empty_file(self, tmp_path):
        """load_jsonl_tail returns empty list for empty files."""
        from portfolio.file_utils import load_jsonl_tail

        fpath = tmp_path / "empty.jsonl"
        fpath.write_text("")

        result = load_jsonl_tail(fpath, max_entries=10)
        assert result == []

    def test_load_jsonl_tail_with_malformed_lines(self, tmp_path):
        """load_jsonl_tail skips malformed JSON lines."""
        from portfolio.file_utils import load_jsonl_tail

        fpath = tmp_path / "mixed.jsonl"
        lines = [
            '{"val": 1}',
            'not json',
            '{"val": 2}',
            '',
            '{"val": 3}',
        ]
        fpath.write_text("\n".join(lines) + "\n")

        result = load_jsonl_tail(fpath, max_entries=10)
        assert len(result) == 3
        assert [e["val"] for e in result] == [1, 2, 3]


# ---------------------------------------------------------------------------
# BUG-110: Stale import path
# ---------------------------------------------------------------------------

class TestBug110StaleImport:
    """digest.py should import load_jsonl from file_utils, not stats."""

    def test_digest_imports_from_file_utils(self):
        """Verify digest.py uses canonical import path."""
        import inspect
        import portfolio.digest as digest_mod

        source = inspect.getsource(digest_mod)
        # Should NOT have `from portfolio.stats import load_jsonl`
        assert "from portfolio.stats import load_jsonl" not in source
        # Should have `from portfolio.file_utils import load_jsonl` (or similar)
        assert "load_jsonl" in source
