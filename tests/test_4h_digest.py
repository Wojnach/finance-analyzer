"""Tests for portfolio.digest — the 4-hour digest builder/sender.

Covers BUG-10, BUG-11, BUG-13, BUG-14 fixes:
- _get_last_digest_time() with missing/corrupt trigger_state.json
- _build_digest_message() with missing "ts" keys in JSONL entries
- _build_digest_message() with missing/corrupt agent_summary.json
- _build_digest_message() with missing/corrupt bold state file
- _maybe_send_digest() end-to-end
"""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path, entries):
    """Write a list of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _hours_ago_iso(h):
    return (datetime.now(timezone.utc) - timedelta(hours=h)).isoformat()


# ---------------------------------------------------------------------------
# _get_last_digest_time
# ---------------------------------------------------------------------------

class TestGetLastDigestTime:
    """BUG-14: narrowed exception types + load_json usage."""

    def test_returns_zero_when_file_missing(self, tmp_path):
        with patch("portfolio.digest.DATA_DIR", tmp_path):
            from portfolio.digest import _get_last_digest_time
            assert _get_last_digest_time() == 0

    def test_returns_stored_time(self, tmp_path):
        ts = time.time() - 3600
        (tmp_path / "trigger_state.json").write_text(
            json.dumps({"last_digest_time": ts})
        )
        with patch("portfolio.digest.DATA_DIR", tmp_path):
            from portfolio.digest import _get_last_digest_time
            assert _get_last_digest_time() == ts

    def test_returns_zero_when_file_corrupt(self, tmp_path):
        (tmp_path / "trigger_state.json").write_text("{{{invalid json")
        with patch("portfolio.digest.DATA_DIR", tmp_path):
            from portfolio.digest import _get_last_digest_time
            assert _get_last_digest_time() == 0

    def test_returns_zero_when_key_missing(self, tmp_path):
        (tmp_path / "trigger_state.json").write_text(json.dumps({"other_key": 1}))
        with patch("portfolio.digest.DATA_DIR", tmp_path):
            from portfolio.digest import _get_last_digest_time
            assert _get_last_digest_time() == 0


# ---------------------------------------------------------------------------
# _build_digest_message — JSONL robustness
# ---------------------------------------------------------------------------

class TestBuildDigestJSONLRobustness:
    """BUG-10: safe .get() for ts in JSONL entries."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        """Set up temp data directory with minimal valid files."""
        self.tmp = tmp_path
        self.data = tmp_path / "data"
        self.data.mkdir()

        # Minimal portfolio state
        self.patient_state = {
            "cash_sek": 500000,
            "initial_value_sek": 500000,
            "total_fees_sek": 0,
            "holdings": {},
            "transactions": [],
        }
        (self.data / "portfolio_state.json").write_text(
            json.dumps(self.patient_state)
        )

        # Empty signal log, journal, invocations by default
        _write_jsonl(self.data / "invocations.jsonl", [])
        _write_jsonl(self.data / "layer2_journal.jsonl", [])
        _write_jsonl(self.data / "signal_log.jsonl", [])

        # Minimal agent summary
        (self.data / "agent_summary.json").write_text(
            json.dumps({"fx_rate": 10.5, "signals": {}})
        )

    def _patch_all(self):
        """Return a context manager that patches all digest paths."""
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch("portfolio.digest.DATA_DIR", self.data))
        stack.enter_context(patch("portfolio.digest.INVOCATIONS_FILE", self.data / "invocations.jsonl"))
        stack.enter_context(patch("portfolio.digest.JOURNAL_FILE", self.data / "layer2_journal.jsonl"))
        stack.enter_context(patch("portfolio.digest.SIGNAL_LOG_FILE", self.data / "signal_log.jsonl"))
        stack.enter_context(patch("portfolio.digest.AGENT_SUMMARY_FILE", self.data / "agent_summary.json"))
        stack.enter_context(patch("portfolio.digest.BOLD_STATE_FILE", self.data / "portfolio_state_bold.json"))
        stack.enter_context(patch("portfolio.portfolio_mgr.STATE_FILE", self.data / "portfolio_state.json"))
        return stack

    def test_invocations_with_missing_ts(self):
        """Entries missing 'ts' key should be skipped, not crash."""
        _write_jsonl(self.data / "invocations.jsonl", [
            {"ts": _now_iso(), "status": "invoked", "reasons": ["consensus"]},
            {"status": "invoked", "reasons": ["price_move"]},  # no ts
            {"ts": "", "status": "invoked", "reasons": ["other"]},  # empty ts
        ])
        with self._patch_all():
            from portfolio.digest import _build_digest_message
            msg = _build_digest_message()
        assert "*4H DIGEST*" in msg
        # Only the first entry (valid ts) should be counted
        assert "Invoked:" in msg

    def test_invocations_with_invalid_ts(self):
        """Entries with unparseable ts should be skipped."""
        _write_jsonl(self.data / "invocations.jsonl", [
            {"ts": "not-a-date", "status": "invoked", "reasons": ["consensus"]},
            {"ts": 12345, "status": "invoked", "reasons": ["other"]},
        ])
        with self._patch_all():
            from portfolio.digest import _build_digest_message
            msg = _build_digest_message()
        assert "*4H DIGEST*" in msg

    def test_journal_with_missing_ts(self):
        """Journal entries missing 'ts' key should be skipped."""
        _write_jsonl(self.data / "layer2_journal.jsonl", [
            {"ts": _now_iso(), "decisions": {"patient": {"action": "HOLD"}, "bold": {"action": "HOLD"}}},
            {"decisions": {"patient": {"action": "BUY"}, "bold": {"action": "SELL"}}},  # no ts
        ])
        with self._patch_all():
            from portfolio.digest import _build_digest_message
            msg = _build_digest_message()
        assert "*4H DIGEST*" in msg

    def test_signal_log_with_missing_ts_uses_fallback(self):
        """Signal log entries use .get('ts', '2000-01-01') which parses but is old."""
        _write_jsonl(self.data / "signal_log.jsonl", [
            {"tickers": {"BTC": {"consensus": "BUY"}}},  # no ts → falls back to 2000
            {"ts": _now_iso(), "tickers": {"ETH": {"consensus": "SELL"}}},
        ])
        with self._patch_all():
            from portfolio.digest import _build_digest_message
            msg = _build_digest_message()
        assert "*4H DIGEST*" in msg

    def test_all_empty_files_produces_valid_message(self):
        """Completely empty JSONL files should produce a valid digest."""
        with self._patch_all():
            from portfolio.digest import _build_digest_message
            msg = _build_digest_message()
        assert "*4H DIGEST*" in msg
        assert "Patient:" in msg


# ---------------------------------------------------------------------------
# _build_digest_message — agent_summary robustness
# ---------------------------------------------------------------------------

class TestBuildDigestAgentSummary:
    """BUG-11: use load_json with fallback for agent_summary.json."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.data = tmp_path / "data"
        self.data.mkdir()
        state = {
            "cash_sek": 500000, "initial_value_sek": 500000,
            "total_fees_sek": 0, "holdings": {}, "transactions": [],
        }
        (self.data / "portfolio_state.json").write_text(json.dumps(state))
        _write_jsonl(self.data / "invocations.jsonl", [])
        _write_jsonl(self.data / "layer2_journal.jsonl", [])
        _write_jsonl(self.data / "signal_log.jsonl", [])

    def _patch_all(self):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch("portfolio.digest.DATA_DIR", self.data))
        stack.enter_context(patch("portfolio.digest.INVOCATIONS_FILE", self.data / "invocations.jsonl"))
        stack.enter_context(patch("portfolio.digest.JOURNAL_FILE", self.data / "layer2_journal.jsonl"))
        stack.enter_context(patch("portfolio.digest.SIGNAL_LOG_FILE", self.data / "signal_log.jsonl"))
        stack.enter_context(patch("portfolio.digest.AGENT_SUMMARY_FILE", self.data / "agent_summary.json"))
        stack.enter_context(patch("portfolio.digest.BOLD_STATE_FILE", self.data / "portfolio_state_bold.json"))
        stack.enter_context(patch("portfolio.portfolio_mgr.STATE_FILE", self.data / "portfolio_state.json"))
        return stack

    def test_missing_agent_summary(self):
        """Missing agent_summary.json should use defaults, not crash."""
        # Don't create agent_summary.json at all
        with self._patch_all():
            from portfolio.digest import _build_digest_message
            msg = _build_digest_message()
        assert "*4H DIGEST*" in msg

    def test_corrupt_agent_summary(self):
        """Corrupt agent_summary.json should use defaults, not crash."""
        (self.data / "agent_summary.json").write_text("NOT JSON {{{{")
        with self._patch_all():
            from portfolio.digest import _build_digest_message
            msg = _build_digest_message()
        assert "*4H DIGEST*" in msg

    def test_empty_agent_summary(self):
        """Empty agent_summary.json should use defaults."""
        (self.data / "agent_summary.json").write_text("{}")
        with self._patch_all():
            from portfolio.digest import _build_digest_message
            msg = _build_digest_message()
        assert "*4H DIGEST*" in msg


# ---------------------------------------------------------------------------
# _build_digest_message — bold state robustness
# ---------------------------------------------------------------------------

class TestBuildDigestBoldState:
    """BUG-13: safe bold state file read with load_json + try/except."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.data = tmp_path / "data"
        self.data.mkdir()
        state = {
            "cash_sek": 500000, "initial_value_sek": 500000,
            "total_fees_sek": 0, "holdings": {}, "transactions": [],
        }
        (self.data / "portfolio_state.json").write_text(json.dumps(state))
        (self.data / "agent_summary.json").write_text(
            json.dumps({"fx_rate": 10.5, "signals": {}})
        )
        _write_jsonl(self.data / "invocations.jsonl", [])
        _write_jsonl(self.data / "layer2_journal.jsonl", [])
        _write_jsonl(self.data / "signal_log.jsonl", [])

    def _patch_all(self):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch("portfolio.digest.DATA_DIR", self.data))
        stack.enter_context(patch("portfolio.digest.INVOCATIONS_FILE", self.data / "invocations.jsonl"))
        stack.enter_context(patch("portfolio.digest.JOURNAL_FILE", self.data / "layer2_journal.jsonl"))
        stack.enter_context(patch("portfolio.digest.SIGNAL_LOG_FILE", self.data / "signal_log.jsonl"))
        stack.enter_context(patch("portfolio.digest.AGENT_SUMMARY_FILE", self.data / "agent_summary.json"))
        stack.enter_context(patch("portfolio.digest.BOLD_STATE_FILE", self.data / "portfolio_state_bold.json"))
        stack.enter_context(patch("portfolio.portfolio_mgr.STATE_FILE", self.data / "portfolio_state.json"))
        return stack

    def test_missing_bold_state(self):
        """Missing bold state file should be skipped gracefully."""
        with self._patch_all():
            from portfolio.digest import _build_digest_message
            msg = _build_digest_message()
        assert "*4H DIGEST*" in msg
        assert "Bold:" not in msg

    def test_corrupt_bold_state(self):
        """Corrupt bold state file should be skipped gracefully."""
        (self.data / "portfolio_state_bold.json").write_text("CORRUPT{{{{")
        with self._patch_all():
            from portfolio.digest import _build_digest_message
            msg = _build_digest_message()
        assert "*4H DIGEST*" in msg
        # Bold section should be skipped (load_json returns {}, no initial_value_sek)
        assert "Bold:" not in msg

    def test_valid_bold_state(self):
        """Valid bold state should produce Bold line in digest."""
        bold = {
            "cash_sek": 464535, "initial_value_sek": 500000,
            "total_fees_sek": 200, "holdings": {}, "transactions": [],
        }
        (self.data / "portfolio_state_bold.json").write_text(json.dumps(bold))
        with self._patch_all():
            from portfolio.digest import _build_digest_message
            msg = _build_digest_message()
        assert "Bold:" in msg

    def test_bold_state_missing_initial_value(self):
        """Bold state without initial_value_sek should be skipped."""
        bold = {"cash_sek": 464535, "holdings": {}}
        (self.data / "portfolio_state_bold.json").write_text(json.dumps(bold))
        with self._patch_all():
            from portfolio.digest import _build_digest_message
            msg = _build_digest_message()
        assert "Bold:" not in msg


# ---------------------------------------------------------------------------
# _maybe_send_digest — integration
# ---------------------------------------------------------------------------

class TestMaybeSendDigest:
    """Integration test for the digest send flow."""

    def test_skips_when_recently_sent(self):
        with patch("portfolio.digest._get_last_digest_time", return_value=time.time() - 100):
            from portfolio.digest import _maybe_send_digest
            # Should return without calling _build_digest_message
            with patch("portfolio.digest._build_digest_message") as mock_build:
                _maybe_send_digest({})
                mock_build.assert_not_called()

    def test_sends_when_interval_elapsed(self):
        with patch("portfolio.digest._get_last_digest_time", return_value=time.time() - 20000), \
             patch("portfolio.digest._build_digest_message", return_value="*4H DIGEST*\ntest"), \
             patch("portfolio.digest.send_or_store") as mock_send, \
             patch("portfolio.digest._set_last_digest_time") as mock_set:
            from portfolio.digest import _maybe_send_digest
            _maybe_send_digest({"telegram": {"token": "x", "chat_id": "y"}})
            mock_send.assert_called_once()
            mock_set.assert_called_once()

    def test_handles_build_failure_gracefully(self):
        with patch("portfolio.digest._get_last_digest_time", return_value=0), \
             patch("portfolio.digest._build_digest_message", side_effect=Exception("boom")), \
             patch("portfolio.digest.send_or_store") as mock_send, \
             patch("portfolio.digest._set_last_digest_time") as mock_set:
            from portfolio.digest import _maybe_send_digest
            _maybe_send_digest({})
            mock_send.assert_not_called()
            mock_set.assert_not_called()
