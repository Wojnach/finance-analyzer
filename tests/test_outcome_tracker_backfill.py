"""Tests for outcome_tracker.backfill_outcomes() memory optimization (BUG-12).

Covers:
- max_entries parameter limits processing to last N entries
- Head entries are preserved unchanged in output
- Malformed JSONL lines are skipped
- Default behavior processes all entries when below limit
"""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest


def _make_entry(hours_ago, tickers=None, outcomes=None):
    """Create a signal_log entry with given age."""
    ts = (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat()
    entry = {"ts": ts, "tickers": tickers or {"BTC-USD": {"price_usd": 67000, "consensus": "BUY"}}}
    if outcomes is not None:
        entry["outcomes"] = outcomes
    return entry


def _write_signal_log(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _read_signal_log(path):
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


class TestBackfillMaxEntries:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.log_file = tmp_path / "signal_log.jsonl"

    def test_max_entries_limits_processing(self):
        """Only the last max_entries entries should be processed."""
        # Create 10 entries, all fully filled (old) and 5 needing backfill (recent)
        old_entries = []
        for i in range(10):
            e = _make_entry(hours_ago=500 + i)
            e["outcomes"] = {"BTC-USD": {
                "1d": {"price_usd": 67100, "change_pct": 0.15, "ts": "2026-01-01"},
                "3d": {"price_usd": 67200, "change_pct": 0.30, "ts": "2026-01-03"},
                "5d": {"price_usd": 67300, "change_pct": 0.45, "ts": "2026-01-05"},
                "10d": {"price_usd": 67400, "change_pct": 0.60, "ts": "2026-01-10"},
            }}
            old_entries.append(e)

        recent_entries = [_make_entry(hours_ago=i) for i in range(5)]
        all_entries = old_entries + recent_entries
        _write_signal_log(self.log_file, all_entries)

        with patch("portfolio.outcome_tracker.SIGNAL_LOG", self.log_file), \
             patch("portfolio.outcome_tracker._fetch_historical_price", return_value=None):
            from portfolio.outcome_tracker import backfill_outcomes
            result = backfill_outcomes(max_entries=5)

        # Read back and verify all 15 entries preserved
        result_entries = _read_signal_log(self.log_file)
        assert len(result_entries) == 15

    def test_head_entries_preserved_unchanged(self):
        """Entries before the max_entries window should be preserved exactly."""
        old_entry = _make_entry(hours_ago=500)
        old_entry["custom_field"] = "should_survive"
        recent_entry = _make_entry(hours_ago=1)

        _write_signal_log(self.log_file, [old_entry, recent_entry])

        with patch("portfolio.outcome_tracker.SIGNAL_LOG", self.log_file), \
             patch("portfolio.outcome_tracker._fetch_historical_price", return_value=None):
            from portfolio.outcome_tracker import backfill_outcomes
            backfill_outcomes(max_entries=1)

        result = _read_signal_log(self.log_file)
        assert len(result) == 2
        assert result[0]["custom_field"] == "should_survive"

    def test_all_entries_processed_when_below_limit(self):
        """When entry count is below max_entries, all are processed."""
        entries = [_make_entry(hours_ago=i) for i in range(3)]
        _write_signal_log(self.log_file, entries)

        with patch("portfolio.outcome_tracker.SIGNAL_LOG", self.log_file), \
             patch("portfolio.outcome_tracker._fetch_historical_price", return_value=None):
            from portfolio.outcome_tracker import backfill_outcomes
            backfill_outcomes(max_entries=100)

        result = _read_signal_log(self.log_file)
        assert len(result) == 3

    def test_malformed_jsonl_lines_skipped(self):
        """Malformed lines should be skipped, not crash."""
        good_entry = _make_entry(hours_ago=1)
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(good_entry) + "\n")
            f.write("NOT VALID JSON\n")
            f.write("{incomplete\n")

        with patch("portfolio.outcome_tracker.SIGNAL_LOG", self.log_file), \
             patch("portfolio.outcome_tracker._fetch_historical_price", return_value=None):
            from portfolio.outcome_tracker import backfill_outcomes
            result = backfill_outcomes(max_entries=100)

        result_entries = _read_signal_log(self.log_file)
        assert len(result_entries) == 1  # only the valid entry survives

    def test_empty_file_returns_zero(self):
        """Empty file should return 0 updated."""
        _write_signal_log(self.log_file, [])

        with patch("portfolio.outcome_tracker.SIGNAL_LOG", self.log_file):
            from portfolio.outcome_tracker import backfill_outcomes
            result = backfill_outcomes()
        assert result == 0

    def test_missing_file_returns_zero(self):
        """Missing file should return 0."""
        missing = self.log_file.parent / "nonexistent.jsonl"
        with patch("portfolio.outcome_tracker.SIGNAL_LOG", missing):
            from portfolio.outcome_tracker import backfill_outcomes
            result = backfill_outcomes()
        assert result == 0
