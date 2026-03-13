"""Tests for I/O safety fixes (BUG-47 through BUG-51).

Validates that file reads use load_json() from file_utils (TOCTOU-safe)
and that JSONL appends use atomic_append_jsonl() with fsync.
"""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# BUG-47: health.py load_health() should use load_json()
# ---------------------------------------------------------------------------

class TestHealthLoadJson:
    """Ensure load_health() uses load_json() and handles edge cases."""

    def test_load_health_missing_file(self, tmp_path):
        """load_health returns defaults when health file doesn't exist."""
        from portfolio.health import load_health, HEALTH_FILE
        fake_path = tmp_path / "no_such_file.json"
        with patch("portfolio.health.HEALTH_FILE", fake_path):
            result = load_health()
        assert result["cycle_count"] == 0
        assert result["error_count"] == 0
        assert "start_time" in result

    def test_load_health_corrupt_file(self, tmp_path):
        """load_health returns defaults when file is corrupt JSON."""
        corrupt = tmp_path / "health_state.json"
        corrupt.write_text("{invalid json", encoding="utf-8")
        with patch("portfolio.health.HEALTH_FILE", corrupt):
            from portfolio.health import load_health
            result = load_health()
        assert result["cycle_count"] == 0

    def test_load_health_valid_file(self, tmp_path):
        """load_health returns parsed content from valid file."""
        valid = tmp_path / "health_state.json"
        data = {"cycle_count": 42, "error_count": 1, "start_time": 1000, "errors": []}
        valid.write_text(json.dumps(data), encoding="utf-8")
        with patch("portfolio.health.HEALTH_FILE", valid):
            from portfolio.health import load_health
            result = load_health()
        assert result["cycle_count"] == 42

    def test_load_health_empty_file(self, tmp_path):
        """load_health returns defaults when file is empty."""
        empty = tmp_path / "health_state.json"
        empty.write_text("", encoding="utf-8")
        with patch("portfolio.health.HEALTH_FILE", empty):
            from portfolio.health import load_health
            result = load_health()
        assert result["cycle_count"] == 0

    def test_load_health_uses_load_json(self):
        """Verify load_health calls load_json, not raw json.loads."""
        import inspect
        from portfolio.health import load_health
        source = inspect.getsource(load_health)
        assert "load_json" in source, "load_health should use load_json()"
        assert "json.loads" not in source, "load_health should NOT use raw json.loads()"


# ---------------------------------------------------------------------------
# BUG-48: reporting._get_held_tickers() should use load_json()
# ---------------------------------------------------------------------------

class TestGetHeldTickersLoadJson:
    """Ensure _get_held_tickers() uses load_json()."""

    def test_get_held_tickers_uses_load_json(self):
        """Verify _get_held_tickers calls load_json, not raw json.loads."""
        import inspect
        from portfolio.reporting import _get_held_tickers
        source = inspect.getsource(_get_held_tickers)
        assert "load_json" in source, "_get_held_tickers should use load_json()"
        assert "json.loads" not in source, "_get_held_tickers should NOT use raw json.loads()"

    def test_get_held_tickers_missing_files(self, tmp_path):
        """Returns empty set when portfolio files don't exist."""
        from portfolio.reporting import _get_held_tickers, _held_tickers_cache
        _held_tickers_cache["cycle_id"] = -1  # force cache miss
        with patch("portfolio.reporting.DATA_DIR", tmp_path), \
             patch("portfolio.reporting._held_tickers_cache", {"cycle_id": -1, "tickers": set()}):
            result = _get_held_tickers()
        assert result == set()

    def test_get_held_tickers_with_positions(self, tmp_path):
        """Returns held tickers from both portfolios."""
        patient = {"holdings": {"BTC-USD": {"shares": 0.5}, "ETH-USD": {"shares": 0}}}
        bold = {"holdings": {"NVDA": {"shares": 10}}}
        (tmp_path / "portfolio_state.json").write_text(json.dumps(patient), encoding="utf-8")
        (tmp_path / "portfolio_state_bold.json").write_text(json.dumps(bold), encoding="utf-8")
        from portfolio.reporting import _get_held_tickers
        with patch("portfolio.reporting.DATA_DIR", tmp_path), \
             patch("portfolio.reporting._held_tickers_cache", {"cycle_id": -1, "tickers": set()}):
            result = _get_held_tickers()
        assert "BTC-USD" in result
        assert "NVDA" in result
        assert "ETH-USD" not in result  # 0 shares


# ---------------------------------------------------------------------------
# BUG-49 & BUG-50: reporting.py warrant state & stale data should use load_json()
# ---------------------------------------------------------------------------

class TestReportingLoadJsonPatterns:
    """Verify reporting.py uses load_json() for all JSON file reads."""

    def test_write_agent_summary_no_raw_json_loads(self):
        """No raw json.loads(path.read_text()) in write_agent_summary or helpers."""
        import inspect
        from portfolio import reporting
        source = inspect.getsource(reporting)
        # Count remaining raw json.loads patterns (there should be zero for file reads)
        # We check for the specific TOCTOU pattern: json.loads(something.read_text(
        import re
        toctou_pattern = re.compile(r'json\.loads\([^)]*\.read_text\(')
        matches = toctou_pattern.findall(source)
        assert len(matches) == 0, (
            f"Found {len(matches)} raw json.loads(path.read_text()) patterns in reporting.py. "
            f"All should use load_json() from file_utils."
        )


# ---------------------------------------------------------------------------
# BUG-51: outcome_tracker.log_signal_snapshot() should use atomic_append_jsonl()
# ---------------------------------------------------------------------------

class TestOutcomeTrackerAtomicAppend:
    """Ensure log_signal_snapshot() uses atomic_append_jsonl()."""

    def test_log_signal_snapshot_uses_atomic_append(self):
        """Verify the function uses atomic_append_jsonl, not raw open/write."""
        import inspect
        from portfolio.outcome_tracker import log_signal_snapshot
        source = inspect.getsource(log_signal_snapshot)
        assert "atomic_append_jsonl" in source, (
            "log_signal_snapshot should use atomic_append_jsonl()"
        )
        # Should NOT have raw open() + write() for JSONL
        assert 'open(SIGNAL_LOG, "a"' not in source, (
            "log_signal_snapshot should NOT use raw open() for JSONL append"
        )

    def test_log_signal_snapshot_writes_valid_jsonl(self, tmp_path):
        """Verify log_signal_snapshot produces valid JSONL output."""
        signal_log = tmp_path / "signal_log.jsonl"
        signals_dict = {
            "BTC-USD": {
                "action": "HOLD",
                "indicators": {"rsi": 50, "close": 67000},
                "extra": {
                    "_votes": {"rsi": "HOLD", "macd": "BUY"},
                },
            }
        }
        prices = {"BTC-USD": 67000}
        with patch("portfolio.outcome_tracker.SIGNAL_LOG", signal_log), \
             patch("portfolio.outcome_tracker.DATA_DIR", tmp_path), \
             patch.dict("sys.modules", {"portfolio.signal_db": MagicMock()}):
            from portfolio.outcome_tracker import log_signal_snapshot
            entry = log_signal_snapshot(signals_dict, prices, 10.5, ["test_trigger"])

        assert signal_log.exists()
        line = signal_log.read_text(encoding="utf-8").strip()
        parsed = json.loads(line)
        assert parsed["fx_rate"] == 10.5
        assert "BTC-USD" in parsed["tickers"]

    def test_atomic_append_jsonl_has_fsync(self):
        """Verify atomic_append_jsonl flushes and fsyncs."""
        import inspect
        from portfolio.file_utils import atomic_append_jsonl
        source = inspect.getsource(atomic_append_jsonl)
        assert "f.flush()" in source
        assert "os.fsync" in source
