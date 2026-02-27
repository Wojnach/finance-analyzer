"""Tests for portfolio.cumulative_tracker — rolling price changes."""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def _make_snapshot(hours_ago, prices):
    """Build a snapshot dict at a given number of hours in the past."""
    ts = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return {
        "ts": ts.isoformat(),
        "prices": prices,
    }


# ============================================================
# maybe_log_hourly_snapshot
# ============================================================

class TestMaybeLogHourlySnapshot:
    """Tests for maybe_log_hourly_snapshot()."""

    @patch("portfolio.cumulative_tracker._get_last_snapshot_ts")
    @patch("portfolio.cumulative_tracker.atomic_append_jsonl")
    def test_logs_when_no_prior(self, mock_append, mock_last_ts):
        from portfolio.cumulative_tracker import maybe_log_hourly_snapshot
        mock_last_ts.return_value = None

        result = maybe_log_hourly_snapshot({"XAG-USD": 89.5, "BTC-USD": 67000})
        assert result is True
        mock_append.assert_called_once()

    @patch("portfolio.cumulative_tracker._get_last_snapshot_ts")
    @patch("portfolio.cumulative_tracker.atomic_append_jsonl")
    def test_skips_when_recent(self, mock_append, mock_last_ts):
        from portfolio.cumulative_tracker import maybe_log_hourly_snapshot
        mock_last_ts.return_value = time.time() - 30 * 60  # 30 min ago

        result = maybe_log_hourly_snapshot({"XAG-USD": 89.5})
        assert result is False
        mock_append.assert_not_called()

    @patch("portfolio.cumulative_tracker._get_last_snapshot_ts")
    @patch("portfolio.cumulative_tracker.atomic_append_jsonl")
    def test_logs_when_old_enough(self, mock_append, mock_last_ts):
        from portfolio.cumulative_tracker import maybe_log_hourly_snapshot
        mock_last_ts.return_value = time.time() - 60 * 60  # 60 min ago

        result = maybe_log_hourly_snapshot({"XAG-USD": 89.5})
        assert result is True
        mock_append.assert_called_once()

    @patch("portfolio.cumulative_tracker._get_last_snapshot_ts")
    @patch("portfolio.cumulative_tracker.atomic_append_jsonl")
    def test_empty_prices_rejected(self, mock_append, mock_last_ts):
        from portfolio.cumulative_tracker import maybe_log_hourly_snapshot
        mock_last_ts.return_value = None

        result = maybe_log_hourly_snapshot({})
        assert result is False
        mock_append.assert_not_called()

    @patch("portfolio.cumulative_tracker._get_last_snapshot_ts")
    @patch("portfolio.cumulative_tracker.atomic_append_jsonl")
    def test_snapshot_format(self, mock_append, mock_last_ts):
        from portfolio.cumulative_tracker import maybe_log_hourly_snapshot
        mock_last_ts.return_value = None

        maybe_log_hourly_snapshot({"XAG-USD": 89.5123456})
        call_args = mock_append.call_args[0]
        entry = call_args[1]  # second arg to atomic_append_jsonl

        assert "ts" in entry
        assert "prices" in entry
        assert entry["prices"]["XAG-USD"] == 89.5123  # rounded to 4 decimals


# ============================================================
# compute_rolling_changes
# ============================================================

class TestComputeRollingChanges:
    """Tests for compute_rolling_changes()."""

    def test_basic_7d_change(self):
        from portfolio.cumulative_tracker import compute_rolling_changes
        snapshots = [
            _make_snapshot(168, {"XAG-USD": 80.0}),  # 7 days ago
            _make_snapshot(72, {"XAG-USD": 85.0}),    # 3 days ago
            _make_snapshot(24, {"XAG-USD": 87.0}),    # 1 day ago
            _make_snapshot(0, {"XAG-USD": 89.5}),     # now
        ]

        result = compute_rolling_changes(snapshots=snapshots)
        assert "XAG-USD" in result
        # 7d: (89.5 - 80) / 80 * 100 = 11.875%
        assert result["XAG-USD"]["change_7d"] == pytest.approx(11.88, abs=0.1)
        # 3d: (89.5 - 85) / 85 * 100 = 5.29%
        assert result["XAG-USD"]["change_3d"] == pytest.approx(5.29, abs=0.1)
        # 1d: (89.5 - 87) / 87 * 100 = 2.87%
        assert result["XAG-USD"]["change_1d"] == pytest.approx(2.87, abs=0.1)

    def test_negative_change(self):
        from portfolio.cumulative_tracker import compute_rolling_changes
        snapshots = [
            _make_snapshot(24, {"BTC-USD": 70000}),
            _make_snapshot(0, {"BTC-USD": 67000}),
        ]

        result = compute_rolling_changes(snapshots=snapshots)
        assert result["BTC-USD"]["change_1d"] == pytest.approx(-4.29, abs=0.1)

    def test_no_snapshots(self):
        from portfolio.cumulative_tracker import compute_rolling_changes
        result = compute_rolling_changes(snapshots=[])
        assert result == {}

    def test_single_snapshot(self):
        from portfolio.cumulative_tracker import compute_rolling_changes
        snapshots = [_make_snapshot(0, {"XAG-USD": 89.5})]
        result = compute_rolling_changes(snapshots=snapshots)
        # No historical price → None changes
        assert result["XAG-USD"]["change_1d"] is None
        assert result["XAG-USD"]["change_7d"] is None

    def test_ticker_filter(self):
        from portfolio.cumulative_tracker import compute_rolling_changes
        snapshots = [
            _make_snapshot(24, {"XAG-USD": 80, "BTC-USD": 65000}),
            _make_snapshot(0, {"XAG-USD": 89, "BTC-USD": 67000}),
        ]

        result = compute_rolling_changes(tickers=["XAG-USD"], snapshots=snapshots)
        assert "XAG-USD" in result
        assert "BTC-USD" not in result

    def test_missing_ticker_in_old_snapshot(self):
        from portfolio.cumulative_tracker import compute_rolling_changes
        snapshots = [
            _make_snapshot(24, {"BTC-USD": 65000}),  # no XAG
            _make_snapshot(0, {"XAG-USD": 89, "BTC-USD": 67000}),
        ]

        result = compute_rolling_changes(snapshots=snapshots)
        assert result["XAG-USD"]["change_1d"] is None  # no old price
        assert result["BTC-USD"]["change_1d"] is not None

    def test_zero_price_skipped(self):
        from portfolio.cumulative_tracker import compute_rolling_changes
        snapshots = [
            _make_snapshot(24, {"XAG-USD": 0}),
            _make_snapshot(0, {"XAG-USD": 89}),
        ]
        result = compute_rolling_changes(snapshots=snapshots)
        # Old price is 0 → can't compute change
        assert result["XAG-USD"]["change_1d"] is None


# ============================================================
# _find_closest_price
# ============================================================

class TestFindClosestPrice:
    """Tests for _find_closest_price() helper."""

    def test_exact_match(self):
        from portfolio.cumulative_tracker import _find_closest_price
        target = datetime.now(timezone.utc) - timedelta(hours=24)
        snapshots = [{"ts": target.isoformat(), "prices": {"XAG-USD": 85.0}}]

        price = _find_closest_price(snapshots, "XAG-USD", target)
        assert price == 85.0

    def test_closest_within_range(self):
        from portfolio.cumulative_tracker import _find_closest_price
        target = datetime.now(timezone.utc) - timedelta(hours=24)
        snap_near = {"ts": (target + timedelta(hours=1)).isoformat(), "prices": {"XAG-USD": 85.0}}
        snap_far = {"ts": (target + timedelta(hours=5)).isoformat(), "prices": {"XAG-USD": 90.0}}

        price = _find_closest_price([snap_far, snap_near], "XAG-USD", target)
        assert price == 85.0  # closer one wins

    def test_none_when_out_of_range(self):
        from portfolio.cumulative_tracker import _find_closest_price
        target = datetime.now(timezone.utc) - timedelta(hours=24)
        snap = {"ts": (target + timedelta(hours=10)).isoformat(), "prices": {"XAG-USD": 85.0}}

        price = _find_closest_price([snap], "XAG-USD", target, max_hours=6)
        assert price is None

    def test_missing_ticker(self):
        from portfolio.cumulative_tracker import _find_closest_price
        target = datetime.now(timezone.utc)
        snap = {"ts": target.isoformat(), "prices": {"BTC-USD": 67000}}

        price = _find_closest_price([snap], "XAG-USD", target)
        assert price is None


# ============================================================
# get_cumulative_summary
# ============================================================

class TestGetCumulativeSummary:
    """Tests for get_cumulative_summary()."""

    @patch("portfolio.cumulative_tracker.compute_rolling_changes")
    def test_movers_detected(self, mock_changes):
        from portfolio.cumulative_tracker import get_cumulative_summary
        mock_changes.return_value = {
            "XAG-USD": {"change_1d": 3.0, "change_3d": 8.0, "change_7d": 12.0},
            "BTC-USD": {"change_1d": -0.5, "change_3d": -2.0, "change_7d": -3.0},
            "NVDA": {"change_1d": 1.0, "change_3d": 6.0, "change_7d": 4.0},
        }

        # Clear cache to force fresh computation
        from portfolio.shared_state import _tool_cache
        keys_to_del = [k for k in _tool_cache if "cumulative" in k]
        for k in keys_to_del:
            del _tool_cache[k]

        result = get_cumulative_summary()
        assert "ticker_changes" in result
        assert "movers" in result

        movers = result["movers"]
        mover_tickers = [m["ticker"] for m in movers]
        assert "XAG-USD" in mover_tickers  # 7d > 10%
        assert "NVDA" in mover_tickers      # 3d > 5%
        assert "BTC-USD" not in mover_tickers  # below thresholds

    @patch("portfolio.cumulative_tracker.compute_rolling_changes")
    def test_no_movers(self, mock_changes):
        from portfolio.cumulative_tracker import get_cumulative_summary
        mock_changes.return_value = {
            "XAG-USD": {"change_1d": 0.1, "change_3d": 0.5, "change_7d": 1.0},
        }
        from portfolio.shared_state import _tool_cache
        keys_to_del = [k for k in _tool_cache if "cumulative" in k]
        for k in keys_to_del:
            del _tool_cache[k]

        result = get_cumulative_summary()
        assert result["movers"] == []
