"""Tests for portfolio/equity_curve.py.

Covers: load_equity_curve, compute_metrics, compare_strategies,
get_latest_values, _daily_returns, _parse_ts.
"""

import json
import math
import pathlib

import pytest

from portfolio.equity_curve import (
    load_equity_curve,
    compute_metrics,
    compare_strategies,
    get_latest_values,
    _daily_returns,
    _parse_ts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: pathlib.Path, entries: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _make_entry(ts, patient_val, bold_val, patient_pnl=None, bold_pnl=None,
                fx_rate=10.0, prices=None):
    if patient_pnl is None:
        patient_pnl = ((patient_val - 500_000) / 500_000) * 100
    if bold_pnl is None:
        bold_pnl = ((bold_val - 500_000) / 500_000) * 100
    return {
        "ts": ts,
        "patient_value_sek": patient_val,
        "bold_value_sek": bold_val,
        "patient_pnl_pct": patient_pnl,
        "bold_pnl_pct": bold_pnl,
        "fx_rate": fx_rate,
        "prices": prices or {},
    }


# A 5-day curve with some up and down movement
FIVE_DAY_CURVE = [
    _make_entry("2026-02-16T12:00:00+00:00", 500_000, 500_000),
    _make_entry("2026-02-17T12:00:00+00:00", 505_000, 510_000),
    _make_entry("2026-02-18T12:00:00+00:00", 502_000, 495_000),
    _make_entry("2026-02-19T12:00:00+00:00", 510_000, 520_000),
    _make_entry("2026-02-20T12:00:00+00:00", 508_000, 515_000),
]


# ===================================================================
# load_equity_curve
# ===================================================================

class TestLoadEquityCurve:
    def test_load_from_file(self, tmp_path):
        """Load entries from a JSONL file."""
        path = tmp_path / "history.jsonl"
        _write_jsonl(path, FIVE_DAY_CURVE)
        result = load_equity_curve(str(path))
        assert len(result) == 5
        assert result[0]["ts"] == "2026-02-16T12:00:00+00:00"

    def test_nonexistent_file(self, tmp_path):
        """Non-existent file -> empty list."""
        path = tmp_path / "nonexistent.jsonl"
        result = load_equity_curve(str(path))
        assert result == []

    def test_malformed_lines_skipped(self, tmp_path):
        """Malformed JSON lines are silently skipped."""
        path = tmp_path / "history.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(_make_entry("2026-02-16T12:00:00+00:00", 500_000, 500_000)) + "\n")
            f.write("NOT JSON\n")
            f.write(json.dumps(_make_entry("2026-02-17T12:00:00+00:00", 510_000, 510_000)) + "\n")
        result = load_equity_curve(str(path))
        assert len(result) == 2

    def test_sorted_by_timestamp(self, tmp_path):
        """Entries are sorted by timestamp."""
        path = tmp_path / "history.jsonl"
        entries = [
            _make_entry("2026-02-20T12:00:00+00:00", 510_000, 510_000),
            _make_entry("2026-02-16T12:00:00+00:00", 500_000, 500_000),
            _make_entry("2026-02-18T12:00:00+00:00", 505_000, 505_000),
        ]
        _write_jsonl(path, entries)
        result = load_equity_curve(str(path))
        assert result[0]["ts"] == "2026-02-16T12:00:00+00:00"
        assert result[-1]["ts"] == "2026-02-20T12:00:00+00:00"

    def test_empty_file(self, tmp_path):
        """Empty file -> empty list."""
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        result = load_equity_curve(str(path))
        assert result == []

    def test_blank_lines_skipped(self, tmp_path):
        """Blank lines in the JSONL file are skipped."""
        path = tmp_path / "history.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(_make_entry("2026-02-16T12:00:00+00:00", 500_000, 500_000)) + "\n")
            f.write("\n")
            f.write("  \n")
            f.write(json.dumps(_make_entry("2026-02-17T12:00:00+00:00", 510_000, 510_000)) + "\n")
        result = load_equity_curve(str(path))
        assert len(result) == 2


# ===================================================================
# _parse_ts
# ===================================================================

class TestParseTs:
    def test_aware_timestamp(self):
        dt = _parse_ts("2026-02-16T12:00:00+00:00")
        assert dt.tzinfo is not None

    def test_naive_timestamp_gets_utc(self):
        """Naive timestamp should get UTC timezone."""
        dt = _parse_ts("2026-02-16T12:00:00")
        assert dt.tzinfo is not None


# ===================================================================
# _daily_returns
# ===================================================================

class TestDailyReturns:
    def test_basic_returns(self):
        """Compute day-over-day returns."""
        returns = _daily_returns(FIVE_DAY_CURVE, "patient_value_sek")
        assert len(returns) == 4  # 5 days -> 4 returns
        # Day 1->2: (505000-500000)/500000 * 100 = 1.0%
        assert abs(returns[0] - 1.0) < 0.01

    def test_empty_curve(self):
        assert _daily_returns([], "patient_value_sek") == []

    def test_single_entry(self):
        """Single entry -> no returns."""
        single = [_make_entry("2026-02-16T12:00:00+00:00", 500_000, 500_000)]
        assert _daily_returns(single, "patient_value_sek") == []

    def test_multiple_entries_same_day_uses_last(self):
        """Multiple entries per day -> uses last value per day."""
        curve = [
            _make_entry("2026-02-16T08:00:00+00:00", 500_000, 500_000),
            _make_entry("2026-02-16T20:00:00+00:00", 502_000, 502_000),
            _make_entry("2026-02-17T08:00:00+00:00", 510_000, 510_000),
            _make_entry("2026-02-17T20:00:00+00:00", 508_000, 508_000),
        ]
        returns = _daily_returns(curve, "patient_value_sek")
        assert len(returns) == 1  # 2 days -> 1 return
        # Last value Feb 16 = 502000, last value Feb 17 = 508000
        # Return = (508000 - 502000) / 502000 * 100
        expected = (508_000 - 502_000) / 502_000 * 100
        assert abs(returns[0] - expected) < 0.01


# ===================================================================
# compute_metrics
# ===================================================================

class TestComputeMetrics:
    def test_empty_curve(self):
        """Empty curve returns default zeros/Nones."""
        result = compute_metrics([], "patient")
        assert result["max_drawdown_pct"] == 0.0
        assert result["sharpe_ratio"] is None
        assert result["sortino_ratio"] is None
        assert result["win_rate"] == 0.0
        assert result["num_data_points"] == 0
        assert result["date_range"] is None

    def test_flat_curve(self):
        """Flat curve -> 0% return, 0% drawdown."""
        curve = [
            _make_entry("2026-02-16T12:00:00+00:00", 500_000, 500_000),
            _make_entry("2026-02-17T12:00:00+00:00", 500_000, 500_000),
            _make_entry("2026-02-18T12:00:00+00:00", 500_000, 500_000),
        ]
        result = compute_metrics(curve, "patient")
        assert result["total_return_pct"] == 0.0
        assert result["max_drawdown_pct"] == 0.0

    def test_total_return(self):
        """Total return calculated from first to last value."""
        result = compute_metrics(FIVE_DAY_CURVE, "patient")
        # 500K -> 508K = +1.6%
        assert abs(result["total_return_pct"] - 1.6) < 0.01

    def test_max_drawdown(self):
        """Max drawdown from peak to trough."""
        curve = [
            _make_entry("2026-02-16T12:00:00+00:00", 500_000, 500_000),
            _make_entry("2026-02-17T12:00:00+00:00", 520_000, 520_000),  # peak
            _make_entry("2026-02-18T12:00:00+00:00", 468_000, 468_000),  # trough (10% DD)
            _make_entry("2026-02-19T12:00:00+00:00", 510_000, 510_000),  # recovery
        ]
        result = compute_metrics(curve, "patient")
        # DD from 520K peak to 468K trough = (520K-468K)/520K * 100 = 10.0%
        assert abs(result["max_drawdown_pct"] - 10.0) < 0.01

    def test_win_rate(self):
        """Win rate is percentage of positive-return days."""
        result = compute_metrics(FIVE_DAY_CURVE, "patient")
        # Returns: +1.0%, -0.59%, +1.59%, -0.39% -> 2 positive out of 4
        assert abs(result["win_rate"] - 50.0) < 1.0

    def test_best_worst_day(self):
        """Best and worst day returns identified."""
        result = compute_metrics(FIVE_DAY_CURVE, "patient")
        assert result["best_day_pct"] > 0
        assert result["worst_day_pct"] < 0

    def test_sharpe_ratio_computed(self):
        """Sharpe ratio computed when sufficient data."""
        result = compute_metrics(FIVE_DAY_CURVE, "patient")
        # 5 days = 4 returns, enough for Sharpe
        assert result["sharpe_ratio"] is not None

    def test_sortino_ratio_computed(self):
        """Sortino ratio computed when there are downside returns."""
        result = compute_metrics(FIVE_DAY_CURVE, "patient")
        # Some negative days exist
        assert result["sortino_ratio"] is not None

    def test_volatility_computed(self):
        """Annualized volatility computed from daily returns."""
        result = compute_metrics(FIVE_DAY_CURVE, "patient")
        assert result["volatility_annual_pct"] > 0

    def test_date_range(self):
        """Date range from first to last entry."""
        result = compute_metrics(FIVE_DAY_CURVE, "patient")
        assert result["date_range"] is not None
        assert result["date_range"][0] == "2026-02-16T12:00:00+00:00"
        assert result["date_range"][1] == "2026-02-20T12:00:00+00:00"

    def test_bold_strategy(self):
        """Bold strategy uses bold_value_sek."""
        result = compute_metrics(FIVE_DAY_CURVE, "bold")
        # Bold: 500K -> 515K = +3.0%
        assert abs(result["total_return_pct"] - 3.0) < 0.01

    def test_all_losses_curve(self):
        """Monotonically decreasing curve."""
        curve = [
            _make_entry("2026-02-16T12:00:00+00:00", 500_000, 500_000),
            _make_entry("2026-02-17T12:00:00+00:00", 490_000, 490_000),
            _make_entry("2026-02-18T12:00:00+00:00", 480_000, 480_000),
            _make_entry("2026-02-19T12:00:00+00:00", 470_000, 470_000),
        ]
        result = compute_metrics(curve, "patient")
        assert result["total_return_pct"] < 0
        assert result["max_drawdown_pct"] > 0
        assert result["win_rate"] == 0.0

    def test_single_data_point(self):
        """Single data point -> minimal metrics."""
        single = [_make_entry("2026-02-16T12:00:00+00:00", 500_000, 500_000)]
        result = compute_metrics(single, "patient")
        assert result["num_data_points"] == 1
        assert result["total_return_pct"] == 0.0
        assert result["sharpe_ratio"] is None  # Not enough returns

    def test_annualized_return(self):
        """Annualized return computed when > 1 day elapsed."""
        result = compute_metrics(FIVE_DAY_CURVE, "patient")
        assert result["annualized_return_pct"] is not None

    def test_days_in_drawdown(self):
        """Count of unique dates when portfolio was below peak."""
        curve = [
            _make_entry("2026-02-16T12:00:00+00:00", 500_000, 500_000),
            _make_entry("2026-02-17T12:00:00+00:00", 510_000, 510_000),  # new peak
            _make_entry("2026-02-18T12:00:00+00:00", 505_000, 505_000),  # below peak
            _make_entry("2026-02-19T12:00:00+00:00", 503_000, 503_000),  # below peak
            _make_entry("2026-02-20T12:00:00+00:00", 515_000, 515_000),  # new peak
        ]
        result = compute_metrics(curve, "patient")
        assert result["days_in_drawdown"] == 2  # Feb 18 and Feb 19


# ===================================================================
# compare_strategies
# ===================================================================

class TestCompareStrategies:
    def test_basic_comparison(self):
        """Compare patient vs bold with known values."""
        result = compare_strategies(FIVE_DAY_CURVE)
        assert "patient" in result
        assert "bold" in result
        assert "comparison" in result

    def test_leader_identified(self):
        """Leader is the strategy with higher return."""
        result = compare_strategies(FIVE_DAY_CURVE)
        # Bold: 500K->515K = +3.0%, Patient: 500K->508K = +1.6%
        assert result["comparison"]["leader"] == "bold"

    def test_drawdown_comparison(self):
        """Lower drawdown strategy identified."""
        result = compare_strategies(FIVE_DAY_CURVE)
        assert "lower_drawdown" in result["comparison"]

    def test_sharpe_comparison_when_available(self):
        """Sharpe leader included when both strategies have Sharpe data."""
        result = compare_strategies(FIVE_DAY_CURVE)
        if result["patient"]["sharpe_ratio"] is not None and result["bold"]["sharpe_ratio"] is not None:
            assert "sharpe_leader" in result["comparison"]

    def test_empty_curve(self):
        """Empty curve -> both strategies have default metrics."""
        result = compare_strategies([])
        assert result["patient"]["total_return_pct"] == 0.0
        assert result["bold"]["total_return_pct"] == 0.0
        assert result["comparison"]["return_diff_pct"] == 0.0

    def test_equal_performance(self):
        """Both strategies have the same values."""
        curve = [
            _make_entry("2026-02-16T12:00:00+00:00", 500_000, 500_000),
            _make_entry("2026-02-17T12:00:00+00:00", 510_000, 510_000),
        ]
        result = compare_strategies(curve)
        assert result["comparison"]["return_diff_pct"] == 0.0


# ===================================================================
# get_latest_values
# ===================================================================

class TestGetLatestValues:
    def test_returns_last_entry(self):
        """Returns the most recent entry."""
        result = get_latest_values(FIVE_DAY_CURVE)
        assert result is not None
        assert result["ts"] == "2026-02-20T12:00:00+00:00"
        assert result["patient_value_sek"] == 508_000
        assert result["bold_value_sek"] == 515_000

    def test_empty_curve(self):
        """Empty curve -> None."""
        assert get_latest_values([]) is None

    def test_single_entry(self):
        """Single entry curve returns that entry."""
        single = [_make_entry("2026-02-16T12:00:00+00:00", 500_000, 500_000)]
        result = get_latest_values(single)
        assert result is not None
        assert result["patient_value_sek"] == 500_000
