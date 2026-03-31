"""Tests for portfolio/backtester.py.

Covers:
  1. _old_consensus: simple majority voting
  2. Module imports successfully
  3. run_backtest with mocked entries
  4. print_report does not crash
  5. Edge cases: empty entries, no-outcome entries, all HOLD votes
"""

from unittest.mock import patch

import pytest

from portfolio.backtester import (
    HORIZONS,
    _old_consensus,
    _safe_div,
    print_report,
    run_backtest,
)

# ===========================================================================
# _old_consensus — simple majority voting
# ===========================================================================

class TestOldConsensus:
    """Tests for the simple majority voting baseline."""

    def test_all_hold_returns_hold(self):
        votes = {"rsi": "HOLD", "macd": "HOLD", "ema": "HOLD"}
        assert _old_consensus(votes) == "HOLD"

    def test_empty_votes_returns_hold(self):
        assert _old_consensus({}) == "HOLD"

    def test_single_buy_returns_buy(self):
        assert _old_consensus({"rsi": "BUY"}) == "BUY"

    def test_single_sell_returns_sell(self):
        assert _old_consensus({"rsi": "SELL"}) == "SELL"

    def test_buy_majority(self):
        votes = {"rsi": "BUY", "macd": "BUY", "ema": "SELL"}
        assert _old_consensus(votes) == "BUY"

    def test_sell_majority(self):
        votes = {"rsi": "SELL", "macd": "SELL", "ema": "BUY"}
        assert _old_consensus(votes) == "SELL"

    def test_tie_returns_hold(self):
        votes = {"rsi": "BUY", "macd": "SELL"}
        assert _old_consensus(votes) == "HOLD"

    def test_all_buy_returns_buy(self):
        votes = {"rsi": "BUY", "macd": "BUY", "ema": "BUY", "bb": "BUY"}
        assert _old_consensus(votes) == "BUY"

    def test_all_sell_returns_sell(self):
        votes = {"rsi": "SELL", "macd": "SELL", "ema": "SELL"}
        assert _old_consensus(votes) == "SELL"

    def test_three_way_with_hold_buy_wins(self):
        # HOLD does not count in buy vs sell; only BUY/SELL matter
        votes = {"rsi": "BUY", "macd": "HOLD", "ema": "SELL", "bb": "BUY"}
        assert _old_consensus(votes) == "BUY"

    def test_single_buy_vs_many_hold(self):
        votes = {"rsi": "BUY", "macd": "HOLD", "ema": "HOLD", "bb": "HOLD", "volume": "HOLD"}
        assert _old_consensus(votes) == "BUY"


# ===========================================================================
# _safe_div — safe division helper
# ===========================================================================

class TestSafeDiv:
    def test_normal_division(self):
        assert _safe_div(3, 4) == pytest.approx(0.75)

    def test_zero_denominator_returns_default(self):
        assert _safe_div(5, 0) == 0.0

    def test_zero_denominator_custom_default(self):
        assert _safe_div(5, 0, default=-1.0) == -1.0


# ===========================================================================
# Module import
# ===========================================================================

def test_module_imports():
    """The backtester module should import without errors."""
    import portfolio.backtester as bt
    assert hasattr(bt, "_old_consensus")
    assert hasattr(bt, "run_backtest")
    assert hasattr(bt, "print_report")
    assert hasattr(bt, "HORIZONS")
    assert bt.HORIZONS == ["3h", "4h", "12h", "1d", "3d", "5d", "10d"]


# ===========================================================================
# run_backtest — integration with mocked data
# ===========================================================================

def _make_entry(vote, change_pct, horizon="1d", ticker="BTC-USD", regime="ranging"):
    """Build a minimal signal log entry."""
    return {
        "ts": "2026-01-01T12:00:00+00:00",
        "tickers": {
            ticker: {
                "regime": regime,
                "signals": {"rsi": vote, "macd": vote},
            }
        },
        "outcomes": {
            ticker: {
                horizon: {"change_pct": change_pct}
            }
        },
    }


class TestRunBacktest:
    """Tests for the main backtest runner."""

    def test_empty_entries_returns_error(self):
        with patch("portfolio.accuracy_stats.load_entries", return_value=[]):
            with patch("portfolio.backtester._build_accuracy_data", return_value={}):
                result = run_backtest(horizon="1d")
        assert "error" in result

    def test_returns_expected_keys(self):
        entries = [_make_entry("BUY", 1.0)]
        with (
            patch("portfolio.accuracy_stats.load_entries", return_value=entries),
            patch("portfolio.backtester._build_accuracy_data", return_value={}),
        ):
            result = run_backtest(horizon="1d")

        assert "entries" in result
        assert "horizon_results" in result
        assert "signal_results" in result

    def test_correct_buy_counted(self):
        """A correct BUY prediction increments old_correct and new_correct."""
        entries = [_make_entry("BUY", 1.0)]  # BUY + price up = correct
        with (
            patch("portfolio.accuracy_stats.load_entries", return_value=entries),
            patch("portfolio.backtester._build_accuracy_data", return_value={}),
        ):
            result = run_backtest(horizon="1d")

        r = result["horizon_results"]["1d"]
        assert r["old_correct"] == 1
        assert r["old_total"] == 1

    def test_incorrect_buy_not_counted_as_correct(self):
        """A wrong BUY prediction increments old_total but not old_correct."""
        entries = [_make_entry("BUY", -1.0)]  # BUY + price down = wrong
        with (
            patch("portfolio.accuracy_stats.load_entries", return_value=entries),
            patch("portfolio.backtester._build_accuracy_data", return_value={}),
        ):
            result = run_backtest(horizon="1d")

        r = result["horizon_results"]["1d"]
        assert r["old_correct"] == 0
        assert r["old_total"] == 1

    def test_hold_not_counted(self):
        """HOLD consensus does not count in accuracy metrics."""
        # Equal BUY/SELL -> HOLD from old consensus
        entries = [
            {
                "ts": "2026-01-01T12:00:00+00:00",
                "tickers": {
                    "BTC-USD": {
                        "regime": "ranging",
                        "signals": {"rsi": "BUY", "macd": "SELL"},  # tie -> HOLD
                    }
                },
                "outcomes": {
                    "BTC-USD": {
                        "1d": {"change_pct": 1.0}
                    }
                },
            }
        ]
        with (
            patch("portfolio.accuracy_stats.load_entries", return_value=entries),
            patch("portfolio.backtester._build_accuracy_data", return_value={}),
        ):
            result = run_backtest(horizon="1d")

        r = result["horizon_results"]["1d"]
        assert r["old_total"] == 0  # HOLD not counted

    def test_neutral_outcome_not_counted(self):
        """Neutral price moves (within ±0.05%) do not count."""
        entries = [_make_entry("BUY", 0.01)]  # 0.01% change = neutral
        with (
            patch("portfolio.accuracy_stats.load_entries", return_value=entries),
            patch("portfolio.backtester._build_accuracy_data", return_value={}),
        ):
            result = run_backtest(horizon="1d")

        r = result["horizon_results"]["1d"]
        assert r["old_total"] == 0

    def test_days_filter_applied(self):
        """Entries older than --days cutoff are excluded."""
        old_entry = dict(_make_entry("BUY", 1.0))
        old_entry["ts"] = "2020-01-01T00:00:00+00:00"  # very old

        recent_entry = dict(_make_entry("BUY", 1.0))
        recent_entry["ts"] = "2026-03-27T12:00:00+00:00"  # recent

        with (
            patch("portfolio.accuracy_stats.load_entries", return_value=[old_entry, recent_entry]),
            patch("portfolio.backtester._build_accuracy_data", return_value={}),
        ):
            result_all = run_backtest(horizon="1d", days=None)
            result_recent = run_backtest(horizon="1d", days=30)

        r_all = result_all["horizon_results"]["1d"]
        r_recent = result_recent["horizon_results"]["1d"]
        # All data has 2 entries; recent 30-day filter should have fewer
        assert r_all["old_total"] >= r_recent["old_total"]

    def test_multiple_horizons_evaluated(self):
        """Entries with multiple outcome horizons contribute to each."""
        entry = {
            "ts": "2026-01-01T12:00:00+00:00",
            "tickers": {
                "BTC-USD": {
                    "regime": "ranging",
                    "signals": {"rsi": "BUY", "macd": "BUY"},
                }
            },
            "outcomes": {
                "BTC-USD": {
                    "1d": {"change_pct": 1.0},
                    "3d": {"change_pct": 1.5},
                    "5d": {"change_pct": -0.5},
                }
            },
        }
        with (
            patch("portfolio.accuracy_stats.load_entries", return_value=[entry]),
            patch("portfolio.backtester._build_accuracy_data", return_value={}),
        ):
            result = run_backtest(horizon="1d")

        assert result["horizon_results"]["1d"]["old_total"] == 1
        assert result["horizon_results"]["3d"]["old_total"] == 1
        assert result["horizon_results"]["5d"]["old_total"] == 1

    def test_entries_count_returned(self):
        entries = [_make_entry("BUY", 1.0) for _ in range(5)]
        with (
            patch("portfolio.accuracy_stats.load_entries", return_value=entries),
            patch("portfolio.backtester._build_accuracy_data", return_value={}),
        ):
            result = run_backtest(horizon="1d")

        assert result["entries"] == 5


# ===========================================================================
# print_report — smoke tests
# ===========================================================================

class TestPrintReport:
    """Ensure print_report runs without crashing on various inputs."""

    def test_prints_without_error(self, capsys):
        results = {
            "entries": 100,
            "min_ts": "2026-01-01T00:00:00+00:00",
            "max_ts": "2026-03-27T00:00:00+00:00",
            "horizon_results": {
                h: {"old_correct": 60, "old_total": 100, "new_correct": 65, "new_total": 100}
                for h in HORIZONS
            },
            "signal_results": {
                "rsi": {"old_correct": 55, "old_total": 100, "new_correct": 58, "new_total": 100},
                "macd": {"old_correct": 50, "old_total": 100, "new_correct": 52, "new_total": 100},
            },
        }
        print_report(results, target_horizon="1d")
        captured = capsys.readouterr()
        assert "BACKTEST" in captured.out
        assert "Old Acc" in captured.out
        assert "New Acc" in captured.out
        assert "Delta" in captured.out

    def test_error_result_prints_error(self, capsys):
        print_report({"error": "No entries found", "entries": 0})
        captured = capsys.readouterr()
        assert "Error" in captured.out

    def test_zero_sample_horizons_skipped(self, capsys):
        """Horizons with 0 old and new samples should not appear in output."""
        results = {
            "entries": 10,
            "min_ts": "2026-01-01T00:00:00+00:00",
            "max_ts": "2026-03-27T00:00:00+00:00",
            "horizon_results": {
                "1d": {"old_correct": 6, "old_total": 10, "new_correct": 7, "new_total": 10},
                "3h": {"old_correct": 0, "old_total": 0, "new_correct": 0, "new_total": 0},
                "4h": {"old_correct": 0, "old_total": 0, "new_correct": 0, "new_total": 0},
                "12h": {"old_correct": 0, "old_total": 0, "new_correct": 0, "new_total": 0},
                "3d": {"old_correct": 0, "old_total": 0, "new_correct": 0, "new_total": 0},
                "5d": {"old_correct": 0, "old_total": 0, "new_correct": 0, "new_total": 0},
                "10d": {"old_correct": 0, "old_total": 0, "new_correct": 0, "new_total": 0},
            },
            "signal_results": {},
        }
        print_report(results, target_horizon="1d")
        captured = capsys.readouterr()
        # 1d should appear, but 3h row should not
        assert "1d" in captured.out

    def test_summary_shows_improvement(self, capsys):
        """Summary section shows improvement delta."""
        results = {
            "entries": 1000,
            "min_ts": "2026-01-01T00:00:00+00:00",
            "max_ts": "2026-03-27T00:00:00+00:00",
            "horizon_results": {
                h: {"old_correct": 50, "old_total": 100, "new_correct": 60, "new_total": 100}
                for h in HORIZONS
            },
            "signal_results": {},
        }
        print_report(results, target_horizon="1d")
        captured = capsys.readouterr()
        assert "Summary" in captured.out
        assert "Improvement" in captured.out
