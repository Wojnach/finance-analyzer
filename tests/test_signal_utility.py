"""Tests for signal_utility() in accuracy_stats.py.

Covers:
  1. Correct BUY with positive change → positive avg_return
  2. Wrong BUY with negative change → negative avg_return
  3. SELL signals → return = -change_pct
  4. HOLD votes excluded
  5. Empty entries → zero / default values
  6. Neutral outcomes (|change| < 0.05) → skipped
  7. utility_score = avg_return * sqrt(samples)
  8. Multiple signals, multiple tickers, multiple entries
"""

import math
from unittest.mock import patch

from portfolio.accuracy_stats import signal_utility
from portfolio.tickers import SIGNAL_NAMES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(ticker, sig_name, vote, change_pct, horizon="1d", ts="2026-01-01T12:00:00"):
    """Build a minimal signal-log entry for one ticker, one signal."""
    return {
        "ts": ts,
        "tickers": {
            ticker: {
                "signals": {sig_name: vote},
            }
        },
        "outcomes": {
            ticker: {
                horizon: {"change_pct": change_pct}
            }
        },
    }


def _make_entries(rows, horizon="1d"):
    """Build a list of entries from (ticker, sig_name, vote, change_pct) tuples."""
    return [
        _make_entry(t, s, v, c, horizon=horizon)
        for t, s, v, c in rows
    ]


# ---------------------------------------------------------------------------
# 1. Correct BUY with positive change → positive avg_return
# ---------------------------------------------------------------------------

class TestBuyCorrect:
    def test_buy_positive_change_positive_avg_return(self):
        entries = _make_entries([
            ("BTC-USD", "rsi", "BUY", 1.5),
            ("BTC-USD", "rsi", "BUY", 2.0),
            ("BTC-USD", "rsi", "BUY", 0.8),
        ])
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_utility("1d")
        rsi = result["rsi"]
        assert rsi["avg_return"] > 0, "avg_return should be positive for correct BUY"
        assert rsi["samples"] == 3
        assert abs(rsi["avg_return"] - (1.5 + 2.0 + 0.8) / 3) < 1e-9

    def test_buy_avg_return_equals_mean_change(self):
        entries = _make_entries([
            ("ETH-USD", "macd", "BUY", 3.0),
            ("ETH-USD", "macd", "BUY", 1.0),
        ])
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_utility("1d")
        macd = result["macd"]
        assert abs(macd["avg_return"] - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# 2. Wrong BUY with negative change → negative avg_return
# ---------------------------------------------------------------------------

class TestBuyWrong:
    def test_buy_negative_change_negative_avg_return(self):
        entries = _make_entries([
            ("BTC-USD", "ema", "BUY", -2.0),
            ("BTC-USD", "ema", "BUY", -1.5),
        ])
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_utility("1d")
        ema = result["ema"]
        assert ema["avg_return"] < 0, "avg_return should be negative for wrong BUY"
        assert abs(ema["avg_return"] - (-1.75)) < 1e-9

    def test_mixed_buy_returns_average(self):
        entries = _make_entries([
            ("NVDA", "rsi", "BUY", 4.0),   # correct  → +4.0
            ("NVDA", "rsi", "BUY", -2.0),  # wrong    → -2.0
        ])
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_utility("1d")
        rsi = result["rsi"]
        assert abs(rsi["avg_return"] - 1.0) < 1e-9   # (4 - 2) / 2


# ---------------------------------------------------------------------------
# 3. SELL signals → directional return = -change_pct
# ---------------------------------------------------------------------------

class TestSellSignals:
    def test_sell_positive_change_negative_return(self):
        """SELL when price went up = bad call → negative return."""
        entries = _make_entries([
            ("XAU-USD", "bb", "SELL", 2.0),
        ])
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_utility("1d")
        bb = result["bb"]
        assert abs(bb["avg_return"] - (-2.0)) < 1e-9

    def test_sell_negative_change_positive_return(self):
        """SELL when price fell = good call → positive return."""
        entries = _make_entries([
            ("XAG-USD", "macd", "SELL", -3.0),
            ("XAG-USD", "macd", "SELL", -1.5),
        ])
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_utility("1d")
        macd = result["macd"]
        # -change_pct for SELL: -(-3.0)=3.0, -(-1.5)=1.5 → avg=2.25
        assert abs(macd["avg_return"] - 2.25) < 1e-9
        assert macd["avg_return"] > 0


# ---------------------------------------------------------------------------
# 4. HOLD votes excluded
# ---------------------------------------------------------------------------

class TestHoldExcluded:
    def test_hold_votes_not_counted(self):
        entries = _make_entries([
            ("BTC-USD", "rsi", "HOLD", 5.0),
            ("BTC-USD", "rsi", "HOLD", -3.0),
        ])
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_utility("1d")
        rsi = result["rsi"]
        assert rsi["samples"] == 0
        assert rsi["avg_return"] == 0.0

    def test_hold_mixed_with_buy_excludes_hold(self):
        entries = _make_entries([
            ("BTC-USD", "ema", "HOLD", 10.0),  # excluded
            ("BTC-USD", "ema", "BUY", 2.0),    # included
        ])
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_utility("1d")
        ema = result["ema"]
        assert ema["samples"] == 1
        assert abs(ema["avg_return"] - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# 5. Empty entries → zero / default values
# ---------------------------------------------------------------------------

class TestEmptyEntries:
    def test_no_entries_returns_zero_defaults(self):
        with patch("portfolio.accuracy_stats.load_entries", return_value=[]):
            result = signal_utility("1d")
        assert isinstance(result, dict)
        # All signal names should be present
        for sig in SIGNAL_NAMES:
            assert sig in result
            assert result[sig]["avg_return"] == 0.0
            assert result[sig]["samples"] == 0
            assert result[sig]["utility_score"] == 0.0

    def test_signal_not_voting_returns_zero(self):
        """A signal present in SIGNAL_NAMES but not in any entry gets zeros."""
        entries = _make_entries([
            ("BTC-USD", "rsi", "BUY", 1.0),
        ])
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_utility("1d")
        # macd has no entries here
        macd = result["macd"]
        assert macd["samples"] == 0
        assert macd["avg_return"] == 0.0
        assert macd["utility_score"] == 0.0


# ---------------------------------------------------------------------------
# 6. Neutral outcomes (|change_pct| < 0.05) → skipped
# ---------------------------------------------------------------------------

class TestNeutralOutcomesSkipped:
    def test_tiny_change_excluded(self):
        entries = _make_entries([
            ("BTC-USD", "rsi", "BUY", 0.03),   # |0.03| < 0.05 → skip
            ("BTC-USD", "rsi", "BUY", -0.04),  # |0.04| < 0.05 → skip
        ])
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_utility("1d")
        rsi = result["rsi"]
        assert rsi["samples"] == 0
        assert rsi["avg_return"] == 0.0

    def test_boundary_exactly_min_change_included(self):
        """change_pct == 0.05 is exactly the threshold — should be included (>= not >)."""
        entries = _make_entries([
            ("BTC-USD", "rsi", "BUY", 0.05),   # |0.05| == 0.05 → included
        ])
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_utility("1d")
        rsi = result["rsi"]
        assert rsi["samples"] == 1

    def test_mixed_neutral_and_real_moves(self):
        entries = _make_entries([
            ("BTC-USD", "rsi", "BUY", 0.02),   # skip (neutral)
            ("BTC-USD", "rsi", "BUY", 2.0),    # include
        ])
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_utility("1d")
        rsi = result["rsi"]
        assert rsi["samples"] == 1
        assert abs(rsi["avg_return"] - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# 7. utility_score = avg_return * sqrt(samples)
# ---------------------------------------------------------------------------

class TestUtilityScore:
    def test_utility_score_formula(self):
        entries = _make_entries([
            ("BTC-USD", "rsi", "BUY", 2.0),
            ("BTC-USD", "rsi", "BUY", 4.0),
            ("BTC-USD", "rsi", "BUY", 3.0),
        ])
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_utility("1d")
        rsi = result["rsi"]
        expected_avg = (2.0 + 4.0 + 3.0) / 3
        expected_utility = expected_avg * math.sqrt(3)
        assert abs(rsi["utility_score"] - expected_utility) < 1e-9

    def test_utility_score_zero_when_no_samples(self):
        with patch("portfolio.accuracy_stats.load_entries", return_value=[]):
            result = signal_utility("1d")
        for sig in SIGNAL_NAMES:
            assert result[sig]["utility_score"] == 0.0

    def test_utility_score_negative_when_avg_return_negative(self):
        entries = _make_entries([
            ("BTC-USD", "ema", "BUY", -3.0),
            ("BTC-USD", "ema", "BUY", -1.0),
        ])
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_utility("1d")
        ema = result["ema"]
        assert ema["utility_score"] < 0
        expected = (-2.0) * math.sqrt(2)
        assert abs(ema["utility_score"] - expected) < 1e-9

    def test_utility_score_rewards_magnitude(self):
        """A 55%-accurate signal on 3% moves beats a 60%-accurate signal on 0.2% moves."""
        entries_high_mag = _make_entries(
            [("BTC-USD", "rsi", "BUY", 3.0)] * 55 +
            [("BTC-USD", "rsi", "BUY", -3.0)] * 45
        )
        entries_low_mag = _make_entries(
            [("ETH-USD", "macd", "BUY", 0.2)] * 60 +
            [("ETH-USD", "macd", "BUY", -0.2)] * 40
        )
        combined = entries_high_mag + entries_low_mag
        with patch("portfolio.accuracy_stats.load_entries", return_value=combined):
            result = signal_utility("1d")
        # rsi: avg_return = (55*3 + 45*(-3))/100 = (165-135)/100 = 0.30
        # macd: avg_return = (60*0.2 + 40*(-0.2))/100 = (12-8)/100 = 0.04
        # rsi utility = 0.30 * sqrt(100) = 3.0
        # macd utility = 0.04 * sqrt(100) = 0.4
        assert result["rsi"]["utility_score"] > result["macd"]["utility_score"]


# ---------------------------------------------------------------------------
# 8. Multiple signals, multiple tickers, multiple entries
# ---------------------------------------------------------------------------

class TestMultipleSignalsTickers:
    def test_two_signals_tracked_independently(self):
        entries = [
            _make_entry("BTC-USD", "rsi", "BUY", 2.0),
            _make_entry("ETH-USD", "macd", "SELL", -1.5),
        ]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_utility("1d")
        assert abs(result["rsi"]["avg_return"] - 2.0) < 1e-9
        # SELL with -1.5 → directional return = -(-1.5) = 1.5
        assert abs(result["macd"]["avg_return"] - 1.5) < 1e-9

    def test_multiple_tickers_same_signal_aggregated(self):
        """Returns from the same signal across different tickers are pooled."""
        entries = [
            _make_entry("BTC-USD", "rsi", "BUY", 4.0),
            _make_entry("ETH-USD", "rsi", "BUY", 2.0),
            _make_entry("NVDA", "rsi", "BUY", 6.0),
        ]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_utility("1d")
        rsi = result["rsi"]
        assert rsi["samples"] == 3
        assert abs(rsi["avg_return"] - 4.0) < 1e-9  # (4+2+6)/3 = 4.0

    def test_total_return_is_sum_of_directional_returns(self):
        entries = _make_entries([
            ("BTC-USD", "rsi", "BUY", 2.0),
            ("BTC-USD", "rsi", "BUY", 3.0),
        ])
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_utility("1d")
        rsi = result["rsi"]
        assert abs(rsi["total_return"] - 5.0) < 1e-9

    def test_horizon_filtering(self):
        """Only outcomes for the requested horizon are used."""
        entry = {
            "ts": "2026-01-01T12:00:00",
            "tickers": {"BTC-USD": {"signals": {"rsi": "BUY"}}},
            "outcomes": {
                "BTC-USD": {
                    "1d": {"change_pct": 5.0},
                    "3d": {"change_pct": -2.0},
                }
            },
        }
        with patch("portfolio.accuracy_stats.load_entries", return_value=[entry]):
            result_1d = signal_utility("1d")
            result_3d = signal_utility("3d")
        assert abs(result_1d["rsi"]["avg_return"] - 5.0) < 1e-9
        assert abs(result_3d["rsi"]["avg_return"] - (-2.0)) < 1e-9

    def test_missing_outcome_for_horizon_skipped(self):
        """Entries without the requested horizon outcome are silently skipped."""
        entries = [
            _make_entry("BTC-USD", "rsi", "BUY", 2.0, horizon="1d"),
        ]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_utility("3d")  # 3d outcomes don't exist
        assert result["rsi"]["samples"] == 0
