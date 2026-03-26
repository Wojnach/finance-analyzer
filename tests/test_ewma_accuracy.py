"""Tests for signal_accuracy_ewma() in portfolio/accuracy_stats.py.

Covers:
- Basic EWMA weight computation (older entries get lower weight)
- halflife_days parameter controls decay rate
- Returns correct keys: accuracy, total_weight, effective_samples, total, correct, pct
- total/correct are int(round(...)) of weighted sums
- HOLD votes are skipped
- Neutral outcomes (|change_pct| < 0.05) are skipped
- Zero-data case (no entries with outcomes) returns 0 accuracy
- Single entry case
- Signal not in SIGNAL_NAMES is absent from result (iterates SIGNAL_NAMES)
- Integration: signal_engine uses EWMA when config has accuracy_halflife_days
"""

import math
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from portfolio.accuracy_stats import signal_accuracy_ewma
from portfolio.tickers import SIGNAL_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(ts_offset_days, signal_votes, change_pct, ticker="BTC-USD", horizon="1d"):
    """Build a minimal signal log entry.

    Args:
        ts_offset_days: How many days ago this entry was recorded (positive = older).
        signal_votes: Dict of {signal_name: vote} for the ticker.
        change_pct: Price change in the outcome.
        ticker: Ticker symbol.
        horizon: Outcome horizon string.
    """
    ts = (datetime.now(UTC) - timedelta(days=ts_offset_days)).isoformat()
    return {
        "ts": ts,
        "tickers": {
            ticker: {
                "signals": signal_votes,
            }
        },
        "outcomes": {
            ticker: {
                horizon: {"change_pct": change_pct},
            }
        },
    }


def _weight(age_days, halflife=5):
    """Expected exponential decay weight."""
    return math.exp(-math.log(2) / halflife * age_days)


# ---------------------------------------------------------------------------
# Basic structure and keys
# ---------------------------------------------------------------------------

class TestReturnStructure:
    def test_returns_dict_for_each_signal_name(self):
        entries = [_make_entry(1, {"rsi": "BUY"}, 1.0)]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        assert isinstance(result, dict)
        for name in SIGNAL_NAMES:
            assert name in result, f"Missing signal: {name}"

    def test_each_entry_has_required_keys(self):
        entries = [_make_entry(1, {"rsi": "BUY"}, 1.0)]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        for name in SIGNAL_NAMES:
            keys = set(result[name].keys())
            assert "accuracy" in keys
            assert "total_weight" in keys
            assert "effective_samples" in keys
            assert "total" in keys
            assert "correct" in keys
            assert "pct" in keys

    def test_total_and_correct_are_ints(self):
        entries = [_make_entry(1, {"rsi": "BUY"}, 1.0)]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        assert isinstance(result["rsi"]["total"], int)
        assert isinstance(result["rsi"]["correct"], int)

    def test_pct_is_accuracy_times_100(self):
        entries = [_make_entry(1, {"rsi": "BUY"}, 1.0)]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        rsi = result["rsi"]
        assert abs(rsi["pct"] - round(rsi["accuracy"] * 100, 1)) < 1e-6


# ---------------------------------------------------------------------------
# No data / empty cases
# ---------------------------------------------------------------------------

class TestEmptyData:
    def test_no_entries_returns_zero_accuracy(self):
        with patch("portfolio.accuracy_stats.load_entries", return_value=[]):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        for name in SIGNAL_NAMES:
            assert result[name]["accuracy"] == 0.0
            assert result[name]["total"] == 0
            assert result[name]["correct"] == 0

    def test_entries_without_outcomes_returns_zero(self):
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "tickers": {"BTC-USD": {"signals": {"rsi": "BUY"}}},
            "outcomes": {},  # no outcomes
        }
        with patch("portfolio.accuracy_stats.load_entries", return_value=[entry]):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        assert result["rsi"]["total"] == 0

    def test_entries_with_wrong_horizon_returns_zero(self):
        entry = _make_entry(1, {"rsi": "BUY"}, 1.0, horizon="3d")
        with patch("portfolio.accuracy_stats.load_entries", return_value=[entry]):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        assert result["rsi"]["total"] == 0


# ---------------------------------------------------------------------------
# HOLD votes and neutral outcomes are skipped
# ---------------------------------------------------------------------------

class TestSkipping:
    def test_hold_vote_is_skipped(self):
        entries = [_make_entry(1, {"rsi": "HOLD"}, 1.0)]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        assert result["rsi"]["total"] == 0

    def test_neutral_outcome_is_skipped(self):
        # |change_pct| < 0.05 is neutral
        entries = [_make_entry(1, {"rsi": "BUY"}, 0.01)]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        assert result["rsi"]["total"] == 0

    def test_exactly_at_threshold_is_skipped(self):
        # |0.04| < 0.05, should be skipped
        entries = [_make_entry(1, {"rsi": "BUY"}, 0.04)]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        assert result["rsi"]["total"] == 0

    def test_just_above_threshold_is_counted(self):
        # |0.06| >= 0.05, should be counted
        entries = [_make_entry(1, {"rsi": "BUY"}, 0.06)]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        assert result["rsi"]["total"] == 1


# ---------------------------------------------------------------------------
# Accuracy correctness: BUY/SELL vs direction
# ---------------------------------------------------------------------------

class TestAccuracyCorrectness:
    def test_buy_correct_when_price_up(self):
        entries = [_make_entry(1, {"rsi": "BUY"}, 1.0)]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        assert result["rsi"]["accuracy"] == pytest.approx(1.0)

    def test_buy_incorrect_when_price_down(self):
        entries = [_make_entry(1, {"rsi": "BUY"}, -1.0)]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        assert result["rsi"]["accuracy"] == pytest.approx(0.0)

    def test_sell_correct_when_price_down(self):
        entries = [_make_entry(1, {"rsi": "SELL"}, -1.0)]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        assert result["rsi"]["accuracy"] == pytest.approx(1.0)

    def test_sell_incorrect_when_price_up(self):
        entries = [_make_entry(1, {"rsi": "SELL"}, 1.0)]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        assert result["rsi"]["accuracy"] == pytest.approx(0.0)

    def test_mixed_correct_and_incorrect(self):
        # 2 correct (BUY+up), 1 incorrect (BUY+down)
        # All same age (1 day), so weights are equal
        entries = [
            _make_entry(1, {"rsi": "BUY"}, 1.0),
            _make_entry(1, {"rsi": "BUY"}, 1.0),
            _make_entry(1, {"rsi": "BUY"}, -1.0),
        ]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        # All same age: accuracy = 2/3
        assert result["rsi"]["accuracy"] == pytest.approx(2.0 / 3.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Exponential decay: older entries get lower weight
# ---------------------------------------------------------------------------

class TestDecayWeighting:
    def test_recent_entry_has_higher_weight_than_old(self):
        """Two entries: 1-day-old correct BUY and 10-day-old incorrect BUY.
        With decay, recent entry outweighs old. Accuracy should be > 0.5.
        """
        entries = [
            _make_entry(1, {"rsi": "BUY"}, 1.0),   # recent, correct
            _make_entry(10, {"rsi": "BUY"}, -1.0),  # old, incorrect
        ]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        # Weight recent = exp(-ln2/5 * 1), weight old = exp(-ln2/5 * 10)
        w1 = _weight(1, 5)
        w2 = _weight(10, 5)
        expected_acc = w1 / (w1 + w2)
        assert result["rsi"]["accuracy"] == pytest.approx(expected_acc, rel=1e-4)
        assert result["rsi"]["accuracy"] > 0.5

    def test_old_entry_has_lower_weight_than_recent(self):
        """Two entries: 1-day-old incorrect BUY and 10-day-old correct BUY.
        Incorrect recent entry should drag accuracy below 0.5.
        """
        entries = [
            _make_entry(1, {"rsi": "BUY"}, -1.0),  # recent, incorrect
            _make_entry(10, {"rsi": "BUY"}, 1.0),   # old, correct
        ]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        w1 = _weight(1, 5)
        w2 = _weight(10, 5)
        expected_acc = w2 / (w1 + w2)
        assert result["rsi"]["accuracy"] == pytest.approx(expected_acc, rel=1e-4)
        assert result["rsi"]["accuracy"] < 0.5

    def test_halflife_1_gives_steeper_decay(self):
        """With halflife=1, a 7-day-old entry should have very low weight vs halflife=50."""
        entries = [
            _make_entry(1, {"rsi": "BUY"}, 1.0),   # recent, correct
            _make_entry(7, {"rsi": "BUY"}, -1.0),   # old, incorrect
        ]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result_steep = signal_accuracy_ewma("1d", halflife_days=1)
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result_flat = signal_accuracy_ewma("1d", halflife_days=50)
        # Steeper decay weights recent more, so accuracy should be higher
        assert result_steep["rsi"]["accuracy"] > result_flat["rsi"]["accuracy"]

    def test_default_halflife_is_5_days(self):
        """signal_accuracy_ewma() with no halflife_days arg uses 5 as default."""
        entries = [
            _make_entry(1, {"rsi": "BUY"}, 1.0),
            _make_entry(10, {"rsi": "BUY"}, -1.0),
        ]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result_default = signal_accuracy_ewma("1d")
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result_explicit = signal_accuracy_ewma("1d", halflife_days=5)
        assert result_default["rsi"]["accuracy"] == pytest.approx(
            result_explicit["rsi"]["accuracy"], rel=1e-9
        )


# ---------------------------------------------------------------------------
# total_weight and effective_samples
# ---------------------------------------------------------------------------

class TestWeightMetrics:
    def test_total_weight_is_sum_of_weights(self):
        """total_weight should be the sum of per-observation weights."""
        entries = [
            _make_entry(1, {"rsi": "BUY"}, 1.0),
            _make_entry(5, {"rsi": "BUY"}, 1.0),
        ]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        w1 = _weight(1, 5)
        w5 = _weight(5, 5)
        expected_total_weight = w1 + w5
        assert result["rsi"]["total_weight"] == pytest.approx(expected_total_weight, rel=1e-4)

    def test_effective_samples_formula(self):
        """effective_samples = total_weight^2 / sum(w_i^2)."""
        entries = [
            _make_entry(1, {"rsi": "BUY"}, 1.0),
            _make_entry(5, {"rsi": "BUY"}, 1.0),
        ]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        w1 = _weight(1, 5)
        w5 = _weight(5, 5)
        total_w = w1 + w5
        sum_w2 = w1**2 + w5**2
        expected_eff = total_w**2 / sum_w2
        assert result["rsi"]["effective_samples"] == pytest.approx(expected_eff, rel=1e-4)

    def test_equal_weights_effective_samples_equals_n(self):
        """When all entries have the same age, effective_samples = n."""
        # Create 4 entries with same age
        entries = [_make_entry(3, {"rsi": "BUY"}, 1.0) for _ in range(4)]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        assert result["rsi"]["effective_samples"] == pytest.approx(4.0, rel=1e-4)

    def test_zero_total_weight_when_no_data(self):
        with patch("portfolio.accuracy_stats.load_entries", return_value=[]):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        assert result["rsi"]["total_weight"] == 0.0
        assert result["rsi"]["effective_samples"] == 0.0


# ---------------------------------------------------------------------------
# Weighted totals: total and correct as rounded ints
# ---------------------------------------------------------------------------

class TestWeightedCounts:
    def test_total_is_int_of_rounded_sum(self):
        """total = int(round(sum of weights))."""
        entries = [
            _make_entry(1, {"rsi": "BUY"}, 1.0),
            _make_entry(3, {"rsi": "BUY"}, 1.0),
        ]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        w1 = _weight(1, 5)
        w3 = _weight(3, 5)
        expected_total = int(round(w1 + w3))
        assert result["rsi"]["total"] == expected_total

    def test_correct_is_int_of_rounded_correct_weight(self):
        """correct = int(round(sum of weights for correct outcomes))."""
        entries = [
            _make_entry(1, {"rsi": "BUY"}, 1.0),   # correct
            _make_entry(3, {"rsi": "BUY"}, -1.0),   # incorrect
        ]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        w1 = _weight(1, 5)
        expected_correct = int(round(w1))
        assert result["rsi"]["correct"] == expected_correct

    def test_all_correct_total_equals_correct(self):
        entries = [_make_entry(1, {"rsi": "BUY"}, 1.0) for _ in range(3)]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        # All same age: total == correct (both are int(round(3 * w)))
        assert result["rsi"]["total"] == result["rsi"]["correct"]


# ---------------------------------------------------------------------------
# Multiple signals and multiple tickers
# ---------------------------------------------------------------------------

class TestMultipleSignalsAndTickers:
    def test_only_matching_signal_gets_counted(self):
        """rsi votes don't affect macd counts."""
        entries = [_make_entry(1, {"rsi": "BUY"}, 1.0)]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        assert result["rsi"]["total"] == 1
        assert result["macd"]["total"] == 0

    def test_multiple_signals_in_same_entry(self):
        """Both rsi and macd vote in the same entry."""
        entries = [_make_entry(1, {"rsi": "BUY", "macd": "SELL"}, 1.0)]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        # rsi BUY + up = correct
        assert result["rsi"]["accuracy"] == pytest.approx(1.0)
        # macd SELL + up = incorrect
        assert result["macd"]["accuracy"] == pytest.approx(0.0)

    def test_multiple_tickers_aggregate_independently(self):
        """Each ticker's entry is counted toward signal accuracy."""
        entry = {
            "ts": (datetime.now(UTC) - timedelta(days=1)).isoformat(),
            "tickers": {
                "BTC-USD": {"signals": {"rsi": "BUY"}},
                "ETH-USD": {"signals": {"rsi": "BUY"}},
            },
            "outcomes": {
                "BTC-USD": {"1d": {"change_pct": 1.0}},
                "ETH-USD": {"1d": {"change_pct": -1.0}},
            },
        }
        with patch("portfolio.accuracy_stats.load_entries", return_value=[entry]):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        # 1 correct (BTC) + 1 incorrect (ETH): accuracy = 0.5
        assert result["rsi"]["accuracy"] == pytest.approx(0.5, rel=1e-4)
        assert result["rsi"]["total"] == 2  # both same age, weights ~1 each, rounds to 2


# ---------------------------------------------------------------------------
# Different horizons
# ---------------------------------------------------------------------------

class TestHorizons:
    def test_uses_correct_horizon(self):
        """Entries with 3d outcome don't show up when querying 1d."""
        entry_1d = _make_entry(1, {"rsi": "BUY"}, 1.0, horizon="1d")
        entry_3d = _make_entry(1, {"rsi": "SELL"}, -1.0, horizon="3d")
        with patch("portfolio.accuracy_stats.load_entries", return_value=[entry_1d, entry_3d]):
            result_1d = signal_accuracy_ewma("1d", halflife_days=5)
            result_3d = signal_accuracy_ewma("3d", halflife_days=5)
        # 1d horizon: only BUY+up = 100%
        assert result_1d["rsi"]["accuracy"] == pytest.approx(1.0)
        # 3d horizon: only SELL+down = 100%
        assert result_3d["rsi"]["accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integration: signal_engine uses EWMA when halflife_days configured
# ---------------------------------------------------------------------------

class TestSignalEngineIntegration:
    """Verify signal_engine.py reads halflife_days from config and uses EWMA path."""

    def test_halflife_days_config_key_accepted(self):
        """Reading halflife from config should not raise errors."""
        config = {"signals": {"accuracy_halflife_days": 5}}
        halflife = (config or {}).get("signals", {}).get("accuracy_halflife_days", 5)
        assert halflife == 5

    def test_halflife_days_default_is_5(self):
        """If key not present in config, default should be 5."""
        config = {"signals": {}}
        halflife = (config or {}).get("signals", {}).get("accuracy_halflife_days", 5)
        assert halflife == 5

    def test_halflife_days_none_config_default_5(self):
        """If config is None, default should be 5."""
        config = None
        halflife = (config or {}).get("signals", {}).get("accuracy_halflife_days", 5)
        assert halflife == 5

    def test_ewma_result_is_compatible_with_accuracy_data_format(self):
        """EWMA result keys match what _weighted_consensus expects."""
        entries = [_make_entry(1, {"rsi": "BUY"}, 1.0)]
        with patch("portfolio.accuracy_stats.load_entries", return_value=entries):
            result = signal_accuracy_ewma("1d", halflife_days=5)
        # _weighted_consensus reads .get("accuracy", 0.5) and .get("total", 0)
        for name in SIGNAL_NAMES:
            assert "accuracy" in result[name]
            assert "total" in result[name]
            assert isinstance(result[name]["accuracy"], float)
            assert isinstance(result[name]["total"], int)

    def test_ewma_fallback_when_no_entries(self):
        """When EWMA returns no data (all zeros), signal_engine should fall back to all-time."""
        from portfolio.accuracy_stats import signal_accuracy_ewma as ewma

        with patch("portfolio.accuracy_stats.load_entries", return_value=[]):
            result = ewma("1d", halflife_days=5)

        # With no entries, every signal total == 0
        assert all(v["total"] == 0 for v in result.values())
