"""Tests for portfolio.ticker_accuracy — per-ticker accuracy and probability engine."""

import math
import pytest
from unittest.mock import patch, MagicMock


# --- Helper: build fake signal_log entries ---

def _make_entry(ts, ticker, signals, outcome_1d=None, outcome_3h=None, outcome_3d=None):
    """Build a signal_log entry dict for testing."""
    entry = {
        "ts": ts,
        "tickers": {
            ticker: {
                "price_usd": 100.0,
                "consensus": "HOLD",
                "signals": signals,
            }
        },
        "outcomes": {},
    }
    if outcome_1d is not None:
        entry["outcomes"][ticker] = {"1d": {"change_pct": outcome_1d, "price_usd": 100 + outcome_1d}}
    if outcome_3h is not None:
        entry["outcomes"].setdefault(ticker, {})["3h"] = {"change_pct": outcome_3h, "price_usd": 100 + outcome_3h}
    if outcome_3d is not None:
        entry["outcomes"].setdefault(ticker, {})["3d"] = {"change_pct": outcome_3d, "price_usd": 100 + outcome_3d}
    return entry


def _make_entries_with_known_accuracy(ticker, signal_name, buys_correct, buys_total,
                                       sells_correct=0, sells_total=0):
    """Build entries giving a known accuracy for one signal on one ticker."""
    entries = []
    ts_base = "2026-02-20T00:00:00+00:00"
    idx = 0

    # BUY votes
    for i in range(buys_total):
        correct = i < buys_correct
        change = 1.0 if correct else -1.0
        ts = f"2026-02-20T{idx:02d}:00:00+00:00"
        entries.append(_make_entry(ts, ticker, {signal_name: "BUY"}, outcome_1d=change))
        idx += 1

    # SELL votes
    for i in range(sells_total):
        correct = i < sells_correct
        change = -1.0 if correct else 1.0
        ts = f"2026-02-20T{idx:02d}:00:00+00:00"
        entries.append(_make_entry(ts, ticker, {signal_name: "SELL"}, outcome_1d=change))
        idx += 1

    return entries


# ============================================================
# accuracy_by_ticker_signal
# ============================================================

class TestAccuracyByTickerSignal:
    """Tests for accuracy_by_ticker_signal()."""

    @patch("portfolio.accuracy_stats.load_entries")
    def test_basic_accuracy(self, mock_load):
        from portfolio.ticker_accuracy import accuracy_by_ticker_signal
        entries = _make_entries_with_known_accuracy("XAG-USD", "rsi", 7, 10)
        mock_load.return_value = entries

        result = accuracy_by_ticker_signal("XAG-USD", horizon="1d")
        assert "rsi" in result
        assert result["rsi"]["accuracy"] == pytest.approx(0.7, abs=0.001)
        assert result["rsi"]["samples"] == 10
        assert result["rsi"]["correct"] == 7

    @patch("portfolio.accuracy_stats.load_entries")
    def test_100_percent_accuracy(self, mock_load):
        from portfolio.ticker_accuracy import accuracy_by_ticker_signal
        entries = _make_entries_with_known_accuracy("XAG-USD", "ema", 5, 5)
        mock_load.return_value = entries

        result = accuracy_by_ticker_signal("XAG-USD", horizon="1d")
        assert result["ema"]["accuracy"] == pytest.approx(1.0)
        assert result["ema"]["samples"] == 5

    @patch("portfolio.accuracy_stats.load_entries")
    def test_0_percent_accuracy(self, mock_load):
        from portfolio.ticker_accuracy import accuracy_by_ticker_signal
        entries = _make_entries_with_known_accuracy("BTC-USD", "macd", 0, 8)
        mock_load.return_value = entries

        result = accuracy_by_ticker_signal("BTC-USD", horizon="1d")
        assert result["macd"]["accuracy"] == pytest.approx(0.0)
        assert result["macd"]["samples"] == 8

    @patch("portfolio.accuracy_stats.load_entries")
    def test_different_ticker_isolated(self, mock_load):
        """Entries for ETH-USD should not affect XAG-USD accuracy."""
        from portfolio.ticker_accuracy import accuracy_by_ticker_signal
        xag_entries = _make_entries_with_known_accuracy("XAG-USD", "rsi", 8, 10)
        eth_entries = _make_entries_with_known_accuracy("ETH-USD", "rsi", 2, 10)
        mock_load.return_value = xag_entries + eth_entries

        result = accuracy_by_ticker_signal("XAG-USD", horizon="1d")
        assert result["rsi"]["accuracy"] == pytest.approx(0.8)
        assert result["rsi"]["samples"] == 10

    @patch("portfolio.accuracy_stats.load_entries")
    def test_no_entries(self, mock_load):
        from portfolio.ticker_accuracy import accuracy_by_ticker_signal
        mock_load.return_value = []

        result = accuracy_by_ticker_signal("XAG-USD", horizon="1d")
        assert result == {}

    @patch("portfolio.accuracy_stats.load_entries")
    def test_hold_signals_ignored(self, mock_load):
        from portfolio.ticker_accuracy import accuracy_by_ticker_signal
        entries = [_make_entry("2026-02-20T01:00:00+00:00", "XAG-USD",
                               {"rsi": "HOLD", "ema": "BUY"}, outcome_1d=1.0)]
        mock_load.return_value = entries

        result = accuracy_by_ticker_signal("XAG-USD", horizon="1d")
        assert "rsi" not in result  # HOLD votes not counted
        assert "ema" in result
        assert result["ema"]["samples"] == 1

    @patch("portfolio.accuracy_stats.load_entries")
    def test_no_outcomes_ignored(self, mock_load):
        from portfolio.ticker_accuracy import accuracy_by_ticker_signal
        # Entry with no outcome
        entry = _make_entry("2026-02-20T01:00:00+00:00", "XAG-USD", {"rsi": "BUY"})
        mock_load.return_value = [entry]

        result = accuracy_by_ticker_signal("XAG-USD", horizon="1d")
        assert result == {}

    @patch("portfolio.accuracy_stats.load_entries")
    def test_sell_accuracy(self, mock_load):
        from portfolio.ticker_accuracy import accuracy_by_ticker_signal
        entries = _make_entries_with_known_accuracy("XAG-USD", "bb", 0, 0, sells_correct=6, sells_total=10)
        mock_load.return_value = entries

        result = accuracy_by_ticker_signal("XAG-USD", horizon="1d")
        assert result["bb"]["accuracy"] == pytest.approx(0.6)
        assert result["bb"]["samples"] == 10

    @patch("portfolio.accuracy_stats.load_entries")
    def test_multiple_signals(self, mock_load):
        from portfolio.ticker_accuracy import accuracy_by_ticker_signal
        entries = []
        for i in range(10):
            change = 1.0 if i < 7 else -1.0
            entries.append(_make_entry(
                f"2026-02-20T{i:02d}:00:00+00:00", "XAG-USD",
                {"rsi": "BUY", "ema": "BUY", "macd": "SELL"},
                outcome_1d=change
            ))
        mock_load.return_value = entries

        result = accuracy_by_ticker_signal("XAG-USD", horizon="1d")
        assert result["rsi"]["accuracy"] == pytest.approx(0.7)
        assert result["ema"]["accuracy"] == pytest.approx(0.7)
        # MACD voted SELL, price went up 7 times → 3 correct (3 times price went down)
        assert result["macd"]["accuracy"] == pytest.approx(0.3)

    @patch("portfolio.accuracy_stats.load_entries")
    def test_days_filter(self, mock_load):
        from portfolio.ticker_accuracy import accuracy_by_ticker_signal
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        old_entries = _make_entries_with_known_accuracy("XAG-USD", "rsi", 2, 10)
        # Override timestamps to be old (30 days ago)
        for e in old_entries:
            e["ts"] = (now - timedelta(days=30)).isoformat()
        new_entries = _make_entries_with_known_accuracy("XAG-USD", "rsi", 9, 10)
        # Override timestamps to be recent (1 hour ago)
        for i, e in enumerate(new_entries):
            e["ts"] = (now - timedelta(hours=1, minutes=i)).isoformat()
        mock_load.return_value = old_entries + new_entries

        # With days=7, old entries should be excluded
        result = accuracy_by_ticker_signal("XAG-USD", horizon="1d", days=7)
        assert result["rsi"]["accuracy"] == pytest.approx(0.9)
        assert result["rsi"]["samples"] == 10


# ============================================================
# direction_probability
# ============================================================

class TestDirectionProbability:
    """Tests for direction_probability()."""

    @patch("portfolio.ticker_accuracy.accuracy_by_ticker_signal")
    def test_basic_buy_probability(self, mock_acc):
        from portfolio.ticker_accuracy import direction_probability
        mock_acc.return_value = {
            "rsi": {"accuracy": 0.7, "samples": 100, "correct": 70},
        }
        votes = {"rsi": "BUY", "ema": "HOLD"}

        result = direction_probability("XAG-USD", votes, horizon="1d")
        assert result["direction"] == "up"
        assert result["probability"] == pytest.approx(0.7, abs=0.01)
        assert result["signals_used"] == 1

    @patch("portfolio.ticker_accuracy.accuracy_by_ticker_signal")
    def test_basic_sell_probability(self, mock_acc):
        from portfolio.ticker_accuracy import direction_probability
        mock_acc.return_value = {
            "rsi": {"accuracy": 0.7, "samples": 100, "correct": 70},
        }
        votes = {"rsi": "SELL"}

        result = direction_probability("XAG-USD", votes, horizon="1d")
        assert result["direction"] == "down"
        # SELL with 70% accuracy → P(up) = 1 - 0.7 = 0.3
        assert result["probability"] == pytest.approx(0.3, abs=0.01)

    @patch("portfolio.ticker_accuracy.accuracy_by_ticker_signal")
    def test_multiple_signals_weighted(self, mock_acc):
        """3 signals: BUY at 70%/100sam, BUY at 60%/50sam, BUY at 80%/200sam.
        Expected P(up) = weighted average with sqrt(samples) weights.
        """
        from portfolio.ticker_accuracy import direction_probability
        mock_acc.return_value = {
            "rsi": {"accuracy": 0.7, "samples": 100, "correct": 70},
            "ema": {"accuracy": 0.6, "samples": 50, "correct": 30},
            "bb": {"accuracy": 0.8, "samples": 200, "correct": 160},
        }
        votes = {"rsi": "BUY", "ema": "BUY", "bb": "BUY"}

        result = direction_probability("XAG-USD", votes, horizon="1d")

        # Manual calc:
        w_rsi = math.sqrt(100)  # 10
        w_ema = math.sqrt(50)   # ~7.07
        w_bb = math.sqrt(200)   # ~14.14
        expected = (0.7 * w_rsi + 0.6 * w_ema + 0.8 * w_bb) / (w_rsi + w_ema + w_bb)
        assert result["probability"] == pytest.approx(expected, abs=0.01)
        assert result["signals_used"] == 3
        assert result["total_samples"] == 350

    @patch("portfolio.ticker_accuracy.accuracy_by_ticker_signal")
    def test_mixed_buy_sell(self, mock_acc):
        from portfolio.ticker_accuracy import direction_probability
        mock_acc.return_value = {
            "rsi": {"accuracy": 0.7, "samples": 100, "correct": 70},
            "ema": {"accuracy": 0.7, "samples": 100, "correct": 70},
        }
        # RSI says BUY, EMA says SELL — both 70% accurate
        votes = {"rsi": "BUY", "ema": "SELL"}

        result = direction_probability("XAG-USD", votes, horizon="1d")
        # BUY at 70% → p_up=0.7, SELL at 70% → p_up=0.3
        # Equal weights → (0.7 + 0.3) / 2 = 0.5
        assert result["probability"] == pytest.approx(0.5, abs=0.01)
        assert result["direction"] == "neutral"

    @patch("portfolio.ticker_accuracy.accuracy_by_ticker_signal")
    def test_no_active_signals(self, mock_acc):
        from portfolio.ticker_accuracy import direction_probability
        mock_acc.return_value = {}
        votes = {"rsi": "HOLD", "ema": "HOLD"}

        result = direction_probability("XAG-USD", votes, horizon="1d")
        assert result["direction"] == "neutral"
        assert result["probability"] == 0.5
        assert result["signals_used"] == 0

    @patch("portfolio.ticker_accuracy.accuracy_by_ticker_signal")
    def test_insufficient_samples_excluded(self, mock_acc):
        from portfolio.ticker_accuracy import direction_probability
        mock_acc.return_value = {
            "rsi": {"accuracy": 0.9, "samples": 3, "correct": 3},  # < min_samples
            "ema": {"accuracy": 0.6, "samples": 50, "correct": 30},
        }
        votes = {"rsi": "BUY", "ema": "BUY"}

        result = direction_probability("XAG-USD", votes, horizon="1d", min_samples=5)
        assert result["signals_used"] == 1  # only ema
        assert result["probability"] == pytest.approx(0.6, abs=0.01)

    @patch("portfolio.ticker_accuracy.accuracy_by_ticker_signal")
    def test_disabled_signals_excluded(self, mock_acc):
        from portfolio.ticker_accuracy import direction_probability
        mock_acc.return_value = {
            "ml": {"accuracy": 0.9, "samples": 100, "correct": 90},
            "rsi": {"accuracy": 0.6, "samples": 100, "correct": 60},
        }
        votes = {"ml": "BUY", "rsi": "BUY"}

        result = direction_probability("XAG-USD", votes, horizon="1d")
        # ml is in DISABLED_SIGNALS, should be excluded
        assert result["signals_used"] == 1
        assert result["probability"] == pytest.approx(0.6, abs=0.01)

    @patch("portfolio.ticker_accuracy.accuracy_by_ticker_signal")
    def test_direction_thresholds(self, mock_acc):
        from portfolio.ticker_accuracy import direction_probability

        # p_up = 0.51 → neutral (below 0.52 threshold)
        mock_acc.return_value = {"rsi": {"accuracy": 0.51, "samples": 100, "correct": 51}}
        r = direction_probability("XAG-USD", {"rsi": "BUY"})
        assert r["direction"] == "neutral"

        # p_up = 0.53 → up (above 0.52 threshold)
        mock_acc.return_value = {"rsi": {"accuracy": 0.53, "samples": 100, "correct": 53}}
        r = direction_probability("XAG-USD", {"rsi": "BUY"})
        assert r["direction"] == "up"

        # p_up = 0.47 → down (below 0.48 threshold)
        mock_acc.return_value = {"rsi": {"accuracy": 0.47, "samples": 100, "correct": 47}}
        r = direction_probability("XAG-USD", {"rsi": "BUY"})
        assert r["direction"] == "down"

    @patch("portfolio.ticker_accuracy.accuracy_by_ticker_signal")
    def test_signal_details_populated(self, mock_acc):
        from portfolio.ticker_accuracy import direction_probability
        mock_acc.return_value = {
            "rsi": {"accuracy": 0.71, "samples": 89, "correct": 63},
        }
        votes = {"rsi": "BUY"}

        result = direction_probability("XAG-USD", votes)
        assert len(result["signal_details"]) == 1
        detail = result["signal_details"][0]
        assert detail["name"] == "rsi"
        assert detail["vote"] == "BUY"
        assert detail["accuracy"] == pytest.approx(0.71, abs=0.01)
        assert detail["samples"] == 89
        assert detail["p_up"] == pytest.approx(0.71, abs=0.01)

    @patch("portfolio.ticker_accuracy.accuracy_by_ticker_signal")
    def test_single_signal_only(self, mock_acc):
        from portfolio.ticker_accuracy import direction_probability
        mock_acc.return_value = {
            "volume": {"accuracy": 0.65, "samples": 50, "correct": 33},
        }
        votes = {"volume": "SELL"}

        result = direction_probability("XAG-USD", votes)
        # SELL at 65% → P(up) = 0.35
        assert result["probability"] == pytest.approx(0.35, abs=0.01)
        assert result["direction"] == "down"


# ============================================================
# get_focus_probabilities
# ============================================================

class TestGetFocusProbabilities:
    """Tests for get_focus_probabilities()."""

    @patch("portfolio.ticker_accuracy.direction_probability")
    @patch("portfolio.ticker_accuracy.accuracy_by_ticker_signal")
    def test_basic_two_tickers(self, mock_acc, mock_prob):
        from portfolio.ticker_accuracy import get_focus_probabilities

        mock_prob.return_value = {
            "direction": "up", "probability": 0.72,
            "signals_used": 4, "total_samples": 312, "signal_details": [],
        }
        mock_acc.return_value = {
            "rsi": {"accuracy": 0.71, "samples": 89, "correct": 63},
        }

        data = {
            "XAG-USD": {"extra": {"_votes": {"rsi": "BUY"}}},
            "BTC-USD": {"extra": {"_votes": {"rsi": "BUY"}}},
        }

        result = get_focus_probabilities(["XAG-USD", "BTC-USD"], data)
        assert "XAG-USD" in result
        assert "BTC-USD" in result
        assert "1d" in result["XAG-USD"]
        assert "3h" in result["XAG-USD"]
        assert "3d" in result["XAG-USD"]
        assert "accuracy_1d" in result["XAG-USD"]
        assert "accuracy_samples" in result["XAG-USD"]

    @patch("portfolio.ticker_accuracy.direction_probability")
    @patch("portfolio.ticker_accuracy.accuracy_by_ticker_signal")
    def test_missing_ticker_data(self, mock_acc, mock_prob):
        from portfolio.ticker_accuracy import get_focus_probabilities

        data = {"XAG-USD": {"extra": {"_votes": {"rsi": "BUY"}}}}
        mock_prob.return_value = {
            "direction": "up", "probability": 0.7,
            "signals_used": 1, "total_samples": 100, "signal_details": [],
        }
        mock_acc.return_value = {"rsi": {"accuracy": 0.7, "samples": 100, "correct": 70}}

        result = get_focus_probabilities(["XAG-USD", "BTC-USD"], data)
        assert "XAG-USD" in result
        assert "BTC-USD" not in result  # no data for BTC-USD

    @patch("portfolio.ticker_accuracy.direction_probability")
    @patch("portfolio.ticker_accuracy.accuracy_by_ticker_signal")
    def test_custom_horizons(self, mock_acc, mock_prob):
        from portfolio.ticker_accuracy import get_focus_probabilities
        mock_prob.return_value = {
            "direction": "up", "probability": 0.6,
            "signals_used": 1, "total_samples": 50, "signal_details": [],
        }
        mock_acc.return_value = {"rsi": {"accuracy": 0.6, "samples": 50, "correct": 30}}

        data = {"XAG-USD": {"extra": {"_votes": {"rsi": "BUY"}}}}
        result = get_focus_probabilities(["XAG-USD"], data, horizons=["1d", "5d"])
        assert "1d" in result["XAG-USD"]
        assert "5d" in result["XAG-USD"]
        assert "3h" not in result["XAG-USD"]

    @patch("portfolio.ticker_accuracy.direction_probability")
    @patch("portfolio.ticker_accuracy.accuracy_by_ticker_signal")
    def test_empty_data(self, mock_acc, mock_prob):
        from portfolio.ticker_accuracy import get_focus_probabilities
        result = get_focus_probabilities(["XAG-USD"], {})
        assert result == {}


# ============================================================
# _extract_votes
# ============================================================

class TestExtractVotes:
    """Tests for _extract_votes() helper."""

    def test_from_extra_votes(self):
        from portfolio.ticker_accuracy import _extract_votes
        data = {"extra": {"_votes": {"rsi": "BUY", "ema": "SELL"}}}
        assert _extract_votes(data) == {"rsi": "BUY", "ema": "SELL"}

    def test_from_signals_dict(self):
        from portfolio.ticker_accuracy import _extract_votes
        data = {"signals": {"rsi": "BUY", "ema": "HOLD"}}
        assert _extract_votes(data) == {"rsi": "BUY", "ema": "HOLD"}

    def test_empty_data(self):
        from portfolio.ticker_accuracy import _extract_votes
        assert _extract_votes({}) == {}

    def test_all_hold_signals(self):
        from portfolio.ticker_accuracy import _extract_votes
        data = {"signals": {"rsi": "HOLD", "ema": "HOLD"}}
        # All HOLD → no active votes → returns empty
        assert _extract_votes(data) == {}

    def test_prefers_extra_votes(self):
        from portfolio.ticker_accuracy import _extract_votes
        data = {
            "extra": {"_votes": {"rsi": "BUY"}},
            "signals": {"rsi": "SELL"},
        }
        # extra._votes takes priority
        assert _extract_votes(data) == {"rsi": "BUY"}


# ============================================================
# Integration-style tests (mock at load_entries level)
# ============================================================

class TestIntegration:
    """End-to-end tests with mock entries."""

    @patch("portfolio.accuracy_stats.load_entries")
    def test_xag_known_accuracy_probability(self, mock_load):
        """Simulate XAG-USD with 71% RSI accuracy and a BUY vote."""
        from portfolio.ticker_accuracy import direction_probability, accuracy_by_ticker_signal
        entries = _make_entries_with_known_accuracy("XAG-USD", "rsi", 71, 100)
        mock_load.return_value = entries

        votes = {"rsi": "BUY"}
        result = direction_probability("XAG-USD", votes, horizon="1d", days=None)
        assert result["probability"] == pytest.approx(0.71, abs=0.01)
        assert result["direction"] == "up"

    @patch("portfolio.accuracy_stats.load_entries")
    def test_btc_coinflip_accuracy(self, mock_load):
        """BTC-USD with 50% accuracy → neutral."""
        from portfolio.ticker_accuracy import direction_probability
        entries = _make_entries_with_known_accuracy("BTC-USD", "rsi", 50, 100)
        mock_load.return_value = entries

        votes = {"rsi": "BUY"}
        result = direction_probability("BTC-USD", votes, horizon="1d", days=None)
        assert result["probability"] == pytest.approx(0.5, abs=0.01)
        assert result["direction"] == "neutral"

    @patch("portfolio.accuracy_stats.load_entries")
    def test_plan_verification_case(self, mock_load):
        """Plan verification: 3 signals BUY at 70%/60%/80% with 100/50/200 samples → ~72.8%."""
        from portfolio.ticker_accuracy import direction_probability

        # Build entries for each signal with known accuracy
        entries = []
        entries.extend(_make_entries_with_known_accuracy("XAG-USD", "rsi", 70, 100))
        entries.extend(_make_entries_with_known_accuracy("XAG-USD", "ema", 30, 50))
        entries.extend(_make_entries_with_known_accuracy("XAG-USD", "bb", 160, 200))
        mock_load.return_value = entries

        votes = {"rsi": "BUY", "ema": "BUY", "bb": "BUY"}
        result = direction_probability("XAG-USD", votes, horizon="1d", days=None)

        # Expected: sqrt-weighted average
        w1, w2, w3 = math.sqrt(100), math.sqrt(50), math.sqrt(200)
        expected = (0.7 * w1 + 0.6 * w2 + 0.8 * w3) / (w1 + w2 + w3)
        assert result["probability"] == pytest.approx(expected, abs=0.02)
        assert result["direction"] == "up"

    @patch("portfolio.accuracy_stats.load_entries")
    def test_3h_horizon(self, mock_load):
        """3h horizon uses 3h outcomes."""
        from portfolio.ticker_accuracy import accuracy_by_ticker_signal
        entries = [
            _make_entry("2026-02-20T01:00:00+00:00", "XAG-USD", {"rsi": "BUY"},
                        outcome_3h=2.0),
            _make_entry("2026-02-20T02:00:00+00:00", "XAG-USD", {"rsi": "BUY"},
                        outcome_3h=-1.0),
        ]
        mock_load.return_value = entries

        result = accuracy_by_ticker_signal("XAG-USD", horizon="3h")
        assert result["rsi"]["samples"] == 2
        assert result["rsi"]["correct"] == 1
        assert result["rsi"]["accuracy"] == pytest.approx(0.5)

    @patch("portfolio.accuracy_stats.load_entries")
    def test_multi_horizon_different_results(self, mock_load):
        """Different horizons can give different accuracy."""
        from portfolio.ticker_accuracy import accuracy_by_ticker_signal
        entries = [
            _make_entry("2026-02-20T01:00:00+00:00", "XAG-USD", {"rsi": "BUY"},
                        outcome_1d=2.0, outcome_3d=-1.0),
            _make_entry("2026-02-20T02:00:00+00:00", "XAG-USD", {"rsi": "BUY"},
                        outcome_1d=1.0, outcome_3d=3.0),
        ]
        mock_load.return_value = entries

        result_1d = accuracy_by_ticker_signal("XAG-USD", horizon="1d")
        result_3d = accuracy_by_ticker_signal("XAG-USD", horizon="3d")

        assert result_1d["rsi"]["accuracy"] == pytest.approx(1.0)  # both correct at 1d
        assert result_3d["rsi"]["accuracy"] == pytest.approx(0.5)  # one correct at 3d
