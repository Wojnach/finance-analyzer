"""Tests for portfolio.accuracy_stats — cost-adjusted signal accuracy."""


import portfolio.accuracy_stats as acc_mod
from portfolio.accuracy_stats import signal_accuracy_cost_adjusted

# ---------------------------------------------------------------------------
# Helper: build fake signal log entries
# ---------------------------------------------------------------------------

def _make_entry(ticker, signal_name, vote, change_pct, horizon="1d"):
    """Build a minimal signal log entry for one ticker + one signal."""
    return {
        "ts": "2026-03-15T12:00:00+00:00",
        "tickers": {
            ticker: {
                "signals": {signal_name: vote},
                "consensus": vote,
            },
        },
        "outcomes": {
            ticker: {horizon: {"change_pct": change_pct}},
        },
    }


# ---------------------------------------------------------------------------
# 3a. Cost-adjusted accuracy tests
# ---------------------------------------------------------------------------

class TestSignalAccuracyCostAdjusted:

    def test_cost_adjusted_accuracy_filters_small_moves(self, monkeypatch):
        """A BUY with +0.03% change should NOT be counted as correct at 5 bps cost.

        cost_pct = 5.0 / 100 = 0.05%.  A +0.03% move is below the cost threshold,
        so even though direction is correct, it's not profitable after costs.
        """
        entries = [_make_entry("BTC-USD", "rsi", "BUY", 0.03)]
        # monkeypatch load_entries so it doesn't hit disk
        monkeypatch.setattr(acc_mod, "load_entries", lambda: entries)

        result = signal_accuracy_cost_adjusted(
            horizon="1d", cost_bps=5.0, entries=entries
        )
        # The entry should not count as correct (move < cost_pct)
        # Note: _MIN_CHANGE_PCT = 0.05 means |0.03| < 0.05, so the entry
        # is skipped as neutral entirely. total should be 0.
        rsi_stats = result.get("rsi", {})
        # Either total=0 (skipped as neutral) or correct=0 (below cost)
        assert rsi_stats.get("correct", 0) == 0

    def test_cost_adjusted_accuracy_filters_small_moves_above_neutral(self, monkeypatch):
        """A BUY with +0.04% change exceeds _MIN_CHANGE_PCT but is below 5bps cost.

        change_pct=0.04 is below 0.05 neutral threshold, so also skipped.
        Let's use 0.06% which is above neutral (0.05) but below 10bps cost.
        """
        entries = [_make_entry("BTC-USD", "rsi", "BUY", 0.06)]
        monkeypatch.setattr(acc_mod, "load_entries", lambda: entries)

        result = signal_accuracy_cost_adjusted(
            horizon="1d", cost_bps=10.0, entries=entries  # 10 bps = 0.10%
        )
        rsi_stats = result.get("rsi", {})
        # 0.06% move > _MIN_CHANGE_PCT (0.05), so it counts as total
        # but 0.06% < cost_pct (0.10%), so not correct
        assert rsi_stats.get("total", 0) == 1
        assert rsi_stats.get("correct", 0) == 0

    def test_cost_adjusted_accuracy_accepts_large_moves(self, monkeypatch):
        """A BUY with +0.10% change should be counted as correct at 5 bps cost.

        cost_pct = 5.0 / 100 = 0.05%.  A +0.10% move exceeds the cost,
        so the directionally-correct BUY counts as profitable.
        """
        entries = [_make_entry("BTC-USD", "rsi", "BUY", 0.10)]
        monkeypatch.setattr(acc_mod, "load_entries", lambda: entries)

        result = signal_accuracy_cost_adjusted(
            horizon="1d", cost_bps=5.0, entries=entries
        )
        rsi_stats = result.get("rsi", {})
        assert rsi_stats["total"] == 1
        assert rsi_stats["correct"] == 1
        assert rsi_stats["accuracy"] == 1.0

    def test_cost_adjusted_sell_large_negative_move(self, monkeypatch):
        """A SELL with -0.20% change should be correct at 5 bps cost."""
        entries = [_make_entry("ETH-USD", "macd", "SELL", -0.20)]
        monkeypatch.setattr(acc_mod, "load_entries", lambda: entries)

        result = signal_accuracy_cost_adjusted(
            horizon="1d", cost_bps=5.0, entries=entries
        )
        macd_stats = result.get("macd", {})
        assert macd_stats["total"] == 1
        assert macd_stats["correct"] == 1

    def test_cost_adjusted_sell_small_negative_move(self, monkeypatch):
        """A SELL with -0.06% is directionally correct but below 10bps cost threshold."""
        entries = [_make_entry("ETH-USD", "macd", "SELL", -0.06)]
        monkeypatch.setattr(acc_mod, "load_entries", lambda: entries)

        result = signal_accuracy_cost_adjusted(
            horizon="1d", cost_bps=10.0, entries=entries
        )
        macd_stats = result.get("macd", {})
        assert macd_stats["total"] == 1
        assert macd_stats["correct"] == 0  # -0.06% not below -0.10% threshold

    def test_cost_adjusted_hold_votes_skipped(self, monkeypatch):
        """HOLD votes should not count in accuracy at all."""
        entries = [_make_entry("BTC-USD", "rsi", "HOLD", 5.0)]
        monkeypatch.setattr(acc_mod, "load_entries", lambda: entries)

        result = signal_accuracy_cost_adjusted(
            horizon="1d", cost_bps=5.0, entries=entries
        )
        rsi_stats = result.get("rsi", {})
        assert rsi_stats["total"] == 0

    def test_cost_adjusted_returns_cost_bps_in_result(self, monkeypatch):
        """Result dict should include the cost_bps used."""
        entries = [_make_entry("BTC-USD", "rsi", "BUY", 1.0)]
        monkeypatch.setattr(acc_mod, "load_entries", lambda: entries)

        result = signal_accuracy_cost_adjusted(
            horizon="1d", cost_bps=7.5, entries=entries
        )
        rsi_stats = result.get("rsi", {})
        assert rsi_stats["cost_bps"] == 7.5

    def test_cost_adjusted_empty_entries(self, monkeypatch):
        """No entries → all signals have 0 total, 0 accuracy."""
        monkeypatch.setattr(acc_mod, "load_entries", lambda: [])

        result = signal_accuracy_cost_adjusted(horizon="1d", entries=[])
        for sig_name, stats in result.items():
            assert stats["total"] == 0
            assert stats["correct"] == 0
            assert stats["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# 3b. Directional accuracy tracking
# ---------------------------------------------------------------------------

class TestDirectionalAccuracy:

    def test_buy_accuracy_tracked_separately(self, monkeypatch):
        """BUY votes with positive change should increment correct_buy."""
        from portfolio.accuracy_stats import signal_accuracy

        entries = [_make_entry("BTC-USD", "rsi", "BUY", 1.0)]
        monkeypatch.setattr(acc_mod, "load_entries", lambda: entries)

        result = signal_accuracy(horizon="1d", entries=entries)
        rsi = result["rsi"]
        assert rsi["total_buy"] == 1
        assert rsi["correct_buy"] == 1
        assert rsi["buy_accuracy"] == 1.0
        assert rsi["total_sell"] == 0
        assert rsi["correct_sell"] == 0

    def test_sell_accuracy_tracked_separately(self, monkeypatch):
        """SELL votes with negative change should increment correct_sell."""
        from portfolio.accuracy_stats import signal_accuracy

        entries = [_make_entry("BTC-USD", "rsi", "SELL", -2.0)]
        monkeypatch.setattr(acc_mod, "load_entries", lambda: entries)

        result = signal_accuracy(horizon="1d", entries=entries)
        rsi = result["rsi"]
        assert rsi["total_sell"] == 1
        assert rsi["correct_sell"] == 1
        assert rsi["sell_accuracy"] == 1.0
        assert rsi["total_buy"] == 0

    def test_wrong_buy_increments_total_but_not_correct(self, monkeypatch):
        """BUY vote with negative change should be total_buy=1, correct_buy=0."""
        from portfolio.accuracy_stats import signal_accuracy

        entries = [_make_entry("BTC-USD", "rsi", "BUY", -3.0)]
        monkeypatch.setattr(acc_mod, "load_entries", lambda: entries)

        result = signal_accuracy(horizon="1d", entries=entries)
        rsi = result["rsi"]
        assert rsi["total_buy"] == 1
        assert rsi["correct_buy"] == 0
        assert rsi["buy_accuracy"] == 0.0
        assert rsi["total"] == 1
        assert rsi["correct"] == 0

    def test_mixed_buy_sell_accuracy(self, monkeypatch):
        """Multiple votes in different directions should be tracked correctly."""
        from portfolio.accuracy_stats import signal_accuracy

        entries = [
            _make_entry("BTC-USD", "rsi", "BUY", 1.0),   # correct BUY
            _make_entry("BTC-USD", "rsi", "BUY", -1.0),  # wrong BUY
            _make_entry("BTC-USD", "rsi", "SELL", -1.0),  # correct SELL
        ]
        monkeypatch.setattr(acc_mod, "load_entries", lambda: entries)

        result = signal_accuracy(horizon="1d", entries=entries)
        rsi = result["rsi"]
        assert rsi["total_buy"] == 2
        assert rsi["correct_buy"] == 1
        assert rsi["buy_accuracy"] == 0.5
        assert rsi["total_sell"] == 1
        assert rsi["correct_sell"] == 1
        assert rsi["sell_accuracy"] == 1.0
        assert rsi["total"] == 3
        assert rsi["correct"] == 2

    def test_hold_not_counted_in_directional(self, monkeypatch):
        """HOLD votes should not appear in buy or sell counts."""
        from portfolio.accuracy_stats import signal_accuracy

        entries = [_make_entry("BTC-USD", "rsi", "HOLD", 5.0)]
        monkeypatch.setattr(acc_mod, "load_entries", lambda: entries)

        result = signal_accuracy(horizon="1d", entries=entries)
        rsi = result["rsi"]
        assert rsi["total_buy"] == 0
        assert rsi["total_sell"] == 0
        assert rsi["total"] == 0


# ---------------------------------------------------------------------------
# BUG-186: Blended accuracy correct field consistency
# ---------------------------------------------------------------------------

class TestBlendAccuracyData:
    """Tests for blend_accuracy_data() — verifies correct/total matches accuracy."""

    def test_correct_derived_from_blended_accuracy(self):
        """BUG-186: correct should equal round(blended_accuracy * total)."""
        from portfolio.accuracy_stats import blend_accuracy_data

        alltime = {"rsi": {"accuracy": 0.5, "total": 100, "correct": 50}}
        recent = {"rsi": {"accuracy": 0.9, "total": 40, "correct": 36}}
        result = blend_accuracy_data(alltime, recent, min_recent_samples=30)
        rsi = result["rsi"]
        # blended = 0.7 * 0.9 + 0.3 * 0.5 = 0.78
        # total = max(100, 40) = 100
        # correct should be round(0.78 * 100) = 78, NOT 50 (all-time)
        assert rsi["correct"] == round(rsi["accuracy"] * rsi["total"])

    def test_correct_total_ratio_equals_accuracy(self):
        """The ratio correct/total should approximately equal accuracy."""
        from portfolio.accuracy_stats import blend_accuracy_data

        alltime = {"ema": {"accuracy": 0.6, "total": 200, "correct": 120}}
        recent = {"ema": {"accuracy": 0.8, "total": 50, "correct": 40}}
        result = blend_accuracy_data(alltime, recent, min_recent_samples=30)
        ema = result["ema"]
        if ema["total"] > 0:
            ratio = ema["correct"] / ema["total"]
            assert abs(ratio - ema["accuracy"]) < 0.02  # within rounding tolerance

    def test_blend_with_insufficient_recent_samples(self):
        """When recent samples < min, use alltime accuracy directly."""
        from portfolio.accuracy_stats import blend_accuracy_data

        alltime = {"rsi": {"accuracy": 0.65, "total": 100, "correct": 65}}
        recent = {"rsi": {"accuracy": 0.9, "total": 5, "correct": 4}}  # < 30
        result = blend_accuracy_data(alltime, recent, min_recent_samples=30)
        rsi = result["rsi"]
        assert rsi["accuracy"] == 0.65  # alltime used directly
        assert rsi["correct"] == round(0.65 * 100)

    def test_blend_carries_directional_fields(self):
        """Directional accuracy fields from alltime should be preserved."""
        from portfolio.accuracy_stats import blend_accuracy_data

        alltime = {
            "rsi": {
                "accuracy": 0.55, "total": 100, "correct": 55,
                "buy_accuracy": 0.60, "total_buy": 50,
                "sell_accuracy": 0.50, "total_sell": 50,
            }
        }
        recent = {"rsi": {"accuracy": 0.70, "total": 40, "correct": 28}}
        result = blend_accuracy_data(alltime, recent, min_recent_samples=30)
        rsi = result["rsi"]
        assert rsi["buy_accuracy"] == 0.60
        assert rsi["sell_accuracy"] == 0.50
        assert rsi["total_buy"] == 50

    def test_empty_inputs_return_empty(self):
        """Both empty → empty result."""
        from portfolio.accuracy_stats import blend_accuracy_data

        assert blend_accuracy_data({}, {}) == {}
        assert blend_accuracy_data(None, None) == {}

    def test_alltime_only_returns_alltime(self):
        """No recent data → alltime used directly."""
        from portfolio.accuracy_stats import blend_accuracy_data

        alltime = {"rsi": {"accuracy": 0.6, "total": 100, "correct": 60}}
        result = blend_accuracy_data(alltime, None)
        assert result == alltime
