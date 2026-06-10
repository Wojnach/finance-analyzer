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
        """No recent data → alltime used directly (2026-04-18: now also
        gets a derived `pct` field = accuracy * 100)."""
        from portfolio.accuracy_stats import blend_accuracy_data

        alltime = {"rsi": {"accuracy": 0.6, "total": 100, "correct": 60}}
        result = blend_accuracy_data(alltime, None)
        # Same core fields, plus the new derived pct
        assert result["rsi"]["accuracy"] == 0.6
        assert result["rsi"]["total"] == 100
        assert result["rsi"]["correct"] == 60
        assert result["rsi"]["pct"] == 60.0


# ---------------------------------------------------------------------------
# Core function: _vote_correct
# ---------------------------------------------------------------------------

class TestVoteCorrect:

    def test_buy_positive_is_correct(self):
        from portfolio.accuracy_stats import _vote_correct
        assert _vote_correct("BUY", 1.5) is True

    def test_buy_negative_is_wrong(self):
        from portfolio.accuracy_stats import _vote_correct
        assert _vote_correct("BUY", -1.5) is False

    def test_sell_negative_is_correct(self):
        from portfolio.accuracy_stats import _vote_correct
        assert _vote_correct("SELL", -2.0) is True

    def test_sell_positive_is_wrong(self):
        from portfolio.accuracy_stats import _vote_correct
        assert _vote_correct("SELL", 2.0) is False

    def test_neutral_zone_returns_none(self):
        from portfolio.accuracy_stats import _vote_correct
        assert _vote_correct("BUY", 0.03) is None
        assert _vote_correct("SELL", -0.03) is None
        assert _vote_correct("BUY", 0.0) is None

    def test_custom_neutral_threshold(self):
        from portfolio.accuracy_stats import _vote_correct
        assert _vote_correct("BUY", 0.08, min_change_pct=0.10) is None
        assert _vote_correct("BUY", 0.12, min_change_pct=0.10) is True

    def test_boundary_at_threshold(self):
        from portfolio.accuracy_stats import _vote_correct
        assert _vote_correct("BUY", 0.05) is True
        assert _vote_correct("BUY", 0.049) is None

    def test_none_change_pct_is_neutral(self):
        """Regression: 2026-04-22 — missing outcome backfill at 4h+ horizons
        stored change_pct=None; abs(None) crashed --accuracy mid-report."""
        from portfolio.accuracy_stats import _vote_correct
        assert _vote_correct("BUY", None) is None
        assert _vote_correct("SELL", None) is None
        assert _vote_correct("BUY", None, min_change_pct=0.5) is None

    def test_signal_accuracy_logs_null_skip_count(self, caplog):
        """Regression: 2026-04-22 follow-up — silently dropping NULL outcomes
        would let an outcome_tracker data-quality regression go unnoticed.
        When any NULLs are encountered, log the count so drift is visible."""
        import logging

        from portfolio.accuracy_stats import signal_accuracy
        entries = [
            _make_entry("BTC-USD", "rsi", "BUY", 2.5),
            _make_entry("ETH-USD", "rsi", "BUY", None),
            _make_entry("XAU-USD", "rsi", "SELL", None),
        ]
        with caplog.at_level(logging.INFO, logger="portfolio.accuracy_stats"):
            signal_accuracy(horizon="1d", entries=entries)
        assert "change_pct=None" in caplog.text
        assert "skipped 2/3" in caplog.text


# ---------------------------------------------------------------------------
# Core function: signal_accuracy
# ---------------------------------------------------------------------------

class TestSignalAccuracy:

    def test_empty_entries_returns_zero_for_all(self, monkeypatch):
        from portfolio.accuracy_stats import signal_accuracy
        monkeypatch.setattr(acc_mod, "load_entries", lambda: [])
        result = signal_accuracy(horizon="1d", entries=[])
        for sig, stats in result.items():
            assert stats["total"] == 0
            assert stats["accuracy"] == 0.0

    # 2026-06-10 (audit batch 2): macd/ema are in DISABLED_SIGNALS, so their
    # untagged-row votes now land in the shadow_* bucket — headline
    # total/accuracy is consensus-eligible only. The all-votes view is
    # preserved in all_*/shadow_* keys.

    def test_single_correct_buy(self, monkeypatch):
        from portfolio.accuracy_stats import signal_accuracy
        entries = [_make_entry("BTC-USD", "macd", "BUY", 2.5)]
        result = signal_accuracy(horizon="1d", entries=entries)
        assert result["macd"]["total"] == 0  # disabled => shadow-only
        assert result["macd"]["shadow_total"] == 1
        assert result["macd"]["shadow_correct"] == 1
        assert result["macd"]["shadow_accuracy"] == 1.0
        assert result["macd"]["all_total"] == 1
        assert result["macd"]["all_accuracy"] == 1.0

    def test_single_wrong_sell(self, monkeypatch):
        from portfolio.accuracy_stats import signal_accuracy
        entries = [_make_entry("ETH-USD", "ema", "SELL", 3.0)]
        result = signal_accuracy(horizon="1d", entries=entries)
        assert result["ema"]["total"] == 0  # disabled => shadow-only
        assert result["ema"]["shadow_total"] == 1
        assert result["ema"]["shadow_correct"] == 0
        assert result["ema"]["shadow_accuracy"] == 0.0
        assert result["ema"]["all_total"] == 1

    def test_multiple_signals_independent(self, monkeypatch):
        from portfolio.accuracy_stats import signal_accuracy
        entry = {
            "ts": "2026-03-15T12:00:00+00:00",
            "tickers": {
                "BTC-USD": {
                    "signals": {"rsi": "BUY", "macd": "SELL", "ema": "HOLD"},
                    "consensus": "BUY",
                },
            },
            "outcomes": {"BTC-USD": {"1d": {"change_pct": 1.5}}},
        }
        result = signal_accuracy(horizon="1d", entries=[entry])
        assert result["rsi"]["total"] == 1  # active signal: headline counts it
        assert result["rsi"]["correct"] == 1
        assert result["macd"]["total"] == 0
        assert result["macd"]["shadow_total"] == 1
        assert result["macd"]["shadow_correct"] == 0
        assert result["ema"]["total"] == 0
        assert result["ema"]["shadow_total"] == 0  # HOLD never counts

    def test_since_filter(self, monkeypatch):
        from portfolio.accuracy_stats import signal_accuracy
        old = _make_entry("BTC-USD", "rsi", "BUY", 1.0)
        old["ts"] = "2026-01-01T00:00:00+00:00"
        new = _make_entry("BTC-USD", "rsi", "BUY", -1.0)
        new["ts"] = "2026-04-01T00:00:00+00:00"
        result = signal_accuracy(horizon="1d", since="2026-03-01", entries=[old, new])
        assert result["rsi"]["total"] == 1
        assert result["rsi"]["correct"] == 0

    def test_multi_ticker_entry(self, monkeypatch):
        from portfolio.accuracy_stats import signal_accuracy
        entry = {
            "ts": "2026-03-15T12:00:00+00:00",
            "tickers": {
                "BTC-USD": {"signals": {"rsi": "BUY"}, "consensus": "BUY"},
                "ETH-USD": {"signals": {"rsi": "SELL"}, "consensus": "SELL"},
            },
            "outcomes": {
                "BTC-USD": {"1d": {"change_pct": 2.0}},
                "ETH-USD": {"1d": {"change_pct": -1.0}},
            },
        }
        result = signal_accuracy(horizon="1d", entries=[entry])
        assert result["rsi"]["total"] == 2
        assert result["rsi"]["correct"] == 2

    def test_missing_outcome_skipped(self, monkeypatch):
        from portfolio.accuracy_stats import signal_accuracy
        entry = {
            "ts": "2026-03-15T12:00:00+00:00",
            "tickers": {"BTC-USD": {"signals": {"rsi": "BUY"}, "consensus": "BUY"}},
            "outcomes": {},
        }
        result = signal_accuracy(horizon="1d", entries=[entry])
        assert result["rsi"]["total"] == 0


# ---------------------------------------------------------------------------
# Core function: consensus_accuracy
# ---------------------------------------------------------------------------

class TestConsensusAccuracy:

    def test_correct_consensus(self, monkeypatch):
        from portfolio.accuracy_stats import consensus_accuracy
        entries = [_make_entry("BTC-USD", "rsi", "BUY", 2.0)]
        result = consensus_accuracy(horizon="1d", entries=entries)
        assert result["total"] == 1
        assert result["correct"] == 1
        assert result["accuracy"] == 1.0

    def test_wrong_consensus(self, monkeypatch):
        from portfolio.accuracy_stats import consensus_accuracy
        entries = [_make_entry("BTC-USD", "rsi", "SELL", 2.0)]
        result = consensus_accuracy(horizon="1d", entries=entries)
        assert result["total"] == 1
        assert result["correct"] == 0

    def test_hold_consensus_skipped(self, monkeypatch):
        from portfolio.accuracy_stats import consensus_accuracy
        entry = {
            "ts": "2026-03-15T12:00:00+00:00",
            "tickers": {"BTC-USD": {"signals": {"rsi": "HOLD"}, "consensus": "HOLD"}},
            "outcomes": {"BTC-USD": {"1d": {"change_pct": 5.0}}},
        }
        result = consensus_accuracy(horizon="1d", entries=[entry])
        assert result["total"] == 0

    def test_empty_entries(self, monkeypatch):
        from portfolio.accuracy_stats import consensus_accuracy
        result = consensus_accuracy(horizon="1d", entries=[])
        assert result["total"] == 0
        assert result["accuracy"] == 0.0

    def test_multiple_tickers_counted(self, monkeypatch):
        from portfolio.accuracy_stats import consensus_accuracy
        entry = {
            "ts": "2026-03-15T12:00:00+00:00",
            "tickers": {
                "BTC-USD": {"signals": {}, "consensus": "BUY"},
                "ETH-USD": {"signals": {}, "consensus": "SELL"},
            },
            "outcomes": {
                "BTC-USD": {"1d": {"change_pct": 1.0}},
                "ETH-USD": {"1d": {"change_pct": 1.0}},
            },
        }
        result = consensus_accuracy(horizon="1d", entries=[entry])
        assert result["total"] == 2
        assert result["correct"] == 1


# ---------------------------------------------------------------------------
# Core function: per_ticker_accuracy
# ---------------------------------------------------------------------------

class TestPerTickerAccuracy:

    def test_basic_per_ticker(self, monkeypatch):
        from portfolio.accuracy_stats import per_ticker_accuracy
        entries = [
            _make_entry("BTC-USD", "rsi", "BUY", 1.0),
            _make_entry("ETH-USD", "rsi", "SELL", -1.0),
        ]
        result = per_ticker_accuracy(horizon="1d", entries=entries)
        assert "BTC-USD" in result
        assert result["BTC-USD"]["correct"] == 1
        assert result["BTC-USD"]["total"] == 1
        assert "ETH-USD" in result
        assert result["ETH-USD"]["correct"] == 1

    def test_mixed_accuracy_per_ticker(self, monkeypatch):
        from portfolio.accuracy_stats import per_ticker_accuracy
        entries = [
            _make_entry("BTC-USD", "rsi", "BUY", 1.0),
            _make_entry("BTC-USD", "rsi", "BUY", -1.0),
            _make_entry("BTC-USD", "rsi", "BUY", 2.0),
        ]
        result = per_ticker_accuracy(horizon="1d", entries=entries)
        btc = result["BTC-USD"]
        assert btc["total"] == 3
        assert btc["correct"] == 2
        assert abs(btc["accuracy"] - 2 / 3) < 0.01

    def test_empty_returns_empty(self, monkeypatch):
        from portfolio.accuracy_stats import per_ticker_accuracy
        result = per_ticker_accuracy(horizon="1d", entries=[])
        assert result == {}


# ---------------------------------------------------------------------------
# 2026-05-05: enabled flag + samples alias + disable-reason helper
#
# These tests guard the dashboard accuracy view's "DISABLED" labelling
# pipeline. signal_accuracy() and friends must emit `enabled=False` and
# `samples == total` for every entry; tickers.get_disabled_reason() must
# return a short reason for each disabled signal and None for active ones.
# ---------------------------------------------------------------------------

class TestEnabledFlagAndSamplesAlias:

    def test_signal_accuracy_includes_enabled_and_samples(self):
        from portfolio.accuracy_stats import signal_accuracy
        from portfolio.tickers import DISABLED_SIGNALS, SIGNAL_NAMES

        entries = [
            _make_entry("BTC-USD", "rsi", "BUY", 1.5),
            _make_entry("BTC-USD", "rsi", "BUY", 1.0),
        ]
        result = signal_accuracy("1d", entries=entries)
        for sig_name in SIGNAL_NAMES:
            entry = result[sig_name]
            assert "enabled" in entry, f"missing enabled flag for {sig_name}"
            assert "samples" in entry, f"missing samples alias for {sig_name}"
            assert entry["samples"] == entry["total"]
            assert entry["enabled"] is (sig_name not in DISABLED_SIGNALS)

    def test_signal_accuracy_ewma_includes_enabled_and_samples(self):
        from portfolio.accuracy_stats import signal_accuracy_ewma
        from portfolio.tickers import DISABLED_SIGNALS

        entries = [_make_entry("BTC-USD", "rsi", "BUY", 1.5)]
        result = signal_accuracy_ewma("1d", entries=entries)
        rsi = result["rsi"]
        assert rsi["enabled"] is True
        assert rsi["samples"] == rsi["total"]
        # A disabled signal should be labelled even with zero samples.
        any_disabled = next(iter(DISABLED_SIGNALS))
        assert result[any_disabled]["enabled"] is False

    def test_signal_accuracy_cost_adjusted_includes_enabled_and_samples(self):
        from portfolio.accuracy_stats import signal_accuracy_cost_adjusted
        entries = [_make_entry("BTC-USD", "rsi", "BUY", 1.5)]
        result = signal_accuracy_cost_adjusted("1d", entries=entries)
        rsi = result["rsi"]
        assert rsi["enabled"] is True
        assert rsi["samples"] == rsi["total"]

    def test_blend_accuracy_data_includes_enabled_and_samples(self):
        from portfolio.accuracy_stats import blend_accuracy_data
        from portfolio.tickers import DISABLED_SIGNALS

        any_disabled = next(iter(DISABLED_SIGNALS))
        alltime = {
            "rsi": {"accuracy": 0.6, "total": 50, "correct": 30},
            any_disabled: {"accuracy": 0.4, "total": 20, "correct": 8},
        }
        recent = {
            "rsi": {"accuracy": 0.7, "total": 30, "correct": 21},
        }
        result = blend_accuracy_data(alltime, recent, min_recent_samples=20)
        assert result["rsi"]["enabled"] is True
        assert result["rsi"]["samples"] == result["rsi"]["total"]
        assert result[any_disabled]["enabled"] is False

    def test_catastrophic_floor_overrides_alltime(self):
        """When recent accuracy < 35% with 15+ samples, force-use recent."""
        from portfolio.accuracy_stats import blend_accuracy_data

        alltime = {"ministral": {"accuracy": 0.58, "total": 6303, "correct": 3655}}
        recent = {"ministral": {"accuracy": 0.25, "total": 20, "correct": 5}}
        result = blend_accuracy_data(alltime, recent, min_recent_samples=30)
        m = result["ministral"]
        assert m["accuracy"] == 0.25

    def test_catastrophic_floor_not_triggered_above_threshold(self):
        """Recent accuracy above 35% follows normal blend logic."""
        from portfolio.accuracy_stats import blend_accuracy_data

        alltime = {"rsi": {"accuracy": 0.52, "total": 33000, "correct": 17160}}
        recent = {"rsi": {"accuracy": 0.40, "total": 20, "correct": 8}}
        result = blend_accuracy_data(alltime, recent, min_recent_samples=30)
        r = result["rsi"]
        assert r["accuracy"] == 0.52  # alltime fallback (20 < 30 min_recent)

    def test_catastrophic_floor_requires_min_samples(self):
        """Below min catastrophic samples (15), alltime still used."""
        from portfolio.accuracy_stats import blend_accuracy_data

        alltime = {"ml": {"accuracy": 0.42, "total": 1714, "correct": 720}}
        recent = {"ml": {"accuracy": 0.10, "total": 10, "correct": 1}}
        result = blend_accuracy_data(alltime, recent, min_recent_samples=30)
        m = result["ml"]
        assert m["accuracy"] == 0.42  # alltime (10 < 15 catastrophic min)

    def test_signal_accuracy_by_regime_includes_enabled_and_samples(self):
        from portfolio.accuracy_stats import signal_accuracy_by_regime
        from portfolio.tickers import DISABLED_SIGNALS

        # Build entries with regime metadata.
        entries = [{
            "ts": "2026-03-15T12:00:00+00:00",
            "tickers": {
                "BTC-USD": {
                    "signals": {"rsi": "BUY"},
                    "consensus": "BUY",
                    "regime": "trending",
                },
            },
            "outcomes": {"BTC-USD": {"1d": {"change_pct": 1.5}}},
        }]
        result = signal_accuracy_by_regime("1d", entries=entries)
        assert "trending" in result
        rsi = result["trending"]["rsi"]
        assert rsi["enabled"] is True
        assert rsi["samples"] == rsi["total"]
        # Disabled signals only appear in the regime map if they had samples,
        # which they don't here — so we only assert the active-signal shape.
        assert "rsi" not in DISABLED_SIGNALS


class TestGetDisabledReason:

    def test_returns_none_for_active_signal(self):
        from portfolio.tickers import get_disabled_reason
        # rsi, bb are core actives. 2026-05-19: swapped ema → bb because
        # ema was disabled 2026-05-15 (46.2% blended 1d, 17917 sam).
        assert get_disabled_reason("rsi") is None
        assert get_disabled_reason("bb") is None

    def test_returns_none_for_unknown_signal(self):
        from portfolio.tickers import get_disabled_reason
        assert get_disabled_reason("not_a_real_signal_xyz") is None

    def test_returns_reason_for_every_disabled_signal(self):
        from portfolio.tickers import DISABLED_SIGNALS, get_disabled_reason
        # Every entry in DISABLED_SIGNALS is annotated with a comment in
        # tickers.py — the parser must extract a non-empty reason for each.
        # If this fails, an entry was added without an inline comment.
        empty = sorted(
            name for name in DISABLED_SIGNALS
            if not (get_disabled_reason(name) or "").strip()
        )
        assert empty == [], f"missing/empty disable reason for: {empty}"

    def test_reason_is_short_and_single_line(self):
        from portfolio.tickers import get_disabled_reason
        r = get_disabled_reason("hash_ribbons")
        assert r is not None
        assert len(r) <= 160
        assert "\n" not in r

    def test_separator_comments_do_not_bleed_into_adjacent_entries(self):
        """Regression guard for the parser's column-aware continuation rule.

        The DISABLED_SIGNALS literal contains flush-left separator comments
        about *re-enabled* signals (cot_positioning, statistical_jump_regime,
        forecast, econ_calendar) sitting between disabled entries. A naive
        line-by-line continuation parser would absorb that text into the
        previous entry's reason. We require the helper to ignore comments
        whose `#` indent is not strictly greater than the entry-name indent.
        """
        from portfolio.tickers import get_disabled_reason

        # Each pair: entry whose reason might absorb the following separator.
        # If any of these substrings appears in the reason, the parser is
        # leaking separator text into the entry above.
        leak_pairs = [
            ("fibonacci", "cot_positioning"),
            ("copper_gold_ratio", "statistical_jump_regime"),
            ("complexity_gap_regime", "econ_calendar"),
            ("orderbook_flow", "forecast"),
        ]
        for entry, leaked in leak_pairs:
            reason = get_disabled_reason(entry) or ""
            assert leaked not in reason, (
                f"reason for {entry!r} leaked separator text about {leaked!r}: {reason!r}"
            )


class TestApiAccuracyEnrichment:
    """Verify the dashboard's response-layer enrichment overwrites stale flags.

    The endpoint must re-derive `enabled` and `disabled_reason` from the
    live DISABLED_SIGNALS on every request, not preserve whatever was
    cached. Otherwise a re-enabled signal stays labelled as disabled
    until the 1h accuracy cache rebuilds.
    """

    def test_enrich_signals_overwrites_stale_enabled_flag(self, monkeypatch):
        from dashboard import app as dash

        # Build a fake stale cache where everything claims to be disabled.
        stale_signals = {
            "rsi": {"total": 100, "correct": 50, "pct": 50.0, "enabled": False,
                    "disabled_reason": "stale leftover from a previous run"},
            # And a genuinely disabled one with no enabled flag at all.
            "hash_ribbons": {"total": 0, "correct": 0, "pct": 0.0},
        }
        stale_consensus = {"correct": 5, "total": 10, "accuracy": 0.5, "pct": 50.0}

        def fake_acc(_horizon):
            # Return a fresh copy each call so tests don't see prior mutations.
            import copy
            return copy.deepcopy(stale_signals)

        def fake_consensus(_horizon):
            return dict(stale_consensus)

        def fake_per_ticker(_horizon):
            return {}

        monkeypatch.setattr(
            "portfolio.accuracy_stats.get_or_compute_accuracy", fake_acc,
        )
        monkeypatch.setattr(
            "portfolio.accuracy_stats.get_or_compute_consensus_accuracy",
            fake_consensus,
        )
        monkeypatch.setattr(
            "portfolio.accuracy_stats.get_or_compute_per_ticker_accuracy",
            fake_per_ticker,
        )
        # Bust the 60s in-process TTL so our monkeypatched compute is hit.
        dash._API_ACCURACY_CACHE["data"] = None
        dash._API_ACCURACY_CACHE["ts"] = 0

        client = dash.app.test_client()
        # 2026-05-10: bearer auth path required a non-empty DASHBOARD_TOKEN
        # which is unset in test config — every assertion 401'd. Use the
        # same _no_auth pattern test_dashboard.py uses (patches
        # ``dashboard.auth._get_dashboard_token`` to return None,
        # disabling the gate cleanly).
        from unittest.mock import patch as _patch
        with _patch("dashboard.auth._get_dashboard_token", return_value=None):
            resp = client.get("/api/accuracy")
        assert resp.status_code == 200, resp.data
        body = resp.get_json()
        assert "1d" in body
        sigs = body["1d"]["signals"]

        # rsi was stale-tagged disabled; enrichment must overwrite to True
        # and clear the bogus disabled_reason.
        assert sigs["rsi"]["enabled"] is True
        assert "disabled_reason" not in sigs["rsi"]
        assert sigs["rsi"]["samples"] == sigs["rsi"]["total"]

        # hash_ribbons is genuinely disabled — flag must be set False
        # and reason populated from the inline comment.
        assert sigs["hash_ribbons"]["enabled"] is False
        assert sigs["hash_ribbons"].get("disabled_reason")


# ---------------------------------------------------------------------------
# 2026-06-10 (audit batch 2): consensus-eligible vs shadow-only separation
# ---------------------------------------------------------------------------

class TestShadowConsensusSeparation:
    """signal_accuracy must not conflate shadow votes with consensus votes.
    Row tag (shadow_signals) wins; untagged rows fall back to config."""

    def test_config_fallback_per_ticker_override(self):
        """realized_skewness: eligible on XAU-USD (override), shadow on XAG-USD."""
        from portfolio.accuracy_stats import is_consensus_eligible, signal_accuracy

        assert is_consensus_eligible("realized_skewness", "XAU-USD") is True
        assert is_consensus_eligible("realized_skewness", "XAG-USD") is False

        entries = [
            _make_entry("XAU-USD", "realized_skewness", "BUY", 2.0),
            _make_entry("XAG-USD", "realized_skewness", "BUY", 2.0),
        ]
        result = signal_accuracy(horizon="1d", entries=entries)
        stats = result["realized_skewness"]
        assert stats["total"] == 1          # XAU only (consensus-eligible)
        assert stats["shadow_total"] == 1   # XAG (shadow scope)
        assert stats["all_total"] == 2

    def test_row_tag_overrides_config(self):
        """A row tagging an ACTIVE signal as shadow is honored (write-time
        truth beats config reconstruction)."""
        from portfolio.accuracy_stats import signal_accuracy

        entry = _make_entry("BTC-USD", "rsi", "BUY", 2.0)
        entry["tickers"]["BTC-USD"]["shadow_signals"] = ["rsi"]
        result = signal_accuracy(horizon="1d", entries=[entry])
        assert result["rsi"]["total"] == 0
        assert result["rsi"]["shadow_total"] == 1

    def test_empty_tag_means_consensus_vote(self):
        """A tagged row with an empty shadow list counts a globally-disabled
        signal as consensus — covers shadow-registry promotions, where the
        signal votes in consensus while DISABLED_SIGNALS still lists it."""
        from portfolio.accuracy_stats import signal_accuracy

        entry = _make_entry("BTC-USD", "macd", "BUY", 2.0)
        entry["tickers"]["BTC-USD"]["shadow_signals"] = []
        result = signal_accuracy(horizon="1d", entries=[entry])
        assert result["macd"]["total"] == 1
        assert result["macd"]["shadow_total"] == 0

    def test_write_time_tag_from_extra_flags(self):
        """outcome_tracker.log_signal_snapshot persists shadow_<name> flags
        as the per-ticker shadow_signals list."""
        from unittest.mock import patch as _patch

        from portfolio.outcome_tracker import log_signal_snapshot

        signals_dict = {
            "BTC-USD": {
                "indicators": {},
                "extra": {
                    "_raw_votes": {"rsi": "BUY", "crypto_evrp": "SELL"},
                    "shadow_crypto_evrp": True,
                },
                "action": "BUY",
            }
        }
        with _patch("portfolio.outcome_tracker.atomic_append_jsonl"), \
             _patch("portfolio.outcome_tracker.SignalDB", create=True) as db:
            db.side_effect = ImportError("no db")
            entry = log_signal_snapshot(signals_dict, {"BTC-USD": 67000.0}, 10.5, ["t"])

        assert entry["tickers"]["BTC-USD"]["shadow_signals"] == ["crypto_evrp"]

    def test_signal_db_roundtrips_shadow_tag(self, tmp_path):
        """SignalDB must persist and restore the shadow_signals tag; legacy
        rows (no tag) come back without the key."""
        from portfolio.signal_db import SignalDB

        db = SignalDB(db_path=tmp_path / "t.db")
        try:
            tagged = {
                "ts": "2026-06-10T10:00:00+00:00",
                "trigger_reasons": [],
                "fx_rate": 10.5,
                "tickers": {"BTC-USD": {
                    "price_usd": 67000.0, "consensus": "BUY",
                    "buy_count": 1, "sell_count": 0, "total_voters": 1,
                    "signals": {"rsi": "BUY"}, "regime": "ranging",
                    "shadow_signals": ["crypto_evrp"],
                }},
                "outcomes": {},
            }
            legacy = {
                "ts": "2026-06-10T11:00:00+00:00",
                "trigger_reasons": [],
                "fx_rate": 10.5,
                "tickers": {"BTC-USD": {
                    "price_usd": 67000.0, "consensus": "BUY",
                    "buy_count": 1, "sell_count": 0, "total_voters": 1,
                    "signals": {"rsi": "BUY"}, "regime": "ranging",
                }},
                "outcomes": {},
            }
            db.insert_snapshot(tagged)
            db.insert_snapshot(legacy)
            entries = db.load_entries()
        finally:
            db.close()

        assert entries[0]["tickers"]["BTC-USD"]["shadow_signals"] == ["crypto_evrp"]
        assert "shadow_signals" not in entries[1]["tickers"]["BTC-USD"]
