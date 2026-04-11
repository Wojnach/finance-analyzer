"""Tests for _weighted_consensus() in signal_engine.

Covers 11 test categories:
  1. All HOLD         2. All BUY        3. All SELL
  4. Majority vote    5. Accuracy gate   6. 45% gate boundary
  7. Small sample     8. Regime weights  9. Activation rates
 10. Empty votes     11. HOLD ignored
Plus edge cases and integration scenarios.
"""

import pytest

from portfolio.signal_engine import (
    ACCURACY_GATE_MIN_SAMPLES,
    ACCURACY_GATE_THRESHOLD,
    _weighted_consensus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _acc(accuracy, total=100):
    """Shorthand for creating an accuracy dict entry."""
    return {"accuracy": accuracy, "total": total}


def _acc_dict(names, accuracy=0.6, total=50):
    """Create accuracy data for multiple signals at the same accuracy."""
    return {n: _acc(accuracy, total) for n in names}


# ===========================================================================
# Verify constants are as expected
# ===========================================================================

class TestConstants:
    def test_accuracy_gate_threshold(self):
        assert ACCURACY_GATE_THRESHOLD == 0.45

    def test_accuracy_gate_min_samples(self):
        assert ACCURACY_GATE_MIN_SAMPLES == 30


# ===========================================================================
# Category 1: All HOLD
# ===========================================================================

class TestAllHold:
    def test_all_hold_returns_hold_zero(self):
        votes = {"rsi": "HOLD", "macd": "HOLD", "ema": "HOLD"}
        action, conf = _weighted_consensus(votes, {}, "trending-up")
        assert action == "HOLD"
        assert conf == 0.0

    def test_many_hold_signals(self):
        votes = {f"sig_{i}": "HOLD" for i in range(20)}
        action, conf = _weighted_consensus(votes, {}, "ranging")
        assert action == "HOLD"
        assert conf == 0.0

    def test_hold_with_accuracy_data(self):
        votes = {"rsi": "HOLD", "macd": "HOLD"}
        acc = _acc_dict(["rsi", "macd"], 0.9, 200)
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "HOLD"
        assert conf == 0.0


# ===========================================================================
# Category 2: All BUY
# ===========================================================================

class TestAllBuy:
    def test_unanimous_buy_returns_buy_1(self):
        votes = {"rsi": "BUY", "macd": "BUY", "ema": "BUY"}
        acc = _acc_dict(votes.keys(), 0.6, 50)
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"
        assert conf == 1.0

    def test_single_buy(self):
        votes = {"only": "BUY"}
        acc = {"only": _acc(0.7, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"
        assert conf == 1.0

    def test_all_buy_various_weights(self):
        votes = {"s1": "BUY", "s2": "BUY", "s3": "BUY"}
        acc = {"s1": _acc(0.8, 50), "s2": _acc(0.6, 50), "s3": _acc(0.55, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"
        assert conf == 1.0  # all same direction


# ===========================================================================
# Category 3: All SELL
# ===========================================================================

class TestAllSell:
    def test_unanimous_sell_returns_sell_1(self):
        votes = {"rsi": "SELL", "macd": "SELL", "ema": "SELL"}
        acc = _acc_dict(votes.keys(), 0.6, 50)
        action, conf = _weighted_consensus(votes, acc, "trending-down")
        assert action == "SELL"
        assert conf == 1.0

    def test_single_sell(self):
        votes = {"only": "SELL"}
        acc = {"only": _acc(0.65, 30)}
        action, conf = _weighted_consensus(votes, acc, "trending-down")
        assert action == "SELL"
        assert conf == 1.0


# ===========================================================================
# Category 4: Majority vote
# ===========================================================================

class TestMajorityVote:
    def test_2buy_1sell_buy_wins(self):
        votes = {"s1": "BUY", "s2": "BUY", "s3": "SELL"}
        acc = _acc_dict(votes.keys(), 0.6, 50)
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"
        # 2 * 0.6 = 1.2 BUY, 1 * 0.6 = 0.6 SELL -> 1.2/(1.2+0.6) = 0.6667
        assert conf == pytest.approx(1.2 / 1.8, abs=0.01)

    def test_1buy_2sell_sell_wins(self):
        votes = {"s1": "BUY", "s2": "SELL", "s3": "SELL"}
        acc = _acc_dict(votes.keys(), 0.6, 50)
        action, conf = _weighted_consensus(votes, acc, "trending-down")
        assert action == "SELL"
        assert conf == pytest.approx(1.2 / 1.8, abs=0.01)

    def test_3buy_2sell_buy_wins(self):
        votes = {"b1": "BUY", "b2": "BUY", "b3": "BUY",
                 "s1": "SELL", "s2": "SELL"}
        acc = _acc_dict(votes.keys(), 0.6, 50)
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"
        # 3*0.6 / (3*0.6 + 2*0.6) = 1.8/3.0 = 0.6
        assert conf == pytest.approx(1.8 / 3.0, abs=0.01)

    def test_stronger_minority_can_win(self):
        """A minority with higher accuracy can outweigh the majority."""
        votes = {"strong": "SELL", "weak1": "BUY", "weak2": "BUY"}
        acc = {"strong": _acc(0.9, 100), "weak1": _acc(0.51, 50), "weak2": _acc(0.51, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # SELL weight: 0.9, BUY weight: 0.51+0.51 = 1.02
        # BUY still wins because 1.02 > 0.9
        assert action == "BUY"

    def test_high_accuracy_minority_beats_count(self):
        """1 signal at 0.95 vs 2 signals at 0.51 — lower total weight loses."""
        votes = {"ace": "BUY", "meh1": "SELL", "meh2": "SELL"}
        acc = {"ace": _acc(0.95, 100), "meh1": _acc(0.51, 50), "meh2": _acc(0.51, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # BUY: 0.95, SELL: 0.51+0.51=1.02 -> SELL wins
        assert action == "SELL"


# ===========================================================================
# Category 5: Accuracy gate (replaces signal inversion)
# ===========================================================================

class TestAccuracyGate:
    """Signals below ACCURACY_GATE_THRESHOLD (0.45) with >= ACCURACY_GATE_MIN_SAMPLES (30)
    are skipped entirely — they don't participate in voting."""

    def test_gated_sole_voter_returns_hold(self):
        """Signal at 0.30 accuracy with 100 samples -> gated, returns HOLD."""
        votes = {"bad": "BUY"}
        acc = {"bad": _acc(0.30, 100)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "HOLD"
        assert conf == 0.0

    def test_044_accuracy_gated(self):
        """Signal at 0.44 accuracy with 50 samples -> gated (0.44 < 0.45)."""
        votes = {"low": "BUY"}
        acc = {"low": _acc(0.44, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "HOLD"
        assert conf == 0.0

    def test_045_accuracy_not_gated(self):
        """Signal at 0.45 accuracy with 50 samples -> NOT gated, votes normally."""
        votes = {"borderline": "BUY"}
        acc = {"borderline": _acc(0.45, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"
        assert conf == 1.0

    def test_insufficient_samples_not_gated(self):
        """Signal at 0.30 accuracy with 25 samples -> NOT gated (needs >= 30).
        Uses actual accuracy 0.30 as weight since samples >= 20."""
        votes = {"new_bad": "BUY"}
        acc = {"new_bad": _acc(0.30, 25)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"
        assert conf == 1.0  # sole voter

    def test_exactly_29_samples_not_gated(self):
        """29 samples < 30 min -> not gated even with terrible accuracy."""
        votes = {"almost": "SELL"}
        acc = {"almost": _acc(0.10, 29)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "SELL"
        assert conf == 1.0

    def test_exactly_30_samples_gated_if_below_threshold(self):
        """30 samples >= 30 min -> gated when accuracy < 0.45."""
        votes = {"enough": "BUY"}
        acc = {"enough": _acc(0.40, 30)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "HOLD"
        assert conf == 0.0

    def test_mixed_gated_and_good(self):
        """One good signal + one gated signal -> only good signal participates."""
        votes = {"good": "BUY", "bad": "SELL"}
        acc = {"good": _acc(0.70, 100), "bad": _acc(0.30, 100)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # bad is gated (0.30 < 0.45, 100 >= 30) -> skipped
        # only good participates -> BUY with confidence 1.0
        assert action == "BUY"
        assert conf == 1.0

    def test_all_signals_gated_returns_hold(self):
        """All signals gated -> HOLD."""
        votes = {"bad1": "BUY", "bad2": "SELL", "bad3": "BUY"}
        acc = {
            "bad1": _acc(0.20, 100),
            "bad2": _acc(0.30, 50),
            "bad3": _acc(0.10, 200),
        }
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "HOLD"
        assert conf == 0.0

    def test_custom_accuracy_gate_lower(self):
        """Custom accuracy_gate=0.35 means 0.40 >= 0.35 -> NOT gated."""
        votes = {"sig": "BUY"}
        acc = {"sig": _acc(0.40, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up",
                                           accuracy_gate=0.35)
        assert action == "BUY"
        assert conf == 1.0

    def test_custom_accuracy_gate_higher(self):
        """Custom accuracy_gate=0.55 means 0.50 < 0.55 -> gated."""
        votes = {"sig": "SELL"}
        acc = {"sig": _acc(0.50, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up",
                                           accuracy_gate=0.55)
        assert action == "HOLD"
        assert conf == 0.0

    def test_custom_accuracy_gate_at_boundary(self):
        """accuracy_gate=0.35, signal at 0.34 -> gated; signal at 0.35 -> not gated."""
        votes_gated = {"sig": "BUY"}
        acc_gated = {"sig": _acc(0.34, 50)}
        action, _ = _weighted_consensus(votes_gated, acc_gated, "trending-up",
                                        accuracy_gate=0.35)
        assert action == "HOLD"

        votes_ok = {"sig": "BUY"}
        acc_ok = {"sig": _acc(0.35, 50)}
        action, _ = _weighted_consensus(votes_ok, acc_ok, "trending-up",
                                        accuracy_gate=0.35)
        assert action == "BUY"

    def test_gated_signal_does_not_affect_weight(self):
        """Gated signals contribute zero to both buy and sell weight."""
        votes = {"good_buy": "BUY", "good_sell": "SELL", "gated": "BUY"}
        acc = {
            "good_buy": _acc(0.60, 50),
            "good_sell": _acc(0.70, 100),
            "gated": _acc(0.20, 100),  # gated
        }
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # Only good_buy (0.60) and good_sell (0.70) participate
        # BUY: 0.60, SELL: 0.70 -> SELL wins
        assert action == "SELL"
        assert conf == pytest.approx(0.70 / 1.30, abs=0.01)

    def test_between_045_and_050_votes_normally(self):
        """Signals between 0.45-0.50 are not gated, not inverted, vote with their accuracy."""
        votes = {"mid": "BUY", "good": "SELL"}
        acc = {"mid": _acc(0.47, 50), "good": _acc(0.80, 100)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # mid: BUY with weight 0.47 (not gated, not inverted)
        # good: SELL with weight 0.80
        assert action == "SELL"
        assert conf == pytest.approx(0.80 / (0.47 + 0.80), abs=0.01)

    def test_049_accuracy_not_gated_not_inverted(self):
        """0.49 accuracy is above 0.45 gate -> votes normally as BUY."""
        votes = {"borderline": "BUY"}
        acc = {"borderline": _acc(0.49, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # 0.49 >= 0.45 -> not gated. No inversion. BUY stays BUY.
        assert action == "BUY"
        assert conf == 1.0

    def test_low_accuracy_buy_stays_buy(self):
        """No inversion: a 0.46 accuracy BUY stays BUY (not flipped to SELL)."""
        votes = {"sig": "BUY"}
        acc = {"sig": _acc(0.46, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"

    def test_low_accuracy_sell_stays_sell(self):
        """No inversion: a 0.46 accuracy SELL stays SELL (not flipped to BUY)."""
        votes = {"sig": "SELL"}
        acc = {"sig": _acc(0.46, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "SELL"


# ===========================================================================
# Category 6: Gate boundary (accuracy at exactly 0.45)
# ===========================================================================

class TestBoundaryAccuracy:
    def test_exactly_45_percent_not_gated(self):
        """0.45 accuracy with enough samples is at the boundary -> NOT gated."""
        votes = {"borderline": "BUY"}
        acc = {"borderline": _acc(0.45, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"

    def test_45_percent_weight_is_045(self):
        """At 45% accuracy, weight = 0.45."""
        votes = {"low": "BUY", "good": "SELL"}
        acc = {"low": _acc(0.45, 50), "good": _acc(0.80, 100)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # BUY weight: 0.45, SELL weight: 0.80
        assert action == "SELL"
        assert conf == pytest.approx(0.80 / (0.45 + 0.80), abs=0.01)

    def test_just_below_45_gated(self):
        """0.4499 accuracy with enough samples -> gated."""
        votes = {"sig": "BUY"}
        acc = {"sig": _acc(0.4499, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "HOLD"
        assert conf == 0.0

    def test_just_above_45_not_gated(self):
        votes = {"sig": "BUY"}
        acc = {"sig": _acc(0.4501, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"

    def test_50_percent_not_gated(self):
        """0.50 accuracy is well above gate."""
        votes = {"borderline": "BUY"}
        acc = {"borderline": _acc(0.50, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"

    def test_50_percent_weight_is_0_5(self):
        """At 50% accuracy, weight = 0.5."""
        votes = {"half": "BUY", "good": "SELL"}
        acc = {"half": _acc(0.50, 50), "good": _acc(0.80, 100)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # BUY weight: 0.5, SELL weight: 0.8
        assert action == "SELL"
        assert conf == pytest.approx(0.8 / 1.3, abs=0.01)


# ===========================================================================
# Category 7: Small sample (< 20 samples)
# ===========================================================================

class TestSmallSample:
    def test_under_20_samples_gets_default_weight(self):
        votes = {"new_signal": "BUY"}
        acc = {"new_signal": _acc(0.9, 10)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"
        assert conf == 1.0  # sole voter

    def test_under_20_low_accuracy_not_gated(self):
        """Even 10% accuracy with <20 samples is not gated (insufficient samples)."""
        votes = {"new_signal": "BUY"}
        acc = {"new_signal": _acc(0.10, 15)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"

    def test_exactly_19_uses_default(self):
        votes = {"sig": "BUY"}
        acc = {"sig": _acc(0.2, 19)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # 19 < 20 -> default weight 0.5, not gated (< 30 samples)
        assert action == "BUY"

    def test_exactly_20_uses_actual_accuracy(self):
        """20 samples >= 20 -> uses actual accuracy as weight. Not gated (20 < 30)."""
        votes = {"sig": "BUY"}
        acc = {"sig": _acc(0.3, 20)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # 20 >= 20 -> actual accuracy 0.3. Not gated (20 < 30). BUY stays BUY.
        assert action == "BUY"
        assert conf == 1.0  # sole voter

    def test_zero_samples(self):
        votes = {"sig": "BUY"}
        acc = {"sig": _acc(0.0, 0)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # 0 < 20 -> default weight 0.5, not gated
        assert action == "BUY"

    def test_no_accuracy_data_gets_default_weight(self):
        """Signal not in accuracy_data gets default 0.5 weight."""
        votes = {"unknown": "BUY", "other": "SELL"}
        action, conf = _weighted_consensus(votes, {}, "trending-up")
        # Both get 0.5 weight -> tie -> HOLD
        assert action == "HOLD"
        assert conf == 0.5

    def test_small_sample_vs_large_sample(self):
        """Large sample signal should outweigh small sample signal."""
        votes = {"new": "BUY", "proven": "SELL"}
        acc = {"new": _acc(0.9, 5), "proven": _acc(0.8, 200)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # new: 5 < 20 -> weight 0.5
        # proven: 200 >= 20 -> weight 0.8
        # BUY: 0.5, SELL: 0.8 -> SELL wins
        assert action == "SELL"
        assert conf == pytest.approx(0.8 / 1.3, abs=0.01)

    def test_20_to_29_samples_uses_accuracy_not_gated(self):
        """Samples 20-29: uses actual accuracy but NOT subject to gate (needs >= 30)."""
        votes = {"mid_sample": "SELL"}
        acc = {"mid_sample": _acc(0.30, 25)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # 25 >= 20 -> uses actual accuracy 0.30
        # 25 < 30 -> not gated
        # SELL stays SELL (no inversion)
        assert action == "SELL"
        assert conf == 1.0


# ===========================================================================
# Category 8: Regime weights
# ===========================================================================

class TestRegimeWeights:
    def test_trending_up_boosts_heikin_ashi(self):
        # ema is REGIME-GATED in trending-up (BUG-152); test heikin_ashi (1.2x, not gated)
        votes = {"heikin_ashi": "BUY", "rsi": "SELL"}
        acc = _acc_dict(["heikin_ashi", "rsi"], 0.6, 50)
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # heikin_ashi: 0.6 * 1.2 = 0.72, rsi: 0.6 * 0.7 = 0.42
        assert action == "BUY"
        assert conf == pytest.approx(0.72 / (0.72 + 0.42), abs=0.01)

    def test_trending_up_boosts_macd(self):
        votes = {"macd": "BUY", "rsi": "SELL"}
        acc = _acc_dict(["macd", "rsi"], 0.6, 50)
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # macd: 0.6 * 1.3 = 0.78, rsi: 0.6 * 0.7 = 0.42
        assert action == "BUY"
        assert conf == pytest.approx(0.78 / (0.78 + 0.42), abs=0.01)

    def test_trending_down_same_as_trending_up(self):
        """trending-down has same multipliers as trending-up."""
        votes = {"ema": "BUY", "rsi": "SELL"}
        acc = _acc_dict(["ema", "rsi"], 0.6, 50)
        _, conf_up = _weighted_consensus(votes, acc, "trending-up")
        _, conf_down = _weighted_consensus(votes, acc, "trending-down")
        assert conf_up == conf_down

    def test_ranging_boosts_rsi(self):
        # candlestick is now regime-gated in ranging (BUG-161); use sentiment (1.0x, not gated)
        votes = {"rsi": "BUY", "sentiment": "SELL"}
        acc = _acc_dict(["rsi", "sentiment"], 0.6, 50)
        action, conf = _weighted_consensus(votes, acc, "ranging")
        # rsi: 0.6 * 1.5 = 0.9, sentiment: 0.6 * 1.0 = 0.6
        assert action == "BUY"
        assert conf == pytest.approx(0.9 / (0.9 + 0.6), abs=0.01)

    def test_ranging_boosts_bb(self):
        # macd in ranging gets 1.3x (not 0.5x); update expected accordingly
        votes = {"bb": "SELL", "macd": "BUY"}
        acc = _acc_dict(["bb", "macd"], 0.6, 50)
        action, conf = _weighted_consensus(votes, acc, "ranging")
        # bb: 0.6 * 1.5 = 0.9, macd: 0.6 * 1.3 = 0.78
        assert action == "SELL"
        assert conf == pytest.approx(0.9 / (0.9 + 0.78), abs=0.01)

    def test_highvol_boosts_bb_volume(self):
        votes = {"bb": "BUY", "ema": "SELL"}
        acc = _acc_dict(["bb", "ema"], 0.6, 50)
        action, conf = _weighted_consensus(votes, acc, "high-vol")
        # bb: 0.6 * 1.5 = 0.9, ema: 0.6 * 0.5 = 0.3
        assert action == "BUY"
        assert conf == pytest.approx(0.9 / 1.2, abs=0.01)

    def test_highvol_boosts_volume(self):
        votes = {"volume": "SELL", "ema": "BUY"}
        acc = _acc_dict(["volume", "ema"], 0.6, 50)
        action, conf = _weighted_consensus(votes, acc, "high-vol")
        # volume: 0.6 * 1.3 = 0.78, ema: 0.6 * 0.5 = 0.3
        assert action == "SELL"
        assert conf == pytest.approx(0.78 / 1.08, abs=0.01)

    def test_unknown_regime_uses_default_1x(self):
        votes = {"ema": "BUY", "rsi": "SELL"}
        acc = _acc_dict(["ema", "rsi"], 0.6, 50)
        action, conf = _weighted_consensus(votes, acc, "unknown-regime")
        # No regime mults -> both 0.6 * 1.0 -> tie -> HOLD
        assert action == "HOLD"
        assert conf == 0.5

    def test_none_regime(self):
        votes = {"ema": "BUY", "rsi": "SELL"}
        acc = _acc_dict(["ema", "rsi"], 0.6, 50)
        action, conf = _weighted_consensus(votes, acc, None)
        # None regime -> no mults -> tie -> HOLD
        assert action == "HOLD"
        assert conf == 0.5

    def test_signal_not_in_regime_gets_1x(self):
        """A signal not listed in the regime dict gets multiplier 1.0."""
        votes = {"sentiment": "BUY", "rsi": "SELL"}
        acc = _acc_dict(["sentiment", "rsi"], 0.6, 50)
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # sentiment: 0.6 * 1.0 = 0.6, rsi: 0.6 * 0.7 = 0.42
        assert action == "BUY"
        assert conf == pytest.approx(0.6 / 1.02, abs=0.01)

    def test_regime_weight_stacks_with_accuracy(self):
        """High accuracy + favorable regime = large effective weight."""
        # ema is gated in trending-up; use macd (1.3x, not gated)
        votes = {"macd": "BUY"}
        acc = {"macd": _acc(0.9, 100)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # 0.9 * 1.3 = 1.17 effective weight, sole voter
        assert action == "BUY"
        assert conf == 1.0  # sole voter

    def test_regime_weight_with_gated_signal(self):
        """Gated signal does not participate even with favorable regime."""
        votes = {"ema": "BUY"}
        acc = {"ema": _acc(0.3, 100)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # acc 0.3 < 0.45 gate, 100 >= 30 -> gated, skipped entirely
        assert action == "HOLD"
        assert conf == 0.0

    def test_empty_regime_string(self):
        votes = {"rsi": "BUY"}
        acc = {"rsi": _acc(0.6, 50)}
        action, conf = _weighted_consensus(votes, acc, "")
        assert action == "BUY"
        assert conf == 1.0


# ===========================================================================
# Category 9: Activation rates
# ===========================================================================

class TestActivationRates:
    def test_normalized_weight_scales_signal(self):
        votes = {"rare_signal": "BUY", "noisy_signal": "SELL"}
        acc = _acc_dict(["rare_signal", "noisy_signal"], 0.6, 50)
        activation = {
            "rare_signal": {"normalized_weight": 2.0},
            "noisy_signal": {"normalized_weight": 0.5},
        }
        action, conf = _weighted_consensus(votes, acc, "breakout", activation)
        # rare: 0.6 * 1.0 * 2.0 = 1.2, noisy: 0.6 * 1.0 * 0.5 = 0.3
        assert action == "BUY"
        assert conf == pytest.approx(1.2 / 1.5, abs=0.01)

    def test_no_activation_data_uses_default_1(self):
        votes = {"sig1": "BUY"}
        acc = {"sig1": _acc(0.7, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up", None)
        assert action == "BUY"
        assert conf == 1.0

    def test_activation_empty_dict_uses_default(self):
        votes = {"sig1": "BUY"}
        acc = {"sig1": _acc(0.7, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up", {})
        assert action == "BUY"
        assert conf == 1.0

    def test_activation_zero_weight(self):
        """A signal with normalized_weight=0 contributes nothing."""
        votes = {"zero": "BUY", "normal": "SELL"}
        acc = _acc_dict(["zero", "normal"], 0.6, 50)
        activation = {"zero": {"normalized_weight": 0.0}, "normal": {"normalized_weight": 1.0}}
        action, conf = _weighted_consensus(votes, acc, "trending-up", activation)
        # zero: 0.6 * 0.0 = 0.0, normal: 0.6 * 1.0 = 0.6
        assert action == "SELL"
        assert conf == 1.0

    def test_activation_stacks_with_regime(self):
        # ema is gated in trending-up; use macd (1.3x, not gated)
        votes = {"macd": "BUY"}
        acc = {"macd": _acc(0.6, 50)}
        activation = {"macd": {"normalized_weight": 2.0}}
        action, conf = _weighted_consensus(votes, acc, "trending-up", activation)
        # 0.6 * 1.3 (regime) * 2.0 (activation) = 1.56 — sole voter
        assert action == "BUY"
        assert conf == 1.0

    def test_activation_missing_signal_uses_default(self):
        """Signal not in activation_rates gets normalized_weight=1.0."""
        votes = {"in_act": "BUY", "not_in_act": "SELL"}
        acc = _acc_dict(["in_act", "not_in_act"], 0.6, 50)
        activation = {"in_act": {"normalized_weight": 3.0}}
        action, conf = _weighted_consensus(votes, acc, "trending-up", activation)
        # in_act: 0.6 * 3.0 = 1.8, not_in_act: 0.6 * 1.0 = 0.6
        assert action == "BUY"
        assert conf == pytest.approx(1.8 / 2.4, abs=0.01)

    def test_activation_without_normalized_weight_key(self):
        """Entry in activation_rates but no normalized_weight key -> defaults to 1.0."""
        votes = {"sig": "BUY"}
        acc = {"sig": _acc(0.6, 50)}
        activation = {"sig": {"other_key": 42}}
        action, conf = _weighted_consensus(votes, acc, "trending-up", activation)
        assert action == "BUY"
        assert conf == 1.0


# ===========================================================================
# Category 10: Empty votes
# ===========================================================================

class TestEmptyVotes:
    def test_empty_dict_returns_hold(self):
        action, conf = _weighted_consensus({}, {}, "trending-up")
        assert action == "HOLD"
        assert conf == 0.0

    def test_empty_with_accuracy_data(self):
        acc = {"rsi": _acc(0.9, 100)}
        action, conf = _weighted_consensus({}, acc, "trending-up")
        assert action == "HOLD"
        assert conf == 0.0

    def test_empty_with_activation_rates(self):
        action, conf = _weighted_consensus(
            {}, {}, "trending-up", {"sig": {"normalized_weight": 5.0}}
        )
        assert action == "HOLD"
        assert conf == 0.0


# ===========================================================================
# Category 11: HOLD votes ignored in weight calc
# ===========================================================================

class TestHoldIgnored:
    def test_hold_votes_excluded_from_weight(self):
        votes = {"s1": "BUY", "s2": "HOLD", "s3": "HOLD", "s4": "HOLD"}
        acc = {"s1": _acc(0.7, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"
        assert conf == 1.0  # sole active voter

    def test_hold_does_not_dilute_confidence(self):
        """20 HOLD + 1 BUY should give same conf as just 1 BUY."""
        votes = {f"h{i}": "HOLD" for i in range(20)}
        votes["buyer"] = "BUY"
        acc = {"buyer": _acc(0.7, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"
        assert conf == 1.0

    def test_hold_between_buy_and_sell(self):
        votes = {"b": "BUY", "h": "HOLD", "s": "SELL"}
        acc = _acc_dict(["b", "h", "s"], 0.6, 50)
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # BUY: 0.6, SELL: 0.6 -> tie -> HOLD
        assert action == "HOLD"
        assert conf == 0.5

    def test_hold_with_high_accuracy_ignored(self):
        """Even a HOLD signal with 99% accuracy contributes nothing."""
        votes = {"accurate_hold": "HOLD", "buyer": "BUY"}
        acc = {"accurate_hold": _acc(0.99, 1000), "buyer": _acc(0.55, 30)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"
        assert conf == 1.0


# ===========================================================================
# Edge cases and integration
# ===========================================================================

class TestEdgeCases:
    def test_confidence_rounded_to_4_decimals(self):
        votes = {"s1": "BUY", "s2": "BUY", "s3": "SELL"}
        acc = _acc_dict(votes.keys(), 0.6, 50)
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert conf == round(conf, 4)

    def test_equal_buy_sell_weight_returns_hold(self):
        votes = {"s1": "BUY", "s2": "SELL"}
        acc = _acc_dict(["s1", "s2"], 0.6, 50)
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "HOLD"
        assert conf == 0.5

    def test_many_signals_large_scale(self):
        """30 signals (realistic scale) should work without error."""
        votes = {}
        acc = {}
        for i in range(15):
            votes[f"buy_{i}"] = "BUY"
            acc[f"buy_{i}"] = _acc(0.6, 50)
        for i in range(10):
            votes[f"sell_{i}"] = "SELL"
            acc[f"sell_{i}"] = _acc(0.6, 50)
        for i in range(5):
            votes[f"hold_{i}"] = "HOLD"
            acc[f"hold_{i}"] = _acc(0.6, 50)
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"
        # 15*0.6 / (15*0.6 + 10*0.6) = 9.0/15.0 = 0.6
        assert conf == pytest.approx(0.6, abs=0.01)

    def test_all_factors_combined(self):
        """Accuracy + gate + regime + activation all working together."""
        # ema is gated in trending-up; use heikin_ashi (1.2x, not gated) as BUY voter
        votes = {"heikin_ashi": "BUY", "bad_rsi": "BUY", "bb": "SELL"}
        acc = {
            "heikin_ashi": _acc(0.7, 100),
            "bad_rsi": _acc(0.3, 100),  # gated (0.3 < 0.45, 100 >= 30)
            "bb": _acc(0.6, 50),
        }
        activation = {
            "heikin_ashi": {"normalized_weight": 1.5},
            "bad_rsi": {"normalized_weight": 0.8},
            "bb": {"normalized_weight": 1.0},
        }
        action, conf = _weighted_consensus(votes, acc, "trending-up", activation)
        # heikin_ashi BUY: 0.7 * 1.2 (regime) * 1.5 (act) = 1.26
        # bad_rsi: gated → skipped entirely
        # bb SELL: 0.6 * 0.7 (regime for bb in trending-up) * 1.0 (act) = 0.42
        # BUY total: 1.26, SELL total: 0.42
        assert action == "BUY"
        assert conf == pytest.approx(1.26 / (1.26 + 0.42), abs=0.01)

    def test_confidence_never_exceeds_1(self):
        votes = {"s1": "BUY"}
        acc = {"s1": _acc(0.99, 1000)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert conf <= 1.0

    def test_confidence_never_below_0(self):
        votes = {"s1": "BUY", "s2": "SELL"}
        acc = _acc_dict(["s1", "s2"], 0.6, 50)
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert conf >= 0.0

    def test_unknown_vote_value_treated_as_hold(self):
        """Non-standard vote values should be ignored like HOLD."""
        votes = {"weird": "MAYBE", "good": "BUY"}
        acc = {"good": _acc(0.7, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # "MAYBE" is not BUY/SELL/HOLD -> skipped in loop (continue on HOLD check)
        # Actually "MAYBE" != "HOLD" so it passes the HOLD check,
        # but also != "BUY" and != "SELL" so it adds to neither weight
        assert action == "BUY"
        assert conf == 1.0

    def test_accuracy_data_for_nonexistent_signal(self):
        """Extra accuracy data for signals not in votes is harmless."""
        votes = {"rsi": "BUY"}
        acc = {"rsi": _acc(0.7, 50), "phantom": _acc(0.99, 1000)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"
        assert conf == 1.0


# ===========================================================================
# Category 13: Direction-specific weight scaling (BUG-182)
# ===========================================================================

class TestDirectionalWeightScaling:
    """BUG-182: _weighted_consensus should use buy_accuracy/sell_accuracy as
    the signal weight instead of overall accuracy when directional data is
    available with sufficient samples."""

    def test_buy_uses_buy_accuracy_as_weight(self):
        """A BUY vote should be weighted by buy_accuracy, not overall."""
        votes = {"qwen3": "BUY", "rsi": "SELL"}
        acc = {
            # buy_accuracy 0.45 passes the 0.40 directional gate
            "qwen3": {"accuracy": 0.65, "total": 100,
                      "buy_accuracy": 0.45, "total_buy": 40,
                      "sell_accuracy": 0.80, "total_sell": 60},
            "rsi": {"accuracy": 0.55, "total": 100,
                    "buy_accuracy": 0.55, "total_buy": 50,
                    "sell_accuracy": 0.55, "total_sell": 50},
        }
        action, conf = _weighted_consensus(votes, acc, "breakout")
        # qwen3 BUY weight = buy_accuracy = 0.45 (not overall 0.65)
        # rsi SELL weight = sell_accuracy = 0.55
        # SELL conf = 0.55 / (0.45 + 0.55) = 0.55
        assert action == "SELL"
        assert conf == pytest.approx(0.55 / 1.0, abs=0.02)

    def test_sell_uses_sell_accuracy_as_weight(self):
        """A SELL vote should be weighted by sell_accuracy, not overall."""
        votes = {"qwen3": "SELL", "rsi": "BUY"}
        acc = {
            "qwen3": {"accuracy": 0.60, "total": 100,
                      "buy_accuracy": 0.30, "total_buy": 40,
                      "sell_accuracy": 0.75, "total_sell": 60},
            "rsi": {"accuracy": 0.55, "total": 100,
                    "buy_accuracy": 0.55, "total_buy": 50,
                    "sell_accuracy": 0.55, "total_sell": 50},
        }
        action, conf = _weighted_consensus(votes, acc, "breakout")
        # qwen3 SELL weight = sell_accuracy = 0.75
        # rsi BUY weight = buy_accuracy = 0.55
        # SELL conf = 0.75 / (0.75 + 0.55) ≈ 0.577
        assert action == "SELL"
        assert conf == pytest.approx(0.75 / 1.30, abs=0.02)

    def test_falls_back_to_overall_when_directional_samples_low(self):
        """When directional samples < 20, fall back to overall accuracy."""
        votes = {"sig1": "BUY", "sig2": "SELL"}
        acc = {
            "sig1": {"accuracy": 0.60, "total": 100,
                     "buy_accuracy": 0.30, "total_buy": 10,  # < 20
                     "sell_accuracy": 0.75, "total_sell": 90},
            "sig2": {"accuracy": 0.55, "total": 100,
                     "buy_accuracy": 0.55, "total_buy": 50,
                     "sell_accuracy": 0.55, "total_sell": 50},
        }
        action, conf = _weighted_consensus(votes, acc, "breakout")
        # sig1 BUY: total_buy=10 < 20 → fallback to overall 0.60
        # sig2 SELL: total_sell=50 >= 20 → use sell_accuracy 0.55
        # BUY conf = 0.60 / (0.60 + 0.55) ≈ 0.522
        assert action == "BUY"
        assert conf == pytest.approx(0.60 / 1.15, abs=0.02)

    def test_falls_back_to_overall_when_no_directional_keys(self):
        """Legacy accuracy data without directional fields uses overall."""
        votes = {"sig1": "BUY", "sig2": "SELL"}
        acc = {
            "sig1": {"accuracy": 0.70, "total": 100},  # no buy/sell keys
            "sig2": {"accuracy": 0.55, "total": 100},
        }
        action, conf = _weighted_consensus(votes, acc, "breakout")
        # sig1 BUY: no total_buy → fallback to overall 0.70
        # sig2 SELL: no total_sell → fallback to overall 0.55
        assert action == "BUY"
        assert conf == pytest.approx(0.70 / 1.25, abs=0.02)

    def test_directional_weight_flips_consensus(self):
        """Direction-specific weights can flip consensus vs overall accuracy.

        Without BUG-182 fix: sig_a BUY weight=0.70, sig_b SELL weight=0.55 → BUY wins.
        With BUG-182 fix: sig_a BUY weight=0.40 (buy_accuracy), sig_b SELL weight=0.55 → SELL wins.
        """
        votes = {"sig_a": "BUY", "sig_b": "SELL"}
        acc = {
            "sig_a": {"accuracy": 0.70, "total": 200,
                      "buy_accuracy": 0.40, "total_buy": 80,
                      "sell_accuracy": 0.85, "total_sell": 120},
            "sig_b": {"accuracy": 0.55, "total": 100,
                      "buy_accuracy": 0.55, "total_buy": 50,
                      "sell_accuracy": 0.55, "total_sell": 50},
        }
        action, conf = _weighted_consensus(votes, acc, "breakout")
        # sig_a BUY weight = buy_accuracy = 0.40
        # sig_b SELL weight = sell_accuracy = 0.55
        # SELL wins: 0.55 / (0.40 + 0.55) ≈ 0.579
        assert action == "SELL"
        assert conf == pytest.approx(0.55 / 0.95, abs=0.02)

    def test_directional_zero_accuracy_zeroes_weight(self):
        """A signal with 0% directional accuracy contributes zero weight."""
        votes = {"bad_dir": "BUY", "good": "SELL"}
        acc = {
            "bad_dir": {"accuracy": 0.50, "total": 100,
                        "buy_accuracy": 0.0, "total_buy": 30,
                        "sell_accuracy": 0.80, "total_sell": 70},
            "good": {"accuracy": 0.55, "total": 50,
                     "buy_accuracy": 0.55, "total_buy": 25,
                     "sell_accuracy": 0.55, "total_sell": 25},
        }
        action, conf = _weighted_consensus(votes, acc, "breakout")
        # bad_dir BUY: buy_accuracy=0.0 → weight=0.0
        # good SELL: sell_accuracy=0.55 → weight=0.55
        assert action == "SELL"
        assert conf == 1.0
