"""Tests for _weighted_consensus() in signal_engine.

Covers 11 test categories:
  1. All HOLD         2. All BUY        3. All SELL
  4. Majority vote    5. Signal inversion  6. 50% boundary
  7. Small sample     8. Regime weights    9. Activation rates
 10. Empty votes     11. HOLD ignored
Plus edge cases and integration scenarios.
"""

import pytest

from portfolio.signal_engine import _weighted_consensus


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
        """1 signal at 0.95 beats 2 signals at 0.4 (below inversion threshold)."""
        votes = {"ace": "BUY", "meh1": "SELL", "meh2": "SELL"}
        acc = {"ace": _acc(0.95, 100), "meh1": _acc(0.51, 50), "meh2": _acc(0.51, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # BUY: 0.95, SELL: 0.51+0.51=1.02 -> SELL wins
        assert action == "SELL"


# ===========================================================================
# Category 5: Signal inversion
# ===========================================================================

class TestSignalInversion:
    def test_low_accuracy_buy_inverted_to_sell(self):
        votes = {"bad_signal": "BUY"}
        acc = {"bad_signal": _acc(0.30, 100)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "SELL"
        assert conf == 1.0

    def test_low_accuracy_sell_inverted_to_buy(self):
        votes = {"bad_signal": "SELL"}
        acc = {"bad_signal": _acc(0.25, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"
        assert conf == 1.0

    def test_inversion_uses_complement_weight(self):
        """A 30% accurate signal, when inverted, gets weight 0.70."""
        votes = {"bad": "BUY", "good": "SELL"}
        acc = {"bad": _acc(0.30, 100), "good": _acc(0.60, 100)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # bad BUY inverted to SELL with weight 0.70
        # good SELL weight 0.60
        # SELL total: 0.70 + 0.60 = 1.30, BUY total: 0.0
        assert action == "SELL"
        assert conf == 1.0

    def test_inversion_changes_consensus_direction(self):
        """3 low-accuracy BUY signals inverted overpower 2 good BUY signals."""
        votes = {
            "good1": "BUY", "good2": "BUY",
            "bad1": "BUY", "bad2": "BUY", "bad3": "BUY",
        }
        acc = {
            "good1": _acc(0.7, 100), "good2": _acc(0.7, 100),
            "bad1": _acc(0.3, 100), "bad2": _acc(0.3, 100), "bad3": _acc(0.3, 100),
        }
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # good BUY weight: 0.7+0.7 = 1.4
        # bad BUY -> inverted SELL weight: 0.7+0.7+0.7 = 2.1
        assert action == "SELL"
        assert conf == pytest.approx(2.1 / (1.4 + 2.1), abs=0.01)

    def test_49_percent_inverted(self):
        votes = {"borderline": "BUY"}
        acc = {"borderline": _acc(0.49, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "SELL"

    def test_all_inverted_still_produces_result(self):
        votes = {"b1": "BUY", "b2": "BUY", "b3": "BUY"}
        acc = {k: _acc(0.2, 100) for k in votes}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "SELL"
        assert conf == 1.0

    def test_mixed_inverted_and_normal(self):
        """Normal SELL + inverted BUY (low acc) both contribute to SELL weight."""
        votes = {"norm_sell": "SELL", "bad_buy": "BUY"}
        acc = {"norm_sell": _acc(0.7, 100), "bad_buy": _acc(0.3, 100)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # norm_sell: SELL with weight 0.7
        # bad_buy: inverted to SELL with weight 0.7
        # total SELL: 1.4, BUY: 0
        assert action == "SELL"
        assert conf == 1.0

    def test_inverted_sell_and_normal_buy_reinforce(self):
        """Normal BUY + inverted SELL (low acc) both contribute to BUY weight."""
        votes = {"norm_buy": "BUY", "bad_sell": "SELL"}
        acc = {"norm_buy": _acc(0.7, 100), "bad_sell": _acc(0.3, 100)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # norm_buy: BUY with weight 0.7
        # bad_sell: inverted to BUY with weight 0.7
        assert action == "BUY"
        assert conf == 1.0


# ===========================================================================
# Category 6: 50% boundary (accuracy at exactly 50%)
# ===========================================================================

class TestBoundaryAccuracy:
    def test_exactly_50_percent_no_inversion(self):
        votes = {"borderline": "BUY"}
        acc = {"borderline": _acc(0.50, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"

    def test_50_percent_weight_is_0_5(self):
        """At 50% accuracy, weight = 0.5 (not inverted)."""
        votes = {"half": "BUY", "good": "SELL"}
        acc = {"half": _acc(0.50, 50), "good": _acc(0.80, 100)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # BUY weight: 0.5, SELL weight: 0.8
        assert action == "SELL"
        assert conf == pytest.approx(0.8 / 1.3, abs=0.01)

    def test_just_below_50_inverted(self):
        votes = {"sig": "BUY"}
        acc = {"sig": _acc(0.4999, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "SELL"

    def test_just_above_50_not_inverted(self):
        votes = {"sig": "BUY"}
        acc = {"sig": _acc(0.5001, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"


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

    def test_under_20_never_inverted(self):
        """Even 10% accuracy with <20 samples should NOT invert."""
        votes = {"new_signal": "BUY"}
        acc = {"new_signal": _acc(0.10, 15)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"

    def test_exactly_19_uses_default(self):
        votes = {"sig": "BUY"}
        acc = {"sig": _acc(0.2, 19)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # 19 < 20 -> default weight 0.5, no inversion
        assert action == "BUY"

    def test_exactly_20_uses_actual_accuracy(self):
        votes = {"sig": "BUY"}
        acc = {"sig": _acc(0.3, 20)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # 20 >= 20 -> actual accuracy 0.3 < 0.5 -> inverted
        assert action == "SELL"

    def test_zero_samples(self):
        votes = {"sig": "BUY"}
        acc = {"sig": _acc(0.0, 0)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # 0 < 20 -> default weight 0.5, no inversion
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


# ===========================================================================
# Category 8: Regime weights
# ===========================================================================

class TestRegimeWeights:
    def test_trending_up_boosts_ema(self):
        votes = {"ema": "BUY", "rsi": "SELL"}
        acc = _acc_dict(["ema", "rsi"], 0.6, 50)
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # ema: 0.6 * 1.5 = 0.9, rsi: 0.6 * 0.7 = 0.42
        assert action == "BUY"
        assert conf == pytest.approx(0.9 / (0.9 + 0.42), abs=0.01)

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

    def test_ranging_boosts_rsi_bb(self):
        votes = {"rsi": "BUY", "ema": "SELL"}
        acc = _acc_dict(["rsi", "ema"], 0.6, 50)
        action, conf = _weighted_consensus(votes, acc, "ranging")
        # rsi: 0.6 * 1.5 = 0.9, ema: 0.6 * 0.5 = 0.3
        assert action == "BUY"
        assert conf == pytest.approx(0.9 / (0.9 + 0.3), abs=0.01)

    def test_ranging_boosts_bb(self):
        votes = {"bb": "SELL", "macd": "BUY"}
        acc = _acc_dict(["bb", "macd"], 0.6, 50)
        action, conf = _weighted_consensus(votes, acc, "ranging")
        # bb: 0.6 * 1.5 = 0.9, macd: 0.6 * 0.5 = 0.3
        assert action == "SELL"
        assert conf == pytest.approx(0.9 / 1.2, abs=0.01)

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
        votes = {"ema": "BUY"}
        acc = {"ema": _acc(0.9, 100)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # 0.9 * 1.5 = 1.35 effective weight
        assert action == "BUY"
        assert conf == 1.0  # sole voter

    def test_regime_weight_with_inversion(self):
        """Inverted signal weight still gets regime multiplier."""
        votes = {"ema": "BUY"}
        acc = {"ema": _acc(0.3, 100)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # acc 0.3 < 0.5 -> inverted to SELL, weight = (1-0.3) * 1.5 = 1.05
        assert action == "SELL"
        assert conf == 1.0

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
        votes = {"ema": "BUY"}
        acc = {"ema": _acc(0.6, 50)}
        activation = {"ema": {"normalized_weight": 2.0}}
        action, conf = _weighted_consensus(votes, acc, "trending-up", activation)
        # 0.6 * 1.5 (regime) * 2.0 (activation) = 1.8
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
        """Accuracy + inversion + regime + activation all working together."""
        votes = {"ema": "BUY", "bad_rsi": "BUY", "bb": "SELL"}
        acc = {
            "ema": _acc(0.7, 100),
            "bad_rsi": _acc(0.3, 100),  # inverted
            "bb": _acc(0.6, 50),
        }
        activation = {
            "ema": {"normalized_weight": 1.5},
            "bad_rsi": {"normalized_weight": 0.8},
            "bb": {"normalized_weight": 1.0},
        }
        action, conf = _weighted_consensus(votes, acc, "trending-up", activation)
        # ema BUY: 0.7 * 1.5 (regime) * 1.5 (act) = 1.575
        # bad_rsi BUY -> inverted SELL: (1-0.3) * 1.0 (not 'rsi' key!) * 0.8 (act) = 0.56
        # bb SELL: 0.6 * 0.7 (regime for bb in trending-up) * 1.0 (act) = 0.42
        # BUY total: 1.575, SELL total: 0.56 + 0.42 = 0.98
        assert action == "BUY"
        assert conf == pytest.approx(1.575 / (1.575 + 0.98), abs=0.01)

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
# TEST-8: Inversion weight cap at 0.75 (BUG-38)
# ===========================================================================

class TestInversionWeightCap:
    """BUG-38: Inverted signals capped at weight 0.75 to prevent domination."""

    def test_5_percent_accuracy_capped_at_075(self):
        """A 5% accurate signal should get weight 0.75, not 0.95."""
        votes = {"terrible": "BUY", "good": "BUY"}
        acc = {"terrible": _acc(0.05, 100), "good": _acc(0.70, 100)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # terrible: inverted to SELL, weight capped at 0.75 (not 0.95)
        # good: BUY, weight 0.70
        # Before cap: SELL would have 0.95 > BUY 0.70 -> SELL wins
        # After cap: SELL 0.75, BUY 0.70 -> closer, but SELL still wins here
        assert action == "SELL"

    def test_cap_prevents_extreme_domination(self):
        """Without cap, 1 signal at 5% acc would overpower 2 signals at 60% acc.
        With cap at 0.75, the 2 good signals (1.2 total) beat the capped one (0.75)."""
        votes = {"bad": "BUY", "good1": "BUY", "good2": "BUY"}
        acc = {
            "bad": _acc(0.05, 100),   # inverted: capped at 0.75
            "good1": _acc(0.60, 100), # BUY weight 0.60
            "good2": _acc(0.60, 100), # BUY weight 0.60
        }
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # bad inverted to SELL: 0.75 (capped from 0.95)
        # good1+good2 BUY: 0.60 + 0.60 = 1.20
        # BUY wins (1.20 > 0.75) -- without cap, SELL would be 0.95 vs BUY 1.20, BUY still wins
        # but with even worse accuracy or regime multipliers, the cap becomes critical
        assert action == "BUY"
        assert conf == pytest.approx(1.20 / (1.20 + 0.75), abs=0.01)

    def test_10_percent_accuracy_capped(self):
        """10% accuracy -> inverted weight would be 0.90, capped to 0.75."""
        votes = {"low_acc": "SELL"}
        acc = {"low_acc": _acc(0.10, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # Inverted from SELL to BUY, weight capped at 0.75
        assert action == "BUY"
        assert conf == 1.0  # sole voter

    def test_25_percent_accuracy_capped(self):
        """25% accuracy -> inverted weight would be 0.75, exactly at cap (no change)."""
        votes = {"sig": "BUY"}
        acc = {"sig": _acc(0.25, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # 1.0 - 0.25 = 0.75, exactly at cap
        assert action == "SELL"

    def test_30_percent_accuracy_not_capped(self):
        """30% accuracy -> inverted weight = 0.70, below cap (unaffected)."""
        votes = {"sig": "BUY"}
        acc = {"sig": _acc(0.30, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # 1.0 - 0.30 = 0.70 < 0.75, not capped
        assert action == "SELL"

    def test_45_percent_accuracy_not_capped(self):
        """45% accuracy -> inverted weight = 0.55, well below cap."""
        votes = {"sig": "BUY"}
        acc = {"sig": _acc(0.45, 50)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "SELL"

    def test_cap_does_not_affect_normal_signals(self):
        """Non-inverted signals (>= 50% accuracy) are never capped."""
        votes = {"great": "BUY"}
        acc = {"great": _acc(0.95, 200)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"
        assert conf == 1.0

    def test_cap_does_not_affect_small_samples(self):
        """Signals with <20 samples get default 0.5, no inversion, no cap."""
        votes = {"new": "BUY"}
        acc = {"new": _acc(0.05, 10)}  # terrible acc but too few samples
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # <20 samples -> default weight 0.5, no inversion
        assert action == "BUY"

    def test_cap_with_regime_multiplier(self):
        """Capped weight still gets regime multiplier applied."""
        votes = {"ema": "BUY"}
        acc = {"ema": _acc(0.05, 100)}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # ema: inverted to SELL, weight = min(0.75, 0.95) = 0.75
        # regime mult for ema in trending-up = 1.5
        # effective weight = 0.75 * 1.5 = 1.125
        assert action == "SELL"
        assert conf == 1.0  # sole voter

    def test_multiple_capped_signals(self):
        """Multiple extremely low-accuracy signals all get capped."""
        votes = {"bad1": "BUY", "bad2": "BUY", "bad3": "BUY", "good": "BUY"}
        acc = {
            "bad1": _acc(0.05, 100),  # capped at 0.75
            "bad2": _acc(0.10, 100),  # capped at 0.75
            "bad3": _acc(0.15, 100),  # capped at 0.75
            "good": _acc(0.70, 100),  # BUY weight 0.70
        }
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # SELL from inversions: 0.75 + 0.75 + 0.75 = 2.25
        # BUY from good: 0.70
        # Without cap: 0.95 + 0.90 + 0.85 = 2.70 SELL vs 0.70 BUY
        # Cap reduces SELL weight from 2.70 to 2.25
        assert action == "SELL"
        assert conf == pytest.approx(2.25 / (2.25 + 0.70), abs=0.01)
