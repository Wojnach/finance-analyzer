"""Tests for _weighted_consensus() in signal_engine."""

import pytest

from portfolio.signal_engine import _weighted_consensus


class TestAllHold:
    def test_all_hold_returns_hold_zero(self):
        votes = {"rsi": "HOLD", "macd": "HOLD", "ema": "HOLD"}
        action, conf = _weighted_consensus(votes, {}, "trending-up")
        assert action == "HOLD"
        assert conf == 0.0

    def test_empty_votes_returns_hold(self):
        action, conf = _weighted_consensus({}, {}, "trending-up")
        assert action == "HOLD"
        assert conf == 0.0


class TestUnanimous:
    def test_unanimous_buy_returns_buy_1(self):
        votes = {"rsi": "BUY", "macd": "BUY", "ema": "BUY"}
        # All have enough samples and 60% accuracy
        acc = {k: {"accuracy": 0.6, "total": 50} for k in votes}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"
        assert conf == 1.0

    def test_unanimous_sell_returns_sell_1(self):
        votes = {"rsi": "SELL", "macd": "SELL", "ema": "SELL"}
        acc = {k: {"accuracy": 0.6, "total": 50} for k in votes}
        action, conf = _weighted_consensus(votes, acc, "trending-down")
        assert action == "SELL"
        assert conf == 1.0


class TestSignalInversion:
    def test_low_accuracy_buy_inverted_to_sell(self):
        votes = {"bad_signal": "BUY"}
        acc = {"bad_signal": {"accuracy": 0.30, "total": 100}}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # 30% BUY → inverted to SELL with weight 0.70
        assert action == "SELL"
        assert conf == 1.0  # sole voter

    def test_low_accuracy_sell_inverted_to_buy(self):
        votes = {"bad_signal": "SELL"}
        acc = {"bad_signal": {"accuracy": 0.25, "total": 50}}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"
        assert conf == 1.0

    def test_exactly_50_percent_no_inversion(self):
        votes = {"borderline": "BUY"}
        acc = {"borderline": {"accuracy": 0.50, "total": 50}}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"  # 50% is NOT below 50%, so no inversion

    def test_49_percent_inverted(self):
        votes = {"borderline": "BUY"}
        acc = {"borderline": {"accuracy": 0.49, "total": 50}}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "SELL"  # 49% < 50% → inverted

    def test_inversion_changes_consensus_direction(self):
        # 2 BUY (good accuracy) + 3 BUY (bad accuracy → inverted to SELL)
        votes = {
            "good1": "BUY", "good2": "BUY",
            "bad1": "BUY", "bad2": "BUY", "bad3": "BUY",
        }
        acc = {
            "good1": {"accuracy": 0.7, "total": 100},
            "good2": {"accuracy": 0.7, "total": 100},
            "bad1": {"accuracy": 0.3, "total": 100},
            "bad2": {"accuracy": 0.3, "total": 100},
            "bad3": {"accuracy": 0.3, "total": 100},
        }
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # good BUY weight = 0.7+0.7 = 1.4
        # bad BUY → inverted SELL weight = 0.7+0.7+0.7 = 2.1
        # SELL should win
        assert action == "SELL"
        assert conf == pytest.approx(2.1 / (1.4 + 2.1), abs=0.01)


class TestSmallSampleDefault:
    def test_under_20_samples_gets_default_weight(self):
        votes = {"new_signal": "BUY"}
        acc = {"new_signal": {"accuracy": 0.9, "total": 10}}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # Weight should be 0.5 (default), not 0.9
        assert action == "BUY"
        assert conf == 1.0  # sole voter still gets 1.0

    def test_under_20_never_inverted(self):
        votes = {"new_signal": "BUY"}
        acc = {"new_signal": {"accuracy": 0.10, "total": 15}}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # Even 10% accuracy with <20 samples should NOT invert
        assert action == "BUY"

    def test_no_accuracy_data_gets_default_weight(self):
        votes = {"unknown": "BUY", "other": "SELL"}
        action, conf = _weighted_consensus(votes, {}, "trending-up")
        # Both get 0.5 weight, equal → tie at 0.5 each
        # buy_conf == sell_conf == 0.5 → falls to max() → HOLD
        assert action == "HOLD"
        assert conf == 0.5


class TestRegimeWeights:
    def test_trending_up_boosts_ema_macd(self):
        votes = {"ema": "BUY", "rsi": "SELL"}
        acc = {
            "ema": {"accuracy": 0.6, "total": 50},
            "rsi": {"accuracy": 0.6, "total": 50},
        }
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # ema weight = 0.6 * 1.5 = 0.9
        # rsi weight = 0.6 * 0.7 = 0.42
        # BUY should win
        assert action == "BUY"
        assert conf == pytest.approx(0.9 / (0.9 + 0.42), abs=0.01)

    def test_ranging_boosts_rsi_bb(self):
        votes = {"rsi": "BUY", "ema": "SELL"}
        acc = {
            "rsi": {"accuracy": 0.6, "total": 50},
            "ema": {"accuracy": 0.6, "total": 50},
        }
        action, conf = _weighted_consensus(votes, acc, "ranging")
        # rsi weight = 0.6 * 1.5 = 0.9
        # ema weight = 0.6 * 0.5 = 0.3
        assert action == "BUY"
        assert conf == pytest.approx(0.9 / (0.9 + 0.3), abs=0.01)

    def test_unknown_regime_uses_default_1x(self):
        votes = {"ema": "BUY", "rsi": "SELL"}
        acc = {
            "ema": {"accuracy": 0.6, "total": 50},
            "rsi": {"accuracy": 0.6, "total": 50},
        }
        action, conf = _weighted_consensus(votes, acc, "unknown-regime")
        # No regime mults → both 0.6 * 1.0 → tie → HOLD
        assert action == "HOLD"
        assert conf == 0.5


class TestActivationRates:
    def test_normalized_weight_scales_signal(self):
        votes = {"rare_signal": "BUY", "noisy_signal": "SELL"}
        acc = {
            "rare_signal": {"accuracy": 0.6, "total": 50},
            "noisy_signal": {"accuracy": 0.6, "total": 50},
        }
        activation = {
            "rare_signal": {"normalized_weight": 2.0},
            "noisy_signal": {"normalized_weight": 0.5},
        }
        action, conf = _weighted_consensus(votes, acc, "trending-up", activation)
        # rare: 0.6 * 1.0 (no regime) * 2.0 = 1.2
        # noisy: 0.6 * 1.0 * 0.5 = 0.3
        # But ema/rsi regime mults don't apply to these names → 1.0
        assert action == "BUY"
        assert conf == pytest.approx(1.2 / (1.2 + 0.3), abs=0.01)

    def test_no_activation_data_uses_default_1(self):
        votes = {"sig1": "BUY"}
        acc = {"sig1": {"accuracy": 0.7, "total": 50}}
        action, conf = _weighted_consensus(votes, acc, "trending-up", None)
        assert action == "BUY"
        assert conf == 1.0


class TestMixedVotes:
    def test_majority_buy_wins(self):
        votes = {"s1": "BUY", "s2": "BUY", "s3": "SELL", "s4": "HOLD"}
        acc = {k: {"accuracy": 0.6, "total": 50} for k in votes}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"
        # 2 BUY * 0.6 = 1.2 vs 1 SELL * 0.6 = 0.6
        assert conf == pytest.approx(1.2 / (1.2 + 0.6), abs=0.01)

    def test_hold_votes_excluded_from_weight(self):
        votes = {"s1": "BUY", "s2": "HOLD", "s3": "HOLD", "s4": "HOLD"}
        acc = {"s1": {"accuracy": 0.7, "total": 50}}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"
        assert conf == 1.0  # sole active voter

    def test_confidence_reflects_weight_ratio(self):
        votes = {"strong": "BUY", "weak": "SELL"}
        acc = {
            "strong": {"accuracy": 0.8, "total": 100},
            "weak": {"accuracy": 0.55, "total": 100},
        }
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        assert action == "BUY"
        # strong: 0.8, weak: 0.55
        assert conf == pytest.approx(0.8 / (0.8 + 0.55), abs=0.01)


class TestEdgeCases:
    def test_single_buy_voter(self):
        action, conf = _weighted_consensus(
            {"only": "BUY"},
            {"only": {"accuracy": 0.65, "total": 30}},
            "trending-up",
        )
        assert action == "BUY"
        assert conf == 1.0

    def test_single_sell_voter(self):
        action, conf = _weighted_consensus(
            {"only": "SELL"},
            {"only": {"accuracy": 0.65, "total": 30}},
            "trending-down",
        )
        assert action == "SELL"
        assert conf == 1.0

    def test_equal_buy_sell_weight_returns_hold(self):
        votes = {"s1": "BUY", "s2": "SELL"}
        acc = {
            "s1": {"accuracy": 0.6, "total": 50},
            "s2": {"accuracy": 0.6, "total": 50},
        }
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # Equal weights → tie → HOLD
        assert action == "HOLD"
        assert conf == 0.5

    def test_confidence_rounded_to_4_decimal(self):
        votes = {"s1": "BUY", "s2": "BUY", "s3": "SELL"}
        acc = {k: {"accuracy": 0.6, "total": 50} for k in votes}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # Confidence should be a float rounded to 4 decimals
        assert conf == round(conf, 4)

    def test_all_inverted_still_produces_result(self):
        votes = {"b1": "BUY", "b2": "BUY", "b3": "BUY"}
        acc = {k: {"accuracy": 0.2, "total": 100} for k in votes}
        action, conf = _weighted_consensus(votes, acc, "trending-up")
        # All inverted: BUY→SELL
        assert action == "SELL"
        assert conf == 1.0
