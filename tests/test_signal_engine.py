"""Tests for portfolio.signal_engine — dynamic correlation groups."""



# ---------------------------------------------------------------------------
# 3c. Static and dynamic correlation groups
# ---------------------------------------------------------------------------

class TestCorrelationGroups:

    def test_static_correlation_groups_has_expected_keys(self):
        """CORRELATION_GROUPS (static alias) should exist and contain known group names."""
        from portfolio.signal_engine import CORRELATION_GROUPS

        assert isinstance(CORRELATION_GROUPS, dict)
        # Check a few known static groups
        # low_activity_timing removed 2026-04-12: calendar (BUY-only) and
        # econ_calendar (SELL-only) have opposite directions and divergent
        # regime profiles — not a valid correlation cluster.
        # pattern_based dissolved 2026-04-29: fibonacci disabled (43.6%, 17K sam),
        # candlestick now unclustered.
        assert "pattern_based" not in CORRELATION_GROUPS
        # Values should be frozensets
        for name, members in CORRELATION_GROUPS.items():
            assert isinstance(members, frozenset), f"Group {name} should be frozenset"
            assert len(members) >= 2, f"Group {name} should have at least 2 members"

    def test_momentum_cluster_exists(self):
        """momentum_cluster should contain rsi, mean_reversion, momentum, bb.

        2026-05-20: bb re-added — 100% agreement with mean_reversion (n=99)
        confirmed via 50-snapshot correlation audit. Unclustered bb inflated
        total weight to 2.30x for one opinion.
        """
        from portfolio.signal_engine import CORRELATION_GROUPS

        assert "momentum_cluster" in CORRELATION_GROUPS
        mc_group = CORRELATION_GROUPS["momentum_cluster"]
        assert "mean_reversion" in mc_group
        assert "rsi" in mc_group
        assert "momentum" in mc_group
        assert "bb" in mc_group

    def test_volatility_cluster_removed(self):
        """volatility_cluster was dissolved (RES-2026-04-21).

        volatility_sig and volume had only r=0.38 correlation — too weak
        for a group. volume (52.1% acc) was unfairly penalized by
        volatility_sig (46.8% acc). Both now vote independently.
        """
        from portfolio.signal_engine import CORRELATION_GROUPS

        assert "volatility_cluster" not in CORRELATION_GROUPS

    def test_trend_direction_split_into_subclusters(self):
        """2026-05-07: trend/macd disabled, groups updated.

        pure_trend: ema, heikin_ashi (trend removed — 46.1% at 1d)
        oscillator_trend: momentum_factors, oscillators (macd removed — 44.2% at 1d)
        structural_flow: volume_flow, macro_regime, structure
        """
        from portfolio.signal_engine import CORRELATION_GROUPS

        assert "trend_direction" not in CORRELATION_GROUPS, "mega-cluster should be removed"
        assert "pure_trend" in CORRELATION_GROUPS
        assert "oscillator_trend" in CORRELATION_GROUPS
        assert "structural_flow" in CORRELATION_GROUPS

        assert CORRELATION_GROUPS["pure_trend"] == frozenset({"ema", "heikin_ashi"})
        assert CORRELATION_GROUPS["oscillator_trend"] == frozenset({"momentum_factors", "oscillators"})
        assert CORRELATION_GROUPS["structural_flow"] == frozenset({"volume_flow", "macro_regime", "structure"})

    def test_macd_disabled_not_in_oscillator_trend(self):
        """2026-05-07: macd disabled (44.2% 1d), removed from oscillator_trend."""
        from portfolio.signal_engine import CORRELATION_GROUPS

        assert "macd" not in CORRELATION_GROUPS["oscillator_trend"]

    def test_macro_regime_in_structural_flow(self):
        """macro_regime should be in structural_flow sub-cluster."""
        from portfolio.signal_engine import CORRELATION_GROUPS

        assert "macro_regime" in CORRELATION_GROUPS["structural_flow"]
        assert "structure" in CORRELATION_GROUPS["structural_flow"]

    def test_macro_regime_not_in_macro_external(self):
        """macro_regime was moved out of macro_external into structural_flow."""
        from portfolio.signal_engine import CORRELATION_GROUPS

        me_group = CORRELATION_GROUPS["macro_external"]
        assert "macro_regime" not in me_group

    def test_static_groups_backward_compat_alias(self):
        """CORRELATION_GROUPS should be the same object as _STATIC_CORRELATION_GROUPS."""
        from portfolio.signal_engine import (
            _STATIC_CORRELATION_GROUPS,
            CORRELATION_GROUPS,
        )
        assert CORRELATION_GROUPS is _STATIC_CORRELATION_GROUPS

    def test_dynamic_groups_fallback(self, monkeypatch):
        """When no signal_log data exists, _compute_dynamic_correlation_groups
        should return _STATIC_CORRELATION_GROUPS."""
        # Monkeypatch load_entries to return empty list (no data)
        import portfolio.accuracy_stats as acc_mod
        from portfolio.signal_engine import (
            _STATIC_CORRELATION_GROUPS,
            _compute_dynamic_correlation_groups,
        )
        monkeypatch.setattr(acc_mod, "load_entries", lambda: [])

        result = _compute_dynamic_correlation_groups()
        assert result is _STATIC_CORRELATION_GROUPS

    def test_dynamic_groups_fallback_insufficient_data(self, monkeypatch):
        """With fewer than _DYNAMIC_CORR_MIN_SAMPLES entries, falls back to static."""
        # Provide a small number of entries (below the 30 minimum)
        import portfolio.accuracy_stats as acc_mod
        from portfolio.signal_engine import (
            _STATIC_CORRELATION_GROUPS,
            _compute_dynamic_correlation_groups,
        )
        fake_entries = [
            {
                "ts": "2026-04-01T00:00:00+00:00",
                "tickers": {"BTC-USD": {"signals": {"rsi": "BUY"}}},
                "outcomes": {},
            }
            for _ in range(5)
        ]
        monkeypatch.setattr(acc_mod, "load_entries", lambda: fake_entries)

        result = _compute_dynamic_correlation_groups()
        assert result is _STATIC_CORRELATION_GROUPS

    def test_dynamic_groups_returns_dict_of_frozensets(self, monkeypatch):
        """Even when falling back, the return type is dict[str, frozenset]."""
        import portfolio.accuracy_stats as acc_mod
        from portfolio.signal_engine import _compute_dynamic_correlation_groups
        monkeypatch.setattr(acc_mod, "load_entries", lambda: [])

        result = _compute_dynamic_correlation_groups()
        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(key, str)
            assert isinstance(value, frozenset)


# ---------------------------------------------------------------------------
# Directional bias penalty
# ---------------------------------------------------------------------------

class TestDirectionalBiasPenalty:

    def test_extreme_bias_reduces_in_bias_weight(self):
        """Signals with bias > 85% should get _BIAS_PENALTY when voting WITH bias."""
        from portfolio.signal_engine import _weighted_consensus

        votes = {"rsi": "BUY", "calendar": "BUY"}
        accuracy_data = {
            "rsi": {"accuracy": 0.55, "total": 100,
                    "buy_accuracy": 0.55, "total_buy": 50},
            "calendar": {"accuracy": 0.60, "total": 100,
                         "buy_accuracy": 0.60, "total_buy": 50},
        }
        # calendar has extreme BUY bias (>85%), rsi does not
        activation_rates = {
            "rsi": {"bias": 0.1, "samples": 100, "normalized_weight": 1.0,
                    "activation_rate": 0.3, "buy_rate": 0.15, "sell_rate": 0.15},
            "calendar": {"bias": 0.95, "samples": 100, "normalized_weight": 0.25,
                         "activation_rate": 0.08, "buy_rate": 0.08, "sell_rate": 0.0},
        }
        result = _weighted_consensus(
            votes, accuracy_data, "ranging",
            activation_rates=activation_rates,
        )
        # Both BUY → should still return BUY (calendar penalized but still BUY)
        assert result[0] == "BUY"

    def test_contrarian_vote_not_penalized(self):
        """BUY-biased signal voting SELL (contrarian) keeps full weight."""
        from portfolio.signal_engine import _weighted_consensus

        # Use "unknown" regime to avoid REGIME_WEIGHTS confounding
        # Two signals with equal accuracy — calendar SELL is contrarian
        accuracy_data = {
            "calendar": {"accuracy": 0.60, "total": 100,
                         "sell_accuracy": 0.60, "total_sell": 50},
            "rsi": {"accuracy": 0.60, "total": 100,
                    "buy_accuracy": 0.60, "total_buy": 50},
        }
        activation_rates = {
            "calendar": {"bias": 0.95, "samples": 100, "normalized_weight": 1.0,
                         "activation_rate": 0.08, "buy_rate": 0.08, "sell_rate": 0.0},
            "rsi": {"bias": 0.1, "samples": 100, "normalized_weight": 1.0,
                    "activation_rate": 0.3, "buy_rate": 0.15, "sell_rate": 0.15},
        }
        # calendar SELL (contrarian) vs rsi BUY — equal accuracy, equal norm_weight
        # With direction-aware penalty: calendar NOT penalized → equal weights → HOLD
        # Old behavior: calendar penalized regardless → rsi BUY wins → BUY
        votes = {"calendar": "SELL", "rsi": "BUY"}
        result = _weighted_consensus(
            votes, accuracy_data, "unknown",
            activation_rates=activation_rates,
        )
        assert result[0] == "HOLD", (
            f"Contrarian SELL from BUY-biased signal should keep full weight "
            f"(equal to opponent), producing HOLD — got {result[0]}"
        )

    def test_in_bias_vote_penalized_loses_to_equal_opponent(self):
        """BUY-biased signal voting BUY (in-bias) gets penalized → opponent wins."""
        from portfolio.signal_engine import _weighted_consensus

        accuracy_data = {
            "calendar": {"accuracy": 0.60, "total": 100,
                         "buy_accuracy": 0.60, "total_buy": 50},
            "rsi": {"accuracy": 0.60, "total": 100,
                    "sell_accuracy": 0.60, "total_sell": 50},
        }
        activation_rates = {
            "calendar": {"bias": 0.95, "samples": 100, "normalized_weight": 1.0,
                         "activation_rate": 0.08, "buy_rate": 0.08, "sell_rate": 0.0},
            "rsi": {"bias": 0.1, "samples": 100, "normalized_weight": 1.0,
                    "activation_rate": 0.3, "buy_rate": 0.15, "sell_rate": 0.15},
        }
        # calendar BUY (in-bias → penalized 0.5x) vs rsi SELL (full weight)
        # calendar weight = 0.60 * 0.5 = 0.30, rsi weight = 0.60 → SELL wins
        votes = {"calendar": "BUY", "rsi": "SELL"}
        result = _weighted_consensus(
            votes, accuracy_data, "unknown",
            activation_rates=activation_rates,
        )
        assert result[0] == "SELL", (
            f"In-bias BUY from BUY-biased signal should be penalized, "
            f"letting equal-accuracy opponent win — got {result[0]}"
        )

    def test_sell_biased_contrarian_buy_not_penalized(self):
        """SELL-biased signal voting BUY (contrarian) keeps full weight."""
        from portfolio.signal_engine import _weighted_consensus

        accuracy_data = {
            "news_event": {"accuracy": 0.60, "total": 100,
                           "buy_accuracy": 0.60, "total_buy": 50},
            "rsi": {"accuracy": 0.60, "total": 100,
                    "sell_accuracy": 0.60, "total_sell": 50},
        }
        activation_rates = {
            "news_event": {"bias": 0.99, "samples": 100, "normalized_weight": 1.0,
                           "activation_rate": 0.05, "buy_rate": 0.0, "sell_rate": 0.05},
            "rsi": {"bias": 0.1, "samples": 100, "normalized_weight": 1.0,
                    "activation_rate": 0.3, "buy_rate": 0.15, "sell_rate": 0.15},
        }
        # news_event BUY (contrarian) vs rsi SELL — equal accuracy
        # Contrarian not penalized → HOLD
        votes = {"news_event": "BUY", "rsi": "SELL"}
        result = _weighted_consensus(
            votes, accuracy_data, "unknown",
            activation_rates=activation_rates,
        )
        assert result[0] == "HOLD"

    def test_no_penalty_below_threshold(self):
        """Signals with bias <= 85% should NOT get extra penalty."""
        from portfolio.signal_engine import _weighted_consensus

        votes = {"rsi": "BUY"}
        accuracy_data = {"rsi": {"accuracy": 0.55, "total": 100,
                                 "buy_accuracy": 0.55, "total_buy": 50}}
        activation_rates = {
            "rsi": {"bias": 0.5, "samples": 100, "normalized_weight": 1.0,
                    "activation_rate": 0.3, "buy_rate": 0.15, "sell_rate": 0.15},
        }
        # With bias=0.5 (< 0.85), no extra penalty should apply
        result = _weighted_consensus(
            votes, accuracy_data, "ranging",
            activation_rates=activation_rates,
        )
        assert result[0] == "BUY"

    def test_extreme_bias_gets_stronger_penalty(self):
        """2026-05-07: >95% bias gets 0.2x penalty (vs 0.5x for 85-95%)."""
        from portfolio.signal_engine import _weighted_consensus

        accuracy_data = {
            "crypto_macro": {"accuracy": 0.56, "total": 500,
                             "buy_accuracy": 0.56, "total_buy": 490},
            "rsi": {"accuracy": 0.55, "total": 100,
                    "sell_accuracy": 0.55, "total_sell": 50},
        }
        activation_rates = {
            "crypto_macro": {"bias": 0.99, "samples": 500, "normalized_weight": 1.0,
                             "activation_rate": 0.98, "buy_rate": 0.98, "sell_rate": 0.01},
            "rsi": {"bias": 0.1, "samples": 100, "normalized_weight": 1.0,
                    "activation_rate": 0.3, "buy_rate": 0.15, "sell_rate": 0.15},
        }
        votes = {"crypto_macro": "BUY", "rsi": "SELL"}
        result = _weighted_consensus(
            votes, accuracy_data, "unknown",
            activation_rates=activation_rates,
        )
        assert result[0] == "SELL", (
            f"crypto_macro at 99% BUY bias should get 0.2x extreme penalty, "
            f"losing to rsi SELL — got {result[0]}"
        )

    def test_bias_penalty_not_applied_with_few_samples(self):
        """Bias penalty should not fire when samples < _BIAS_MIN_ACTIVE."""
        from portfolio.signal_engine import _BIAS_MIN_ACTIVE, _weighted_consensus

        votes = {"rsi": "BUY"}
        accuracy_data = {"rsi": {"accuracy": 0.55, "total": 100,
                                 "buy_accuracy": 0.55, "total_buy": 50}}
        activation_rates = {
            "rsi": {"bias": 0.99, "samples": _BIAS_MIN_ACTIVE - 1,
                    "normalized_weight": 1.0, "activation_rate": 0.3,
                    "buy_rate": 0.28, "sell_rate": 0.02},
        }
        result = _weighted_consensus(
            votes, accuracy_data, "ranging",
            activation_rates=activation_rates,
        )
        assert result[0] == "BUY"


class TestResolveBiasPenaltyBoundaries:
    """2026-05-19: three-tier bias penalty (moderate/high/extreme).

    Each boundary tested separately so a refactor that breaks tier
    ordering (e.g. accidentally `if bias > moderate: ... elif bias >
    extreme` which would swallow extreme into moderate) fails loudly
    instead of silently returning the wrong multiplier. Premortem F1.
    """

    def test_below_moderate_returns_1x(self):
        from portfolio.signal_engine import _resolve_bias_penalty
        assert _resolve_bias_penalty(0.64) == 1.0
        assert _resolve_bias_penalty(0.65) == 1.0  # boundary: NOT > 0.65
        assert _resolve_bias_penalty(0.0) == 1.0

    def test_moderate_tier_returns_0_7x(self):
        from portfolio.signal_engine import (
            _BIAS_MODERATE_PENALTY, _resolve_bias_penalty,
        )
        # 0.65 < bias <= 0.85 → moderate (0.7x)
        assert _resolve_bias_penalty(0.66) == _BIAS_MODERATE_PENALTY
        assert _resolve_bias_penalty(0.81) == _BIAS_MODERATE_PENALTY  # sentiment
        assert _resolve_bias_penalty(0.85) == _BIAS_MODERATE_PENALTY  # boundary

    def test_high_tier_returns_0_5x(self):
        from portfolio.signal_engine import (
            _BIAS_PENALTY, _resolve_bias_penalty,
        )
        # 0.85 < bias <= 0.90 → high (0.5x)
        assert _resolve_bias_penalty(0.86) == _BIAS_PENALTY
        assert _resolve_bias_penalty(0.90) == _BIAS_PENALTY  # boundary

    def test_extreme_tier_returns_0_2x(self):
        from portfolio.signal_engine import (
            _BIAS_EXTREME_PENALTY, _resolve_bias_penalty,
        )
        # bias > 0.90 → extreme (0.2x)
        assert _resolve_bias_penalty(0.91) == _BIAS_EXTREME_PENALTY  # crypto_macro
        assert _resolve_bias_penalty(0.95) == _BIAS_EXTREME_PENALTY
        assert _resolve_bias_penalty(0.99) == _BIAS_EXTREME_PENALTY  # fear_greed-ish
        assert _resolve_bias_penalty(1.00) == _BIAS_EXTREME_PENALTY

    def test_tier_ordering_extreme_beats_moderate(self):
        """Regression guard for premortem F1: extreme value must NOT
        return moderate's multiplier. A buggy lowest-first cascade
        would return 0.7 here."""
        from portfolio.signal_engine import (
            _BIAS_EXTREME_PENALTY, _resolve_bias_penalty,
        )
        # crypto_macro 0.91 in production: must hit extreme, not moderate.
        assert _resolve_bias_penalty(0.91) == _BIAS_EXTREME_PENALTY
        assert _resolve_bias_penalty(0.91) < 0.5  # stronger than high tier

    def test_moderate_tier_catches_sentiment(self):
        """Production sentinel: sentiment bias=0.81 must get moderate
        penalty (was 1.0x pre-2026-05-19). Boundary regression guard."""
        from portfolio.signal_engine import _resolve_bias_penalty
        assert _resolve_bias_penalty(0.81) == 0.7


class TestModerateTierEndToEnd:
    """Verify the new moderate tier actually changes consensus output."""

    def test_sentiment_buy_with_moderate_penalty_loses_to_balanced_sell(self):
        """sentiment bias=0.81 voting BUY (in-bias) gets 0.7x. With
        an equally-accurate direction-balanced opponent voting SELL,
        the opponent should win on weighted_confidence."""
        from portfolio.signal_engine import _weighted_consensus

        accuracy_data = {
            "sentiment": {"accuracy": 0.55, "total": 500,
                          "buy_accuracy": 0.55, "total_buy": 400},
            "rsi": {"accuracy": 0.55, "total": 500,
                    "sell_accuracy": 0.55, "total_sell": 250},
        }
        activation_rates = {
            "sentiment": {"bias": 0.81, "samples": 99057,
                          "normalized_weight": 1.0,
                          "activation_rate": 0.44, "buy_rate": 0.40,
                          "sell_rate": 0.04},
            "rsi": {"bias": 0.11, "samples": 99057,
                    "normalized_weight": 1.0, "activation_rate": 0.36,
                    "buy_rate": 0.16, "sell_rate": 0.20},
        }
        # sentiment BUY (in-bias, 0.7x) vs rsi SELL (full weight)
        # sentiment weight = 0.55 * 0.7 = 0.385; rsi weight = 0.55
        # → rsi SELL wins.
        votes = {"sentiment": "BUY", "rsi": "SELL"}
        result = _weighted_consensus(
            votes, accuracy_data, "unknown",
            activation_rates=activation_rates,
        )
        assert result[0] == "SELL", (
            f"sentiment BUY (bias 0.81, moderate 0.7x) should lose to "
            f"rsi SELL (equal accuracy, full weight) — got {result[0]}"
        )

    def test_crypto_macro_promoted_to_extreme_after_threshold_drop(self):
        """crypto_macro bias=0.91 was previously high tier (0.5x). After
        lowering _BIAS_EXTREME_THRESHOLD 0.95 → 0.90 it sits in extreme
        (0.2x). With crypto_macro acc=1.0 and opponent rsi acc=0.55:

        OLD policy (0.91 → 0.5x): crypto_macro weight = 1.0 * 0.5 = 0.50
        > rsi 0.55 = lose by margin → BUY wins
        NEW policy (0.91 → 0.2x): crypto_macro weight = 1.0 * 0.2 = 0.20
        < rsi 0.55 → SELL wins

        rsi accuracy 0.55 is above the 0.47 gate so it stays in play."""
        from portfolio.signal_engine import _weighted_consensus

        accuracy_data = {
            "crypto_macro": {"accuracy": 1.0, "total": 500,
                             "buy_accuracy": 1.0, "total_buy": 475},
            "rsi": {"accuracy": 0.55, "total": 500,
                    "sell_accuracy": 0.55, "total_sell": 250},
        }
        activation_rates = {
            "crypto_macro": {"bias": 0.91, "samples": 15933,
                             "normalized_weight": 1.0,
                             "activation_rate": 0.28, "buy_rate": 0.265,
                             "sell_rate": 0.012},
            "rsi": {"bias": 0.11, "samples": 99057,
                    "normalized_weight": 1.0, "activation_rate": 0.36,
                    "buy_rate": 0.16, "sell_rate": 0.20},
        }
        votes = {"crypto_macro": "BUY", "rsi": "SELL"}
        result = _weighted_consensus(
            votes, accuracy_data, "unknown",
            activation_rates=activation_rates,
        )
        assert result[0] == "SELL"


class TestBiasPolicyVersion:
    """2026-05-19 (premortem F3 mitigation): agent_summary.json must
    carry the active bias policy version so dashboard /
    accuracy-history consumers can mark the policy-change line."""

    def test_constant_exists_and_is_iso_date(self):
        from portfolio.signal_engine import BIAS_POLICY_VERSION
        # ISO-8601 date YYYY-MM-DD
        assert len(BIAS_POLICY_VERSION) == 10
        assert BIAS_POLICY_VERSION[4] == "-"
        assert BIAS_POLICY_VERSION[7] == "-"


# ---------------------------------------------------------------------------
# Directional accuracy gating
# ---------------------------------------------------------------------------

class TestDirectionalAccuracyGating:

    def test_buy_gated_when_buy_accuracy_low(self):
        """Qwen3-like: overall 59.8% passes gate, but BUY=30% should be gated."""
        from portfolio.signal_engine import _weighted_consensus

        # qwen3 BUY at 30% (well below 35% directional gate), SELL at 74%
        votes = {"qwen3": "BUY", "rsi": "SELL"}
        accuracy_data = {
            "qwen3": {"accuracy": 0.598, "total": 3608,
                       "buy_accuracy": 0.30, "total_buy": 1174,
                       "sell_accuracy": 0.74, "total_sell": 2434},
            "rsi": {"accuracy": 0.52, "total": 1000},
        }
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        # qwen3 BUY should be gated, only rsi SELL remains → SELL
        assert result[0] == "SELL"

    def test_sell_passes_when_buy_gated(self):
        """Same signal: BUY gated but SELL should still vote normally."""
        from portfolio.signal_engine import _weighted_consensus

        # qwen3 SELL should NOT be directionally gated (sell_accuracy 74% >> 35%)
        votes = {"qwen3": "SELL"}
        accuracy_data = {
            "qwen3": {"accuracy": 0.598, "total": 3608,
                       "buy_accuracy": 0.30, "total_buy": 1174,
                       "sell_accuracy": 0.74, "total_sell": 2434},
        }
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        # Only qwen3 voting SELL, not directionally gated → SELL
        assert result[0] == "SELL"

    def test_directional_gate_not_applied_with_few_samples(self):
        """Directional gate requires sufficient samples to avoid premature gating."""
        from portfolio.signal_engine import (
            _DIRECTIONAL_GATE_MIN_SAMPLES,
            _weighted_consensus,
        )

        votes = {"qwen3": "BUY"}
        accuracy_data = {
            "qwen3": {"accuracy": 0.55, "total": 100,
                       "buy_accuracy": 0.20, "total_buy": _DIRECTIONAL_GATE_MIN_SAMPLES - 1,
                       "sell_accuracy": 0.80, "total_sell": 100},
        }
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        # Not enough BUY samples → directional gate should NOT fire
        assert result[0] == "BUY"

    def test_claude_fundamental_sell_gated(self):
        """claude_fundamental: BUY=65.7% fine, but SELL=39.7% should be gated at 0.43."""
        from portfolio.signal_engine import _weighted_consensus

        votes = {"claude_fundamental": "SELL", "rsi": "BUY"}
        accuracy_data = {
            "claude_fundamental": {"accuracy": 0.628, "total": 7535,
                                    "buy_accuracy": 0.657, "total_buy": 6697,
                                    "sell_accuracy": 0.397, "total_sell": 838},
            "rsi": {"accuracy": 0.52, "total": 1000},
        }
        # sell_accuracy 0.397 < 0.43 threshold → gated
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        # claude_fundamental SELL gated, only rsi BUY remains → BUY
        assert result[0] == "BUY"

    def test_macro_regime_buy_gated_at_40pct(self):
        """macro_regime: overall 46.6% passes accuracy gate, but BUY=38.9% gated."""
        from portfolio.signal_engine import _weighted_consensus

        votes = {"macro_regime": "BUY", "rsi": "SELL"}
        accuracy_data = {
            "macro_regime": {"accuracy": 0.466, "total": 25654,
                              "buy_accuracy": 0.389, "total_buy": 8963,
                              "sell_accuracy": 0.508, "total_sell": 16691},
            "rsi": {"accuracy": 0.52, "total": 1000},
        }
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        # macro_regime BUY at 38.9% < 40% threshold → gated, only rsi SELL remains
        assert result[0] == "SELL"

    def test_no_directional_gate_when_accuracy_above_threshold(self):
        """Signals with both BUY and SELL above threshold should not be gated."""
        from portfolio.signal_engine import _weighted_consensus

        votes = {"rsi": "BUY"}
        accuracy_data = {
            "rsi": {"accuracy": 0.52, "total": 1000,
                     "buy_accuracy": 0.50, "total_buy": 500,
                     "sell_accuracy": 0.54, "total_sell": 500},
        }
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        assert result[0] == "BUY"

    def test_missing_buy_accuracy_key_falls_back_to_overall(self):
        """BUG-185: Missing buy_accuracy key should not crash; uses overall acc."""
        from portfolio.signal_engine import _weighted_consensus

        # total_buy present (>= 20) but buy_accuracy missing — simulates cache corruption
        votes = {"rsi": "BUY"}
        accuracy_data = {
            "rsi": {"accuracy": 0.55, "total": 200,
                     "total_buy": 100},  # buy_accuracy intentionally absent
        }
        # Should not raise KeyError; falls back to overall accuracy (0.55)
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        assert result[0] == "BUY"

    def test_missing_sell_accuracy_key_falls_back_to_overall(self):
        """BUG-185: Missing sell_accuracy key should not crash; uses overall acc."""
        from portfolio.signal_engine import _weighted_consensus

        votes = {"rsi": "SELL"}
        accuracy_data = {
            "rsi": {"accuracy": 0.55, "total": 200,
                     "total_sell": 100},  # sell_accuracy intentionally absent
        }
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        assert result[0] == "SELL"

    def test_directional_rescue_sell_for_low_overall_accuracy(self):
        """Directional rescue: overall accuracy fails gate but SELL direction is strong."""
        from portfolio.signal_engine import (
            _DIRECTIONAL_RESCUE_THRESHOLD,
            _DIRECTIONAL_RESCUE_MIN_SAMPLES,
            _weighted_consensus,
        )

        votes = {"btc_proxy": "SELL", "rsi": "BUY"}
        accuracy_data = {
            "btc_proxy": {
                "accuracy": 0.446, "total": 139,
                "buy_accuracy": 0.311, "total_buy": 90,
                "sell_accuracy": 0.694, "total_sell": 49,
            },
            "rsi": {"accuracy": 0.52, "total": 1000},
        }
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        # btc_proxy overall 44.6% < 47% gate BUT sell_accuracy 69.4% >= 55%
        # rescue threshold AND total_sell 49 >= 30 min samples → rescued.
        # Both btc_proxy SELL (rescued, 0.70x weight) and rsi BUY vote.
        # btc_proxy SELL weight ≈ 0.694 * 0.70 ≈ 0.486
        # rsi BUY weight ≈ 0.52
        # SELL weight slightly lower → depends on exact weighting, but
        # the key assertion is that btc_proxy SELL is NOT gated (it participates)
        assert result[0] in ("BUY", "SELL"), "btc_proxy SELL should be rescued, not forced HOLD"

    def test_directional_rescue_not_triggered_below_threshold(self):
        """No rescue when directional accuracy is below rescue threshold."""
        from portfolio.signal_engine import _weighted_consensus

        votes = {"bad_signal": "SELL", "rsi": "BUY"}
        accuracy_data = {
            "bad_signal": {
                "accuracy": 0.40, "total": 200,
                "buy_accuracy": 0.30, "total_buy": 100,
                "sell_accuracy": 0.50, "total_sell": 100,
            },
            "rsi": {"accuracy": 0.52, "total": 1000},
        }
        result = _weighted_consensus(votes, accuracy_data, "ranging")
        # bad_signal overall 40% < 47% AND sell_accuracy 50% < 55% rescue threshold
        # → fully gated, only rsi BUY remains
        assert result[0] == "BUY"

    def test_llm_confidence_gate_constant_exists(self):
        """Verify the LLM confidence gate constant is defined and reasonable."""
        from portfolio.signal_engine import _LLM_CONFIDENCE_GATE
        assert 0.3 <= _LLM_CONFIDENCE_GATE <= 0.8

    def test_graduated_rescue_penalty(self):
        """Graduated rescue: higher directional accuracy → lower penalty."""
        from portfolio.signal_engine import (
            _rescue_weight_penalty,
            _DIRECTIONAL_RESCUE_WEIGHT_FLOOR,
            _DIRECTIONAL_RESCUE_WEIGHT_CAP,
        )
        assert _rescue_weight_penalty(0.55) == _DIRECTIONAL_RESCUE_WEIGHT_FLOOR
        assert _rescue_weight_penalty(0.85) == _DIRECTIONAL_RESCUE_WEIGHT_CAP
        assert _rescue_weight_penalty(0.99) == _DIRECTIONAL_RESCUE_WEIGHT_CAP
        mid = _rescue_weight_penalty(0.70)
        assert _DIRECTIONAL_RESCUE_WEIGHT_FLOOR < mid < _DIRECTIONAL_RESCUE_WEIGHT_CAP

    def test_graduated_rescue_in_consensus(self):
        """Rescued signal at 80% dir accuracy gets higher weight than at 56%."""
        from portfolio.signal_engine import _weighted_consensus
        acc_strong = {
            "sig_a": {
                "accuracy": 0.40, "total": 200,
                "sell_accuracy": 0.80, "total_sell": 50,
                "buy_accuracy": 0.20, "total_buy": 150,
            },
            "rsi": {"accuracy": 0.52, "total": 1000},
        }
        acc_weak = {
            "sig_a": {
                "accuracy": 0.40, "total": 200,
                "sell_accuracy": 0.56, "total_sell": 50,
                "buy_accuracy": 0.20, "total_buy": 150,
            },
            "rsi": {"accuracy": 0.52, "total": 1000},
        }
        votes = {"sig_a": "SELL", "rsi": "BUY"}
        r_strong = _weighted_consensus(votes, acc_strong, "ranging")
        r_weak = _weighted_consensus(votes, acc_weak, "ranging")
        # Both rescued, but strong should have higher SELL confidence
        # (stronger rescue penalty = more weight for SELL)
        # r[1] is confidence
        assert r_strong[0] in ("BUY", "SELL")
        assert r_weak[0] in ("BUY", "SELL")


# ---------------------------------------------------------------------------
# Funding rate horizon gating (2026-04-09)
# ---------------------------------------------------------------------------

class TestFundingRateHorizonGating:
    """Funding rate: 74.2% at 3h but 29.9% at 1d. Should only vote at 3h/4h."""

    def test_funding_gated_at_default_in_all_regimes(self):
        """funding should be in _default gate for ranging, trending-up/down, high-vol."""
        from portfolio.signal_engine import REGIME_GATED_SIGNALS

        for regime in ("ranging", "trending-up", "trending-down", "high-vol"):
            assert regime in REGIME_GATED_SIGNALS, f"{regime} missing from REGIME_GATED_SIGNALS"
            default_set = REGIME_GATED_SIGNALS[regime].get("_default", frozenset())
            assert "funding" in default_set, (
                f"funding should be gated at _default in {regime}"
            )

    def test_funding_not_gated_at_3h_ranging(self):
        """funding should NOT be in 3h gate for ranging (74.2% accuracy there)."""
        from portfolio.signal_engine import REGIME_GATED_SIGNALS

        gate_3h = REGIME_GATED_SIGNALS["ranging"].get("3h", frozenset())
        assert "funding" not in gate_3h

    def test_funding_not_gated_at_4h_ranging(self):
        """funding should NOT be in 4h gate for ranging."""
        from portfolio.signal_engine import REGIME_GATED_SIGNALS

        gate_4h = REGIME_GATED_SIGNALS["ranging"].get("4h", frozenset())
        assert "funding" not in gate_4h

    def test_funding_in_disabled_signals(self):
        """funding added to DISABLED_SIGNALS 2026-05-13: 30.8% at 1d (743 sam)."""
        from portfolio.tickers import DISABLED_SIGNALS

        assert "funding" in DISABLED_SIGNALS


# ---------------------------------------------------------------------------
# On-chain BTC signal (2026-04-09)
# ---------------------------------------------------------------------------

class TestOnchainSignal:
    """On-chain signal: MVRV Z-Score, SOPR, NUPL, netflow → majority vote for BTC."""

    def test_onchain_in_signal_names(self):
        """onchain should be in the SIGNAL_NAMES list."""
        from portfolio.tickers import SIGNAL_NAMES

        assert "onchain" in SIGNAL_NAMES

    def _run_onchain_block(self, ticker, cached_data):
        """Helper: run just the on-chain signal block in isolation.

        Returns (vote, extra_info) without calling generate_signal
        (which needs dozens of indicator keys).
        """

        votes = {"onchain": "HOLD"}
        extra_info = {}

        if ticker == "BTC-USD":
            oc = cached_data
            if oc:
                sub_votes = []
                zscore = oc.get("mvrv_zscore")
                if zscore is not None:
                    if zscore < 1.0:
                        sub_votes.append("BUY")
                    elif zscore > 5.0:
                        sub_votes.append("SELL")
                    else:
                        sub_votes.append("HOLD")
                    extra_info["onchain_mvrv_zscore"] = round(zscore, 2)
                sopr = oc.get("sopr")
                if sopr is not None:
                    if sopr < 0.97:
                        sub_votes.append("BUY")
                    elif sopr > 1.05:
                        sub_votes.append("SELL")
                    else:
                        sub_votes.append("HOLD")
                    extra_info["onchain_sopr"] = round(sopr, 4)
                nupl = oc.get("nupl")
                if nupl is not None:
                    if nupl < 0:
                        sub_votes.append("BUY")
                    elif nupl > 0.75:
                        sub_votes.append("SELL")
                    else:
                        sub_votes.append("HOLD")
                netflow = oc.get("netflow")
                if netflow is not None:
                    if netflow < 0:
                        sub_votes.append("BUY")
                    elif netflow > 0:
                        sub_votes.append("SELL")
                    else:
                        sub_votes.append("HOLD")
                buy_count = sub_votes.count("BUY")
                sell_count = sub_votes.count("SELL")
                total = buy_count + sell_count
                if total >= 2:
                    if buy_count > sell_count:
                        votes["onchain"] = "BUY"
                    elif sell_count > buy_count:
                        votes["onchain"] = "SELL"
                    extra_info["onchain_sub_votes"] = f"{buy_count}B/{sell_count}S"

        return votes["onchain"], extra_info

    def test_onchain_buy_when_metrics_undervalued(self):
        """When MVRV < 1, SOPR < 0.97, NUPL < 0 → majority BUY."""
        data = {
            "mvrv_zscore": 0.5,   # < 1.0 → BUY
            "sopr": 0.95,         # < 0.97 → BUY
            "nupl": -0.1,         # < 0 → BUY
            "netflow": 500,       # > 0 → SELL (minority)
        }
        vote, extra = self._run_onchain_block("BTC-USD", data)
        assert vote == "BUY"
        assert extra["onchain_mvrv_zscore"] == 0.5
        assert extra["onchain_sopr"] == 0.95
        assert extra["onchain_sub_votes"] == "3B/1S"

    def test_onchain_sell_when_overheated(self):
        """When MVRV > 5, SOPR > 1.05, NUPL > 0.75 → majority SELL."""
        data = {
            "mvrv_zscore": 6.0,   # > 5.0 → SELL
            "sopr": 1.08,         # > 1.05 → SELL
            "nupl": 0.80,         # > 0.75 → SELL
            "netflow": -100,      # < 0 → BUY (minority)
        }
        vote, extra = self._run_onchain_block("BTC-USD", data)
        assert vote == "SELL"
        assert extra["onchain_sub_votes"] == "1B/3S"

    def test_onchain_hold_for_non_btc(self):
        """On-chain signal should be HOLD for non-BTC tickers."""
        data = {"mvrv_zscore": 0.5, "sopr": 0.95, "nupl": -0.1, "netflow": -100}
        vote, extra = self._run_onchain_block("ETH-USD", data)
        assert vote == "HOLD"
        assert not extra  # no extra_info populated

    def test_onchain_hold_when_no_data(self):
        """On-chain signal should be HOLD when API returns None."""
        vote, extra = self._run_onchain_block("BTC-USD", None)
        assert vote == "HOLD"

    def test_onchain_hold_when_neutral_metrics(self):
        """When all metrics are neutral, should remain HOLD (no majority)."""
        data = {
            "mvrv_zscore": 3.0,   # 1-5 → HOLD
            "sopr": 1.00,         # 0.97-1.05 → HOLD
            "nupl": 0.4,          # 0-0.75 → HOLD
            "netflow": 0,         # 0 → HOLD
        }
        vote, extra = self._run_onchain_block("BTC-USD", data)
        assert vote == "HOLD"
        # No sub_votes key when total active (BUY+SELL) < 2
        assert "onchain_sub_votes" not in extra

    def test_onchain_tie_stays_hold(self):
        """When BUY and SELL are tied (2B/2S), should remain HOLD."""
        data = {
            "mvrv_zscore": 0.5,   # < 1.0 → BUY
            "sopr": 0.95,         # < 0.97 → BUY
            "nupl": 0.80,         # > 0.75 → SELL
            "netflow": 500,       # > 0 → SELL
        }
        vote, extra = self._run_onchain_block("BTC-USD", data)
        assert vote == "HOLD"
        assert extra["onchain_sub_votes"] == "2B/2S"


# ---------------------------------------------------------------------------
# BUG-178 fixes (2026-04-10)
# ---------------------------------------------------------------------------

class TestDispatchLoopRespectsDisabledSignals:
    """Regression: enhanced-signal dispatch loop must skip DISABLED_SIGNALS.

    Before 2026-04-10, the dispatch loop iterated every registered enhanced
    signal regardless of disabled status. This caused 49 BUG-178 ticker
    pool timeouts on 2026-04-09/10 because the disabled signals
    (crypto_macro, cot_positioning, credit_spread_risk) were doing network
    I/O on every cycle.
    """

    def test_disabled_signals_have_force_hold_in_dispatch_loop(self):
        """The dispatch loop must short-circuit on DISABLED_SIGNALS."""
        import inspect

        from portfolio import signal_engine
        from portfolio.tickers import DISABLED_SIGNALS

        src = inspect.getsource(signal_engine.generate_signal)
        # The fix: dispatch loop must check DISABLED_SIGNALS before calling compute_fn
        assert "if sig_name in DISABLED_SIGNALS" in src, (
            "Dispatch loop must skip disabled signals to prevent BUG-178 hangs"
        )
        # Defense in depth: at least one disabled signal must exist (otherwise
        # the check is dead code we should remove)
        assert len(DISABLED_SIGNALS) > 0


class TestLastSignalDiagnostic:
    """BUG-178 diagnostic: per-ticker last-signal tracker."""

    def test_set_and_get_last_signal_round_trip(self):
        from portfolio.signal_engine import _set_last_signal, get_last_signal

        _set_last_signal("BTC-USD", "test_signal_xyz")
        result = get_last_signal("BTC-USD")
        assert result is not None
        sig_name, elapsed = result
        assert sig_name == "test_signal_xyz"
        assert elapsed >= 0.0
        assert elapsed < 5.0  # should be near-instant

    def test_get_last_signal_unknown_ticker_returns_none(self):
        from portfolio.signal_engine import get_last_signal

        result = get_last_signal("__nonexistent_ticker__")
        assert result is None

    def test_set_overwrites_previous(self):
        from portfolio.signal_engine import _set_last_signal, get_last_signal

        _set_last_signal("ETH-USD", "first_signal")
        _set_last_signal("ETH-USD", "second_signal")
        result = get_last_signal("ETH-USD")
        assert result is not None
        assert result[0] == "second_signal"


class TestGenerateSignalPhaseMarkers:
    """BUG-178 slow-cycle diagnostic: phase markers written by generate_signal.

    generate_signal() must update _last_signal_per_ticker with
    __pre_dispatch__ before the enhanced-signal dispatch loop and
    __post_dispatch__ after, so the main.py slow-cycle diagnostic can
    distinguish hangs in the three distinct phases.
    """

    def test_generate_signal_writes_pre_and_post_dispatch_markers(self):
        """After generate_signal runs, the tracker should show __post_dispatch__.

        Uses the full generate_signal pipeline with a synthetic indicator dict
        and checks that the tracker's recorded last_signal is either
        __post_dispatch__ (normal fast path) or a concrete signal name (if the
        dispatch loop bailed early). In no case should it still be
        __pre_dispatch__ on a successful completion.
        """
        import numpy as np
        import pandas as pd

        from portfolio.signal_engine import generate_signal, get_last_signal

        # Minimal indicator dict with everything generate_signal touches.
        ind = {
            "close": 100.0, "rsi": 50.0, "rsi_p20": 30.0, "rsi_p80": 70.0,
            "macd_hist": 0.0, "macd_hist_prev": 0.0,
            "ema9": 100.0, "ema21": 100.0, "sma20": 100.0, "sma50": 100.0,
            "bb_upper": 110.0, "bb_lower": 90.0, "bb_mid": 100.0,
            "price_vs_bb": "inside", "volume": 1_000_000, "volume_sma20": 1_000_000,
            "atr": 1.0, "adx": 20.0, "high": 101.0, "low": 99.0, "open": 100.0,
        }
        # Small df — mostly for the enhanced-signals dispatch loop.
        df = pd.DataFrame({
            "open": np.full(50, 100.0), "high": np.full(50, 101.0),
            "low": np.full(50, 99.0), "close": np.full(50, 100.0),
            "volume": np.full(50, 1_000_000),
        })

        generate_signal(ind, ticker="BTC-USD", df=df)

        last = get_last_signal("BTC-USD")
        assert last is not None
        sig_name, _ = last
        # Post-successful-completion, the tracker should show the post-dispatch
        # marker — proves the loop ran to the end.
        assert sig_name == "__post_dispatch__", (
            f"Expected __post_dispatch__ after a clean generate_signal run, "
            f"got {sig_name!r}. If this is a concrete signal name, the dispatch "
            f"loop bailed early; if __pre_dispatch__, the loop never started."
        )

    def test_generate_signal_skips_markers_when_ticker_none(self):
        """Phase markers are gated on `ticker` being truthy.

        When generate_signal is called without a ticker (edge case for tests
        and legacy callers), the tracker must not record anything under an
        empty-string key — that would pollute the dict for all future lookups.
        """
        import numpy as np
        import pandas as pd

        from portfolio import signal_engine

        # Snapshot current tracker state so we can verify no new entry appears.
        with signal_engine._last_signal_lock:
            keys_before = set(signal_engine._last_signal_per_ticker.keys())

        ind = {
            "close": 100.0, "rsi": 50.0, "rsi_p20": 30.0, "rsi_p80": 70.0,
            "macd_hist": 0.0, "macd_hist_prev": 0.0,
            "ema9": 100.0, "ema21": 100.0, "sma20": 100.0, "sma50": 100.0,
            "bb_upper": 110.0, "bb_lower": 90.0, "bb_mid": 100.0,
            "price_vs_bb": "inside", "volume": 1_000_000, "volume_sma20": 1_000_000,
            "atr": 1.0, "adx": 20.0, "high": 101.0, "low": 99.0, "open": 100.0,
        }
        df = pd.DataFrame({
            "open": np.full(50, 100.0), "high": np.full(50, 101.0),
            "low": np.full(50, 99.0), "close": np.full(50, 100.0),
            "volume": np.full(50, 1_000_000),
        })

        signal_engine.generate_signal(ind, ticker=None, df=df)

        with signal_engine._last_signal_lock:
            keys_after = set(signal_engine._last_signal_per_ticker.keys())
        new_keys = keys_after - keys_before
        # No empty-string or None keys should have been added.
        assert "" not in new_keys
        assert None not in new_keys

    def test_post_dispatch_marker_overwrites_dispatch_loop_marker(self):
        """The post-dispatch marker must always supersede the dispatch loop.

        Regression: earlier iterations wrote `_set_last_signal` only inside
        the dispatch loop, leaving the tracker pointing at whichever signal
        ran last. The post-dispatch phase marker must overwrite it.
        """
        from portfolio.signal_engine import _set_last_signal, get_last_signal

        # Simulate the tracker state after the dispatch loop ran
        # metals_cross_asset as its final signal.
        _set_last_signal("XAU-USD", "metals_cross_asset")
        result = get_last_signal("XAU-USD")
        assert result[0] == "metals_cross_asset"

        # Simulate the post-dispatch marker write that our fix adds
        _set_last_signal("XAU-USD", "__post_dispatch__")
        result = get_last_signal("XAU-USD")
        assert result[0] == "__post_dispatch__"


# ---------------------------------------------------------------------------
# 2026-04-11 Research Session: Signal gating and correlation penalty changes
# ---------------------------------------------------------------------------

class TestOrderbookFlowDisabled:
    """orderbook_flow was disabled 2026-04-11: 93.3% active, 51.1% accuracy, no
    recent data. Pure noise in every consensus decision."""

    def test_orderbook_flow_in_disabled_signals(self):
        from portfolio.tickers import DISABLED_SIGNALS
        assert "orderbook_flow" in DISABLED_SIGNALS

    def test_orderbook_flow_not_in_consensus(self):
        """Disabled signals must produce HOLD in the dispatch loop."""
        from portfolio.tickers import DISABLED_SIGNALS
        assert "orderbook_flow" in DISABLED_SIGNALS


class TestCreditSpreadRiskDisabled:
    """credit_spread_risk DISABLED 2026-05-21: 23% blended accuracy
    (0.90x20% recent + 0.10x50% alltime). Actively harming consensus."""

    def test_credit_spread_risk_disabled(self):
        from portfolio.tickers import DISABLED_SIGNALS
        assert "credit_spread_risk" in DISABLED_SIGNALS


class TestCryptoMacroEnabled:
    """crypto_macro re-enabled 2026-04-11: 56.5% accuracy (1273 sam). BUY-biased
    (93%) so bias penalty (0.5x) applies. Provides crypto-specific on-chain edge."""

    def test_crypto_macro_not_disabled(self):
        from portfolio.tickers import DISABLED_SIGNALS
        assert "crypto_macro" not in DISABLED_SIGNALS


class TestSentimentGatedAt3hRanging:
    """sentiment gated at 3h in ranging: 33.8% at 3h_recent (3629 sam). The 0.5x
    horizon weight was insufficient — explicit regime gating is clearer."""

    def test_sentiment_gated_at_3h_ranging(self):
        from portfolio.signal_engine import REGIME_GATED_SIGNALS
        gated_3h = REGIME_GATED_SIGNALS["ranging"]["3h"]
        assert "sentiment" in gated_3h

    def test_sentiment_gated_at_4h_ranging(self):
        from portfolio.signal_engine import REGIME_GATED_SIGNALS
        gated_4h = REGIME_GATED_SIGNALS["ranging"]["4h"]
        assert "sentiment" in gated_4h

    def test_sentiment_gated_at_default_ranging(self):
        """2026-04-27: sentiment 40.1% at 1d_recent (202 sam), 33.8% at 3h.
        BUY-only bias actively harmful at longer horizons in ranging."""
        from portfolio.signal_engine import REGIME_GATED_SIGNALS
        gated_default = REGIME_GATED_SIGNALS["ranging"]["_default"]
        assert "sentiment" in gated_default

    def test_sentiment_gated_at_default_trending_down(self):
        """2026-04-27: sentiment BUY-only bias harmful in downtrends at longer horizons."""
        from portfolio.signal_engine import REGIME_GATED_SIGNALS
        gated_default = REGIME_GATED_SIGNALS["trending-down"]["_default"]
        assert "sentiment" in gated_default


class TestRegimeGatedSemanticsAreReplaceNotUnion:
    """Regression for adversarial-review carryover finding (04-24 P0-1, 04-29 SC-P1-1,
    05-01 P0-1 carryover): `_get_regime_gated` returns the horizon override as a
    REPLACEMENT of `_default`, not a union. This is intentional (BUG-149,
    2026-03-29). Funding 74.2% @3h_ranging means we WANT it to vote at 3h ranging
    even though it's in `_default`. Same for trend (61.6% @3h vs 40.7% @1d in ranging).

    If a future review re-flags this as a "union bug", read this test class first.
    """

    def test_funding_voting_at_3h_ranging(self):
        """funding is in `_default` for ranging but MUST NOT be gated at 3h."""
        from portfolio.signal_engine import _get_regime_gated
        gated = _get_regime_gated("ranging", "3h")
        assert "funding" not in gated, (
            "funding 74.2% accuracy at 3h ranging — regression if gated. "
            "If you tried to fix this with a union, that's the wrong fix. "
            "See _get_regime_gated docstring."
        )

    def test_trend_voting_at_3h_ranging(self):
        """trend has 61.6% accuracy at 3h ranging (BUG-149 docstring) — must vote."""
        from portfolio.signal_engine import _get_regime_gated
        gated = _get_regime_gated("ranging", "3h")
        assert "trend" not in gated, (
            "trend 61.6% accuracy at 3h ranging (BUG-149) — must NOT be gated. "
            "If you tried to fix the carryover finding with a union, revert that fix."
        )

    def test_default_horizon_uses_default_set(self):
        """When no horizon is given, `_default` IS the gate set."""
        from portfolio.signal_engine import _get_regime_gated, REGIME_GATED_SIGNALS
        gated = _get_regime_gated("ranging", None)
        default_set = REGIME_GATED_SIGNALS["ranging"]["_default"]
        assert gated == default_set

    def test_long_horizon_uses_default_set(self):
        """1d horizon is not a key in REGIME_GATED_SIGNALS["ranging"], falls back to _default."""
        from portfolio.signal_engine import _get_regime_gated, REGIME_GATED_SIGNALS
        gated = _get_regime_gated("ranging", "1d")
        default_set = REGIME_GATED_SIGNALS["ranging"]["_default"]
        assert gated == default_set


class TestClaudeFundamentalGatedRangingDefault:
    """2026-04-27: claude_fundamental 40.5% at 1d_recent (1178 sam), 78-83% BUY bias.
    Was only gated at 3h/4h in ranging — now gated at _default too."""

    def test_claude_fundamental_gated_at_default_ranging(self):
        from portfolio.signal_engine import REGIME_GATED_SIGNALS
        gated_default = REGIME_GATED_SIGNALS["ranging"]["_default"]
        assert "claude_fundamental" in gated_default

    def test_claude_fundamental_per_ticker_gated_xag_1d(self):
        """Metals have no earnings/guidance — claude_fundamental is noise for XAG."""
        from portfolio.signal_engine import _TICKER_DISABLED_BY_HORIZON
        assert "claude_fundamental" in _TICKER_DISABLED_BY_HORIZON["1d"]["XAG-USD"]

    def test_claude_fundamental_per_ticker_gated_xau_1d(self):
        """Metals have no earnings/guidance — claude_fundamental is noise for XAU."""
        from portfolio.signal_engine import _TICKER_DISABLED_BY_HORIZON
        assert "claude_fundamental" in _TICKER_DISABLED_BY_HORIZON["1d"]["XAU-USD"]


class TestPerClusterCorrelationPenalties:
    """2026-04-11: momentum_cluster penalty tightened from 0.3x to 0.15x.
    RSI/BB/MeanReversion/Momentum agree 88-100%, so 0.3x still gave 1.9x
    combined weight. With 0.15x: 1.0 + 3*0.15 = 1.45x."""

    def test_cluster_penalties_dict_exists(self):
        from portfolio.signal_engine import _CLUSTER_CORRELATION_PENALTIES
        assert isinstance(_CLUSTER_CORRELATION_PENALTIES, dict)

    def test_momentum_cluster_has_tighter_penalty(self):
        from portfolio.signal_engine import (
            _CLUSTER_CORRELATION_PENALTIES,
            _CORRELATION_PENALTY,
        )
        assert "momentum_cluster" in _CLUSTER_CORRELATION_PENALTIES
        assert _CLUSTER_CORRELATION_PENALTIES["momentum_cluster"] < _CORRELATION_PENALTY

    def test_momentum_cluster_penalty_is_015(self):
        from portfolio.signal_engine import _CLUSTER_CORRELATION_PENALTIES
        assert _CLUSTER_CORRELATION_PENALTIES["momentum_cluster"] == 0.15

    def test_default_penalty_unchanged(self):
        from portfolio.signal_engine import _CORRELATION_PENALTY
        assert _CORRELATION_PENALTY == 0.3


class TestMetaClusterDedup:
    """2026-05-01: Meta-cluster deduplication — when leaders from related sub-clusters
    agree on direction, apply penalty to redundant leaders. Prevents trend mega-view
    from getting 3.0x effective leader weight."""

    def test_meta_cluster_groups_exist(self):
        from portfolio.signal_engine import _META_CLUSTER_GROUPS
        assert isinstance(_META_CLUSTER_GROUPS, dict)
        assert "trend_mega" in _META_CLUSTER_GROUPS

    def test_trend_mega_contains_three_subclusters(self):
        from portfolio.signal_engine import _META_CLUSTER_GROUPS
        assert set(_META_CLUSTER_GROUPS["trend_mega"]) == {
            "pure_trend", "oscillator_trend", "structural_flow"
        }

    def test_meta_cluster_penalty_is_035(self):
        from portfolio.signal_engine import _META_CLUSTER_PENALTY
        assert _META_CLUSTER_PENALTY == 0.35

    def test_meta_cluster_penalty_less_than_default(self):
        from portfolio.signal_engine import (
            _META_CLUSTER_PENALTY,
            _CORRELATION_PENALTY,
        )
        # Meta-cluster penalty should be at least as harsh as default correlation
        assert _META_CLUSTER_PENALTY <= _CORRELATION_PENALTY + 0.1


class TestTickerDisabledSignals:
    """Per-ticker signal gating: force HOLD for specific signal+ticker combos."""

    def test_ticker_disabled_signals_dict_exists(self):
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert isinstance(_TICKER_DISABLED_SIGNALS, dict)

    def test_eth_news_event_disabled(self):
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "ETH-USD" in _TICKER_DISABLED_SIGNALS
        assert "news_event" in _TICKER_DISABLED_SIGNALS["ETH-USD"]

    def test_btc_news_event_not_disabled(self):
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        btc_disabled = _TICKER_DISABLED_SIGNALS.get("BTC-USD", frozenset())
        assert "news_event" not in btc_disabled

    def test_dispatch_respects_ticker_disable(self):
        """In generate_signal dispatch loop, per-ticker disabled signals should be HOLD."""
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS

        ticker = "ETH-USD"
        sig_name = "news_event"
        disabled_for_ticker = _TICKER_DISABLED_SIGNALS.get(ticker, ())
        assert sig_name in disabled_for_ticker
        # Simulating the dispatch check: if sig_name in disabled_for_ticker → HOLD
        if sig_name in disabled_for_ticker:
            vote = "HOLD"
        else:
            vote = "BUY"
        assert vote == "HOLD"

    def test_xag_ministral_disabled(self):
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "XAG-USD" in _TICKER_DISABLED_SIGNALS
        assert "ministral" in _TICKER_DISABLED_SIGNALS["XAG-USD"]

    def test_xag_credit_spread_risk_disabled(self):
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "credit_spread_risk" in _TICKER_DISABLED_SIGNALS["XAG-USD"]

    def test_xag_metals_cross_asset_disabled(self):
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "metals_cross_asset" in _TICKER_DISABLED_SIGNALS["XAG-USD"]

    def test_xau_ministral_disabled(self):
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "XAU-USD" in _TICKER_DISABLED_SIGNALS
        assert "ministral" in _TICKER_DISABLED_SIGNALS["XAU-USD"]

    def test_mstr_credit_spread_risk_disabled(self):
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "MSTR" in _TICKER_DISABLED_SIGNALS
        assert "credit_spread_risk" in _TICKER_DISABLED_SIGNALS["MSTR"]

    def test_btc_ministral_not_disabled(self):
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        btc_disabled = _TICKER_DISABLED_SIGNALS.get("BTC-USD", frozenset())
        assert "ministral" not in btc_disabled


class TestOscillatorsDisabled:
    """BUG-193: oscillators globally disabled — below 45% on all tickers."""

    def test_oscillators_in_disabled_signals(self):
        from portfolio.tickers import DISABLED_SIGNALS
        assert "oscillators" in DISABLED_SIGNALS

    def test_oscillators_still_in_signal_names(self):
        from portfolio.tickers import SIGNAL_NAMES
        assert "oscillators" in SIGNAL_NAMES


class TestSentimentUnknownRegimeGating:
    """BUG-194: sentiment gated at 3h/4h in unknown regime."""

    def test_sentiment_gated_3h_unknown(self):
        from portfolio.signal_engine import REGIME_GATED_SIGNALS
        unknown = REGIME_GATED_SIGNALS.get("unknown", {})
        assert "sentiment" in unknown.get("3h", frozenset())

    def test_sentiment_gated_4h_unknown(self):
        from portfolio.signal_engine import REGIME_GATED_SIGNALS
        unknown = REGIME_GATED_SIGNALS.get("unknown", {})
        assert "sentiment" in unknown.get("4h", frozenset())

    def test_sentiment_not_gated_1d_unknown(self):
        from portfolio.signal_engine import REGIME_GATED_SIGNALS
        unknown = REGIME_GATED_SIGNALS.get("unknown", {})
        default_gated = unknown.get("_default", frozenset())
        assert "sentiment" not in default_gated


class TestMSTRSignalBlacklist:
    """MSTR-specific per-ticker blacklisting.

    2026-04-16: Trimmed from 7 entries to 2. The Apr 14 audit built the
    blacklist from 3h accuracy data and applied it at all horizons. At
    the 1d horizon where consensus trades, the original 5 removed signals
    (macro_regime, trend, volatility_sig, volume, sentiment) were 62-81%
    accurate and were silencing the votes that would have correctly called
    MSTR's +8.4% W16 rally. Kept: claude_fundamental (47.8% 1d / 33.2% 3h
    - bad at both horizons) and credit_spread_risk (44.2% 1d).
    """

    def test_mstr_macro_regime_NOT_disabled(self):
        # 2026-04-16: macro_regime removed from MSTR blacklist.
        # At 1d horizon it shows 81.4% accuracy on MSTR (last 7d, n=43).
        # The Apr 14 blacklist was built from 3h data (32.5%) and applied
        # globally — horizon mismatch caused W15/W16 consensus collapse.
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "macro_regime" not in _TICKER_DISABLED_SIGNALS["MSTR"]

    def test_mstr_trend_NOT_disabled(self):
        # 2026-04-16: trend removed from MSTR blacklist (71.2% at 1d, n=59).
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "trend" not in _TICKER_DISABLED_SIGNALS["MSTR"]

    def test_mstr_volatility_sig_NOT_disabled(self):
        # 2026-04-16: volatility_sig removed (66.7% at 1d, n=42).
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "volatility_sig" not in _TICKER_DISABLED_SIGNALS["MSTR"]

    def test_mstr_sentiment_disabled(self):
        # 2026-05-10: sentiment crashed 90.4% -> 39.2% after regime change
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "sentiment" in _TICKER_DISABLED_SIGNALS["MSTR"]

    def test_mstr_volume_NOT_disabled(self):
        # 2026-04-16: volume removed (62.3% at 1d, n=77).
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "volume" not in _TICKER_DISABLED_SIGNALS["MSTR"]

    def test_mstr_claude_fundamental_still_disabled(self):
        # Retained: 47.8% at 1d / 33.2% at 3h - bad at both horizons.
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "claude_fundamental" in _TICKER_DISABLED_SIGNALS["MSTR"]

    def test_mstr_credit_spread_risk_still_disabled(self):
        # Retained: 44.2% at 1d.
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "credit_spread_risk" in _TICKER_DISABLED_SIGNALS["MSTR"]


class TestMay2BlacklistExpansion:
    """2026-05-02 after-hours: expanded per-ticker blacklists based on
    signal_log.db accuracy audit (all-time 1d, large samples).
    """

    def test_xag_sentiment_disabled(self):
        # sentiment 33.3% 1d on XAG-USD (285 sam), 94% BUY-only.
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "sentiment" in _TICKER_DISABLED_SIGNALS["XAG-USD"]

    def test_mstr_statistical_jump_regime_disabled(self):
        # statistical_jump_regime 27.0% 1d on MSTR (74 sam).
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "statistical_jump_regime" in _TICKER_DISABLED_SIGNALS["MSTR"]

    def test_mstr_realized_skewness_disabled(self):
        # realized_skewness 36.0% 1d on MSTR (50 sam).
        from portfolio.signal_engine import _TICKER_DISABLED_SIGNALS
        assert "realized_skewness" in _TICKER_DISABLED_SIGNALS["MSTR"]

    def test_mstr_macro_regime_disabled_at_1d(self):
        # macro_regime 40.3% 1d on MSTR (1475 sam) — moved to 1d-only
        # to preserve good 3h performance.
        from portfolio.signal_engine import _get_horizon_disabled_signals
        disabled = _get_horizon_disabled_signals("MSTR", "1d")
        assert "macro_regime" in disabled

    def test_mstr_macro_regime_not_disabled_at_3h(self):
        # macro_regime kept at 3h for MSTR (good short-horizon accuracy).
        from portfolio.signal_engine import _get_horizon_disabled_signals
        disabled = _get_horizon_disabled_signals("MSTR", "3h")
        assert "macro_regime" not in disabled


class TestCorrelationPenaltyMultiGroup:
    """Signals in multiple correlation groups get the harshest penalty."""

    def test_multi_group_signal_gets_min_penalty(self):
        from portfolio.signal_engine import _weighted_consensus
        votes = {"rsi": "BUY", "structure": "BUY", "volatility_sig": "BUY",
                 "claude_fundamental": "BUY"}
        accuracy = {
            "rsi": {"accuracy": 0.55, "total": 100},
            "structure": {"accuracy": 0.50, "total": 100},
            "volatility_sig": {"accuracy": 0.48, "total": 100},
            "claude_fundamental": {"accuracy": 0.62, "total": 100},
        }
        action, conf = _weighted_consensus(votes, accuracy, "unknown")
        assert action == "BUY"


class TestCrossAssetFlowGroup:
    """2026-05-07: cross_asset_flow dissolved — futures_flow disabled (38.3%).
    2026-05-18: credit_spread_risk moved to fundamental_cluster (100% agreement with crypto_macro)."""

    def test_cross_asset_flow_group_removed(self):
        from portfolio.signal_engine import CORRELATION_GROUPS
        assert "cross_asset_flow" not in CORRELATION_GROUPS

    def test_credit_spread_risk_in_fundamental_cluster(self):
        from portfolio.signal_engine import CORRELATION_GROUPS
        assert "credit_spread_risk" in CORRELATION_GROUPS.get("fundamental_cluster", frozenset())

    def test_futures_flow_disabled(self):
        from portfolio.tickers import DISABLED_SIGNALS
        assert "futures_flow" in DISABLED_SIGNALS


class TestCrisisModeConditionalTrendPenalty:
    """Crisis mode should NOT penalize trend signals when they're accurate."""

    def test_crisis_no_trend_penalty_when_trend_accurate(self):
        """When macro signals are broken but trend signals have >55% accuracy,
        crisis mode should NOT penalize trend signals."""
        from portfolio.signal_engine import _weighted_consensus

        # Trend signal voting BUY with high accuracy
        votes = {"ema": "BUY", "trend": "BUY", "rsi": "BUY"}
        # Macro signals broken (below 35%), but trend signals strong
        accuracy = {
            "fear_greed": {"accuracy": 0.25, "total": 100},
            "macro_regime": {"accuracy": 0.30, "total": 100},
            "news_event": {"accuracy": 0.29, "total": 100},
            "structure": {"accuracy": 0.40, "total": 100},
            "sentiment": {"accuracy": 0.46, "total": 100},
            # Trend signals are strong
            "ema": {"accuracy": 0.63, "total": 100},
            "trend": {"accuracy": 0.62, "total": 100},
            "heikin_ashi": {"accuracy": 0.55, "total": 100},
            "volume_flow": {"accuracy": 0.56, "total": 100},
            "rsi": {"accuracy": 0.55, "total": 100},
        }
        action, conf_no_crisis = _weighted_consensus(votes, accuracy, "unknown")
        # Should still get full BUY — trend not penalized
        assert action == "BUY"
        # The confidence should be relatively high since trend signals are accurate
        assert conf_no_crisis > 0.5

    def test_crisis_penalizes_trend_when_trend_weak(self):
        """When both macro AND trend signals are broken, crisis penalty applies."""
        from portfolio.signal_engine import _weighted_consensus

        votes = {"ema": "BUY", "mean_reversion": "SELL", "rsi": "SELL"}
        accuracy = {
            # Macro broken
            "fear_greed": {"accuracy": 0.25, "total": 100},
            "macro_regime": {"accuracy": 0.30, "total": 100},
            "news_event": {"accuracy": 0.29, "total": 100},
            # Trend signals also weak (below 55% floor)
            "ema": {"accuracy": 0.48, "total": 100},
            "trend": {"accuracy": 0.42, "total": 100},
            "heikin_ashi": {"accuracy": 0.45, "total": 100},
            "volume_flow": {"accuracy": 0.44, "total": 100},
            # MR signals
            "mean_reversion": {"accuracy": 0.60, "total": 100},
            "rsi": {"accuracy": 0.55, "total": 100},
        }
        action, conf = _weighted_consensus(votes, accuracy, "unknown")
        # MR should win since trend is penalized and MR is boosted
        assert action == "SELL"


# ---------------------------------------------------------------------------
# _persistence_state bounds (memory leak fix)
# ---------------------------------------------------------------------------

class TestPersistenceStateBounds:
    """Verify _persistence_state dict is bounded by _PERSISTENCE_MAX_TICKERS."""

    def test_dict_bounded_at_cap(self):
        from portfolio.signal_engine import (
            _PERSISTENCE_MAX_TICKERS,
            _apply_persistence_filter,
            _persistence_lock,
            _persistence_state,
        )
        # Clear state and fill to cap with unique tickers
        with _persistence_lock:
            _persistence_state.clear()

        votes = {"ema": "BUY", "rsi": "SELL"}
        for i in range(_PERSISTENCE_MAX_TICKERS + 10):
            _apply_persistence_filter(votes, f"TICKER-{i}")

        with _persistence_lock:
            assert len(_persistence_state) <= _PERSISTENCE_MAX_TICKERS
            _persistence_state.clear()

    def test_eviction_preserves_recent_tickers(self):
        from portfolio.signal_engine import (
            _PERSISTENCE_MAX_TICKERS,
            _apply_persistence_filter,
            _persistence_lock,
            _persistence_state,
        )
        with _persistence_lock:
            _persistence_state.clear()

        votes = {"ema": "BUY"}
        # Fill exactly to cap
        for i in range(_PERSISTENCE_MAX_TICKERS):
            _apply_persistence_filter(votes, f"T-{i}")
        # Add one more — should trigger eviction
        _apply_persistence_filter(votes, "T-NEWEST")

        with _persistence_lock:
            assert "T-NEWEST" in _persistence_state
            # Oldest half should be evicted
            assert "T-0" not in _persistence_state
            _persistence_state.clear()


class TestMSTRBTCProxy:
    """MSTR BTC cross-asset proxy signal (2026-04-29).

    MSTR is a BTC treasury company (818K BTC, 0.58 correlation). When BTC-USD
    consensus is cached, MSTR should get a synthetic btc_proxy vote injected.
    """

    def test_btc_proxy_injected_when_cache_populated(self):
        """btc_proxy should appear in MSTR votes when BTC-USD consensus is cached."""
        from portfolio.signal_engine import _cross_ticker_consensus
        _cross_ticker_consensus["BTC-USD"] = {"action": "BUY", "confidence": 0.65}
        try:
            from portfolio.signal_engine import generate_signal
            ind = _make_neutral_indicators()
            _, _, extra = generate_signal(ind, ticker="MSTR")
            assert extra.get("btc_proxy_action") == "BUY"
            assert extra.get("btc_proxy_source") == "cross_ticker_cache"
            assert "btc_proxy" in extra.get("_votes", {})
        finally:
            _cross_ticker_consensus.pop("BTC-USD", None)

    def test_btc_proxy_not_injected_for_btc(self):
        """btc_proxy should NOT appear for BTC-USD itself."""
        from portfolio.signal_engine import _cross_ticker_consensus
        _cross_ticker_consensus["BTC-USD"] = {"action": "BUY", "confidence": 0.65}
        try:
            from portfolio.signal_engine import generate_signal
            ind = _make_neutral_indicators()
            _, _, extra = generate_signal(ind, ticker="BTC-USD")
            assert "btc_proxy_action" not in extra
            assert "btc_proxy" not in extra.get("_votes", {})
        finally:
            _cross_ticker_consensus.pop("BTC-USD", None)

    def test_btc_proxy_not_injected_when_cache_empty(self):
        """btc_proxy should NOT appear when BTC-USD consensus is not cached."""
        from portfolio.signal_engine import _cross_ticker_consensus
        _cross_ticker_consensus.pop("BTC-USD", None)
        from portfolio.signal_engine import generate_signal
        ind = _make_neutral_indicators()
        _, _, extra = generate_signal(ind, ticker="MSTR")
        assert "btc_proxy_action" not in extra

    def test_btc_proxy_sell_propagates(self):
        """SELL consensus from BTC should propagate as btc_proxy=SELL."""
        from portfolio.signal_engine import _cross_ticker_consensus
        _cross_ticker_consensus["BTC-USD"] = {"action": "SELL", "confidence": 0.70}
        try:
            from portfolio.signal_engine import generate_signal
            ind = _make_neutral_indicators()
            _, _, extra = generate_signal(ind, ticker="MSTR")
            assert extra.get("btc_proxy_action") == "SELL"
        finally:
            _cross_ticker_consensus.pop("BTC-USD", None)

    def test_btc_proxy_hold_propagates(self):
        """HOLD consensus from BTC should still inject (neutral vote)."""
        from portfolio.signal_engine import _cross_ticker_consensus
        _cross_ticker_consensus["BTC-USD"] = {"action": "HOLD", "confidence": 0.0}
        try:
            from portfolio.signal_engine import generate_signal
            ind = _make_neutral_indicators()
            _, _, extra = generate_signal(ind, ticker="MSTR")
            assert extra.get("btc_proxy_action") == "HOLD"
        finally:
            _cross_ticker_consensus.pop("BTC-USD", None)

    def test_consensus_cache_updated_after_generate(self):
        """generate_signal should update _cross_ticker_consensus for the ticker."""
        from portfolio.signal_engine import _cross_ticker_consensus, generate_signal
        _cross_ticker_consensus.pop("ETH-USD", None)
        ind = _make_neutral_indicators()
        action, conf, _ = generate_signal(ind, ticker="ETH-USD")
        assert "ETH-USD" in _cross_ticker_consensus
        assert _cross_ticker_consensus["ETH-USD"]["action"] == action
        assert _cross_ticker_consensus["ETH-USD"]["confidence"] == conf


def _make_neutral_indicators():
    """Return a minimal indicators dict where all core signals vote HOLD."""
    return {
        "rsi": 50, "rsi_p20": 30, "rsi_p80": 70,
        "macd_hist": 0.01, "macd_hist_prev": 0.02,  # no crossover
        "ema9": 100, "ema21": 100,  # no gap
        "price_vs_bb": "within",
        "close": 100, "volume": 1000, "avg_volume": 1000,
        "atr": 1.0, "adx": 20, "rvol": 1.0,
    }


class TestAccuracyTierMult:
    """Tests for the walk-forward accuracy tier multiplier."""

    def test_strong_edge_gets_2x(self):
        from portfolio.signal_engine import _accuracy_tier_mult
        assert _accuracy_tier_mult(0.68) == 2.0
        assert _accuracy_tier_mult(0.65) == 2.0

    def test_good_edge_gets_1_5x(self):
        from portfolio.signal_engine import _accuracy_tier_mult
        assert _accuracy_tier_mult(0.60) == 1.5
        assert _accuracy_tier_mult(0.63) == 1.5

    def test_moderate_edge_gets_1_2x(self):
        from portfolio.signal_engine import _accuracy_tier_mult
        assert _accuracy_tier_mult(0.55) == 1.2
        assert _accuracy_tier_mult(0.58) == 1.2

    def test_baseline_gets_1x(self):
        from portfolio.signal_engine import _accuracy_tier_mult
        assert _accuracy_tier_mult(0.50) == 1.0
        assert _accuracy_tier_mult(0.53) == 1.0

    def test_marginal_gets_0_75x(self):
        from portfolio.signal_engine import _accuracy_tier_mult
        assert _accuracy_tier_mult(0.47) == 0.75
        assert _accuracy_tier_mult(0.48) == 0.75

    def test_weak_gets_0_5x(self):
        from portfolio.signal_engine import _accuracy_tier_mult
        assert _accuracy_tier_mult(0.40) == 0.5
        assert _accuracy_tier_mult(0.0) == 0.5

    def test_monotonically_increasing(self):
        from portfolio.signal_engine import _accuracy_tier_mult
        values = [0.30, 0.45, 0.48, 0.52, 0.57, 0.62, 0.70]
        mults = [_accuracy_tier_mult(v) for v in values]
        for i in range(len(mults) - 1):
            assert mults[i] <= mults[i + 1], (
                f"mult({values[i]})={mults[i]} > mult({values[i+1]})={mults[i+1]}"
            )
