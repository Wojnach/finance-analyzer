"""Tests for portfolio.price_targets -- optimal buy/sell price targets."""

import math

import pytest

from portfolio.price_targets import (
    fill_probability,
    fill_probability_buy,
    running_extremes,
    structural_levels,
    expected_value,
    compute_targets,
)


class TestFillProbability:
    def test_target_below_price_sell(self):
        """Selling below current price = guaranteed fill."""
        p = fill_probability(100, 95, 0.20, 0.05, 3.0)
        assert p >= 0.99

    def test_target_at_price(self):
        """Target at current price = guaranteed fill."""
        p = fill_probability(100, 100, 0.20, 0.05, 3.0)
        assert p >= 0.99

    def test_target_far_above(self):
        """Target far above current price = very low fill prob."""
        p = fill_probability(100, 200, 0.20, 0.0, 1.0)
        assert p < 0.01

    def test_monotonic_in_target(self):
        """Higher target = lower fill probability."""
        p1 = fill_probability(100, 101, 0.20, 0.0, 3.0)
        p2 = fill_probability(100, 103, 0.20, 0.0, 3.0)
        p3 = fill_probability(100, 106, 0.20, 0.0, 3.0)
        assert p1 > p2 > p3

    def test_more_time_higher_prob(self):
        """More time remaining = higher fill probability."""
        p1 = fill_probability(100, 105, 0.20, 0.0, 1.0)
        p2 = fill_probability(100, 105, 0.20, 0.0, 6.0)
        p3 = fill_probability(100, 105, 0.20, 0.0, 24.0)
        assert p1 < p2 < p3

    def test_higher_vol_higher_prob(self):
        """Higher volatility = higher fill probability."""
        p1 = fill_probability(100, 105, 0.10, 0.0, 3.0)
        p2 = fill_probability(100, 105, 0.40, 0.0, 3.0)
        assert p1 < p2

    def test_positive_drift_helps_sell(self):
        """Positive drift increases probability of reaching higher targets."""
        p_no_drift = fill_probability(100, 105, 0.20, 0.0, 3.0)
        p_pos_drift = fill_probability(100, 105, 0.20, 0.50, 3.0)
        assert p_pos_drift > p_no_drift

    def test_result_bounded(self):
        """Fill probability always in [0, 1]."""
        for target in [50, 90, 100, 110, 200]:
            p = fill_probability(100, target, 0.20, 0.05, 3.0)
            assert 0.0 <= p <= 1.0

    def test_zero_hours(self):
        """Zero hours remaining: target above price -> 0, at/below -> 1."""
        assert fill_probability(100, 110, 0.20, 0.0, 0.0) == 0.0
        assert fill_probability(100, 90, 0.20, 0.0, 0.0) == 1.0

    def test_zero_price(self):
        """Zero price should not crash."""
        p = fill_probability(0, 100, 0.20, 0.0, 3.0)
        assert 0.0 <= p <= 1.0

    def test_extreme_volatility(self):
        """Very high volatility should give high fill prob for nearby targets."""
        p = fill_probability(100, 101, 2.0, 0.0, 3.0)
        assert p > 0.5


class TestFillProbabilityBuy:
    def test_target_above_price(self):
        """Buy target above current price = guaranteed fill."""
        p = fill_probability_buy(100, 110, 0.20, 0.0, 3.0)
        assert p >= 0.99

    def test_target_far_below(self):
        """Buy target far below = very low fill prob."""
        p = fill_probability_buy(100, 50, 0.20, 0.0, 1.0)
        assert p < 0.05

    def test_negative_drift_helps_buy(self):
        """Negative drift (price dropping) helps buy targets below price."""
        p_no = fill_probability_buy(100, 95, 0.20, 0.0, 3.0)
        p_neg = fill_probability_buy(100, 95, 0.20, -0.50, 3.0)
        assert p_neg > p_no


class TestRunningExtremes:
    def test_sell_max_above_spot(self):
        """Running max should be >= spot for all quantiles."""
        result = running_extremes(100, 0.20, 0.0, 3.0, side="sell", n_paths=5000)
        for key in ("p10", "p25", "p50", "p75", "p90"):
            assert result[key] >= 100.0

    def test_buy_min_below_spot(self):
        """Running min should be <= spot for all quantiles."""
        result = running_extremes(100, 0.20, 0.0, 3.0, side="buy", n_paths=5000)
        for key in ("p10", "p25", "p50", "p75", "p90"):
            assert result[key] <= 100.0

    def test_higher_vol_wider_spread(self):
        """Higher vol = wider spread between p10 and p90."""
        r_low = running_extremes(100, 0.10, 0.0, 3.0, side="sell", n_paths=5000)
        r_high = running_extremes(100, 0.40, 0.0, 3.0, side="sell", n_paths=5000)
        spread_low = r_low["p90"] - r_low["p10"]
        spread_high = r_high["p90"] - r_high["p10"]
        assert spread_high > spread_low

    def test_returns_correct_keys(self):
        result = running_extremes(100, 0.20, 0.0, 3.0)
        assert set(result.keys()) == {"p10", "p25", "p50", "p75", "p90"}

    def test_zero_hours(self):
        """Zero hours -> all quantiles at spot."""
        result = running_extremes(100, 0.20, 0.0, 0.0, side="sell")
        for key in ("p10", "p25", "p50", "p75", "p90"):
            assert result[key] == 100.0

    def test_more_time_wider_spread(self):
        """More time -> wider spread."""
        r_short = running_extremes(100, 0.20, 0.0, 1.0, side="sell", n_paths=5000)
        r_long = running_extremes(100, 0.20, 0.0, 24.0, side="sell", n_paths=5000)
        assert r_long["p90"] > r_short["p90"]

    def test_quantile_ordering_sell(self):
        """p10 <= p25 <= p50 <= p75 <= p90 for sell (running max)."""
        r = running_extremes(100, 0.20, 0.0, 6.0, side="sell", n_paths=5000)
        assert r["p10"] <= r["p25"] <= r["p50"] <= r["p75"] <= r["p90"]

    def test_quantile_ordering_buy(self):
        """p10 <= p25 <= p50 <= p75 <= p90 for buy (running min)."""
        r = running_extremes(100, 0.20, 0.0, 6.0, side="buy", n_paths=5000)
        assert r["p10"] <= r["p25"] <= r["p50"] <= r["p75"] <= r["p90"]


class TestStructuralLevels:
    def test_extracts_bb_levels(self):
        ind = {"bb_mid": 85.0, "bb_upper": 87.0, "bb_lower": 83.0}
        result = structural_levels(85.0, ind)
        assert "bb_mid" in result
        assert "bb_upper" in result
        assert "bb_lower" in result

    def test_handles_missing_indicators(self):
        result = structural_levels(85.0, {})
        assert result == {}

    def test_handles_none_indicators(self):
        result = structural_levels(85.0, None)
        assert result == {}

    def test_partial_indicators(self):
        ind = {"bb_mid": 85.0}
        result = structural_levels(85.0, ind)
        assert "bb_mid" in result
        assert "bb_upper" not in result

    def test_non_numeric_ignored(self):
        ind = {"bb_mid": "not_a_number", "bb_upper": 87.0}
        result = structural_levels(85.0, ind)
        assert "bb_mid" not in result
        assert "bb_upper" in result


class TestExpectedValue:
    def test_certain_fill(self):
        ev = expected_value(1.0, 100, 50)
        assert ev == 100

    def test_zero_fill(self):
        ev = expected_value(0.0, 100, 50)
        assert ev == 50

    def test_half_probability(self):
        ev = expected_value(0.5, 100, 0)
        assert ev == 50.0

    def test_negative_fallback(self):
        ev = expected_value(0.5, 200, -100)
        assert ev == 50.0

    def test_negative_gain(self):
        ev = expected_value(0.5, -100, 0)
        assert ev == -50.0


class TestComputeTargets:
    def test_sell_returns_structure(self):
        result = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                                 atr_pct=0.59, p_up=0.45, hours_remaining=3.0)
        assert result["ticker"] == "XAG-USD"
        assert result["side"] == "sell"
        assert "targets" in result
        assert "extremes" in result
        assert "recommended" in result or result["recommended"] is None

    def test_buy_returns_structure(self):
        result = compute_targets("BTC-USD", side="buy", price_usd=85000,
                                 atr_pct=2.5, p_up=0.65, hours_remaining=6.0)
        assert result["side"] == "buy"
        assert len(result["targets"]) > 0

    def test_targets_sorted_by_ev(self):
        result = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                                 atr_pct=0.59, p_up=0.50, hours_remaining=3.0)
        evs = [t["ev_sek"] for t in result["targets"]]
        assert evs == sorted(evs, reverse=True)

    def test_sell_targets_above_price(self):
        result = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                                 atr_pct=0.59, p_up=0.50, hours_remaining=3.0)
        for t in result["targets"]:
            assert t["price"] >= 85.0

    def test_buy_targets_below_price(self):
        result = compute_targets("BTC-USD", side="buy", price_usd=85000,
                                 atr_pct=2.5, p_up=0.50, hours_remaining=3.0)
        for t in result["targets"]:
            assert t["price"] <= 85000

    def test_fill_prob_monotonic_for_sell(self):
        """Targets further from price should have lower fill probability."""
        result = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                                 atr_pct=0.59, p_up=0.50, hours_remaining=3.0)
        targets = sorted(result["targets"], key=lambda t: t["price"])
        probs = [t["fill_prob"] for t in targets]
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1] - 0.01  # small tolerance

    def test_with_indicators(self):
        # Use higher ATR and closer BB levels so BB targets are reachable
        ind = {"bb_mid": 85.05, "bb_upper": 85.12, "bb_lower": 84.88}
        result = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                                 atr_pct=2.5, p_up=0.50, hours_remaining=6.0,
                                 indicators=ind)
        labels = [t["label"] for t in result["targets"]]
        assert any("bb" in l.lower() for l in labels)

    def test_with_warrant_leverage(self):
        r1 = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                             atr_pct=0.59, p_up=0.50, hours_remaining=3.0,
                             warrant_leverage=1.0, position_units=1105, fx_rate=9.2)
        r5 = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                             atr_pct=0.59, p_up=0.50, hours_remaining=3.0,
                             warrant_leverage=5.0, position_units=1105, fx_rate=9.2)
        # Leveraged EV should be ~5x the unleveraged for same targets
        if r1["targets"] and r5["targets"]:
            ev1 = r1["targets"][0]["ev_sek"]
            ev5 = r5["targets"][0]["ev_sek"]
            if ev1 > 0:
                assert 3.0 < ev5 / ev1 < 7.0  # roughly 5x with some tolerance

    def test_zero_hours_returns_empty_targets(self):
        result = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                                 atr_pct=0.59, p_up=0.50, hours_remaining=0.0)
        assert result["targets"] == []

    def test_zero_atr_returns_empty(self):
        result = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                                 atr_pct=0.0, p_up=0.50, hours_remaining=3.0)
        assert result["targets"] == []

    def test_zero_price_returns_empty(self):
        result = compute_targets("XAG-USD", side="sell", price_usd=0.0,
                                 atr_pct=0.59, p_up=0.50, hours_remaining=3.0)
        assert result["targets"] == []

    def test_recommended_is_best_ev(self):
        result = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                                 atr_pct=0.59, p_up=0.50, hours_remaining=3.0)
        if result["targets"]:
            assert result["recommended"] == result["targets"][0]

    def test_stock_uses_correct_time_fraction(self):
        """Non-24h asset uses stock trading hours."""
        result = compute_targets("NVDA", side="sell", price_usd=185.0,
                                 atr_pct=1.5, p_up=0.50, hours_remaining=3.0,
                                 is_24h=False)
        assert result["ticker"] == "NVDA"
        assert len(result["targets"]) > 0

    def test_each_target_has_required_keys(self):
        result = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                                 atr_pct=0.59, p_up=0.50, hours_remaining=3.0)
        for t in result["targets"]:
            assert "price" in t
            assert "fill_prob" in t
            assert "ev_sek" in t
            assert "label" in t
