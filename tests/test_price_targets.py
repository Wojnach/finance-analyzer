"""Tests for portfolio.price_targets -- optimal buy/sell price targets."""



from portfolio.price_targets import (
    _apply_regime_adjustment,
    compute_targets,
    expected_value,
    fill_probability,
    fill_probability_buy,
    running_extremes,
    structural_levels,
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
    """Tests for structural_levels() including enriched signal data extraction."""

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

    def test_handles_both_none(self):
        """Both indicators and extra None -> empty."""
        result = structural_levels(85.0, None, None)
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

    # --- Fibonacci levels ---
    def test_extracts_fibonacci_levels(self):
        extra = {
            "fibonacci_indicators": {
                "fib_levels": {
                    "0.236": 84.0,
                    "0.382": 83.0,
                    "0.5": 82.0,
                    "0.618": 81.0,
                    "0.786": 80.0,
                },
            },
        }
        result = structural_levels(85.0, None, extra)
        assert result["fib_236"] == 84.0
        assert result["fib_382"] == 83.0
        assert result["fib_5"] == 82.0
        assert result["fib_618"] == 81.0
        assert result["fib_786"] == 80.0

    def test_extracts_pivot_levels(self):
        extra = {
            "fibonacci_indicators": {
                "pivot": 85.0,
                "r1": 86.0,
                "r2": 87.0,
                "s1": 84.0,
                "s2": 83.0,
            },
        }
        result = structural_levels(85.0, None, extra)
        assert result["pivot_pp"] == 85.0
        assert result["pivot_r1"] == 86.0
        assert result["pivot_r2"] == 87.0
        assert result["pivot_s1"] == 84.0
        assert result["pivot_s2"] == 83.0

    def test_extracts_camarilla_pivots(self):
        extra = {
            "fibonacci_indicators": {
                "cam_r3": 86.5,
                "cam_s3": 83.5,
                "cam_r4": 88.0,
                "cam_s4": 82.0,
            },
        }
        result = structural_levels(85.0, None, extra)
        assert result["cam_r3"] == 86.5
        assert result["cam_s3"] == 83.5
        assert result["cam_r4"] == 88.0
        assert result["cam_s4"] == 82.0

    def test_extracts_golden_pocket(self):
        extra = {
            "fibonacci_indicators": {
                "gp_upper": 81.5,
                "gp_lower": 80.8,
            },
        }
        result = structural_levels(85.0, None, extra)
        assert result["gp_upper"] == 81.5
        assert result["gp_lower"] == 80.8

    def test_extracts_fibonacci_swings(self):
        extra = {
            "fibonacci_indicators": {
                "swing_high": 90.0,
                "swing_low": 78.0,
            },
        }
        result = structural_levels(85.0, None, extra)
        assert result["fib_swing_high"] == 90.0
        assert result["fib_swing_low"] == 78.0

    # --- Volatility levels ---
    def test_extracts_keltner_channels(self):
        extra = {
            "volatility_sig_indicators": {
                "keltner_upper": 87.0,
                "keltner_lower": 83.0,
            },
        }
        result = structural_levels(85.0, None, extra)
        assert result["keltner_upper"] == 87.0
        assert result["keltner_lower"] == 83.0

    def test_extracts_donchian_channels(self):
        extra = {
            "volatility_sig_indicators": {
                "donchian_upper": 89.0,
                "donchian_lower": 81.0,
            },
        }
        result = structural_levels(85.0, None, extra)
        assert result["donchian_upper"] == 89.0
        assert result["donchian_lower"] == 81.0

    # --- Volume flow ---
    def test_extracts_vwap(self):
        extra = {
            "volume_flow_indicators": {
                "vwap": 85.5,
            },
        }
        result = structural_levels(85.0, None, extra)
        assert result["vwap"] == 85.5

    # --- Smart money ---
    def test_extracts_smart_money_swings(self):
        extra = {
            "smart_money_indicators": {
                "last_swing_high": 88.0,
                "last_swing_low": 82.0,
            },
        }
        result = structural_levels(85.0, None, extra)
        assert result["smc_swing_high"] == 88.0
        assert result["smc_swing_low"] == 82.0

    # --- Edge cases ---
    def test_nan_values_skipped(self):
        """NaN values in any indicator source should be skipped."""
        extra = {
            "fibonacci_indicators": {
                "fib_levels": {"0.5": float("nan")},
                "pivot": float("nan"),
                "swing_high": float("nan"),
            },
            "volatility_sig_indicators": {
                "keltner_upper": float("nan"),
            },
            "volume_flow_indicators": {
                "vwap": float("nan"),
            },
            "smart_money_indicators": {
                "last_swing_high": float("nan"),
            },
        }
        result = structural_levels(85.0, None, extra)
        assert result == {}

    def test_zero_values_skipped(self):
        """Zero price levels should be skipped."""
        extra = {
            "fibonacci_indicators": {
                "pivot": 0.0,
                "r1": 0,
            },
            "volatility_sig_indicators": {
                "keltner_upper": 0,
                "donchian_upper": 0.0,
            },
            "volume_flow_indicators": {
                "vwap": 0.0,
            },
        }
        result = structural_levels(85.0, None, extra)
        assert result == {}

    def test_combined_bb_and_extra(self):
        """BB indicators + extra dict should both contribute levels."""
        ind = {"bb_mid": 85.0, "bb_upper": 87.0}
        extra = {
            "fibonacci_indicators": {"pivot": 85.5, "fib_levels": {"0.5": 82.0}},
            "volume_flow_indicators": {"vwap": 84.8},
        }
        result = structural_levels(85.0, ind, extra)
        assert "bb_mid" in result
        assert "bb_upper" in result
        assert "pivot_pp" in result
        assert "fib_5" in result
        assert "vwap" in result

    def test_empty_extra_dicts_handled(self):
        """Empty sub-dicts in extra should not crash."""
        extra = {
            "fibonacci_indicators": {},
            "volatility_sig_indicators": {},
            "volume_flow_indicators": {},
            "smart_money_indicators": {},
        }
        result = structural_levels(85.0, None, extra)
        assert result == {}


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


class TestRegimeAdjustment:
    """Tests for _apply_regime_adjustment()."""

    def _make_targets(self, prices, fill_prob=0.5, side="sell", base_price=85.0):
        targets = []
        for p in prices:
            targets.append({
                "price": p,
                "fill_prob": fill_prob,
                "ev_sek": 0.0,
                "label": "test",
            })
        return targets

    def test_ranging_penalizes_far_targets(self):
        """Ranging regime should penalize targets >1% from price."""
        targets = self._make_targets([86.5])  # ~1.8% from 85.0
        _apply_regime_adjustment(targets, "ranging", "sell", 85.0, None)
        assert targets[0]["fill_prob"] == round(0.5 * 0.85, 4)

    def test_ranging_does_not_penalize_near_targets(self):
        """Ranging regime should not penalize targets <1% from price."""
        targets = self._make_targets([85.5])  # ~0.6% from 85.0
        _apply_regime_adjustment(targets, "ranging", "sell", 85.0, None)
        assert targets[0]["fill_prob"] == 0.5  # unchanged

    def test_ranging_boosts_near_bb_mid(self):
        """Ranging regime should boost targets near BB mid."""
        targets = self._make_targets([85.0])  # right at bb_mid
        _apply_regime_adjustment(targets, "range-bound", "sell", 84.5, 85.0)
        assert targets[0]["fill_prob"] > 0.5

    def test_trending_up_boosts_sell(self):
        """Trending-up should boost sell targets above price."""
        targets = self._make_targets([86.0])
        _apply_regime_adjustment(targets, "trending-up", "sell", 85.0, None)
        assert targets[0]["fill_prob"] == round(0.5 * 1.10, 4)

    def test_trending_up_penalizes_buy(self):
        """Trending-up should penalize buy targets below price."""
        targets = self._make_targets([84.0])
        _apply_regime_adjustment(targets, "trending-up", "buy", 85.0, None)
        assert targets[0]["fill_prob"] == round(0.5 * 0.90, 4)

    def test_trending_down_boosts_buy(self):
        """Trending-down should boost buy targets below price."""
        targets = self._make_targets([84.0])
        _apply_regime_adjustment(targets, "trending-down", "buy", 85.0, None)
        assert targets[0]["fill_prob"] == round(0.5 * 1.10, 4)

    def test_trending_down_penalizes_sell(self):
        """Trending-down should penalize sell targets above price."""
        targets = self._make_targets([86.0])
        _apply_regime_adjustment(targets, "trending-down", "sell", 85.0, None)
        assert targets[0]["fill_prob"] == round(0.5 * 0.90, 4)

    def test_empty_regime_no_change(self):
        """Empty regime string should not modify targets."""
        targets = self._make_targets([86.0])
        _apply_regime_adjustment(targets, "", "sell", 85.0, None)
        assert targets[0]["fill_prob"] == 0.5

    def test_unknown_regime_no_change(self):
        """Unknown regime string should not modify targets."""
        targets = self._make_targets([86.0])
        _apply_regime_adjustment(targets, "breakout", "sell", 85.0, None)
        assert targets[0]["fill_prob"] == 0.5

    def test_fill_prob_capped_at_1(self):
        """Regime boost should never push fill_prob above 1.0."""
        targets = self._make_targets([86.0], fill_prob=0.98)
        _apply_regime_adjustment(targets, "trending-up", "sell", 85.0, None)
        assert targets[0]["fill_prob"] <= 1.0


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

    def test_with_extra_fibonacci_levels(self):
        """Extra dict with fibonacci levels should produce structural targets."""
        extra = {
            "fibonacci_indicators": {
                "r1": 85.50,
                "r2": 86.00,
                "pivot": 85.10,
            },
        }
        result = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                                 atr_pct=2.5, p_up=0.50, hours_remaining=6.0,
                                 extra=extra)
        labels = [t["label"] for t in result["targets"]]
        assert any("pivot" in l for l in labels)

    def test_with_extra_keltner(self):
        """Keltner channel levels should appear as targets when reachable."""
        extra = {
            "volatility_sig_indicators": {
                # Level within p90 range so fill_prob > min_fill (0.05)
                "keltner_upper": 85.15,
            },
        }
        result = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                                 atr_pct=2.5, p_up=0.50, hours_remaining=6.0,
                                 extra=extra)
        labels = [t["label"] for t in result["targets"]]
        assert any("keltner" in l for l in labels)

    def test_with_extra_vwap_buy(self):
        """VWAP below price should appear as buy target."""
        extra = {
            "volume_flow_indicators": {
                "vwap": 84.50,
            },
        }
        result = compute_targets("XAG-USD", side="buy", price_usd=85.0,
                                 atr_pct=2.5, p_up=0.50, hours_remaining=6.0,
                                 extra=extra)
        labels = [t["label"] for t in result["targets"]]
        assert any("vwap" in l for l in labels)

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

    # --- Chronos drift blending ---
    def test_chronos_drift_positive_boosts_sell(self):
        """Positive Chronos drift should help sell targets (more upside expected)."""
        r_no = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                               atr_pct=1.5, p_up=0.50, hours_remaining=6.0)
        # Strongly positive drift: 5% annualized from Chronos
        r_chrono = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                                   atr_pct=1.5, p_up=0.50, hours_remaining=6.0,
                                   chronos_drift=5.0)
        # With positive drift, recommended sell target should have higher EV
        if r_no["recommended"] and r_chrono["recommended"]:
            assert r_chrono["recommended"]["ev_sek"] >= r_no["recommended"]["ev_sek"] - 0.01

    def test_chronos_drift_none_has_no_effect(self):
        """chronos_drift=None should behave identically to the base case."""
        r_base = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                                 atr_pct=1.5, p_up=0.50, hours_remaining=6.0)
        r_none = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                                 atr_pct=1.5, p_up=0.50, hours_remaining=6.0,
                                 chronos_drift=None)
        # Should produce identical extremes and targets
        assert r_base["extremes"] == r_none["extremes"]

    # --- BB squeeze warning ---
    def test_squeeze_warning_flag(self):
        """bb_squeeze=True should set squeeze_warning in result."""
        result = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                                 atr_pct=1.5, p_up=0.50, hours_remaining=6.0,
                                 bb_squeeze=True)
        assert result.get("squeeze_warning") is True

    def test_squeeze_reduces_fill_probs(self):
        """BB squeeze should reduce fill probabilities by 0.7x."""
        r_no = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                               atr_pct=1.5, p_up=0.50, hours_remaining=6.0,
                               bb_squeeze=False)
        r_sq = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                               atr_pct=1.5, p_up=0.50, hours_remaining=6.0,
                               bb_squeeze=True)
        # Each matching target in squeeze result should have lower fill_prob
        if r_no["targets"] and r_sq["targets"]:
            # Compare first targets (same label since same inputs)
            no_labels = {t["label"]: t for t in r_no["targets"]}
            for t_sq in r_sq["targets"]:
                if t_sq["label"] in no_labels:
                    # fill_prob should be ~0.7x of the no-squeeze version
                    expected = round(no_labels[t_sq["label"]]["fill_prob"] * 0.7, 4)
                    assert abs(t_sq["fill_prob"] - expected) < 0.01

    def test_squeeze_false_no_warning(self):
        """bb_squeeze=False should not set squeeze_warning."""
        result = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                                 atr_pct=1.5, p_up=0.50, hours_remaining=6.0,
                                 bb_squeeze=False)
        assert "squeeze_warning" not in result

    # --- Regime in compute_targets ---
    def test_regime_trending_up_affects_targets(self):
        """Trending-up regime should boost sell fill probs vs no regime."""
        r_no = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                               atr_pct=1.5, p_up=0.50, hours_remaining=6.0,
                               regime="")
        r_up = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                               atr_pct=1.5, p_up=0.50, hours_remaining=6.0,
                               regime="trending-up")
        # At least one sell target should have boosted fill_prob
        if r_no["targets"] and r_up["targets"]:
            no_labels = {t["label"]: t["fill_prob"] for t in r_no["targets"]}
            boosted = False
            for t in r_up["targets"]:
                if t["label"] in no_labels and t["fill_prob"] > no_labels[t["label"]]:
                    boosted = True
                    break
            assert boosted

    def test_regime_empty_string_no_effect(self):
        """Empty regime string should not change results."""
        r_base = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                                 atr_pct=1.5, p_up=0.50, hours_remaining=6.0)
        r_empty = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                                  atr_pct=1.5, p_up=0.50, hours_remaining=6.0,
                                  regime="")
        # Should produce identical targets
        assert len(r_base["targets"]) == len(r_empty["targets"])
        for a, b in zip(r_base["targets"], r_empty["targets"]):
            assert a["fill_prob"] == b["fill_prob"]

    def test_targets_still_sorted_after_regime(self):
        """After regime adjustment, targets must still be sorted by EV desc."""
        result = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                                 atr_pct=1.5, p_up=0.50, hours_remaining=6.0,
                                 regime="trending-up")
        evs = [t["ev_sek"] for t in result["targets"]]
        assert evs == sorted(evs, reverse=True)

    def test_targets_sorted_after_squeeze(self):
        """After squeeze adjustment, targets must still be sorted by EV desc."""
        result = compute_targets("XAG-USD", side="sell", price_usd=85.0,
                                 atr_pct=1.5, p_up=0.50, hours_remaining=6.0,
                                 bb_squeeze=True)
        evs = [t["ev_sek"] for t in result["targets"]]
        assert evs == sorted(evs, reverse=True)
