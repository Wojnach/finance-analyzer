"""Tests for portfolio.grid_tiers — pure tier math, no I/O."""

from __future__ import annotations

import pytest

from portfolio.grid_tiers import (
    MIN_LEG_SEK,
    Tier,
    active_tiers,
    build_buy_ladder,
    build_exit_levels,
    total_planned_notional,
)


# ---------------------------------------------------------------------------
# build_buy_ladder
# ---------------------------------------------------------------------------


class TestBuyLadderBasics:
    def test_three_tiers_default_spacing(self):
        tiers = build_buy_ladder(bid=100.0, leg_sek=1200, n_tiers=3,
                                 spacing_pct=(0.3, 0.8, 1.5))
        assert len(tiers) == 3
        # Tier 0 at -0.3%, tier 1 at -0.8%, tier 2 at -1.5%
        assert tiers[0].price == pytest.approx(99.70)
        assert tiers[1].price == pytest.approx(99.20)
        assert tiers[2].price == pytest.approx(98.50)

    def test_qty_clears_min_order_at_high_price(self):
        tiers = build_buy_ladder(bid=100.0, leg_sek=1200, n_tiers=1,
                                 spacing_pct=(0.5,))
        t = tiers[0]
        # 1200 / 99.50 = 12 (floor), notional = 12 * 99.50 = 1194 — clears 1000
        assert t.qty == 12
        assert t.is_active
        assert t.notional_sek == pytest.approx(12 * t.price)

    def test_low_priced_warrant_qty_scales_up(self):
        tiers = build_buy_ladder(bid=9.20, leg_sek=1200, n_tiers=1,
                                 spacing_pct=(0.5,))
        t = tiers[0]
        # 1200 / 9.1540 ~ 131
        assert t.qty == int(1200 // t.price)
        assert t.notional_sek >= MIN_LEG_SEK
        assert t.is_active


class TestBuyLadderMinOrder:
    def test_below_min_order_marked_skipped(self):
        # leg_sek too low to clear MIN_LEG_SEK at any price
        tiers = build_buy_ladder(bid=100.0, leg_sek=500, n_tiers=1,
                                 spacing_pct=(0.5,))
        t = tiers[0]
        assert not t.is_active
        assert "below_min_order" in (t.skip_reason or "")

    def test_high_priced_warrant_with_low_qty_still_passes_min(self):
        # Very high-priced warrant — qty 1, notional ~ price
        tiers = build_buy_ladder(bid=2000.0, leg_sek=1200, n_tiers=1,
                                 spacing_pct=(0.5,))
        t = tiers[0]
        # 1200 / 1990 = 0 (floor) -> below_min_order
        assert t.qty == 0
        assert not t.is_active


class TestBuyLadderValidation:
    def test_rejects_zero_bid(self):
        with pytest.raises(ValueError, match="bid"):
            build_buy_ladder(bid=0.0)

    def test_rejects_negative_bid(self):
        with pytest.raises(ValueError, match="bid"):
            build_buy_ladder(bid=-1.0)

    def test_rejects_nan_bid(self):
        with pytest.raises(ValueError, match="bid"):
            build_buy_ladder(bid=float("nan"))

    def test_rejects_zero_tiers(self):
        with pytest.raises(ValueError, match="n_tiers"):
            build_buy_ladder(bid=100.0, n_tiers=0)

    def test_rejects_short_spacing_array(self):
        with pytest.raises(ValueError, match="spacing"):
            build_buy_ladder(bid=100.0, n_tiers=3, spacing_pct=(0.3, 0.8))

    def test_rejects_non_positive_spacing(self):
        with pytest.raises(ValueError, match="spacing_pct"):
            build_buy_ladder(bid=100.0, n_tiers=2, spacing_pct=(0.3, 0.0))

    def test_rejects_invalid_direction(self):
        with pytest.raises(ValueError, match="direction"):
            build_buy_ladder(bid=100.0, direction="FLAT")


class TestKnockoutSafety:
    def test_long_tier_skipped_when_near_barrier(self):
        # LONG warrant: barrier below current. Tier 1.5% below bid implies
        # underlying drop = 1.5% / 5x = 0.3%. With underlying at 75 and
        # barrier at 74.5 (0.67% away), that's within the 8% safety band.
        tiers = build_buy_ladder(
            bid=100.0,
            leg_sek=1200,
            n_tiers=3,
            spacing_pct=(0.3, 0.8, 1.5),
            direction="LONG",
            underlying_price=75.0,
            barrier=74.5,
            leverage=5.0,
        )
        # All tiers should hit knockout skip — barrier is already too close
        for t in tiers:
            assert not t.is_active, f"tier {t.index} unexpectedly active"
            assert "knockout" in (t.skip_reason or "")

    def test_long_tier_active_when_barrier_far(self):
        tiers = build_buy_ladder(
            bid=100.0,
            leg_sek=1200,
            n_tiers=1,
            spacing_pct=(0.5,),
            direction="LONG",
            underlying_price=75.0,
            barrier=50.0,  # ~33% away — well clear
            leverage=5.0,
        )
        assert tiers[0].is_active

    def test_short_tier_uses_opposite_barrier_logic(self):
        # SHORT warrant: barrier above current. Tier 1.5% below warrant bid
        # means underlying rose 0.3%, moving *toward* barrier.
        tiers = build_buy_ladder(
            bid=100.0,
            leg_sek=1200,
            n_tiers=1,
            spacing_pct=(0.5,),
            direction="SHORT",
            underlying_price=75.0,
            barrier=75.5,  # 0.67% above — within safety band
            leverage=5.0,
        )
        assert not tiers[0].is_active
        assert "knockout" in (tiers[0].skip_reason or "")

    def test_no_barrier_skips_knockout_check(self):
        tiers = build_buy_ladder(
            bid=100.0,
            leg_sek=1200,
            n_tiers=1,
            spacing_pct=(0.5,),
            direction="LONG",
            underlying_price=75.0,
            barrier=None,
            leverage=5.0,
        )
        assert tiers[0].is_active


class TestHelperFunctions:
    def test_total_planned_notional_sums_only_active(self):
        tiers = [
            Tier(index=0, price=10.0, qty=120, notional_sek=1200.0, spacing_pct=0.3),
            Tier(index=1, price=10.0, qty=0, notional_sek=0.0, spacing_pct=0.8,
                 skip_reason="below_min_order:0sek"),
            Tier(index=2, price=10.0, qty=120, notional_sek=1200.0, spacing_pct=1.5),
        ]
        assert total_planned_notional(tiers) == pytest.approx(2400.0)

    def test_active_tiers_filters_skipped(self):
        tiers = [
            Tier(index=0, price=10.0, qty=120, notional_sek=1200.0, spacing_pct=0.3),
            Tier(index=1, price=10.0, qty=0, notional_sek=0.0, spacing_pct=0.8,
                 skip_reason="skipped"),
        ]
        assert len(active_tiers(tiers)) == 1
        assert active_tiers(tiers)[0].index == 0


# ---------------------------------------------------------------------------
# build_exit_levels
# ---------------------------------------------------------------------------


class TestExitLevels:
    def test_basic_target_and_stop(self):
        sell, stop = build_exit_levels(fill_price=100.0, target_pct=1.2,
                                       stop_pct=3.5)
        assert sell == pytest.approx(101.20)
        assert stop == pytest.approx(96.50)

    def test_rounded_to_ore(self):
        sell, stop = build_exit_levels(fill_price=42.137, target_pct=1.2,
                                       stop_pct=3.5)
        assert sell == round(42.137 * 1.012, 2)
        assert stop == round(42.137 * 0.965, 2)

    def test_rejects_non_positive_fill(self):
        with pytest.raises(ValueError, match="fill_price"):
            build_exit_levels(fill_price=0.0, target_pct=1.2, stop_pct=3.5)

    def test_rejects_non_positive_target(self):
        with pytest.raises(ValueError, match="target_pct"):
            build_exit_levels(fill_price=100.0, target_pct=0.0, stop_pct=3.5)

    def test_rejects_non_positive_stop(self):
        with pytest.raises(ValueError, match="stop_pct"):
            build_exit_levels(fill_price=100.0, target_pct=1.2, stop_pct=0.0)
