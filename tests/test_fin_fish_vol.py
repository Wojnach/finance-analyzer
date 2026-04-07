"""Tests for fin_fish volatility fix — correct annualization of daily range data.

Bug: _compute_vol_and_drift fed daily-range sigma into volatility_from_atr()
which uses sqrt(252/14), correct for hourly ATR but 3.7x too low for daily data.
Fix: separate annualization path for daily ranges using sqrt(252).
"""

from __future__ import annotations

import math
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, ".")

from portfolio.fin_fish import (
    _compute_vol_and_drift,
    compute_fishing_levels_bull,
    evaluate_warrants,
)
from portfolio.monte_carlo import MIN_VOLATILITY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_signal(atr_pct=0.55, rsi=46.0, p_up_3h=0.52):
    """Minimal signal dict for testing."""
    return {
        "entry": {
            "extra": {
                "forecast_indicators": {},
            },
        },
        "focus": {
            "3h": {"probability": p_up_3h},
        },
        "atr_pct": atr_pct,
        "rsi": rsi,
        "regime": "ranging",
        "action": "HOLD",
        "weighted_confidence": 0.5,
        "price_usd": 72.0,
    }


def _make_daily_ranges(range_pcts=None):
    """Minimal daily ranges for testing."""
    if range_pcts is None:
        range_pcts = [0.2, 0.1, 2.4, 2.7, 2.5]
    result = []
    spot = 72.0
    for i, rp in enumerate(range_pcts):
        high = spot * (1 + rp / 200)
        low = spot * (1 - rp / 200)
        result.append({
            "date": f"04-{3 + i:02d}",
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(spot, 2),
            "range_pct": rp,
        })
    return result


# ---------------------------------------------------------------------------
# Fix 1: Vol annualization
# ---------------------------------------------------------------------------

class TestVolAnnualization:
    """_compute_vol_and_drift must annualize daily ranges correctly."""

    def test_daily_range_gives_reasonable_annual_vol(self):
        """Daily ranges of ~2.5% should yield annual vol ~25-30%, not ~7%."""
        signal = _make_signal(atr_pct=0.55)
        daily_ranges = _make_daily_ranges([2.4, 2.7, 2.5])

        vol, drift = _compute_vol_and_drift(signal, daily_ranges, "LONG")

        # With daily range avg=2.53%, daily sigma ~ 1.69%
        # Annual vol = 1.69% * sqrt(252) ~ 26.8%
        assert vol > 0.20, f"Annual vol {vol:.3f} too low (expected >0.20)"
        assert vol < 0.50, f"Annual vol {vol:.3f} too high (expected <0.50)"

    def test_low_atr_overridden_by_daily_range(self):
        """When hourly ATR is low but daily ranges are wide, daily wins."""
        signal = _make_signal(atr_pct=0.3)  # tiny hourly ATR
        daily_ranges = _make_daily_ranges([3.0, 3.5, 2.8])

        vol, _ = _compute_vol_and_drift(signal, daily_ranges, "LONG")

        # Daily range avg=3.1%, sigma=2.07%, annual=32.8%
        assert vol > 0.25, f"Daily range should dominate: got {vol:.3f}"

    def test_no_daily_ranges_falls_back_to_atr(self):
        """With no daily ranges, uses hourly ATR annualization."""
        signal = _make_signal(atr_pct=2.0)
        vol, _ = _compute_vol_and_drift(signal, [], "LONG")

        # hourly ATR 2% -> volatility_from_atr -> atr_frac * sqrt(252/14)
        expected = 0.02 * math.sqrt(252 / 14)
        assert abs(vol - expected) < 0.01

    def test_narrow_daily_ranges_ignored(self):
        """Daily ranges < 0.5% are filtered, so atr path used instead."""
        signal = _make_signal(atr_pct=0.55)
        daily_ranges = _make_daily_ranges([0.1, 0.2, 0.3])

        vol, _ = _compute_vol_and_drift(signal, daily_ranges, "LONG")
        # All ranges < 0.5%, so daily path skipped, hourly ATR used
        # vol = 0.0055 * sqrt(18) ~ 0.023
        assert vol < 0.10, f"Should use ATR path: got {vol:.3f}"

    def test_vol_floor_respected(self):
        """Returned vol is never below MIN_VOLATILITY."""
        signal = _make_signal(atr_pct=0.01)
        vol, _ = _compute_vol_and_drift(signal, [], "LONG")
        assert vol >= MIN_VOLATILITY


# ---------------------------------------------------------------------------
# Fix 1 integration: multiple levels pass filter
# ---------------------------------------------------------------------------

class TestFishingLevelsWithFixedVol:
    """With correct vol, 5+ fishing levels should pass the 5% fill threshold."""

    def test_multiple_bull_levels_pass(self):
        """At least 4 BULL levels should have fill_prob >= 5% over 13h."""
        signal = _make_signal(atr_pct=0.55)
        daily_ranges = _make_daily_ranges([2.4, 2.7, 2.5])
        spot = 72.30

        levels = compute_fishing_levels_bull("XAG-USD", spot, signal, 13.0, daily_ranges)

        passing = [l for l in levels if l["fill_prob"] >= 0.05]
        debug = [(l["level"], round(l["fill_prob"], 3)) for l in levels[:8]]
        assert len(passing) >= 4, (
            f"Only {len(passing)} levels pass 5% filter, expected >=4. Levels: {debug}"
        )

    def test_4day_floor_has_fill_probability(self):
        """A dip to the 4-day low ($71.26) should have measurable fill prob."""
        signal = _make_signal(atr_pct=0.55)
        daily_ranges = _make_daily_ranges([0.2, 0.1, 2.4, 2.7, 2.5])
        # Daily low on 04-05 with range 2.4%: low ~ 72*(1-2.4/200) = 71.14
        spot = 72.30

        levels = compute_fishing_levels_bull("XAG-USD", spot, signal, 13.0, daily_ranges)

        # Find the level closest to the daily low
        deep_levels = [l for l in levels if l["dip_pct"] >= 1.0]
        assert len(deep_levels) >= 1, "No levels at 1%+ dip"
        assert any(l["fill_prob"] > 0.01 for l in deep_levels), (
            f"Deep levels all have zero fill prob: {[(l['level'], l['fill_prob']) for l in deep_levels]}"
        )


# ---------------------------------------------------------------------------
# Fix 2: One warrant per level in output
# ---------------------------------------------------------------------------

class TestBestWarrantPerLevel:
    """evaluate_warrants should return distinct price levels."""

    def test_distinct_levels_in_results(self):
        """Top results should have multiple distinct price levels."""
        signal = _make_signal(atr_pct=0.55)
        daily_ranges = _make_daily_ranges([2.4, 2.7, 2.5])
        spot = 72.30

        levels = compute_fishing_levels_bull("XAG-USD", spot, signal, 13.0, daily_ranges)
        results = evaluate_warrants("XAG-USD", spot, levels, 20000, 9.5, direction="LONG")

        if len(results) < 2:
            pytest.skip("Not enough warrant results to test dedup")

        # Check that the top 6 results aren't all the same level
        top_levels = [r["level"] for r in results[:6]]
        unique = len(set(round(l, 1) for l in top_levels))
        assert unique >= 2, (
            f"Top 6 results have only {unique} unique level(s): {top_levels}"
        )


# ---------------------------------------------------------------------------
# Fix 3: Dynamic leverage
# ---------------------------------------------------------------------------

class TestDynamicLeverage:
    """Warrants with barriers should have leverage computed from spot."""

    def test_turbo_leverage_computed_from_spot(self):
        """TURBO L with barrier $68 at spot $72.30 should be ~16.8x, not 9.8x."""
        signal = _make_signal(atr_pct=0.55)
        daily_ranges = _make_daily_ranges([2.4, 2.7, 2.5])
        spot = 72.30

        levels = compute_fishing_levels_bull("XAG-USD", spot, signal, 13.0, daily_ranges)
        results = evaluate_warrants("XAG-USD", spot, levels, 20000, 9.5, direction="LONG")

        turbo_results = [r for r in results if "TURBO" in r["warrant"]]
        if not turbo_results:
            pytest.skip("No TURBO warrants in results")

        for r in turbo_results:
            barrier = r["barrier"]
            if barrier > 0:
                expected_lev = spot / (spot - barrier)
                assert abs(r["leverage"] - expected_lev) < 1.0, (
                    f"{r['warrant']}: leverage={r['leverage']:.1f} but "
                    f"expected ~{expected_lev:.1f} from spot={spot}, barrier={barrier}"
                )

    def test_daily_cert_keeps_config_leverage(self):
        """Daily certificates (barrier=0) should keep their config leverage."""
        signal = _make_signal(atr_pct=0.55)
        daily_ranges = _make_daily_ranges([2.4, 2.7, 2.5])
        spot = 72.30

        levels = compute_fishing_levels_bull("XAG-USD", spot, signal, 13.0, daily_ranges)
        results = evaluate_warrants("XAG-USD", spot, levels, 20000, 9.5, direction="LONG")

        cert_results = [r for r in results if r.get("is_daily_cert")]
        if not cert_results:
            pytest.skip("No daily certs in results")

        for r in cert_results:
            # Daily certs should have leverage 3x or 5x (from config)
            assert r["leverage"] in (3.0, 5.0), (
                f"{r['warrant']}: daily cert leverage={r['leverage']}, expected 3.0 or 5.0"
            )
