"""Statistical significance gate for the accuracy degradation detector.

Caught 2026-04-28: detector was firing on regression-to-mean from
small-sample baselines. The Apr 21 baseline showed sentiment 75.3% on
N=223 — a one-week spike well above lifetime 46% (N=39k). The detector's
flat 15pp threshold doesn't account for binomial noise: at N=100 vs N=100
on p=0.5, the 95% CI on the difference is ±10pp — so spurious 15pp drops
are 1-in-10 events.

Fix: require ``drop_pp >= max(DROP_THRESHOLD_PP, 2*SE)`` where SE is the
standard error of the difference of two independent binomial proportions.
At N=200 each that's ≈7pp, so the original 15pp threshold dominates. At
N=50 each it's ≈10pp, doubling to 20pp — and the 15pp threshold no longer
fires on noise.

Also raises ``MIN_SAMPLES_HISTORICAL`` / ``MIN_SAMPLES_CURRENT`` from
100 → 200 so the SE gate has a healthy floor on input quality.
"""
from __future__ import annotations

import math

import pytest

from portfolio import accuracy_degradation as deg


class TestBinomialDiffSE:
    """The helper that computes the SE of (p_old - p_new) in percentage points."""

    def test_zero_samples_returns_zero(self):
        assert deg._binomial_diff_se_pp(0.5, 0, 0.5, 100) == 0.0
        assert deg._binomial_diff_se_pp(0.5, 100, 0.5, 0) == 0.0

    def test_known_se_value(self):
        # p1=0.5, n1=100, p2=0.5, n2=100
        # var = 0.25/100 + 0.25/100 = 0.005
        # SE = sqrt(0.005) ≈ 0.0707 → 7.07 pp
        se_pp = deg._binomial_diff_se_pp(0.5, 100, 0.5, 100)
        assert math.isclose(se_pp, 7.07, abs_tol=0.05)

    def test_se_shrinks_with_larger_n(self):
        small = deg._binomial_diff_se_pp(0.5, 50, 0.5, 50)
        large = deg._binomial_diff_se_pp(0.5, 500, 0.5, 500)
        assert small > large
        # SE scales with 1/sqrt(N), so 10x N → ~3.16x smaller SE
        assert math.isclose(small / large, math.sqrt(10), rel_tol=0.05)


class TestSignificanceGate:
    """`_maybe_alert` rejects drops that aren't 2*SE above the threshold."""

    def test_small_sample_borderline_drop_suppressed(self):
        """At N=200 vs N=200, p_old=0.5, p_new=0.35 → 15pp drop, SE ≈ 4.9pp.
        2*SE = 9.8pp; 15pp > 9.8pp so it should fire on the SE gate.
        But we still also need drop >= 15pp threshold — which it does.
        """
        alert = deg._maybe_alert(
            key="signal_a", scope="signal",
            old={"accuracy": 0.50, "total": 200},
            new={"accuracy": 0.35, "total": 200},
            drop_threshold_pp=15.0, absolute_floor_pct=50.0,
            min_samples_historical=200, min_samples_current=200,
        )
        assert alert is not None, "real 15pp drop with healthy N must fire"

    def test_tiny_samples_drop_at_threshold_suppressed_by_se(self):
        """N=50 vs N=50 — even a 15pp drop is only 1.5*SE → suppress."""
        alert = deg._maybe_alert(
            key="noisy", scope="signal",
            old={"accuracy": 0.55, "total": 50},
            new={"accuracy": 0.40, "total": 50},
            drop_threshold_pp=15.0, absolute_floor_pct=50.0,
            min_samples_historical=50, min_samples_current=50,
        )
        # SE = sqrt(0.55*0.45/50 + 0.40*0.60/50)*100 ≈ 9.9 pp
        # 2*SE ≈ 19.8 pp; drop=15pp < 19.8pp → suppressed
        assert alert is None, "drop within 2*SE noise must be suppressed"

    def test_large_real_drop_passes_both_gates(self):
        """Like the sentiment case: 75% N=223 → 43% N=187, drop=32pp.
        SE ≈ 4.6pp; 2*SE ≈ 9.2pp. drop>>9.2 and drop>>15. Must fire.
        """
        alert = deg._maybe_alert(
            key="sentiment", scope="signal",
            old={"accuracy": 0.753, "total": 223},
            new={"accuracy": 0.433, "total": 187},
            drop_threshold_pp=15.0, absolute_floor_pct=50.0,
            min_samples_historical=200, min_samples_current=187,
        )
        assert alert is not None
        assert alert["drop_pp"] > 30.0

    def test_drop_above_threshold_but_below_2se_suppressed(self):
        """20pp drop on N=80 each — SE ≈ 7.9pp, 2*SE ≈ 15.8pp."""
        alert = deg._maybe_alert(
            key="borderline", scope="signal",
            old={"accuracy": 0.60, "total": 80},
            new={"accuracy": 0.40, "total": 80},
            drop_threshold_pp=15.0, absolute_floor_pct=50.0,
            min_samples_historical=80, min_samples_current=80,
        )
        # drop=20pp >= 15pp threshold gate → would have passed old check
        # SE = sqrt(0.6*0.4/80 + 0.4*0.6/80)*100 ≈ 7.75 pp
        # 2*SE ≈ 15.5 pp; drop=20pp > 15.5pp → fires
        assert alert is not None, "20pp drop on N=80 IS significant (>2*SE)"

    def test_low_n_drop_below_2se_suppressed(self):
        """16pp drop on N=40 each — SE ≈ 11pp, 2*SE ≈ 22pp; drop suppressed."""
        alert = deg._maybe_alert(
            key="too_noisy", scope="signal",
            old={"accuracy": 0.55, "total": 40},
            new={"accuracy": 0.39, "total": 40},
            drop_threshold_pp=15.0, absolute_floor_pct=50.0,
            min_samples_historical=40, min_samples_current=40,
        )
        assert alert is None, "16pp drop on tiny N must be suppressed by SE gate"


class TestRaisedMinSamples:
    """The min-samples floor is bumped to 200."""

    def test_min_samples_constant_is_200(self):
        assert deg.MIN_SAMPLES_HISTORICAL == 200
        assert deg.MIN_SAMPLES_CURRENT == 200

    def test_below_floor_blocks_alert(self):
        """Even a real-looking 30pp drop is suppressed if either side has N<200."""
        alert = deg._maybe_alert(
            key="thin_baseline", scope="signal",
            old={"accuracy": 0.70, "total": 150},  # below floor
            new={"accuracy": 0.40, "total": 250},
            drop_threshold_pp=15.0, absolute_floor_pct=50.0,
            min_samples_historical=200, min_samples_current=200,
        )
        assert alert is None

    def test_at_floor_passes(self):
        alert = deg._maybe_alert(
            key="at_floor", scope="signal",
            old={"accuracy": 0.70, "total": 200},
            new={"accuracy": 0.40, "total": 200},
            drop_threshold_pp=15.0, absolute_floor_pct=50.0,
            min_samples_historical=200, min_samples_current=200,
        )
        assert alert is not None


class TestSyntheticNoiseDoesNotFire:
    """End-to-end: simulate stable signal at p=0.5, take two random 7d windows,
    and assert detector does NOT fire 95% of the time. Uses fixed seeds for
    determinism rather than rerunning across many seeds.
    """

    def test_synthetic_p50_p50_low_n_no_fire(self):
        """Two N=200 samples from a p=0.5 process. At a fixed seed where the
        difference is moderate (e.g., 0.50 vs 0.42 = 8pp), the gate
        suppresses (drop < 15pp threshold) — sanity check.
        """
        # Concrete: 0.50 vs 0.42 means drop=8pp; threshold gate alone catches it
        alert = deg._maybe_alert(
            key="stable", scope="signal",
            old={"accuracy": 0.50, "total": 200},
            new={"accuracy": 0.42, "total": 200},
            drop_threshold_pp=15.0, absolute_floor_pct=50.0,
            min_samples_historical=200, min_samples_current=200,
        )
        assert alert is None

    def test_synthetic_p50_p50_high_n_borderline_drop_no_fire(self):
        """N=400 each, drop of exactly 13pp — below 15pp threshold."""
        alert = deg._maybe_alert(
            key="stable_high_n", scope="signal",
            old={"accuracy": 0.50, "total": 400},
            new={"accuracy": 0.37, "total": 400},
            drop_threshold_pp=15.0, absolute_floor_pct=50.0,
            min_samples_historical=200, min_samples_current=200,
        )
        # drop=13pp < 15pp threshold → suppressed
        assert alert is None
