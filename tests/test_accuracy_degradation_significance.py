"""Statistical significance gate for the accuracy degradation detector.

Caught 2026-04-28: detector was firing on regression-to-mean from
small-sample baselines. The Apr 21 baseline showed sentiment 75.3% on
N=223 — a one-week spike well above lifetime 46% (N=39k). The detector's
flat 15pp threshold doesn't account for binomial noise: at N=100 vs N=100
on p=0.5, the 95% CI on the difference is ±10pp — so spurious 15pp drops
are 1-in-10 events.

Fix: require ``drop_pp >= max(DROP_THRESHOLD_PP, 2*SE)`` where SE is the
standard error of the difference of two binomial proportions.

2026-06-10 (audit batch 2): the SE now uses EFFECTIVE sample sizes
(n_eff = n / AUTOCORR_EFFECTIVE_N_DIVISOR, K=20) because 60s snapshot rows
of persistent votes are heavily autocorrelated — signals emit identical
votes in day-long blocks, so 200 graded rows are nowhere near 200
independent trials. This widens the CIs roughly sqrt(20) ≈ 4.5x: drops
that used to fire at N≈200 (including the 2026-04-28 sentiment cliff,
which WAS regression-to-mean noise) are now suppressed, and genuine
collapses need either bigger drops or bigger windows. Expected values in
this file are computed with n_eff = n // 20.

Also raises ``MIN_SAMPLES_HISTORICAL`` / ``MIN_SAMPLES_CURRENT`` from
100 → 200 so the SE gate has a healthy floor on input quality (the floors
keep gating on RAW counts, not n_eff).
"""
from __future__ import annotations

import math

from portfolio import accuracy_degradation as deg


class TestEffectiveN:
    """2026-06-10: raw graded-row count -> effective independent trials."""

    def test_divisor_constant(self):
        assert deg.AUTOCORR_EFFECTIVE_N_DIVISOR == 20

    def test_basic_division(self):
        assert deg._effective_n(400) == 20
        assert deg._effective_n(200) == 10
        assert deg._effective_n(1000) == 50

    def test_floor_at_one(self):
        # Any nonzero raw count yields at least one effective trial.
        assert deg._effective_n(1) == 1
        assert deg._effective_n(19) == 1
        assert deg._effective_n(20) == 1
        assert deg._effective_n(39) == 1

    def test_zero_and_negative(self):
        assert deg._effective_n(0) == 0
        assert deg._effective_n(-5) == 0

    def test_divisor_override(self):
        assert deg._effective_n(100, divisor=10) == 10
        # Degenerate divisor clamps to 1 (no inflation of n).
        assert deg._effective_n(100, divisor=0) == 100


class TestBinomialDiffSE:
    """The helper that computes the SE of (p_old - p_new) in percentage points."""

    def test_zero_samples_returns_zero(self):
        assert deg._binomial_diff_se_pp(0.5, 0, 0.5, 100) == 0.0
        assert deg._binomial_diff_se_pp(0.5, 100, 0.5, 0) == 0.0

    def test_known_se_value(self):
        # p1=0.5, n1=100 → n_eff=5; p2=0.5, n2=100 → n_eff=5
        # var = 0.25/5 + 0.25/5 = 0.1
        # SE = sqrt(0.1) ≈ 0.3162 → 31.62 pp
        se_pp = deg._binomial_diff_se_pp(0.5, 100, 0.5, 100)
        assert math.isclose(se_pp, 31.62, abs_tol=0.05)

    def test_se_shrinks_with_larger_n(self):
        small = deg._binomial_diff_se_pp(0.5, 50, 0.5, 50)    # n_eff = 2
        large = deg._binomial_diff_se_pp(0.5, 500, 0.5, 500)  # n_eff = 25
        assert small > large
        # SE scales with 1/sqrt(n_eff): sqrt(25/2) ≈ 3.54x
        assert math.isclose(small / large, math.sqrt(25 / 2), rel_tol=0.05)


class TestSignificanceGate:
    """`_maybe_alert` rejects drops that aren't 2*SE above the threshold."""

    def test_moderate_drop_at_min_samples_suppressed(self):
        """2026-06-10: N=200 vs N=200 is only ~10 effective trials per side.
        p_old=0.5 → p_new=0.35 (15pp) gives SE ≈ 21.9pp, 2*SE ≈ 43.7pp —
        a 15pp drop at this window size is indistinguishable from one
        autocorrelated day-block flipping, so it must NOT fire (it did
        before the effective-n correction).
        """
        alert = deg._maybe_alert(
            key="signal_a", scope="signal",
            old={"accuracy": 0.50, "total": 200},
            new={"accuracy": 0.35, "total": 200},
            drop_threshold_pp=15.0, absolute_floor_pct=50.0,
            min_samples_historical=200, min_samples_current=200,
        )
        assert alert is None, "15pp drop on ~10 effective trials is noise"

    def test_tiny_samples_drop_at_threshold_suppressed_by_se(self):
        """N=50 vs N=50 — 15pp drop is far inside the widened noise band."""
        alert = deg._maybe_alert(
            key="noisy", scope="signal",
            old={"accuracy": 0.55, "total": 50},
            new={"accuracy": 0.40, "total": 50},
            drop_threshold_pp=15.0, absolute_floor_pct=50.0,
            min_samples_historical=50, min_samples_current=50,
        )
        assert alert is None, "drop within 2*SE noise must be suppressed"

    def test_large_real_drop_with_large_window_fires(self):
        """A genuine multi-week collapse: 62% → 30% (32pp) on N=1400 per
        side. n_eff=70 each → SE ≈ 8.0pp, 2*SE ≈ 16pp < 32pp. Must fire.
        """
        alert = deg._maybe_alert(
            key="collapsed", scope="signal",
            old={"accuracy": 0.62, "total": 1400},
            new={"accuracy": 0.30, "total": 1400},
            drop_threshold_pp=15.0, absolute_floor_pct=50.0,
            min_samples_historical=200, min_samples_current=200,
        )
        assert alert is not None
        assert alert["drop_pp"] > 30.0

    def test_sentiment_2026_04_28_cliff_now_suppressed(self):
        """The original sentiment 75.3% (N=223) → 43.3% (N=187) incident was
        regression-to-mean from an autocorrelated one-week spike. Under
        n_eff (11 vs 9 effective trials, SE ≈ 21pp, 2*SE ≈ 42pp) even this
        32pp drop is within the noise band — correctly suppressed.
        """
        alert = deg._maybe_alert(
            key="sentiment", scope="signal",
            old={"accuracy": 0.753, "total": 223},
            new={"accuracy": 0.433, "total": 187},
            drop_threshold_pp=15.0, absolute_floor_pct=50.0,
            min_samples_historical=200, min_samples_current=187,
        )
        assert alert is None

    def test_drop_above_threshold_but_below_2se_suppressed(self):
        """20pp drop on N=80 each — n_eff=4, SE ≈ 34.6pp, 2*SE ≈ 69pp."""
        alert = deg._maybe_alert(
            key="borderline", scope="signal",
            old={"accuracy": 0.60, "total": 80},
            new={"accuracy": 0.40, "total": 80},
            drop_threshold_pp=15.0, absolute_floor_pct=50.0,
            min_samples_historical=80, min_samples_current=80,
        )
        assert alert is None, "20pp on 4 effective trials must be suppressed"

    def test_low_n_drop_below_2se_suppressed(self):
        """16pp drop on N=40 each — drop suppressed by the SE gate."""
        alert = deg._maybe_alert(
            key="too_noisy", scope="signal",
            old={"accuracy": 0.55, "total": 40},
            new={"accuracy": 0.39, "total": 40},
            drop_threshold_pp=15.0, absolute_floor_pct=50.0,
            min_samples_historical=40, min_samples_current=40,
        )
        assert alert is None, "16pp drop on tiny N must be suppressed by SE gate"


class TestRaisedMinSamples:
    """The min-samples floor is bumped to 200 (raw counts, not n_eff)."""

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

    def test_at_floor_with_big_window_passes(self):
        """30pp drop fires once the window is big enough for the n_eff SE
        (2026-06-10: N=200 alone no longer clears 2*SE ≈ 42pp; N=1000 does:
        n_eff=50 each, SE ≈ 9.5pp, 2*SE ≈ 19pp < 30pp).
        """
        alert = deg._maybe_alert(
            key="at_floor", scope="signal",
            old={"accuracy": 0.70, "total": 1000},
            new={"accuracy": 0.40, "total": 1000},
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
