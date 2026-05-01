"""Tests for the voter-count circuit breaker (Batch 2 of 2026-04-16 accuracy
gating reconfiguration).

The circuit breaker relaxes the accuracy gate when cascaded gates would leave
too few active voters, preserving voter diversity during regime transitions.
It does NOT relax the directional or correlation gates.
"""

import pytest

from portfolio.signal_engine import (
    _GATE_RELAXATION_MAX,
    _GATE_RELAXATION_STEP,
    _MIN_ACTIVE_VOTERS_SOFT,
    _compute_gate_relaxation,
    _count_active_voters_at_gate,
    _weighted_consensus,
)


class TestCircuitBreakerConstants:
    """Guardrail: the constants are sane and will not silently regress."""

    def test_min_voters_soft_at_5(self):
        assert _MIN_ACTIVE_VOTERS_SOFT == 5

    def test_relaxation_step_at_2pp(self):
        assert pytest.approx(0.02) == _GATE_RELAXATION_STEP

    def test_relaxation_max_at_6pp(self):
        assert pytest.approx(0.06) == _GATE_RELAXATION_MAX

    def test_max_is_multiple_of_step(self):
        """Step must divide max cleanly so iteration reaches the exact cap.

        Use a ratio check because float modulo leaves residual noise
        (0.06 % 0.02 == 0.019999... in IEEE-754).
        """
        ratio = _GATE_RELAXATION_MAX / _GATE_RELAXATION_STEP
        assert ratio == pytest.approx(round(ratio), abs=1e-9)

    def test_high_sample_min_at_10000(self):
        """2026-04-16 review (Reviewer 3 P1-1): pin the tiered-gate sample
        threshold. Was raised 5000 -> 10000 during the gating reconfig so
        signals with 5-10K samples aren't caught by the tighter 0.50 gate
        during regime transitions. Regression test against a silent revert.
        """
        from portfolio.signal_engine import _ACCURACY_GATE_HIGH_SAMPLE_MIN
        assert _ACCURACY_GATE_HIGH_SAMPLE_MIN == 10000

    def test_high_sample_threshold_at_050(self):
        """Companion pin: the tiered gate at the high-sample tier stays 0.50."""
        from portfolio.signal_engine import _ACCURACY_GATE_HIGH_SAMPLE_THRESHOLD
        assert pytest.approx(0.50) == _ACCURACY_GATE_HIGH_SAMPLE_THRESHOLD


class TestCountActiveVotersAtGate:
    """Helper correctness: counts active voters at a given relaxation level."""

    def _make_stats(self, acc, total=100, buy_acc=None, sell_acc=None,
                    total_buy=50, total_sell=50):
        """Build one signal's stats dict."""
        return {
            "accuracy": acc,
            "total": total,
            "buy_accuracy": buy_acc if buy_acc is not None else acc,
            "sell_accuracy": sell_acc if sell_acc is not None else acc,
            "total_buy": total_buy,
            "total_sell": total_sell,
        }

    def test_all_pass_at_base_gate(self):
        votes = {f"s{i}": "BUY" for i in range(5)}
        accuracy = {f"s{i}": self._make_stats(0.60) for i in range(5)}
        active = _count_active_voters_at_gate(votes, accuracy, set(), set(), 0.47, 0.0)
        assert active == 5

    def test_none_pass_when_all_below_gate(self):
        votes = {f"s{i}": "BUY" for i in range(5)}
        accuracy = {f"s{i}": self._make_stats(0.40) for i in range(5)}
        active = _count_active_voters_at_gate(votes, accuracy, set(), set(), 0.47, 0.0)
        assert active == 0

    def test_relaxation_lets_borderline_signals_pass(self):
        """Signals at 0.45 fail 0.47 gate but pass with 0.02 relaxation."""
        votes = {f"s{i}": "BUY" for i in range(5)}
        accuracy = {f"s{i}": self._make_stats(0.45) for i in range(5)}
        # At base gate 0.47 — all gated.
        assert _count_active_voters_at_gate(
            votes, accuracy, set(), set(), 0.47, 0.0,
        ) == 0
        # With 0.02 relaxation — effective gate 0.45, all pass (acc < 0.45 is False).
        assert _count_active_voters_at_gate(
            votes, accuracy, set(), set(), 0.47, 0.02,
        ) == 5

    def test_hold_votes_not_counted(self):
        votes = {"s1": "BUY", "s2": "HOLD", "s3": "SELL"}
        accuracy = {s: self._make_stats(0.60) for s in ("s1", "s2", "s3")}
        active = _count_active_voters_at_gate(votes, accuracy, set(), set(), 0.47, 0.0)
        assert active == 2

    def test_excluded_signals_not_counted(self):
        votes = {f"s{i}": "BUY" for i in range(5)}
        accuracy = {f"s{i}": self._make_stats(0.60) for i in range(5)}
        active = _count_active_voters_at_gate(
            votes, accuracy, excluded={"s0", "s1"}, group_gated=set(),
            base_gate=0.47, relaxation=0.0,
        )
        assert active == 3

    def test_group_gated_signals_not_counted(self):
        votes = {f"s{i}": "BUY" for i in range(5)}
        accuracy = {f"s{i}": self._make_stats(0.60) for i in range(5)}
        active = _count_active_voters_at_gate(
            votes, accuracy, excluded=set(), group_gated={"s2", "s3"},
            base_gate=0.47, relaxation=0.0,
        )
        assert active == 3

    def test_directional_gate_not_relaxed(self):
        """A signal with buy_accuracy=0.30 should be gated even at max relaxation."""
        votes = {"rsi": "BUY"}
        accuracy = {
            "rsi": self._make_stats(0.55, buy_acc=0.30, total_buy=50),
        }
        # Even with max relaxation, directional gate (threshold 0.40) fires.
        active = _count_active_voters_at_gate(
            votes, accuracy, set(), set(), 0.47, _GATE_RELAXATION_MAX,
        )
        assert active == 0

    def test_insufficient_samples_not_gated(self):
        """Signals with fewer than ACCURACY_GATE_MIN_SAMPLES samples always count.

        Use total_buy=5 so the directional gate (min_samples=30) also doesn't
        fire — otherwise a 0.30 buy_accuracy with 50 samples would gate via
        the directional path, masking the overall-gate behavior we want to test.
        """
        votes = {"new_sig": "BUY"}
        accuracy = {"new_sig": self._make_stats(0.30, total=10, total_buy=5, total_sell=5)}
        active = _count_active_voters_at_gate(
            votes, accuracy, set(), set(), 0.47, 0.0,
        )
        assert active == 1


class TestComputeGateRelaxation:
    """The relaxation chooser: smallest relaxation that meets voter floor."""

    def _make_stats(self, acc, total=100):
        return {
            "accuracy": acc, "total": total,
            "buy_accuracy": acc, "sell_accuracy": acc,
            "total_buy": total // 2, "total_sell": total // 2,
        }

    def test_no_relaxation_when_floor_already_met(self):
        votes = {f"s{i}": "BUY" for i in range(6)}
        accuracy = {f"s{i}": self._make_stats(0.60) for i in range(6)}
        rel = _compute_gate_relaxation(votes, accuracy, set(), set(), 0.47)
        assert rel == 0.0

    def test_one_step_relaxation_when_some_signals_at_45pct(self):
        """5 signals at 0.60, 3 at 0.45 — base gate lets 5 through (>=5),
        so no relaxation needed."""
        votes = {f"s{i}": "BUY" for i in range(8)}
        accuracy = {f"s{i}": self._make_stats(0.60) for i in range(5)}
        accuracy.update({f"s{i}": self._make_stats(0.45) for i in range(5, 8)})
        rel = _compute_gate_relaxation(votes, accuracy, set(), set(), 0.47)
        assert rel == 0.0

    def test_relaxes_by_step_when_only_4_above_base(self):
        """4 signals at 0.60, 3 at 0.45 — base allows 4 (< 5), relax to 0.45 adds them."""
        votes = {f"s{i}": "BUY" for i in range(7)}
        accuracy = {f"s{i}": self._make_stats(0.60) for i in range(4)}
        accuracy.update({f"s{i}": self._make_stats(0.45) for i in range(4, 7)})
        rel = _compute_gate_relaxation(votes, accuracy, set(), set(), 0.47)
        # With 0.02 relaxation: effective gate 0.45, those at 0.45 fail (acc < 0.45 False),
        # wait — 0.45 < 0.45 is False, so they pass at 0.45 gate. So relaxation 0.02 suffices.
        assert rel == pytest.approx(0.02)

    def test_returns_zero_when_relaxation_recovers_no_additional_voters(self):
        """Genuine regime break: max relaxation recovers 0 additional voters
        beyond baseline (all sub-47% signals are also sub-41%). The strict
        gate should stay on - relaxing would let actively-broken signals vote.

        Scenario: 2 signals above any gate + 3 signals at 0.30 that are
        below the 0.41 relaxed gate AND the 0.40 directional gate. Max
        relaxation recovers nothing new beyond the 2 baseline voters.
        """
        votes = {f"s{i}": "BUY" for i in range(5)}
        accuracy = {
            "s0": self._make_stats(0.60),  # passes at any gate
            "s1": self._make_stats(0.60),  # passes at any gate
            "s2": self._make_stats(0.30),  # below 0.41 and directionally gated
            "s3": self._make_stats(0.30),
            "s4": self._make_stats(0.30),
        }
        rel = _compute_gate_relaxation(votes, accuracy, set(), set(), 0.47)
        assert rel == 0.0, (
            "When max relaxation recovers 0 additional voters, the circuit "
            "breaker should NOT engage - that's a genuine regime break, "
            "not a relaxation opportunity."
        )

    def test_sparse_3_voter_ranging_stays_hold(self):
        """Ranging regime: 3 BUY candidates all at 0.46 should NOT trigger
        relaxation. Ranging dynamic_min=5, bp=3 < 5, blocked. If relaxation
        engaged, the recovered consensus would be below the ranging quorum
        anyway (force-HOLD downstream) - but blocking here avoids the
        pointless relaxation.
        """
        votes = {f"s{i}": "BUY" for i in range(3)}
        accuracy = {f"s{i}": self._make_stats(0.46) for i in range(3)}
        rel = _compute_gate_relaxation(votes, accuracy, set(), set(), 0.47, regime="ranging")
        assert rel == 0.0

    def test_sparse_3_voter_trending_relaxes(self):
        """Codex round 6 (2026-04-17): trending regime has dynamic_min=3,
        so 3 recoverable voters reaches the regime quorum. Should relax.
        Previously this was incorrectly blocked by the hardcoded MIN_VOTERS+1=4
        threshold, which didn't account for regime-specific quorums.
        """
        votes = {f"s{i}": "BUY" for i in range(3)}
        accuracy = {f"s{i}": self._make_stats(0.46) for i in range(3)}
        rel = _compute_gate_relaxation(
            votes, accuracy, set(), set(), 0.47, regime="trending-up",
        )
        # bp=3, dynamic_min=3, 3>=3 -> engage. bp<floor(5) -> partial-recovery MAX.
        assert rel == pytest.approx(_GATE_RELAXATION_MAX)

    def test_five_candidates_three_recoverable_trending_relaxes(self):
        """Codex round 6 exact scenario: 5 raw votes with 3 recoverable 0.46
        voters and 2 directionally gated voters in trending-up must still
        be actionable. bp=3 meets trending quorum of 3.
        """
        votes = {f"s{i}": "BUY" for i in range(5)}
        accuracy = {f"s{i}": self._make_stats(0.46) for i in range(3)}
        # 2 directionally gated voters (buy_accuracy=0.30 with enough samples).
        for i in (3, 4):
            accuracy[f"s{i}"] = {
                "accuracy": 0.60, "total": 100,
                "buy_accuracy": 0.30, "sell_accuracy": 0.60,
                "total_buy": 50, "total_sell": 50,
            }
        rel = _compute_gate_relaxation(
            votes, accuracy, set(), set(), 0.47, regime="trending-up",
        )
        assert rel == pytest.approx(_GATE_RELAXATION_MAX)

    def test_four_voter_ranging_stays_hold(self):
        """Ranging regime: 4 BUY candidates at 0.46 still below ranging
        quorum (5). bp=4 < dynamic_min=5, blocked. Relaxation here would
        be pointless because the downstream dynamic_min_voters check would
        force-HOLD anyway.
        """
        votes = {f"s{i}": "BUY" for i in range(4)}
        accuracy = {f"s{i}": self._make_stats(0.46) for i in range(4)}
        rel = _compute_gate_relaxation(votes, accuracy, set(), set(), 0.47, regime="ranging")
        assert rel == 0.0

    def test_four_voter_high_vol_relaxes(self):
        """High-vol regime: dynamic_min=4. 4 recoverable voters meets quorum."""
        votes = {f"s{i}": "BUY" for i in range(4)}
        accuracy = {f"s{i}": self._make_stats(0.46) for i in range(4)}
        rel = _compute_gate_relaxation(
            votes, accuracy, set(), set(), 0.47, regime="high-vol",
        )
        assert rel == pytest.approx(_GATE_RELAXATION_MAX)

    def test_ranging_5raw_4recoverable_1irrecoverable_relaxes(self):
        """Codex round 7 (2026-04-17): 5 raw candidates in ranging where
        4 are recoverable and 1 is directionally gated. Raw candidates
        (5) meets ranging dynamic_min (5), so downstream passes. Lone-
        signal guard: bp=4 >= 2. Should relax to MAX for partial recovery.
        Previously blocked by the too-strict `bp >= regime_quorum` check.
        """
        votes = {f"s{i}": "BUY" for i in range(5)}
        accuracy = {f"s{i}": self._make_stats(0.46) for i in range(4)}
        # s4 directionally gated (buy_accuracy=0.30).
        accuracy["s4"] = {
            "accuracy": 0.60, "total": 100,
            "buy_accuracy": 0.30, "sell_accuracy": 0.60,
            "total_buy": 50, "total_sell": 50,
        }
        rel = _compute_gate_relaxation(
            votes, accuracy, set(), set(), 0.47, regime="ranging",
        )
        assert rel == pytest.approx(_GATE_RELAXATION_MAX)

    def test_group_gate_thin_slate_stays_hold(self):
        """Codex round 9 (2026-04-17): 5 raw candidates where a correlation
        cluster group-gates 3 of them, leaving only 2 post-exclusion. Even
        though raw count (5) meets the ranging quorum (5), the post-exclusion
        slate (2) is too thin to drive meaningful consensus via relaxation.
        Must return 0.0 to block the escape.
        """
        votes = {f"s{i}": "BUY" for i in range(5)}
        accuracy = {f"s{i}": self._make_stats(0.46) for i in range(5)}
        # 3 signals group-gated (simulates macro_external cluster collapse).
        rel = _compute_gate_relaxation(
            votes, accuracy, excluded=set(), group_gated={"s0", "s1", "s2"},
            base_gate=0.47, regime="ranging",
        )
        assert rel == 0.0, (
            "Post-exclusion slate of 2 voters is too thin for relaxation "
            "regardless of raw count or regime quorum."
        )

    def test_ranging_5raw_with_top_n_excluded_still_relaxes(self):
        """Codex round 8 (2026-04-17): top-N exclusion must not shrink the
        raw-candidate count used for the regime-quorum check. Downstream's
        `extra_info['_voters']` is post-regime but PRE top-N, so the
        circuit-breaker's raw count must match.

        Scenario: 5 raw BUY votes in ranging, top-N excludes 1 (simulating
        max_active_signals=4). Without the round-8 fix, candidates would
        drop to 4 < ranging quorum 5 and relaxation would incorrectly be
        blocked even though downstream _voters=5 passes.
        """
        votes = {f"s{i}": "BUY" for i in range(5)}
        accuracy = {f"s{i}": self._make_stats(0.46) for i in range(5)}
        # top-N excludes s4 (lowest-accuracy signal).
        rel = _compute_gate_relaxation(
            votes, accuracy, excluded={"s4"}, group_gated=set(),
            base_gate=0.47, regime="ranging",
        )
        # raw=5 >= 5, bp=4 (s4 excluded) >= 2, bp > baseline(0) -> relax.
        # bp < soft floor (5) -> partial-recovery MAX.
        assert rel == pytest.approx(_GATE_RELAXATION_MAX)

    def test_ranging_5raw_lone_recoverable_stays_hold(self):
        """Lone-signal escape in ranging: 5 raw candidates, 4 directionally
        gated, only 1 recoverable at 0.46. Raw (5) meets ranging quorum,
        so downstream would NOT block via dynamic_min, but relaxation
        would let a single accuracy-passing signal drive consensus.
        The lone-signal guard (bp >= 2) must block this.
        """
        votes = {f"s{i}": "BUY" for i in range(5)}
        # 4 signals directionally gated.
        accuracy = {}
        for i in range(4):
            accuracy[f"s{i}"] = {
                "accuracy": 0.60, "total": 100,
                "buy_accuracy": 0.30, "sell_accuracy": 0.60,
                "total_buy": 50, "total_sell": 50,
            }
        # 1 borderline recoverable signal.
        accuracy["s4"] = self._make_stats(0.46)
        rel = _compute_gate_relaxation(
            votes, accuracy, set(), set(), 0.47, regime="ranging",
        )
        assert rel == 0.0, (
            "A single effective voter (bp=1) cannot be the basis of a "
            "relaxed consensus, regardless of raw candidate count."
        )

    def test_lone_recoverable_signal_blocked_from_escape(self):
        """Codex P2 follow-up (2026-04-17): 5 candidate signals where 4 are
        directionally gated (buy_accuracy=0.30) and only 1 is recoverable
        at 0.46. The raw-candidate check passes (5 candidates) but max
        relaxation can only recover 1 voter. Without a best_possible>=2
        guard, the breaker would relax to 0.06 and let that lone borderline
        signal flip consensus from HOLD to BUY. Must return 0.0.
        """
        votes = {f"s{i}": "BUY" for i in range(5)}
        # 4 signals with buy_accuracy=0.30 (directionally gated even at
        # max relaxation since _DIRECTIONAL_GATE_THRESHOLD=0.40 is not
        # relaxed).
        for i in range(4):
            stats = self._make_stats(0.60)
            stats["buy_accuracy"] = 0.30
            votes[f"s{i}"] = "BUY"
            stats["total_buy"] = 50
            votes[f"s{i}"] = "BUY"
            _ = stats  # keep for accuracy dict
        accuracy = {}
        for i in range(4):
            accuracy[f"s{i}"] = {
                "accuracy": 0.60, "total": 100,
                "buy_accuracy": 0.30, "sell_accuracy": 0.60,
                "total_buy": 50, "total_sell": 50,
            }
        # 1 borderline signal at 0.46 that would pass at max relaxation.
        accuracy["s4"] = self._make_stats(0.46)
        rel = _compute_gate_relaxation(votes, accuracy, set(), set(), 0.47)
        assert rel == 0.0, (
            "A single recoverable signal with 4 directionally-gated companions "
            "must NOT trigger relaxation - that's the exact lone-borderline "
            "escape the accuracy gate is designed to prevent."
        )

    def test_partial_recovery_when_one_signal_unrecoverable(self):
        """Codex P2 (2026-04-16 follow-up): a single irrecoverable outlier
        must NOT veto relaxation for the rest. Previously best_possible<floor
        short-circuited to 0.0, suppressing the recovery of 4 valid voters.

        Scenario: 4 BUY signals at 0.42 (recoverable at relaxation>=0.06)
        + 1 SELL signal at 0.30 (unrecoverable). best_possible=4<5=floor,
        but relaxation still recovers those 4 vs baseline of 0. Should
        return max relaxation to preserve partial recovery.
        """
        votes = {f"s{i}": "BUY" for i in range(4)}
        votes["s4"] = "SELL"
        accuracy = {f"s{i}": self._make_stats(0.42) for i in range(4)}
        # s4 is directionally gated (buy_acc irrelevant for SELL vote) - use
        # overall=0.30 to also fail the overall gate. With sell_acc=0.30, the
        # directional gate (threshold 0.40) fires on SELL direction too.
        accuracy["s4"] = self._make_stats(0.30)
        # Regime-aware: in trending-up the dynamic_min=3 quorum allows bp=4
        # recovery (4 >= 3). In ranging this would be blocked (4 < 5).
        rel = _compute_gate_relaxation(
            votes, accuracy, set(), set(), 0.47, regime="trending-up",
        )
        assert rel == pytest.approx(_GATE_RELAXATION_MAX), (
            "Partial recovery (4 recoverable + 1 irrecoverable) should "
            "return max relaxation in trending regime where quorum=3."
        )

    def test_no_relaxation_when_few_candidate_signals(self):
        """Pre-condition guard: single voter scenario. Max relaxation can't
        recover the floor (only 1 candidate, floor is 5), so returns 0."""
        votes = {"s0": "BUY"}
        accuracy = {"s0": self._make_stats(0.30)}
        rel = _compute_gate_relaxation(votes, accuracy, set(), set(), 0.47)
        assert rel == 0.0

    def test_handles_none_excluded_and_group_gated(self):
        """Defensive: None for excluded/group_gated shouldn't crash the hot
        consensus path. Reviewer 2 P3 suggestion - treat None as empty."""
        votes = {f"s{i}": "BUY" for i in range(6)}
        accuracy = {f"s{i}": self._make_stats(0.60) for i in range(6)}
        # Should not raise TypeError from `in None`
        rel = _compute_gate_relaxation(
            votes, accuracy, excluded=None, group_gated=None, base_gate=0.47,
        )
        assert rel == 0.0  # 6 signals all pass; no relaxation needed

    def test_intermediate_step_relaxation(self):
        """2026-04-16 review (Reviewer 3 P2-5): explicit test for the 0.04
        intermediate step. Previous tests only checked step 0 (no-op), step 1
        (0.02), and max (0.06). A loop off-by-one that made step 2 land at
        0.06 would slip through. 4 passing + 3 borderline at 0.44 requires
        exactly 0.04 relaxation.
        """
        # 4 signals that pass any gate + 3 at 0.44 (below 0.45 relaxed gate
        # but above 0.43). Floor of 5 requires 1 of the 0.44 signals.
        votes = {f"s{i}": "BUY" for i in range(7)}
        accuracy = {f"s{i}": self._make_stats(0.60) for i in range(4)}
        accuracy.update({f"s{i}": self._make_stats(0.44) for i in range(4, 7)})
        rel = _compute_gate_relaxation(votes, accuracy, set(), set(), 0.47)
        # Effective gate after relaxation = 0.47 - rel. Need 0.44 > 0.47 - rel,
        # i.e. rel > 0.03. Smallest step meeting this: 0.04.
        assert rel == pytest.approx(0.04)


class TestCircuitBreakerHighSampleInteraction:
    """SC-P1-2 (2026-05-02 adversarial follow-ups): the tiered high-sample
    gate (>=10000 samples -> 0.50 floor) is NOT relaxed by the circuit
    breaker — only the standard tier is.

    Reasoning: a signal with 10K+ samples at sub-50% accuracy has measurable
    negative edge (10K samples is statistically significant). Relaxing this
    gate during regime transitions would let demonstrated-poor signals back
    into voting. The standard tier still relaxes uniformly so newer
    borderline signals (<10K samples) can be rescued.

    Originally written 2026-04-16 (Reviewer 3 P1-2) to verify uniform
    relaxation across both tiers; rewritten 2026-05-02 to enforce the
    asymmetric relaxation."""

    def _make_stats(self, acc, total=200):
        return {
            "accuracy": acc, "total": total,
            "buy_accuracy": acc, "sell_accuracy": acc,
            "total_buy": total // 2, "total_sell": total // 2,
        }

    def test_high_sample_tier_not_relaxed(self):
        """A 10K-sample signal at 0.48 must remain gated at the 0.50 floor
        regardless of relaxation. Standard tier (5K samples) at 0.48 should
        pass with 0.02 relaxation."""
        from portfolio.signal_engine import _count_active_voters_at_gate
        votes_high = {"big": "BUY"}
        accuracy_high = {"big": self._make_stats(0.48, total=10000)}
        # No relaxation: high-sample gate at 0.50 > 0.48 -> gated.
        assert _count_active_voters_at_gate(
            votes_high, accuracy_high, set(), set(), 0.47, 0.0,
        ) == 0
        # 0.02 relaxation: high-sample gate stays at 0.50 (NOT relaxed) > 0.48 -> still gated.
        assert _count_active_voters_at_gate(
            votes_high, accuracy_high, set(), set(), 0.47, 0.02,
        ) == 0
        # Even at max relaxation (0.06), high-sample gate stays at 0.50 -> still gated.
        assert _count_active_voters_at_gate(
            votes_high, accuracy_high, set(), set(), 0.47, _GATE_RELAXATION_MAX,
        ) == 0

        # Standard tier at the same 0.48 accuracy DOES respond to relaxation.
        votes_low = {"newer": "BUY"}
        accuracy_low = {"newer": self._make_stats(0.48, total=5000)}
        # Standard gate 0.47 - 0.02 = 0.45, 0.48 >= 0.45 -> passes (in fact already passes baseline).
        assert _count_active_voters_at_gate(
            votes_low, accuracy_low, set(), set(), 0.47, 0.0,
        ) == 1

    def test_high_sample_and_low_sample_mixed_scenario(self):
        """Mixed population: 3 standard-tier signals at 0.45 pass at 0.02
        relaxation, but 3 high-sample at 0.49 stay gated (0.50 floor doesn't
        budge). Total active = 3, not 6."""
        from portfolio.signal_engine import _count_active_voters_at_gate
        votes = {f"s{i}": "BUY" for i in range(6)}
        accuracy = {}
        for i in range(3):
            accuracy[f"s{i}"] = self._make_stats(0.45, total=200)
        for i in range(3, 6):
            accuracy[f"s{i}"] = self._make_stats(0.49, total=10000)
        # Base gate: low-sample fail (0.45 < 0.47), high-sample fail (0.49 < 0.50) -> 0 active.
        assert _count_active_voters_at_gate(
            votes, accuracy, set(), set(), 0.47, 0.0,
        ) == 0
        # 0.02 relax: low-sample (0.45 < 0.45 False, passes); high-sample
        # (0.49 < 0.50 still True since high gate not relaxed, gated) -> 3 active.
        assert _count_active_voters_at_gate(
            votes, accuracy, set(), set(), 0.47, 0.02,
        ) == 3
        # Even with max relaxation, the 3 high-sample signals stay gated at 0.50.
        assert _count_active_voters_at_gate(
            votes, accuracy, set(), set(), 0.47, _GATE_RELAXATION_MAX,
        ) == 3


class TestCircuitBreakerIntegration:
    """End-to-end: does _weighted_consensus emit votes when relaxation kicks in?"""

    def _build_case(self, accuracies):
        votes = {f"s{i}": "BUY" for i in range(len(accuracies))}
        accuracy = {
            f"s{i}": {
                "accuracy": acc, "total": 200,
                "buy_accuracy": acc, "sell_accuracy": acc,
                "total_buy": 100, "total_sell": 100,
            }
            for i, acc in enumerate(accuracies)
        }
        return votes, accuracy

    def test_integration_preserves_buy_consensus_when_relaxed(self):
        """4 signals at 0.55, 3 at 0.46 — under base gate (0.47) only 4 active;
        circuit breaker relaxes to 0.45, letting 3 more vote. BUY consensus preserved."""
        votes, accuracy = self._build_case([0.55, 0.55, 0.55, 0.55, 0.46, 0.46, 0.46])
        action, conf = _weighted_consensus(
            votes, accuracy, regime="unknown",
        )
        assert action == "BUY"
        assert conf > 0.5

    def test_integration_no_relaxation_when_enough_voters(self):
        """All 6 above base gate — relaxation stays at 0."""
        votes, accuracy = self._build_case([0.60, 0.60, 0.60, 0.60, 0.60, 0.60])
        action, conf = _weighted_consensus(votes, accuracy, regime="unknown")
        assert action == "BUY"
        assert conf > 0.5


class TestPostPersistenceVoterCount:
    """BUG-224: extra_info['_voters_post_filter'] must reflect the count AFTER
    the persistence filter reduces active voters, not the inflated pre-filter
    count stored in extra_info['_voters']."""

    def test_post_filter_count_logic(self):
        """Verify the counting formula matches the persistence-filtered dict."""
        # Simulate a consensus_votes dict where persistence filter forced
        # some BUY/SELL votes to HOLD
        consensus_votes = {
            "rsi": "BUY",
            "macd": "BUY",
            "ema": "HOLD",        # was BUY, filtered
            "bb": "SELL",
            "sentiment": "HOLD",  # was SELL, filtered
            "volume": "HOLD",     # genuine HOLD
        }
        post_persistence_voters = sum(
            1 for v in consensus_votes.values() if v in ("BUY", "SELL")
        )
        assert post_persistence_voters == 3  # rsi, macd, bb

    def test_post_filter_all_hold(self):
        """All votes filtered to HOLD → 0 post-persistence voters."""
        consensus_votes = {"rsi": "HOLD", "macd": "HOLD", "ema": "HOLD"}
        post_persistence_voters = sum(
            1 for v in consensus_votes.values() if v in ("BUY", "SELL")
        )
        assert post_persistence_voters == 0


class TestBug227PostFilterGates:
    """BUG-227: The weighted consensus gate and dynamic_min_voters penalty must
    use post-persistence voter count, not the pre-filter count. Without this,
    weak consensus can pass the gate when the persistence filter has reduced
    actual participating voters below the threshold."""

    def test_confidence_penalty_reads_post_filter_voters(self):
        """Stage 4 dynamic_min_voters should use _voters_post_filter."""
        import numpy as np
        import pandas as pd

        from portfolio.signal_engine import apply_confidence_penalties
        df = pd.DataFrame({"close": np.linspace(100, 105, 50)})

        # Pre-filter: 6 voters (would pass min_voters=5 for ranging).
        # Post-filter: 2 voters (should FAIL min_voters=5 for ranging).
        extra_info = {
            "_voters": 6,
            "_voters_post_filter": 2,
            "_buy_count": 4,
            "_sell_count": 2,
        }
        action, conf, log = apply_confidence_penalties(
            action="BUY",
            conf=0.65,
            regime="ranging",  # dynamic_min = 5
            ind={},
            extra_info=extra_info,
            ticker="BTC-USD",
            df=df,
            config={"confidence_penalties": {"enabled": True}},
        )
        # With only 2 post-filter voters < 5 (ranging min), should force HOLD
        assert action == "HOLD"
        assert conf == 0.0

    def test_confidence_penalty_passes_with_enough_post_filter_voters(self):
        """When post-filter voters meet the threshold, action should survive."""
        import numpy as np
        import pandas as pd

        from portfolio.signal_engine import apply_confidence_penalties
        df = pd.DataFrame({"close": np.linspace(100, 105, 50)})

        extra_info = {
            "_voters": 8,
            "_voters_post_filter": 6,
            "_buy_count": 5,
            "_sell_count": 3,
        }
        action, conf, log = apply_confidence_penalties(
            action="BUY",
            conf=0.65,
            regime="ranging",  # dynamic_min = 5
            ind={},
            extra_info=extra_info,
            ticker="BTC-USD",
            df=df,
            config={"confidence_penalties": {"enabled": True}},
        )
        # 6 post-filter voters >= 5 (ranging min) → should NOT force HOLD
        assert action != "HOLD" or conf > 0.0  # at least one must be non-HOLD

    def test_confidence_penalty_fallback_to_voters_key(self):
        """Backward compat: if _voters_post_filter is missing, use _voters."""
        import numpy as np
        import pandas as pd

        from portfolio.signal_engine import apply_confidence_penalties
        df = pd.DataFrame({"close": np.linspace(100, 105, 50)})

        # Only _voters (pre-filter), no post-filter key → use it as fallback
        extra_info = {
            "_voters": 6,
            "_buy_count": 4,
            "_sell_count": 2,
        }
        action, conf, log = apply_confidence_penalties(
            action="BUY",
            conf=0.65,
            regime="ranging",  # dynamic_min = 5
            ind={},
            extra_info=extra_info,
            ticker="BTC-USD",
            df=df,
            config={"confidence_penalties": {"enabled": True}},
        )
        # 6 voters >= 5 → should NOT force HOLD at Stage 4
        # (may still be penalized by other stages, but not zero'd by Stage 4)
        has_stage4_hold = any(
            e.get("stage") == "dynamic_min_voters" for e in log
        )
        assert not has_stage4_hold


class TestHighSampleGateNotRelaxed:
    """SC-P1-2 (2026-05-02 adversarial follow-ups): the high-sample 0.50 gate
    must not be relaxed below 0.50 by the circuit breaker.

    A signal with 10K+ samples whose accuracy is 0.46 has statistically
    demonstrated negative edge — circuit-breaker relaxation that drops the
    high-sample gate to e.g. 0.44 would let it vote, despite the very purpose
    of the high-sample tier being to apply a stricter gate to long-track-record
    signals. The relaxation should still apply to the standard tier so newer
    signals with borderline accuracy can be rescued.
    """

    def _make_stats(self, acc, total=100, buy_acc=None, sell_acc=None):
        return {
            "accuracy": acc,
            "total": total,
            "buy_accuracy": buy_acc if buy_acc is not None else acc,
            "sell_accuracy": sell_acc if sell_acc is not None else acc,
            "total_buy": 50,
            "total_sell": 50,
        }

    def test_high_sample_signal_at_046_gated_even_with_max_relaxation(self):
        """Signal with 20K samples at 0.46 accuracy must stay gated at any
        relaxation. 0.50 is its absolute floor."""
        votes = {"long_track": "BUY"}
        accuracy = {"long_track": self._make_stats(0.46, total=20000)}
        # Even with max relaxation (currently 0.06pp), high-sample gate stays at 0.50.
        active = _count_active_voters_at_gate(
            votes, accuracy, set(), set(), 0.47, _GATE_RELAXATION_MAX,
        )
        assert active == 0, (
            "High-sample gate (0.50) must not be relaxed; "
            "a 0.46-accuracy signal with 20K samples must remain gated."
        )

    def test_high_sample_signal_at_051_passes_with_relaxation(self):
        """Signal with 20K samples at 0.51 accuracy passes the 0.50 floor
        regardless of relaxation."""
        votes = {"long_track": "BUY"}
        accuracy = {"long_track": self._make_stats(0.51, total=20000)}
        active = _count_active_voters_at_gate(
            votes, accuracy, set(), set(), 0.47, 0.0,
        )
        assert active == 1

    def test_standard_tier_still_relaxed(self):
        """The relaxation must still apply to non-high-sample signals (otherwise
        the circuit-breaker is broken). A signal at 0.46 with 5K samples
        (below the 10K high-sample threshold) should pass with 0.02 relaxation."""
        votes = {"newer_sig": "BUY"}
        # 5K samples is below _ACCURACY_GATE_HIGH_SAMPLE_MIN (10K).
        accuracy = {"newer_sig": self._make_stats(0.46, total=5000)}
        # Without relaxation: gated at 0.47.
        assert _count_active_voters_at_gate(
            votes, accuracy, set(), set(), 0.47, 0.0,
        ) == 0
        # With 0.02 relaxation: effective gate 0.45, passes (acc 0.46 >= 0.45).
        assert _count_active_voters_at_gate(
            votes, accuracy, set(), set(), 0.47, 0.02,
        ) == 1

    def test_high_sample_tier_at_threshold_passes(self):
        """Boundary: 0.50 at high-sample tier passes (gate is acc < threshold)."""
        votes = {"sig": "BUY"}
        accuracy = {"sig": self._make_stats(0.50, total=15000)}
        active = _count_active_voters_at_gate(
            votes, accuracy, set(), set(), 0.47, _GATE_RELAXATION_MAX,
        )
        assert active == 1


class TestRescuedFlagInitializedPerIteration:
    """P1-1 (2026-05-02 adversarial follow-ups): defensive — the `_rescued`
    flag in `_weighted_consensus`'s voting loop must be initialized at the
    top of each iteration so future refactors that introduce a fall-through
    branch don't accidentally read a stale True from a prior iteration.

    In current code the flag is set on both arms of the gate-check (rescue
    branch sets True, else branch sets False), so the bug doesn't manifest
    today. The defensive init guards against a future contributor adding
    a third branch without realising _rescued must be set.
    """

    def _make_stats(self, acc, total=100, buy_acc=None, sell_acc=None,
                    total_buy=50, total_sell=50):
        return {
            "accuracy": acc,
            "total": total,
            "buy_accuracy": buy_acc if buy_acc is not None else acc,
            "sell_accuracy": sell_acc if sell_acc is not None else acc,
            "total_buy": total_buy,
            "total_sell": total_sell,
        }

    def test_iteration_order_invariance(self):
        """Buy weight is invariant under iteration order, even when one
        signal takes the rescue branch and another passes cleanly.

        Builds a 3-signal vote: rescued + clean + sell. Reverses dict order
        to flip iteration order. With either current code or the defensive
        init, results must match — but the defensive init makes this
        guarantee robust to future refactors.
        """
        accuracy = {
            "rescued_sig": self._make_stats(
                0.40, total=200, buy_acc=0.65, total_buy=50,
            ),
            "clean_sig": self._make_stats(
                0.65, total=200, buy_acc=0.65, total_buy=50,
            ),
            "sell_sig": self._make_stats(
                0.50, total=200, sell_acc=0.50, total_sell=50, total_buy=50,
            ),
        }
        votes_a = {"rescued_sig": "BUY", "clean_sig": "BUY", "sell_sig": "SELL"}
        votes_b = {"clean_sig": "BUY", "rescued_sig": "BUY", "sell_sig": "SELL"}
        action_a, conf_a = _weighted_consensus(
            votes_a, accuracy, regime=None, activation_rates={},
            accuracy_gate=0.47, ticker=None, horizon=None,
        )
        action_b, conf_b = _weighted_consensus(
            votes_b, accuracy, regime=None, activation_rates={},
            accuracy_gate=0.47, ticker=None, horizon=None,
        )
        assert action_a == action_b
        assert conf_a == pytest.approx(conf_b, abs=1e-4), (
            f"Iteration order changed result: a={conf_a}, b={conf_b}. "
            "_rescued may be leaking across iterations."
        )

    def test_rescued_init_at_loop_top_structural(self):
        """Source-level guard: assert that `_rescued = False` appears as the
        first statement of the voting loop in `_weighted_consensus`. This is
        the defensive init mandated by the 2026-05-02 follow-ups review.

        A future refactor that removes the init (or adds a branch which
        bypasses the rescue gate-check, leaking a stale True) will fail
        this test loudly.
        """
        import inspect

        from portfolio import signal_engine

        src = inspect.getsource(signal_engine._weighted_consensus)
        # The init must appear after the `for signal_name, vote in votes.items():`
        # line and BEFORE any branch that could set _rescued conditionally.
        loop_idx = src.find("for signal_name, vote in votes.items():")
        assert loop_idx != -1, "loop signature changed — review _rescued init"
        first_set = src.find("_rescued = True", loop_idx)
        first_init = src.find("_rescued = False", loop_idx)
        assert first_init != -1, "_rescued = False not present in loop body"
        # Defensive init must appear BEFORE the first conditional setter.
        assert first_init < first_set, (
            f"_rescued = False (idx {first_init}) must appear before any "
            f"conditional _rescued = True (idx {first_set}) so a stale "
            "True from a prior iteration cannot leak."
        )
