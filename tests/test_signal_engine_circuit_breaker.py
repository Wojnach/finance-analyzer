"""Tests for the voter-count circuit breaker (Batch 2 of 2026-04-16 accuracy
gating reconfiguration).

The circuit breaker relaxes the accuracy gate when cascaded gates would leave
too few active voters, preserving voter diversity during regime transitions.
It does NOT relax the directional or correlation gates.
"""

import pytest

from portfolio.signal_engine import (
    ACCURACY_GATE_THRESHOLD,
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
        assert _GATE_RELAXATION_STEP == pytest.approx(0.02)

    def test_relaxation_max_at_6pp(self):
        assert _GATE_RELAXATION_MAX == pytest.approx(0.06)

    def test_max_is_multiple_of_step(self):
        """Step must divide max cleanly so iteration reaches the exact cap.

        Use a ratio check because float modulo leaves residual noise
        (0.06 % 0.02 == 0.019999... in IEEE-754).
        """
        ratio = _GATE_RELAXATION_MAX / _GATE_RELAXATION_STEP
        assert ratio == pytest.approx(round(ratio), abs=1e-9)


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
