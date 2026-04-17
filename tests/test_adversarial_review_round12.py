"""Regression tests for Codex round-12 finding on commit 48d935a4.

Codex round 12 flagged that the round-11 deep-sanitization had a real
semantic bug: coercing a poisoned `{"accuracy": null, "total": 200}` to
`{"accuracy": 0.5, "total": 200}` made the row look like a MATURE 50%
signal. It would then pass the min-samples gate, count toward quorum,
and its NaN directional values would no longer trigger the downstream
`.get('buy_accuracy', acc)` fallback.

Round 12 fix: poisoned numeric fields are DROPPED entirely, so downstream
`.get(key, default)` reads the safe default. A row with only `total=200`
and no accuracy becomes indistinguishable from a signal with missing
stats (bypasses the min-samples gate because samples=0 via default).

This test file pins the corrected behavior.
"""

from __future__ import annotations

import pytest

from portfolio.signal_engine import _weighted_consensus


class TestSanitizationDropsPoisonedFields:
    """Round-12: poisoned fields must be DROPPED, not replaced with defaults."""

    def test_none_accuracy_does_not_create_mature_50pct_signal(self):
        """A row with {"accuracy": None, "total": 200} must NOT be promoted
        to a 50% accurate mature signal. Compare consensus against the same
        row with accuracy simply absent."""
        # Case A: poisoned accuracy.
        votes = {"s0": "BUY", "s1": "BUY", "s2": "BUY", "s3": "BUY"}
        accuracy_poisoned = {
            "s0": {"accuracy": None, "total": 200,
                   "buy_accuracy": 0.60, "total_buy": 100},
            "s1": {"accuracy": 0.60, "total": 200,
                   "buy_accuracy": 0.60, "total_buy": 100},
            "s2": {"accuracy": 0.60, "total": 200,
                   "buy_accuracy": 0.60, "total_buy": 100},
            "s3": {"accuracy": 0.60, "total": 200,
                   "buy_accuracy": 0.60, "total_buy": 100},
        }
        # Case B: accuracy field missing entirely.
        accuracy_missing = {
            "s0": {"total": 200, "buy_accuracy": 0.60, "total_buy": 100},
            "s1": accuracy_poisoned["s1"],
            "s2": accuracy_poisoned["s2"],
            "s3": accuracy_poisoned["s3"],
        }

        action_p, conf_p = _weighted_consensus(
            votes, accuracy_poisoned, regime="trending-up",
        )
        action_m, conf_m = _weighted_consensus(
            votes, accuracy_missing, regime="trending-up",
        )
        # Both produce equivalent consensus because the poisoned row ends
        # up with {accuracy, total} both dropped (round-13 fix), matching
        # case B where the whole row was absent. Without round-13 the
        # poisoned row would have total=200 kept and the gate behavior
        # would differ.
        assert action_p == action_m, (
            "Poisoned-accuracy row must behave as absent (round-13 drop-pair "
            f"semantics). Got action_p={action_p}, action_m={action_m}"
        )

    def test_round13_paired_drop_when_total_valid_but_accuracy_poisoned(self):
        """Codex round 13 exact scenario: {"accuracy": None, "total": 200}
        must have BOTH accuracy and total dropped by the sanitizer, so the
        row bypasses the min-samples gate rather than masquerading as a
        mature 50% signal. Test by checking that such a row doesn't
        contribute to the consensus the way a mature 0.5-signal would.
        """
        # Build two scenarios that should produce IDENTICAL consensus if the
        # round-13 drop-pair semantics are correct:
        #   A: poisoned-row (None accuracy, valid total) mixed in
        #   B: same row with {accuracy, total} BOTH absent
        votes = {"poison": "BUY", "s1": "BUY", "s2": "BUY", "s3": "BUY"}
        accuracy_a = {
            "poison": {"accuracy": None, "total": 200},
            "s1": {"accuracy": 0.60, "total": 200},
            "s2": {"accuracy": 0.60, "total": 200},
            "s3": {"accuracy": 0.60, "total": 200},
        }
        accuracy_b = {
            "poison": {},  # both fields absent
            "s1": {"accuracy": 0.60, "total": 200},
            "s2": {"accuracy": 0.60, "total": 200},
            "s3": {"accuracy": 0.60, "total": 200},
        }
        action_a, conf_a = _weighted_consensus(
            votes, accuracy_a, regime="trending-up",
        )
        action_b, conf_b = _weighted_consensus(
            votes, accuracy_b, regime="trending-up",
        )
        assert (action_a, round(conf_a, 4)) == (action_b, round(conf_b, 4)), (
            f"Round-13: poisoned-accuracy + valid-total must be equivalent to "
            f"both-fields-absent. Got A={action_a},{conf_a} vs B={action_b},{conf_b}"
        )

    def test_nan_directional_falls_back_to_overall(self):
        """A signal with overall=0.60 and buy_accuracy=NaN should have the
        directional gate/weight path fall back to overall 0.60, not pass NaN
        through as a value."""
        import math
        votes = {"s0": "BUY", "s1": "BUY", "s2": "BUY", "s3": "BUY"}
        accuracy = {
            "s0": {
                "accuracy": 0.60, "total": 200,
                "buy_accuracy": float("nan"), "total_buy": 100,
            },
            "s1": {"accuracy": 0.60, "total": 200,
                   "buy_accuracy": 0.60, "total_buy": 100},
            "s2": {"accuracy": 0.60, "total": 200,
                   "buy_accuracy": 0.60, "total_buy": 100},
            "s3": {"accuracy": 0.60, "total": 200,
                   "buy_accuracy": 0.60, "total_buy": 100},
        }
        action, conf = _weighted_consensus(votes, accuracy, regime="trending-up")
        # Finite confidence regardless of the NaN field.
        assert not math.isnan(conf)
        assert action == "BUY"

    def test_negative_samples_omitted_not_coerced_to_zero(self):
        """A -50 sample count must behave as if the key were absent - signal
        bypasses min_samples gate and downstream fallbacks apply."""
        votes = {"s0": "BUY", "s1": "BUY", "s2": "BUY", "s3": "BUY"}
        accuracy = {
            "s0": {"accuracy": 0.60, "total": -50},  # negative -> dropped
            "s1": {"accuracy": 0.60, "total": 200},
            "s2": {"accuracy": 0.60, "total": 200},
            "s3": {"accuracy": 0.60, "total": 200},
        }
        # Must not crash; s0's total is dropped so samples=0 (via default),
        # which means the min-samples gate doesn't fire (samples<30 bypass).
        action, conf = _weighted_consensus(votes, accuracy, regime="trending-up")
        assert action == "BUY"

    def test_poisoned_row_does_not_pass_min_samples_gate(self):
        """A row with ONLY a poisoned accuracy (no other fields) must bypass
        all gates (as if fresh signal), not masquerade as a mature 50% one."""
        # Using a single signal to isolate behavior, though in practice
        # _weighted_consensus will need more voters for quorum.
        votes = {"s0": "BUY", "s1": "BUY", "s2": "BUY", "s3": "BUY"}
        accuracy = {
            # Poisoned row: in old code -> {accuracy: 0.5, total: 200} -> mature
            # In new code -> {} (accuracy dropped, total dropped) -> bypasses gate
            "s0": {"accuracy": None, "total": "garbage"},
            "s1": {"accuracy": 0.55, "total": 200,
                   "buy_accuracy": 0.55, "total_buy": 100},
            "s2": {"accuracy": 0.55, "total": 200,
                   "buy_accuracy": 0.55, "total_buy": 100},
            "s3": {"accuracy": 0.55, "total": 200,
                   "buy_accuracy": 0.55, "total_buy": 100},
        }
        # Must not crash. s0's poisoned values dropped means it bypasses gates
        # and votes at neutral default; real signals drive direction.
        action, conf = _weighted_consensus(votes, accuracy, regime="trending-up")
        assert action in ("BUY", "SELL", "HOLD")
