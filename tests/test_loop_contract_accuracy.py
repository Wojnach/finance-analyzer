"""Integration tests for the BUG-178/W15-W16 accuracy degradation invariant.

Asserts:
- check_signal_accuracy_degradation_safe() is called from verify_contract()
- The wrapper swallows exceptions so a broken accuracy stack can't take down
  the rest of the contract framework.
- The wrapper returns the Violations produced by check_degradation().
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import portfolio.accuracy_degradation as deg
import portfolio.loop_contract as lc
from portfolio.loop_contract import (
    CycleReport,
    Violation,
    check_signal_accuracy_degradation_safe,
    verify_contract,
)


def _make_passing_report() -> CycleReport:
    """Build a CycleReport that passes the other 11 invariants.

    The degradation invariant is the only one we want to exercise here;
    everything else needs to be in a state that wouldn't itself emit
    a violation, otherwise we can't tell which one fired.
    """
    report = CycleReport(cycle_id=1)
    report.cycle_start = 100.0
    report.cycle_end = 130.0
    report.active_tickers = {"BTC-USD"}
    report.signals_ok = 1
    report.signals_failed = 0
    report.signals = {
        "BTC-USD": {"action": "HOLD", "confidence": 0.5},
    }
    report.health_updated = True
    report.summary_written = True
    report.heartbeat_updated = True
    report.llm_batch_flushed = True
    report.post_cycle_results = {}
    report.errors = []
    return report


class TestInvariantWiring:

    def test_safe_wrapper_swallows_exceptions(self, monkeypatch):
        def boom():
            raise RuntimeError("synthetic failure to test the safety net")
        monkeypatch.setattr(deg, "check_degradation", boom)
        result = check_signal_accuracy_degradation_safe()
        assert result == []

    def test_safe_wrapper_returns_violations_from_check(self, monkeypatch):
        v = Violation(
            invariant=deg.DEGRADATION_INVARIANT,
            severity=deg.SEVERITY_WARNING,
            message="synthetic",
            details={"alert_count": 1},
        )
        monkeypatch.setattr(deg, "check_degradation", lambda: [v])
        result = check_signal_accuracy_degradation_safe()
        assert result == [v]

    def test_invariant_wired_into_verify_contract(self, monkeypatch):
        """verify_contract() must call our safe wrapper."""
        sentinel = Violation(
            invariant=deg.DEGRADATION_INVARIANT,
            severity=deg.SEVERITY_WARNING,
            message="sentinel from accuracy degradation",
            details={},
        )
        # Patch the wrapper at its loop_contract import site so verify_contract
        # picks up the stub regardless of how it imports it.
        monkeypatch.setattr(
            lc, "check_signal_accuracy_degradation_safe",
            lambda: [sentinel],
        )
        # Also stub the layer2 check so the only violation we see is ours.
        monkeypatch.setattr(lc, "check_layer2_journal_activity", lambda: [])

        report = _make_passing_report()
        violations = verify_contract(report)

        names = [v.invariant for v in violations]
        assert deg.DEGRADATION_INVARIANT in names

    def test_check_uses_throttle_state_path(self, monkeypatch, tmp_path):
        """The check should never block the loop more than ~1s.

        Approximated by asserting that two back-to-back calls (the second
        hits the hourly throttle) both return very quickly. We don't
        measure wall time precisely — the worktree CI has variable IO —
        but we DO assert the throttle replay path is taken.
        """
        monkeypatch.setattr(deg, "ALERT_STATE_FILE", tmp_path / "alert.json")
        monkeypatch.setattr(deg, "SNAPSHOT_STATE_FILE", tmp_path / "snap.json")

        # Empty snapshots file -> first call returns []
        from portfolio.accuracy_stats import ACCURACY_SNAPSHOTS_FILE
        monkeypatch.setattr(
            "portfolio.accuracy_stats.ACCURACY_SNAPSHOTS_FILE",
            tmp_path / "snap.jsonl",
        )

        # First call updates last_full_check_time
        first = deg.check_degradation()
        assert first == []  # no baseline

        # Second call within throttle: replay path. Even with no baseline
        # this hits the cached-empty list, NOT a fresh compute.
        with patch.object(deg, "_load_snapshots") as mock_load:
            second = deg.check_degradation()
            assert mock_load.call_count == 0  # throttle short-circuited
        assert second == []
