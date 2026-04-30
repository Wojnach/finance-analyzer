"""Tests for the snapshot_freshness invariant.

The 2026-04-21 → 2026-04-28 detection blackout (docs/PLAN_detection_blackout_20260501.md)
exposed a structural gap: when accuracy_snapshots.jsonl stops growing, the
degradation detector silently goes dark — the cascade gates ("baseline within
36h", "baseline ≥6 days old") all return [] in the no-baseline branch with no
external surfacing.

This invariant adds an INDEPENDENT freshness check that flags
accuracy_snapshots.jsonl staleness via WARNING (36h+) → CRITICAL (48h+).
Independent of maybe_save_daily_snapshot's own size-check (which only fires when
that function actually runs), so a wedged loop or never-called writer is
caught.

Wired into the existing CRITICAL_ERROR_DISPATCH_INVARIANTS so escalation hits
the auto-fix-agent dispatcher.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import portfolio.loop_contract as lc


@pytest.fixture
def isolated_snapshots_file(tmp_path, monkeypatch):
    """Redirect ACCURACY_SNAPSHOTS_FILE to a tmp path so each test owns it.

    Patching at the accuracy_stats import location matters because
    loop_contract.check_snapshot_freshness_safe lazy-imports it from there.
    """
    snapshots_path = tmp_path / "accuracy_snapshots.jsonl"
    from portfolio import accuracy_stats
    monkeypatch.setattr(accuracy_stats, "ACCURACY_SNAPSHOTS_FILE", snapshots_path)
    return snapshots_path


def _set_mtime_hours_ago(path: Path, hours: float) -> None:
    """Set both atime and mtime to ``hours`` hours before now."""
    target_ts = time.time() - hours * 3600.0
    os.utime(path, (target_ts, target_ts))


class TestSnapshotFreshness:

    def test_fresh_snapshot_no_violation(self, isolated_snapshots_file):
        """A snapshot file modified <36h ago must produce zero violations."""
        isolated_snapshots_file.write_text('{"ts": "2026-05-01T06:00:00+00:00"}\n')
        _set_mtime_hours_ago(isolated_snapshots_file, 12.0)

        violations = lc.check_snapshot_freshness_safe()

        assert violations == [], (
            f"Fresh snapshot must not produce violations; got {violations}"
        )

    def test_stale_36h_warning(self, isolated_snapshots_file):
        """File mtime ~36h+ stale produces a WARNING violation."""
        isolated_snapshots_file.write_text('{"ts": "2026-04-29T06:00:00+00:00"}\n')
        _set_mtime_hours_ago(isolated_snapshots_file, 36.5)

        violations = lc.check_snapshot_freshness_safe()

        assert len(violations) == 1, (
            f"Expected exactly 1 violation; got {len(violations)}"
        )
        v = violations[0]
        assert v.invariant == lc.SNAPSHOT_FRESHNESS_INVARIANT
        assert v.severity == "WARNING"
        assert "stale" in v.message.lower() or "36" in v.message

    def test_stale_48h_critical(self, isolated_snapshots_file):
        """File mtime 48h+ stale escalates to CRITICAL.

        At ≥48h the daily writer has missed two consecutive cycles; the
        degradation detector's 36h baseline tolerance is also being
        violated, so the detector is effectively dark.
        """
        isolated_snapshots_file.write_text('{"ts": "2026-04-29T06:00:00+00:00"}\n')
        _set_mtime_hours_ago(isolated_snapshots_file, 50.0)

        violations = lc.check_snapshot_freshness_safe()

        assert len(violations) == 1
        v = violations[0]
        assert v.invariant == lc.SNAPSHOT_FRESHNESS_INVARIANT
        assert v.severity == "CRITICAL"

    def test_missing_jsonl_warning(self, isolated_snapshots_file):
        """Missing file → WARNING (could be day-1 deployment).

        Don't escalate to CRITICAL on absence: a fresh repo with no
        snapshots yet is a normal day-1 state, distinct from a writer
        that USED to work and now doesn't. Operators should see the
        WARN, but auto-fix-agent shouldn't engage on a virgin install.
        """
        # File deliberately not created
        assert not isolated_snapshots_file.exists()

        violations = lc.check_snapshot_freshness_safe()

        assert len(violations) == 1
        v = violations[0]
        assert v.invariant == lc.SNAPSHOT_FRESHNESS_INVARIANT
        assert v.severity == "WARNING"
        assert "no snapshot file" in v.message.lower() or \
               "not present" in v.message.lower() or \
               "missing" in v.message.lower()

    def test_invariant_does_not_raise_on_io_error(
        self, isolated_snapshots_file, monkeypatch
    ):
        """If Path.stat() raises, the wrapper returns [] and logs a warning.

        Mirrors the design of check_signal_accuracy_degradation_safe: the
        contract framework runs every cycle, and a single broken stat()
        call must never take down the rest of the framework. The shape
        of the safe wrapper is what's tested — actual stat failures are
        OS-level and rare.
        """
        from portfolio import accuracy_stats

        class StatExploder:
            """Path-like that explodes on stat()."""
            def exists(self):
                return True
            def stat(self):
                raise OSError("simulated I/O failure")

        monkeypatch.setattr(
            accuracy_stats, "ACCURACY_SNAPSHOTS_FILE", StatExploder(),
        )

        violations = lc.check_snapshot_freshness_safe()

        assert violations == [], (
            f"Wrapper must swallow stat errors; got {violations}"
        )

    def test_critical_dispatch_set_includes_snapshot_freshness(self):
        """The auto-fix-agent dispatcher only engages on invariants in
        CRITICAL_ERROR_DISPATCH_INVARIANTS. snapshot_freshness must be in
        that set so a stale-snapshot CRITICAL triggers automatic recovery
        instead of waiting for the next interactive session.
        """
        assert lc.SNAPSHOT_FRESHNESS_INVARIANT in lc.CRITICAL_ERROR_DISPATCH_INVARIANTS

    def test_invariant_constant_exposed(self):
        """The invariant name must be exposed as a module-level constant
        so callers (tests, dashboard, fix_agent_dispatcher) can reference
        it without hard-coding the string."""
        assert hasattr(lc, "SNAPSHOT_FRESHNESS_INVARIANT")
        assert lc.SNAPSHOT_FRESHNESS_INVARIANT == "snapshot_freshness"
