"""Tests for routing accuracy_degradation CRITICAL violations into critical_errors.jsonl.

Background (2026-04-28): the auto-fix-agent dispatcher (PF-FixAgentDispatcher,
every 10 min) reads data/critical_errors.jsonl. The accuracy_degradation check
emits CRITICAL violations to contract_violations.jsonl + Telegram, but never
to critical_errors.jsonl. As a result the dispatcher has been blind to a 32 h
streak of degradation alerts and the only escalation path was Telegram spam.

This test file pins the expected behavior: ``_dispatch_critical_errors_for_degradation``
writes a critical_errors.jsonl row for every CRITICAL accuracy_degradation
violation it sees, deduplicated on (invariant, message_hash) so identical
replays don't blow up the journal — but with a TTL so a long-running issue
keeps reappearing in the dispatcher's lookback window.

The dispatch hook is wired into ``verify_and_act`` AFTER the
ViolationTracker has had its chance to escalate consecutive WARNINGs to
CRITICAL. The ``TestVerifyAndActDispatchAfterTracker`` class below covers
that integration; the per-unit dedup/TTL behavior here calls the hook
directly with synthetic violations.
"""

import hashlib
import json

import pytest

from portfolio.loop_contract import (
    Violation,
    _dispatch_critical_errors_for_degradation,
    check_signal_accuracy_degradation_safe,
)


@pytest.fixture()
def critical_errors_paths(tmp_path, monkeypatch):
    """Redirect critical_errors.jsonl + the dedup state file to tmp."""
    crit_file = tmp_path / "critical_errors.jsonl"
    state_file = tmp_path / "contract_state.json"
    monkeypatch.setattr(
        "portfolio.claude_gate.CRITICAL_ERRORS_LOG", crit_file,
    )
    monkeypatch.setattr(
        "portfolio.loop_contract.CONTRACT_STATE_FILE", state_file,
    )
    return crit_file, state_file


def _read_jsonl(path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _make_critical_violation(message="12 signal(s) dropped..."):
    return Violation(
        invariant="accuracy_degradation",
        severity="CRITICAL",
        message=message,
        details={"alert_count": 12, "baseline_age_days": 6.7, "alerts": []},
    )


class TestAccuracyDegradationDispatcherWire:
    """check_signal_accuracy_degradation_safe writes critical_errors when CRITICAL."""

    def test_critical_violation_writes_critical_error(self, critical_errors_paths):
        crit_file, _state_file = critical_errors_paths
        v = _make_critical_violation()
        _dispatch_critical_errors_for_degradation([v])

        rows = _read_jsonl(crit_file)
        assert len(rows) == 1
        assert rows[0]["category"] == "accuracy_degradation"
        assert rows[0]["caller"] == "accuracy_degradation"
        assert rows[0]["resolution"] is None
        assert rows[0]["level"] == "critical"
        assert "12 signal(s)" in rows[0]["message"]

    def test_warning_violation_does_not_write_critical_error(self, critical_errors_paths):
        crit_file, _state_file = critical_errors_paths
        # WARNING from the inner check (1-2 simultaneous drops). The
        # tracker can later escalate this to CRITICAL but until that
        # promotion happens the dispatcher should stay quiet.
        v = Violation(
            invariant="accuracy_degradation",
            severity="WARNING",
            message="1 signal dropped >15pp...",
        )
        _dispatch_critical_errors_for_degradation([v])
        assert _read_jsonl(crit_file) == []

    def test_check_signal_accuracy_degradation_safe_does_not_write_directly(
        self, critical_errors_paths,
    ):
        """Sanity: the inner check should NOT write critical_errors itself
        anymore — that responsibility moved to verify_and_act post-tracker
        so escalations are captured (Codex P2 2026-04-28). Confirms we
        don't accidentally double-write."""
        from unittest.mock import patch

        crit_file, _state_file = critical_errors_paths
        v = _make_critical_violation()
        with patch(
            "portfolio.accuracy_degradation.check_degradation",
            return_value=[v],
        ):
            result = check_signal_accuracy_degradation_safe()
        assert len(result) == 1
        assert result[0].severity == "CRITICAL"
        # No row — only the verify_and_act hook is allowed to write.
        assert _read_jsonl(crit_file) == []

    def test_identical_message_does_not_re_append_within_ttl(self, critical_errors_paths):
        """Replays of the same cached violation must not blow up the journal.
        Otherwise every 10 min cycle of a 32 h streak adds 192 identical rows."""
        crit_file, _state_file = critical_errors_paths
        v = _make_critical_violation()
        _dispatch_critical_errors_for_degradation([v])
        _dispatch_critical_errors_for_degradation([v])
        _dispatch_critical_errors_for_degradation([v])

        rows = _read_jsonl(crit_file)
        assert len(rows) == 1, (
            "Expected a single critical_errors row across 3 identical replays "
            f"within the TTL; got {len(rows)} — dedup is broken"
        )

    def test_message_change_appends_new_row(self, critical_errors_paths):
        """When the alert text changes (e.g. one more signal joins), the
        dispatcher should see a fresh row so the auto-fix-agent re-engages."""
        crit_file, _state_file = critical_errors_paths
        v_old = _make_critical_violation("12 signal(s) dropped...")
        v_new = _make_critical_violation("13 signal(s) dropped...")

        _dispatch_critical_errors_for_degradation([v_old])
        _dispatch_critical_errors_for_degradation([v_new])

        rows = _read_jsonl(crit_file)
        assert len(rows) == 2

    def test_check_degradation_failure_yields_empty_violations_no_writes(
        self, critical_errors_paths,
    ):
        """Wrapped exception still degrades to []; nothing should land in
        critical_errors.jsonl on a check_degradation crash."""
        from unittest.mock import patch

        crit_file, _state_file = critical_errors_paths
        with patch(
            "portfolio.accuracy_degradation.check_degradation",
            side_effect=RuntimeError("snapshot file corrupt"),
        ):
            result = check_signal_accuracy_degradation_safe()

        assert result == []
        assert _read_jsonl(crit_file) == []

    def test_dedup_state_persists_in_contract_state_file(self, critical_errors_paths):
        """The dedup marker for critical_errors lives in contract_state.json
        (not in degradation_alert_state, to keep accuracy_degradation a leaf
        module)."""
        from portfolio.file_utils import load_json

        crit_file, state_file = critical_errors_paths
        v = _make_critical_violation()
        _dispatch_critical_errors_for_degradation([v])

        state = load_json(state_file, default={}) or {}
        assert "critical_error_dispatch" in state
        sha1 = hashlib.sha1(v.message.encode("utf-8")).hexdigest()
        per_inv = state["critical_error_dispatch"].get("accuracy_degradation") or {}
        assert per_inv.get("last_message_hash") == sha1

    def test_dedup_ttl_lets_persistent_issue_re_appear(self, critical_errors_paths):
        """Codex P2 follow-up: dedup must not skip on hash equality forever.
        Otherwise a persistent regression's row ages out of the dispatcher's
        24 h lookback and the auto-fix-agent stops engaging on a still-
        broken issue. After the TTL window passes, the same message must
        re-append a fresh row."""
        import time as _time

        from portfolio.file_utils import atomic_write_json, load_json

        crit_file, state_file = critical_errors_paths
        v = _make_critical_violation()

        # First fire — one row, dedup state recorded.
        _dispatch_critical_errors_for_degradation([v])
        rows1 = _read_jsonl(crit_file)
        assert len(rows1) == 1

        # Forge the dedup ts to be older than the TTL (any window > 6 h
        # must trigger a re-emit per the contract pinned here).
        state = load_json(state_file, default={}) or {}
        per_inv = state["critical_error_dispatch"]["accuracy_degradation"]
        per_inv["ts"] = _time.time() - 24 * 3600  # one full day ago
        atomic_write_json(state_file, state)

        # Second fire with identical text — TTL elapsed, so a fresh row.
        _dispatch_critical_errors_for_degradation([v])
        rows2 = _read_jsonl(crit_file)
        assert len(rows2) == 2, (
            "Same-hash replay after the TTL window must re-append; "
            "otherwise PF-FixAgentDispatcher loses sight of long-running issues"
        )


class TestVerifyAndActDispatchAfterTracker:
    """Codex P2 follow-up: tracker-escalated WARNING -> CRITICAL must also
    reach critical_errors.jsonl. The original implementation hooked into
    check_signal_accuracy_degradation_safe (pre-tracker) so escalations were
    invisible to the dispatcher."""

    def test_warning_escalated_to_critical_writes_critical_error(
        self, critical_errors_paths,
    ):
        from unittest.mock import patch

        from portfolio.loop_contract import (
            CycleReport,
            ESCALATION_THRESHOLD,
            ViolationTracker,
            verify_and_act,
        )

        crit_file, state_file = critical_errors_paths

        # Inner check returns the WARNING form (1-2 simultaneous drops).
        warning = Violation(
            invariant="accuracy_degradation",
            severity="WARNING",
            message="2 signal(s) dropped >15pp...",
            details={"alert_count": 2},
        )

        # Pre-load the tracker so the next fire is the (ESCALATION_THRESHOLD)th
        # consecutive — the one that escalates WARNING -> CRITICAL.
        tracker = ViolationTracker(state_file)
        tracker._consecutive["accuracy_degradation"] = ESCALATION_THRESHOLD - 1
        tracker._save()

        # Use a clean report that wouldn't otherwise trigger violations.
        report = CycleReport(cycle_id=1, active_tickers={"BTC-USD"})
        report.signals_ok = 1
        report.signals_failed = 0
        report.signals = {"BTC-USD": {
            "action": "HOLD", "confidence": 0.5,
            "extra": {"active_voters": 5},
        }}
        report.cycle_start = 1.0
        report.cycle_end = 50.0
        report.llm_batch_flushed = True
        report.health_updated = True
        report.heartbeat_updated = True
        report.summary_written = True

        with patch(
            "portfolio.accuracy_degradation.check_degradation",
            return_value=[warning],
        ), patch("portfolio.loop_contract._alert_violations"), \
             patch("portfolio.loop_contract._trigger_self_heal"):
            verify_and_act(
                report, {}, tracker=ViolationTracker(state_file),
            )

        rows = _read_jsonl(crit_file)
        assert len(rows) == 1, (
            "WARNING escalated to CRITICAL by tracker must reach "
            "critical_errors.jsonl too — otherwise PF-FixAgentDispatcher "
            f"never sees the promoted alert. Got: {rows}"
        )
        assert rows[0]["category"] == "accuracy_degradation"
