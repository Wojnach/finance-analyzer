"""Tests for routing accuracy_degradation CRITICAL violations into critical_errors.jsonl.

Background (2026-04-28): the auto-fix-agent dispatcher (PF-FixAgentDispatcher,
every 10 min) reads data/critical_errors.jsonl. The accuracy_degradation check
emits CRITICAL violations to contract_violations.jsonl + Telegram, but never
to critical_errors.jsonl. As a result the dispatcher has been blind to a 32 h
streak of degradation alerts and the only escalation path was Telegram spam.

This test file pins the expected behavior: the loop_contract wrapper around
the accuracy_degradation check writes a critical_errors.jsonl row when the
inner check returns CRITICAL violations, with dedup keyed on
(invariant, message_hash) so we don't append the same row every cycle.
"""

import hashlib
import json

import pytest

from portfolio.loop_contract import (
    Violation,
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
        from unittest.mock import patch

        crit_file, _state_file = critical_errors_paths
        v = _make_critical_violation()
        with patch(
            "portfolio.accuracy_degradation.check_degradation",
            return_value=[v],
        ):
            result = check_signal_accuracy_degradation_safe()

        # Violation still propagates through to the contract framework.
        assert len(result) == 1
        assert result[0].severity == "CRITICAL"

        # critical_errors.jsonl now has exactly one row.
        rows = _read_jsonl(crit_file)
        assert len(rows) == 1
        assert rows[0]["category"] == "accuracy_degradation"
        assert rows[0]["caller"] == "accuracy_degradation"
        assert rows[0]["resolution"] is None
        assert rows[0]["level"] == "critical"
        assert "12 signal(s)" in rows[0]["message"]

    def test_warning_violation_does_not_write_critical_error(self, critical_errors_paths):
        from unittest.mock import patch

        crit_file, _state_file = critical_errors_paths
        v = Violation(
            invariant="accuracy_degradation",
            severity="WARNING",  # not CRITICAL — single drop
            message="1 signal dropped >15pp...",
        )
        with patch(
            "portfolio.accuracy_degradation.check_degradation",
            return_value=[v],
        ):
            check_signal_accuracy_degradation_safe()

        # critical_errors.jsonl gets nothing — WARNINGs aren't auto-fix-agent material.
        assert _read_jsonl(crit_file) == []

    def test_identical_message_does_not_re_append(self, critical_errors_paths):
        """Replays of the same cached violation must not blow up the journal.
        Otherwise every 10 min cycle of a 32 h streak adds 192 identical rows."""
        from unittest.mock import patch

        crit_file, _state_file = critical_errors_paths
        v = _make_critical_violation()
        with patch(
            "portfolio.accuracy_degradation.check_degradation",
            return_value=[v],
        ):
            check_signal_accuracy_degradation_safe()
            check_signal_accuracy_degradation_safe()
            check_signal_accuracy_degradation_safe()

        rows = _read_jsonl(crit_file)
        assert len(rows) == 1, (
            "Expected a single critical_errors row across 3 identical replays; "
            f"got {len(rows)} — dedup is broken"
        )

    def test_message_change_appends_new_row(self, critical_errors_paths):
        """When the alert text changes (e.g. one more signal joins), the
        dispatcher should see a fresh row so the auto-fix-agent re-engages."""
        from unittest.mock import patch

        crit_file, _state_file = critical_errors_paths
        v_old = _make_critical_violation("12 signal(s) dropped...")
        v_new = _make_critical_violation("13 signal(s) dropped...")

        with patch(
            "portfolio.accuracy_degradation.check_degradation",
            return_value=[v_old],
        ):
            check_signal_accuracy_degradation_safe()
        with patch(
            "portfolio.accuracy_degradation.check_degradation",
            return_value=[v_new],
        ):
            check_signal_accuracy_degradation_safe()

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
        from unittest.mock import patch

        from portfolio.file_utils import load_json

        crit_file, state_file = critical_errors_paths
        v = _make_critical_violation()
        with patch(
            "portfolio.accuracy_degradation.check_degradation",
            return_value=[v],
        ):
            check_signal_accuracy_degradation_safe()

        state = load_json(state_file, default={}) or {}
        assert "critical_error_dispatch" in state
        sha1 = hashlib.sha1(v.message.encode("utf-8")).hexdigest()
        per_inv = state["critical_error_dispatch"].get("accuracy_degradation") or {}
        assert per_inv.get("last_message_hash") == sha1
