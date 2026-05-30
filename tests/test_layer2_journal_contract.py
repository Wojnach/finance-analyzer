"""Tests for the 2026-04-13 layer2_journal_activity contract.

The contract fires when Layer 2 is enabled, a trigger fired recently,
the grace window has elapsed, but no journal entry has been written
since the trigger. Background context in
``docs/plans/2026-04-13-claude-auth-detection.md``.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from portfolio import loop_contract

# ---------------------------------------------------------------------------
# Helpers — build tmp fixture layouts
# ---------------------------------------------------------------------------

def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")


@pytest.fixture
def contract_env(tmp_path, monkeypatch):
    """Redirect all file paths used by the contract into tmp_path."""
    paths = {
        "CONFIG_FILE": tmp_path / "config.json",
        "HEALTH_STATE_FILE": tmp_path / "data" / "health_state.json",
        "LAYER2_JOURNAL_FILE": tmp_path / "data" / "layer2_journal.jsonl",
        "CLAUDE_INVOCATIONS_FILE": tmp_path / "data" / "claude_invocations.jsonl",
        "LAYER2_INVOCATIONS_FILE": tmp_path / "data" / "invocations.jsonl",
        "CONTRACT_STATE_FILE": tmp_path / "data" / "contract_state.json",
    }
    for name, p in paths.items():
        monkeypatch.setattr(loop_contract, name, p)

    # Also redirect the critical_errors.jsonl that the contract writes to
    # (via record_critical_error) so we don't pollute production data/.
    from portfolio import claude_gate
    monkeypatch.setattr(
        claude_gate, "CRITICAL_ERRORS_LOG", tmp_path / "data" / "critical_errors.jsonl"
    )
    return tmp_path, paths


# ---------------------------------------------------------------------------
# Preconditions — no violation when gates block the check
# ---------------------------------------------------------------------------

def test_no_violation_when_layer2_disabled(contract_env):
    tmp_path, p = contract_env
    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": False}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(datetime.now(UTC) - timedelta(hours=2)),
        "last_trigger_reason": "test trigger",
    })

    assert loop_contract.check_layer2_journal_activity() == []


def test_no_violation_when_no_recent_trigger(contract_env):
    tmp_path, p = contract_env
    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(datetime.now(UTC) - timedelta(hours=12)),
        "last_trigger_reason": "old trigger",
    })
    assert loop_contract.check_layer2_journal_activity() == []


def test_grace_window_is_18_minutes(contract_env):
    """BUG-202 (2026-04-16): pin the grace window to 18 min.

    The constant was 60m before — a value that predates T3's 15-min
    subprocess timeout and let three consecutive overnight auth-silent
    outages pass undetected (Apr 14–16). If someone widens it again,
    this test forces them to justify it in review.
    """
    assert loop_contract.LAYER2_JOURNAL_GRACE_S == 18 * 60


def test_no_violation_before_grace_window_elapses(contract_env):
    """Trigger just fired 10 minutes ago — agent still has time to journal."""
    tmp_path, p = contract_env
    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(datetime.now(UTC) - timedelta(minutes=10)),
        "last_trigger_reason": "fresh trigger",
    })
    assert loop_contract.check_layer2_journal_activity() == []


def test_no_violation_when_missing_state_files(contract_env):
    """Fail-safe: if the state files don't exist we don't cry wolf."""
    tmp_path, p = contract_env
    # No files written at all.
    assert loop_contract.check_layer2_journal_activity() == []


def test_no_violation_when_config_missing_layer2_key(contract_env):
    """Config without layer2 key — default is enabled=True so the check
    runs, but no health file → precondition 2 fails → no violation."""
    tmp_path, p = contract_env
    _write_json(p["CONFIG_FILE"], {"other_key": "value"})
    assert loop_contract.check_layer2_journal_activity() == []


# ---------------------------------------------------------------------------
# Happy path — recent trigger + journal written after
# ---------------------------------------------------------------------------

def test_no_violation_when_journal_written_after_trigger(contract_env):
    tmp_path, p = contract_env
    now = datetime.now(UTC)
    trigger_ts = now - timedelta(hours=2)
    journal_ts = now - timedelta(hours=1, minutes=50)

    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(trigger_ts),
        "last_trigger_reason": "test",
    })
    _write_jsonl(p["LAYER2_JOURNAL_FILE"], [
        {"timestamp": _iso(journal_ts), "decision": "HOLD"},
    ])

    assert loop_contract.check_layer2_journal_activity() == []


# ---------------------------------------------------------------------------
# Violation paths — the core regression guard
# ---------------------------------------------------------------------------

def test_violation_when_trigger_fired_but_no_journal(contract_env):
    """The exact scenario the silent --bare outage produced."""
    tmp_path, p = contract_env
    now = datetime.now(UTC)
    trigger_ts = now - timedelta(hours=3)

    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(trigger_ts),
        "last_trigger_reason": "BTC-USD consensus SELL",
    })
    # No journal entries.
    p["LAYER2_JOURNAL_FILE"].parent.mkdir(parents=True, exist_ok=True)
    p["LAYER2_JOURNAL_FILE"].write_text("", encoding="utf-8")

    violations = loop_contract.check_layer2_journal_activity()
    assert len(violations) == 1
    v = violations[0]
    assert v.invariant == "layer2_journal_activity"
    assert v.severity == "CRITICAL"
    assert "BTC-USD consensus SELL" in v.message
    assert v.details["trigger_age_s"] == pytest.approx(3 * 3600, abs=60)


def test_violation_when_journal_is_stale_before_trigger(contract_env):
    """Journal has old entries from BEFORE the trigger — must still fire."""
    tmp_path, p = contract_env
    now = datetime.now(UTC)
    trigger_ts = now - timedelta(hours=3)
    old_journal_ts = now - timedelta(days=2)

    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(trigger_ts),
        "last_trigger_reason": "test",
    })
    _write_jsonl(p["LAYER2_JOURNAL_FILE"], [
        {"timestamp": _iso(old_journal_ts), "decision": "BUY"},
    ])

    violations = loop_contract.check_layer2_journal_activity()
    assert len(violations) == 1


def test_violation_writes_to_critical_errors_journal(contract_env):
    """The violation must also land in critical_errors.jsonl so the
    CLAUDE.md STARTUP CHECK surfaces it."""
    tmp_path, p = contract_env
    now = datetime.now(UTC)
    trigger_ts = now - timedelta(hours=3)

    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(trigger_ts),
        "last_trigger_reason": "test",
    })
    p["LAYER2_JOURNAL_FILE"].parent.mkdir(parents=True, exist_ok=True)
    p["LAYER2_JOURNAL_FILE"].write_text("", encoding="utf-8")

    loop_contract.check_layer2_journal_activity()

    crit_file = tmp_path / "data" / "critical_errors.jsonl"
    assert crit_file.exists()
    entries = [json.loads(line) for line in crit_file.read_text().splitlines() if line.strip()]
    assert len(entries) == 1
    assert entries[0]["category"] == "contract_violation"
    assert entries[0]["caller"] == "layer2_journal_activity"
    assert entries[0]["resolution"] is None


def test_violation_includes_last_invocation_context(contract_env):
    """If invocations.jsonl has a recent L2 entry, it's quoted in
    the violation details — saves the user from doing the correlation.

    2026-05-04: now reads LAYER2_INVOCATIONS_FILE (the L2-specific log)
    instead of the global claude log. The global log has entries from
    claude_fundamental, bigbet, iskbets, golddigger fix-agent — its tail
    is essentially noise here, and produced misleading details on
    2026-05-03/04 (last_invocation_caller="loop_contract_golddigger" on
    a Layer 2 violation).
    """
    tmp_path, p = contract_env
    now = datetime.now(UTC)
    trigger_ts = now - timedelta(hours=3)

    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(trigger_ts),
        "last_trigger_reason": "test",
    })
    p["LAYER2_JOURNAL_FILE"].parent.mkdir(parents=True, exist_ok=True)
    p["LAYER2_JOURNAL_FILE"].write_text("", encoding="utf-8")
    inv_ts = trigger_ts + timedelta(seconds=30)
    _write_jsonl(p["LAYER2_INVOCATIONS_FILE"], [
        {"ts": _iso(inv_ts),
         "reasons": ["test"], "status": "success",
         "tier": 3, "exit_code": 1, "duration_s": 0.5},
    ])

    violations = loop_contract.check_layer2_journal_activity()
    assert len(violations) == 1
    details = violations[0].details
    assert details["last_invocation_status"] == "success"
    assert details["last_invocation_ts"] == _iso(inv_ts)
    assert details["last_invocation_tier"] == 3
    assert details["last_invocation_exit_code"] == 1


def test_violation_context_ignores_unrelated_global_caller(contract_env):
    """Regression for 2026-05-03/04: the global claude_invocations.jsonl
    had a stale ``loop_contract_golddigger`` timeout entry from yesterday
    that bled into Layer 2 violation details, making it look like Layer 2
    was broken when only the journal write timing missed the grace
    window. Confirm the L2-specific log wins now."""
    tmp_path, p = contract_env
    now = datetime.now(UTC)
    trigger_ts = now - timedelta(hours=3)

    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(trigger_ts),
        "last_trigger_reason": "test",
    })
    p["LAYER2_JOURNAL_FILE"].parent.mkdir(parents=True, exist_ok=True)
    p["LAYER2_JOURNAL_FILE"].write_text("", encoding="utf-8")

    # Stale unrelated entry in the GLOBAL log — should NOT appear in details.
    _write_jsonl(p["CLAUDE_INVOCATIONS_FILE"], [
        {"timestamp": _iso(now - timedelta(hours=20)),
         "caller": "loop_contract_golddigger", "status": "timeout",
         "exit_code": 124},
    ])
    # Fresh L2 entry — should appear in details.
    fresh_ts = trigger_ts - timedelta(seconds=10)
    _write_jsonl(p["LAYER2_INVOCATIONS_FILE"], [
        {"ts": _iso(fresh_ts),
         "reasons": ["test"], "status": "success",
         "tier": 1, "exit_code": 0,
         "duration_s": 480.0, "journal_written": True},
    ])

    violations = loop_contract.check_layer2_journal_activity()
    assert len(violations) == 1
    details = violations[0].details
    # L2-specific status, NOT the stale "timeout" from the global log.
    assert details["last_invocation_status"] == "success"
    assert details["last_invocation_ts"] == _iso(fresh_ts)
    # No "caller" leakage from the global log either.
    assert "last_invocation_caller" not in details


# ---------------------------------------------------------------------------
# Integration: verify_contract() includes the check
# ---------------------------------------------------------------------------

def test_verify_contract_runs_layer2_check(contract_env):
    """verify_contract should call through to check_layer2_journal_activity
    so the main loop picks up violations without extra wiring."""
    tmp_path, p = contract_env
    now = datetime.now(UTC)
    trigger_ts = now - timedelta(hours=3)

    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(trigger_ts),
        "last_trigger_reason": "test",
    })
    p["LAYER2_JOURNAL_FILE"].parent.mkdir(parents=True, exist_ok=True)
    p["LAYER2_JOURNAL_FILE"].write_text("", encoding="utf-8")

    # Minimal valid CycleReport — no other invariants fire.
    from portfolio.loop_contract import CycleReport
    now_t = time.time()
    report = CycleReport(
        cycle_id=1,
        active_tickers={"BTC-USD"},
        signals_ok=1,
        signals_failed=0,
        signals={"BTC-USD": {"action": "HOLD", "confidence": 0.5}},
        cycle_start=now_t,
        cycle_end=now_t + 1,
        llm_batch_flushed=True,
        health_updated=True,
        heartbeat_updated=True,
        summary_written=True,
    )

    violations = loop_contract.verify_contract(report)
    # Only the layer2 violation should appear (all other invariants OK).
    assert any(v.invariant == "layer2_journal_activity" for v in violations)


# ---------------------------------------------------------------------------
# Autonomous status suppression (2026-05-30)
# ---------------------------------------------------------------------------

def test_no_violation_when_autonomous_invocation_recent(contract_env):
    """autonomous_* invocations should suppress the journal contract
    because autonomous writes its own journal entry (or failure stub)."""
    tmp_path, p = contract_env
    now = datetime.now(UTC)
    trigger_ts = now - timedelta(hours=2)

    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(trigger_ts),
        "last_trigger_reason": "BTC-USD consensus BUY",
    })
    _write_jsonl(p["LAYER2_INVOCATIONS_FILE"], [
        {"ts": _iso(trigger_ts + timedelta(seconds=1)),
         "reasons": ["BTC-USD consensus BUY"],
         "status": "autonomous_layer2_disabled",
         "tier": 2},
    ])
    p["LAYER2_JOURNAL_FILE"].parent.mkdir(parents=True, exist_ok=True)
    p["LAYER2_JOURNAL_FILE"].write_text("", encoding="utf-8")

    assert loop_contract.check_layer2_journal_activity() == []


def test_no_violation_when_autonomous_failed_status(contract_env):
    """autonomous_failed is a known failure status — suppresses violation."""
    tmp_path, p = contract_env
    now = datetime.now(UTC)
    trigger_ts = now - timedelta(hours=2)

    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(trigger_ts),
        "last_trigger_reason": "test",
    })
    _write_jsonl(p["LAYER2_INVOCATIONS_FILE"], [
        {"ts": _iso(trigger_ts + timedelta(seconds=1)),
         "reasons": ["test"],
         "status": "autonomous_failed",
         "tier": 2},
    ])
    p["LAYER2_JOURNAL_FILE"].parent.mkdir(parents=True, exist_ok=True)
    p["LAYER2_JOURNAL_FILE"].write_text("", encoding="utf-8")

    assert loop_contract.check_layer2_journal_activity() == []
