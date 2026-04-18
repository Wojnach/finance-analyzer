"""Tests for the 2026-04-17 per-tier dynamic grace window + in-flight suppression.

Context: the flat 18-minute grace in ``loop_contract.LAYER2_JOURNAL_GRACE_S``
fired false-positives during overnight hours when a T3 invocation (900s /
15m timeout) itself timed out and respawned. Two critical_errors entries
at 2026-04-16T23:11 and 2026-04-17T05:19 both showed
``last_invocation_status = "timeout"`` (not auth_failure). The wall-clock
gap between original trigger and the new invocation's journal write
exceeded 18m but nothing was actually silent — the new subprocess was
still booting.

Fix: per-tier grace (T1=3m, T2=12m, T3=20m, default=T3) + a 4th
precondition in ``check_layer2_journal_activity`` that suppresses the
alert when the most recent invocation has status="invoked" within the
effective grace window. See ``loop_contract._get_layer2_grace_s`` and the
design note above ``LAYER2_JOURNAL_GRACE_S_BY_TIER``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from portfolio import loop_contract


# ---------------------------------------------------------------------------
# Helpers — build tmp fixture layouts (matches test_layer2_journal_contract.py)
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
        # 2026-04-17 Codex P1 follow-up: in-flight suppression reads the
        # Layer-2-specific log, not the global claude log, to avoid being
        # false-negatived by unrelated claude_fundamental/bigbet calls.
        "LAYER2_INVOCATIONS_FILE": tmp_path / "data" / "invocations.jsonl",
        # 2026-04-18 violation-dedup (precondition 5) reads/writes this.
        "CONTRACT_STATE_FILE": tmp_path / "data" / "contract_state.json",
    }
    for name, p in paths.items():
        monkeypatch.setattr(loop_contract, name, p)

    # Redirect critical_errors.jsonl to keep the sandbox clean.
    from portfolio import claude_gate
    monkeypatch.setattr(
        claude_gate, "CRITICAL_ERRORS_LOG", tmp_path / "data" / "critical_errors.jsonl"
    )
    return tmp_path, paths


# ---------------------------------------------------------------------------
# Grace table — unit tests on the helper + constants
# ---------------------------------------------------------------------------

def test_grace_table_pins_tier_values():
    """Pin the per-tier grace table. If someone widens or shrinks these,
    this test forces them to justify it in review and update the CLAUDE.md
    critical-rules doc."""
    assert loop_contract.LAYER2_JOURNAL_GRACE_S_BY_TIER[1] == 3 * 60
    assert loop_contract.LAYER2_JOURNAL_GRACE_S_BY_TIER[2] == 12 * 60
    assert loop_contract.LAYER2_JOURNAL_GRACE_S_BY_TIER[3] == 20 * 60


def test_grace_default_is_t3():
    """Fail-safe: missing tier info must use T3 grace, not T1. A shorter
    default would re-introduce the false-positive problem for any tier
    whose invocation pre-dates the new health_state field."""
    assert (
        loop_contract.LAYER2_JOURNAL_GRACE_S_DEFAULT
        == loop_contract.LAYER2_JOURNAL_GRACE_S_BY_TIER[3]
    )


def test_get_layer2_grace_returns_default_for_none():
    """_get_layer2_grace_s(None) returns T3 grace."""
    assert loop_contract._get_layer2_grace_s(None) == 20 * 60


def test_get_layer2_grace_returns_default_for_missing_tier():
    """_get_layer2_grace_s({}) returns T3 grace when key is absent."""
    assert loop_contract._get_layer2_grace_s({}) == 20 * 60


def test_get_layer2_grace_returns_default_for_non_int_tier():
    """Malformed tier (string, None, etc.) falls back to T3 grace."""
    assert loop_contract._get_layer2_grace_s({"last_invocation_tier": "3"}) == 20 * 60
    assert loop_contract._get_layer2_grace_s({"last_invocation_tier": None}) == 20 * 60


@pytest.mark.parametrize("tier,expected", [(1, 180), (2, 720), (3, 1200)])
def test_get_layer2_grace_maps_each_tier(tier, expected):
    """_get_layer2_grace_s returns the correct per-tier value for 1/2/3."""
    assert loop_contract._get_layer2_grace_s({"last_invocation_tier": tier}) == expected


def test_get_layer2_grace_unknown_tier_uses_default():
    """An unknown tier number (e.g. 99) falls back to the default."""
    assert loop_contract._get_layer2_grace_s({"last_invocation_tier": 99}) == 20 * 60


# ---------------------------------------------------------------------------
# Per-tier grace — integration with check_layer2_journal_activity
# ---------------------------------------------------------------------------

def test_t1_grace_suppresses_alert_at_4m(contract_env):
    """Trigger 4m old with T1 last_tier → no alert (T1 grace is 3m, but
    this also tests that we don't false-positive before the grace has
    elapsed when the most recent invocation is T1)."""
    tmp_path, p = contract_env
    now = datetime.now(UTC)
    trigger_ts = now - timedelta(minutes=4)

    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(trigger_ts),
        "last_trigger_reason": "quick check",
        "last_invocation_tier": 1,
    })
    # Empty journal (post-grace, agent should have journaled but an
    # in-flight T1 invocation is still running — covered by precondition 4).
    # Write to BOTH files: LAYER2_INVOCATIONS_FILE for in-flight check,
    # CLAUDE_INVOCATIONS_FILE for violation-context enrichment.
    l2_entry = {"ts": _iso(now - timedelta(minutes=1)),
                "reasons": ["quick check"], "status": "invoked", "tier": 1}
    _write_jsonl(p["LAYER2_INVOCATIONS_FILE"], [l2_entry])
    _write_jsonl(p["CLAUDE_INVOCATIONS_FILE"], [
        {"timestamp": _iso(now - timedelta(minutes=1)),
         "caller": "layer2_t1", "status": "invoked", "tier": 1},
    ])
    p["LAYER2_JOURNAL_FILE"].parent.mkdir(parents=True, exist_ok=True)
    p["LAYER2_JOURNAL_FILE"].write_text("", encoding="utf-8")

    assert loop_contract.check_layer2_journal_activity() == []


def test_t3_grace_suppresses_alert_at_19m(contract_env):
    """Trigger 19m old with T3 last_tier → no alert (T3 grace is 20m).
    Under the old flat 18m grace this would have fired — this is the
    exact false-positive the fix eliminates."""
    tmp_path, p = contract_env
    now = datetime.now(UTC)
    trigger_ts = now - timedelta(minutes=19)

    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(trigger_ts),
        "last_trigger_reason": "BTC-USD consensus SELL",
        "last_invocation_tier": 3,
    })
    p["LAYER2_JOURNAL_FILE"].parent.mkdir(parents=True, exist_ok=True)
    p["LAYER2_JOURNAL_FILE"].write_text("", encoding="utf-8")

    assert loop_contract.check_layer2_journal_activity() == []


def test_default_grace_uses_t3_when_tier_absent(contract_env):
    """Trigger 19m old, no last_invocation_tier in health_state → no
    alert (fail-safe: default = T3 grace = 20m)."""
    tmp_path, p = contract_env
    now = datetime.now(UTC)
    trigger_ts = now - timedelta(minutes=19)

    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(trigger_ts),
        "last_trigger_reason": "legacy trigger pre-dating tier field",
        # last_invocation_tier intentionally omitted
    })
    p["LAYER2_JOURNAL_FILE"].parent.mkdir(parents=True, exist_ok=True)
    p["LAYER2_JOURNAL_FILE"].write_text("", encoding="utf-8")

    assert loop_contract.check_layer2_journal_activity() == []


# ---------------------------------------------------------------------------
# In-flight suppression (precondition 4)
# ---------------------------------------------------------------------------

def test_in_flight_suppresses_alert(contract_env):
    """Trigger 25m old, most recent invocation status="invoked" 2m ago →
    no alert. This is the overnight-cascade case: the first invocation
    timed out, a new one started, and we must wait for the new one to
    complete before firing the contract."""
    tmp_path, p = contract_env
    now = datetime.now(UTC)
    trigger_ts = now - timedelta(minutes=25)
    invoked_ts = now - timedelta(minutes=2)

    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(trigger_ts),
        "last_trigger_reason": "ETH-USD consensus BUY (57%)",
        "last_invocation_tier": 3,
    })
    # L2-specific log drives the in-flight check. Most recent entry:
    # fresh "invoked" after a prior "timeout" → cascade in progress.
    _write_jsonl(p["LAYER2_INVOCATIONS_FILE"], [
        {"ts": _iso(now - timedelta(minutes=20)),
         "reasons": ["ETH"], "status": "timeout", "tier": 3},
        {"ts": _iso(invoked_ts),
         "reasons": ["ETH"], "status": "invoked", "tier": 3},
    ])
    # Mirror into claude_invocations for violation-context enrichment.
    _write_jsonl(p["CLAUDE_INVOCATIONS_FILE"], [
        {"timestamp": _iso(now - timedelta(minutes=20)),
         "caller": "layer2_t3", "status": "timeout", "tier": 3},
        {"timestamp": _iso(invoked_ts),
         "caller": "layer2_t3", "status": "invoked", "tier": 3},
    ])
    p["LAYER2_JOURNAL_FILE"].parent.mkdir(parents=True, exist_ok=True)
    p["LAYER2_JOURNAL_FILE"].write_text("", encoding="utf-8")

    assert loop_contract.check_layer2_journal_activity() == []


def test_real_silent_failure_still_fires(contract_env):
    """Trigger 25m old, most recent invocation is a terminal "timeout"
    with no follow-up "invoked" entry → contract fires. This verifies
    that the in-flight suppression is narrow enough to still catch the
    original silent-stall pattern it was designed for."""
    tmp_path, p = contract_env
    now = datetime.now(UTC)
    trigger_ts = now - timedelta(minutes=25)

    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(trigger_ts),
        "last_trigger_reason": "BTC-USD consensus SELL (47%)",
        "last_invocation_tier": 3,
    })
    # Most recent invocation is terminal (timeout), no fresh in-flight.
    _write_jsonl(p["CLAUDE_INVOCATIONS_FILE"], [
        {"timestamp": _iso(now - timedelta(minutes=22)),
         "caller": "layer2_t3", "status": "timeout", "tier": 3},
    ])
    p["LAYER2_JOURNAL_FILE"].parent.mkdir(parents=True, exist_ok=True)
    p["LAYER2_JOURNAL_FILE"].write_text("", encoding="utf-8")

    violations = loop_contract.check_layer2_journal_activity()
    assert len(violations) == 1
    v = violations[0]
    assert v.invariant == "layer2_journal_activity"
    assert v.severity == "CRITICAL"
    # Violation details now include the effective grace + tier so
    # postmortems can tell at a glance which window was in play.
    assert v.details["grace_s"] == 20 * 60
    assert v.details["last_invocation_tier"] == 3
    assert v.details["last_invocation_status"] == "timeout"


def test_in_flight_invoked_older_than_grace_does_not_suppress(contract_env):
    """Edge case: an "invoked" entry older than the per-tier grace is
    almost certainly a crashed process whose terminal status never got
    written (e.g. SIGKILL from OOM, Windows taskkill). In that case we
    should NOT suppress — it's indistinguishable from a real silent
    failure and the user needs to know."""
    tmp_path, p = contract_env
    now = datetime.now(UTC)
    trigger_ts = now - timedelta(minutes=30)
    # "invoked" older than T3 grace (20m) + should fall through.
    invoked_ts = now - timedelta(minutes=25)

    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(trigger_ts),
        "last_trigger_reason": "stuck invocation",
        "last_invocation_tier": 3,
    })
    _write_jsonl(p["LAYER2_INVOCATIONS_FILE"], [
        {"ts": _iso(invoked_ts),
         "reasons": ["stuck"], "status": "invoked", "tier": 3},
    ])
    _write_jsonl(p["CLAUDE_INVOCATIONS_FILE"], [
        {"timestamp": _iso(invoked_ts),
         "caller": "layer2_t3", "status": "invoked", "tier": 3},
    ])
    p["LAYER2_JOURNAL_FILE"].parent.mkdir(parents=True, exist_ok=True)
    p["LAYER2_JOURNAL_FILE"].write_text("", encoding="utf-8")

    violations = loop_contract.check_layer2_journal_activity()
    assert len(violations) == 1


# ---------------------------------------------------------------------------
# Precondition 4b — skipped_* status suppresses the alert (2026-04-18)
# ---------------------------------------------------------------------------

def test_skipped_offhours_suppresses_alert(contract_env):
    """Overnight: Layer 2 correctly skipped off-hours → no journal write
    is expected, so the contract must not fire. Prior bug: 12 overnight
    violations were fired against triggers the loop had legitimately
    skipped."""
    tmp_path, p = contract_env
    now = datetime.now(UTC)
    trigger_ts = now - timedelta(minutes=25)
    skipped_ts = now - timedelta(minutes=20)  # AFTER the trigger

    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(trigger_ts),
        "last_trigger_reason": "XAU-USD consensus BUY (80%)",
        "last_invocation_tier": 3,
    })
    _write_jsonl(p["LAYER2_INVOCATIONS_FILE"], [
        {"ts": _iso(skipped_ts),
         "reasons": ["XAU-USD consensus BUY (80%)"],
         "status": "skipped_offhours", "tier": 2},
    ])
    p["LAYER2_JOURNAL_FILE"].parent.mkdir(parents=True, exist_ok=True)
    p["LAYER2_JOURNAL_FILE"].write_text("", encoding="utf-8")

    assert loop_contract.check_layer2_journal_activity() == []


def test_skipped_busy_does_NOT_suppress(contract_env):
    """Codex P1 2026-04-18: skipped_busy is written on real failure paths
    (couldn't kill old agent, no agent binary, etc). Must NOT be treated
    as a legitimate skip — those would mask silent failures. Only
    offhours/gate/stack_overflow qualify."""
    tmp_path, p = contract_env
    now = datetime.now(UTC)
    trigger_ts = now - timedelta(minutes=25)

    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(trigger_ts),
        "last_trigger_reason": "XAU-USD consensus BUY",
        "last_invocation_tier": 3,
    })
    _write_jsonl(p["LAYER2_INVOCATIONS_FILE"], [
        {"ts": _iso(now - timedelta(minutes=20)),
         "reasons": ["X"], "status": "skipped_busy", "tier": 3},
    ])
    p["LAYER2_JOURNAL_FILE"].parent.mkdir(parents=True, exist_ok=True)
    p["LAYER2_JOURNAL_FILE"].write_text("", encoding="utf-8")

    violations = loop_contract.check_layer2_journal_activity()
    assert len(violations) == 1, "skipped_busy must not suppress the alert"


def test_skipped_gate_suppresses_alert(contract_env):
    """skipped_gate (perception gate decided not to invoke) IS a
    legitimate non-run and should suppress."""
    tmp_path, p = contract_env
    now = datetime.now(UTC)
    trigger_ts = now - timedelta(minutes=25)

    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(trigger_ts),
        "last_trigger_reason": "XAU-USD consensus BUY",
        "last_invocation_tier": 3,
    })
    _write_jsonl(p["LAYER2_INVOCATIONS_FILE"], [
        {"ts": _iso(now - timedelta(minutes=20)),
         "reasons": ["X"], "status": "skipped_gate", "tier": 3},
    ])
    p["LAYER2_JOURNAL_FILE"].parent.mkdir(parents=True, exist_ok=True)
    p["LAYER2_JOURNAL_FILE"].write_text("", encoding="utf-8")

    assert loop_contract.check_layer2_journal_activity() == []


def test_stale_skipped_does_not_suppress(contract_env):
    """A skipped_* entry from BEFORE the trigger doesn't prove the
    current trigger was skipped — the contract must still fire. This
    protects against a single stale skipped entry from hours ago
    masking a genuine new silent-failure."""
    tmp_path, p = contract_env
    now = datetime.now(UTC)
    trigger_ts = now - timedelta(minutes=25)
    # skipped entry is OLDER than the trigger
    stale_skipped_ts = trigger_ts - timedelta(minutes=10)

    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(trigger_ts),
        "last_trigger_reason": "XAU-USD consensus BUY",
        "last_invocation_tier": 3,
    })
    _write_jsonl(p["LAYER2_INVOCATIONS_FILE"], [
        {"ts": _iso(stale_skipped_ts),
         "reasons": ["old"], "status": "skipped_offhours", "tier": 1},
    ])
    p["LAYER2_JOURNAL_FILE"].parent.mkdir(parents=True, exist_ok=True)
    p["LAYER2_JOURNAL_FILE"].write_text("", encoding="utf-8")

    violations = loop_contract.check_layer2_journal_activity()
    assert len(violations) == 1


# ---------------------------------------------------------------------------
# Precondition 5 — violation dedup per trigger (2026-04-18)
# ---------------------------------------------------------------------------

def test_second_call_for_same_trigger_does_not_fire(contract_env):
    """Once we've fired a violation for a given trigger_time, subsequent
    cycles must not re-fire. Without this, the same trigger generated
    dozens of duplicate alerts in the overnight 2026-04-17 window."""
    tmp_path, p = contract_env
    now = datetime.now(UTC)
    trigger_ts = now - timedelta(minutes=25)

    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(trigger_ts),
        "last_trigger_reason": "ETH-USD consensus BUY",
        "last_invocation_tier": 3,
    })
    # No skip, no in-flight — the violation should fire first call
    _write_jsonl(p["LAYER2_INVOCATIONS_FILE"], [
        {"ts": _iso(now - timedelta(minutes=22)),
         "reasons": ["ETH-USD consensus BUY"],
         "status": "timeout", "tier": 3},
    ])
    p["LAYER2_JOURNAL_FILE"].parent.mkdir(parents=True, exist_ok=True)
    p["LAYER2_JOURNAL_FILE"].write_text("", encoding="utf-8")

    v1 = loop_contract.check_layer2_journal_activity()
    assert len(v1) == 1
    # Second call — same trigger_time — should be deduped
    v2 = loop_contract.check_layer2_journal_activity()
    assert v2 == []


def test_dedup_marker_survives_violation_tracker_save(contract_env):
    """Codex P1 2026-04-18: ViolationTracker._save() previously rewrote
    CONTRACT_STATE_FILE with only its own 3 keys, wiping the new
    layer2_last_violation_trigger_ts dedup marker on the next cycle.
    This test pins the preserve-unknown-keys behavior."""
    tmp_path, p = contract_env
    now = datetime.now(UTC)
    trigger_ts = now - timedelta(minutes=25)

    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(trigger_ts),
        "last_trigger_reason": "ETH-USD consensus BUY",
        "last_invocation_tier": 3,
    })
    _write_jsonl(p["LAYER2_INVOCATIONS_FILE"], [
        {"ts": _iso(now - timedelta(minutes=22)),
         "reasons": ["x"], "status": "timeout", "tier": 3},
    ])
    p["LAYER2_JOURNAL_FILE"].parent.mkdir(parents=True, exist_ok=True)
    p["LAYER2_JOURNAL_FILE"].write_text("", encoding="utf-8")

    # Fire a violation — writes layer2_last_violation_trigger_ts to state
    v1 = loop_contract.check_layer2_journal_activity()
    assert len(v1) == 1

    # Simulate ViolationTracker doing its own save cycle
    from portfolio.file_utils import load_json
    tracker = loop_contract.ViolationTracker(state_file=p["CONTRACT_STATE_FILE"])
    tracker.update(v1, None)

    # Dedup marker must still be there after the save
    final_state = load_json(p["CONTRACT_STATE_FILE"])
    assert "layer2_last_violation_trigger_ts" in final_state
    assert final_state["layer2_last_violation_trigger_ts"] == _iso(trigger_ts)
    # And the tracker's own keys are also present
    assert "consecutive" in final_state


def test_new_trigger_after_dedup_fires(contract_env):
    """Dedup only applies to the SAME trigger_time. A fresh trigger
    (new timestamp) should fire a new violation even if the prior one
    was for a similar reason."""
    tmp_path, p = contract_env
    now = datetime.now(UTC)
    first_trigger_ts = now - timedelta(minutes=40)

    _write_json(p["CONFIG_FILE"], {"layer2": {"enabled": True}})
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(first_trigger_ts),
        "last_trigger_reason": "ETH-USD consensus BUY",
        "last_invocation_tier": 3,
    })
    _write_jsonl(p["LAYER2_INVOCATIONS_FILE"], [
        {"ts": _iso(now - timedelta(minutes=35)),
         "reasons": ["ETH-USD consensus BUY"],
         "status": "timeout", "tier": 3},
    ])
    p["LAYER2_JOURNAL_FILE"].parent.mkdir(parents=True, exist_ok=True)
    p["LAYER2_JOURNAL_FILE"].write_text("", encoding="utf-8")

    # First fire
    assert len(loop_contract.check_layer2_journal_activity()) == 1

    # A fresh trigger arrives
    second_trigger_ts = now - timedelta(minutes=25)
    _write_json(p["HEALTH_STATE_FILE"], {
        "last_trigger_time": _iso(second_trigger_ts),
        "last_trigger_reason": "ETH-USD consensus BUY",  # same reason
        "last_invocation_tier": 3,
    })
    # Dedup must NOT suppress — trigger_time changed
    assert len(loop_contract.check_layer2_journal_activity()) == 1


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def test_legacy_constant_still_exported(contract_env):
    """``LAYER2_JOURNAL_GRACE_S`` is retained for backward compatibility
    (docs and the 18-min regression guard). The active code path uses
    the per-tier table — this just verifies we didn't remove the name."""
    assert hasattr(loop_contract, "LAYER2_JOURNAL_GRACE_S")
    assert loop_contract.LAYER2_JOURNAL_GRACE_S == 18 * 60
