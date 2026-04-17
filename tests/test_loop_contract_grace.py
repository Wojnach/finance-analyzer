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
    _write_jsonl(p["CLAUDE_INVOCATIONS_FILE"], [
        # Prior invocation timed out (terminal state).
        {"timestamp": _iso(now - timedelta(minutes=20)),
         "caller": "layer2_t3", "status": "timeout", "tier": 3},
        # New one is in-flight — don't alert while it works.
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
    _write_jsonl(p["CLAUDE_INVOCATIONS_FILE"], [
        {"timestamp": _iso(invoked_ts),
         "caller": "layer2_t3", "status": "invoked", "tier": 3},
    ])
    p["LAYER2_JOURNAL_FILE"].parent.mkdir(parents=True, exist_ok=True)
    p["LAYER2_JOURNAL_FILE"].write_text("", encoding="utf-8")

    violations = loop_contract.check_layer2_journal_activity()
    assert len(violations) == 1


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def test_legacy_constant_still_exported(contract_env):
    """``LAYER2_JOURNAL_GRACE_S`` is retained for backward compatibility
    (docs and the 18-min regression guard). The active code path uses
    the per-tier table — this just verifies we didn't remove the name."""
    assert hasattr(loop_contract, "LAYER2_JOURNAL_GRACE_S")
    assert loop_contract.LAYER2_JOURNAL_GRACE_S == 18 * 60
