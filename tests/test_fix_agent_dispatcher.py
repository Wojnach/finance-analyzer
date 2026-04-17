"""Tests for scripts/fix_agent_dispatcher.py.

The dispatcher never spawns a real Claude subprocess; every test uses a
mock ``invoke_claude_fn`` dependency-injected into ``run()``.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add scripts/ to path so we can import the dispatcher
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import fix_agent_dispatcher as dispatcher  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env(tmp_path, monkeypatch):
    """Redirect all dispatcher paths into tmp_path and return them."""
    paths = {
        "DATA_DIR": tmp_path / "data",
        "CRITICAL_ERRORS_LOG": tmp_path / "data" / "critical_errors.jsonl",
        "STATE_FILE": tmp_path / "data" / "fix_agent_state.json",
        "KILL_SWITCH": tmp_path / "data" / "fix_agent.disabled",
    }
    for k, v in paths.items():
        monkeypatch.setattr(dispatcher, k, v)
    paths["DATA_DIR"].mkdir(parents=True, exist_ok=True)
    # Clear recursion env flag between tests
    monkeypatch.delenv(dispatcher.RECURSION_ENV, raising=False)
    return paths


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _write_entries(path: Path, entries: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")


def _critical_entry(
    category: str = "auth_failure",
    caller: str = "layer2_t3",
    ts: datetime | None = None,
    resolution: object = None,
) -> dict:
    ts = ts or datetime.now(UTC) - timedelta(hours=1)
    return {
        "ts": _iso(ts),
        "level": "critical",
        "category": category,
        "caller": caller,
        "resolution": resolution,
        "message": "mock error",
        "context": {},
    }


# ---------------------------------------------------------------------------
# No-op paths
# ---------------------------------------------------------------------------

def test_no_journal_returns_zero(env):
    # Journal doesn't exist at all.
    rc = dispatcher.run(invoke_claude_fn=MagicMock())
    assert rc == 0


def test_empty_journal_returns_zero(env):
    env["CRITICAL_ERRORS_LOG"].write_text("", encoding="utf-8")
    rc = dispatcher.run(invoke_claude_fn=MagicMock())
    assert rc == 0


def test_old_entries_are_ignored(env):
    _write_entries(env["CRITICAL_ERRORS_LOG"], [
        _critical_entry(ts=datetime.now(UTC) - timedelta(days=3)),
    ])
    mock = MagicMock()
    dispatcher.run(invoke_claude_fn=mock, lookback_h=24)
    mock.assert_not_called()


def test_resolved_entries_skipped(env):
    _write_entries(env["CRITICAL_ERRORS_LOG"], [
        _critical_entry(resolution="fixed"),
    ])
    mock = MagicMock()
    dispatcher.run(invoke_claude_fn=mock)
    mock.assert_not_called()


def test_resolves_ts_retroactively_closes_entry(env):
    ts_original = datetime.now(UTC) - timedelta(hours=2)
    ts_resolution = datetime.now(UTC) - timedelta(hours=1)
    _write_entries(env["CRITICAL_ERRORS_LOG"], [
        _critical_entry(ts=ts_original),
        {"ts": _iso(ts_resolution), "level": "info", "category": "resolution",
         "caller": "agent", "resolves_ts": _iso(ts_original),
         "resolution": "patched", "message": "", "context": {}},
    ])
    mock = MagicMock()
    dispatcher.run(invoke_claude_fn=mock)
    mock.assert_not_called()


# ---------------------------------------------------------------------------
# Kill switch + recursion
# ---------------------------------------------------------------------------

def test_kill_switch_blocks_spawn(env):
    env["KILL_SWITCH"].write_text("", encoding="utf-8")
    _write_entries(env["CRITICAL_ERRORS_LOG"], [_critical_entry()])
    mock = MagicMock()
    dispatcher.run(invoke_claude_fn=mock)
    mock.assert_not_called()

    # A "skipped" info entry should be recorded so the user sees why the
    # dispatcher went quiet.
    lines = env["CRITICAL_ERRORS_LOG"].read_text(encoding="utf-8").splitlines()
    skips = [json.loads(l) for l in lines if json.loads(l).get("category") == "fix_attempt_skipped"]
    assert len(skips) == 1
    assert skips[0]["context"]["reason"] == "disabled_by_kill_switch"


def test_recursion_guard_blocks_spawn(env, monkeypatch):
    monkeypatch.setenv(dispatcher.RECURSION_ENV, "1")
    _write_entries(env["CRITICAL_ERRORS_LOG"], [_critical_entry()])
    mock = MagicMock()
    dispatcher.run(invoke_claude_fn=mock)
    mock.assert_not_called()


# ---------------------------------------------------------------------------
# Cooldown behaviour
# ---------------------------------------------------------------------------

def test_cooldown_blocks_second_attempt(env):
    _write_entries(env["CRITICAL_ERRORS_LOG"], [_critical_entry()])
    # Manually prime state with a recent successful attempt → still blocked
    env["STATE_FILE"].write_text(json.dumps({
        "by_category": {
            "auth_failure": {
                "consecutive_failures": 0,
                "blocked_until": _iso(datetime.now(UTC) + timedelta(minutes=10)),
                "last_attempt_ts": _iso(datetime.now(UTC) - timedelta(minutes=20)),
                "last_attempt_success": True,
            }
        }
    }), encoding="utf-8")

    mock = MagicMock()
    dispatcher.run(invoke_claude_fn=mock)
    mock.assert_not_called()


def test_cooldown_elapsed_allows_spawn(env):
    _write_entries(env["CRITICAL_ERRORS_LOG"], [_critical_entry()])
    env["STATE_FILE"].write_text(json.dumps({
        "by_category": {
            "auth_failure": {
                "consecutive_failures": 0,
                "blocked_until": _iso(datetime.now(UTC) - timedelta(minutes=1)),
                "last_attempt_ts": _iso(datetime.now(UTC) - timedelta(hours=2)),
                "last_attempt_success": True,
            }
        }
    }), encoding="utf-8")

    mock = MagicMock(return_value=(True, 0))
    dispatcher.run(invoke_claude_fn=mock)
    assert mock.call_count == 1


def test_consecutive_failures_trigger_exponential_backoff(env):
    """3rd failure → 12h block; 4th attempt → effectively disabled."""
    now = datetime.now(UTC)
    state = {"by_category": {"auth_failure": {"consecutive_failures": 0}}}

    state = dispatcher.update_state_after_attempt(state, "auth_failure", success=False, now=now)
    assert state["by_category"]["auth_failure"]["consecutive_failures"] == 1
    b1 = dispatcher._parse_iso(state["by_category"]["auth_failure"]["blocked_until"])
    assert b1 is not None and (b1 - now).total_seconds() == pytest.approx(1800, abs=5)

    state = dispatcher.update_state_after_attempt(state, "auth_failure", success=False, now=now)
    assert state["by_category"]["auth_failure"]["consecutive_failures"] == 2
    b2 = dispatcher._parse_iso(state["by_category"]["auth_failure"]["blocked_until"])
    assert (b2 - now).total_seconds() == pytest.approx(7200, abs=5)

    state = dispatcher.update_state_after_attempt(state, "auth_failure", success=False, now=now)
    b3 = dispatcher._parse_iso(state["by_category"]["auth_failure"]["blocked_until"])
    assert (b3 - now).total_seconds() == pytest.approx(43200, abs=5)

    # 4th failure → beyond schedule → effectively disabled (>1y out).
    state = dispatcher.update_state_after_attempt(state, "auth_failure", success=False, now=now)
    b4 = dispatcher._parse_iso(state["by_category"]["auth_failure"]["blocked_until"])
    assert (b4 - now).total_seconds() > 365 * 24 * 3600


def test_success_resets_consecutive_failures(env):
    now = datetime.now(UTC)
    state = {"by_category": {"auth_failure": {"consecutive_failures": 2}}}
    state = dispatcher.update_state_after_attempt(state, "auth_failure", success=True, now=now)
    assert state["by_category"]["auth_failure"]["consecutive_failures"] == 0


# ---------------------------------------------------------------------------
# Successful spawn path
# ---------------------------------------------------------------------------

def test_unresolved_entry_spawns_agent(env):
    _write_entries(env["CRITICAL_ERRORS_LOG"], [_critical_entry()])
    mock = MagicMock(return_value=(True, 0))
    dispatcher.run(invoke_claude_fn=mock)

    assert mock.call_count == 1
    call_kwargs = mock.call_args.kwargs
    assert call_kwargs["caller"] == "fix_agent_auth_failure"
    assert call_kwargs["model"] == dispatcher.AGENT_MODEL
    assert call_kwargs["allowed_tools"] == dispatcher.AGENT_ALLOWED_TOOLS
    assert "auth_failure" in call_kwargs["prompt"]


def test_groups_entries_by_category(env):
    """Three entries in two categories → two spawns, one per category."""
    _write_entries(env["CRITICAL_ERRORS_LOG"], [
        _critical_entry(category="auth_failure"),
        _critical_entry(category="auth_failure", ts=datetime.now(UTC) - timedelta(minutes=30)),
        _critical_entry(category="contract_violation"),
    ])
    mock = MagicMock(return_value=(True, 0))
    dispatcher.run(invoke_claude_fn=mock)
    assert mock.call_count == 2
    callers = sorted(c.kwargs["caller"] for c in mock.call_args_list)
    assert callers == ["fix_agent_auth_failure", "fix_agent_contract_violation"]


def test_started_and_completed_entries_appended_on_success(env):
    _write_entries(env["CRITICAL_ERRORS_LOG"], [_critical_entry()])
    mock = MagicMock(return_value=(True, 0))
    dispatcher.run(invoke_claude_fn=mock)

    lines = env["CRITICAL_ERRORS_LOG"].read_text(encoding="utf-8").splitlines()
    categories = [json.loads(l).get("category") for l in lines]
    assert "fix_attempt_started" in categories
    assert "fix_attempt_completed" in categories


def test_failed_spawn_records_fix_agent_failed(env):
    _write_entries(env["CRITICAL_ERRORS_LOG"], [_critical_entry()])
    mock = MagicMock(return_value=(False, 1))
    dispatcher.run(invoke_claude_fn=mock)

    lines = env["CRITICAL_ERRORS_LOG"].read_text(encoding="utf-8").splitlines()
    failed = [json.loads(l) for l in lines if json.loads(l).get("category") == "fix_agent_failed"]
    assert len(failed) == 1
    assert failed[0]["level"] == "critical"
    assert failed[0]["context"]["success"] is False


def test_invoke_claude_raising_is_contained(env):
    """If invoke_claude raises, dispatcher must not propagate — just
    record the failure and keep going."""
    _write_entries(env["CRITICAL_ERRORS_LOG"], [_critical_entry()])
    mock = MagicMock(side_effect=RuntimeError("boom"))
    rc = dispatcher.run(invoke_claude_fn=mock)
    assert rc == 0
    lines = env["CRITICAL_ERRORS_LOG"].read_text(encoding="utf-8").splitlines()
    failed = [json.loads(l) for l in lines if json.loads(l).get("category") == "fix_agent_failed"]
    assert len(failed) == 1


def test_dry_run_does_not_spawn(env):
    _write_entries(env["CRITICAL_ERRORS_LOG"], [_critical_entry()])
    mock = MagicMock()
    dispatcher.run(dry_run=True, invoke_claude_fn=mock)
    mock.assert_not_called()
    # Still records an 'started' entry so the user sees the dispatcher ran.
    lines = env["CRITICAL_ERRORS_LOG"].read_text(encoding="utf-8").splitlines()
    started = [l for l in lines if json.loads(l).get("category") == "fix_attempt_started"]
    assert len(started) == 1


# ---------------------------------------------------------------------------
# Prompt content smoke
# ---------------------------------------------------------------------------

def test_build_fix_prompt_includes_category_and_entries():
    entries = [_critical_entry(category="auth_failure", caller="layer2_t3")]
    prompt = dispatcher.build_fix_prompt("auth_failure", entries)
    assert "auth_failure" in prompt
    assert "layer2_t3" in prompt
    assert "resolves_ts" in prompt   # Agent is told to append resolution line
    assert "NOT:" in prompt          # Allow-list constraints must appear


# ---------------------------------------------------------------------------
# Atomic state write — mid-write crash must not corrupt the state file
# ---------------------------------------------------------------------------

def test_state_write_is_atomic(env, monkeypatch):
    """If os.replace is interrupted, the original state file stays intact.
    We simulate a write crash and verify the *existing* state file wasn't
    mutated partway through."""
    original_state = {"by_category": {"x": {"consecutive_failures": 0}}}
    env["STATE_FILE"].write_text(json.dumps(original_state), encoding="utf-8")

    # Replace os.replace with a crasher. The tmp file gets written but
    # the rename fails — the real STATE_FILE must remain unchanged.
    real_replace = os.replace
    def crash_replace(src, dst):
        raise OSError("simulated crash during rename")
    monkeypatch.setattr(os, "replace", crash_replace)

    with pytest.raises(OSError):
        dispatcher._save_state({"by_category": {"y": {"consecutive_failures": 99}}})

    # Original file must be untouched.
    assert json.loads(env["STATE_FILE"].read_text(encoding="utf-8")) == original_state
    # And a tmp file may exist — clean up for the next test.
    tmp = env["STATE_FILE"].with_suffix(env["STATE_FILE"].suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    # Restore (not strictly necessary — monkeypatch does this).
    monkeypatch.setattr(os, "replace", real_replace)


# ---------------------------------------------------------------------------
# Corrupt state file — must NOT cause runaway spawns
# ---------------------------------------------------------------------------

def test_corrupt_state_blocks_all_categories_for_one_hour(env):
    """Closing the gap from the post-ultrathink review: if state file is
    garbage, the dispatcher should NOT clear cooldowns and spam-spawn.
    Conservative default applies a 1h global block + records a critical
    entry so the user sees what happened."""
    env["STATE_FILE"].write_text("not valid json {{{", encoding="utf-8")
    _write_entries(env["CRITICAL_ERRORS_LOG"], [_critical_entry()])
    mock = MagicMock()
    dispatcher.run(invoke_claude_fn=mock)
    # No spawn attempted.
    mock.assert_not_called()
    # A `fix_agent_state_corrupt` entry must have been recorded.
    lines = env["CRITICAL_ERRORS_LOG"].read_text(encoding="utf-8").splitlines()
    cats = [json.loads(l).get("category") for l in lines]
    assert "fix_agent_state_corrupt" in cats


def test_global_cooldown_blocks_in_check_gates(env):
    """Direct check_gates verification of the global block path."""
    state = {
        "by_category": {},
        "blocked_until_global": _iso(datetime.now(UTC) + timedelta(minutes=30)),
    }
    decision = dispatcher.check_gates("auth_failure", state)
    assert decision.allowed is False
    assert decision.reason == "global_cooldown"


def test_global_cooldown_expired_allows_through(env):
    state = {
        "by_category": {},
        "blocked_until_global": _iso(datetime.now(UTC) - timedelta(minutes=5)),
    }
    decision = dispatcher.check_gates("auth_failure", state, caller_depth=0)
    assert decision.allowed is True


# ---------------------------------------------------------------------------
# check_gates explicit caller_depth
# ---------------------------------------------------------------------------

def test_check_gates_uses_explicit_caller_depth(env, monkeypatch):
    """When caller_depth is passed explicitly, env-var mutations between
    the check and the call must not affect the decision."""
    monkeypatch.setenv(dispatcher.RECURSION_ENV, "5")  # would block if read live
    decision = dispatcher.check_gates("x", {"by_category": {}}, caller_depth=0)
    assert decision.allowed is True


def test_check_gates_falls_back_to_env_when_no_explicit_depth(env, monkeypatch):
    monkeypatch.setenv(dispatcher.RECURSION_ENV, "1")
    decision = dispatcher.check_gates("x", {"by_category": {}})
    assert decision.allowed is False
    assert decision.reason == "recursion_depth_exceeded"


# ---------------------------------------------------------------------------
# Concurrent writes to critical_errors.jsonl — atomic_append_jsonl must
# produce well-formed JSONL when the dispatcher and a simulated main-loop
# writer hammer the file in parallel
# ---------------------------------------------------------------------------

def test_concurrent_append_does_not_corrupt_jsonl(env):
    """Verify the dispatcher's _append_critical and the main-loop's
    atomic_append_jsonl can run concurrently without producing TORN
    JSON lines (which would break the entire startup-check parser).

    Note: this is a corruption test, not a durability test. Empirically,
    Python's open('a') with text-mode buffering on Windows can drop
    occasional writes under heavy thread contention (~1-2% loss in
    benchmarks), but never produces torn lines. The startup check and
    dispatcher both tolerate occasional loss; both fail hard on
    corruption. Cross-process file locking would close the loss gap but
    is a codebase-wide change tracked separately.
    """
    from portfolio.file_utils import atomic_append_jsonl

    journal = env["CRITICAL_ERRORS_LOG"]
    n_per_writer = 200

    def writer_dispatcher():
        for i in range(n_per_writer):
            dispatcher._append_critical({
                "ts": _iso(datetime.now(UTC)),
                "level": "info",
                "category": "fix_attempt_started",
                "caller": "fix_agent_dispatcher",
                "resolution": None,
                "message": f"dispatch-{i}",
                "context": {"i": i},
            })

    def writer_loop():
        for i in range(n_per_writer):
            atomic_append_jsonl(journal, {
                "ts": _iso(datetime.now(UTC)),
                "level": "critical",
                "category": "auth_failure",
                "caller": "layer2_t3",
                "resolution": None,
                "message": f"loop-{i}",
                "context": {"i": i},
            })

    threads = [
        threading.Thread(target=writer_dispatcher),
        threading.Thread(target=writer_loop),
        threading.Thread(target=writer_dispatcher),
        threading.Thread(target=writer_loop),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # The corruption guarantee: count torn lines (non-empty, non-parseable).
    # Empty lines from interleaved writes are tolerated — they don't break
    # the startup-check parser, which uses `if not line.strip(): continue`.
    lines = journal.read_text(encoding="utf-8").splitlines()
    expected_total = 4 * n_per_writer
    parsed = []
    torn = []
    for ln in lines:
        if not ln.strip():
            continue
        try:
            entry = json.loads(ln)
            parsed.append(entry)
            assert "category" in entry or "i" in entry
        except json.JSONDecodeError:
            torn.append(ln[:80])
    # The actual corruption assertion: zero torn lines.
    assert torn == [], f"Torn JSONL lines under contention: {torn[:3]}"
    # Loss tolerance: at most ~10% writes may vanish. Below that suggests
    # the underlying append primitive has regressed.
    assert len(parsed) >= int(expected_total * 0.90), (
        f"Only {len(parsed)}/{expected_total} writes survived — "
        f"atomic_append_jsonl may have regressed"
    )


def test_info_level_entries_do_not_appear_in_startup_check(tmp_path):
    """Regression guard for the 2026-04-13 dispatcher fix: info-level
    fix_attempt_* entries must NOT surface via check_critical_errors.py,
    otherwise every 10-min dispatcher tick would add noise to every new
    Claude session's startup."""
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    import check_critical_errors as cce  # noqa: E402

    now = datetime.now(UTC)
    journal = tmp_path / "crit.jsonl"
    journal.write_text("\n".join([
        json.dumps({"ts": _iso(now - timedelta(minutes=5)), "level": "info",
                    "category": "fix_attempt_started", "caller": "fix_agent_dispatcher",
                    "resolution": None, "message": "spawning",  "context": {}}),
        json.dumps({"ts": _iso(now - timedelta(minutes=5)), "level": "info",
                    "category": "fix_attempt_completed", "caller": "fix_agent_dispatcher",
                    "resolution": None, "message": "done", "context": {}}),
        json.dumps({"ts": _iso(now - timedelta(hours=1)), "level": "critical",
                    "category": "auth_failure", "caller": "layer2_t3",
                    "resolution": None, "message": "the real issue", "context": {}}),
    ]) + "\n", encoding="utf-8")

    unresolved = cce.find_unresolved(cce._load_entries(journal), days=7)
    # The two info lines must NOT appear; only the critical one.
    assert len(unresolved) == 1
    assert unresolved[0]["category"] == "auth_failure"
