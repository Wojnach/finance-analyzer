"""Tests for the Layer 2 completion-detection watchdog.

Covers the 2026-05-05 fix for item (3a) of the dashboard-noise follow-ups:
without the watchdog, ``check_agent_completion`` ran only at the start of
each ``main.run()`` cycle, so when the cycle bloated to 480-540s the
subprocess completion (and the wall-clock timeout kill) was detected up
to 8 minutes late. See ``docs/plans/2026-05-05-l2-completion-watchdog.md``.

Test scope:
  - watchdog is started by ``try_invoke_agent`` on a successful spawn
  - watchdog tick is a no-op when no agent is running
  - main + watchdog calling ``check_agent_completion`` concurrently
    log exactly one row to invocations.jsonl
  - the lock-protected timeout-kill path still fires correctly
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import portfolio.agent_invocation as ai


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_agent_globals():
    """Reset every test so the watchdog/lock state is fresh.

    Critical for xdist parallel runs: a stuck watchdog from one test
    must not bleed into another test in the same worker.
    """
    ai._stop_completion_watchdog(timeout_s=2.0)
    ai._agent_proc = None
    ai._agent_log = None
    ai._agent_start = 0
    ai._agent_start_wall = 0.0
    ai._agent_timeout = 900
    ai._agent_tier = None
    ai._agent_reasons = None
    ai._journal_ts_before = None
    ai._telegram_ts_before = None
    yield
    ai._stop_completion_watchdog(timeout_s=2.0)
    ai._agent_proc = None
    ai._agent_log = None
    ai._agent_start = 0
    ai._agent_start_wall = 0.0
    ai._agent_timeout = 900
    ai._agent_tier = None
    ai._agent_reasons = None
    ai._journal_ts_before = None
    ai._telegram_ts_before = None


@pytest.fixture
def fast_watchdog(monkeypatch):
    """Shrink the watchdog interval so tests don't sleep 30s each."""
    monkeypatch.setattr(ai, "_COMPLETION_WATCHDOG_INTERVAL_S", 0.05)


# ---------------------------------------------------------------------------
# (a) Watchdog lifecycle
# ---------------------------------------------------------------------------


def test_ensure_watchdog_starts_idempotently(fast_watchdog):
    assert ai._watchdog_thread is None

    ai._ensure_completion_watchdog()
    first = ai._watchdog_thread
    assert first is not None
    assert first.is_alive()
    assert first.daemon is True

    # Second call is a no-op while the thread is alive.
    ai._ensure_completion_watchdog()
    assert ai._watchdog_thread is first


def test_ensure_watchdog_replaces_dead_thread(fast_watchdog):
    """If the thread died for any reason, _ensure must spawn a fresh one."""
    ai._ensure_completion_watchdog()
    first = ai._watchdog_thread
    assert first is not None and first.is_alive()

    # Simulate the prior thread having exited cleanly.
    ai._stop_completion_watchdog(timeout_s=1.0)
    assert ai._watchdog_thread is None

    ai._ensure_completion_watchdog()
    second = ai._watchdog_thread
    assert second is not None and second.is_alive()
    assert second is not first


# ---------------------------------------------------------------------------
# (b) Watchdog tick is harmless when no agent is running
# ---------------------------------------------------------------------------


def test_watchdog_tick_is_noop_with_no_agent(fast_watchdog, tmp_path, monkeypatch):
    """The watchdog must not write to invocations.jsonl when idle."""
    inv_path = tmp_path / "invocations.jsonl"
    monkeypatch.setattr(ai, "INVOCATIONS_FILE", inv_path)

    assert ai._agent_proc is None
    ai._ensure_completion_watchdog()

    # Wait long enough for several ticks to fire.
    time.sleep(0.3)

    assert not inv_path.exists() or inv_path.read_text() == ""


# ---------------------------------------------------------------------------
# (c) Concurrent main + watchdog detect completion exactly once
# ---------------------------------------------------------------------------


def _make_completed_proc(exit_code=0):
    """Return a Popen-like mock whose poll() reports a finished process."""
    proc = MagicMock()
    proc.poll.return_value = exit_code
    proc.pid = 4242
    return proc


def _seed_running_agent(tmp_path, monkeypatch, *, elapsed_s=10.0,
                       timeout_s=120, tier=1):
    """Set module globals to simulate an agent that just spawned."""
    journal = tmp_path / "layer2_journal.jsonl"
    telegram = tmp_path / "telegram_messages.jsonl"
    inv = tmp_path / "invocations.jsonl"
    journal.touch()
    telegram.touch()
    monkeypatch.setattr(ai, "JOURNAL_FILE", journal)
    monkeypatch.setattr(ai, "TELEGRAM_FILE", telegram)
    monkeypatch.setattr(ai, "INVOCATIONS_FILE", inv)
    # Bypass the post-completion side-effects we don't care about here.
    monkeypatch.setattr(ai, "_record_new_trades", lambda: None)
    monkeypatch.setattr(ai, "_scan_agent_log_for_auth_failure",
                       lambda *a, **kw: False)
    ai._agent_proc = _make_completed_proc(exit_code=0)
    ai._agent_log = None  # already-closed handle path is logger.warning only
    ai._agent_start = time.monotonic() - elapsed_s
    ai._agent_start_wall = time.time() - elapsed_s
    ai._agent_timeout = timeout_s
    ai._agent_tier = tier
    ai._agent_reasons = ["test-trigger"]
    ai._journal_ts_before = None
    ai._telegram_ts_before = None
    return inv


def test_concurrent_check_does_not_double_log(tmp_path, monkeypatch):
    """When main and watchdog race, exactly one invocations.jsonl row lands.

    Reproduces the worry behind the lock: two threads observing
    ``_agent_proc != None``, both calling poll, both writing the
    completion row. The lock must serialise them so the second call
    sees ``_agent_proc = None`` (cleared at the end of the handler) and
    returns ``None`` without logging.

    To make the lock contention real (rather than relying on GIL
    serialisation, which would also pass even WITHOUT the lock — see
    review feedback P2-4), we monkeypatch the auth-failure scan to
    sleep 100 ms while the lock is held. That guarantees the second
    thread reaches `with _completion_lock` while the first is still
    inside the locked body. Without the lock the first would clear
    ``_agent_proc`` only after the sleep, so the second would either
    re-poll a stale proc and double-log, or — more likely — both
    callers would write before either reaches the cleanup. With the
    lock the second thread blocks until the first finishes cleanup
    and clears ``_agent_proc``, and then returns ``None``.
    """
    inv_path = _seed_running_agent(tmp_path, monkeypatch)
    monkeypatch.setattr(
        ai, "_scan_agent_log_for_auth_failure",
        lambda *a, **kw: (time.sleep(0.1), False)[1],
    )

    barrier = threading.Barrier(2)
    results = {}

    def caller(name):
        barrier.wait()
        results[name] = ai.check_agent_completion()

    t1 = threading.Thread(target=caller, args=("main",))
    t2 = threading.Thread(target=caller, args=("watchdog",))
    t1.start(); t2.start()
    t1.join(); t2.join()

    # Exactly one of the two callers got the dict; the other got None.
    got_dict = sum(1 for v in results.values() if v is not None)
    assert got_dict == 1, f"expected 1 winner, got results={results}"

    # Exactly one row in invocations.jsonl.
    rows = [json.loads(ln) for ln in inv_path.read_text().splitlines() if ln]
    assert len(rows) == 1, f"expected 1 row, got {rows}"
    assert rows[0]["status"] in ("success", "incomplete")
    assert rows[0]["tier"] == 1


def test_concurrent_check_without_lock_would_double_log(tmp_path, monkeypatch):
    """Negative regression guard: with the lock bypassed, the same setup
    DOES double-log. If this test starts passing without changes, the
    lock wrap on ``check_agent_completion`` was silently removed and
    the previous test became reassurance theatre. See review P2-4.

    We swap ``_completion_lock`` for a no-op context manager so the
    test exercises the unprotected path. A 100 ms scan delay forces
    real overlap. Both callers therefore proceed to the file-write
    section, producing two rows.
    """
    inv_path = _seed_running_agent(tmp_path, monkeypatch)
    monkeypatch.setattr(
        ai, "_scan_agent_log_for_auth_failure",
        lambda *a, **kw: (time.sleep(0.1), False)[1],
    )

    # Replace the real lock with a no-op so the locked path becomes
    # equivalent to the pre-fix unprotected path.
    class _NoopLock:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(ai, "_completion_lock", _NoopLock())

    barrier = threading.Barrier(2)
    results = {}

    def caller(name):
        barrier.wait()
        results[name] = ai.check_agent_completion()

    t1 = threading.Thread(target=caller, args=("main",))
    t2 = threading.Thread(target=caller, args=("watchdog",))
    t1.start(); t2.start()
    t1.join(); t2.join()

    rows = [json.loads(ln) for ln in inv_path.read_text().splitlines() if ln]
    # Without the lock both callers reach the file-write section; the
    # exact count is timing-dependent but must be > 1 for the lock to
    # be load-bearing.
    assert len(rows) > 1, (
        f"expected >1 rows when lock bypassed, got {len(rows)}; lock no "
        f"longer protects double-log. results={results}"
    )


# ---------------------------------------------------------------------------
# (d) Timeout kill via the locked path
# ---------------------------------------------------------------------------


def test_timeout_kill_fires_through_locked_path(tmp_path, monkeypatch):
    """A still-running subprocess past _agent_timeout returns 'timeout'.

    Sets up the same mocked-process state as the concurrent test but
    with poll() returning None (still running) and elapsed > timeout.
    The locked check_agent_completion must invoke the kill helper and
    return a 'timeout' dict.
    """
    inv = tmp_path / "invocations.jsonl"
    monkeypatch.setattr(ai, "INVOCATIONS_FILE", inv)

    proc = MagicMock()
    proc.poll.return_value = None  # still running
    proc.pid = 1234

    kill_calls = []
    monkeypatch.setattr(ai, "_kill_overrun_agent",
                       lambda: kill_calls.append(time.time()))

    ai._agent_proc = proc
    ai._agent_start = time.monotonic() - 200.0  # 200s elapsed
    ai._agent_start_wall = time.time() - 200.0
    ai._agent_timeout = 120  # T1 budget
    ai._agent_tier = 1
    ai._agent_reasons = ["test"]

    result = ai.check_agent_completion()
    assert result is not None
    assert result["status"] == "timeout"
    assert result["tier"] == 1
    assert result["duration_s"] >= 120
    assert len(kill_calls) == 1
