"""Tests for portfolio.claude_gate — A-IN-2 process tree kill on timeout.

Focus on the tree-kill helpers (_kill_process_tree, _run_with_tree_kill).
We do NOT spawn the real claude CLI — we use a tiny Python child that
sleeps long enough to time out. This keeps the test deterministic and
fast across platforms.
"""

import os
import platform
import subprocess
import sys
import textwrap
import time

import pytest

from portfolio import claude_gate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _python_sleep_script(seconds: float) -> str:
    """A standalone Python script that sleeps for `seconds` then exits 0.
    Used as a subprocess we can race against the timeout."""
    return textwrap.dedent(f"""
        import time, sys
        time.sleep({seconds})
        sys.exit(0)
    """).strip()


def _python_spawn_child_script(child_seconds: float) -> str:
    """Python script that spawns a child and waits for it. We use this to
    verify tree-kill: when the parent is killed, the grandchild must also
    die. We write the grandchild's PID to a temp file so the test can
    verify it's gone."""
    return textwrap.dedent(f"""
        import os, subprocess, sys, time, tempfile
        pidfile = sys.argv[1]
        # Spawn a sleeping grandchild
        child = subprocess.Popen([sys.executable, '-c',
            'import time; time.sleep({child_seconds})'])
        with open(pidfile, 'w') as f:
            f.write(str(child.pid))
        # Wait so the parent stays alive while the test races us
        time.sleep({child_seconds})
        sys.exit(0)
    """).strip()


def _process_alive(pid: int) -> bool:
    """Check whether a PID is still running. Cross-platform."""
    try:
        if platform.system() == "Windows":
            res = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True, text=True, timeout=5,
            )
            return str(pid) in res.stdout
        else:
            os.kill(pid, 0)
            return True
    except (OSError, subprocess.SubprocessError):
        return False


# ---------------------------------------------------------------------------
# _run_with_tree_kill — direct child timeout
# ---------------------------------------------------------------------------

class TestRunWithTreeKillDirectChild:
    def test_normal_completion_returns_returncode(self, tmp_path):
        """A short-running command should return exit_code=0, no timeout."""
        rc, stdout, stderr, timed_out = claude_gate._run_with_tree_kill(
            [sys.executable, "-c", "print('hello'); import sys; sys.exit(0)"],
            timeout=10.0,
            env=os.environ.copy(),
            cwd=str(tmp_path),
            label="test",
        )
        assert timed_out is False
        assert rc == 0
        assert "hello" in stdout

    def test_nonzero_exit_code_passed_through(self, tmp_path):
        rc, _, _, timed_out = claude_gate._run_with_tree_kill(
            [sys.executable, "-c", "import sys; sys.exit(7)"],
            timeout=10.0,
            env=os.environ.copy(),
            cwd=str(tmp_path),
            label="test",
        )
        assert timed_out is False
        assert rc == 7

    def test_timeout_kills_child_and_returns_timed_out(self, tmp_path):
        """A child that sleeps longer than the timeout must be killed
        and the function must return timed_out=True quickly."""
        t0 = time.time()
        rc, _, _, timed_out = claude_gate._run_with_tree_kill(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=1.0,  # 1 second
            env=os.environ.copy(),
            cwd=str(tmp_path),
            label="test",
        )
        elapsed = time.time() - t0
        assert timed_out is True
        assert rc == -1
        # Must return promptly (kill+drain budget is ~5s); 10s is generous.
        assert elapsed < 10, f"Tree-kill took too long: {elapsed:.1f}s (expected <10s)"


# ---------------------------------------------------------------------------
# _run_with_tree_kill — grandchild propagation
# ---------------------------------------------------------------------------

class TestTreeKillReachesGrandchildren:
    """A-IN-2 invariant: when claude is killed, its grandchildren (the
    actual claude API helper, MCP servers, local-LLM helpers, etc.) must
    ALSO be killed. Plain proc.kill() leaves them as zombies."""

    def test_grandchild_dies_when_parent_killed(self, tmp_path):
        """Spawn a parent that spawns a grandchild, then time out the parent.
        The grandchild PID (written to a file by the parent) must NOT be
        running after _run_with_tree_kill returns."""
        pidfile = tmp_path / "grandchild.pid"
        script = _python_spawn_child_script(child_seconds=30)

        rc, _, _, timed_out = claude_gate._run_with_tree_kill(
            [sys.executable, "-c", script, str(pidfile)],
            timeout=1.5,  # parent sleeps 30s, will time out
            env=os.environ.copy(),
            cwd=str(tmp_path),
            label="treekill-test",
        )

        assert timed_out is True

        # Give the OS a moment to reap the killed grandchild.
        time.sleep(0.5)

        if not pidfile.exists():
            pytest.skip("Parent script didn't write pidfile in time — test is racy on this platform")

        grandchild_pid = int(pidfile.read_text().strip())
        assert not _process_alive(grandchild_pid), (
            f"Grandchild PID {grandchild_pid} is still alive after tree-kill — "
            "_kill_process_tree did not propagate to descendants"
        )


# ---------------------------------------------------------------------------
# _popen_kwargs_for_tree_kill — platform behavior
# ---------------------------------------------------------------------------

class TestPopenKwargs:
    def test_returns_dict(self):
        kwargs = claude_gate._popen_kwargs_for_tree_kill()
        assert isinstance(kwargs, dict)

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only")
    def test_windows_uses_create_new_process_group(self):
        kwargs = claude_gate._popen_kwargs_for_tree_kill()
        assert "creationflags" in kwargs
        # CREATE_NEW_PROCESS_GROUP = 0x00000200
        assert kwargs["creationflags"] & 0x200 == 0x200

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix-only")
    def test_unix_uses_start_new_session(self):
        kwargs = claude_gate._popen_kwargs_for_tree_kill()
        assert kwargs.get("start_new_session") is True


# ---------------------------------------------------------------------------
# _kill_process_tree — already-exited process is no-op
# ---------------------------------------------------------------------------

class TestKillProcessTreeIdempotent:
    def test_already_exited_process_is_noop(self, tmp_path):
        """Calling _kill_process_tree on a process that already exited
        must NOT raise."""
        proc = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.exit(0)"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        proc.wait()
        assert proc.poll() is not None  # confirmed exited
        # This must not raise.
        claude_gate._kill_process_tree(proc, label="exited-test")


class TestInvokeLockSerialization:
    """A-IN-3 (2026-04-11): in-process invocations must serialize through
    _invoke_lock so the 8-worker ticker pool can't spawn 5 concurrent
    Claude processes."""

    def test_invoke_lock_exists_and_is_a_lock(self):
        import threading
        # Should be a Lock or RLock object
        assert hasattr(claude_gate, "_invoke_lock")
        # It should support context-manager protocol
        with claude_gate._invoke_lock:
            pass

    def test_concurrent_invokes_serialize(self, monkeypatch, tmp_path):
        """Spawn 5 threads that all call invoke_claude. Mock the
        _run_with_tree_kill helper to track concurrency. Max concurrent
        executions must be exactly 1 — the lock serializes them."""
        import threading
        import time

        # Block the kill switch / config / claude lookup so we always
        # reach the locked subprocess section.
        monkeypatch.setattr(claude_gate, "CLAUDE_ENABLED", True)
        monkeypatch.setattr(claude_gate, "_load_config_layer2_enabled", lambda: True)
        monkeypatch.setattr(claude_gate, "_find_claude_cmd", lambda: "/fake/claude")
        # Redirect log writes to a temp dir so they don't poison the real log.
        monkeypatch.setattr(claude_gate, "INVOCATIONS_LOG", tmp_path / "log.jsonl")
        monkeypatch.setattr(claude_gate, "_count_today_invocations", lambda: 0)

        active = [0]
        max_active = [0]
        active_lock = threading.Lock()

        def fake_run_with_tree_kill(*args, **kwargs):
            with active_lock:
                active[0] += 1
                if active[0] > max_active[0]:
                    max_active[0] = active[0]
            time.sleep(0.05)  # hold the "claude process" briefly
            with active_lock:
                active[0] -= 1
            return (0, "ok", "", False)

        monkeypatch.setattr(claude_gate, "_run_with_tree_kill", fake_run_with_tree_kill)

        results = []
        def worker(i):
            ok, code = claude_gate.invoke_claude(
                prompt=f"prompt {i}",
                caller=f"test-{i}",
                timeout=1,
            )
            results.append((ok, code))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert max_active[0] == 1, (
            f"Expected serialized invocations (max=1), got max={max_active[0]} "
            "concurrent _run_with_tree_kill calls — _invoke_lock not held"
        )
        assert len(results) == 5
        assert all(ok for ok, _ in results)
