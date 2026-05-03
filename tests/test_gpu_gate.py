"""Tests for portfolio/gpu_gate.py.

Covers:
- BUG-182: _pid_alive helper for stale lock validation
- 2026-05-03: stale-lock background sweeper. Background — chronos pid 13152
  died holding the file lock 2026-05-02 02:14 and the loop wedged 25 hours
  because _is_stale() was only checked inside the acquire retry loop. The
  sweeper closes that liveness hole. See
  docs/plans/2026-05-03-gpu-gate-sweeper.md.
"""

import os
import time
from pathlib import Path

import pytest

from portfolio import gpu_gate as gpu_gate_mod
from portfolio.gpu_gate import _pid_alive


class TestPidAlive:
    """BUG-182: Verify PID validation before breaking stale locks."""

    def test_current_process_alive(self):
        """Current process should always be alive."""
        assert _pid_alive(os.getpid()) is True

    def test_zero_pid_not_alive(self):
        """PID 0 is not a valid process."""
        assert _pid_alive(0) is False

    def test_nonexistent_pid_not_alive(self):
        """Very high PID should not exist."""
        assert _pid_alive(999999999) is False

    def test_negative_pid_not_alive(self):
        """Negative PID is invalid."""
        assert _pid_alive(-1) is False


# ---------------------------------------------------------------------------
# 2026-05-03: stale-lock sweeper
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_lock(tmp_path, monkeypatch):
    """Redirect _GPU_LOCK_FILE to tmp_path for xdist safety.

    The module reads _GPU_LOCK_FILE at call time (not at import), so a
    monkeypatch on the module attribute is sufficient — no test sees the
    real Q:/models/.gpu_lock.
    """
    lock_file = tmp_path / ".gpu_lock"
    monkeypatch.setattr(gpu_gate_mod, "_GPU_LOCK_FILE", lock_file)
    return lock_file


def _write_lock_raw(path: Path, model: str, pid: int, ts: float, tid: int = 1234):
    path.write_text(f"{model}|{pid}|{ts}|{tid}", encoding="utf-8")


class TestTryBreakStaleLock:
    """Direct unit tests on the helper used by both reactive and sweeper paths."""

    def test_reaps_stale_lock_with_dead_pid(self, isolated_lock, caplog):
        """Stale-by-mtime AND dead-pid → lock removed + warning logged."""
        _write_lock_raw(isolated_lock, "kronos", 999999999, time.time() - 600)
        os.utime(isolated_lock, (time.time() - 600, time.time() - 600))

        with caplog.at_level("WARNING", logger="portfolio.gpu_gate"):
            reaped = gpu_gate_mod._try_break_stale_lock()

        assert reaped is True
        assert not isolated_lock.exists(), "stale lock with dead pid should be reaped"
        assert any("Breaking stale GPU lock" in rec.message for rec in caplog.records), \
            "must emit Breaking stale GPU lock warning so log-grep tools keep working"

    def test_keeps_lock_with_live_pid(self, isolated_lock):
        """Stale-by-mtime but pid still alive → lock kept (process may be busy)."""
        _write_lock_raw(isolated_lock, "kronos", os.getpid(), time.time() - 600)
        os.utime(isolated_lock, (time.time() - 600, time.time() - 600))

        reaped = gpu_gate_mod._try_break_stale_lock()

        assert reaped is False
        assert isolated_lock.exists(), "live-pid lock must NOT be reaped"

    def test_keeps_fresh_lock(self, isolated_lock):
        """Fresh mtime (within _STALE_SECONDS) → lock kept regardless of pid."""
        _write_lock_raw(isolated_lock, "kronos", 999999999, time.time())

        reaped = gpu_gate_mod._try_break_stale_lock()

        assert reaped is False
        assert isolated_lock.exists(), "fresh lock must NOT be reaped"

    def test_handles_missing_lock_silently(self, isolated_lock):
        """No lock file present → no-op, no exception."""
        assert not isolated_lock.exists()

        reaped = gpu_gate_mod._try_break_stale_lock()

        assert reaped is False

    def test_handles_malformed_lock_silently(self, isolated_lock):
        """Malformed lock content → no-op (defensive)."""
        isolated_lock.write_text("not-a-valid-lock-format", encoding="utf-8")
        os.utime(isolated_lock, (time.time() - 600, time.time() - 600))

        # Should not raise. The daemon thread must never crash on bad input.
        gpu_gate_mod._try_break_stale_lock()


class TestSweeperLifecycle:
    """The daemon registration is idempotent and uses a daemon thread."""

    def test_start_sweeper_is_idempotent(self):
        """Calling _start_sweeper twice spawns at most one daemon."""
        with gpu_gate_mod._SWEEPER_LOCK:
            gpu_gate_mod._sweeper_thread = None
        gpu_gate_mod._start_sweeper()
        first = gpu_gate_mod._sweeper_thread
        gpu_gate_mod._start_sweeper()
        second = gpu_gate_mod._sweeper_thread
        assert first is not None, "first call must spawn the daemon"
        assert first is second, "second call must NOT spawn a new daemon"

    def test_sweeper_thread_is_daemon(self):
        """Thread must be daemon=True so it dies with the process."""
        with gpu_gate_mod._SWEEPER_LOCK:
            gpu_gate_mod._sweeper_thread = None
        gpu_gate_mod._start_sweeper()
        t = gpu_gate_mod._sweeper_thread
        assert t is not None
        assert t.daemon is True, "sweeper must be daemon=True (no shutdown hook needed)"

    def test_sweeper_started_lazily_by_gpu_gate(self, isolated_lock, monkeypatch):
        """First gpu_gate() call spawns the sweeper; subsequent calls reuse it."""
        with gpu_gate_mod._SWEEPER_LOCK:
            gpu_gate_mod._sweeper_thread = None

        # nvidia-smi may not exist in test env; stub the VRAM probe.
        monkeypatch.setattr(gpu_gate_mod, "get_vram_usage", lambda: None)

        with gpu_gate_mod.gpu_gate("test-model", timeout=2) as acquired:
            assert acquired is True

        assert gpu_gate_mod._sweeper_thread is not None, \
            "first gpu_gate() call should have spawned the sweeper"
