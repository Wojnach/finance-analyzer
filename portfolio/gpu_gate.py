"""GPU gating and VRAM monitoring for LLM model inference.

Ensures only one LLM model uses the GPU at a time. Logs VRAM usage
before and after each model load for monitoring.

Uses a threading lock for in-process concurrency (ThreadPoolExecutor workers)
plus a file-based lock at Q:/models/.gpu_lock for cross-process protection.

Stale-lock recovery (2026-05-03):
- Reactive: ``gpu_gate()`` calls ``_try_break_stale_lock()`` when another
  caller blocks on the lock — same predicate as before BUG-182.
- Background: a daemon thread (lazily spawned on first ``gpu_gate()`` call)
  runs the same predicate every 30 s. This closes the liveness hole that
  let the loop wedge for ~25 hours after chronos pid 13152 died holding
  the lock 2026-05-02 02:14 (no other acquirer = no break = no recovery).
  See ``docs/plans/2026-05-03-gpu-gate-sweeper.md``.
"""

import logging
import os
import subprocess
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path

logger = logging.getLogger("portfolio.gpu_gate")

# In-process lock — prevents ThreadPoolExecutor workers from racing
_THREAD_LOCK = threading.Lock()

# File-based lock for cross-process protection
_GPU_LOCK_DIR = Path("Q:/models")
_GPU_LOCK_FILE = _GPU_LOCK_DIR / ".gpu_lock"
_STALE_SECONDS = 300  # 5 min

# Stale-lock sweeper daemon (2026-05-03). Module-level singleton so subprocess
# workers that import this module only spawn one sweeper, not one per import.
_SWEEPER_INTERVAL_SECONDS = 30
_SWEEPER_LOCK = threading.Lock()
_sweeper_thread: "threading.Thread | None" = None


def get_vram_usage() -> dict:
    """Query nvidia-smi for current VRAM usage. Returns dict or None on error."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parts = [p.strip() for p in proc.stdout.strip().split(",")]
            if len(parts) >= 4:
                return {
                    "used_mb": int(parts[0]),
                    "free_mb": int(parts[1]),
                    "total_mb": int(parts[2]),
                    "gpu_util_pct": int(parts[3]),
                }
    except Exception:
        logger.debug("GPU info query failed", exc_info=True)
    return None


def _is_stale() -> bool:
    try:
        return (time.time() - _GPU_LOCK_FILE.stat().st_mtime) > _STALE_SECONDS
    except OSError:
        return True


def _pid_alive(pid: int) -> bool:
    """Check if a process is still running. BUG-182."""
    if not pid or pid < 0:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        # Fallback: assume alive if we can't check
        return True


def _read_lock() -> dict:
    try:
        text = _GPU_LOCK_FILE.read_text(encoding="utf-8").strip()
        parts = text.split("|")
        return {
            "model": parts[0] if len(parts) > 0 else "unknown",
            "pid": int(parts[1]) if len(parts) > 1 else 0,
            "ts": float(parts[2]) if len(parts) > 2 else 0,
        }
    except (OSError, ValueError):
        return {}


def _release_lock():
    with suppress(OSError):
        _GPU_LOCK_FILE.unlink(missing_ok=True)


def _try_break_stale_lock() -> bool:
    """Reap the lock file iff stale-by-mtime AND owner pid is dead.

    Returns True if the lock was broken (caller can retry acquire), False
    otherwise. Defensive: never raises — the sweeper daemon depends on this.

    Called from two paths:
    - Reactive: ``gpu_gate()`` retry loop, when another caller is waiting.
    - Sweeper: the background daemon, when no one is waiting.

    Both paths must agree on the predicate so behaviour is identical
    regardless of which path reaped the lock. Emits the same
    ``Breaking stale GPU lock`` warning either way so log-grep tools and
    postmortem audits work uniformly.
    """
    try:
        if not _GPU_LOCK_FILE.exists():
            return False
        if not _is_stale():
            return False
        info = _read_lock()
        pid = info.get("pid", 0)
        if _pid_alive(pid):
            return False
        logger.warning("Breaking stale GPU lock: %s (pid=%s, dead)",
                       info.get("model"), pid)
        _release_lock()
        return True
    except Exception as exc:
        # The sweeper must NEVER crash — a dead daemon stops sweeping forever.
        logger.debug("Stale-lock sweep error: %s", exc)
        return False


def _sweeper_loop():
    """Background daemon: reap stale-dead locks every 30 s.

    Wedge-recovery story (2026-05-02): chronos pid 13152 died holding the
    lock at 02:14. No one tried to acquire while the loop was stuck inside
    its LLM batch, so ``_is_stale()`` was never checked. Loop wedged for
    ~25 hours until a system reboot. This daemon closes that hole.
    """
    while True:
        try:
            time.sleep(_SWEEPER_INTERVAL_SECONDS)
            _try_break_stale_lock()
        except Exception as exc:
            # Defence-in-depth — _try_break_stale_lock already swallows but
            # any future code added here must also keep the daemon alive.
            logger.debug("Sweeper loop error: %s", exc)


def _start_sweeper():
    """Spawn the sweeper daemon (idempotent, thread-safe).

    Lazily called from ``gpu_gate()`` so:
    - Subprocess workers that import this module but never call
      ``gpu_gate()`` (e.g. ``portfolio.signal_engine``'s import-time scan)
      do NOT spawn a redundant daemon.
    - Tests can reset ``_sweeper_thread = None`` and re-trigger spawn.

    If the daemon ever dies (it shouldn't — both layers swallow exceptions)
    a future call will respawn it.
    """
    global _sweeper_thread
    with _SWEEPER_LOCK:
        if _sweeper_thread is None or not _sweeper_thread.is_alive():
            t = threading.Thread(
                target=_sweeper_loop,
                name="gpu-gate-sweeper",
                daemon=True,
            )
            _sweeper_thread = t
            t.start()


@contextmanager
def gpu_gate(model_name: str, timeout: float = 60):
    """Acquire exclusive GPU access, log VRAM before/after.

    Uses a two-layer lock:
    1. threading.Lock for in-process concurrency (ThreadPoolExecutor workers)
    2. File-based lock for cross-process protection (metals loop, etc.)

    Args:
        model_name: e.g. "ministral-3", "qwen3", "chronos"
        timeout: max seconds to wait for lock

    Yields:
        True if acquired, False if timed out.
    """
    # Lazy-spawn the stale-lock sweeper. Idempotent so no cost after the
    # first call. See _start_sweeper() for the rationale.
    _start_sweeper()

    deadline = time.time() + timeout

    # Layer 1: In-process thread lock (prevents ThreadPoolExecutor races)
    remaining = deadline - time.time()
    thread_acquired = _THREAD_LOCK.acquire(timeout=max(0, remaining))
    if not thread_acquired:
        logger.warning("GPU thread-lock timeout (%ss) for %s", timeout, model_name)
        yield False
        return

    try:
        # Layer 2: File-based lock (cross-process)
        file_acquired = False
        while time.time() < deadline:
            try:
                # Atomic create — fails if file already exists (no TOCTOU race)
                fd = os.open(str(_GPU_LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                # H23/CI1: Always close fd in finally to prevent leak if write raises.
                try:
                    os.write(fd, f"{model_name}|{os.getpid()}|{time.time()}|{threading.get_ident()}".encode())
                finally:
                    os.close(fd)
                file_acquired = True
                break
            except FileExistsError:
                # Lock file exists — check if same process (re-entry) or stale.
                info = _read_lock()
                if info.get("pid") == os.getpid():
                    # Re-entry from same process (shouldn't happen with thread lock, but safe)
                    file_acquired = True
                    break
                # BUG-182: Only break stale lock if owning process is dead.
                # Helper is shared with the sweeper daemon so the two paths
                # agree on the predicate.
                if _try_break_stale_lock():
                    continue  # retry atomic create
                logger.debug("GPU file-locked by %s, waiting...", info.get("model", "?"))
                time.sleep(1.0)

        if not file_acquired:
            info = _read_lock()
            logger.warning("GPU file-lock timeout (%ss) — held by %s", timeout, info.get("model", "?"))
            yield False
            return

        # Log VRAM at acquire
        vram = get_vram_usage()
        if vram:
            logger.info(
                "GPU gate ACQUIRED by %s — VRAM: %dMB used / %dMB free / %dMB total (GPU %d%%)",
                model_name, vram["used_mb"], vram["free_mb"], vram["total_mb"], vram["gpu_util_pct"],
            )

        t0 = time.time()
        try:
            yield True
        finally:
            elapsed = time.time() - t0
            vram = get_vram_usage()
            if vram:
                logger.info(
                    "GPU gate RELEASED by %s after %.1fs — VRAM: %dMB used / %dMB free",
                    model_name, elapsed, vram["used_mb"], vram["free_mb"],
                )
            _release_lock()
    finally:
        _THREAD_LOCK.release()
