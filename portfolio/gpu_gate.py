"""GPU gating and VRAM monitoring for LLM model inference.

Ensures only one LLM model uses the GPU at a time. Logs VRAM usage
before and after each model load for monitoring.

Uses a threading lock for in-process concurrency (ThreadPoolExecutor workers)
plus a file-based lock at Q:/models/.gpu_lock for cross-process protection.
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
    if not pid:
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


def _write_lock(model_name: str):
    _GPU_LOCK_FILE.write_text(
        f"{model_name}|{os.getpid()}|{time.time()}|{threading.get_ident()}",
        encoding="utf-8",
    )


def _release_lock():
    with suppress(OSError):
        _GPU_LOCK_FILE.unlink(missing_ok=True)


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
                # Lock file exists — check if stale or same process
                info = _read_lock()
                if info.get("pid") == os.getpid():
                    # Re-entry from same process (shouldn't happen with thread lock, but safe)
                    file_acquired = True
                    break
                if _is_stale() and not _pid_alive(info.get("pid", 0)):
                    # BUG-182: Only break stale lock if owning process is dead
                    logger.warning("Breaking stale GPU lock: %s (pid=%s, dead)",
                                   info.get("model"), info.get("pid"))
                    _release_lock()
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
