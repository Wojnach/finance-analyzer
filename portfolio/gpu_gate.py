"""GPU gating and VRAM monitoring for LLM model inference.

Ensures only one LLM model uses the GPU at a time. Logs VRAM usage
before and after each model load for monitoring.

Uses the existing file-based lock at Q:/models/.gpu_lock.
"""

import logging
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger("portfolio.gpu_gate")

# Import the existing GPU lock
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
    import os
    _GPU_LOCK_FILE.write_text(
        f"{model_name}|{os.getpid()}|{time.time()}", encoding="utf-8"
    )


def _release_lock():
    try:
        _GPU_LOCK_FILE.unlink(missing_ok=True)
    except OSError:
        pass


@contextmanager
def gpu_gate(model_name: str, timeout: float = 60):
    """Acquire exclusive GPU access, log VRAM before/after.

    Args:
        model_name: e.g. "ministral-3", "qwen3", "chronos"
        timeout: max seconds to wait for lock

    Yields:
        None (use as context manager)

    Logs VRAM usage at acquire and release for monitoring.
    """
    import os

    deadline = time.time() + timeout
    acquired = False

    while time.time() < deadline:
        if not _GPU_LOCK_FILE.exists():
            _write_lock(model_name)
            acquired = True
            break

        info = _read_lock()

        # Re-entry from same process
        if info.get("model") == model_name and info.get("pid") == os.getpid():
            acquired = True
            break

        # Stale lock
        if _is_stale():
            logger.warning("Breaking stale GPU lock: %s (pid=%s)", info.get("model"), info.get("pid"))
            _release_lock()
            _write_lock(model_name)
            acquired = True
            break

        # Wait
        holder = info.get("model", "?")
        logger.debug("GPU locked by %s, waiting...", holder)
        time.sleep(1.0)

    if not acquired:
        info = _read_lock()
        logger.warning("GPU lock timeout (%ss) — held by %s", timeout, info.get("model", "?"))
        # Don't raise — just skip this model. Better to return HOLD than crash.
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
        # Log VRAM at release
        vram = get_vram_usage()
        if vram:
            logger.info(
                "GPU gate RELEASED by %s after %.1fs — VRAM: %dMB used / %dMB free",
                model_name, elapsed, vram["used_mb"], vram["free_mb"],
            )
        # Release lock
        info = _read_lock()
        if info.get("pid") == os.getpid():
            _release_lock()
