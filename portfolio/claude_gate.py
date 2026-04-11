"""Centralized Claude Code invocation gatekeeper.

This module is the ONLY approved way to invoke Claude Code (``claude -p``)
from anywhere in the codebase.  All callers — agent_invocation, metals_loop,
silver_monitor, claude_fundamental, analyze, bigbet, iskbets, etc. — MUST
route through ``invoke_claude()`` defined here.

Direct ``subprocess.Popen([claude_cmd, "-p", ...])`` calls are FORBIDDEN.
Doing so bypasses the kill switch, rate limiter, and invocation tracking.

Usage::

    from portfolio.claude_gate import invoke_claude

    success, exit_code = invoke_claude(
        prompt="Analyze BTC-USD",
        caller="silver_monitor",
        model="sonnet",
        max_turns=20,
        timeout=180,
    )
"""

import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_append_jsonl, load_jsonl

logger = logging.getLogger("portfolio.claude_gate")

import threading

# ---------------------------------------------------------------------------
# Master kill switch.  Set to False to block ALL Claude Code invocations
# across the entire codebase — no exceptions.
# ---------------------------------------------------------------------------
CLAUDE_ENABLED = True

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_FILE = BASE_DIR / "config.json"
INVOCATIONS_LOG = DATA_DIR / "claude_invocations.jsonl"

# A-IN-3 (2026-04-11): In-process concurrency lock. Without this, the main
# loop's 8-worker ticker pool + the metals loop's fast-tick + signal
# subprocesses can all call invoke_claude in parallel. The Claude CLI is
# expensive (sonnet ~30s, opus ~3-5min) and the rate limiter is per-day,
# not per-second — uncoordinated parallel invocations can:
#   1. Race past the kill switch (CLAUDE_ENABLED check is non-atomic)
#   2. Spawn 5+ concurrent Claude processes, each holding ~500MB RAM
#   3. Confuse the invocation log (timestamps interleave)
# Serializing in-process invocations is the simplest robust fix. For
# cross-process coordination (multiple Python processes), see the file
# lock TODO below.
_invoke_lock = threading.Lock()

# Rate-limit threshold: warn when daily invocations exceed this count.
_DAILY_WARN_THRESHOLD = 50


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config_layer2_enabled() -> bool:
    """Check ``config.json -> layer2.enabled``.

    Returns True if the key is missing or the file cannot be read (fail-open
    for the config check — the module-level CLAUDE_ENABLED flag is the hard
    gate).
    """
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("layer2", {}).get("enabled", True)
    except Exception:
        # Config unreadable — don't block on that alone.
        return True


def _clean_env() -> dict:
    """Return a copy of ``os.environ`` with Claude session markers removed.

    Prevents the "nested session" error when invoking ``claude -p`` from a
    process tree that already has a Claude Code session active.
    """
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    return env


def _find_claude_cmd() -> str | None:
    """Locate the ``claude`` CLI executable on PATH."""
    return shutil.which("claude")


def _log_invocation(entry: dict) -> None:
    """Append an invocation record to the JSONL log."""
    try:
        atomic_append_jsonl(INVOCATIONS_LOG, entry)
    except Exception as e:
        logger.warning("Failed to write invocation log: %s", e)


def _count_today_invocations() -> int:
    """Count invocation records from today (UTC)."""
    today_str = datetime.now(UTC).strftime("%Y-%m-%d")
    count = 0
    for entry in load_jsonl(INVOCATIONS_LOG):
        ts = entry.get("timestamp", "")
        if ts.startswith(today_str):
            count += 1
    return count


# A-IN-2 (2026-04-11): The previous code used `subprocess.run(timeout=...)`.
# CPython's run() does kill the *direct* child on TimeoutExpired, but the
# Claude CLI is a Node.js process that spawns its own helpers (MCP servers,
# the actual claude API client process, etc.). Killing the direct child
# leaves all of its descendants running as zombies on Windows. Over a long
# session this leaks file handles, sockets, and (worst) GPU VRAM held by
# any local-LLM helpers Claude may have spawned.
#
# Fix: explicitly Popen with a new process group/session so we can kill the
# entire tree, not just the direct child. On Windows we use taskkill /T /F
# (kills the whole tree by PID); on Unix we use os.killpg(SIGKILL) on the
# process group started via start_new_session=True.
def _popen_kwargs_for_tree_kill() -> dict:
    """Return Popen kwargs that allow tree-killing the spawned process."""
    if platform.system() == "Windows":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def _kill_process_tree(proc: subprocess.Popen, *, label: str = "claude") -> None:
    """Kill a Popen process and all of its descendants. Best-effort:
    falls back to proc.kill() if the platform-specific path fails.
    Always returns; never raises."""
    if proc.poll() is not None:
        return  # already exited
    pid = proc.pid
    try:
        if platform.system() == "Windows":
            # taskkill /T = terminate this PID and all child processes,
            # /F = force, /PID = the parent PID. Capture stderr to keep
            # logs clean if the process already exited between poll() and here.
            res = subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                capture_output=True, timeout=5,
            )
            if res.returncode not in (0, 128):  # 128 = "process not found"
                logger.warning(
                    "%s tree kill via taskkill returned %d (stderr=%r) — "
                    "falling back to proc.kill()",
                    label, res.returncode, res.stderr.decode("utf-8", "replace")[:200],
                )
                proc.kill()
        else:
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError) as e:
                logger.warning("%s killpg(%d) failed: %s — falling back to proc.kill()", label, pid, e)
                proc.kill()
    except Exception as e:
        # Last-ditch fallback so a kill failure never propagates.
        logger.error("%s tree kill encountered unexpected error: %s — proc.kill()", label, e)
        try:
            proc.kill()
        except Exception:
            pass


def _run_with_tree_kill(
    cmd: list[str],
    *,
    timeout: float,
    env: dict | None,
    cwd: str,
    label: str,
) -> tuple[int, str, str, bool]:
    """Run a subprocess with proper timeout + tree-kill cleanup.

    Returns:
        (returncode, stdout, stderr, timed_out)

    On timeout, kills the entire process tree (not just the direct child)
    and waits up to 5s for the tree to actually exit before returning.
    Logs an error if the tree refused to exit.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        env=env,
        cwd=cwd,
        **_popen_kwargs_for_tree_kill(),
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout or "", stderr or "", False
    except subprocess.TimeoutExpired:
        logger.warning("%s timed out after %ds — killing process tree (pid=%d)",
                       label, timeout, proc.pid)
        _kill_process_tree(proc, label=label)
        # Drain pipes after kill so the OS can release them.
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            logger.error("%s process tree did not exit within 5s of kill — possible zombie", label)
            try:
                proc.kill()
            except Exception:
                pass
            stdout, stderr = "", ""
        return -1, stdout or "", stderr or "", True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def invoke_claude(
    prompt: str,
    caller: str,
    model: str = "sonnet",
    max_turns: int = 20,
    allowed_tools: str = "Read,Edit,Bash,Write",
    timeout: int = 180,
    cwd: str | None = None,
) -> tuple[bool, int]:
    """Invoke Claude Code via ``claude -p`` and wait for completion.

    Args:
        prompt: The prompt text to send.
        caller: Identifier of the calling module (e.g. ``"silver_monitor"``).
        model: Claude model to use (``"sonnet"``, ``"haiku"``, ``"opus"``).
        max_turns: Maximum agentic turns.
        allowed_tools: Comma-separated tool names for ``--allowedTools``.
        timeout: Subprocess timeout in seconds.
        cwd: Working directory for the subprocess.  Defaults to the repo root.

    Returns:
        ``(success, exit_code)`` where *success* is True when exit_code == 0.
        If the invocation is blocked, returns ``(False, -1)``.
    """
    now_iso = datetime.now(UTC).isoformat()
    working_dir = cwd or str(BASE_DIR)

    # --- Gate 1: module-level kill switch ---
    if not CLAUDE_ENABLED:
        logger.info("Claude invocation BLOCKED (CLAUDE_ENABLED=False) caller=%s", caller)
        _log_invocation({
            "timestamp": now_iso,
            "caller": caller,
            "status": "blocked",
            "reason": "CLAUDE_ENABLED=False",
            "model": model,
            "max_turns": max_turns,
            "duration_seconds": 0,
            "exit_code": -1,
        })
        return False, -1

    # --- Gate 2: config.json layer2.enabled ---
    if not _load_config_layer2_enabled():
        logger.info("Claude invocation BLOCKED (config layer2.enabled=false) caller=%s", caller)
        _log_invocation({
            "timestamp": now_iso,
            "caller": caller,
            "status": "blocked",
            "reason": "config.layer2.enabled=false",
            "model": model,
            "max_turns": max_turns,
            "duration_seconds": 0,
            "exit_code": -1,
        })
        return False, -1

    # --- Rate-limit warning ---
    today_count = _count_today_invocations()
    if today_count >= _DAILY_WARN_THRESHOLD:
        logger.warning(
            "Daily invocation count (%d) exceeds threshold (%d) — caller=%s",
            today_count, _DAILY_WARN_THRESHOLD, caller,
        )

    # --- Locate claude CLI ---
    claude_cmd = _find_claude_cmd()
    if not claude_cmd:
        logger.error("claude CLI not found on PATH — caller=%s", caller)
        _log_invocation({
            "timestamp": now_iso,
            "caller": caller,
            "status": "error",
            "reason": "claude not on PATH",
            "model": model,
            "max_turns": max_turns,
            "duration_seconds": 0,
            "exit_code": -1,
        })
        return False, -1

    # --- Build command ---
    cmd = [
        claude_cmd, "-p", prompt,
        "--allowedTools", allowed_tools,
        "--max-turns", str(max_turns),
        "--model", model,
        "--output-format", "text",
    ]

    # --- Execute ---
    t0 = time.time()
    exit_code = -1
    status = "error"

    try:
        # A-IN-3: serialize all in-process Claude invocations so the
        # 8-worker ticker pool / metals fast-tick / signal subprocesses
        # don't spawn 5 concurrent expensive Claude processes.
        # A-IN-2: tree-killing helper for grandchild cleanup on timeout.
        with _invoke_lock:
            rc, _stdout, _stderr, timed_out = _run_with_tree_kill(
                cmd,
                timeout=timeout,
                env=_clean_env(),
                cwd=working_dir,
                label=f"claude({caller})",
            )
        if timed_out:
            status = "timeout"
        else:
            exit_code = rc
            status = "invoked" if exit_code == 0 else "error"
    except Exception as e:
        status = "error"
        logger.error("Claude invocation failed — caller=%s: %s", caller, e)

    duration = round(time.time() - t0, 2)

    _log_invocation({
        "timestamp": now_iso,
        "caller": caller,
        "status": status,
        "model": model,
        "max_turns": max_turns,
        "duration_seconds": duration,
        "exit_code": exit_code,
    })

    logger.info(
        "Claude invocation: caller=%s model=%s status=%s exit=%d duration=%.1fs",
        caller, model, status, exit_code, duration,
    )

    return exit_code == 0, exit_code


def invoke_claude_text(
    prompt: str,
    caller: str,
    model: str = "sonnet",
    timeout: int = 60,
) -> tuple[str, bool, int]:
    """Invoke Claude CLI for text-only Q&A (no tools, single turn).

    Unlike ``invoke_claude()``, this captures stdout and returns the text
    response.  Used by signals that need Claude's analysis as structured
    text (e.g., claude_fundamental).

    Returns:
        ``(text, success, exit_code)``
    """
    now_iso = datetime.now(UTC).isoformat()

    if not CLAUDE_ENABLED or not _load_config_layer2_enabled():
        _log_invocation({
            "timestamp": now_iso, "caller": caller, "status": "blocked",
            "reason": "disabled", "model": model, "max_turns": 1,
            "duration_seconds": 0, "exit_code": -1,
        })
        return "", False, -1

    claude_cmd = _find_claude_cmd()
    if not claude_cmd:
        logger.error("claude CLI not found — caller=%s", caller)
        return "", False, -1

    cmd = [
        claude_cmd, "-p", prompt,
        "--model", model,
        "--output-format", "text",
        "--max-turns", "1",
        "--allowedTools", "",
    ]

    t0 = time.time()
    text = ""
    exit_code = -1
    status = "error"

    try:
        # A-IN-3 + A-IN-2: serialized + tree-killing.
        with _invoke_lock:
            rc, stdout, _stderr, timed_out = _run_with_tree_kill(
                cmd,
                timeout=timeout,
                env=_clean_env(),
                cwd=str(BASE_DIR),
                label=f"claude_text({caller})",
            )
        if timed_out:
            status = "timeout"
        else:
            exit_code = rc
            text = stdout
            status = "invoked" if exit_code == 0 else "error"
    except Exception as e:
        logger.error("Claude text invocation failed — caller=%s: %s", caller, e)

    duration = round(time.time() - t0, 2)
    _log_invocation({
        "timestamp": now_iso, "caller": caller, "status": status,
        "model": model, "max_turns": 1,
        "duration_seconds": duration, "exit_code": exit_code,
    })

    logger.info(
        "Claude text: caller=%s model=%s status=%s exit=%d duration=%.1fs len=%d",
        caller, model, status, exit_code, duration, len(text),
    )

    return text, exit_code == 0, exit_code


def get_invocation_stats() -> dict:
    """Return summary statistics from the invocation log.

    Returns:
        Dict with keys: ``total_invocations``, ``today_invocations``,
        ``last_invocation_ts``, ``last_caller``, ``enabled``.
    """
    entries = load_jsonl(INVOCATIONS_LOG)
    today_str = datetime.now(UTC).strftime("%Y-%m-%d")

    total = len(entries)
    today_count = 0
    last_ts = None
    last_caller = None

    for entry in entries:
        ts = entry.get("timestamp", "")
        if ts.startswith(today_str):
            today_count += 1
        # Track the latest entry by position (JSONL is append-only,
        # so the last entry in the list is the most recent).
        last_ts = ts or last_ts
        last_caller = entry.get("caller") or last_caller

    # Combine both gates for the overall enabled status.
    enabled = CLAUDE_ENABLED and _load_config_layer2_enabled()

    return {
        "total_invocations": total,
        "today_invocations": today_count,
        "last_invocation_ts": last_ts,
        "last_caller": last_caller,
        "enabled": enabled,
    }
