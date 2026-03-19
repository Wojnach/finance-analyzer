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
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_append_jsonl, load_jsonl

logger = logging.getLogger("portfolio.claude_gate")

# ---------------------------------------------------------------------------
# Master kill switch.  Set to False to block ALL Claude Code invocations
# across the entire codebase — no exceptions.
# ---------------------------------------------------------------------------
CLAUDE_ENABLED = False

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_FILE = BASE_DIR / "config.json"
INVOCATIONS_LOG = DATA_DIR / "claude_invocations.jsonl"

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
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_clean_env(),
            cwd=working_dir,
            stdin=subprocess.DEVNULL,
        )
        exit_code = result.returncode
        status = "invoked" if exit_code == 0 else "error"
    except subprocess.TimeoutExpired:
        status = "timeout"
        logger.warning("Claude invocation timed out after %ds — caller=%s", timeout, caller)
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
