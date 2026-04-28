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

import contextlib
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
# 2026-04-13: Append-only journal of failures that EVERY future Claude Code
# session must see. Intentionally separate from claude_invocations.jsonl so
# hooks and startup scripts can cheaply poll it without parsing routine
# invocation noise. Consumed by scripts/check_critical_errors.py, which is
# referenced from CLAUDE.md to guarantee surfacing at session start.
CRITICAL_ERRORS_LOG = DATA_DIR / "critical_errors.jsonl"

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


# 2026-04-13: Detector for silent auth failures. The `--bare` flag (removed
# from agent_invocation.py and multi_agent_layer2.py on 2026-04-13) disables
# OAuth/keychain auth and requires ANTHROPIC_API_KEY. Since this user runs
# on a Max subscription with no API key, `--bare` caused every Layer 2
# invocation between 2026-03-27 and 2026-04-13 to print "Not logged in —
# Please run /login" on stdout and exit 0. Nothing surfaced the failure
# because exit_code=0 was treated as success across all three invocation
# paths. Do not re-add `--bare`. If a new CLI flag or env tweak
# re-introduces this class of silent auth error, this detector should
# catch it.
_AUTH_ERROR_MARKERS = ("Not logged in", "Please run /login", "Invalid API key")

# 2026-04-16: feedback-loop fix. CLAUDE.md tells every agent to surface
# unresolved critical_errors.jsonl entries verbatim at session start. Those
# entries CONTAIN the literal string "Not logged in", so the substring scan
# below was treating every echo as a new auth failure, journaling it,
# triggering the next agent to re-surface, ad infinitum (today's entries
# 13:45:45 + 14:15:01 are both echoes, not real failures).
#
# The fix narrows the match: a marker only counts when it's the START of a
# line, NOT preceded by quote/backtick/paren/blockquote, with no leading
# indentation, AND within the first _AUTH_SCAN_LINE_LIMIT lines of output.
# Real Claude CLI auth errors print as standalone preamble — they never
# appear deep in agent chat. Echoes always appear quoted, indented, in
# code blocks, or wrapped in conversational context.
_AUTH_SCAN_LINE_LIMIT = 16
# Characters that, when they precede the marker, mean "this is quoted, not
# CLI output". `'` `"` and `` ` `` cover plain quotes; `(` covers
# parentheticals; `>` covers Markdown blockquotes; `[` covers JSON-style
# log entries (`["ts": ..., "message": "...Not logged in..."]`); whitespace
# at line start covers code-block indentation.
_AUTH_MARKER_PREFIX_REJECT = ("'", '"', "`", "(", ">", "[", " ", "\t")


def _is_real_auth_marker_line(line: str, marker: str) -> bool:
    """Return True if `line` looks like an actual CLI auth-error line.

    The CLI prints markers as standalone lines without quoting. Anything
    quoted, indented, blockquoted, or embedded in conversational text is
    almost certainly an echo of a previously-journaled error.
    """
    if not line:
        return False
    # Reject lines that begin with a wrapper character before the marker.
    if line[0] in _AUTH_MARKER_PREFIX_REJECT:
        return False
    # The marker must appear at the very start (after any leading wrapper
    # check above has already passed — i.e. no leading whitespace).
    if not line.startswith(marker):
        return False
    # Defense in depth: even if startswith matches, reject if any wrapper
    # char appears in the slice BEFORE the marker (handles bullet lists
    # like `- Not logged in` that tests pre-empt by checking line[0]).
    return True


def record_critical_error(
    category: str,
    caller: str,
    message: str,
    context: dict | None = None,
) -> bool:
    """Append a critical error to ``data/critical_errors.jsonl``.

    The journal is the single source of truth consulted by
    ``scripts/check_critical_errors.py`` at Claude session start (via
    CLAUDE.md). Writing here guarantees the failure is visible to every
    future Claude session until it's resolved with a follow-up entry.

    Never raises — logging failures here must not cascade into the caller.

    Returns ``True`` when the append landed, ``False`` when it failed.
    The boolean lets dedup-aware callers (e.g. loop_contract's
    ``_dispatch_critical_errors_for_degradation``) avoid claiming a
    dedup slot for a row that never made it to disk — otherwise a
    transient IO problem would silence 6+ h of unrecorded incidents
    (Codex P2 2026-04-28). Callers that don't need the signal can
    safely ignore the return.
    """
    try:
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "level": "critical",
            "category": category,
            "caller": caller,
            "resolution": None,
            "message": message,
            "context": context or {},
        }
        atomic_append_jsonl(CRITICAL_ERRORS_LOG, entry)
        return True
    except Exception as e:
        logger.error("Failed to write critical_errors.jsonl: %s", e)
        return False


def detect_auth_failure(output: str, caller: str, context: dict | None = None) -> bool:
    """Scan subprocess output for claude-CLI auth errors and escalate.

    Returns True if an auth failure pattern is detected. On match, logs at
    CRITICAL level AND records the failure to ``critical_errors.jsonl`` so
    future Claude sessions see it via the CLAUDE.md startup check. Callers
    should downgrade ``success`` to False and mark the invocation status as
    ``auth_error`` so the failure also shows up in the invocation log.

    Deliberately logger.critical rather than an exception — the finance
    loop runs 24/7 and raising here would tear down a tick. The
    critical-level log + critical_errors.jsonl entry + invocation-log
    status="auth_error" together make the failure impossible to miss.
    """
    if not output:
        return False

    # Scan only the top of the output. Real CLI auth errors print as
    # preamble before any agent turn output; echoes always appear later
    # in conversational chat. See _AUTH_SCAN_LINE_LIMIT comment above
    # for the full feedback-loop rationale (BUG-ECHO 2026-04-16).
    candidate_lines = output.splitlines()[:_AUTH_SCAN_LINE_LIMIT]
    in_fenced_code_block = False
    for line in candidate_lines:
        # Track Markdown fenced code blocks (```). Lines inside the block
        # are quoted content even if they don't have leading whitespace.
        if line.startswith("```"):
            in_fenced_code_block = not in_fenced_code_block
            continue
        if in_fenced_code_block:
            continue
        for marker in _AUTH_ERROR_MARKERS:
            if not _is_real_auth_marker_line(line, marker):
                continue
            logger.critical(
                "[AUTH_FAILURE] caller=%s — claude CLI printed %r. "
                "OAuth session not being read. Likely causes: "
                "--bare flag re-added, ANTHROPIC_API_KEY set to an invalid "
                "value, or ~/.claude/.credentials.json expired/missing. "
                "Run `claude` interactively to re-login.",
                caller, marker,
            )
            record_critical_error(
                category="auth_failure",
                caller=caller,
                message=(
                    f"claude CLI subprocess printed {marker!r} — OAuth session "
                    f"not being read. Check for --bare flag, invalid "
                    f"ANTHROPIC_API_KEY, or expired ~/.claude/.credentials.json."
                ),
                context={**(context or {}), "marker": marker},
            )
            return True
    return False


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
        logger.error(
            "%s tree kill encountered unexpected error: %s — proc.kill()",
            label, e, exc_info=True,
        )
        try:
            proc.kill()
        except Exception as kill_err:  # 2026-04-17: surface orphan risk
            logger.error(
                "%s proc.kill() also failed after tree-kill error: %s — "
                "process pid=%s may be orphaned",
                label, kill_err, getattr(proc, "pid", "?"),
            )


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
            with contextlib.suppress(Exception):
                proc.kill()
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
            # 2026-04-13: Silent-failure detector. claude CLI can exit 0 while
            # printing "Not logged in" when OAuth/keychain auth can't be read
            # (e.g. --bare flag, missing ANTHROPIC_API_KEY). Override status
            # so the failure surfaces instead of being lost to exit_code=0.
            # BUG-ECHO follow-up (Codex P2 finding 2026-04-16): scan stdout
            # and stderr SEPARATELY rather than concatenating without a
            # newline. Concat-without-newline could merge the marker into
            # the last stdout line ("...stdoutNot logged in"), defeating
            # the start-of-line check shipped today. Scanning each stream
            # independently preserves both streams' line-1 position.
            stdout_hit = detect_auth_failure(
                _stdout or "", caller,
                context={"model": model, "max_turns": max_turns, "exit_code": exit_code},
            )
            stderr_hit = detect_auth_failure(
                _stderr or "", caller,
                context={"model": model, "max_turns": max_turns, "exit_code": exit_code},
            ) if not stdout_hit else False
            if stdout_hit or stderr_hit:
                status = "auth_error"
                exit_code = exit_code or 1
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

    return status == "invoked", exit_code


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
            # 2026-04-13: Same auth-failure detection as invoke_claude — see
            # the comment there for the full context. Need to scan both
            # stdout and stderr because the CLI can write "Not logged in"
            # to either depending on version.
            # BUG-ECHO follow-up (Codex P2 finding 2026-04-16): scan each
            # stream independently so concat-without-newline can't merge
            # the marker into the last stdout line. See invoke_claude for
            # the full rationale.
            stdout_hit = detect_auth_failure(
                stdout or "", caller,
                context={"model": model, "max_turns": 1, "exit_code": exit_code},
            )
            stderr_hit = detect_auth_failure(
                _stderr or "", caller,
                context={"model": model, "max_turns": 1, "exit_code": exit_code},
            ) if not stdout_hit else False
            if stdout_hit or stderr_hit:
                status = "auth_error"
                exit_code = exit_code or 1
                text = ""  # don't let the error message leak into the caller's "text"
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

    return text, status == "invoked", exit_code


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
