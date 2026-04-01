"""Layer 2 agent invocation — manages the Claude Code subprocess lifecycle."""

import logging
import os
import platform
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from portfolio.api_utils import load_config as _load_config
from portfolio.file_utils import atomic_append_jsonl, last_jsonl_entry, load_jsonl
from portfolio.message_store import send_or_store
from portfolio.telegram_notifications import escape_markdown_v1

logger = logging.getLogger("portfolio.agent")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INVOCATIONS_FILE = DATA_DIR / "invocations.jsonl"
JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
TELEGRAM_FILE = DATA_DIR / "telegram_messages.jsonl"

_agent_proc = None
_agent_log = None
_agent_start = 0
_agent_timeout = 900  # per-invocation timeout (set from tier config)
_agent_tier = None  # tier of the currently running agent
_agent_reasons = None  # trigger reasons for the current invocation
_journal_ts_before = None  # last journal timestamp before agent started
_telegram_ts_before = None  # last telegram timestamp before agent started

# Stack overflow detection — exit code 3221225794 = Windows STATUS_STACK_OVERFLOW (0xC00000FD)
_STACK_OVERFLOW_EXIT_CODE = 3221225794
_consecutive_stack_overflows = 0
_MAX_STACK_OVERFLOWS = 5  # auto-disable after this many consecutive stack overflow crashes

# Per-tier configuration
TIER_CONFIG = {
    1: {"max_turns": 15, "timeout": 120, "label": "QUICK CHECK"},
    2: {"max_turns": 40, "timeout": 600, "label": "SIGNAL ANALYSIS"},
    3: {"max_turns": 40, "timeout": 900, "label": "FULL REVIEW"},
}


def _build_tier_prompt(tier, reasons):
    """Build a tier-specific prompt for the Claude Code agent."""
    reason_str = ", ".join(reasons[:5])

    playbook = "docs/TRADING_PLAYBOOK.md"

    if tier == 1:
        return (
            "You are the Layer 2 trading agent (QUICK CHECK). "
            f"Trigger: {reason_str}. "
            f"Read {playbook} for trading rules, then data/layer2_context.md "
            "then data/agent_context_t1.json. "
            "This is a routine check. Confirm held positions are OK (check ATR stops). "
            "If no positions are held, briefly assess macro state. "
            "Write a brief journal entry and send a short Telegram message. "
            "Do NOT analyze all tickers — focus only on held positions and macro headline."
        )
    elif tier == 2:
        return (
            "You are the Layer 2 trading agent (SIGNAL ANALYSIS). "
            f"Trigger: {reason_str}. "
            "If data/trading_insights.md exists, read it first for recent signal performance context. "
            f"Read {playbook} for trading rules, then data/layer2_context.md, "
            "then data/agent_context_t2.json, "
            "data/portfolio_state.json, and data/portfolio_state_bold.json. "
            "Analyze triggered tickers and held positions. Decide for BOTH strategies. "
            "Write journal entry and send Telegram per the playbook instructions."
        )
    else:
        # Tier 3 — full review
        return (
            "You are the Layer 2 trading agent. "
            "If data/trading_insights.md exists, read it first for recent signal performance context. "
            f"FIRST read {playbook} for trading rules. "
            "Then read data/layer2_context.md (your memory from previous invocations). "
            "Then read data/agent_summary_compact.json (signals, trigger reasons, timeframes), "
            "data/portfolio_state.json (Patient portfolio), and data/portfolio_state_bold.json "
            "(Bold portfolio). Follow the playbook to analyze, decide, and act "
            "for BOTH strategies independently. Compare your previous theses and prices with "
            "current data — were you right? Always write a journal entry and send a Telegram message."
        )


def _log_trigger(reasons, status, tier=None):
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "reasons": reasons,
        "status": status,
    }
    if tier is not None:
        entry["tier"] = tier
    atomic_append_jsonl(INVOCATIONS_FILE, entry)


def _last_jsonl_ts(path):
    """Return the 'ts' value from the last entry of a JSONL file, or None.

    Uses efficient tail-read via last_jsonl_entry() (reads last 4KB only).
    """
    return last_jsonl_entry(path, field="ts")


def _safe_last_jsonl_ts(path, label):
    """Return the last JSONL timestamp without failing the invocation flow."""
    try:
        return _last_jsonl_ts(path)
    except Exception as e:
        logger.debug("%s baseline read failed: %s", label, e)
        return None


def invoke_agent(reasons, tier=3):
    global _agent_proc, _agent_log, _agent_start, _agent_timeout
    global _agent_tier, _agent_reasons, _journal_ts_before, _telegram_ts_before

    # Check if Layer 2 is auto-disabled due to consecutive stack overflows
    if _consecutive_stack_overflows >= _MAX_STACK_OVERFLOWS:
        logger.info(
            "Layer 2 skipped: auto-disabled after %d consecutive stack overflows",
            _consecutive_stack_overflows,
        )
        _log_trigger(reasons, "skipped_stack_overflow", tier=tier)
        return False

    # Check if Layer 2 is enabled — allows running data loop without Claude quota
    try:
        config = _load_config()
        l2_cfg = config.get("layer2", {})
        if not l2_cfg.get("enabled", True):
            logger.info("Layer 2 disabled (config.layer2.enabled=false), skipping")
            return False
    except Exception as e:
        logger.warning("Failed to load config for layer2 check: %s", e)

    tier_cfg = TIER_CONFIG.get(tier, TIER_CONFIG[3])
    timeout = tier_cfg["timeout"]

    if _agent_proc and _agent_proc.poll() is None:
        elapsed = time.time() - _agent_start
        if elapsed > _agent_timeout:
            logger.info("Agent pid=%s timed out (%.0fs), killing", _agent_proc.pid, elapsed)
            kill_ok = True
            if platform.system() == "Windows":
                # BUG-92: Check taskkill return code to detect kill failure
                result = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(_agent_proc.pid)],
                    capture_output=True,
                )
                if result.returncode != 0:
                    logger.error(
                        "taskkill failed (rc=%d): %s",
                        result.returncode, result.stderr.decode(errors="replace").strip(),
                    )
                    kill_ok = False
            else:
                _agent_proc.kill()
            try:
                _agent_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if kill_ok:
                    logger.error("Agent pid=%s did not exit after kill+wait", _agent_proc.pid)
                kill_ok = False
            if _agent_log:
                _agent_log.close()
                _agent_log = None
            # BUG-91: Log the timed-out invocation before spawning a new one
            _log_trigger(
                _agent_reasons or reasons, "timeout",
                tier=_agent_tier or tier,
            )
            # BUG-92: If kill failed, don't spawn new agent (old one may still be running)
            if not kill_ok:
                logger.error("Not spawning new agent — old process may still be running")
                _agent_proc = None
                return False
            _agent_proc = None
        else:
            logger.info(
                "Agent still running (pid %s, %.0fs), skipping",
                _agent_proc.pid, elapsed,
            )
            return False

    if _agent_log:
        _agent_log.close()
        _agent_log = None

    try:
        from portfolio.journal import write_context

        n = write_context()
        logger.info("Layer 2 context: %d journal entries", n)
    except Exception as e:
        logger.warning("journal context failed: %s", e)

    # Perception gate: skip low-value invocations
    try:
        from portfolio.perception_gate import should_invoke as _should_invoke
        should, gate_reason = _should_invoke(reasons, tier)
        if not should:
            logger.info("Perception gate skipped: %s", gate_reason)
            _log_trigger(reasons, "skipped_gate", tier=tier)
            return False
    except Exception as e:
        logger.warning("perception gate error (passing through): %s", e)

    prompt = _build_tier_prompt(tier, reasons)
    max_turns = tier_cfg["max_turns"]

    # Try direct claude invocation first; fall back to bat file for T3
    claude_cmd = shutil.which("claude")
    if claude_cmd:
        cmd = [
            claude_cmd, "-p", prompt,
            "--allowedTools", "Edit,Read,Bash,Write",
            "--max-turns", str(max_turns),
            "--bare",
        ]
    else:
        # Fallback: use pf-agent.bat (always Tier 3)
        agent_bat = BASE_DIR / "scripts" / "win" / "pf-agent.bat"
        if not agent_bat.exists():
            logger.warning("Agent script not found at %s", agent_bat)
            return False
        cmd = ["cmd", "/c", str(agent_bat)]
        logger.info("claude not on PATH, falling back to pf-agent.bat (T3)")

    log_fh = None
    try:
        log_fh = open(DATA_DIR / "agent.log", "a", encoding="utf-8")
        # Strip Claude Code session markers to avoid "nested session" error
        # when the parent process tree has Claude Code running
        agent_env = os.environ.copy()
        agent_env.pop("CLAUDECODE", None)
        agent_env.pop("CLAUDE_CODE_ENTRYPOINT", None)
        # Increase Node.js stack size to prevent stack overflow in Claude CLI
        agent_env["NODE_OPTIONS"] = "--stack-size=16384"
        _agent_proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=agent_env,
        )
        _agent_log = log_fh  # transfer ownership on success
        log_fh = None  # prevent cleanup below from closing it
        _agent_start = time.time()
        _agent_timeout = timeout
        _agent_tier = tier
        _agent_reasons = list(reasons)
        _journal_ts_before = _safe_last_jsonl_ts(JOURNAL_FILE, "journal")
        _telegram_ts_before = _safe_last_jsonl_ts(TELEGRAM_FILE, "telegram")
        logger.info(
            "Agent T%d invoked pid=%s max_turns=%d timeout=%ds (%s)",
            tier, _agent_proc.pid, max_turns, timeout,
            ", ".join(reasons[:3]),
        )
        # Save Layer 2 invocation notification (save-only, not sent to Telegram)
        try:
            config = _load_config()
            reason_str = escape_markdown_v1(", ".join(reasons[:3]))
            if len(reasons) > 3:
                reason_str += f" (+{len(reasons) - 3} more)"
            tier_label = tier_cfg["label"]
            notify_msg = f"_Layer 2 T{tier} ({tier_label}): {reason_str}_"
            send_or_store(notify_msg, config, category="invocation")
        except Exception as e:
            logger.debug("invocation notification failed: %s", e)
        return True
    except Exception as e:
        logger.error("invoking agent: %s", e)
        if log_fh is not None:
            log_fh.close()
        return False


def check_agent_completion():
    """Check if a running agent has completed and log completion info.

    Returns:
        dict with completion info (status, exit_code, duration_s, tier,
        journal_written, telegram_sent), or None if no agent is running
        or the agent is still in progress.
    """
    global _agent_proc, _agent_log, _agent_start
    global _agent_tier, _agent_reasons, _journal_ts_before, _telegram_ts_before

    if _agent_proc is None:
        return None

    exit_code = _agent_proc.poll()
    if exit_code is None:
        # Still running
        return None

    # Process has finished — collect completion info
    duration_s = round(time.time() - _agent_start, 1)
    completed_at = datetime.now(UTC).isoformat()

    # BUG-97: _last_jsonl_ts can raise OSError if file is locked on Windows
    try:
        journal_ts_after = _last_jsonl_ts(JOURNAL_FILE)
    except Exception:
        logger.warning("Failed to read journal timestamp after agent completion")
        journal_ts_after = None
    journal_written = (
        _journal_ts_before is not None
        and journal_ts_after is not None
        and journal_ts_after != _journal_ts_before
    )

    # BUG-97: Same protection for telegram file
    try:
        telegram_ts_after = _last_jsonl_ts(TELEGRAM_FILE)
    except Exception:
        logger.warning("Failed to read telegram timestamp after agent completion")
        telegram_ts_after = None
    telegram_sent = (
        _telegram_ts_before is not None
        and telegram_ts_after is not None
        and telegram_ts_after != _telegram_ts_before
    )

    # Without a baseline from invoke_agent(), stay conservative and do not infer
    # success from pre-existing files in the workspace.
    if _journal_ts_before is None:
        journal_written = False
    if _telegram_ts_before is None:
        telegram_sent = False

    # Determine status
    if exit_code != 0:
        status = "failed"
    elif journal_written and telegram_sent:
        status = "success"
    else:
        status = "incomplete"

    result = {
        "status": status,
        "exit_code": exit_code,
        "duration_s": duration_s,
        "tier": _agent_tier,
        "completed_at": completed_at,
        "journal_written": journal_written,
        "telegram_sent": telegram_sent,
    }

    # Log to invocations file
    log_entry = {
        "ts": completed_at,
        "reasons": _agent_reasons or [],
        "status": status,
        "tier": _agent_tier,
        "exit_code": exit_code,
        "duration_s": duration_s,
        "journal_written": journal_written,
        "telegram_sent": telegram_sent,
    }
    try:
        atomic_append_jsonl(INVOCATIONS_FILE, log_entry)
    except Exception as e:
        logger.warning("Failed to log agent completion: %s", e)

    logger.info(
        "Agent completed: status=%s exit=%d duration=%.1fs tier=%s journal=%s telegram=%s",
        status, exit_code, duration_s, _agent_tier, journal_written, telegram_sent,
    )

    # Telegram alert on any agent failure (not just stack overflow)
    if status == "failed":
        try:
            config = _load_config()
            send_or_store(
                f"*L2 FAILED* T{_agent_tier} exit={exit_code} "
                f"({duration_s:.0f}s) journal={journal_written} tg={telegram_sent}",
                config, category="error",
            )
        except Exception as e:
            logger.warning("Agent failure alert failed: %s", e)

    # Track consecutive stack overflow crashes
    global _consecutive_stack_overflows
    if exit_code == _STACK_OVERFLOW_EXIT_CODE:
        _consecutive_stack_overflows += 1
        logger.error(
            "Claude CLI stack overflow (exit %d), %d consecutive. "
            "Check project root for problematic files or update Claude Code.",
            exit_code, _consecutive_stack_overflows,
        )
        if _consecutive_stack_overflows == _MAX_STACK_OVERFLOWS:
            logger.error(
                "Layer 2 auto-disabled after %d consecutive stack overflows",
                _MAX_STACK_OVERFLOWS,
            )
            try:
                config = _load_config()
                send_or_store(
                    f"*ALERT* Layer 2 auto-disabled after {_MAX_STACK_OVERFLOWS} "
                    f"consecutive stack overflows (exit {exit_code}). "
                    "Claude CLI is crashing — investigate project root.",
                    config, category="alert",
                )
            except Exception as e:
                logger.warning("Stack overflow alert failed: %s", e)
    else:
        # BUG-95: Reset counter on any non-stack-overflow completion (success or otherwise).
        # This prevents false positive auto-disable when the consecutive chain is broken.
        _consecutive_stack_overflows = 0

    # Clean up
    if _agent_log:
        try:
            _agent_log.close()
        except Exception as e:
            logger.debug("Agent log close failed: %s", e)
    _agent_proc = None
    _agent_log = None
    _agent_start = 0
    _agent_tier = None
    _agent_reasons = None
    _journal_ts_before = None
    _telegram_ts_before = None

    return result


def get_completion_stats(hours=24):
    """Compute rolling completion stats from the invocations log.

    Args:
        hours: Number of hours to look back (default 24).

    Returns:
        dict with keys: total, success, incomplete, failed, completion_rate.
        Returns zeroed stats if no data is available.
    """
    entries = load_jsonl(INVOCATIONS_FILE)
    cutoff = datetime.now(UTC).timestamp() - (hours * 3600)

    total = 0
    success = 0
    incomplete = 0
    failed = 0

    for entry in entries:
        # Only count entries that have a completion status (not "invoked" or "skipped_gate")
        entry_status = entry.get("status", "")
        if entry_status not in ("success", "incomplete", "failed"):
            continue

        ts_str = entry.get("ts", "")
        if not ts_str:
            continue

        try:
            # Parse ISO-8601 timestamp
            ts_str_clean = ts_str.replace("+00:00", "+0000").replace("Z", "+0000")
            if "+" in ts_str_clean[10:]:
                dt = datetime.fromisoformat(ts_str)
            else:
                dt = datetime.fromisoformat(ts_str).replace(tzinfo=UTC)
            entry_ts = dt.timestamp()
        except (ValueError, TypeError):
            continue

        if entry_ts < cutoff:
            continue

        total += 1
        if entry_status == "success":
            success += 1
        elif entry_status == "incomplete":
            incomplete += 1
        elif entry_status == "failed":
            failed += 1

    completion_rate = (success / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "success": success,
        "incomplete": incomplete,
        "failed": failed,
        "completion_rate": round(completion_rate, 1),
    }
