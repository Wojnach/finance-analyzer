"""Layer 2 agent invocation — manages the Claude Code subprocess lifecycle."""

import json
import logging
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from portfolio.api_utils import load_config as _load_config
from portfolio.file_utils import atomic_append_jsonl
from portfolio.message_store import send_or_store
from portfolio.telegram_notifications import escape_markdown_v1

logger = logging.getLogger("portfolio.agent")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INVOCATIONS_FILE = DATA_DIR / "invocations.jsonl"

_agent_proc = None
_agent_log = None
_agent_start = 0
_agent_timeout = 900  # per-invocation timeout (set from tier config)

# Per-tier configuration
TIER_CONFIG = {
    1: {"max_turns": 15, "timeout": 120, "label": "QUICK CHECK"},
    2: {"max_turns": 25, "timeout": 300, "label": "SIGNAL ANALYSIS"},
    3: {"max_turns": 40, "timeout": 900, "label": "FULL REVIEW"},
}


def _build_tier_prompt(tier, reasons):
    """Build a tier-specific prompt for the Claude Code agent."""
    reason_str = ", ".join(reasons[:5])

    if tier == 1:
        return (
            "You are the Layer 2 trading agent (QUICK CHECK). "
            f"Trigger: {reason_str}. "
            "Read data/layer2_context.md then data/agent_context_t1.json. "
            "This is a routine check. Confirm held positions are OK (check ATR stops). "
            "If no positions are held, briefly assess macro state. "
            "Write a brief journal entry and send a short Telegram message. "
            "Do NOT analyze all tickers — focus only on held positions and macro headline."
        )
    elif tier == 2:
        return (
            "You are the Layer 2 trading agent (SIGNAL ANALYSIS). "
            f"Trigger: {reason_str}. "
            "Read data/layer2_context.md, then data/agent_context_t2.json, "
            "data/portfolio_state.json, and data/portfolio_state_bold.json. "
            "Analyze triggered tickers and held positions. Decide for BOTH strategies. "
            "Write journal entry and send Telegram per CLAUDE.md instructions."
        )
    else:
        # Tier 3 — full review, same as the original pf-agent.bat prompt
        return (
            "You are the Layer 2 trading agent. "
            "FIRST read data/layer2_context.md (your memory from previous invocations). "
            "Then read data/agent_summary_compact.json (signals, trigger reasons, timeframes), "
            "data/portfolio_state.json (Patient portfolio), and data/portfolio_state_bold.json "
            "(Bold portfolio). Follow the instructions in CLAUDE.md to analyze, decide, and act "
            "for BOTH strategies independently. Compare your previous theses and prices with "
            "current data — were you right? Always write a journal entry and send a Telegram message."
        )


def _log_trigger(reasons, status, tier=None):
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "reasons": reasons,
        "status": status,
    }
    if tier is not None:
        entry["tier"] = tier
    atomic_append_jsonl(INVOCATIONS_FILE, entry)


def invoke_agent(reasons, tier=3):
    global _agent_proc, _agent_log, _agent_start, _agent_timeout

    tier_cfg = TIER_CONFIG.get(tier, TIER_CONFIG[3])
    timeout = tier_cfg["timeout"]

    if _agent_proc and _agent_proc.poll() is None:
        elapsed = time.time() - _agent_start
        if elapsed > _agent_timeout:
            logger.info("Agent pid=%s timed out (%.0fs), killing", _agent_proc.pid, elapsed)
            if platform.system() == "Windows":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(_agent_proc.pid)],
                    capture_output=True,
                )
            else:
                _agent_proc.kill()
            try:
                _agent_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            if _agent_log:
                _agent_log.close()
                _agent_log = None
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
        except Exception:
            pass  # non-critical
        return True
    except Exception as e:
        logger.error("invoking agent: %s", e)
        if log_fh is not None:
            log_fh.close()
        return False
