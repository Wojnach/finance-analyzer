"""Layer 2 agent invocation — manages the Claude Code subprocess lifecycle."""

import logging
import os
import platform
import shutil
import subprocess
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from portfolio.api_utils import load_config as _load_config
from portfolio.claude_gate import detect_auth_failure
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
_agent_log_start_offset = 0  # byte offset of agent.log at invoke time, for auth-error scan on completion
_agent_start = 0
_agent_timeout = 900  # per-invocation timeout (set from tier config)
_agent_tier = None  # tier of the currently running agent
_agent_reasons = None  # trigger reasons for the current invocation
_journal_ts_before = None  # last journal timestamp before agent started
_telegram_ts_before = None  # last telegram timestamp before agent started

# Stack overflow detection — exit code 3221225794 = Windows STATUS_STACK_OVERFLOW (0xC00000FD)
_STACK_OVERFLOW_EXIT_CODE = 3221225794
_MAX_STACK_OVERFLOWS = 5  # auto-disable after this many consecutive stack overflow crashes
_STACK_OVERFLOW_FILE = DATA_DIR / "stack_overflow_counter.json"


def _load_stack_overflow_counter() -> int:
    """Load persisted stack overflow counter. Returns 0 if missing/corrupt."""
    from portfolio.file_utils import load_json
    data = load_json(_STACK_OVERFLOW_FILE)
    if data and isinstance(data.get("count"), int):
        return data["count"]
    return 0


def _save_stack_overflow_counter(count: int) -> None:
    """Persist stack overflow counter to survive loop restarts."""
    from portfolio.file_utils import atomic_write_json
    atomic_write_json(_STACK_OVERFLOW_FILE, {
        "count": count,
        "updated": datetime.now(UTC).isoformat(),
    })


_consecutive_stack_overflows = _load_stack_overflow_counter()

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


def _extract_ticker(reasons):
    """Extract the primary ticker from trigger reasons.

    Looks for common ticker patterns like 'XAG-USD', 'BTC-USD', 'NVDA'.
    Falls back to 'XAG-USD' if no ticker found.
    """
    import re
    for r in reasons:
        # Match patterns like XAG-USD, BTC-USD, ETH-USD
        m = re.search(r'\b([A-Z]{2,5}-USD)\b', r)
        if m:
            return m.group(1)
        # Match stock tickers like NVDA, PLTR
        m = re.search(r'\b([A-Z]{2,5})\b(?:\s+flipped|\s+crossed|\s+broke)', r)
        if m:
            return m.group(1)
    return "XAG-USD"  # default to silver


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
        logger.warning("%s baseline read failed: %s", label, e)
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
        # BUG-203: use monotonic clock for elapsed — wall clock is NTP-jump-prone.
        elapsed = time.monotonic() - _agent_start
        if elapsed > _agent_timeout:
            logger.info("Agent pid=%s timed out (%.0fs), killing", _agent_proc.pid, elapsed)
            kill_ok = True
            if platform.system() == "Windows":
                # BUG-92: Check taskkill return code to detect kill failure
                # BUG-189: rc=128 means process already exited — treat as success
                result = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(_agent_proc.pid)],
                    capture_output=True,
                )
                if result.returncode not in (0, 128):
                    logger.error(
                        "taskkill failed (rc=%d): %s",
                        result.returncode, result.stderr.decode(errors="replace").strip(),
                    )
                    kill_ok = False
                elif result.returncode == 128:
                    logger.info("Agent pid=%s already exited (rc=128)", _agent_proc.pid)
            else:
                _agent_proc.kill()
            try:
                _agent_proc.wait(timeout=15)  # BUG-189: 15s for Claude CLI Node.js teardown
            except subprocess.TimeoutExpired:
                if kill_ok:
                    logger.error("Agent pid=%s did not exit after kill+15s wait", _agent_proc.pid)
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

    # Multi-agent mode: parallel specialists + synthesis (Coordinator Mode pattern)
    # Enabled via config.layer2.multi_agent = true, only for T2/T3
    try:
        config = _load_config()
        multi_agent = config.get("layer2", {}).get("multi_agent", False)
    except Exception:
        multi_agent = False

    if multi_agent and tier >= 2:
        try:
            from portfolio.multi_agent_layer2 import (
                build_synthesis_prompt,
                launch_specialists,
                wait_for_specialists,
            )
            # Extract primary ticker from reasons
            ticker = _extract_ticker(reasons)
            logger.info("Multi-agent T%d: launching 3 specialists for %s", tier, ticker)
            procs = launch_specialists(ticker, reasons)
            if procs:
                # C3/NEW-1: timeout reduced from 150s to 30s (configurable via
                # layer2.specialist_timeout_s) to avoid blocking the main loop.
                # TODO: run specialists in background thread, collect results async.
                specialist_timeout = config.get("layer2", {}).get("specialist_timeout_s", 30)
                results = wait_for_specialists(procs, timeout=specialist_timeout)
                success_count = sum(1 for v in results.values() if v)
                logger.info("Specialists complete: %d/%d succeeded", success_count, len(results))
                # Even if some fail, proceed with synthesis using available reports
                prompt = build_synthesis_prompt(ticker, reasons)
                # Fall through to normal agent launch with synthesis prompt
            else:
                logger.warning("No specialists launched, falling back to single-agent")
                prompt = _build_tier_prompt(tier, reasons)
        except Exception as e:
            logger.warning("Multi-agent failed (%s), falling back to single-agent", e)
            prompt = _build_tier_prompt(tier, reasons)
    else:
        prompt = _build_tier_prompt(tier, reasons)

    max_turns = tier_cfg["max_turns"]

    # Try direct claude invocation first; fall back to bat file for T3
    claude_cmd = shutil.which("claude")
    if claude_cmd:
        # 2026-04-13: DO NOT add `--bare`. It disables OAuth/keychain auth
        # and only accepts ANTHROPIC_API_KEY. This user runs on a Max
        # subscription with no API key, so `--bare` silently breaks every
        # invocation ("Not logged in" to stdout, exit 0). Commit b4bb57d
        # added it on 2026-03-27; removed on 2026-04-13 after 3 weeks of
        # silent Layer 2 failures. See portfolio/claude_gate.py
        # (detect_auth_failure) for the runtime guard.
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
        agent_log_path = DATA_DIR / "agent.log"
        # Capture the current file size BEFORE opening in append mode, so
        # check_agent_completion() can read only this invocation's output
        # (for auth-error detection) and not the entire log history.
        global _agent_log_start_offset
        _agent_log_start_offset = agent_log_path.stat().st_size if agent_log_path.exists() else 0
        log_fh = open(agent_log_path, "a", encoding="utf-8")
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
        _agent_start = time.monotonic()
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
            logger.warning("invocation notification failed: %s", e)
        return True
    except Exception as e:
        logger.error("invoking agent: %s", e)
        if log_fh is not None:
            log_fh.close()
        return False


def _write_fishing_context(journal_entry):
    """Extract fishing context from Layer 2 journal entry.

    Called after Layer 2 completes. Creates a structured context file
    that the fish engine reads as its strongest tactic vote.
    """
    try:
        tickers = journal_entry.get('tickers', {})
        xag = tickers.get('XAG-USD')
        if not xag:
            return

        outlook = xag.get('outlook', '')
        conviction = float(xag.get('conviction', 0))
        levels = xag.get('levels', [])
        thesis = xag.get('thesis', '')

        # Determine direction bias
        if outlook == 'bullish' and conviction >= 0.4:
            direction_bias = 'bullish'
            tactic_vote = 'LONG'
            allow_long = True
            allow_short = conviction < 0.6  # block short only if very bullish
        elif outlook == 'bearish' and conviction >= 0.4:
            direction_bias = 'bearish'
            tactic_vote = 'SHORT'
            allow_long = conviction < 0.6
            allow_short = True
        else:
            direction_bias = 'neutral'
            tactic_vote = None
            allow_long = True
            allow_short = True

        # Check for event context from watchlist
        watchlist = journal_entry.get('watchlist', [])
        event_context = ''
        for item in watchlist:
            if isinstance(item, str) and any(
                w in item.lower() for w in ['event', 'fomc', 'cpi', 'tariff', 'opec']
            ):
                event_context = item[:100]
                break

        # Determine position size multiplier from regime
        regime = journal_entry.get('regime', 'ranging')
        if regime == 'high-vol':
            position_size_multiplier = 0.5
        elif regime in ('trending-up', 'trending-down'):
            position_size_multiplier = 1.0
        else:
            position_size_multiplier = 0.75  # ranging = slightly reduced

        context = {
            'timestamp': journal_entry.get('ts', ''),
            'valid_until': '',  # fish engine uses 4h staleness check
            'ticker': 'XAG-USD',
            'direction_bias': direction_bias,
            'bias_confidence': conviction,
            'bias_reasoning': thesis[:200] if thesis else '',
            'allow_long': allow_long,
            'allow_short': allow_short,
            'max_hold_minutes': 120,
            'position_size_multiplier': position_size_multiplier,
            'allow_overnight': conviction >= 0.6 and outlook == 'bullish',
            'event_context': event_context,
            'bull_case': '',
            'bear_case': '',
            'journal_action': '',
            'journal_confidence': conviction,
            'tactic_vote': tactic_vote,
            'tactic_weight': 2.0,
            'levels': levels,
        }

        # Extract bull/bear cases from decisions
        decisions = journal_entry.get('decisions', {})
        for strategy in ('patient', 'bold'):
            dec = decisions.get(strategy, {})
            reasoning = dec.get('reasoning', '')
            action = dec.get('action', 'HOLD')
            if action != 'HOLD':
                context['journal_action'] = action
            if reasoning:
                if not context['bull_case'] and 'bullish' in reasoning.lower():
                    context['bull_case'] = reasoning[:150]
                elif not context['bear_case'] and (
                    'bearish' in reasoning.lower() or 'sell' in reasoning.lower()
                ):
                    context['bear_case'] = reasoning[:150]

        from portfolio.file_utils import atomic_write_json

        # H22/NEW-3: use DATA_DIR absolute path instead of relative 'data/...'
        atomic_write_json(DATA_DIR / 'fishing_context.json', context)

    except Exception as e:
        logger.warning('Fishing context error: %s', e)
        # BUG-181: Write neutral context on failure to prevent stale bias
        try:
            from datetime import UTC, datetime

            from portfolio.file_utils import atomic_write_json
            atomic_write_json(DATA_DIR / 'fishing_context.json', {
                'timestamp': datetime.now(UTC).isoformat(),
                'ticker': 'XAG-USD',
                'direction_bias': 'neutral',
                'bias_confidence': 0.0,
                'bias_reasoning': f'Context extraction failed: {e}',
                'allow_long': True,
                'allow_short': True,
                'tactic_vote': None,
                'tactic_weight': 0.0,
            })
        except Exception:
            pass  # last resort: can't even write neutral


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
    duration_s = round(time.monotonic() - _agent_start, 1)
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

    # 2026-04-13: Scan agent.log for auth-error markers (see claude_gate.py
    # detect_auth_failure). Claude CLI can exit 0 while printing "Not logged
    # in" to stdout — that's exactly the 3-week silent Layer 2 outage that
    # motivated this detection. We captured _agent_log_start_offset before
    # spawning the subprocess, so we only scan output from this invocation.
    auth_error_detected = False
    try:
        agent_log_path = DATA_DIR / "agent.log"
        if agent_log_path.exists():
            with open(agent_log_path, "rb") as f:
                f.seek(_agent_log_start_offset)
                new_output = f.read().decode("utf-8", errors="replace")
            auth_error_detected = detect_auth_failure(
                new_output,
                caller=f"layer2_t{_agent_tier}",
                context={
                    "tier": _agent_tier,
                    "exit_code": exit_code,
                    "duration_s": duration_s,
                    "reasons": (_agent_reasons or [])[:5],
                },
            )
    except Exception as e:
        logger.warning("Auth-error scan of agent.log failed: %s", e)

    # Determine status
    if auth_error_detected:
        status = "auth_error"
    elif exit_code != 0:
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

    # Post-process: extract fishing context from journal for metals fish engine
    if journal_written:
        with suppress(Exception):
            new_journal_entry = last_jsonl_entry(JOURNAL_FILE)
            if new_journal_entry:
                _write_fishing_context(new_journal_entry)

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
        _save_stack_overflow_counter(_consecutive_stack_overflows)
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
        if _consecutive_stack_overflows > 0:
            _consecutive_stack_overflows = 0
            _save_stack_overflow_counter(0)

    # Clean up
    if _agent_log:
        try:
            _agent_log.close()
        except Exception as e:
            logger.warning("Agent log close failed: %s", e)
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
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
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
