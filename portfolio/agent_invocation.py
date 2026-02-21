"""Layer 2 agent invocation — manages the Claude Code subprocess lifecycle."""

import json
import logging
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from portfolio.api_utils import load_config as _load_config
from portfolio.telegram_notifications import send_telegram, escape_markdown_v1

logger = logging.getLogger("portfolio.agent")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INVOCATIONS_FILE = DATA_DIR / "invocations.jsonl"

_agent_proc = None
_agent_log = None
_agent_start = 0
AGENT_TIMEOUT = 600


def _log_trigger(reasons, status):
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "reasons": reasons,
        "status": status,
    }
    with open(INVOCATIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def invoke_agent(reasons):
    global _agent_proc, _agent_log, _agent_start
    if _agent_proc and _agent_proc.poll() is None:
        elapsed = time.time() - _agent_start
        if elapsed > AGENT_TIMEOUT:
            logger.info(f"Agent pid={_agent_proc.pid} timed out ({elapsed:.0f}s), killing")
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
                f"Agent still running (pid {_agent_proc.pid}, {elapsed:.0f}s), skipping"
            )
            return False

    if _agent_log:
        _agent_log.close()
        _agent_log = None

    try:
        from portfolio.journal import write_context

        n = write_context()
        logger.info(f"Layer 2 context: {n} journal entries")
    except Exception as e:
        logger.warning(f"journal context failed: {e}")

    agent_bat = BASE_DIR / "scripts" / "win" / "pf-agent.bat"
    if not agent_bat.exists():
        logger.warning(f"Agent script not found at {agent_bat}")
        return False
    try:
        _agent_log = open(DATA_DIR / "agent.log", "a", encoding="utf-8")
        # Strip Claude Code session markers to avoid "nested session" error
        # when the parent process tree has Claude Code running
        agent_env = os.environ.copy()
        agent_env.pop("CLAUDECODE", None)
        agent_env.pop("CLAUDE_CODE_ENTRYPOINT", None)
        _agent_proc = subprocess.Popen(
            ["cmd", "/c", str(agent_bat)],
            cwd=str(BASE_DIR),
            stdout=_agent_log,
            stderr=subprocess.STDOUT,
            env=agent_env,
        )
        _agent_start = time.time()
        logger.info(f"Agent invoked pid={_agent_proc.pid} ({', '.join(reasons)})")
        # Send brief Telegram notification that Layer 2 was triggered
        try:
            config = _load_config()
            reason_str = escape_markdown_v1(", ".join(reasons[:3]))
            if len(reasons) > 3:
                reason_str += f" (+{len(reasons) - 3} more)"
            notify_msg = f"_Layer 2 invoked: {reason_str}_"
            send_telegram(notify_msg, config)
        except Exception:
            pass  # non-critical
        return True
    except Exception as e:
        logger.error(f"invoking agent: {e}")
        return False
