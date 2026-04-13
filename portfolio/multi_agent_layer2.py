"""Multi-agent Layer 2 orchestration — parallel specialists + synthesis.

Inspired by Claude Code's Coordinator Mode. Instead of one monolithic
agent reading everything, splits analysis into parallel specialists:

    1. Technical Agent: signals, regime, momentum, trend
    2. Risk Agent: portfolio state, exposure, drawdown, stops
    3. Microstructure Agent: order flow, depth, cross-asset context

Each specialist writes a brief report to a temp file. A synthesis agent
reads all three and makes the final BUY/SELL/HOLD decision.

Key design principles (from Claude Code's Agent architecture):
    - Fresh context per agent (no context pollution)
    - 5-word task description forces clarity
    - Standardized report format for mechanical parsing
    - Parent owns the gate — synthesis agent makes final call
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

from portfolio.claude_gate import detect_auth_failure

logger = logging.getLogger("portfolio.multi_agent_layer2")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

SPECIALISTS = {
    "technical": {
        "focus": "Technical analysis: signals, regime, momentum, trend direction",
        "data_files": [
            "data/agent_context_t2.json",
            "data/accuracy_cache.json",
        ],
        "output_file": "data/_specialist_technical.md",
        "timeout": 120,
        "max_turns": 10,
    },
    "risk": {
        "focus": "Risk assessment: portfolio exposure, drawdown, stop levels, position sizing",
        "data_files": [
            "data/portfolio_state.json",
            "data/portfolio_state_bold.json",
            "data/portfolio_state_warrants.json",
        ],
        "output_file": "data/_specialist_risk.md",
        "timeout": 90,
        "max_turns": 8,
    },
    "microstructure": {
        "focus": "Order flow and cross-asset: depth imbalance, trade flow, VPIN, copper, GVZ, gold/silver ratio",
        "data_files": [
            "data/microstructure_state.json",
            "data/seasonality_profiles.json",
        ],
        "output_file": "data/_specialist_microstructure.md",
        "timeout": 90,
        "max_turns": 8,
    },
}


def build_specialist_prompts(
    ticker: str,
    trigger_reasons: list[str],
) -> dict[str, str]:
    """Build prompts for each specialist agent.

    Returns dict keyed by specialist name with prompt strings.
    """
    reason_str = ", ".join(trigger_reasons[:5])
    prompts = {}

    for name, spec in SPECIALISTS.items():
        data_reads = " ".join(f"Read {f}." for f in spec["data_files"])
        prompts[name] = (
            f"You are a {name} specialist for the trading system. "
            f"Ticker: {ticker}. Trigger: {reason_str}. "
            f"Focus: {spec['focus']}. "
            f"{data_reads} "
            f"Write a brief analysis (max 500 words) to {spec['output_file']}. "
            "Include: current state, key signals, recommendation "
            "(bullish/bearish/neutral), and confidence (low/medium/high). "
            "Be concise and data-driven. Do NOT make trade decisions."
        )

    return prompts


def build_synthesis_prompt(
    ticker: str,
    trigger_reasons: list[str],
    report_paths: list[str] | None = None,
) -> str:
    """Build the synthesis agent prompt that reads all specialist reports."""
    reason_str = ", ".join(trigger_reasons[:5])
    if report_paths is None:
        report_paths = get_report_paths()
    reads = " ".join(f"Read {p}." for p in report_paths)

    return (
        "You are the Layer 2 synthesis agent. "
        f"Ticker: {ticker}. Trigger: {reason_str}. "
        "Read docs/TRADING_PLAYBOOK.md for trading rules. "
        "If data/trading_insights.md exists, read it for recent performance context. "
        f"{reads} "
        "These are reports from 3 specialist agents (technical, risk, microstructure). "
        "Synthesize their findings into a trading decision for BOTH Patient and Bold strategies. "
        "If specialists disagree, explain why you sided with one over the other. "
        "Read data/portfolio_state.json and data/portfolio_state_bold.json for current positions. "
        "Write journal entry and send Telegram per the playbook."
    )


def get_report_paths() -> list[str]:
    """Get output file paths for all specialists."""
    return [spec["output_file"] for spec in SPECIALISTS.values()]


def launch_specialists(
    ticker: str,
    trigger_reasons: list[str],
) -> list[subprocess.Popen]:
    """Launch all specialist agents in parallel.

    Returns list of Popen processes. Caller must wait for them.
    """
    prompts = build_specialist_prompts(ticker, trigger_reasons)
    claude_cmd = shutil.which("claude")
    if not claude_cmd:
        logger.warning("claude not on PATH, cannot launch specialists")
        return []

    procs = []
    agent_env = os.environ.copy()
    agent_env.pop("CLAUDECODE", None)
    agent_env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    agent_env["NODE_OPTIONS"] = "--stack-size=16384"

    for name, prompt in prompts.items():
        spec = SPECIALISTS[name]
        # 2026-04-13: DO NOT add `--bare` here either. Same reason as
        # agent_invocation.py: `--bare` disables OAuth/keychain auth and
        # requires ANTHROPIC_API_KEY. User runs Max-subscription OAuth only.
        # Commit 857fd45 (2026-04-01) added `--bare` to specialist launches;
        # removed 2026-04-13 after confirming it broke all specialist runs.
        cmd = [
            claude_cmd, "-p", prompt,
            "--allowedTools", "Read,Write",
            "--max-turns", str(spec["max_turns"]),
        ]
        try:
            log_path = DATA_DIR / f"_specialist_{name}.log"
            log_fh = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=agent_env,
            )
            proc._log_fh = log_fh  # attach for cleanup
            proc._name = name
            procs.append(proc)
            logger.info("Specialist %s launched pid=%s", name, proc.pid)
        except Exception as e:
            logger.error("Failed to launch specialist %s: %s", name, e)

    return procs


def wait_for_specialists(
    procs: list[subprocess.Popen],
    timeout: int = 150,
) -> dict[str, bool]:
    """Wait for all specialist agents to complete.

    Returns dict of specialist_name -> success (True/False).
    """
    results = {}
    deadline = time.time() + timeout

    for proc in procs:
        remaining = max(1, deadline - time.time())
        name = getattr(proc, "_name", "unknown")
        try:
            proc.wait(timeout=remaining)
            success = proc.returncode == 0
            results[name] = success
            if not success:
                logger.warning("Specialist %s exited with code %d", name, proc.returncode)
        except subprocess.TimeoutExpired:
            logger.warning("Specialist %s timed out, killing", name)
            proc.kill()
            proc.wait(timeout=5)
            results[name] = False
        finally:
            log_fh = getattr(proc, "_log_fh", None)
            if log_fh:
                log_fh.close()

        # 2026-04-13: Auth-error scan — specialist log is truncated per run
        # ("w" mode in launch_specialists), so reading the whole file is safe.
        # Override success to False if auth failure detected so synthesis
        # doesn't proceed with an empty specialist report masquerading as OK.
        try:
            log_path = DATA_DIR / f"_specialist_{name}.log"
            if log_path.exists():
                text = log_path.read_text(encoding="utf-8", errors="replace")
                if detect_auth_failure(text, caller=f"layer2_specialist_{name}",
                                       context={"specialist": name}):
                    results[name] = False
        except Exception as e:
            logger.warning("Auth-error scan of specialist %s log failed: %s", name, e)

    return results


def cleanup_reports() -> None:
    """Remove specialist report files after synthesis."""
    for spec in SPECIALISTS.values():
        path = BASE_DIR / spec["output_file"]
        if path.exists():
            path.unlink()
    # Also clean up log files
    for name in SPECIALISTS:
        log_path = DATA_DIR / f"_specialist_{name}.log"
        if log_path.exists():
            log_path.unlink()
