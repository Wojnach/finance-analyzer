"""Wrapper to call Ministral trading model.

Prefers persistent llama-server (HTTP) to eliminate cold-start CPU cost.
Falls back to subprocess llama-completion if server unavailable.
Uses GPU gate to ensure exclusive GPU access during model load/inference.
"""

import json
import logging
import platform
from contextlib import suppress
from pathlib import Path

from portfolio.gpu_gate import gpu_gate
from portfolio.llama_server import query_llama_server
from portfolio.subprocess_utils import kill_orphaned_llama, run_safe

logger = logging.getLogger("portfolio.ministral_signal")


def _extract_json_from_stdout(stdout):
    """Extract the first JSON object from subprocess stdout."""
    if not stdout:
        return None
    text = stdout.strip()
    if not text:
        return None
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _call_model(context, lora_path=None):
    # Try persistent llama-server first (no cold-start cost)
    from portfolio.ministral_trader import _build_prompt, _parse_response
    prompt = _build_prompt(context)

    text = query_llama_server("ministral3", prompt, stop=["[INST]"])
    if text is not None:
        decision, reasoning, confidence = _parse_response(text)
        result = {"action": decision, "reasoning": reasoning, "model": "Ministral-3-8B"}
        if confidence is not None:
            result["confidence"] = confidence
        return result

    # Fallback: subprocess (cold start)
    logger.info("llama-server unavailable, falling back to subprocess")
    repo_root = Path(__file__).resolve().parent.parent
    if platform.system() == "Windows":
        python = r"Q:\models\.venv-llm\Scripts\python.exe"
        script = repo_root / "portfolio" / "ministral_trader.py"
    else:
        python = "/home/deck/models/.venv-llm/bin/python"
        script = repo_root / "portfolio" / "ministral_trader.py"

    cmd = [python, str(script)]
    if lora_path:
        cmd.extend(["--lora", str(lora_path)])

    result = run_safe(
        cmd,
        input=json.dumps(context),
        capture_output=True,
        text=True,
        timeout=240,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Ministral failed: {result.stderr[-500:]}")
    payload = _extract_json_from_stdout(result.stdout)
    if payload is None:
        raise RuntimeError(f"Ministral returned invalid JSON: {result.stdout[-500:]}")
    return payload


def get_ministral_signal(context):
    """Get trading signal from Ministral with GPU gating."""
    with suppress(Exception):
        killed = kill_orphaned_llama()
        if killed:
            logger.warning("Reaped %d orphaned llama process(es)", killed)
    with gpu_gate("ministral", timeout=300) as acquired:
        if not acquired:
            logger.warning("GPU gate timeout — returning HOLD")
            return {"original": {"action": "HOLD", "reasoning": "GPU busy", "model": "skipped"}, "custom": None}
        original = _call_model(context)
    return {"original": original, "custom": None}
