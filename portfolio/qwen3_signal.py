"""Wrapper to call Qwen3-8B trading model via subprocess.

Runs for ALL tickers (crypto, stocks, metals). Uses GPU lock to coordinate
with Ministral (only one GGUF model can be on GPU at a time).
"""

import json
import logging
import platform
import subprocess
from pathlib import Path

logger = logging.getLogger("portfolio.qwen3_signal")


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


def _call_qwen3(context):
    """Call Qwen3-8B inference subprocess."""
    repo_root = Path(__file__).resolve().parent.parent
    if platform.system() == "Windows":
        python = r"Q:\models\.venv-llm\Scripts\python.exe"
    else:
        python = "/home/deck/models/.venv-llm/bin/python"

    script = repo_root / "portfolio" / "qwen3_trader.py"
    cmd = [python, str(script)]

    result = subprocess.run(
        cmd,
        input=json.dumps(context),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Qwen3 failed: {result.stderr[-500:]}")
    payload = _extract_json_from_stdout(result.stdout)
    if payload is None:
        raise RuntimeError(f"Qwen3 returned invalid JSON: {result.stdout[-500:]}")
    return payload


def get_qwen3_signal(context):
    """Get trading signal from Qwen3-8B.

    Returns dict with 'action', 'reasoning', 'model' keys.
    """
    return _call_qwen3(context)
