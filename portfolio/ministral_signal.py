"""Wrapper to call Ministral-8B trading model via subprocess.

Only runs the original CryptoTrader-LM LoRA.  The custom LoRA has been
fully disabled (20.9% accuracy, 97% SELL bias — worse than random).
Shadow A/B testing data is preserved in data/ab_test_log.jsonl.
"""

import json
import platform
import subprocess
from pathlib import Path


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

    result = subprocess.run(
        cmd,
        input=json.dumps(context),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Ministral failed: {result.stderr[-500:]}")
    payload = _extract_json_from_stdout(result.stdout)
    if payload is None:
        raise RuntimeError(f"Ministral returned invalid JSON: {result.stdout[-500:]}")
    return payload


def get_ministral_signal(context):
    original = _call_model(context)
    return {"original": original, "custom": None}
