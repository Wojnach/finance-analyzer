"""Wrapper to call Ministral-8B trading model via subprocess.

Only runs the original CryptoTrader-LM LoRA.  The custom LoRA has been
fully disabled (20.9% accuracy, 97% SELL bias — worse than random).
Shadow A/B testing data is preserved in data/ab_test_log.jsonl.
"""

import json
import platform
import subprocess
from pathlib import Path


def _call_model(context, lora_path=None):
    if platform.system() == "Windows":
        python = r"Q:\models\.venv-llm\Scripts\python.exe"
        script = r"Q:\models\ministral_trader.py"
    else:
        python = "/home/deck/models/.venv-llm/bin/python"
        script = "/home/deck/models/ministral_trader.py"

    cmd = [python, script]
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
    return json.loads(result.stdout.strip())


def get_ministral_signal(context):
    original = _call_model(context)
    return {"original": original, "custom": None}
