"""Wrapper to call Ministral-8B trading model via subprocess."""

import json
import platform
import subprocess


def get_ministral_signal(context):
    if platform.system() == "Windows":
        python = r"Q:\models\.venv-llm\Scripts\python.exe"
        script = r"Q:\models\ministral_trader.py"
    else:
        python = "/home/deck/models/.venv-llm/bin/python"
        script = "/home/deck/models/ministral_trader.py"

    result = subprocess.run(
        [python, script],
        input=json.dumps(context),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Ministral failed: {result.stderr[-500:]}")
    return json.loads(result.stdout.strip())
