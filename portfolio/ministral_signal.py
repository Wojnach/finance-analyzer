"""Wrapper to call Ministral-8B trading model via subprocess.

Supports A/B shadow testing: runs both original and custom LoRA,
logs both results, returns original's decision as the production signal.
"""

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
SHADOW_CONFIG = REPO_DIR / "data" / "shadow_lora_config.json"


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
    custom = None

    try:
        if SHADOW_CONFIG.exists():
            cfg = json.loads(SHADOW_CONFIG.read_text(encoding="utf-8"))
            if cfg.get("enabled"):
                custom = _call_model(context, lora_path=cfg["custom_lora"])
                log_entry = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "ticker": context.get("ticker", "?"),
                    "original": original,
                    "custom": custom,
                    "agree": original["action"] == custom["action"],
                }
                log_file = Path(
                    cfg.get("log_file", REPO_DIR / "data" / "ab_test_log.jsonl")
                )
                with open(log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass

    return {"original": original, "custom": custom}
