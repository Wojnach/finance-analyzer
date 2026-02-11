#!/usr/bin/env python3
"""LoRA Training Pipeline -- Crash-resistant, resumable, monitorable.

Usage:
    python pipeline.py --run        # Execute next incomplete step
    python pipeline.py --run-all    # Execute all remaining steps
    python pipeline.py --status     # Show current progress
    python pipeline.py --step N     # Run specific step (0-indexed)
    python pipeline.py --reset      # Reset state (start over)
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
STATE_FILE = PIPELINE_DIR / "state.json"
LOGS_DIR = PIPELINE_DIR / "logs"
OUTPUT_DIR = PIPELINE_DIR / "output"

REPO_DIR = PIPELINE_DIR.parent.parent
VENV_TRAIN = REPO_DIR / ".venv-train"
FEATHER_DIR = REPO_DIR / "user_data" / "data" / "binance" / "futures"
TRAINING_DATA = PIPELINE_DIR / "training_data.jsonl"

if os.name == "nt":
    VENV_PYTHON = VENV_TRAIN / "Scripts" / "python.exe"
    MAIN_PYTHON = REPO_DIR / ".venv" / "Scripts" / "python.exe"
    HF_MODEL_DIR = Path(r"Q:\models\ministral-8b-hf")
    LORA_OUTPUT = OUTPUT_DIR / "final"
    FINAL_GGUF = Path(r"Q:\models\custom-trading-lora.gguf")
    BASE_GGUF = Path(
        r"Q:\models\ministral-8b-gguf\Ministral-8B-Instruct-2410-Q4_K_M.gguf"
    )
    LLM_PYTHON = Path(r"Q:\models\.venv-llm\Scripts\python.exe")
    ORIGINAL_LORA = Path(r"Q:\models\cryptotrader-lm\cryptotrader-lm-lora.gguf")
else:
    VENV_PYTHON = VENV_TRAIN / "bin" / "python"
    MAIN_PYTHON = REPO_DIR / ".venv" / "bin" / "python"
    HF_MODEL_DIR = Path.home() / "models" / "ministral-8b-hf"
    LORA_OUTPUT = OUTPUT_DIR / "final"
    FINAL_GGUF = Path.home() / "models" / "custom-trading-lora.gguf"
    BASE_GGUF = (
        Path.home()
        / "models"
        / "ministral-8b-gguf"
        / "Ministral-8B-Instruct-2410-Q4_K_M.gguf"
    )
    LLM_PYTHON = Path.home() / "models" / ".venv-llm" / "bin" / "python"
    ORIGINAL_LORA = (
        Path.home() / "models" / "cryptotrader-lm" / "cryptotrader-lm-lora.gguf"
    )

STEPS = [
    "setup_env",
    "generate_data",
    "download_model",
    "train",
    "convert_gguf",
    "verify",
    "deploy_shadow",
]
STEP_NAMES = {
    "setup_env": "Setup Training Environment",
    "generate_data": "Generate Training Data",
    "download_model": "Download HuggingFace Model",
    "train": "QLoRA Training",
    "convert_gguf": "Convert to GGUF",
    "verify": "Verify GGUF",
    "deploy_shadow": "Deploy Shadow Mode",
}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    state = {"started": now_iso(), "steps": {s: {"status": "pending"} for s in STEPS}}
    save_state(state)
    return state


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def update_step(state, step, status, **extra):
    state["steps"][step]["status"] = status
    state["steps"][step][status + "_at"] = now_iso()
    state["steps"][step].update(extra)
    save_state(state)


def log_path(step):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR / f"{step}.log"


def show_status(state):
    print("=" * 50)
    print("  LoRA Training Pipeline")
    print("=" * 50)

    progress_file = PIPELINE_DIR / "training_progress.json"
    tp = None
    if progress_file.exists():
        try:
            tp = json.loads(progress_file.read_text())
        except Exception:
            pass

    for i, step_id in enumerate(STEPS):
        info = state["steps"].get(step_id, {})
        status = info.get("status", "pending")
        name = STEP_NAMES.get(step_id, step_id)

        icons = {"done": "[OK]", "running": "[>>]", "failed": "[!!]"}
        icon = icons.get(status, "[  ]")
        line = f"  {icon} Step {i}: {name}"

        if status == "done" and "done_at" in info:
            line += f"  ({info['done_at'][:19]})"
        elif status == "running" and step_id == "train" and tp:
            ep, total_ep = tp.get("epoch", "?"), tp.get("total_epochs", "?")
            sn, total_s = tp.get("step", "?"), tp.get("total_steps", "?")
            line += f"  [epoch {ep}/{total_ep}, step {sn}/{total_s}] {tp.get('percent', 0):.0f}%"
        elif status == "running" and "running_at" in info:
            line += f"  (since {info['running_at'][:19]})"
        elif status == "failed":
            line += f"  ERROR: {info.get('error', 'unknown')}"

        print(line)
        if status == "done":
            if "examples" in info:
                print(f"         {info['examples']} training examples")
            if "size_mb" in info:
                print(f"         GGUF: {info['size_mb']} MB")
            if "test_output" in info:
                print(f"         Test: {info['test_output'][:80]}")

    done = sum(1 for s in STEPS if state["steps"].get(s, {}).get("status") == "done")
    print(f"\n  Progress: {done}/{len(STEPS)} steps complete")
    if done == len(STEPS):
        print("  STATUS: PIPELINE COMPLETE")
    print()


def run_next(state):
    from steps import STEP_FUNCS

    for step_id in STEPS:
        status = state["steps"].get(step_id, {}).get("status", "pending")
        if status in ("pending", "failed", "running"):
            name = STEP_NAMES.get(step_id, step_id)
            print(f"\n--- Step: {name} ---")
            ok = STEP_FUNCS[step_id](state)
            if not ok:
                print(f"\n  STOPPED: {name} failed. Fix the issue and run again.")
                return False
            return True
    print("\n  All steps complete!")
    return True


def run_all(state):
    while True:
        has_pending = any(
            state["steps"].get(s, {}).get("status", "pending")
            in ("pending", "failed", "running")
            for s in STEPS
        )
        if not has_pending:
            print("\n  All steps complete!")
            return True
        if not run_next(state):
            return False


if __name__ == "__main__":
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    state = load_state()
    args = sys.argv[1:]

    if not args or "--status" in args:
        show_status(state)
    elif "--run" in args:
        run_next(state)
    elif "--run-all" in args:
        run_all(state)
    elif "--step" in args:
        idx = args.index("--step")
        n = int(args[idx + 1]) if idx + 1 < len(args) else 0
        if n < 0 or n >= len(STEPS):
            print(f"Invalid step {n}. Valid: 0-{len(STEPS)-1}")
        else:
            from steps import STEP_FUNCS

            step_id = STEPS[n]
            print(f"\n--- Step {n}: {STEP_NAMES[step_id]} ---")
            STEP_FUNCS[step_id](state)
    elif "--reset" in args:
        STATE_FILE.unlink(missing_ok=True)
        print("State reset. Run --run to start fresh.")
    else:
        print(__doc__)
