"""Step implementations for the LoRA training pipeline."""

import json
import os
import subprocess
import sys
from pathlib import Path

from pipeline import (
    BASE_GGUF,
    FEATHER_DIR,
    FINAL_GGUF,
    HF_MODEL_DIR,
    LLM_PYTHON,
    LORA_OUTPUT,
    MAIN_PYTHON,
    ORIGINAL_LORA,
    OUTPUT_DIR,
    PIPELINE_DIR,
    REPO_DIR,
    TRAINING_DATA,
    VENV_PYTHON,
    VENV_TRAIN,
    log_path,
    now_iso,
    update_step,
)


def _run(cmd, log_file=None, cwd=None, timeout=None):
    cmd = [str(c) for c in cmd]
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"\n{'='*60}\n[{now_iso()}] {' '.join(cmd)}\n{'='*60}\n")
            r = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=cwd,
                timeout=timeout,
            )
        return r.returncode
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def step_setup_env(state):
    log = log_path("setup_env")
    update_step(state, "setup_env", "running", log=str(log))

    if not VENV_TRAIN.exists():
        print("  Creating .venv-train...")
        rc = _run([sys.executable, "-m", "venv", VENV_TRAIN], log)
        if rc != 0:
            update_step(state, "setup_env", "failed", error="venv creation failed")
            return False

    pip = VENV_TRAIN / ("Scripts" if os.name == "nt" else "bin") / "pip"

    print("  Installing PyTorch with CUDA (this takes a few minutes)...")
    rc = _run(
        [
            pip,
            "install",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ],
        log,
        timeout=600,
    )
    if rc != 0:
        update_step(state, "setup_env", "failed", error="torch install failed")
        return False

    print("  Installing training libraries...")
    rc = _run(
        [
            pip,
            "install",
            "transformers",
            "peft",
            "bitsandbytes",
            "datasets",
            "accelerate",
            "scipy",
            "sentencepiece",
            "protobuf",
        ],
        log,
        timeout=300,
    )
    if rc != 0:
        update_step(state, "setup_env", "failed", error="training libs failed")
        return False

    print("  Verifying CUDA...")
    rc, out, _ = _run(
        [
            VENV_PYTHON,
            "-c",
            "import torch; print(f'CUDA:{torch.cuda.is_available()},"
            "GPU:{torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')",
        ]
    )
    if rc != 0 or "CUDA:True" not in (out or ""):
        update_step(state, "setup_env", "failed", error=f"CUDA failed: {out}")
        return False

    update_step(state, "setup_env", "done", cuda=out.strip())
    print(f"  {out.strip()}")
    return True


def step_generate_data(state):
    log = log_path("generate_data")
    update_step(state, "generate_data", "running", log=str(log))
    print("  Generating training data...")

    script = PIPELINE_DIR / "generate_data.py"
    rc = _run(
        [MAIN_PYTHON, script, "--feather-dir", FEATHER_DIR, "--output", TRAINING_DATA],
        log,
        timeout=300,
    )

    if rc != 0 or not TRAINING_DATA.exists():
        update_step(state, "generate_data", "failed", error="data generation failed")
        return False

    n = sum(1 for _ in open(TRAINING_DATA))
    update_step(state, "generate_data", "done", examples=n)
    print(f"  Generated {n} training examples")
    return True


def step_download_model(state):
    log = log_path("download_model")
    update_step(state, "download_model", "running", log=str(log))

    if HF_MODEL_DIR.exists() and any(HF_MODEL_DIR.glob("*.safetensors")):
        update_step(state, "download_model", "done", note="already exists")
        print("  Model already downloaded, skipping")
        return True

    print(f"  Downloading Ministral-8B to {HF_MODEL_DIR} (~16GB)...")
    HF_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    dl_script = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download('mistralai/Ministral-8B-Instruct-2410', "
        f"local_dir=r'{HF_MODEL_DIR}')"
    )
    rc = _run([VENV_PYTHON, "-c", dl_script], log, timeout=3600)
    if rc != 0:
        update_step(
            state,
            "download_model",
            "failed",
            error="Download failed. May need: huggingface-cli login",
        )
        return False

    update_step(state, "download_model", "done")
    print("  Download complete")
    return True


def step_train(state):
    log = log_path("train")
    update_step(state, "train", "running", log=str(log))
    print("  Starting QLoRA training...")

    script = PIPELINE_DIR / "train_lora.py"
    progress_file = PIPELINE_DIR / "training_progress.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        VENV_PYTHON,
        script,
        "--model-dir",
        HF_MODEL_DIR,
        "--data",
        TRAINING_DATA,
        "--output-dir",
        OUTPUT_DIR,
        "--progress-file",
        progress_file,
    ]

    checkpoints = sorted(OUTPUT_DIR.glob("checkpoint-*"))
    if checkpoints:
        cmd.extend(["--resume-from", checkpoints[-1]])
        print(f"  Resuming from {checkpoints[-1].name}")

    rc = _run(cmd, log, timeout=7200)
    if rc != 0:
        update_step(state, "train", "failed", error="training failed, check log")
        return False
    if not LORA_OUTPUT.exists():
        update_step(state, "train", "failed", error="no final output")
        return False

    update_step(state, "train", "done")
    print("  Training complete")
    return True


def step_convert_gguf(state):
    log = log_path("convert_gguf")
    update_step(state, "convert_gguf", "running", log=str(log))
    print("  Converting LoRA to GGUF...")

    llama_cpp_dir = REPO_DIR / "llama.cpp"
    convert_script = llama_cpp_dir / "convert_lora_to_gguf.py"

    if not convert_script.exists():
        print("  Cloning llama.cpp for conversion tool...")
        rc = _run(
            [
                "git",
                "clone",
                "--depth=1",
                "https://github.com/ggerganov/llama.cpp.git",
                llama_cpp_dir,
            ],
            log,
            timeout=300,
        )
        if not convert_script.exists():
            update_step(state, "convert_gguf", "failed", error="clone failed")
            return False

    FINAL_GGUF.parent.mkdir(parents=True, exist_ok=True)
    rc = _run(
        [
            VENV_PYTHON,
            convert_script,
            LORA_OUTPUT,
            "--outfile",
            FINAL_GGUF,
            "--base",
            HF_MODEL_DIR,
        ],
        log,
        timeout=600,
    )
    if rc != 0 or not FINAL_GGUF.exists():
        update_step(state, "convert_gguf", "failed", error="conversion failed")
        return False

    size_mb = round(FINAL_GGUF.stat().st_size / (1024 * 1024), 1)
    update_step(state, "convert_gguf", "done", size_mb=size_mb, path=str(FINAL_GGUF))
    print(f"  GGUF: {FINAL_GGUF} ({size_mb} MB)")
    return True


def step_verify(state):
    log = log_path("verify")
    update_step(state, "verify", "running", log=str(log))
    print("  Verifying GGUF...")

    test_code = (
        "import json; from llama_cpp import Llama; "
        f"m=Llama(model_path=r'{BASE_GGUF}',lora_path=r'{FINAL_GGUF}',"
        "n_ctx=2048,n_gpu_layers=-1,verbose=False); "
        "r=m('[INST]Asset:BTC,$67000,RSI:28. BUY/SELL/HOLD?[/INST]',"
        "max_tokens=50,temperature=0.1); "
        "t=r['choices'][0]['text'].strip(); "
        "print(json.dumps({'out':t,'ok':any(w in t.upper() for w in ['BUY','SELL','HOLD'])}))"
    )
    rc, out, err = _run([LLM_PYTHON, "-c", test_code], timeout=120)
    if rc != 0:
        update_step(
            state, "verify", "failed", error=f"load failed: {(err or '')[:200]}"
        )
        return False

    try:
        result = json.loads(out.strip())
        if not result.get("ok"):
            update_step(
                state, "verify", "failed", error=f"no decision: {result['out']}"
            )
            return False
    except Exception as e:
        update_step(state, "verify", "failed", error=str(e))
        return False

    update_step(state, "verify", "done", test_output=result["out"][:200])
    print(f"  OK: {result['out'][:100]}")
    return True


def step_deploy_shadow(state):
    update_step(state, "deploy_shadow", "running")
    print("  Writing shadow config...")

    config = {
        "enabled": True,
        "original_lora": str(ORIGINAL_LORA),
        "custom_lora": str(FINAL_GGUF),
        "log_file": str(REPO_DIR / "data" / "ab_test_log.jsonl"),
        "deployed_at": now_iso(),
    }
    config_path = REPO_DIR / "data" / "shadow_lora_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2))

    update_step(state, "deploy_shadow", "done", config=str(config_path))
    print(f"  Config: {config_path}")
    print("  Modify ministral_signal.py to enable A/B testing")
    return True


STEP_FUNCS = {
    "setup_env": step_setup_env,
    "generate_data": step_generate_data,
    "download_model": step_download_model,
    "train": step_train,
    "convert_gguf": step_convert_gguf,
    "verify": step_verify,
    "deploy_shadow": step_deploy_shadow,
}
