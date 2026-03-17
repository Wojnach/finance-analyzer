#!/usr/bin/env python3
"""Download upgraded models for the trading agent.

Downloads:
1. Ministral-3-8B-Instruct-2512 (Q5_K_M) — replaces Ministral-8B-2410
2. Qwen3-8B (Q4_K_M) — new LLM signal #31
3. Chronos-2 — auto-downloaded by HuggingFace on first use

Usage:
    python scripts/download_models.py [--ministral] [--qwen3] [--all]
"""

import argparse
import os
import subprocess
import sys


MODELS = {
    "ministral": {
        "repo": "mistralai/Ministral-3-8B-Instruct-2512-GGUF",
        "file": "Ministral-3-8B-Instruct-2512-Q5_K_M.gguf",
        "dest": "Q:/models/ministral-3-8b-gguf",
        "size_gb": 6.1,
    },
    "qwen3": {
        "repo": "Qwen/Qwen3-8B-GGUF",
        "file": "Qwen3-8B-Q4_K_M.gguf",
        "dest": "Q:/models/qwen3-8b-gguf",
        "size_gb": 5.0,
    },
}


def download_model(name, info):
    """Download a model using huggingface-cli."""
    print(f"\n{'='*60}")
    print(f"Downloading {name}: {info['repo']}")
    print(f"File: {info['file']} (~{info['size_gb']:.1f} GB)")
    print(f"Destination: {info['dest']}")
    print(f"{'='*60}")

    os.makedirs(info["dest"], exist_ok=True)

    dest_path = os.path.join(info["dest"], info["file"])
    if os.path.exists(dest_path):
        size_gb = os.path.getsize(dest_path) / (1024**3)
        print(f"Already exists ({size_gb:.2f} GB). Skipping.")
        return True

    cmd = [
        sys.executable, "-m", "huggingface_hub", "download",
        info["repo"],
        "--include", info["file"],
        "--local-dir", info["dest"],
    ]
    print(f"Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print(f"Downloaded {name} successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {name}: {e}")
        return False
    except FileNotFoundError:
        print("huggingface_hub not installed. Install with:")
        print("  pip install huggingface_hub")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download trading agent models")
    parser.add_argument("--ministral", action="store_true", help="Download Ministral-3-8B")
    parser.add_argument("--qwen3", action="store_true", help="Download Qwen3-8B")
    parser.add_argument("--all", action="store_true", help="Download all models")
    args = parser.parse_args()

    if not args.ministral and not args.qwen3 and not args.all:
        args.all = True

    targets = []
    if args.all or args.ministral:
        targets.append(("ministral", MODELS["ministral"]))
    if args.all or args.qwen3:
        targets.append(("qwen3", MODELS["qwen3"]))

    print("Model Download Script")
    print(f"Models to download: {[t[0] for t in targets]}")
    total_gb = sum(t[1]["size_gb"] for t in targets)
    print(f"Total download size: ~{total_gb:.1f} GB")
    print()
    print("Note: Chronos-2 (120M, ~500MB) is auto-downloaded on first use by HuggingFace.")

    results = {}
    for name, info in targets:
        results[name] = download_model(name, info)

    print(f"\n{'='*60}")
    print("Summary:")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")
    print(f"{'='*60}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
