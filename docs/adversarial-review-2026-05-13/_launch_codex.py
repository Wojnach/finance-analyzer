"""Launch 8 codex adversarial reviews in parallel.

Each review is run as `codex review --base <branch>` with a tailored prompt.
Output goes to _codex_out/<n>-<subsystem>.txt. Processes are started detached
so the orchestrator can move on to other work.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

WORKTREE = Path(r"Q:/fa-adv-2026-05-13")
OUT_DIR = WORKTREE / "docs/adversarial-review-2026-05-13/_codex_out"
PROMPTS = WORKTREE / "docs/adversarial-review-2026-05-13/_prompts"

SUBSYSTEMS = [
    (1, "signals-core"),
    (2, "orchestration"),
    (3, "portfolio-risk"),
    (4, "metals-core"),
    (5, "avanza-api"),
    (6, "signals-modules"),
    (7, "data-external"),
    (8, "infrastructure"),
]

TEMPLATE = (PROMPTS / "_template.txt").read_text(encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pids: dict[str, int] = {}
    for n, name in SUBSYSTEMS:
        prompt = TEMPLATE.replace("{SUBSYSTEM}", name)
        prompt_path = PROMPTS / f"{n}-{name}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        out_path = OUT_DIR / f"{n}-{name}.txt"
        err_path = OUT_DIR / f"{n}-{name}.err"
        branch = f"adv-2026-05-13/baseline-{name}"
        codex_exe = r"C:\Users\Herc2\AppData\Roaming\npm\codex.cmd"
        # codex review's `--base` is exclusive with positional [PROMPT]. Use default
        # review prompt; we'll supplement with our own Claude adversarial review.
        cmd = [codex_exe, "review", "--base", branch]

        with open(out_path, "wb") as fout, open(err_path, "wb") as ferr:
            proc = subprocess.Popen(
                cmd,
                cwd=str(WORKTREE),
                stdout=fout,
                stderr=ferr,
                stdin=subprocess.DEVNULL,
                shell=False,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
        pids[name] = proc.pid
        print(f"launched {name} branch={branch} pid={proc.pid} -> {out_path.name}")

    state_path = OUT_DIR / "_pids.txt"
    state_path.write_text(
        "\n".join(f"{name}={pid}" for name, pid in pids.items()) + "\n",
        encoding="utf-8",
    )
    print(f"\npids saved -> {state_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
