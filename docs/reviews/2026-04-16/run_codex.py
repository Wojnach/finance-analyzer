"""Launch 8 codex review subprocesses concurrently, one per subsystem commit."""
import json, subprocess, pathlib, os, sys, time

WORKTREE = r"Q:\finance-analyzer-fgl-review"
OUT_DIR = pathlib.Path(r"Q:\finance-analyzer\docs\reviews\2026-04-16\codex")
OUT_DIR.mkdir(exist_ok=True)

branches = json.loads(pathlib.Path(r"Q:\finance-analyzer\docs\reviews\2026-04-16\branches.json").read_text())

procs = []
for sub in branches["subsystems"]:
    name = sub["name"]
    sha = sub["sha"]
    out_file = OUT_DIR / f"{name}.txt"
    err_file = OUT_DIR / f"{name}.err"
    print(f"Launching codex review for {name} (sha={sha[:8]}) -> {out_file.name}")
    p = subprocess.Popen(
        [r"C:\Users\Herc2\AppData\Roaming\npm\codex.cmd", "review", "--commit", sha],
        cwd=WORKTREE,
        stdout=open(out_file, "w", encoding="utf-8"),
        stderr=open(err_file, "w", encoding="utf-8"),
        env={**os.environ},
        shell=False,
    )
    procs.append((name, p, out_file, err_file))

print(f"\nLaunched {len(procs)} codex reviews. PIDs:")
for name, p, _, _ in procs:
    print(f"  {name}: PID {p.pid}")

# Write PID file so we can monitor later
pathlib.Path(r"Q:\finance-analyzer\docs\reviews\2026-04-16\codex_pids.json").write_text(
    json.dumps({name: {"pid": p.pid, "out": str(of), "err": str(ef)}
                for name, p, of, ef in procs}, indent=2)
)
print("PIDs written. Exiting (children continue).")
