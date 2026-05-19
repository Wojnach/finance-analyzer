#!/bin/bash
# Build 8 subsystem commits chained on empty-baseline.
# Each commit isolates one subsystem; codex review --commit <SHA> sees that subsystem as "added".
set -e
cd /q/finance-analyzer-fgl-review

python <<'PYEOF'
import json, subprocess, pathlib, os
os.chdir("/q/finance-analyzer-fgl-review")

spec = json.loads(pathlib.Path("/q/finance-analyzer/docs/reviews/2026-04-16/subsystems.json").read_text())
baseline = spec["baseline_commit"]
main_ref = spec["main_commit"]

results = {"baseline_commit": baseline, "main_commit": main_ref, "subsystems": []}

# Start from empty-baseline
subprocess.run(["git", "checkout", "review/empty-baseline"], check=True, capture_output=True)

for sub in spec["subsystems"]:
    name = sub["name"]
    branch = sub["branch"]
    paths = sub["paths"]
    print(f"=== {name} ===")

    # Checkout empty-baseline, create subsystem branch fresh
    subprocess.run(["git", "checkout", "review/empty-baseline"], check=True, capture_output=True)
    # Delete branch if exists
    subprocess.run(["git", "branch", "-D", branch], capture_output=True)
    subprocess.run(["git", "checkout", "-b", branch], check=True, capture_output=True)

    # Copy subsystem files from main
    for p in paths:
        r = subprocess.run(["git", "checkout", main_ref, "--", p], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  WARN checkout {p}: {r.stderr.strip()}")

    # Stage & commit
    subprocess.run(["git", "add", "-A"], check=True)
    msg = f"review({name}): subsystem snapshot for adversarial review\n\n{sub['description']}"
    r = subprocess.run(["git", "-c", "commit.gpgsign=false", "commit", "-m", msg], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAIL commit: {r.stderr}")
        continue

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    # count files
    file_count = subprocess.run(
        ["git", "show", "--stat", "--name-only", "--format=", sha],
        capture_output=True, text=True, check=True
    ).stdout.strip().splitlines()
    print(f"  SHA: {sha}  files: {len(file_count)}")
    results["subsystems"].append({
        "name": name, "branch": branch, "sha": sha,
        "file_count": len(file_count), "description": sub["description"]
    })

pathlib.Path("/q/finance-analyzer/docs/reviews/2026-04-16/branches.json").write_text(
    json.dumps(results, indent=2)
)
print("Done.")
PYEOF
