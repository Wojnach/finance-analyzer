import json, subprocess, pathlib, os, sys

WORKTREE = r"Q:\finance-analyzer-fgl-review"
SPEC = r"Q:\finance-analyzer\docs\reviews\2026-04-16\subsystems.json"
OUT = r"Q:\finance-analyzer\docs\reviews\2026-04-16\branches.json"

os.chdir(WORKTREE)
spec = json.loads(pathlib.Path(SPEC).read_text())
baseline = spec["baseline_commit"]
main_ref = spec["main_commit"]

results = {"baseline_commit": baseline, "main_commit": main_ref, "subsystems": []}

def run(args, check=True, capture=True):
    r = subprocess.run(args, capture_output=capture, text=True)
    if check and r.returncode != 0:
        print(f"FAIL {' '.join(args)}: {r.stderr}", file=sys.stderr)
        raise SystemExit(1)
    return r

run(["git", "checkout", "review/empty-baseline"])

for sub in spec["subsystems"]:
    name = sub["name"]
    branch = sub["branch"]
    paths = sub["paths"]
    print(f"=== {name} ===")

    run(["git", "checkout", "review/empty-baseline"])
    subprocess.run(["git", "branch", "-D", branch], capture_output=True)
    run(["git", "checkout", "-b", branch])

    for p in paths:
        r = subprocess.run(["git", "checkout", main_ref, "--", p], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  WARN checkout {p}: {r.stderr.strip()}")

    run(["git", "add", "-A"])
    msg = f"review({name}): subsystem snapshot\n\n{sub['description']}"
    r = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-m", msg],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"  commit output: {r.stdout}{r.stderr}")
        continue
    sha = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    files = run(["git", "show", "--stat", "--name-only", "--format=", sha]).stdout.strip().splitlines()
    print(f"  SHA={sha}  files={len(files)}")
    results["subsystems"].append({
        "name": name, "branch": branch, "sha": sha,
        "file_count": len(files), "description": sub["description"]
    })

pathlib.Path(OUT).write_text(json.dumps(results, indent=2))
print("Wrote", OUT)
