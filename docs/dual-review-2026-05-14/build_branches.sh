#!/usr/bin/env bash
# Build 8 review branches off baseline-2026-05-14, each adding only its
# subsystem files. Records SHAs to branch-shas.txt.
set -euo pipefail
WORKTREE=Q:/fa-review-0514
SRC=Q:/finance-analyzer
SUBSYS=$SRC/docs/dual-review-2026-05-14/subsystems.txt
OUT=$SRC/docs/dual-review-2026-05-14/branch-shas.txt

cd "$WORKTREE"
: > "$OUT"

# Parse subsystems.txt into 8 file lists
declare -A SUBSYS_FILES
declare -a SUBSYS_NAMES
n=0
current=""
while IFS= read -r line; do
    if [[ "$line" =~ ^##\ Subsystem\ ([0-9]+):\ (.+)$ ]]; then
        n="${BASH_REMATCH[1]}"
        current="${BASH_REMATCH[2]}"
        SUBSYS_NAMES[$n]="$current"
        SUBSYS_FILES[$n]=""
    elif [[ -n "$line" && ! "$line" =~ ^# ]]; then
        SUBSYS_FILES[$n]+="$line"$'\n'
    fi
done < "$SUBSYS"

for i in 1 2 3 4 5 6 7 8; do
    name="${SUBSYS_NAMES[$i]}"
    branch="review/sub-$i-$name"
    echo "=== Subsystem $i: $name → $branch ==="

    # reset to empty baseline
    git checkout -q review/baseline-2026-05-14
    git checkout -B "$branch" 2>&1 | tail -1

    # Clean working tree
    git ls-files | xargs -r rm -f 2>/dev/null || true

    # Copy in the files for this subsystem
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        src="$SRC/$f"
        if [[ ! -f "$src" ]]; then
            echo "WARN: missing $f"
            continue
        fi
        mkdir -p "$(dirname "$f")"
        cp "$src" "$f"
        git add "$f"
    done <<< "${SUBSYS_FILES[$i]}"

    git -c user.email=review@local -c user.name=review commit -q -m "review: subsystem $i — $name"
    sha=$(git rev-parse HEAD)
    echo "$i $name $branch $sha" >> "$OUT"
    echo "  → $sha"
done

git checkout -q review/baseline-2026-05-14
echo "DONE. branch-shas.txt:"
cat "$OUT"
