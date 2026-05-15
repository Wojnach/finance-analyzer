#!/usr/bin/env bash
set -uo pipefail
WORKTREE=Q:/fa-review-0514
SRC=Q:/finance-analyzer
SHAS=$SRC/docs/dual-review-2026-05-14/branch-shas.txt
OUT=$SRC/docs/dual-review-2026-05-14/codex-raw

cd "$WORKTREE"
mkdir -p "$OUT"

echo "[$(date +%H:%M:%S)] starting runner" > "$OUT/_runner.log"

# Proper quota probe: looks at WHOLE output for "usage limit"
probe_quota() {
    local out
    out=$(timeout 20 codex exec review --commit a9e03d0c --output-last-message /tmp/codex-probe.txt 2>&1)
    if echo "$out" | grep -q "usage limit"; then
        return 1
    fi
    return 0
}

# Wait for quota (cap 30 minutes from start)
for attempt in $(seq 1 30); do
    if probe_quota; then
        echo "[$(date +%H:%M:%S)] attempt $attempt: quota AVAILABLE" >> "$OUT/_runner.log"
        break
    fi
    echo "[$(date +%H:%M:%S)] attempt $attempt: still rate-limited" >> "$OUT/_runner.log"
    sleep 60
done

# If still limited after 30 attempts, exit with marker
if ! probe_quota; then
    echo "[$(date +%H:%M:%S)] QUOTA-NEVER-RECOVERED" >> "$OUT/_runner.log"
    exit 1
fi

# Sequential review of each commit
while read -r n name branch sha; do
    [[ -z "$sha" ]] && continue
    log="$OUT/${n}-${name}.log"
    msg="$OUT/${n}-${name}.lastmsg.txt"
    echo "[$(date +%H:%M:%S)] === Reviewing $n $name @ $sha ===" >> "$OUT/_runner.log"
    # Retry up to 3 times if quota hits mid-stream
    for r in 1 2 3; do
        codex exec review --commit "$sha" --output-last-message "$msg" > "$log" 2>&1
        if grep -q "usage limit" "$log"; then
            echo "[$(date +%H:%M:%S)] $n $name retry $r quota hit, sleeping 90s" >> "$OUT/_runner.log"
            sleep 90
            continue
        fi
        break
    done
    echo "[$(date +%H:%M:%S)] done $n $name" >> "$OUT/_runner.log"
done < "$SHAS"

echo "[$(date +%H:%M:%S)] ALL DONE" >> "$OUT/_runner.log"
