#!/usr/bin/env bash
# Push a compressed snapshot of data/ to herc2 (Q:\deck-backups) when it is
# awake. Skips silently when herc2 is unreachable — it sleeps most of the
# time, so the twice-daily timer just catches whichever attempts land.
# Keeps the 14 newest snapshots on herc2.
set -uo pipefail

REPO=/home/deck/projects/finance-analyzer
HOST=herc2@100.78.196.30
STAMP=$(date +%Y%m%d-%H%M)
TMP=$(mktemp /tmp/deck-data-backup-XXXX.tgz)
trap 'rm -f "$TMP"' EXIT

ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" "exit 0" 2>/dev/null || {
    echo "herc2 unreachable — backup skipped"
    exit 0
}

tar -C "$REPO" -czf "$TMP" \
    --exclude='data/*_out.txt' \
    --exclude='data/rc-server*' \
    --exclude='data/*.db-shm' \
    --exclude='data/*.db-wal' \
    data

ssh -o BatchMode=yes "$HOST" "cmd /c if not exist Q:\\deck-backups mkdir Q:\\deck-backups" \
    && scp -q "$TMP" "$HOST:Q:/deck-backups/data-$STAMP.tgz" \
    && ssh -o BatchMode=yes "$HOST" "powershell -NoProfile -Command \"Get-ChildItem Q:\\deck-backups\\data-*.tgz | Sort-Object CreationTime -Descending | Select-Object -Skip 14 | Remove-Item\"" \
    && echo "backup ok: data-$STAMP.tgz ($(du -h "$TMP" | cut -f1))"
