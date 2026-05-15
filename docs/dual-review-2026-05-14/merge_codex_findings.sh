#!/usr/bin/env bash
# Read all 8 codex lastmsg.txt files and produce a consolidated findings doc.
set -uo pipefail
SRC=Q:/finance-analyzer/docs/dual-review-2026-05-14
RAW=$SRC/codex-raw
OUT=$SRC/codex-real-findings.md

{
  echo "# Real Codex CLI Findings — All 8 Subsystems"
  echo
  echo "Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) from \`codex-raw/*.lastmsg.txt\`."
  echo
  echo "Each subsystem section is the verbatim 'last message' from"
  echo "\`codex exec review --commit <SHA>\`. Findings here SUPPLEMENT the"
  echo "Claude reviews and Codex-substitute reviews. See \`SYNTHESIS.md\`"
  echo "for the prioritized punch-list."
  echo
  for n in 1 2 3 4 5 6 7 8; do
    name=$(grep "^$n " $SRC/branch-shas.txt | awk '{print $2}')
    msg="$RAW/$n-$name.lastmsg.txt"
    if [[ ! -f "$msg" ]]; then
      echo "## $n. $name"
      echo
      echo "_(no codex output — quota or runtime failure)_"
      echo
      continue
    fi
    echo "## $n. $name"
    echo
    cat "$msg"
    echo
    echo
  done
} > "$OUT"

echo "Wrote $OUT ($(wc -l < $OUT) lines)"
