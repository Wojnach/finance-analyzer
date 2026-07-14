#!/usr/bin/env bash
# Orchestrate a historical LLM backtest on herc2 from the Deck.
#
#   run-llm-backtest-on-herc.sh                    # launch with defaults
#   run-llm-backtest-on-herc.sh --status           # task state + log tail
#   run-llm-backtest-on-herc.sh --results          # fetch results + score locally
#   MODELS=fin_r1 START=2026-06-01 END=2026-07-11 STEP_HOURS=4 \
#       run-llm-backtest-on-herc.sh                # custom run
#
# Launch sequence: wake herc2 -> safety-check repo on main -> git pull
# (ff-only) -> disable sleep -> register+run one-shot scheduled task
# PF-LLMBacktest (survives SSH disconnect; gate flag handled by
# scripts/win/llm-backtest-run.ps1). Run is resumable: relaunching skips
# already-completed (model, timestamp) pairs in the output jsonl.
set -uo pipefail

HOST=herc2@100.78.196.30
RREPO='Q:\finance-analyzer'
LREPO=/home/deck/projects/finance-analyzer
MODELS=${MODELS:-ministral3,qwen3,phi4_mini,fin_r1}
START=${START:-2026-02-01}
END=${END:-2026-07-11}
STEP_HOURS=${STEP_HOURS:-8}
OUT=${OUT:-data\\llm_backtest_results.jsonl}
TICKERS=${TICKERS:-BTC-USD,ETH-USD}
INTERVAL=${INTERVAL:-1h}
KEEP_RAW=${KEEP_RAW:-}

hssh() { timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" "$@"; }

case "${1:-}" in
--status)
    hssh "schtasks /query /tn PF-LLMBacktest /fo list | findstr /i \"Status Task\"" 2>/dev/null
    hssh "powershell -NoProfile -Command \"Get-Content $RREPO\\data\\llm_backtest_run.log -Tail 8\"" 2>/dev/null
    exit 0
    ;;
--results)
    RFILE=$(echo "${OUT}" | tr '\\\\' '/')
    scp -q "$HOST:Q:/finance-analyzer/$RFILE" /tmp/llm_backtest_results.jsonl \
        || { echo "no results file on herc2"; exit 1; }
    "$LREPO/.venv/bin/python" "$LREPO/scripts/llm_backtest.py" --score /tmp/llm_backtest_results.jsonl
    exit 0
    ;;
esac

echo "== waking herc2"
timeout 2 bash -c '</dev/tcp/192.168.0.36/3389' 2>/dev/null || ~/wake-herc.sh >/dev/null 2>&1
for _ in $(seq 1 18); do hssh "echo ok" >/dev/null 2>&1 && break; sleep 10; done
hssh "echo ok" >/dev/null || { echo "herc2 unreachable"; exit 1; }

echo "== repo safety check + pull"
BR=$(hssh "cd /d $RREPO && git branch --show-current" | tr -d '\r')
[ "$BR" = "main" ] || { echo "herc2 repo on '$BR' (another agent?) — abort"; exit 1; }
timeout 90 ssh -o BatchMode=yes "$HOST" "cd /d $RREPO && git -c core.sshCommand=\"ssh -o BatchMode=yes -o ConnectTimeout=15\" pull --ff-only" || { echo "pull failed/timed out"; exit 1; }

echo "== disabling sleep"
hssh "powercfg /change standby-timeout-ac 0 & powercfg /change hibernate-timeout-ac 0" >/dev/null

echo "== launching PF-LLMBacktest (models=$MODELS $START..$END interval=$INTERVAL step=${STEP_HOURS}h)"
KR=false; [ -n "$KEEP_RAW" ] && KR=true
HL=false; [ -n "${HEADLINES:-}" ] && HL=true
OUT_J=${OUT//\\//}
ARGS_JSON=$(printf '{"models":"%s","start":"%s","end":"%s","step_hours":%s,"out":"%s","tickers":"%s","interval":"%s","keep_raw":%s,"headlines":%s}' \
    "$MODELS" "$START" "$END" "$STEP_HOURS" "$OUT_J" "$TICKERS" "$INTERVAL" "$KR" "$HL")
echo "$ARGS_JSON" | timeout 60 ssh -o BatchMode=yes "$HOST" "cmd /c more > Q:\\finance-analyzer\\data\\llm_backtest_args.json" \
    || { echo "args upload failed"; exit 1; }
# /tr truncates ~261 chars (bit us 2026-07-13) — keep this SHORT and fixed.
PSCMD="powershell -NoProfile -ExecutionPolicy Bypass -File $RREPO\\scripts\\win\\llm-backtest-run.ps1 -FromArgsFile"
hssh "schtasks /create /f /tn PF-LLMBacktest /sc once /st 23:59 /tr \"$PSCMD\" & schtasks /run /tn PF-LLMBacktest" \
    || { echo "task launch failed"; exit 1; }

echo "launched. check: $0 --status | results: $0 --results"
echo "NOTE: herc2 sleep disabled for the run — re-enable manually or shut down after."
