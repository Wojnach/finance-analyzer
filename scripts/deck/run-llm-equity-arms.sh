#!/usr/bin/env bash
# Equity two-arm LLM experiment (docs/plans/2026-07-14-equity-news-llm-test-plan.md):
#   Arm A = indicators-only, Arm B = indicators + Finnhub headlines.
# Prefetches headline caches on the Deck (finnhub_key lives in Deck config
# only), pushes them to herc2, runs both arms sequentially, scores A vs B.
#
#   scripts/deck/run-llm-equity-arms.sh                      # defaults
#   MODEL=phi4_mini TICKERS=MSTR,NVDA STEP_HOURS=8 ...       # custom
#   SKIP_A=1 ... # rerun only Arm B
set -uo pipefail

HOST=herc2@100.78.196.30
LREPO=/home/deck/projects/finance-analyzer
MODEL=${MODEL:-phi4_mini}
TICKERS=${TICKERS:-MSTR,NVDA,TSLA,PLTR}
START=${START:-2026-02-01}
END=${END:-2026-07-11}
STEP_HOURS=${STEP_HOURS:-4}
DIR=$(cd "$(dirname "$0")" && pwd)
MAX_WAIT_MIN=${MAX_WAIT_MIN:-240}

marker_count() {
    timeout 45 ssh -o BatchMode=yes "$HOST" \
        "powershell -NoProfile -Command \"(Select-String -Path Q:\\finance-analyzer\\data\\llm_backtest_run.log -Pattern 'RUN COMPLETE').Count\"" \
        2>/dev/null | tr -dc '0-9'
}

wait_phase() {
    local base=$1 waited=0
    until [ "$(marker_count || echo "$base")" -gt "$base" ] 2>/dev/null; do
        sleep 120
        waited=$((waited + 2))
        [ $waited -ge $MAX_WAIT_MIN ] && { echo "phase exceeded ${MAX_WAIT_MIN}min — abort"; exit 1; }
    done
}

echo "== prefetching headline caches on Deck (finnhub key is Deck-only)"
"$LREPO/.venv/bin/python" - "$TICKERS" "$START" "$END" <<'EOF'
import importlib.util, sys
sys.path.insert(0, "/home/deck/projects/finance-analyzer")
spec = importlib.util.spec_from_file_location(
    "bt", "/home/deck/projects/finance-analyzer/scripts/llm_backtest.py")
bt = importlib.util.module_from_spec(spec); spec.loader.exec_module(bt)
import os
os.chdir("/home/deck/projects/finance-analyzer")
for t in sys.argv[1].split(","):
    sym = bt.YF_TICKERS[t]
    h = bt.fetch_headlines_finnhub(sym, sys.argv[2], sys.argv[3])
    print(f"{t}: {len(h)} headlines")
EOF
[ $? -eq 0 ] || { echo "prefetch failed"; exit 1; }

echo "== waking herc2 + pushing caches"
timeout 2 bash -c '</dev/tcp/192.168.0.36/3389' 2>/dev/null || ~/wake-herc.sh >/dev/null 2>&1
for _ in $(seq 1 18); do timeout 15 ssh -o BatchMode=yes "$HOST" "echo ok" >/dev/null 2>&1 && break; sleep 10; done
scp -q "$LREPO"/data/backtest_headlines_*.json "$HOST:Q:/finance-analyzer/data/" || { echo "cache push failed"; exit 1; }

if [ -z "${SKIP_A:-}" ]; then
    BASE=$(marker_count); BASE=${BASE:-0}
    echo "== Arm A (indicators only)"
    MODELS=$MODEL TICKERS=$TICKERS INTERVAL=1h START=$START END=$END STEP_HOURS=$STEP_HOURS \
        OUT='data\llm_equity_armA.jsonl' "$DIR/run-llm-backtest-on-herc.sh" || exit 1
    wait_phase "$BASE"
    echo "== Arm A complete"
fi

BASE=$(marker_count); BASE=${BASE:-0}
echo "== Arm B (indicators + headlines)"
MODELS=$MODEL TICKERS=$TICKERS INTERVAL=1h START=$START END=$END STEP_HOURS=$STEP_HOURS \
    HEADLINES=1 OUT='data\llm_equity_armB.jsonl' "$DIR/run-llm-backtest-on-herc.sh" || exit 1
wait_phase "$BASE"
echo "== Arm B complete"

echo "== scores"
for arm in A B; do
    scp -q "$HOST:Q:/finance-analyzer/data/llm_equity_arm$arm.jsonl" "/tmp/llm_equity_arm$arm.jsonl" &&
        { echo "--- Arm $arm ---"; "$LREPO/.venv/bin/python" "$LREPO/scripts/llm_backtest.py" --score "/tmp/llm_equity_arm$arm.jsonl"; }
done
echo "done. herc2 left RUNNING — shut down manually if idle."
