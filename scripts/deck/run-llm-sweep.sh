#!/usr/bin/env bash
# Single-model competence sweep: run one model across candle intervals and
# tickers, sequentially (herc2 GPU fits one job). 60%+ on a cell = the model
# earns that ticker/horizon slot (user policy 2026-07-13).
#
#   MODEL=phi4_mini scripts/deck/run-llm-sweep.sh
#   MODEL=fin_r1 INTERVALS="4h 1d" scripts/deck/run-llm-sweep.sh
#
# Each phase launches via run-llm-backtest-on-herc.sh (resumable) and waits
# (bounded) for the herc2 COMPLETE marker to increment before the next.
# Results accumulate in one file: data/llm_sweep_<model>.jsonl. Score:
#   scripts/deck/run-llm-backtest-on-herc.sh --results  (with OUT set), or
#   .venv/bin/python scripts/llm_backtest.py --score <fetched file>
set -uo pipefail

HOST=herc2@100.78.196.30
MODEL=${MODEL:-phi4_mini}
INTERVALS=${INTERVALS:-"15m 4h 1d"}
TICKERS=${TICKERS:-BTC-USD,ETH-USD,XAU-USD,XAG-USD}
OUT="data\\llm_sweep_${MODEL}.jsonl"
DIR=$(cd "$(dirname "$0")" && pwd)
MAX_WAIT_MIN=${MAX_WAIT_MIN:-600}

marker_count() {
    timeout 45 ssh -o BatchMode=yes "$HOST" \
        "powershell -NoProfile -Command \"(Select-String -Path Q:\\finance-analyzer\\data\\llm_backtest_run.log -Pattern 'RUN COMPLETE').Count\"" \
        2>/dev/null | tr -dc '0-9'
}

for IV in $INTERVALS; do
    # 15m context is heavy on rows — use a shorter window to keep the
    # phase under ~3.5h; coarser intervals cover the full 5 months.
    case "$IV" in
    15m) START=2026-05-01 ;;
    *) START=2026-02-01 ;;
    esac
    BASE=$(marker_count); BASE=${BASE:-0}
    echo "=== sweep phase: $MODEL interval=$IV tickers=$TICKERS start=$START"
    MODELS=$MODEL TICKERS=$TICKERS INTERVAL=$IV START=$START OUT=$OUT \
        "$DIR/run-llm-backtest-on-herc.sh" || { echo "phase launch failed"; exit 1; }
    WAITED=0
    until [ "$(marker_count || echo "$BASE")" -gt "$BASE" ] 2>/dev/null; do
        sleep 120
        WAITED=$((WAITED + 2))
        if [ $WAITED -ge $MAX_WAIT_MIN ]; then
            echo "phase $IV exceeded ${MAX_WAIT_MIN}min — aborting sweep"
            exit 1
        fi
    done
    echo "=== phase $IV complete (waited ${WAITED}min)"
done

echo "=== sweep done. Fetching + scoring:"
RFILE=$(echo "$OUT" | tr '\\' '/')
scp -q "$HOST:Q:/finance-analyzer/$RFILE" "/tmp/llm_sweep_${MODEL}.jsonl" &&
    /home/deck/projects/finance-analyzer/.venv/bin/python \
        /home/deck/projects/finance-analyzer/scripts/llm_backtest.py \
        --score "/tmp/llm_sweep_${MODEL}.jsonl"
