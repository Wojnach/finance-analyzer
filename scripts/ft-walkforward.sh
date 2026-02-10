#!/usr/bin/env bash
# Walk-forward validation — hyperopt on train windows, backtest on test windows.
# Usage: ./scripts/ft-walkforward.sh [args...]
# Example: ./scripts/ft-walkforward.sh --epochs 200 --train-days 180 --test-days 60

set -euo pipefail
SCRIPT_DIR="$(dirname "$0")"
exec python3 -u "$SCRIPT_DIR/ft-walkforward.py" "$@"
