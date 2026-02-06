#!/usr/bin/env bash
# Run backtests with TABaseStrategy.
# Usage: ./scripts/ft-backtest.sh [extra args...]
# Example: ./scripts/ft-backtest.sh --timerange 20260101-

set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"

exec "$SCRIPT_DIR/ft.sh" backtesting \
    --config /freqtrade/config.json \
    --strategy TABaseStrategy \
    --strategy-path /freqtrade/user_data/strategies \
    "$@"
