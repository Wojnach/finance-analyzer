#!/usr/bin/env bash
# Run hyperparameter optimization for TABaseStrategy.
# Usage: ./scripts/ft-hyperopt.sh [epochs=100] [extra args...]
# Example: ./scripts/ft-hyperopt.sh 500 --timerange 20240101-

set -euo pipefail

EPOCHS="${1:-100}"
shift 2>/dev/null || true
SCRIPT_DIR="$(dirname "$0")"

exec "$SCRIPT_DIR/ft.sh" hyperopt \
    --config /freqtrade/config.json \
    --strategy TABaseStrategy \
    --strategy-path /freqtrade/user_data/strategies \
    --hyperopt-loss SharpeHyperOptLossDaily \
    --spaces buy sell \
    --epochs "$EPOCHS" \
    "$@"
