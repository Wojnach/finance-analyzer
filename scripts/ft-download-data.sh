#!/usr/bin/env bash
# Download historical OHLCV data for backtesting.
# Usage: ./scripts/ft-download-data.sh [days=30]

set -euo pipefail

DAYS="${1:-30}"
SCRIPT_DIR="$(dirname "$0")"

exec "$SCRIPT_DIR/ft.sh" download-data \
    --config /freqtrade/config.json \
    --timeframe 5m \
    --days "$DAYS"
