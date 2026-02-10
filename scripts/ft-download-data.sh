#!/usr/bin/env bash
# Download historical OHLCV data for backtesting.
# Usage: ./scripts/ft-download-data.sh [days=30] [timeframes...]
# Example: ./scripts/ft-download-data.sh 730 5m 1h 4h 1d
#          ./scripts/ft-download-data.sh          (30d of 5m data)

set -euo pipefail

DAYS="${1:-30}"
shift 2>/dev/null || true

# Default to 5m if no timeframes specified
TIMEFRAMES=("${@:-5m}")
if [[ ${#TIMEFRAMES[@]} -eq 0 ]]; then
    TIMEFRAMES=("5m")
fi

SCRIPT_DIR="$(dirname "$0")"

for TF in "${TIMEFRAMES[@]}"; do
    echo "==> Downloading ${DAYS}d of ${TF} data..."
    "$SCRIPT_DIR/ft.sh" download-data \
        --config /freqtrade/config.json \
        --timeframe "$TF" \
        --days "$DAYS"
done
