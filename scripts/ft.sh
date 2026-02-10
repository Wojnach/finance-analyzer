#!/usr/bin/env bash
# Main Freqtrade wrapper — runs any freqtrade command inside Podman.
# Usage: ./scripts/ft.sh [freqtrade args...]
# Example: ./scripts/ft.sh --version
#          ./scripts/ft.sh trade --config /freqtrade/config.json --strategy TABaseStrategy

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="docker.io/freqtradeorg/freqtrade:stable"
CONFIG="${PROJECT_DIR}/config.json"

# Fall back to example config if no config.json exists
if [[ ! -f "$CONFIG" ]]; then
    CONFIG="${PROJECT_DIR}/config.example.json"
fi

exec podman run --rm -it \
    --name ft-"$$" \
    --network=host \
    -v "${PROJECT_DIR}/user_data:/freqtrade/user_data" \
    -v "${CONFIG}:/freqtrade/config.json:ro" \
    "$IMAGE" \
    "$@"
