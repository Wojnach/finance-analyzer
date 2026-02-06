#!/usr/bin/env bash
# Start paper trading (dry run).
# Usage: ./scripts/ft-dry-run.sh [extra args...]

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="docker.io/freqtradeorg/freqtrade:stable"
CONFIG="${PROJECT_DIR}/config.json"

if [[ ! -f "$CONFIG" ]]; then
    CONFIG="${PROJECT_DIR}/config.example.json"
fi

exec podman run --rm -it \
    --name ft-dry-run \
    -p 8080:8080 \
    -v "${PROJECT_DIR}/user_data:/freqtrade/user_data" \
    -v "${CONFIG}:/freqtrade/config.json:ro" \
    "$IMAGE" \
    trade \
    --config /freqtrade/config.json \
    --strategy TABaseStrategy \
    --strategy-path /freqtrade/user_data/strategies \
    "$@"
