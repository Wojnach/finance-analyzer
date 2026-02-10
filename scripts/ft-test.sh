#!/usr/bin/env bash
# Run pytest inside Freqtrade container (for integration tests).
# Usage: ./scripts/ft-test.sh [pytest args...]
# Example: ./scripts/ft-test.sh tests/integration/ -v

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="docker.io/freqtradeorg/freqtrade:stable"

# Default test path if none specified
ARGS=("${@:-tests/integration/ -v}")

exec podman run --rm -it \
    --name ft-test-"$$" \
    --entrypoint "" \
    -v "${PROJECT_DIR}:/freqtrade/project:ro" \
    -v "${PROJECT_DIR}/user_data:/freqtrade/user_data:U" \
    "$IMAGE" \
    bash -c "pip install -q pytest && python -m pytest --override-ini='addopts=' --rootdir=/freqtrade/project /freqtrade/project/${ARGS[*]}"
