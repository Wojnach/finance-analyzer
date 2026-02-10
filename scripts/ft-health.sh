#!/usr/bin/env bash
# Health check for Freqtrade dry-run container.
# Usage: ./scripts/ft-health.sh [--telegram]
#   --telegram  Send alert via Telegram if unhealthy (reads config.json for credentials)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$PROJECT_DIR/config.json"
CONTAINER="ft-dry-run"
ISSUES=()

send_telegram() {
    local msg="$1"
    if [[ ! -f "$CONFIG" ]]; then
        echo "ERROR: No config.json found, cannot send Telegram alert"
        return 1
    fi
    local token chat_id
    token=$(python3 -c "import json; print(json.load(open('$CONFIG'))['telegram']['token'])")
    chat_id=$(python3 -c "import json; print(json.load(open('$CONFIG'))['telegram']['chat_id'])")
    curl -s -X POST "https://api.telegram.org/bot${token}/sendMessage" \
        -d chat_id="$chat_id" \
        -d text="$msg" \
        -d parse_mode="Markdown" > /dev/null
}

# 1. Is container running?
status=$(podman inspect "$CONTAINER" --format '{{.State.Status}}' 2>/dev/null || echo "not_found")
if [[ "$status" == "running" ]]; then
    echo "  Container: running"
else
    echo "  Container: $status"
    ISSUES+=("Container is $status")
fi

# 2. Last log activity
if [[ "$status" == "running" ]]; then
    last_log=$(podman logs --tail 1 --timestamps "$CONTAINER" 2>&1 | head -1)
    if [[ -n "$last_log" ]]; then
        log_time=$(echo "$last_log" | grep -oP '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}' || echo "")
        if [[ -n "$log_time" ]]; then
            log_epoch=$(date -d "$log_time" +%s 2>/dev/null || echo 0)
            now_epoch=$(date +%s)
            age=$(( now_epoch - log_epoch ))
            echo "  Last log: ${age}s ago ($log_time)"
            if (( age > 600 )); then
                ISSUES+=("No log activity for ${age}s")
            fi
        else
            echo "  Last log: $last_log"
        fi
    else
        echo "  Last log: (empty)"
        ISSUES+=("No log output")
    fi
fi

# 3. Check for errors in recent logs
if [[ "$status" == "running" ]]; then
    error_count=$(podman logs --tail 100 "$CONTAINER" 2>&1 | grep -ciE "(error|exception|traceback)" || true)
    if (( error_count > 0 )); then
        echo "  Errors in last 100 lines: $error_count"
        ISSUES+=("$error_count errors in recent logs")
    else
        echo "  Errors in last 100 lines: 0"
    fi
fi

# 4. API health (if running)
if [[ "$status" == "running" ]]; then
    api_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/api/v1/ping 2>/dev/null || echo "000")
    if [[ "$api_response" == "200" ]]; then
        echo "  API (port 8080): healthy"
    else
        echo "  API (port 8080): HTTP $api_response"
        ISSUES+=("API returned HTTP $api_response")
    fi
fi

# Summary
echo ""
if [[ ${#ISSUES[@]} -eq 0 ]]; then
    echo "  Status: HEALTHY"
else
    echo "  Status: UNHEALTHY"
    for issue in "${ISSUES[@]}"; do
        echo "    - $issue"
    done

    if [[ "${1:-}" == "--telegram" ]]; then
        msg="⚠️ *Freqtrade Health Alert*"$'\n'
        for issue in "${ISSUES[@]}"; do
            msg+="• $issue"$'\n'
        done
        send_telegram "$msg"
        echo "  Telegram alert sent."
    fi
fi

exit ${#ISSUES[@]}
