#!/usr/bin/env bash
# with-herc.sh — run a command against herc2, restoring its power state after.
#
# Pattern requested 2026-07-18: any automation that WAKES herc2 must shut it
# down again once the work is done; if herc2 was already awake, leave it
# alone. Wrap any herc2-touching job in this script instead of hand-rolling
# wake/shutdown logic:
#
#   scripts/deck/with-herc.sh ssh herc2 "cd /d Q:\\finance-analyzer && ..."
#
# Behavior:
#   - Probes 3389 (RDP) to decide awake/asleep — ICMP is firewalled, and a
#     mid-shutdown host still answers SSH, so 3389 is the reliable signal.
#   - Asleep -> ~/wake-herc.sh (WOL), wait up to WAKE_WAIT_S for ssh.
#   - Runs the wrapped command; its exit code is preserved.
#   - Only if WE woke it: shutdown /s /f /t 0 afterwards — UNLESS the busy
#     guard trips (PF-LLMBacktest scheduled task Running = the GPU matrix
#     campaign; never kill it). Skipped shutdowns are reported, not silent.
#   - Already awake -> never shuts down (someone/something else owns it).
set -uo pipefail

HOST=herc2
TS_IP=100.78.196.30
WAKE_WAIT_S=${WAKE_WAIT_S:-180}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=5 $HOST"

rdp_open() { timeout 3 bash -c "</dev/tcp/$TS_IP/3389" 2>/dev/null; }

WAS_AWAKE=0
if rdp_open; then
    WAS_AWAKE=1
else
    echo "with-herc: herc2 asleep — waking"
    "$HOME/wake-herc.sh" >/dev/null 2>&1 || true
    waited=0
    until $SSH "exit 0" 2>/dev/null; do
        sleep 10
        waited=$((waited + 10))
        if [ "$waited" -ge "$WAKE_WAIT_S" ]; then
            echo "with-herc: wake FAILED after ${WAKE_WAIT_S}s" >&2
            exit 111
        fi
    done
    echo "with-herc: awake after ${waited}s"
fi

"$@"
rc=$?

if [ "$WAS_AWAKE" -eq 0 ]; then
    if $SSH 'schtasks /query /tn PF-LLMBacktest /fo csv /nh 2>nul' 2>/dev/null | grep -qi running; then
        echo "with-herc: PF-LLMBacktest is Running — leaving herc2 ON (busy guard)"
    else
        echo "with-herc: we woke it, work done — shutting herc2 down"
        $SSH "shutdown /s /f /t 0" 2>/dev/null
        sleep 20
        if rdp_open; then
            echo "with-herc: WARNING — herc2 still answering 3389 after shutdown" >&2
        else
            echo "with-herc: herc2 confirmed down"
        fi
    fi
fi

exit $rc
