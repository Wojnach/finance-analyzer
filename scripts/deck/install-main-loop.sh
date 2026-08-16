#!/usr/bin/env bash
# Install the Layer 1 main data loop as a systemd user unit.
#
# Why this exists: the loop was the only pf-* process on this machine without a
# unit. On 2026-08-11 a transient DNS outage killed it mid-cycle and nothing
# restarted it — the loop stayed dead for 5.7 days before anyone noticed, while
# pf-dashboard and pf-swedbank rode out the same outage untouched.
#
# Like install-swedbank-loop.sh this does NOT enable or start the unit, because
# the loop is usually already running by hand in tmux. Switch over explicitly:
#
#   tmux kill-session -t pf-loop            # release the singleton flock first
#   systemctl --user enable --now pf-loop
#
set -euo pipefail

REPO="${REPO:-$HOME/projects/finance-analyzer}"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="$UNIT_DIR/pf-loop.service"

[ -d "$REPO" ] || { echo "repo not found: $REPO" >&2; exit 1; }
[ -x "$REPO/.venv/bin/python" ] || { echo "venv missing: $REPO/.venv" >&2; exit 1; }
[ -e "$REPO/config.json" ] || { echo "config.json missing (symlink to the external config?)" >&2; exit 1; }
mkdir -p "$UNIT_DIR"

# -m portfolio.main, never portfolio/main.py: script mode leaves the repo root
# off sys.path and the package imports fail.
#
# RestartPreventExitStatus=11 is _DUPLICATE_EXIT_CODE (main.py:53). A second
# instance cannot take the fcntl flock and exits 11; without this the unit would
# crash-loop against a loop the operator is deliberately running in tmux.
cat > "$UNIT" <<EOF
[Unit]
Description=Finance Analyzer Layer 1 data loop (600s cycle)
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=$REPO
Environment=PYTHONPATH=$REPO
Environment=PYTHONUNBUFFERED=1
ExecStart=$REPO/.venv/bin/python -u -m portfolio.main --loop
Restart=always
RestartSec=30
RestartPreventExitStatus=11
TimeoutStopSec=60

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
echo "installed $UNIT (not enabled)"
echo
echo "the loop may already be running in tmux — that instance holds the singleton"
echo "lock and the unit will exit 11 against it. Switch over with:"
echo
echo "  tmux kill-session -t pf-loop"
echo "  systemctl --user enable --now pf-loop"
echo
echo "watch it with:  journalctl --user -u pf-loop -f"
