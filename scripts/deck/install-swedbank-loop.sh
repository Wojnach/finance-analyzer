#!/usr/bin/env bash
# Install the Swedbank monitoring loop as a systemd user unit.
#
# Deliberately does NOT enable or start the unit. Every other pf-* loop on this
# machine is currently disabled; starting one silently would override a pause the
# operator chose. Enable explicitly:
#
#   systemctl --user enable --now pf-swedbank
#
# The loop is monitoring-only and cannot place orders.
set -euo pipefail

REPO="${REPO:-$HOME/projects/finance-analyzer}"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="$UNIT_DIR/pf-swedbank.service"

[ -d "$REPO" ] || { echo "repo not found: $REPO" >&2; exit 1; }
[ -x "$REPO/.venv/bin/python" ] || { echo "venv missing: $REPO/.venv" >&2; exit 1; }
mkdir -p "$UNIT_DIR"

cat > "$UNIT" <<EOF
[Unit]
Description=Swedbank book monitoring loop (read-only, never trades)
After=network-online.target

[Service]
WorkingDirectory=$REPO
Environment=PYTHONPATH=$REPO
ExecStart=$REPO/.venv/bin/python -u data/swedbank_loop.py --loop
Restart=always
RestartSec=30

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
echo "installed $UNIT (not enabled)"
echo
echo "start it with:  systemctl --user enable --now pf-swedbank"
echo "watch it with:  journalctl --user -u pf-swedbank -f"
