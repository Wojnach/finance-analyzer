# WSL + Claude Code Setup Script
# Run this in PowerShell as Administrator (right-click PowerShell -> Run as Admin)
# It will pause between steps so you can see progress.

Write-Host "=== Step 1: Update WSL kernel ===" -ForegroundColor Cyan
wsl --update
wsl --shutdown
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "=== Step 2: Fix WSL DNS + Install Node + Claude Code ===" -ForegroundColor Cyan
# Run everything inside WSL in one shot
wsl -e bash -c @'
set -e

# Fix DNS permanently
echo -e "[network]\ngenerateResolvConf = false" | sudo tee /etc/wsl.conf > /dev/null
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf > /dev/null
echo "[OK] DNS fixed"

# Update apt
sudo apt-get update -q

# Remove old nodejs if broken
sudo apt-get remove -y nodejs npm 2>/dev/null || true

# Install Node 22 via NodeSource
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
echo "[OK] Node $(node --version) installed"

# Install Claude Code globally
sudo npm install -g @anthropic-ai/claude-code
echo "[OK] Claude Code installed: $(which claude)"

# Verify
echo ""
echo "=== VERIFICATION ==="
echo "Node: $(node --version)"
echo "npm: $(npm --version)"
echo "Claude: $(which claude)"
echo ""
echo "SUCCESS! Open Windows Terminal -> Ubuntu tab, then run:"
echo "  cd /mnt/q/finance-analyzer"
echo "  claude"
'@

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host "Open Windows Terminal -> click dropdown -> Ubuntu"
Write-Host "Then run: cd /mnt/q/finance-analyzer && claude"
Write-Host ""
Read-Host "Press Enter to close"
