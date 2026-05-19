# pf-loop-ensure.ps1 — Idempotent loop launcher
# Safe to call on logon, wake-from-sleep, or manually.
# If pf-loop.bat is already running, exits silently.

$batPath = "Q:\finance-analyzer\scripts\win\pf-loop.bat"
$lockFile = "Q:\finance-analyzer\data\pf-loop.pid"

# Check if a pf-loop.bat process is already running
$existing = Get-Process cmd -ErrorAction SilentlyContinue |
    Where-Object { $_.MainWindowTitle -match 'pf-loop' -or $_.CommandLine -match 'pf-loop' }

# Fallback: check if main.py --loop is running
if (-not $existing) {
    $pyLoop = Get-WmiObject Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'main\.py.*--loop' }
    if ($pyLoop) {
        Write-Host "[pf-loop-ensure] Loop already running (python PID $($pyLoop.ProcessId)). Exiting."
        exit 0
    }
}

if ($existing) {
    Write-Host "[pf-loop-ensure] pf-loop.bat already running (PID $($existing.Id)). Exiting."
    exit 0
}

Write-Host "[pf-loop-ensure] No loop detected. Starting pf-loop.bat..."
Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$batPath`"" -WindowStyle Minimized
Write-Host "[pf-loop-ensure] Launched pf-loop.bat at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
