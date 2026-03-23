# rc-server-ensure.ps1 — Idempotent RC server launcher (3 independent servers)
# Safe to call on logon, wake-from-sleep, or manually.
# Checks each server individually and only launches missing ones.

$basePath = "Q:\finance-analyzer\scripts\win"
$servers = @(
    @{ Name = "Trading";     Bat = "$basePath\rc-server.bat";   Pattern = '--name "?Trading' },
    @{ Name = "Development"; Bat = "$basePath\rc-server-2.bat"; Pattern = '--name "?Development' },
    @{ Name = "Research";    Bat = "$basePath\rc-server-3.bat"; Pattern = '--name "?Research' }
)

$ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

# Get all claude remote-control processes once
$rcProcs = Get-CimInstance Win32_Process -Filter "Name='claude.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'remote-control' }

$launched = 0
foreach ($srv in $servers) {
    $running = $rcProcs | Where-Object { $_.CommandLine -match $srv.Pattern }
    if ($running) {
        Write-Host "[$ts] $($srv.Name) already running (PID $($running.ProcessId)). Skipping."
    } else {
        Write-Host "[$ts] $($srv.Name) not running. Launching $($srv.Bat)..."
        Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$($srv.Bat)`"" -WindowStyle Minimized
        $launched++
    }
}

if ($launched -eq 0) {
    Write-Host "[$ts] All 3 RC servers already running."
} else {
    Write-Host "[$ts] Launched $launched server(s)."
}
