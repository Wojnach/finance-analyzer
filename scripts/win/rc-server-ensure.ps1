# rc-server-ensure.ps1 — Idempotent RC server launcher (3 independent servers)
# Safe to call on logon, wake-from-sleep, or manually. Also runs every 30min via task repetition.
#
# Health check strategy (two-layer):
#   1. Log heartbeat: each server writes "Reconnected" to its output file every 2-3 min.
#      If the log file hasn't been modified in 10+ min, the server is stale — even if the
#      process is alive and has a TCP socket open. This catches the case where a session
#      looks healthy by TCP but is no longer registered with Anthropic's remote-control API.
#   2. TCP fallback: if the log file doesn't exist yet (first launch), fall back to checking
#      for an ESTABLISHED TCP connection to port 443.
#
# Design: sessions should live as long as the PC is on. Recycling kills context and can
# leave spawned scripts/agents running unsupervised. Only recycle truly dead sessions.
#
# When called with -WakeDelay, waits 30s first to give servers time to reconnect after sleep.

param(
    [switch]$WakeDelay
)

$basePath = "Q:\finance-analyzer\scripts\win"
$dataPath = "Q:\finance-analyzer\data"
$logFile  = "$dataPath\rc-server-ensure.log"
$staleMinutes = 10  # if log not updated in this many minutes, server is dead

$servers = @(
    @{ Name = "Trading";     Bat = "$basePath\rc-server.bat";   Pattern = '--name "?Trading';  Log = "$dataPath\rc-server_out.txt" },
    @{ Name = "Development"; Bat = "$basePath\rc-server-2.bat"; Pattern = '--name "?Development'; Log = "$dataPath\rc-server-2_out.txt" },
    @{ Name = "Research";    Bat = "$basePath\rc-server-3.bat"; Pattern = '--name "?Research'; Log = "$dataPath\rc-server-3_out.txt" }
)

$configFile = "C:\Users\Herc2\.config\finance-analyzer\config.json"

function Log($msg) {
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $line = "[$ts] $msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line -ErrorAction SilentlyContinue
}

function Send-Telegram($msg) {
    try {
        $cfg = Get-Content $configFile -Raw | ConvertFrom-Json
        $token  = $cfg.telegram.token
        $chatId = $cfg.telegram.chat_id
        if (-not $token -or -not $chatId) { return }
        $body = @{ chat_id = $chatId; text = $msg; parse_mode = "Markdown" } | ConvertTo-Json
        Invoke-RestMethod -Uri "https://api.telegram.org/bot$token/sendMessage" `
            -Method Post -ContentType "application/json" -Body $body `
            -TimeoutSec 15 | Out-Null
    } catch {
        Log "Telegram send failed: $_"
    }
}

# On wake-from-sleep, wait for servers to attempt their own reconnection
# (built-in: 6 attempts over ~17s). 30s gives ample margin.
if ($WakeDelay) {
    Log "Wake-from-sleep trigger. Waiting 30s for auto-reconnect..."
    Start-Sleep -Seconds 30
}

# Get all claude remote-control processes once
$rcProcs = Get-CimInstance Win32_Process -Filter "Name='claude.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'remote-control' }

# Get all ESTABLISHED TCP connections to port 443 (Anthropic API) — used as fallback
$tcpConns = Get-NetTCPConnection -State Established -RemotePort 443 -ErrorAction SilentlyContinue

function Test-ServerAlive($srv, $procId) {
    # Primary check: is the server's output log fresh?
    # The RC server writes "Reconnected after Xs" every 2-3 minutes as part of its
    # long-poll heartbeat. If the file hasn't been touched, the server is stuck.
    $logPath = $srv.Log
    if (Test-Path $logPath) {
        $lastWrite = (Get-Item $logPath).LastWriteTime
        $staleMins = [math]::Round(((Get-Date) - $lastWrite).TotalMinutes, 1)
        if ($staleMins -lt $staleMinutes) {
            return @{ Alive = $true; Method = "log"; Detail = "log updated ${staleMins}m ago" }
        } else {
            return @{ Alive = $false; Method = "log"; Detail = "log stale (${staleMins}m, threshold ${staleMinutes}m)" }
        }
    }

    # Fallback: no log file yet (first launch). Check TCP connection.
    $conn = $tcpConns | Where-Object { $_.OwningProcess -eq $procId }
    if ($null -ne $conn -and @($conn).Count -gt 0) {
        return @{ Alive = $true; Method = "tcp"; Detail = "ESTABLISHED to :443 (no log file yet)" }
    }
    return @{ Alive = $false; Method = "tcp"; Detail = "no connection and no log file" }
}

# Also find bat-loop cmd.exe processes (parents of the claude.exe servers)
$batProcs = Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'rc-server' }

$launched = 0
$skipped  = 0
$killed   = 0

foreach ($srv in $servers) {
    $running = $rcProcs | Where-Object { $_.CommandLine -match $srv.Pattern }
    $batName = [System.IO.Path]::GetFileName($srv.Bat)
    $batRunning = $batProcs | Where-Object { $_.CommandLine -match [regex]::Escape($batName) }

    if ($running) {
        $procId = $running.ProcessId
        $ageHrs = [math]::Round(((Get-Date) - $running.CreationDate).TotalHours, 1)
        $check = Test-ServerAlive $srv $procId

        if ($check.Alive) {
            Log "$($srv.Name) healthy (PID $procId, ${ageHrs}h uptime, $($check.Detail)). Skipping."
            $skipped++
        } else {
            # Dead: process alive but not actually working.
            Log "$($srv.Name) dead (PID $procId, ${ageHrs}h uptime, $($check.Detail)). Killing; bat loop will restart."
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            $killed++
            Send-Telegram "*RC Ensure* $($srv.Name) was dead after ${ageHrs}h ($($check.Detail)) -- recycled. Check if any spawned work was interrupted."
            if (-not $batRunning) {
                Log "$($srv.Name) bat loop also missing. Launching $($srv.Bat)..."
                Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$($srv.Bat)`"" -WindowStyle Minimized
                $launched++
            }
        }
    } else {
        if ($batRunning) {
            # claude.exe died but bat loop is alive; it will restart on its own
            Log "$($srv.Name) claude.exe gone but bat loop alive (PID $($batRunning.ProcessId)). Will auto-restart."
            $skipped++
        } else {
            Log "$($srv.Name) not running. Launching $($srv.Bat)..."
            Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$($srv.Bat)`"" -WindowStyle Minimized
            $launched++
        }
    }
}

if ($launched -eq 0 -and $killed -eq 0) {
    Log "All 3 RC servers healthy and connected."
} else {
    $summary = "Result: $skipped healthy, $killed dead recycled, $launched fresh launch(es)."
    Log $summary
    $trigger = if ($WakeDelay) { "wake-from-sleep" } else { "logon/periodic" }
    Send-Telegram "*RC Ensure* ($trigger)`n$summary"
}
