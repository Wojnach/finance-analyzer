# rc-watchdog.ps1 - RC server health monitor + proactive recycle
# Runs on a schedule (every 30 min). Two jobs:
#   1. Proactive recycle: kill RC servers older than $MaxAgeHours so sessions
#      never hit the 24h server-side timeout. The bat loop auto-restarts.
#   2. Zombie detection: kill RC servers with no ESTABLISHED :443 connection.
# Sends Telegram alert on every action taken.

param(
    [int]$MaxAgeHours = 20
)

$ErrorActionPreference = "Continue"
$logFile  = "Q:\finance-analyzer\data\rc-watchdog.log"
$configFile = "C:\Users\Herc2\.config\finance-analyzer\config.json"

$servers = @(
    @{ Name = "Trading";     BatFile = "rc-server.bat";   Pattern = '--name "?Trading' },
    @{ Name = "Development"; BatFile = "rc-server-2.bat"; Pattern = '--name "?Development' },
    @{ Name = "Research";    BatFile = "rc-server-3.bat"; Pattern = '--name "?Research' }
)

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
        if (-not $token -or -not $chatId) {
            Log "Telegram: missing token or chat_id in config"
            return
        }
        $body = @{ chat_id = $chatId; text = $msg; parse_mode = "Markdown" } | ConvertTo-Json
        Invoke-RestMethod -Uri "https://api.telegram.org/bot$token/sendMessage" `
            -Method Post -ContentType "application/json" -Body $body `
            -TimeoutSec 15 | Out-Null
        Log "Telegram alert sent"
    } catch {
        Log "Telegram send failed: $_"
    }
}

# --- Gather state ---
$rcProcs = Get-CimInstance Win32_Process -Filter "Name='claude.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'remote-control' }

$tcpConns = Get-NetTCPConnection -State Established -RemotePort 443 -ErrorAction SilentlyContinue

$batProcs = Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'rc-server' }

function Test-Connected($procId) {
    $conn = $tcpConns | Where-Object { $_.OwningProcess -eq $procId }
    return ($null -ne $conn -and @($conn).Count -gt 0)
}

$now = Get-Date
$actions = @()
$recycled = 0
$zombied  = 0
$healthy  = 0
$missing  = 0
$basePath = "Q:\finance-analyzer\scripts\win"

foreach ($srv in $servers) {
    $proc = $rcProcs | Where-Object { $_.CommandLine -match $srv.Pattern }
    $bat  = $batProcs | Where-Object { $_.CommandLine -match [regex]::Escape($srv.BatFile) }

    if (-not $proc) {
        if ($bat) {
            Log "$($srv.Name): claude.exe gone, bat loop alive - will auto-restart"
            $healthy++
        } else {
            Log "$($srv.Name): not running. Launching $($srv.BatFile)..."
            Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$basePath\$($srv.BatFile)`"" -WindowStyle Minimized
            $missing++
            $actions += "$($srv.Name): launched - was missing"
        }
        continue
    }

    $pid_val = $proc.ProcessId
    $created = $proc.CreationDate
    $ageHours = ($now - $created).TotalHours
    $ageStr = "{0:N1}h" -f $ageHours

    # Check 1: Proactive age-based recycle
    if ($ageHours -ge $MaxAgeHours) {
        Log "$($srv.Name): PID $pid_val age $ageStr >= ${MaxAgeHours}h threshold. Recycling..."
        Stop-Process -Id $pid_val -Force -ErrorAction SilentlyContinue
        $recycled++
        $msg = "$($srv.Name): recycled at $ageStr, limit ${MaxAgeHours}h"

        if (-not $bat) {
            Log "$($srv.Name): bat loop missing after recycle. Launching..."
            Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$basePath\$($srv.BatFile)`"" -WindowStyle Minimized
            $msg += " + relaunched bat"
        }
        $actions += $msg
        continue
    }

    # Check 2: Zombie detection (alive but no connection)
    if (-not (Test-Connected $pid_val)) {
        Log "$($srv.Name): PID $pid_val zombie - no :443 conn, age $ageStr. Killing..."
        Stop-Process -Id $pid_val -Force -ErrorAction SilentlyContinue
        $zombied++
        $msg = "$($srv.Name): killed zombie at $ageStr, no connection"

        if (-not $bat) {
            Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$basePath\$($srv.BatFile)`"" -WindowStyle Minimized
            $msg += " + relaunched bat"
        }
        $actions += $msg
        continue
    }

    # Healthy
    Log "$($srv.Name): healthy - PID $pid_val, connected, age $ageStr"
    $healthy++
}

# --- Summary ---
if ($actions.Count -gt 0) {
    $time = Get-Date -Format 'HH:mm'
    $summary = "*RC Watchdog* $time`n"
    foreach ($a in $actions) {
        $summary += "- $a`n"
    }
    Log "Actions taken: $($actions -join '; ')"
    Send-Telegram $summary
} else {
    Log "All $healthy server(s) healthy. No action needed."
}
