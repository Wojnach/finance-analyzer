# rc-keepalive.ps1 - Soft-recycle idle RC servers to prevent Anthropic session delisting
#
# Problem: Anthropic server-side TTL is ~20 minutes. Only real user/model activity
# resets it - transport keepalives do NOT count. After ~20 min idle, the server deregisters
# the session. The CLI keeps polling (logs stay fresh), but the session becomes invisible
# in the claude.ai/code picker. Known bug: #28571, #29313, #34255, #37605, #38049.
#
# Solution: Every 5 min, check each RC server. If idle longer than its threshold
# (staggered: 13/15/17 min per server), kill it so the bat loop restarts fresh.
# This refreshes the Anthropic-side registration in ~15s.
#
# Staggered thresholds ensure at least one server is always fresh - they never all
# recycle at the same time after the first cycle.
#
# Servers with active work (>1 child session) are NEVER touched.
#
# Flags:
#   -Wake  Wake-from-sleep mode. Recycles ALL idle servers immediately (sleep guarantees
#          the 20-min TTL has expired, so all idle servers are stale).
#
# Designed to run via Task Scheduler every 5 min (PF-RCKeepalive).

param(
    [switch]$Wake
)

$logFile = "Q:\finance-analyzer\data\rc-keepalive.log"
$configFile = "C:\Users\Herc2\.config\finance-analyzer\config.json"

# Staggered thresholds per server - keeps recycling naturally offset by ~2 min.
# All well under the ~20 min Anthropic TTL (4-8 min margin).
$serverThresholds = @{
    "Trading"     = 13
    "Development" = 15
    "Research"    = 17
}
$defaultThresholdMin = 15

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
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

if ($Wake) {
    Log "Wake-from-sleep mode - all idle servers will be recycled immediately."
}

# Get all claude.exe processes
$allClaude = Get-CimInstance Win32_Process -Filter "Name='claude.exe'" -ErrorAction SilentlyContinue
$rcProcs = $allClaude | Where-Object { $_.CommandLine -match 'remote-control' }
$childProcs = $allClaude | Where-Object { $_.CommandLine -match 'session-id' }

if (-not $rcProcs) {
    Log "No RC servers running. Nothing to do."
    exit
}

$recycled = 0
$skippedActive = 0
$skippedYoung = 0
$nameRx = '--name\s+"?(\w+)'

foreach ($rc in $rcProcs) {
    $name = "Unknown"
    if ($rc.CommandLine -match $nameRx) { $name = $matches[1] }
    $ageMin = [math]::Round(((Get-Date) - $rc.CreationDate).TotalMinutes, 1)
    $children = @($childProcs | Where-Object { $_.ParentProcessId -eq $rc.ProcessId })
    $childCount = $children.Count

    # Active sessions are NEVER touched, even on wake
    if ($childCount -gt 1) {
        Log "$name`: ${ageMin}m old, $childCount child sessions [ACTIVE WORK]. Protected."
        $skippedActive++
        continue
    }

    # Determine threshold: 0 on wake (recycle everything idle), per-server otherwise
    $threshold = $defaultThresholdMin
    if ($Wake) {
        $threshold = 0
    } elseif ($serverThresholds.ContainsKey($name)) {
        $threshold = $serverThresholds[$name]
    }

    if ($ageMin -lt $threshold) {
        Log "$name`: ${ageMin}m old, under ${threshold}m threshold. Fresh."
        $skippedYoung++
        continue
    }

    # Old + idle = safe to recycle
    if ($Wake) {
        $reason = "wake-from-sleep"
    } else {
        $reason = "idle ${ageMin}m, threshold ${threshold}m"
    }
    Log "$name`: recycling [$reason]. $childCount child sessions."
    Stop-Process -Id $rc.ProcessId -Force -ErrorAction SilentlyContinue
    $recycled++
}

if ($recycled -gt 0) {
    $total = @($rcProcs).Count
    if ($Wake) { $mode = "wake" } else { $mode = "periodic" }
    $summary = "Refreshed $recycled/$total idle servers [$mode]. $skippedActive active, $skippedYoung fresh."
    Log $summary
    Send-Telegram "*RC Keepalive* $summary"
} else {
    $total = @($rcProcs).Count
    Log "All $total servers OK: $skippedYoung fresh, $skippedActive active. No refresh needed."
}
