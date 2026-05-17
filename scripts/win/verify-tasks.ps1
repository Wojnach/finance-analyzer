# verify-tasks.ps1 — assert every PF-* scheduled task launches cleanly
# under the hidden-window wrapper (run-hidden.vbs).
#
# Run as:
#   powershell -ExecutionPolicy Bypass -File scripts\win\verify-tasks.ps1
#
# What it does, per registered PF-* task:
#   1. Print the Action[0].Execute + .Arguments verbatim so the operator
#      can eyeball quoting.
#   2. (Optional, -Run) Start the task, then wait up to $WaitSec for the
#      task's expected log file mtime to advance.
#
# Why this exists: docs/PLAN.md premortem N3. Wrapping every action via
# wscript→cmd→target introduces three layers of quoting (PowerShell here-
# string + Task Scheduler XML round-trip + Win32 CreateProcessW
# re-tokenisation). A misquote silently breaks the launch on the
# production machine while passing every Linux CI smoke test.
#
# This script is the cheap detection hook. Run it after any install-*.ps1
# pass before declaring victory.

param(
    [switch]$Run,
    [int]$WaitSec = 90
)

# Per-task expected log file. Tasks NOT listed here are inspected
# (Action printed) but not run-asserted, because they either have no
# stable log output or are one-shot tasks that shouldn't be triggered
# from a verify script.
$expectedLogs = @{
    "PF-DataLoop"           = "Q:\finance-analyzer\data\loop_out.txt"
    "PF-MetalsLoop"         = "Q:\finance-analyzer\data\metals_loop_out.txt"
    "PF-CryptoLoop"         = "Q:\finance-analyzer\data\crypto_loop_out.txt"
    "PF-OilLoop"            = "Q:\finance-analyzer\data\oil_loop_out.txt"
    "PF-MstrLoop"           = "Q:\finance-analyzer\logs\mstr_loop_out.txt"
    "PF-GoldDigger"         = "Q:\finance-analyzer\data\golddigger_out.txt"
    "PF-SilverMonitor"      = "Q:\finance-analyzer\data\silver_monitor_out.txt"
    "PF-LogRotate"          = "Q:\finance-analyzer\data\log_rotation.log"
    "PF-FixAgentDispatcher" = "Q:\finance-analyzer\data\fix_agent_dispatcher.log"
}

$tasks = Get-ScheduledTask | Where-Object { $_.TaskName -like 'PF-*' } | Sort-Object TaskName

$failures = @()
foreach ($task in $tasks) {
    Write-Host ""
    Write-Host "=== $($task.TaskName) ==="
    $action = $task.Actions[0]
    Write-Host "  Execute:   $($action.Execute)"
    Write-Host "  Arguments: $($action.Arguments)"

    # Sanity check: anything still using bare cmd.exe / powershell.exe /
    # python.exe at the top level (i.e. NOT going through wscript) will
    # still pop a window on this machine.
    if ($action.Execute -notlike "*wscript*") {
        Write-Host "  WARNING: Execute is not wscript — task will still pop a window." -ForegroundColor Yellow
        $failures += "$($task.TaskName): not wrapped via wscript"
    }

    if (-not $Run) {
        continue
    }

    $logPath = $expectedLogs[$task.TaskName]
    if (-not $logPath) {
        Write-Host "  (skipped run-assertion: no expected log mapping)"
        continue
    }

    $beforeMtime = $null
    if (Test-Path $logPath) {
        $beforeMtime = (Get-Item $logPath).LastWriteTime
    }

    Write-Host "  Starting task; waiting up to $WaitSec s for $logPath to advance..."
    Start-ScheduledTask -TaskName $task.TaskName

    $deadline = (Get-Date).AddSeconds($WaitSec)
    $advanced = $false
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
        if (Test-Path $logPath) {
            $cur = (Get-Item $logPath).LastWriteTime
            if (($beforeMtime -eq $null) -or ($cur -gt $beforeMtime)) {
                $advanced = $true
                Write-Host "  OK: $logPath mtime advanced." -ForegroundColor Green
                break
            }
        }
    }

    if (-not $advanced) {
        Write-Host "  FAIL: $logPath did not advance within $WaitSec s." -ForegroundColor Red
        $failures += "$($task.TaskName): log did not advance"
    }
}

Write-Host ""
if ($failures.Count -eq 0) {
    Write-Host "=== All PF-* tasks verified ===" -ForegroundColor Green
    exit 0
} else {
    Write-Host "=== $($failures.Count) failure(s) ===" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}
