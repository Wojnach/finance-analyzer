# install-shadow-review-task.ps1 — Run as Administrator
# Creates PF-ShadowReview: fires daily at 03:30 local time, runs
# scripts/review_shadow_signals.py --promote --retire to auto-promote
# eligible shadows and auto-retire degraded promoted ones. Output goes
# to data/shadow_review.log so the morning briefing can pick up flips.
# See docs/plans/2026-05-15-llm-shadow-enrollment for the broader plan.

$taskName    = "PF-ShadowReview"
$pythonPath  = "Q:\finance-analyzer\.venv\Scripts\python.exe"
$scriptPath  = "Q:\finance-analyzer\scripts\review_shadow_signals.py"
$workingDir  = "Q:\finance-analyzer"
$logPath     = "Q:\finance-analyzer\data\shadow_review.log"

# Remove existing task if present (idempotent install)
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

# Trigger: daily at 03:30 (lowest market activity, after midnight outcome backfill)
$trigger = New-ScheduledTaskTrigger -Daily -At "03:30"

# Action: python review_shadow_signals.py --promote --retire, with output
# tee'd to data/shadow_review.log via cmd.exe wrapper (PowerShell adds BOM).
$cmdLine = "/c `"$pythonPath -u `"$scriptPath`" --promote --retire >> `"$logPath`" 2>&1`""
$action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument $cmdLine `
    -WorkingDirectory $workingDir

# Settings: cap runtime at 15 minutes (review reads ~85K JSONL rows, normally
# <30s). Don't run on battery, start when available after wake, skip if a
# previous instance is somehow still running.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Daily 03:30 shadow-registry review. Auto-promote shadows that meet promotion_criteria, auto-retire promoted ones whose 30d rolling accuracy drops > 0.05 below threshold. Output: data/shadow_review.log." `
    | Out-Null

Write-Host ""
Write-Host "=== $taskName installed ==="
Write-Host "Daily 03:30: $pythonPath -u $scriptPath --promote --retire"
Write-Host "Working dir:  $workingDir"
Write-Host "Log:          $logPath"
Write-Host ""
Write-Host "To verify: schtasks /Query /TN '$taskName' /V /FO LIST"
Write-Host "To run now: schtasks /Run /TN '$taskName'"
Write-Host "To remove: Unregister-ScheduledTask -TaskName '$taskName' -Confirm:`$false"
