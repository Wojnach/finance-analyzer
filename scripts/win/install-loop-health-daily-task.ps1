# Install the DAILY loop-health summary scheduled task.
#
# Run as:
#   powershell -ExecutionPolicy Bypass -File scripts\win\install-loop-health-daily-task.ps1
#
# Fires every morning at 08:00 local time. Sends a Telegram summary with:
#   - heartbeat freshness for crypto + oil
#   - mstr_loop_poll.jsonl existence (mstr proxy)
#   - oil + mstr scorecard rollups (paper-mode trade counts, win rate,
#     time-to-live-flip readiness gates)
#
# Complements two other scheduled tasks:
#   - PF-LoopHealthWatchdog (every 30min) — alerts on stale/missing
#     heartbeats with 4h cooldown.
#   - PF-LoopHealthReport-20260515 (one-shot, T+2w) — auto-disables.
#
# This daily task is the "watchdog of the watchdog" — if the daily
# summary stops arriving, you know the WHOLE monitoring chain is dead
# (Telegram, scheduled-task service, the script itself), not just one
# loop. Cheap insurance: one Telegram per day.

$TaskName = "PF-LoopHealthDaily"
$scriptDir = "Q:\finance-analyzer\scripts"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing $TaskName"
}

# Hidden launch via run-hidden.vbs — see docs/HIDDEN_TASKS.md.
$vbs = "Q:\finance-analyzer\scripts\win\run-hidden.vbs"
$pythonExe = "Q:\finance-analyzer\.venv\Scripts\python.exe"
$action = New-ScheduledTaskAction -Execute "wscript.exe" `
    -Argument "`"$vbs`" `"cmd.exe`" `"/c`" `"$pythonExe`" `"$scriptDir\loop_health_report.py`"" `
    -WorkingDirectory "Q:\finance-analyzer"

# 08:00 local — well after EU pre-open prep (07:00) so the loops have
# logged a few cycles, and before the user's typical morning check.
$trigger = New-ScheduledTaskTrigger -Daily -At "08:00"

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

Register-ScheduledTask -TaskName $TaskName `
    -Action $action -Trigger $trigger -Settings $settings `
    -Description "Daily loop-health summary at 08:00 local. Sends Telegram with heartbeat freshness + paper-mode scorecards. Insurance against the WHOLE monitoring chain dying silently."

Write-Host "Registered $TaskName (daily 08:00 local)"
Write-Host "To verify:    Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
Write-Host "To run now:   Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "To stop:      Stop-ScheduledTask -TaskName '$TaskName'"
