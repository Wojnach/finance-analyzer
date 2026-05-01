# One-time scheduled task: paper-mode health check 2 weeks after the
# 2026-05-01 midfinance merge.
#
# Run as:
#   powershell -ExecutionPolicy Bypass -File scripts\win\install-loop-health-report-task.ps1
#
# Fires once on 2026-05-15 at 18:00 local time (post-EU close, before
# US close). The task auto-disables itself after firing. To re-arm for
# a later date, edit $RunOnce below and re-run this script.

$TaskName = "PF-LoopHealthReport-20260515"
$RunOnce = "2026-05-15T18:00:00"   # local time (Europe/Stockholm)

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing $TaskName"
}

$action = New-ScheduledTaskAction -Execute "Q:\finance-analyzer\.venv\Scripts\python.exe" `
    -Argument "Q:\finance-analyzer\scripts\loop_health_report.py" `
    -WorkingDirectory "Q:\finance-analyzer"

$trigger = New-ScheduledTaskTrigger -Once -At $RunOnce

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

Register-ScheduledTask -TaskName $TaskName `
    -Action $action -Trigger $trigger -Settings $settings `
    -Description "One-time paper-mode health check for crypto+MSTR+oil swing loops. Fires 2026-05-15 18:00 local. Sends Telegram summary."

Write-Host "Registered $TaskName for $RunOnce"
Write-Host "To verify:    Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
Write-Host "To run early: Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "To cancel:    Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
