# install-rc-watchdog-task.ps1 — Run as Administrator
# Creates a scheduled task that runs rc-watchdog.ps1 every 30 minutes.
# The watchdog proactively recycles RC servers before the 24h session timeout
# and detects/kills zombies. Sends Telegram alerts on any action.

$taskName   = "PF-RC-Watchdog"
$scriptPath = "Q:\finance-analyzer\scripts\win\rc-watchdog.ps1"

# Remove existing task if present
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

# Hidden launch via run-hidden.vbs + powershell -WindowStyle Hidden.
# See docs/HIDDEN_TASKS.md and docs/PLAN.md N3.
$vbs = "Q:\finance-analyzer\scripts\win\run-hidden.vbs"
$action = New-ScheduledTaskAction -Execute "wscript.exe" `
    -Argument "`"$vbs`" `"powershell.exe`" `"-NoProfile`" `"-ExecutionPolicy`" `"Bypass`" `"-WindowStyle`" `"Hidden`" `"-File`" `"$scriptPath`""

# Trigger: every 30 minutes, starting now, repeating indefinitely
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 30) `
    -RepetitionDuration ([TimeSpan]::MaxValue)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 2) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "RC server watchdog: proactive 20h recycle + zombie detection + Telegram alerts (every 30 min)" `
    -RunLevel Highest

Write-Host ""
Write-Host "=== Installed ==="
Write-Host "  Task:     $taskName"
Write-Host "  Interval: every 30 minutes"
Write-Host "  Script:   $scriptPath"
Write-Host "  Actions:  recycle at 20h, kill zombies, Telegram alert"
Write-Host ""
Write-Host "To verify:"
Write-Host "  schtasks /Query /TN '$taskName' /V /FO LIST"
Write-Host ""
Write-Host "To test now:"
Write-Host "  schtasks /Run /TN '$taskName'"
Write-Host "  # or: powershell -File `"$scriptPath`""
Write-Host ""
