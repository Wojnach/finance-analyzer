# Install the periodic loop-health watchdog scheduled task.
#
# Run as:
#   powershell -ExecutionPolicy Bypass -File scripts\win\install-loop-health-watchdog-task.ps1
#
# Fires every 30 minutes from logon onward. Sends a consolidated telegram
# alert when any loop heartbeat is stale or missing. Per-loop cooldown
# (4h default in scripts/loop_health_watchdog.py) prevents alert spam
# from a persistently-dead loop.

$TaskName = "PF-LoopHealthWatchdog"
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
    -Argument "`"$vbs`" `"cmd.exe`" `"/c`" `"$pythonExe`" `"$scriptDir\loop_health_watchdog.py`"" `
    -WorkingDirectory "Q:\finance-analyzer"

# Trigger every 30 minutes, indefinitely, starting at logon
$trigger = New-ScheduledTaskTrigger -AtLogOn
$trigger.Repetition = (New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 30) `
    -RepetitionDuration (New-TimeSpan -Days 365)).Repetition

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

Register-ScheduledTask -TaskName $TaskName `
    -Action $action -Trigger $trigger -Settings $settings `
    -Description "Periodic loop-health watchdog. Reads data/*_loop.heartbeat every 30min and sends telegram alerts on stale/missing. 4h cooldown per loop."

Write-Host "Registered $TaskName (every 30min)"
Write-Host "To verify:    Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
Write-Host "To run now:   Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "To stop:      Stop-ScheduledTask -TaskName '$TaskName'"
Write-Host "Cooldown state: data\loop_health_watchdog_state.json"
