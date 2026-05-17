# Install scheduled tasks for Silver Monitor and GoldDigger
# Run as: powershell -ExecutionPolicy Bypass -File scripts\win\install-market-tasks.ps1
#
# Schedule:
#   Start: 07:00 CET daily (Mon-Fri) = EU market pre-open
#   Auto-stop: bat files exit after 22:00 CET
#   Auto-restart on crash: bat loop with 30s delay

$scriptDir = "Q:\finance-analyzer\scripts\win"
$vbs = "$scriptDir\run-hidden.vbs"

# --- PF-MetalsLoop ---
& "$scriptDir\install-metals-loop-task.ps1"
Write-Host ""

# --- PF-SilverMonitor ---
$taskName1 = "PF-SilverMonitor"
$existing1 = Get-ScheduledTask -TaskName $taskName1 -ErrorAction SilentlyContinue
if ($existing1) {
    Unregister-ScheduledTask -TaskName $taskName1 -Confirm:$false
    Write-Host "Removed existing $taskName1"
}

# Hidden launch via run-hidden.vbs — see docs/HIDDEN_TASKS.md.
$action1 = New-ScheduledTaskAction -Execute "wscript.exe" `
    -Argument "`"$vbs`" `"cmd.exe`" `"/c`" `"$scriptDir\silver-monitor.bat`"" `
    -WorkingDirectory "Q:\finance-analyzer"

$trigger1 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "07:00"
$trigger1.Repetition = $null
$triggerLogon1 = New-ScheduledTaskTrigger -AtLogon

$settings1 = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 16) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $taskName1 `
    -Action $action1 -Trigger @($trigger1, $triggerLogon1) -Settings $settings1 `
    -Description "Silver price monitor with Claude analysis. Starts at logon and 07:00 CET weekdays. Auto-restarts on crash."

Write-Host "Registered $taskName1 (logon + 07:00 CET Mon-Fri)"

# --- PF-GoldDigger ---
$taskName2 = "PF-GoldDigger"
$existing2 = Get-ScheduledTask -TaskName $taskName2 -ErrorAction SilentlyContinue
if ($existing2) {
    Unregister-ScheduledTask -TaskName $taskName2 -Confirm:$false
    Write-Host "Removed existing $taskName2"
}

# Hidden launch via run-hidden.vbs — see docs/HIDDEN_TASKS.md.
$action2 = New-ScheduledTaskAction -Execute "wscript.exe" `
    -Argument "`"$vbs`" `"cmd.exe`" `"/c`" `"$scriptDir\golddigger-loop.bat`"" `
    -WorkingDirectory "Q:\finance-analyzer"

$trigger2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "07:00"

$settings2 = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 16) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $taskName2 `
    -Action $action2 -Trigger $trigger2 -Settings $settings2 `
    -Description "GoldDigger intraday gold signal tracker (dry-run). Auto-restarts on crash. Runs 07:00-22:00 CET Mon-Fri."

Write-Host "Registered $taskName2 (07:00 CET Mon-Fri)"

# --- Summary ---
Write-Host ""
Write-Host "=== Scheduled Tasks Installed ==="
Write-Host "PF-MetalsLoop:   on logon + 07:00 CET Mon-Fri, auto-restart on crash"
Write-Host "PF-SilverMonitor: on logon + 07:00 CET Mon-Fri, auto-restart on crash"
Write-Host "PF-GoldDigger:    07:00-22:00 CET Mon-Fri, auto-restart on crash"
Write-Host ""
Write-Host "To start NOW:  Start-ScheduledTask -TaskName 'PF-MetalsLoop'"
Write-Host "               Start-ScheduledTask -TaskName 'PF-SilverMonitor'"
Write-Host "               Start-ScheduledTask -TaskName 'PF-GoldDigger'"
Write-Host "To stop:       Stop-ScheduledTask -TaskName 'PF-SilverMonitor'"
Write-Host "To remove:     Unregister-ScheduledTask -TaskName 'PF-SilverMonitor'"
