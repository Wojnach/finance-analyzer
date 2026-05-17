# Install the canonical scheduled task for the brokered metals loop.
# Run as:
#   powershell -ExecutionPolicy Bypass -File scripts\win\install-metals-loop-task.ps1

$TaskName = "PF-MetalsLoop"
$scriptDir = "Q:\finance-analyzer\scripts\win"
$vbs = "$scriptDir\run-hidden.vbs"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing $TaskName"
}

# Hidden launch: wscript.exe run-hidden.vbs "cmd.exe" "/c" "<bat>"
# See docs/HIDDEN_TASKS.md. Multi-arg form survives Task Scheduler XML
# round-trip with one quoting layer; do not collapse into a single arg.
$action = New-ScheduledTaskAction -Execute "wscript.exe" `
    -Argument "`"$vbs`" `"cmd.exe`" `"/c`" `"$scriptDir\metals-loop.bat`"" `
    -WorkingDirectory "Q:\finance-analyzer"

$trigger1 = New-ScheduledTaskTrigger -AtLogOn
$trigger2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "07:00"

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Days 3) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $TaskName `
    -Action $action -Trigger $trigger1,$trigger2 -Settings $settings `
    -Description "Brokered metals execution loop. Runs scripts\\win\\metals-loop.bat and auto-restarts on crash."

Write-Host "Registered $TaskName"
Write-Host "To start now: Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "To stop:      Stop-ScheduledTask -TaskName '$TaskName'"
