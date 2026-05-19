# Install the canonical scheduled task for the main Layer 1 data loop.
# Run as:
#   powershell -ExecutionPolicy Bypass -File scripts\win\install-data-loop-task.ps1
#
# Added 2026-05-19: the original PF-DataLoop was registered by hand
# before the hide-windows merge (commit a2f462b1) was written, so it
# never picked up the run-hidden.vbs wrapper that every other PF-*
# install script uses. As a result, PF-DataLoop -- which is always
# running -- kept popping a visible cmd window on every logon. This
# install script gives it the same wrapper as PF-MetalsLoop et al.

$TaskName = "PF-DataLoop"
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
    -Argument "`"$vbs`" `"cmd.exe`" `"/c`" `"$scriptDir\pf-loop.bat`"" `
    -WorkingDirectory "Q:\finance-analyzer"

$trigger = New-ScheduledTaskTrigger -AtLogOn

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Days 3) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $TaskName `
    -Action $action -Trigger $trigger -Settings $settings `
    -Description "Layer 1 main data loop (60s cycle, signals, triggers). Runs scripts\win\pf-loop.bat with crash recovery."

Write-Host "Registered $TaskName"
Write-Host "To start now: Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "To stop:      Stop-ScheduledTask -TaskName '$TaskName'"
