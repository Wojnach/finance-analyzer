# Install the canonical scheduled task for the oil (WTI) loop.
# Run as:
#   powershell -ExecutionPolicy Bypass -File scripts\win\install-oil-loop-task.ps1
#
# Parity with PF-CryptoLoop / PF-MetalsLoop. Oil futures trade nearly 24/7
# on CME (Sun 23:00 CET to Fri 22:00 CET), so the weekly trigger covers
# Mon-Fri (Saturday is the only fully closed day).

$TaskName = "PF-OilLoop"
$scriptDir = "Q:\finance-analyzer\scripts\win"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing $TaskName"
}

$action = New-ScheduledTaskAction -Execute "cmd.exe" `
    -Argument "/c `"$scriptDir\oil-loop.bat`"" `
    -WorkingDirectory "Q:\finance-analyzer"

$trigger1 = New-ScheduledTaskTrigger -AtLogOn
$trigger2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday,Monday,Tuesday,Wednesday,Thursday,Friday -At "07:00"

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
    -Description "Oil WTI paper-mode swing loop. Runs scripts\win\oil-loop.bat. DRY_RUN=True until manually flipped via data\oil_swing_config.py."

Write-Host "Registered $TaskName (NOT started - DRY_RUN=True)"
Write-Host "To start now: Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "To stop:      Stop-ScheduledTask -TaskName '$TaskName'"
Write-Host "Logs:         data\oil_loop_out.txt"
Write-Host "Heartbeat:    data\oil_loop.heartbeat"
Write-Host ""
Write-Host "First-run prerequisite: oil warrant catalog must be populated."
Write-Host "Run a one-shot probe with a live Avanza session to fill the catalog:"
Write-Host "  .venv\Scripts\python.exe -u data\oil_loop.py --once --debug"
