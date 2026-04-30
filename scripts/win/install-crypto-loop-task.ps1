# Install the canonical scheduled task for the crypto (BTC+ETH) loop.
# Run as:
#   powershell -ExecutionPolicy Bypass -File scripts\win\install-crypto-loop-task.ps1
#
# Parity with PF-MetalsLoop:
#   - AtLogOn + weekly Mon-Fri 07:00 triggers
#   - 3-day execution time limit (the wrapper auto-restarts on crash)
#   - Multiple-instance ignored (singleton lock at the Python level)
#   - 3 restarts with 1-min interval on hard failures
#
# This script DOES NOT auto-start the task. After install, the user runs:
#   Start-ScheduledTask -TaskName 'PF-CryptoLoop'
# Until then, the loop is registered but inert (paper-mode means zero
# trading risk anyway, but explicit user action is the canonical pattern).

$TaskName = "PF-CryptoLoop"
$scriptDir = "Q:\finance-analyzer\scripts\win"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing $TaskName"
}

$action = New-ScheduledTaskAction -Execute "cmd.exe" `
    -Argument "/c `"$scriptDir\crypto-loop.bat`"" `
    -WorkingDirectory "Q:\finance-analyzer"

$trigger1 = New-ScheduledTaskTrigger -AtLogOn
$trigger2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday,Saturday,Sunday -At "07:00"

# Crypto trades 24/7 — Sat+Sun included in the weekly trigger (vs metals
# weekday-only). Singleton lock prevents double-start when AtLogOn fires
# while a Saturday-morning trigger also runs.
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
    -Description "Crypto BTC+ETH paper-mode swing loop. Runs scripts\win\crypto-loop.bat. DRY_RUN=True until manually flipped via data\crypto_swing_config.py."

Write-Host "Registered $TaskName (NOT started — DRY_RUN=True)"
Write-Host "To start now: Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "To stop:      Stop-ScheduledTask -TaskName '$TaskName'"
Write-Host "Logs:         data\crypto_loop_out.txt"
Write-Host "Heartbeat:    data\crypto_loop.heartbeat"
