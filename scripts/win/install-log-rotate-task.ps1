# Install the scheduled task that runs portfolio/log_rotation.py hourly.
# Run as:
#   powershell -ExecutionPolicy Bypass -File scripts\win\install-log-rotate-task.ps1
#
# Why hourly: loop_out.txt grows ~600KB/h at full verbosity. With a 5MB
# rotate threshold that's a ~8h cycle, so an hourly check keeps total
# (live + .1 .. .5.gz) under ~30MB. JSONL files in the policy mostly
# use age-based archive — checking them hourly is harmless idempotent.
#
# This script DOES auto-start the task once registered. Rotation is
# read-only-ish (rename + truncate) — no risk to live trading.

$TaskName = "PF-LogRotate"
$pythonExe = "Q:\finance-analyzer\.venv\Scripts\python.exe"
$workDir   = "Q:\finance-analyzer"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing $TaskName"
}

# Hidden launch via run-hidden.vbs — see docs/HIDDEN_TASKS.md.
$vbs = "Q:\finance-analyzer\scripts\win\run-hidden.vbs"
$action = New-ScheduledTaskAction -Execute "wscript.exe" `
    -Argument "`"$vbs`" `"cmd.exe`" `"/c`" `"$pythonExe`" -m portfolio.log_rotation" `
    -WorkingDirectory $workDir

# Hourly trigger starting in 5 minutes (so first run isn't immediately
# on install).
$startBoundary = (Get-Date).AddMinutes(5)
$trigger = New-ScheduledTaskTrigger -Once -At $startBoundary `
    -RepetitionInterval (New-TimeSpan -Hours 1)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $TaskName `
    -Action $action -Trigger $trigger -Settings $settings `
    -Description "Hourly log rotation for finance-analyzer. Runs portfolio.log_rotation. Rotates loop_out.txt, golddigger_out.txt (size-based) and JSONL files (age-based). Archive: data\archive\."

Write-Host "Registered $TaskName (next run: $startBoundary)"
Write-Host "To run now:      Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "To inspect dry:  $pythonExe -m portfolio.log_rotation --dry-run"
Write-Host "To see sizes:    $pythonExe -m portfolio.log_rotation --status"
Write-Host "Archive dir:     data\archive\"
