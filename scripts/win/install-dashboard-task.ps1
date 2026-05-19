# Install the canonical scheduled task for the Flask dashboard.
# Run as:
#   powershell -ExecutionPolicy Bypass -File scripts\win\install-dashboard-task.ps1
#
# Added 2026-05-19: like PF-DataLoop, the original PF-Dashboard task
# pre-dated the hide-windows merge (commit a2f462b1) and was never
# wrapped through run-hidden.vbs. Result: a python.exe console
# window stayed open for the dashboard's entire lifetime. This
# install script gives it the same wrapper every other PF-* task
# already has.
#
# The dashboard is a Flask process running `python -m dashboard.app`
# on port 5055. It logs to data\portfolio.log via the shared logger.

$TaskName = "PF-Dashboard"
$scriptDir = "Q:\finance-analyzer\scripts\win"
$vbs = "$scriptDir\run-hidden.vbs"
$python = "Q:\finance-analyzer\.venv\Scripts\python.exe"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing $TaskName"
}

# Hidden launch: wscript.exe run-hidden.vbs "python.exe" "-m" "dashboard.app"
# See docs/HIDDEN_TASKS.md. Multi-arg form survives Task Scheduler XML
# round-trip with one quoting layer; do not collapse into a single arg.
$action = New-ScheduledTaskAction -Execute "wscript.exe" `
    -Argument "`"$vbs`" `"$python`" `"-m`" `"dashboard.app`"" `
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
    -Description "Flask dashboard (port 5055). Runs python -m dashboard.app with crash recovery."

Write-Host "Registered $TaskName"
Write-Host "To start now: Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "To stop:      Stop-ScheduledTask -TaskName '$TaskName'"
