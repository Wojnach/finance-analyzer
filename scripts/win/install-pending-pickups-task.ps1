# install-pending-pickups-task.ps1 -- Run as Administrator
#
# Creates PF-PendingPickups: daily 08:00 CET, runs
# scripts/process_pending_pickups.py to dispatch any pickup whose
# due_ts has passed. Each pickup is a one-shot job recorded in
# data/pending_pickups.json with a handler whitelisted in
# scripts/process_pending_pickups.py:_HANDLERS.
#
# Idempotent: re-running deletes + re-registers.

$taskName  = "PF-PendingPickups"
$repoRoot  = "Q:\finance-analyzer"
$python    = "$repoRoot\.venv\Scripts\python.exe"
$script    = "$repoRoot\scripts\process_pending_pickups.py"
$logDir    = "$repoRoot\data"
$logFile   = "$logDir\pending_pickups_task.log"

Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

# Hide the console window via the standard run-hidden.vbs shim
# (see docs/HIDDEN_TASKS.md). Output captured to a log so failures
# can be diagnosed without an interactive console.
$vbs = "$repoRoot\scripts\win\run-hidden.vbs"
$cmd = "cmd.exe"
$args = "/c `"$python`" -u `"$script`" >> `"$logFile`" 2>&1"

$action = New-ScheduledTaskAction -Execute "wscript.exe" `
    -Argument "`"$vbs`" `"$cmd`" `"/c`" `"$python`" `"-u`" `"$script`" `">>`" `"$logFile`" `"2>&1`"" `
    -WorkingDirectory $repoRoot

# Daily 08:00 CET. Pickups whose due_ts > now stay pending until their
# due date passes; cron just polls daily.
$trigger = New-ScheduledTaskTrigger -Daily -At "08:00"

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15)

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Processes data/pending_pickups.json each morning. Verdicts land in SESSION_PROGRESS.md and Telegram." `
    -RunLevel Highest

Write-Host "Registered $taskName (daily 08:00, runs process_pending_pickups.py)."
Write-Host "Log: $logFile"
Write-Host "Smoke test:"
Write-Host "  schtasks /run /tn $taskName"
Write-Host "  Get-Content $logFile -Tail 20"
