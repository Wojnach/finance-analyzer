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
$bat       = "$repoRoot\scripts\win\pending-pickups.bat"
$logFile   = "$repoRoot\data\pending_pickups_task.log"

Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

# Hide the console window via the standard run-hidden.vbs shim
# (see docs/HIDDEN_TASKS.md), in "wait" mode so the child's exit code
# propagates to Task Scheduler "Last Result".
#
# 2026-06-10: the previous action passed ">>" and "2>&1" as quoted argv
# tokens through run-hidden.vbs; cmd.exe only honors redirection when
# UNQUOTED, so python received them as literal arguments and argparse
# exited 2 on every scheduled run for ~20 days while the detached vbs
# made Last Result show 0 (audit docs/IMPROVEMENT_AUDIT_2026-06-10.md).
# Redirection now lives inside pending-pickups.bat, where it works.
# (The old dead $cmd/$args variables that suggested working redirection
# are gone — $args also shadowed PowerShell's automatic variable.)
$vbs = "$repoRoot\scripts\win\run-hidden.vbs"

$action = New-ScheduledTaskAction -Execute "wscript.exe" `
    -Argument "`"$vbs`" `"wait`" `"cmd.exe`" `"/c`" `"$bat`"" `
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
