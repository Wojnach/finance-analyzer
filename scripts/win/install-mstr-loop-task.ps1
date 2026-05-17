# Install the canonical scheduled task for the MSTR loop.
# Run as:
#   powershell -ExecutionPolicy Bypass -File scripts\win\install-mstr-loop-task.ps1
#
# Defaults to PHASE=shadow via MSTR_LOOP_PHASE env var (per
# docs/MSTR_LOOP_NOTES.md). Phase A (live) requires 90 days of shadow data
# and explicit user approval — flip MSTR_LOOP_PHASE=live in the task's
# action only after that gate clears.

$TaskName = "PF-MstrLoop"
$scriptDir = "Q:\finance-analyzer\scripts\win"

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing $TaskName"
}

# Hidden launch via run-hidden.vbs. The phase env var is injected inside
# a wrapper bat invocation: cmd /c "set VAR=x && bat". We give that whole
# string to cmd as a single /c argument so wscript→cmd parses it correctly.
# (set/&& inside one /c-string is the standard cmd pattern.)
$vbs = "$scriptDir\run-hidden.vbs"
$cmdInner = "set MSTR_LOOP_PHASE=shadow && `"$scriptDir\mstr-loop.bat`""
$action = New-ScheduledTaskAction -Execute "wscript.exe" `
    -Argument "`"$vbs`" `"cmd.exe`" `"/c`" `"$cmdInner`"" `
    -WorkingDirectory "Q:\finance-analyzer"

$trigger1 = New-ScheduledTaskTrigger -AtLogOn
# MSTR is US-listed — only relevant during US session (15:30-22:00 CET).
# The loop itself early-exits outside session, but starting earlier means
# the heartbeat is fresh by the time the open hits.
$trigger2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "14:00"

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Days 1) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $TaskName `
    -Action $action -Trigger $trigger1,$trigger2 -Settings $settings `
    -Description "MSTR shadow-mode loop. Runs scripts\win\mstr-loop.bat with PHASE=shadow. Decisions logged to data\mstr_loop_shadow.jsonl, no live orders. Phase A requires 90d shadow data + manual approval."

Write-Host "Registered $TaskName (PHASE=shadow)"
Write-Host "To start now: Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "To stop:      Stop-ScheduledTask -TaskName '$TaskName'"
Write-Host "Logs:         logs\mstr_loop_out.txt"
Write-Host "Shadow log:   data\mstr_loop_shadow.jsonl"
Write-Host "Phase notes:  docs\MSTR_LOOP_NOTES.md"
