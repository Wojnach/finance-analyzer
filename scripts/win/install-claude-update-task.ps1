# install-claude-update-task.ps1 -- Run as Administrator
#
# Creates PF-ClaudeUpdate: weekly Sunday 17:30 CET, runs `claude update`
# on the WINDOWS production Claude Code binary
# (C:\Users\Herc2\.local\bin\claude.exe).
#
# Why: that binary runs Layer 2 (`claude -p` in the loop .bat files) and is
# almost never launched interactively, so its native auto-updater never fires
# (unlike the WSL/interactive install, which self-updates on launch). It froze
# at 2.1.144 and missed the opus-4-8 model until a manual update on 2026-05-28.
# Sunday 17:30 chosen: machine is OFF overnight (so the original 04:00 idea was
# dead), 17-18 is a known on-window, and US market is closed Sunday -> a new
# binary settles before Monday trading. StartWhenAvailable covers a missed slot.
#
# Idempotent: re-running deletes + re-registers.

$taskName  = "PF-ClaudeUpdate"
$repoRoot  = "Q:\finance-analyzer"
$bat       = "$repoRoot\scripts\win\claude-update.bat"
$logDir    = "$repoRoot\data"
$logFile   = "$logDir\claude_update_task.log"

Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

# Hide the console window via the standard run-hidden.vbs shim
# (see docs/HIDDEN_TASKS.md). NOTE: run-hidden.vbs quotes every arg, so a
# ">>" passed here arrives as a literal token, not a redirect. Logging is
# therefore done PER-LINE inside claude-update.bat, not via task args.
$vbs = "$repoRoot\scripts\win\run-hidden.vbs"

$action = New-ScheduledTaskAction -Execute "wscript.exe" `
    -Argument "`"$vbs`" `"cmd.exe`" `"/c`" `"$bat`"" `
    -WorkingDirectory $repoRoot

# Weekly Sunday 17:30 CET (machine is off overnight; 17-18 is a known on-window).
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "17:30"

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
    -Description "Weekly Sunday 17:30 CET: updates the Windows production Claude Code binary so the Layer 2 trading agent stays current and gets new models. Log: data\claude_update_task.log" `
    -RunLevel Highest

Write-Host "Registered $taskName (weekly Sunday 17:30, runs claude-update.bat)."
Write-Host "Log: $logFile"
Write-Host "Smoke test:"
Write-Host "  schtasks /run /tn $taskName"
Write-Host "  Get-Content $logFile -Tail 20"
