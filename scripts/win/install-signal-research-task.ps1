# Install PF-SignalResearch scheduled task
# Run as: powershell -ExecutionPolicy Bypass -File scripts\win\install-signal-research-task.ps1
#
# Schedule:
#   Daily at 18:30 CET (after EU market close, before after-hours research at 22:30)
#   Runs Claude Code CLI with signal research prompt

$taskName = "PF-SignalResearch"
$scriptPath = "Q:\finance-analyzer\scripts\signal-research.bat"

# Remove existing task if present
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Removed existing task: $taskName"
}

# Create the action
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$scriptPath`"" -WorkingDirectory "Q:\finance-analyzer"

# Trigger: daily at 18:30 (after EU close, before after-hours at 22:30)
$trigger = New-ScheduledTaskTrigger -Daily -At "18:30"

# Settings: allow long runs, restart on failure, run whether logged in or not
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -StartWhenAvailable

# Register
Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Daily AI signal research: academic papers, web search, scoring, implementation, backtest, codex review. Runs Claude Opus." `
    -RunLevel Highest

Write-Host ""
Write-Host "Installed: $taskName"
Write-Host "Schedule:  Daily at 18:30 (after EU close)"
Write-Host "Script:    $scriptPath"
Write-Host "Prompt:    Q:\finance-analyzer\docs\signal-research-prompt.md"
Write-Host "Output:    Q:\finance-analyzer\data\signal_research_out.txt"
Write-Host "Timeout:   2 hours max"
Write-Host ""
Write-Host "To run manually:  Start-ScheduledTask -TaskName '$taskName'"
Write-Host "To check status:  Get-ScheduledTask -TaskName '$taskName'"
Write-Host "To remove:        Unregister-ScheduledTask -TaskName '$taskName'"
