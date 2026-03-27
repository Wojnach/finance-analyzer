# Install PF-AfterHoursResearch scheduled task
# Runs daily at 22:30 CET (after US market close)
# Uses Claude Opus to do deep research on markets and quant strategies

$taskName = "PF-AfterHoursResearch"
$scriptPath = "Q:\finance-analyzer\scripts\after-hours-research.bat"

# Remove existing task if present
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Removed existing task: $taskName"
}

# Create the action
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$scriptPath`"" -WorkingDirectory "Q:\finance-analyzer"

# Trigger: daily at 22:30 (10:30 PM local time — after US close)
$trigger = New-ScheduledTaskTrigger -Daily -At "22:30"

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
    -Description "After-hours research agent: market review, quant research, signal audit, morning briefing" `
    -RunLevel Highest

Write-Host ""
Write-Host "Installed: $taskName"
Write-Host "Schedule: Daily at 22:30 (after US market close)"
Write-Host "Script: $scriptPath"
Write-Host "Prompt: Q:\finance-analyzer\docs\after-hours-research-prompt.md"
Write-Host "Output: Q:\finance-analyzer\data\after-hours-research-out.txt"
Write-Host ""
Write-Host "To run manually: $scriptPath"
Write-Host "To check status: Get-ScheduledTask -TaskName $taskName"
