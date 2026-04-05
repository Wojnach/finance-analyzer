# Install PF-AdversarialReview scheduled task
# Run as: powershell -ExecutionPolicy Bypass -File scripts\win\install-adversarial-review-task.ps1
#
# Schedule:
#   Daily at 17:20 CET/CEST (after market close, before after-hours research)
#   Runs Claude Code CLI with /fgl protocol for dual adversarial review

$taskName = "PF-AdversarialReview"
$scriptPath = "Q:\finance-analyzer\scripts\win\adversarial-review.bat"

# Remove existing task if present
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Removed existing task: $taskName"
}

$action = New-ScheduledTaskAction `
    -Execute $scriptPath `
    -WorkingDirectory "Q:\finance-analyzer"

$trigger = New-ScheduledTaskTrigger -Daily -At "17:20"

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

Register-ScheduledTask -TaskName $taskName `
    -Action $action -Trigger $trigger -Settings $settings `
    -Description "Daily dual adversarial review of finance-analyzer codebase (Codex + Claude with cross-critique). Uses /fgl protocol."

Write-Host ""
Write-Host "Installed: $taskName"
Write-Host "Schedule:  Daily at 17:20 local time"
Write-Host "Script:    $scriptPath"
Write-Host "Output:    Q:\finance-analyzer\data\adversarial_review_out.txt"
Write-Host "Timeout:   2 hours max"
Write-Host ""
Write-Host "To run manually: Start-ScheduledTask -TaskName '$taskName'"
Write-Host "To check status: Get-ScheduledTask -TaskName '$taskName'"
Write-Host "To remove:       Unregister-ScheduledTask -TaskName '$taskName'"
