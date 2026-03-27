# Install PF-MetaLearnerRetrain scheduled task
# Run as: powershell -ExecutionPolicy Bypass -File scripts\win\install-meta-learner-task.ps1
#
# Schedule:
#   Daily at 19:00 CET (after market close, after PF-OutcomeCheck at 18:00)
#   Low priority (start /LOW + os.nice(19) + num_threads=1) to avoid disrupting trading loop

$taskName = "PF-MetaLearnerRetrain"
$scriptPath = "Q:\finance-analyzer\scripts\win\meta-learner-retrain.bat"

# Remove existing task if present
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Removed existing task: $taskName"
}

$action = New-ScheduledTaskAction `
    -Execute $scriptPath `
    -WorkingDirectory "Q:\finance-analyzer"

$trigger = New-ScheduledTaskTrigger -Daily -At "19:00"

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType S4U -RunLevel Limited

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Daily LightGBM meta-learner retraining (low priority, 1 thread)" `
    -Force

Write-Host ""
Write-Host "Installed: $taskName"
Write-Host "Schedule:  Daily at 19:00 (after PF-OutcomeCheck at 18:00)"
Write-Host "Script:    $scriptPath"
Write-Host "Output:    Q:\finance-analyzer\data\meta_learner_retrain_out.txt"
Write-Host ""
Write-Host "To run manually: Start-ScheduledTask -TaskName '$taskName'"
Write-Host "To check status: Get-ScheduledTask -TaskName '$taskName'"
Write-Host "To remove:       Unregister-ScheduledTask -TaskName '$taskName'"
