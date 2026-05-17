# Install PF-HealthCheck scheduled tasks (3 tiers)
# Run as Administrator

$pythonExe = "Q:\finance-analyzer\.venv\Scripts\python.exe"
$script = "Q:\finance-analyzer\scripts\health_check.py"
$workDir = "Q:\finance-analyzer"
$vbs = "Q:\finance-analyzer\scripts\win\run-hidden.vbs"

# Helper: build a hidden-launch action (cmd /c invocation under wscript).
# See docs/HIDDEN_TASKS.md.
function New-HiddenAction {
    param([string]$Tier)
    New-ScheduledTaskAction -Execute "wscript.exe" `
        -Argument "`"$vbs`" `"cmd.exe`" `"/c`" `"$pythonExe`" -u `"$script`" --tier $Tier" `
        -WorkingDirectory $workDir
}

# Tier 1: Full check at 11:00 CET (09:00 UTC summer / 10:00 UTC winter)
$action1 = New-HiddenAction -Tier "full"
$trigger1 = New-ScheduledTaskTrigger -Daily -At "11:00"
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
Register-ScheduledTask -TaskName "PF-HealthCheck-Full" -Action $action1 -Trigger $trigger1 -Settings $settings -Description "System health contract - full check (11:00 CET)" -Force

# Tier 2: Pre-US check at 15:25 CET (13:25 UTC summer / 14:25 UTC winter)
$action2 = New-HiddenAction -Tier "pre-us"
$trigger2 = New-ScheduledTaskTrigger -Daily -At "15:25"
Register-ScheduledTask -TaskName "PF-HealthCheck-PreUS" -Action $action2 -Trigger $trigger2 -Settings $settings -Description "System health contract - pre-US-open check (15:25 CET)" -Force

# Tier 3: Post-US check at 22:05 CET (20:05 UTC summer / 21:05 UTC winter)
$action3 = New-HiddenAction -Tier "post-us"
$trigger3 = New-ScheduledTaskTrigger -Daily -At "22:05"
Register-ScheduledTask -TaskName "PF-HealthCheck-PostUS" -Action $action3 -Trigger $trigger3 -Settings $settings -Description "System health contract - post-US-close check (22:05 CET)" -Force

Write-Host "Installed 3 health check tasks:"
Get-ScheduledTask | Where-Object {$_.TaskName -like 'PF-HealthCheck*'} | Select-Object TaskName, State | Format-Table -AutoSize
