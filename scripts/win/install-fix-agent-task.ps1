# install-fix-agent-task.ps1 — Run as Administrator
# Creates PF-FixAgentDispatcher: fires every 10 minutes, runs
# scripts/fix_agent_dispatcher.py. The dispatcher is a no-op when
# data/critical_errors.jsonl has no unresolved entries, so firing
# frequently is cheap. See docs/plans/2026-04-13-auto-spawn-fix-agent.md.

$taskName    = "PF-FixAgentDispatcher"
$pythonPath  = "Q:\finance-analyzer\.venv\Scripts\python.exe"
$scriptPath  = "Q:\finance-analyzer\scripts\fix_agent_dispatcher.py"
$workingDir  = "Q:\finance-analyzer"

# Remove existing task if present (idempotent install)
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

# Trigger: every 10 minutes, indefinitely
$trigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 10) `
    -RepetitionDuration ([TimeSpan]::MaxValue)

# Action: python scripts/fix_agent_dispatcher.py
$action = New-ScheduledTaskAction `
    -Execute $pythonPath `
    -Argument "-u `"$scriptPath`"" `
    -WorkingDirectory $workingDir

# Settings: cap runtime at 20 minutes (agent timeout is 15 min + buffer).
# Don't run if battery, skip if a previous instance is still running (
# MultipleInstances=IgnoreNew), start when available after wake.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 20) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Auto-spawn a Claude fix agent when data/critical_errors.jsonl has unresolved entries. Kill switch: touch data/fix_agent.disabled." `
    | Out-Null

Write-Host ""
Write-Host "=== $taskName installed ==="
Write-Host "Every 10 minutes: $pythonPath -u $scriptPath"
Write-Host "Working dir:       $workingDir"
Write-Host "Kill switch:       touch Q:\finance-analyzer\data\fix_agent.disabled"
Write-Host ""
Write-Host "To verify: schtasks /Query /TN '$taskName' /V /FO LIST"
Write-Host "To remove: Unregister-ScheduledTask -TaskName '$taskName' -Confirm:`$false"
