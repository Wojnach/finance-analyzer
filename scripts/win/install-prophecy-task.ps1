# Install PF-Prophecy scheduled task
# Runs daily at 10:00 Europe/Stockholm — the Prophecy AI price-prediction agent.
#
# SHIPS DISABLED + the data\prophecy_runs\SYSTEM_DISABLED sentinel is present, so
# installing this does NOT start spending tokens (respects the active freeze).
# Going live is a deliberate two-step human action (see bottom). The .bat bypasses
# claude_gate, so BOTH guards matter.

$taskName = "PF-Prophecy"
$scriptPath = "Q:\finance-analyzer\scripts\prophecy-daily.bat"

# Remove existing task if present
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Removed existing task: $taskName"
}

# Hidden launch via run-hidden.vbs — see docs/HIDDEN_TASKS.md.
$vbs = "Q:\finance-analyzer\scripts\win\run-hidden.vbs"
$action = New-ScheduledTaskAction -Execute "wscript.exe" `
    -Argument "`"$vbs`" `"cmd.exe`" `"/c`" `"$scriptPath`"" `
    -WorkingDirectory "Q:\finance-analyzer"

# Trigger: daily at 10:00 local (Europe/Stockholm)
$trigger = New-ScheduledTaskTrigger -Daily -At "10:00"

# Settings: ExecutionTimeLimit is the wall-clock cost backstop (premortem #2) —
# 13 instruments x deep-research can run long; cap it.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -StartWhenAvailable

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Prophecy: daily AI price-prediction (13 instruments x 10 horizons) via deep-research" `
    -RunLevel Highest

# Ship DISABLED (premortem #3 — do not auto-spend on install during the freeze).
Disable-ScheduledTask -TaskName $taskName | Out-Null

Write-Host ""
Write-Host "Installed: $taskName  (DISABLED — will NOT run yet)"
Write-Host "Schedule:  Daily at 10:00 Europe/Stockholm"
Write-Host "Script:    $scriptPath"
Write-Host "Prompt:    Q:\finance-analyzer\docs\prophecy-prompt.md"
Write-Host ""
Write-Host "TWO guards are active (both must be cleared to go live):"
Write-Host "  1. Task is DISABLED"
Write-Host "  2. data\prophecy_runs\SYSTEM_DISABLED sentinel is present"
Write-Host ""
Write-Host "TO GO LIVE (deliberate):"
Write-Host "  del Q:\finance-analyzer\data\prophecy_runs\SYSTEM_DISABLED"
Write-Host "  Enable-ScheduledTask -TaskName $taskName"
Write-Host ""
Write-Host "TO TEST ONE RUN NOW (still respects sentinel — remove it first to actually spend):"
Write-Host "  $scriptPath"
