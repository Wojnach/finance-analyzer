# install-rc-server-task.ps1 — Run as Administrator
# Creates two tasks for always-on Claude Code RC servers:
#   PF-RemoteControl       — At logon (immediate launch, no delay)
#   PF-RemoteControl-Wake  — On wake from sleep (30s delay for auto-reconnect)

$scriptPath = "Q:\finance-analyzer\scripts\win\rc-server-ensure.ps1"

# ---------- Task 1: Logon trigger (no delay) ----------
$logonTask = "PF-RemoteControl"
Unregister-ScheduledTask -TaskName $logonTask -Confirm:$false -ErrorAction SilentlyContinue

# Hidden launch via run-hidden.vbs + powershell -WindowStyle Hidden.
# See docs/HIDDEN_TASKS.md and docs/PLAN.md N3.
$vbs = "Q:\finance-analyzer\scripts\win\run-hidden.vbs"
$logonAction = New-ScheduledTaskAction -Execute "wscript.exe" `
    -Argument "`"$vbs`" `"powershell.exe`" `"-NoProfile`" `"-ExecutionPolicy`" `"Bypass`" `"-WindowStyle`" `"Hidden`" `"-File`" `"$scriptPath`""

$logonTrigger = New-ScheduledTaskTrigger -AtLogOn

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask `
    -TaskName $logonTask `
    -Action $logonAction `
    -Trigger $logonTrigger `
    -Settings $settings `
    -Description "Launch Claude Code RC servers on logon" `
    -RunLevel Highest

# ---------- Task 2: Wake-from-sleep trigger (with -WakeDelay) ----------
$wakeTask = "PF-RemoteControl-Wake"
Unregister-ScheduledTask -TaskName $wakeTask -Confirm:$false -ErrorAction SilentlyContinue

# Hidden launch via run-hidden.vbs + powershell -WindowStyle Hidden.
$wakeAction = New-ScheduledTaskAction -Execute "wscript.exe" `
    -Argument "`"$vbs`" `"powershell.exe`" `"-NoProfile`" `"-ExecutionPolicy`" `"Bypass`" `"-WindowStyle`" `"Hidden`" `"-File`" `"$scriptPath`" `"-WakeDelay`""

# Register with a placeholder trigger first, then replace via XML for event trigger
$placeholderTrigger = New-ScheduledTaskTrigger -AtLogOn
Register-ScheduledTask `
    -TaskName $wakeTask `
    -Action $wakeAction `
    -Trigger $placeholderTrigger `
    -Settings $settings `
    -Description "Check and restart Claude Code RC servers after wake from sleep" `
    -RunLevel Highest

# Replace logon trigger with wake-from-sleep event trigger
$xml = Export-ScheduledTask -TaskName $wakeTask

# Remove the placeholder LogonTrigger and insert EventTrigger
$xml = $xml -replace '<LogonTrigger>[\s\S]*?</LogonTrigger>', @"
<EventTrigger>
      <Enabled>true</Enabled>
      <Subscription>&lt;QueryList&gt;&lt;Query Id="0" Path="System"&gt;&lt;Select Path="System"&gt;*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]&lt;/Select&gt;&lt;/Query&gt;&lt;/QueryList&gt;</Subscription>
    </EventTrigger>
"@

Unregister-ScheduledTask -TaskName $wakeTask -Confirm:$false
Register-ScheduledTask -TaskName $wakeTask -Xml $xml

# ---------- Summary ----------
Write-Host ""
Write-Host "=== Installed ==="
Write-Host "  $logonTask        — At logon (immediate)"
Write-Host "  $wakeTask   — On wake from sleep (30s delay, connection check)"
Write-Host "  Script: $scriptPath"
Write-Host ""
Write-Host "To verify:"
Write-Host "  schtasks /Query /TN '$logonTask' /V /FO LIST"
Write-Host "  schtasks /Query /TN '$wakeTask' /V /FO LIST"
Write-Host ""
Write-Host "To test now:"
Write-Host "  schtasks /Run /TN '$wakeTask'"
Write-Host ""
