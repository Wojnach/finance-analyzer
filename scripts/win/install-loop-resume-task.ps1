# install-loop-resume-task.ps1 — Run as Administrator
# Creates PF-LoopResume: fires on wake-from-sleep, runs pf-loop-ensure.ps1

$taskName = "PF-LoopResume"
$scriptPath = "Q:\finance-analyzer\scripts\win\pf-loop-ensure.ps1"

# Remove existing task if present
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

# Trigger: Event ID 1 from Microsoft-Windows-Power-Troubleshooter = resume from sleep
$trigger = New-ScheduledTaskTrigger -AtLogOn  # placeholder, replaced by XML below

# Action: run the ensure script hidden via run-hidden.vbs.
# See docs/HIDDEN_TASKS.md. wscript → powershell -WindowStyle Hidden
# is belt-and-braces: wscript hides the console, -WindowStyle Hidden
# prevents PowerShell from briefly flashing a window on cold start.
$vbs = "Q:\finance-analyzer\scripts\win\run-hidden.vbs"
$action = New-ScheduledTaskAction -Execute "wscript.exe" `
    -Argument "`"$vbs`" `"powershell.exe`" `"-NoProfile`" `"-ExecutionPolicy`" `"Bypass`" `"-WindowStyle`" `"Hidden`" `"-File`" `"$scriptPath`""

# Settings
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

# Register with logon trigger first (we'll add the event trigger via XML)
$task = Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Ensures pf-loop.bat is running after wake-from-sleep or logon" `
    -RunLevel Highest

# Now add the Event trigger (Power-Troubleshooter Event ID 1 = resume from suspend)
# Export, modify XML, re-import
$xml = Export-ScheduledTask -TaskName $taskName

# Insert event trigger XML before </Triggers>
$eventTrigger = @"
    <EventTrigger>
      <Enabled>true</Enabled>
      <Subscription>&lt;QueryList&gt;&lt;Query Id="0" Path="System"&gt;&lt;Select Path="System"&gt;*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]&lt;/Select&gt;&lt;/Query&gt;&lt;/QueryList&gt;</Subscription>
    </EventTrigger>
"@

$xml = $xml -replace '</Triggers>', "$eventTrigger`n  </Triggers>"

# Re-register with updated XML
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
Register-ScheduledTask -TaskName $taskName -Xml $xml

Write-Host ""
Write-Host "=== $taskName installed ==="
Write-Host "Triggers: (1) At logon, (2) On wake from sleep (Event ID 1)"
Write-Host "Action: powershell -File $scriptPath"
Write-Host ""
Write-Host "To verify: schtasks /Query /TN '$taskName' /V /FO LIST"
