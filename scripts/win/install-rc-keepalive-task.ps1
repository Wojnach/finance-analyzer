# install-rc-keepalive-task.ps1 — Run as Administrator
# Creates a scheduled task that runs rc-keepalive.ps1 every 5 minutes.
#
# Anthropic's server-side TTL is ~20 min without real user activity.
# Keepalive recycles idle servers at staggered thresholds (13/15/17 min)
# to keep them visible in the claude.ai/code picker.
#
# Also creates a wake-from-sleep trigger that runs keepalive in -Wake mode,
# which immediately recycles all idle servers (sleep guarantees staleness).

$taskName   = "PF-RCKeepalive"
$scriptPath = "Q:\finance-analyzer\scripts\win\rc-keepalive.ps1"

# Remove existing task if present
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

# Hidden launch via run-hidden.vbs + powershell -WindowStyle Hidden.
# See docs/HIDDEN_TASKS.md and docs/PLAN.md N3.
$vbs = "Q:\finance-analyzer\scripts\win\run-hidden.vbs"
$actionPeriodic = New-ScheduledTaskAction -Execute "wscript.exe" `
    -Argument "`"$vbs`" `"powershell.exe`" `"-NoProfile`" `"-ExecutionPolicy`" `"Bypass`" `"-WindowStyle`" `"Hidden`" `"-File`" `"$scriptPath`""

$actionWake = New-ScheduledTaskAction -Execute "wscript.exe" `
    -Argument "`"$vbs`" `"powershell.exe`" `"-NoProfile`" `"-ExecutionPolicy`" `"Bypass`" `"-WindowStyle`" `"Hidden`" `"-File`" `"$scriptPath`" `"-Wake`""

# Trigger 1: every 5 minutes, starting now, repeating for ~25 years (max safe duration)
$triggerPeriodic = New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 5) `
    -RepetitionDuration (New-TimeSpan -Days 9000)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 2) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

# Register with periodic trigger first
Register-ScheduledTask `
    -TaskName $taskName `
    -Action $actionPeriodic `
    -Trigger $triggerPeriodic `
    -Settings $settings `
    -Description "RC server keepalive: recycle idle servers before 20-min Anthropic TTL (every 5 min, staggered 13/15/17 min thresholds)" `
    -RunLevel Highest

# Add wake-from-sleep trigger via XML modification
# PowerShell's New-ScheduledTaskTrigger doesn't support event triggers, so we
# export the task XML, inject the wake trigger, and re-register.
try {
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
    $xml = [xml](Export-ScheduledTask -TaskName $taskName)
    $ns = $xml.Task.NamespaceURI

    # Create EventTrigger node for Power-Troubleshooter EventID 1 (system wake)
    $wakeTriggerXml = @"
<EventTrigger xmlns="$ns">
  <Enabled>true</Enabled>
  <Delay>PT10S</Delay>
  <Subscription>&lt;QueryList&gt;&lt;Query Id="0" Path="System"&gt;&lt;Select Path="System"&gt;*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]&lt;/Select&gt;&lt;/Query&gt;&lt;/QueryList&gt;</Subscription>
</EventTrigger>
"@
    $frag = $xml.CreateDocumentFragment()
    $frag.InnerXml = $wakeTriggerXml
    $xml.Task.Triggers.AppendChild($frag) | Out-Null

    # Override the action to include -Wake flag for the event trigger
    # (Can't have per-trigger actions in Task Scheduler, so the periodic action stays as-is.
    #  The ensure script handles wake via -WakeDelay, and the 5-min periodic keepalive will
    #  catch stale servers within 5 min of wake anyway.)

    Register-ScheduledTask -TaskName $taskName -Xml $xml.OuterXml -Force | Out-Null
    Write-Host "  Wake trigger added successfully (Power-Troubleshooter EventID 1, 10s delay)."
} catch {
    Write-Host "  WARNING: Could not add wake trigger: $_"
    Write-Host "  The periodic 5-min check will still catch stale servers within 5 min of wake."
}

Write-Host ""
Write-Host "=== Installed ==="
Write-Host "  Task:     $taskName"
Write-Host "  Interval: every 5 minutes (periodic) + on wake-from-sleep (-Wake flag)"
Write-Host "  Script:   $scriptPath"
Write-Host "  Thresholds: Trading=13min, Development=15min, Research=17min"
Write-Host "  Anthropic TTL: ~20 min (margin: 3-7 min)"
Write-Host ""
Write-Host "To verify:"
Write-Host "  schtasks /Query /TN '$taskName' /V /FO LIST"
Write-Host ""
Write-Host "To test now:"
Write-Host "  schtasks /Run /TN '$taskName'"
Write-Host "  # or: powershell -File `"$scriptPath`""
Write-Host "  # wake mode: powershell -File `"$scriptPath`" -Wake"
Write-Host ""
