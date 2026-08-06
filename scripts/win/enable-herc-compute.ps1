# Turn herc2's compute back on, explicitly, for one work session.
#
# Companion to disable-herc-autostart.ps1. Nothing here runs on boot — you run
# this when you actually want the GPU working, which is the whole point.
#
# Run as Administrator on herc2:
#     powershell -ExecutionPolicy Bypass -File enable-herc-compute.ps1           # LLM server only
#     powershell -ExecutionPolicy Bypass -File enable-herc-compute.ps1 -All      # every compute task
#     powershell -ExecutionPolicy Bypass -File enable-herc-compute.ps1 -Stop     # stop + re-disable
#
# Re-disabling on -Stop is deliberate: leaving them enabled after a session is
# how they end up auto-starting on the next boot again.

param(
    [switch]$All,
    [switch]$Stop
)

$ErrorActionPreference = "Continue"

$core  = @("PF-LlamaRemote")
$extra = @("PF-LLMBackfill", "PF-LocalLlmReport", "PF-LoraTraining",
           "PF-LLMBacktest", "PF-MicroBacktest", "PF-TSBacktest")
$targets = if ($All) { $core + $extra } else { $core }

if ($Stop) {
    Write-Host "Stopping compute and re-disabling auto-start..." -ForegroundColor Cyan
    foreach ($t in ($core + $extra)) {
        schtasks /End    /TN $t 2>$null | Out-Null
        schtasks /Change /TN $t /DISABLE 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { Write-Host ("  {0,-24} stopped + disabled" -f $t) -ForegroundColor Yellow }
    }
    Write-Host "`nherc is idle. Safe to shut down." -ForegroundColor Green
    exit 0
}

Write-Host "Enabling compute on herc2 (this session only)..." -ForegroundColor Cyan
foreach ($t in $targets) {
    $existing = schtasks /Query /TN $t 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Host ("  {0,-24} not present" -f $t) -ForegroundColor DarkGray; continue }
    schtasks /Change /TN $t /ENABLE 2>$null | Out-Null
    schtasks /Run    /TN $t 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { Write-Host ("  {0,-24} ENABLED + started" -f $t) -ForegroundColor Green }
    else { Write-Host ("  {0,-24} FAILED (Administrator?)" -f $t) -ForegroundColor Red }
}

Write-Host ""
Write-Host "When finished, run with -Stop so it does not auto-start next boot:" -ForegroundColor Cyan
Write-Host "  powershell -File enable-herc-compute.ps1 -Stop"
