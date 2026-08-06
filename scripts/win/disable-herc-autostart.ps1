# Disable auto-start of compute/LLM scheduled tasks on herc2.
#
# Why: herc is a wake-on-demand box. Booting it just to fetch a file used to
# start PF-LlamaRemote (llama-server, GPU) and could fire PF-LLMBackfill /
# PF-LocalLlmReport, burning power for work nobody asked for. Compute should be
# opt-in: wake the machine, then start what you actually want.
#
# Run as Administrator on herc2:
#     powershell -ExecutionPolicy Bypass -File disable-herc-autostart.ps1
#
# Re-enable later with enable-herc-compute.ps1, or per task:
#     schtasks /Change /TN "PF-LlamaRemote" /ENABLE
#
# Deliberately does NOT touch PF-Dashboard (read-only, useful on boot) or
# PF-VerifyTunnel / PF-LogRotate (housekeeping, negligible cost). Nothing here
# places orders — herc has no trading loop installed.

$ErrorActionPreference = "Continue"

$tasks = @(
    "PF-LlamaRemote",        # llama-server, holds the GPU
    "PF-LLMBackfill",
    "PF-LocalLlmReport",
    "PF-LoraTraining",
    "PF-LLMBacktest",
    "PF-MicroBacktest",
    "PF-TSBacktest",
    "PF-Stage1",
    "PF-ProbeEarlystop",
    "PF-AdversarialReview",
    "PF-AfterHoursResearch",
    "PF-AutoImprove"
)

Write-Host "Disabling auto-start compute tasks on herc2..." -ForegroundColor Cyan
$disabled = 0
$missing  = 0

foreach ($t in $tasks) {
    $existing = schtasks /Query /TN $t 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("  {0,-26} not present" -f $t) -ForegroundColor DarkGray
        $missing++
        continue
    }
    schtasks /Change /TN $t /DISABLE 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host ("  {0,-26} DISABLED" -f $t) -ForegroundColor Yellow
        $disabled++
    } else {
        Write-Host ("  {0,-26} FAILED (run as Administrator?)" -f $t) -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "$disabled disabled, $missing not present." -ForegroundColor Green
Write-Host ""
Write-Host "Still enabled on purpose:" -ForegroundColor Cyan
Write-Host "  PF-Dashboard      read-only, serves cached data"
Write-Host "  PF-VerifyTunnel   housekeeping"
Write-Host "  PF-LogRotate      housekeeping"
Write-Host ""
Write-Host "Start compute explicitly when you want it:" -ForegroundColor Cyan
Write-Host "  schtasks /Run /TN PF-LlamaRemote"
Write-Host "  or: powershell -File enable-herc-compute.ps1"
Write-Host ""
Write-Host "Current state of every PF- task:" -ForegroundColor Cyan
schtasks /Query /FO TABLE | Select-String "PF-"
