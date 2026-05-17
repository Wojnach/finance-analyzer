# install-shadow-review-task.ps1 — Run from any shell (no Admin required).
# Creates PF-ShadowReview via schtasks: fires daily at 03:30 local time,
# runs scripts/win/shadow-review.bat which invokes review_shadow_signals.py
# --promote --retire. Output tee'd to data/shadow_review.log so the morning
# briefing can pick up flips.
#
# Earlier revision used Register-ScheduledTask + New-ScheduledTaskPrincipal,
# which failed under WSL→cmd.exe→PowerShell nesting because $env:USERNAME
# resolved to an empty/ambiguous identity (HRESULT 0x80070057). The schtasks
# CLI works regardless of nesting and matches the pattern PF-DataLoop /
# PF-Dashboard already use.

$taskName = "PF-ShadowReview"
$batPath  = "Q:\finance-analyzer\scripts\win\shadow-review.bat"

# Remove existing task if present (idempotent install). /F suppresses the
# "no such task" error so a fresh install does not surface as a failure.
schtasks /Delete /TN $taskName /F 2>$null | Out-Null

# Create: daily 03:30, run the wrapper batch hidden via run-hidden.vbs.
# /RL HIGHEST not used because the task only reads JSONL + writes a log —
# no admin operations. See docs/HIDDEN_TASKS.md for the VBS shim.
$vbs = "Q:\finance-analyzer\scripts\win\run-hidden.vbs"
$taskCommand = "wscript.exe `"$vbs`" `"cmd.exe`" `"/c`" `"$batPath`""
schtasks /Create /SC DAILY /ST 03:30 /TN $taskName /TR $taskCommand /F | Out-Null

if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED to install $taskName (schtasks exit $LASTEXITCODE)"
    exit 1
}

Write-Host ""
Write-Host "=== $taskName installed ==="
Write-Host "Daily 03:30: $batPath"
Write-Host "Log:         Q:\finance-analyzer\data\shadow_review.log"
Write-Host ""
Write-Host "Verify:  schtasks /Query /TN '$taskName' /V /FO LIST"
Write-Host "Run now: schtasks /Run /TN '$taskName'"
Write-Host "Remove:  schtasks /Delete /TN '$taskName' /F"
