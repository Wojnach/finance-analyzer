<#
.SYNOPSIS
    Registers the PF-LLMBackfill scheduled task.

.DESCRIPTION
    Runs scripts/win/pf-llm-backfill.bat, which executes:
      * scripts/backfill_llm_outcomes.py   — joins llm_probability_log.jsonl
        rows to realized price moves -> data/llm_probability_outcomes.jsonl
      * scripts/backfill_sentiment_shadow.py --horizon 1d — shadow A/B outcomes

    Those outcome files are what make Brier / log-loss calibration computable
    (portfolio/llm_calibration.compute_metrics), and they feed the dashboard
    /api/calibration + _compute_llm_leaderboard and the shadow promotion gate
    (scripts/review_shadow_signals.py). Without this task the probability log
    grows but never gets outcome-paired, so calibration silently stalls.

    Created 2026-05-29: the PF-LLMBackfill task existed on the production box
    (pf-llm-backfill.bat references it by name) but had NO committed install
    script, unlike every other PF-* task. This file makes the schedule
    reproducible on a fresh machine. See docs handoff for the discovery.

    Cadence: hourly. The backfill is idempotent (already-written
    (ts,signal,ticker,horizon) keys are skipped) and cheap (no GPU, no LLM
    inference — it only reads JSONL + the price-snapshot file). Hourly keeps
    short-horizon rows (3h/12h/1d) paired promptly without waiting for the
    once-daily local-LLM report export.

    Runs ONLY in the interactive user session, NOT as SYSTEM: the repo lives
    on the Q: drive, which is a per-user mapped drive and is not visible to
    the SYSTEM account. The .bat does `cd /d Q:\finance-analyzer`, which would
    fail under SYSTEM.

.NOTES
    Must be run from an elevated (Administrator) PowerShell to register a task.
    Re-running is safe: -Force replaces the existing task definition.
#>

[CmdletBinding()]
param(
    [string]$TaskName = "PF-LLMBackfill",
    # Repo root: default to two levels up from this script (scripts/win/ -> repo).
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    # Hour boundary offset; :05 matches the historical run pattern observed in
    # data/llm_probability_outcomes.jsonl (backfilled_at HH:05).
    [int]$StartMinute = 5
)

$ErrorActionPreference = "Stop"

# --- Admin check -----------------------------------------------------------
$currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
$principalCheck = New-Object Security.Principal.WindowsPrincipal($currentUser)
if (-not $principalCheck.IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)) {
    Write-Error "This script must be run from an elevated (Administrator) PowerShell."
    exit 1
}

# --- Resolve the action target --------------------------------------------
$batPath = Join-Path $RepoRoot "scripts\win\pf-llm-backfill.bat"
if (-not (Test-Path $batPath)) {
    Write-Error "Cannot find backfill batch file at: $batPath"
    exit 2
}

Write-Host "Registering scheduled task '$TaskName'"
Write-Host "  Repo root : $RepoRoot"
Write-Host "  Action    : $batPath"
Write-Host "  User      : $($currentUser.Name) (interactive session — Q: drive required)"

# --- Build task definition -------------------------------------------------
# Launch hidden via run-hidden.vbs (repo convention, see docs/HIDDEN_TASKS.md)
# so this hourly task never flashes a console window. The .bat does its own
# `cd /d Q:\finance-analyzer` and redirects output to data/llm_backfill_out.txt,
# so wscript only needs to fire cmd.exe /c against it.
$vbs = Join-Path $RepoRoot "scripts\win\run-hidden.vbs"
if (-not (Test-Path $vbs)) {
    Write-Error "Cannot find run-hidden.vbs at: $vbs"
    exit 2
}
$action = New-ScheduledTaskAction `
    -Execute "wscript.exe" `
    -Argument "`"$vbs`" `"cmd.exe`" `"/c`" `"$batPath`"" `
    -WorkingDirectory $RepoRoot

# Fire once at HH:$StartMinute today, then repeat every hour forever. Using a
# -Once trigger + repetition (rather than -Daily) so the hourly cadence does
# not reset at midnight.
$startAt = (Get-Date).Date.AddMinutes($StartMinute)
if ($startAt -lt (Get-Date)) { $startAt = $startAt.AddHours(1) }
$trigger = New-ScheduledTaskTrigger -Once -At $startAt `
    -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration ([TimeSpan]::MaxValue)

# Interactive logon so the per-user Q: mapped drive is available. RunLevel
# Limited: the backfill needs no elevation once registered.
$principal = New-ScheduledTaskPrincipal `
    -UserId $currentUser.Name `
    -LogonType Interactive `
    -RunLevel Limited

# StartWhenAvailable: if the box was asleep at the scheduled minute, run as
# soon as it wakes instead of skipping the slot. The job is idempotent so a
# late/extra run is harmless.
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "Hourly outcome backfill for LLM probability log -> Brier/log-loss calibration. Installed by scripts/win/install-llm-backfill-task.ps1." `
    -Force | Out-Null

Write-Host "Done. '$TaskName' runs hourly starting $startAt."
Write-Host "Verify : schtasks /query /tn `"$TaskName`" /v /fo LIST"
Write-Host "Run now: schtasks /run /tn `"$TaskName`""
