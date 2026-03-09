param(
    [string]$TaskName = "PF-LocalLlmReport",
    [string]$Time = "18:10",
    [int]$Days = 30,
    [switch]$Remove
)

$ErrorActionPreference = "Stop"

if ($Time -notmatch '^\d{2}:\d{2}$') {
    throw "Time must be in HH:mm format."
}

if ($Days -lt 1) {
    throw "Days must be >= 1."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$runner = Join-Path $repoRoot "scripts\win\pf-local-llm-report.bat"

if (-not (Test-Path $runner)) {
    throw "Runner not found: $runner"
}

if ($Remove) {
    & schtasks /Delete /TN $TaskName /F
    exit $LASTEXITCODE
}

$taskCommand = '"' + $runner + '" ' + $Days

& schtasks /Create /TN $TaskName /SC DAILY /ST $Time /TR $taskCommand /F
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Created or updated task: $TaskName"
Write-Host "Schedule: daily at $Time"
Write-Host "Command: $taskCommand"
