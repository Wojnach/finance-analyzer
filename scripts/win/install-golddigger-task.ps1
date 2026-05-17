param(
    [string]$TaskName = "PF-GoldDigger",
    [switch]$Remove
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$runner = Join-Path $repoRoot "scripts\win\golddigger.bat"

if (-not (Test-Path $runner)) {
    throw "Runner not found: $runner"
}

if ($Remove) {
    & schtasks /Delete /TN $TaskName /F
    exit $LASTEXITCODE
}

# Hidden launch via run-hidden.vbs — see docs/HIDDEN_TASKS.md.
$vbs = Join-Path $repoRoot "scripts\win\run-hidden.vbs"
$taskCommand = 'wscript.exe "' + $vbs + '" "cmd.exe" "/c" "' + $runner + '" --live'

& schtasks /Create /TN $TaskName /SC ONLOGON /TR $taskCommand /F
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Created or updated task: $TaskName"
Write-Host "Schedule: on logon"
Write-Host "Command: $taskCommand"
