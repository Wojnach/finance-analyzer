<#
.SYNOPSIS
  Cleanly restart a PF scheduled task (DataLoop, MetalsLoop, or both).

.DESCRIPTION
  `schtasks /end <task>` only signals the wrapper batch script — the
  child python.exe spawned by `START /B /WAIT` survives as an orphan,
  keeping the old code in memory and blocking the file-singleton lock
  the new instance tries to acquire (exit code 11). Result: schtasks
  /run silently no-ops because the orphan still holds the resources
  the new wrapper would need.

  This script does the right thing in one shot:
    1. Finds python.exe processes whose CommandLine matches the loop
       entry-point (portfolio.main --loop or portfolio.metals_loop).
    2. Stop-Process -Force on every matching PID.
    3. Calls schtasks /end + /run on the corresponding task name.

.PARAMETER Target
  loop   — restart PF-DataLoop (main signal cycle)
  metals — restart PF-MetalsLoop (metals warrant trading)
  all    — restart both
  Default: loop.

.EXAMPLE
  pwsh -File scripts/win/pf-restart.ps1
  pwsh -File scripts/win/pf-restart.ps1 -Target metals
  pwsh -File scripts/win/pf-restart.ps1 -Target all
#>
param(
    [ValidateSet('loop', 'metals', 'all')]
    [string]$Target = 'loop'
)

$targets = @()
# Match patterns are regexes (NOT regex-escaped); they need to handle both
# the .venv launcher and the python312 actual interpreter, plus path
# separator variations. The `\\` in the patterns matches a literal backslash
# in the regex, which is what the live `tasklist` output shows.
switch ($Target) {
    'loop'   { $targets += @{Task='PF-DataLoop';   Match='main\.py.*--loop'} }
    'metals' { $targets += @{Task='PF-MetalsLoop'; Match='metals_loop\.py'} }
    'all'    {
        $targets += @{Task='PF-DataLoop';   Match='main\.py.*--loop'}
        $targets += @{Task='PF-MetalsLoop'; Match='metals_loop\.py'}
    }
}

foreach ($t in $targets) {
    Write-Host "==> Restarting $($t.Task)" -ForegroundColor Cyan

    # Find orphan-prone python processes by CommandLine match. The match
    # string is treated as a regex (already escaped where literal characters
    # need it).
    $procs = Get-CimInstance Win32_Process -Filter 'name="python.exe"' |
        Where-Object { $_.CommandLine -match $t.Match }

    if ($procs) {
        foreach ($p in $procs) {
            Write-Host "    killing PID $($p.ProcessId) ($([datetime]$p.CreationDate))"
            try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop }
            catch { Write-Warning "    Stop-Process failed for $($p.ProcessId): $_" }
        }
        Start-Sleep -Seconds 2
    } else {
        Write-Host "    no running python.exe matching '$($t.Match)'"
    }

    # Stop the scheduled task wrapper (cmd.exe running pf-*.bat). Safe even
    # if it's already gone — schtasks returns 1 in that case which we ignore.
    & schtasks.exe /end /tn $t.Task 2>&1 | Out-Null

    # Start fresh
    & schtasks.exe /run /tn $t.Task
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "    schtasks /run returned $LASTEXITCODE for $($t.Task)"
    }
}

Write-Host ""
Write-Host "Verifying new processes..." -ForegroundColor Cyan
Start-Sleep -Seconds 5
foreach ($t in $targets) {
    $procs = Get-CimInstance Win32_Process -Filter 'name="python.exe"' |
        Where-Object { $_.CommandLine -match $t.Match }
    if ($procs) {
        foreach ($p in $procs) {
            Write-Host "  $($t.Task): PID $($p.ProcessId) started $([datetime]$p.CreationDate)" -ForegroundColor Green
        }
    } else {
        Write-Warning "  $($t.Task): no python.exe found yet — task may still be starting"
    }
}
