@echo off
REM ============================================================
REM reinstall-all-tasks-elevated.bat
REM
REM Self-elevating launcher that re-registers every PF-* scheduled
REM task with the hide-windows action wrapper (run-hidden.vbs).
REM
REM Created 2026-05-18 as a one-click follow-up to the hide-windows
REM merge (commit a2f462b1). When this script is double-clicked from
REM Explorer:
REM   1. UAC prompt appears once
REM   2. After Yes, an Administrator PowerShell window opens
REM   3. PS iterates Q:\finance-analyzer\scripts\win\install-*.ps1
REM      and runs each one (each install script calls
REM      Unregister-ScheduledTask before Register-ScheduledTask, so
REM      existing tasks are replaced atomically)
REM   4. Then verify-tasks.ps1 -Run is executed to confirm every
REM      task wrapper points through wscript.exe and that each task
REM      writes to its log within 90s of being started
REM
REM Existing running loop processes are NOT killed by this script.
REM They will continue running (with their old launch wrappers) until
REM the next logon, at which point Windows will use the new task
REM definitions. There is therefore no trading interruption.
REM ============================================================

setlocal
set "PS=powershell.exe"
set "SCRIPT_DIR=%~dp0"
set "RUNNER=%SCRIPT_DIR%_reinstall_runner.ps1"

REM Create the runner script that will execute under elevation.
> "%RUNNER%" echo $ErrorActionPreference = 'Continue'
>> "%RUNNER%" echo $scriptDir = 'Q:\finance-analyzer\scripts\win'
>> "%RUNNER%" echo Write-Host '== Reinstalling all PF-* scheduled tasks =='
>> "%RUNNER%" echo $installers = Get-ChildItem (Join-Path $scriptDir 'install-*.ps1')
>> "%RUNNER%" echo foreach ($s in $installers) {
>> "%RUNNER%" echo     Write-Host ''
>> "%RUNNER%" echo     Write-Host "---- $($s.Name) ----"
>> "%RUNNER%" echo     try { ^& $s.FullName } catch { Write-Warning $_ }
>> "%RUNNER%" echo }
>> "%RUNNER%" echo Write-Host ''
>> "%RUNNER%" echo Write-Host '== Verifying tasks =='
>> "%RUNNER%" echo ^& (Join-Path $scriptDir 'verify-tasks.ps1') -Run
>> "%RUNNER%" echo Write-Host ''
>> "%RUNNER%" echo Write-Host 'Done. Review output above for any task that failed to register or did not advance its log.'
>> "%RUNNER%" echo Read-Host 'Press Enter to close'

REM Self-elevate via Start-Process -Verb RunAs.
%PS% -NoProfile -ExecutionPolicy Bypass -Command "Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','%RUNNER%' -Verb RunAs"
endlocal
