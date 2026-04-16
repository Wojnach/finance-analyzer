@echo off
REM Thin wrapper around pf-restart.ps1 so the loop can be restarted from cmd.exe
REM or `cmd.exe /c scripts\win\pf-restart.bat` from WSL. See pf-restart.ps1 for
REM the full rationale on why schtasks /end leaves orphan python.exe children.
REM
REM Usage:
REM   pf-restart.bat            (default: loop / PF-DataLoop)
REM   pf-restart.bat metals     (PF-MetalsLoop only)
REM   pf-restart.bat all        (both)

set TARGET=%~1
if "%TARGET%"=="" set TARGET=loop

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0pf-restart.ps1" -Target %TARGET%
exit /b %ERRORLEVEL%
