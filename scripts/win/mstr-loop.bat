@echo off
REM MSTR Loop scheduled-task wrapper.
REM Usage: schtasks /run /tn "PF-MstrLoop"
REM Phase is read from MSTR_LOOP_PHASE env var, default "shadow".
REM
REM Auto-restarts on crash with 30s delay (matches scripts/win/pf-loop.bat
REM + scripts/win/metals-loop.bat conventions). Exit code 11 = duplicate
REM instance -- stop looping so we don't fight another live process.

cd /d Q:\finance-analyzer

:restart
REM Clear Claude Code session markers so any subprocess can launch cleanly
set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=
if "%MSTR_LOOP_PHASE%"=="" set MSTR_LOOP_PHASE=shadow
echo [%date% %time%] Starting mstr loop (phase=%MSTR_LOOP_PHASE%)...
.venv\Scripts\python.exe -u -m portfolio.mstr_loop >> logs\mstr_loop_out.txt 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] mstr loop exited (code %EXIT_CODE%).

REM Duplicate instance detected -- do not loop-restart into the active loop
if %EXIT_CODE% EQU 11 (
    echo [%date% %time%] Another mstr loop instance already holds the lock -- stopping wrapper.
    goto :eof
)

echo [%date% %time%] Restarting in 30s...
REM ping not timeout: timeout errors when stdin is not a console
REM (i.e. when launched hidden via run-hidden.vbs). See docs/HIDDEN_TASKS.md.
ping -n 31 127.0.0.1 >nul
goto restart
