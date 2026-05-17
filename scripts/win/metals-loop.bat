@echo off
REM Metals Intraday Trading Loop — Layer 1 data collection + Claude Layer 2 decisions
REM Auto-restarts on crash with 30s delay.
cd /d Q:\finance-analyzer

:restart
REM Clear Claude Code session markers so Layer 2 agent can launch
set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=
echo [%date% %time%] Starting metals loop...
.venv\Scripts\python.exe -u data\metals_loop.py > data\metals_loop_out.txt 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] Metals loop exited (code %EXIT_CODE%).

REM Duplicate instance detected -- do not loop-restart into the active metals loop
if %EXIT_CODE% EQU 11 (
    echo [%date% %time%] Another metals loop instance already holds the lock -- stopping wrapper.
    goto :eof
)

echo [%date% %time%] Restarting in 30s...
REM ping not timeout: timeout errors when stdin is not a console
REM (i.e. when launched hidden via run-hidden.vbs). See docs/HIDDEN_TASKS.md.
ping -n 31 127.0.0.1 >nul
goto restart
