@echo off
REM Portfolio Intelligence — Continuous Loop (market-aware scheduling)
REM Auto-restarts on crash with 30s delay.
REM Uses START /WAIT to isolate Python in its own process group,
REM preventing Ctrl+C from killing the restart loop.
cd /d Q:\finance-analyzer

:restart
REM Clear Claude Code session markers so Layer 2 agent can launch
set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=
set PYTHONPATH=Q:\finance-analyzer
echo [%date% %time%] Starting loop...
START /B /WAIT .venv\Scripts\python.exe -u portfolio\main.py --loop >> data\loop_out.txt 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] Loop exited (code %EXIT_CODE%).

REM Duplicate instance detected -- do not loop-restart into the active main loop
if %EXIT_CODE% EQU 11 (
    echo [%date% %time%] Another main loop instance already holds the lock -- stopping wrapper.
    goto :eof
)

echo [%date% %time%] Restarting in 30s...
REM ping not timeout: timeout errors when stdin is not a console
REM (i.e. when launched hidden via run-hidden.vbs). See docs/HIDDEN_TASKS.md.
ping -n 31 127.0.0.1 >nul
goto restart
