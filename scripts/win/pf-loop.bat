@echo off
REM Portfolio Intelligence — Continuous Loop (market-aware scheduling)
REM Auto-restarts on crash with 30s delay.
cd /d Q:\finance-analyzer

:restart
REM Clear Claude Code session markers so Layer 2 agent can launch
set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=
echo [%date% %time%] Starting loop...
.venv\Scripts\python.exe -u portfolio\main.py --loop
echo [%date% %time%] Loop exited (code %ERRORLEVEL%). Restarting in 30s...
timeout /t 30 /nobreak >nul
goto restart
