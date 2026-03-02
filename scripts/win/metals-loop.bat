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
echo [%date% %time%] Metals loop exited (code %ERRORLEVEL%). Restarting in 30s...
timeout /t 30 /nobreak >nul
goto restart
