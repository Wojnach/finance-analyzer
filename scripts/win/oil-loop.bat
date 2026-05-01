@echo off
REM Oil Intraday Trading Loop — WTI paper-mode swing subsystem.
REM Auto-restarts on crash with 30s delay. Exit code 11 means another
REM instance already holds the singleton lock — we stop instead of
REM fork-bombing into the live instance.
cd /d Q:\finance-analyzer

:restart
set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=
echo [%date% %time%] Starting oil loop...
.venv\Scripts\python.exe -u data\oil_loop.py --loop > data\oil_loop_out.txt 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] Oil loop exited (code %EXIT_CODE%).

if %EXIT_CODE% EQU 11 (
    echo [%date% %time%] Another oil loop instance already holds the lock -- stopping wrapper.
    goto :eof
)

echo [%date% %time%] Restarting in 30s...
timeout /t 30 /nobreak >nul
goto restart
