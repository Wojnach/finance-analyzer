@echo off
REM Crypto Intraday Trading Loop — BTC + ETH paper-mode swing subsystem.
REM Auto-restarts on crash with 30s delay. Exit code 11 means another
REM instance already holds the singleton lock — we stop instead of
REM fork-bombing into the live instance.
cd /d Q:\finance-analyzer

:restart
REM Clear Claude Code session markers so any subagent invocation can launch.
set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=
echo [%date% %time%] Starting crypto loop...
.venv\Scripts\python.exe -u data\crypto_loop.py --loop > data\crypto_loop_out.txt 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] Crypto loop exited (code %EXIT_CODE%).

REM Duplicate instance detected -- do not loop-restart into the active loop
if %EXIT_CODE% EQU 11 (
    echo [%date% %time%] Another crypto loop instance already holds the lock -- stopping wrapper.
    goto :eof
)

echo [%date% %time%] Restarting in 30s...
REM ping not timeout: timeout errors when stdin is not a console
REM (i.e. when launched hidden via run-hidden.vbs). See docs/HIDDEN_TASKS.md.
ping -n 31 127.0.0.1 >nul
goto restart
