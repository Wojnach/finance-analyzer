@echo off
title Silver Monitor (auto-restart)
cd /d Q:\finance-analyzer

:restart
echo [%date% %time%] Starting Silver Monitor...
.venv\Scripts\python.exe -u data\silver_monitor.py >> data\silver_monitor_out.txt 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] Silver Monitor exited (code %EXIT_CODE%).

REM Duplicate instance detected -- do not loop-restart into the active monitor
if %EXIT_CODE% EQU 11 (
    echo [%date% %time%] Another Silver Monitor instance already holds the lock -- stopping wrapper.
    goto :eof
)

REM Check if within market hours (07:00-22:00 CET = 06:00-21:00 UTC)
REM If outside hours, exit instead of restarting
for /f "tokens=1-2 delims=:" %%a in ("%time: =0%") do set HOUR=%%a
if %HOUR% GEQ 22 (
    echo [%date% %time%] Outside market hours -- stopping.
    goto :eof
)
if %HOUR% LSS 7 (
    echo [%date% %time%] Outside market hours -- stopping.
    goto :eof
)

echo [%date% %time%] Restarting in 30s...
REM ping not timeout: timeout errors when stdin is not a console
REM (i.e. when launched hidden via run-hidden.vbs). See docs/HIDDEN_TASKS.md.
ping -n 31 127.0.0.1 >nul
goto restart
