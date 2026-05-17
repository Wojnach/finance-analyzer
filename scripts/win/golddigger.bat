@echo off
title GoldDigger Signal Tracker
cd /d Q:\finance-analyzer

:restart
echo [%date% %time%] Starting GoldDigger...
.venv\Scripts\python.exe -u -m portfolio.golddigger %* >> data\golddigger_out.txt 2>&1
echo [%date% %time%] GoldDigger exited (code %ERRORLEVEL%). Restarting in 30s...
REM ping not timeout: timeout errors when stdin is not a console
REM (i.e. when launched hidden via run-hidden.vbs). See docs/HIDDEN_TASKS.md.
ping -n 31 127.0.0.1 >nul
goto restart
