@echo off
title GoldDigger Signal Tracker
cd /d Q:\finance-analyzer

:restart
echo [%date% %time%] Starting GoldDigger...
.venv\Scripts\python.exe -u -m portfolio.golddigger
echo [%date% %time%] GoldDigger exited (code %ERRORLEVEL%). Restarting in 30s...
timeout /t 30 /nobreak
goto restart
