@echo off
REM Dashboard sync loop — runs continuously, syncs every 5 minutes
REM Scheduled via Task Scheduler "PF-DashboardSync" (At logon)

cd /d Q:\finance-analyzer

:restart
echo [%date% %time%] Starting dashboard sync loop...
.venv\Scripts\python.exe -u scripts\sync_dashboard.py --loop --interval 300 >> logs\sync_dashboard.log 2>&1
echo [%date% %time%] Sync loop exited, restarting in 30s...
timeout /t 30 /nobreak >nul
goto restart
