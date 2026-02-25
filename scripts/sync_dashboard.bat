@echo off
REM ============================================================
REM  Sync dashboard data to raanman.lol/bets (GitHub Pages)
REM
REM  Usage:
REM    sync_dashboard.bat              Single sync
REM    sync_dashboard.bat --loop       Continuous (every 5 min)
REM    sync_dashboard.bat --loop --interval 600  Custom interval
REM
REM  For Task Scheduler: use --loop mode, or schedule single runs
REM ============================================================

cd /d Q:\CaludesRoom\finance-analyzer
.venv\Scripts\python.exe scripts\sync_dashboard.py %* >> logs\sync_dashboard.log 2>&1
