@echo off
REM Portfolio Intelligence — Continuous Loop (every 60s)
REM Run this in a terminal or via Task Scheduler
cd /d Q:\finance-analyzer
.venv\Scripts\python.exe -u portfolio\main.py --loop 60
