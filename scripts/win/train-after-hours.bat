@echo off
REM Scheduled via Windows Task Scheduler at 22:30 CET daily
cd /d Q:\finance-analyzer
.venv\Scripts\python.exe -m portfolio.tinylora_trainer
