@echo off
REM GoldDigger — Intraday gold certificate trading bot
REM Usage: scripts\golddigger.bat [--live] [--dry-run]

cd /d Q:\finance-analyzer
.venv\Scripts\python.exe -u -m portfolio.golddigger %*
