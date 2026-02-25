@echo off
cd /d Q:\finance-analyzer
echo [%date% %time%] Starting Crypto Monitor (BTC/ETH/MSTR)...
.venv\Scripts\python.exe -u data\crypto_monitor.py >> logs\crypto_monitor.log 2>&1
echo [%date% %time%] Crypto Monitor exited.
