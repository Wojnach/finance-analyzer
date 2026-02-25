@echo off
cd /d Q:\finance-analyzer
echo [%date% %time%] Starting Silver ORB Monitor...
.venv\Scripts\python.exe -u data\silver_monitor.py >> logs\silver_monitor.log 2>&1
echo [%date% %time%] Silver ORB Monitor exited.
