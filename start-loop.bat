@echo off  
REM Double-click this to start the trading loop  
cd /d Q:\finance-analyzer  
start "PF-Loop" /min .venv\Scripts\python.exe -u portfolio\main.py --loop 
