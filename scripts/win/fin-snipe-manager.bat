@echo off
cd /d %~dp0\..\..
.venv\Scripts\python.exe -u -m portfolio.fin_snipe_manager %*
