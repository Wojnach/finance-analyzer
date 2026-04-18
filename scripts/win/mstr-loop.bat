@echo off
REM MSTR Loop scheduled-task wrapper.
REM Usage: schtasks /run /tn "PF-MstrLoop"
REM Phase is read from MSTR_LOOP_PHASE env var, default "shadow".

cd /d Q:\finance-analyzer
.venv\Scripts\python.exe -u -m portfolio.mstr_loop >> logs\mstr_loop_out.txt 2>&1
