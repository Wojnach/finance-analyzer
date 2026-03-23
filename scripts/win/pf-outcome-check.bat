@echo off
REM PF-OutcomeCheck — Backfill price outcomes for signal accuracy tracking
cd /d Q:\finance-analyzer
set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=
set PYTHONPATH=Q:\finance-analyzer

echo [%date% %time%] Starting outcome backfill... >> data\outcome_check_out.txt 2>&1
.venv\Scripts\python.exe -u portfolio\main.py --check-outcomes >> data\outcome_check_out.txt 2>&1
echo [%date% %time%] Outcome backfill finished (code %ERRORLEVEL%). >> data\outcome_check_out.txt 2>&1
