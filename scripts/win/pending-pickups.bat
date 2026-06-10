@echo off
REM PF-PendingPickups -- process due entries in data\pending_pickups.json
REM
REM 2026-06-10: redirection must live INSIDE this batch file. The previous
REM installer passed ">>" and "2>&1" as individually quoted argv tokens
REM through run-hidden.vbs; cmd.exe only honors redirection operators when
REM they are UNQUOTED, so python received them as literal arguments and
REM argparse exited 2 on every scheduled run -- the root cause of the
REM LLM-CRYPTOTRADER-72H pickup sitting 20 days overdue with an empty log
REM (audit docs/IMPROVEMENT_AUDIT_2026-06-10.md, Ops automation P1).
cd /d Q:\finance-analyzer
set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=
set PYTHONPATH=Q:\finance-analyzer

echo [%date% %time%] Starting pending-pickups run... >> data\pending_pickups_task.log 2>&1
.venv\Scripts\python.exe -u scripts\process_pending_pickups.py >> data\pending_pickups_task.log 2>&1
set RC=%ERRORLEVEL%
echo [%date% %time%] pending-pickups finished (code %RC%). >> data\pending_pickups_task.log 2>&1
exit /b %RC%
