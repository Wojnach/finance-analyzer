@echo off
REM PF-ShadowReview scheduled task command.
REM Reports shadow signals older than 30 d without resolution.
REM Exit code 1 when stale shadows exist — Windows Task Scheduler treats
REM that as a failure so the task history surfaces the alert.
cd /d Q:\finance-analyzer
.venv\Scripts\python.exe scripts\review_shadow_signals.py >> data\shadow_review_out.txt 2>&1
