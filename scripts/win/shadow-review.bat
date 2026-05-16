@echo off
REM Daily shadow-registry review wrapper for PF-ShadowReview task.
REM Runs review_shadow_signals.py --promote --retire and tees output to
REM data/shadow_review.log. Invoked by schtasks at 03:30 local.
cd /d Q:\finance-analyzer
"Q:\finance-analyzer\.venv\Scripts\python.exe" -u "Q:\finance-analyzer\scripts\review_shadow_signals.py" --promote --retire >> "Q:\finance-analyzer\data\shadow_review.log" 2>&1
