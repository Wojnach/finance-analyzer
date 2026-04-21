@echo off
REM PF-LLMBackfill scheduled task command.
REM Runs the probability-log outcome backfill + sentiment A/B shadow backfill.
REM Idempotent; rows without elapsed horizons are skipped and retried later.
cd /d Q:\finance-analyzer
.venv\Scripts\python.exe scripts\backfill_llm_outcomes.py >> data\llm_backfill_out.txt 2>&1
.venv\Scripts\python.exe scripts\backfill_sentiment_shadow.py --horizon 1d >> data\llm_backfill_out.txt 2>&1
