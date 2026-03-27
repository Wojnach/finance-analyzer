@echo off
REM PF-MetaLearnerRetrain — Daily LightGBM meta-learner retraining (low priority)
cd /d Q:\finance-analyzer
set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=
set PYTHONPATH=Q:\finance-analyzer

echo [%date% %time%] Starting meta-learner retraining... >> data\meta_learner_retrain_out.txt 2>&1
start /LOW /B .venv\Scripts\python.exe -u portfolio/meta_learner.py >> data\meta_learner_retrain_out.txt 2>&1
echo [%date% %time%] Meta-learner retraining finished (code %ERRORLEVEL%). >> data\meta_learner_retrain_out.txt 2>&1
