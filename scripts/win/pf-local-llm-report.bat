@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM Standalone local-LLM report export. Safe to run outside the trading loop.

for %%I in ("%~dp0..\..") do set REPO_ROOT=%%~fI
cd /d "%REPO_ROOT%"

set REPORT_DAYS=%~1
if not defined REPORT_DAYS set REPORT_DAYS=30

set LOG_FILE=%REPO_ROOT%\data\local_llm_report_task.log
set PYTHON_EXE=%REPO_ROOT%\.venv\Scripts\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=Q:\finance-analyzer\.venv\Scripts\python.exe

if not exist "%PYTHON_EXE%" (
  echo [%date% %time%] Missing python executable: %PYTHON_EXE% >> "%LOG_FILE%"
  exit /b 3
)

echo [%date% %time%] Starting local LLM report export (%REPORT_DAYS%d) >> "%LOG_FILE%"
"%PYTHON_EXE%" portfolio\main.py --export-local-llm-report %REPORT_DAYS% >> "%LOG_FILE%" 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] Finished local LLM report export exit=%EXIT_CODE% >> "%LOG_FILE%"

exit /b %EXIT_CODE%
