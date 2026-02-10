@echo off
REM Run hyperparameter optimization for TABaseStrategy.
REM Usage: ft-hyperopt.bat [epochs] [extra args...]
REM Example: ft-hyperopt.bat 500 --timerange 20240101-

setlocal
set "PROJECT_DIR=%~dp0..\.."
call "%PROJECT_DIR%\.venv\Scripts\activate.bat"

set "EPOCHS=%1"
if "%EPOCHS%"=="" set "EPOCHS=100"
shift

freqtrade hyperopt ^
    --config "%PROJECT_DIR%\config.json" ^
    --strategy TABaseStrategy ^
    --strategy-path "%PROJECT_DIR%\user_data\strategies" ^
    --hyperopt-loss SharpeHyperOptLossDaily ^
    --spaces buy sell ^
    --epochs %EPOCHS% ^
    %1 %2 %3 %4 %5 %6 %7 %8 %9
