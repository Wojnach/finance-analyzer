@echo off
REM Run backtests with TABaseStrategy.
REM Usage: ft-backtest.bat [extra args...]
REM Example: ft-backtest.bat --timerange 20260101-

setlocal
set "PROJECT_DIR=%~dp0..\.."
call "%PROJECT_DIR%\.venv\Scripts\activate.bat"

freqtrade backtesting ^
    --config "%PROJECT_DIR%\config.json" ^
    --strategy TABaseStrategy ^
    --strategy-path "%PROJECT_DIR%\user_data\strategies" ^
    %*
