@echo off
REM Download historical OHLCV data for backtesting.
REM Usage: ft-download-data.bat [days] [timeframes...]
REM Example: ft-download-data.bat 730 5m 1h 4h 1d

setlocal enabledelayedexpansion
set "PROJECT_DIR=%~dp0..\.."
call "%PROJECT_DIR%\.venv\Scripts\activate.bat"

set "DAYS=%1"
if "%DAYS%"=="" set "DAYS=30"
shift

set "HAS_TF=0"
:loop
if "%1"=="" goto :done
set "HAS_TF=1"
echo ==^> Downloading %DAYS%d of %1 data...
freqtrade download-data ^
    --config "%PROJECT_DIR%\config.json" ^
    --timeframe %1 ^
    --days %DAYS%
shift
goto :loop

:done
if "%HAS_TF%"=="0" (
    echo ==^> Downloading %DAYS%d of 5m data...
    freqtrade download-data ^
        --config "%PROJECT_DIR%\config.json" ^
        --timeframe 5m ^
        --days %DAYS%
)
