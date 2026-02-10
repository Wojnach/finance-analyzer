@echo off
REM Start paper trading (dry run).
REM Usage: ft-dry-run.bat [extra args...]

setlocal
set "PROJECT_DIR=%~dp0..\.."
call "%PROJECT_DIR%\.venv\Scripts\activate.bat"

if exist "%PROJECT_DIR%\config.json" (
    set "CONFIG=%PROJECT_DIR%\config.json"
) else (
    set "CONFIG=%PROJECT_DIR%\config.example.json"
)

freqtrade trade ^
    --config "%CONFIG%" ^
    --strategy TABaseStrategy ^
    --strategy-path "%PROJECT_DIR%\user_data\strategies" ^
    %*
