@echo off
REM Run pytest for unit and integration tests.
REM Usage: ft-test.bat [pytest args...]
REM Example: ft-test.bat tests/unit/ -v

setlocal
set "PROJECT_DIR=%~dp0..\.."
call "%PROJECT_DIR%\.venv\Scripts\activate.bat"

if "%1"=="" (
    python -m pytest "%PROJECT_DIR%\tests" -v
) else (
    python -m pytest %*
)
