@echo off
REM ============================================================
REM  PF-AutoImprove — Daily autonomous improvement session
REM  Scheduled: 13:30 daily via Task Scheduler
REM  Logs to: data\auto-improve-log.jsonl
REM ============================================================

set CLAUDECODE=
cd /d Q:\finance-analyzer

REM --- Timestamp ---
for /f "tokens=*" %%T in ('powershell -NoProfile -Command "[datetime]::UtcNow.ToString('o')"') do set TS_START=%%T

REM --- Log: session starting ---
powershell -NoProfile -Command ^
  "Add-Content -Path 'Q:\finance-analyzer\data\auto-improve-log.jsonl' -Value ('{\"ts\":\"%TS_START%\",\"event\":\"started\"}' )"

REM --- Run claude with the prompt file ---
REM Pipe prompt via type; capture exit code
type Q:\finance-analyzer\docs\auto-improve-prompt.md | claude -p --verbose > Q:\finance-analyzer\data\auto-improve-out.txt 2>&1
set EXIT_CODE=%ERRORLEVEL%

REM --- Timestamp end ---
for /f "tokens=*" %%T in ('powershell -NoProfile -Command "[datetime]::UtcNow.ToString('o')"') do set TS_END=%%T

REM --- Calculate duration ---
for /f "tokens=*" %%D in ('powershell -NoProfile -Command ^
  "$s=[datetime]::Parse('%TS_START%'); $e=[datetime]::Parse('%TS_END%'); [int]($e - $s).TotalSeconds"') do set DURATION=%%D

REM --- Count previous runs from log ---
set SUCCESS_COUNT=0
set FAIL_COUNT=0
if exist Q:\finance-analyzer\data\auto-improve-log.jsonl (
    for /f %%N in ('powershell -NoProfile -Command ^
      "(Select-String -Path 'Q:\finance-analyzer\data\auto-improve-log.jsonl' -Pattern '\"event\":\"success\"' -SimpleMatch).Count"') do set SUCCESS_COUNT=%%N
    for /f %%N in ('powershell -NoProfile -Command ^
      "(Select-String -Path 'Q:\finance-analyzer\data\auto-improve-log.jsonl' -Pattern '\"event\":\"failed\"' -SimpleMatch).Count"') do set FAIL_COUNT=%%N
)

REM --- Log result ---
if %EXIT_CODE%==0 (
    set /a NEW_SUCCESS=%SUCCESS_COUNT%+1
    powershell -NoProfile -Command ^
      "Add-Content -Path 'Q:\finance-analyzer\data\auto-improve-log.jsonl' -Value ('{\"ts\":\"%TS_END%\",\"event\":\"success\",\"exit_code\":0,\"duration_s\":%DURATION%,\"total_success\":' + '%NEW_SUCCESS%' + ',\"total_failed\":%FAIL_COUNT%}')"
    echo [%TS_END%] SUCCESS in %DURATION%s (total: %NEW_SUCCESS% ok, %FAIL_COUNT% failed)
) else (
    set /a NEW_FAIL=%FAIL_COUNT%+1
    powershell -NoProfile -Command ^
      "Add-Content -Path 'Q:\finance-analyzer\data\auto-improve-log.jsonl' -Value ('{\"ts\":\"%TS_END%\",\"event\":\"failed\",\"exit_code\":%EXIT_CODE%,\"duration_s\":%DURATION%,\"total_success\":%SUCCESS_COUNT%,\"total_failed\":' + '%NEW_FAIL%' + '}')"
    echo [%TS_END%] FAILED exit=%EXIT_CODE% in %DURATION%s (total: %SUCCESS_COUNT% ok, %NEW_FAIL% failed)
)

exit /b %EXIT_CODE%
