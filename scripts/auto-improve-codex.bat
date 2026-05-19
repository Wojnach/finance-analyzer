@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM Codex auto-improve runner

cd /d Q:\finance-analyzer

set PROMPT_FILE=Q:\finance-analyzer\docs\auto-improve-prompt-codex.md
set PROGRESS_FILE=Q:\finance-analyzer\data\auto-improve-codex-progress.json
set LOG_FILE=Q:\finance-analyzer\data\auto-improve-codex-log.jsonl
set OUT_FILE=Q:\finance-analyzer\data\auto-improve-codex-out.txt
set PROBE_FILE=Q:\finance-analyzer\data\auto-improve-codex-model-probe.txt

if not exist "%PROMPT_FILE%" goto :missing_prompt

for /f "tokens=*" %%T in ('powershell -NoProfile -Command "[datetime]::UtcNow.ToString('o')"') do set TS_START=%%T

if not defined CODEX_MODEL call :pick_model
if defined CODEX_MODEL goto :have_model

echo [%TS_START%] FAILED no usable Codex model.
powershell -NoProfile -Command "Set-Content -Path '%PROGRESS_FILE%' -Value ('{\"session_started\":\"%TS_START%\",\"current_phase\":\"startup\",\"status\":\"failed\",\"phases_completed\":[],\"notes\":\"No usable Codex model found.\"}')"
powershell -NoProfile -Command "Add-Content -Path '%LOG_FILE%' -Value ('{\"ts\":\"%TS_START%\",\"event\":\"failed\",\"exit_code\":99,\"reason\":\"no_model_available\"}')"
set EXIT_CODE=99
goto :done

:have_model
powershell -NoProfile -Command "Set-Content -Path '%PROGRESS_FILE%' -Value ('{\"session_started\":\"%TS_START%\",\"current_phase\":\"starting\",\"status\":\"starting\",\"phases_completed\":[],\"notes\":\"Session launched with %CODEX_MODEL%, waiting for Codex to begin\",\"model\":\"%CODEX_MODEL%\"}')"
powershell -NoProfile -Command "Add-Content -Path '%LOG_FILE%' -Value ('{\"ts\":\"%TS_START%\",\"event\":\"started\",\"model\":\"%CODEX_MODEL%\"}')"

echo [%TS_START%] AutoImprove Codex starting model=%CODEX_MODEL%
type "%PROMPT_FILE%" | codex exec -m %CODEX_MODEL% --dangerously-bypass-approvals-and-sandbox --color never - > "%OUT_FILE%" 2>&1
set EXIT_CODE=%ERRORLEVEL%

for /f "tokens=*" %%T in ('powershell -NoProfile -Command "[datetime]::UtcNow.ToString('o')"') do set TS_END=%%T
for /f "tokens=*" %%D in ('powershell -NoProfile -Command "$s=[datetime]::Parse('%TS_START%'); $e=[datetime]::Parse('%TS_END%'); [int]($e - $s).TotalSeconds"') do set DURATION=%%D

set LAST_PHASE=unknown
if exist "%PROGRESS_FILE%" for /f "tokens=*" %%P in ('powershell -NoProfile -Command "$j = Get-Content '%PROGRESS_FILE%' -Raw | ConvertFrom-Json; $j.current_phase"') do set LAST_PHASE=%%P

set PHASES_DONE=0
if exist "%PROGRESS_FILE%" for /f "tokens=*" %%N in ('powershell -NoProfile -Command "$j = Get-Content '%PROGRESS_FILE%' -Raw | ConvertFrom-Json; $j.phases_completed.Count"') do set PHASES_DONE=%%N

set SUCCESS_COUNT=0
set FAIL_COUNT=0
if exist "%LOG_FILE%" for /f %%N in ('powershell -NoProfile -Command "$m = Select-String -Path '%LOG_FILE%' -Pattern '\"event\":\"success\"' -SimpleMatch; if($m){$m.Count}else{0}"') do set SUCCESS_COUNT=%%N
if exist "%LOG_FILE%" for /f %%N in ('powershell -NoProfile -Command "$m = Select-String -Path '%LOG_FILE%' -Pattern '\"event\":\"failed\"' -SimpleMatch; if($m){$m.Count}else{0}"') do set FAIL_COUNT=%%N

if not %EXIT_CODE%==0 goto :log_failed

set /a NEW_SUCCESS=%SUCCESS_COUNT%+1
powershell -NoProfile -Command "Add-Content -Path '%LOG_FILE%' -Value ('{\"ts\":\"%TS_END%\",\"event\":\"success\",\"model\":\"%CODEX_MODEL%\",\"exit_code\":0,\"duration_s\":%DURATION%,\"last_phase\":\"%LAST_PHASE%\",\"phases_done\":%PHASES_DONE%,\"total_success\":%NEW_SUCCESS%,\"total_failed\":%FAIL_COUNT%}')"
echo [%TS_END%] SUCCESS in %DURATION%s model=%CODEX_MODEL% last_phase=%LAST_PHASE% phases_done=%PHASES_DONE%
goto :done

:log_failed
set /a NEW_FAIL=%FAIL_COUNT%+1
powershell -NoProfile -Command "Add-Content -Path '%LOG_FILE%' -Value ('{\"ts\":\"%TS_END%\",\"event\":\"failed\",\"model\":\"%CODEX_MODEL%\",\"exit_code\":%EXIT_CODE%,\"duration_s\":%DURATION%,\"last_phase\":\"%LAST_PHASE%\",\"phases_done\":%PHASES_DONE%,\"total_success\":%SUCCESS_COUNT%,\"total_failed\":%NEW_FAIL%}')"
echo [%TS_END%] FAILED exit=%EXIT_CODE% in %DURATION%s model=%CODEX_MODEL% last_phase=%LAST_PHASE% phases_done=%PHASES_DONE%
goto :done

:missing_prompt
echo Missing prompt file: %PROMPT_FILE%
set EXIT_CODE=2

:done
exit /b %EXIT_CODE%

:pick_model
set CODEX_MODEL=
for %%M in (codex-xhigh codex-high gpt-5.3-codex codex) do (
  call :probe_model %%M
  if "!PROBE_OK!"=="1" (
    set CODEX_MODEL=%%M
    goto :eof
  )
)
goto :eof

:probe_model
set PROBE_OK=0
echo Reply with exactly: OK | codex exec -m %1 --dangerously-bypass-approvals-and-sandbox --color never -o "%PROBE_FILE%" - >nul 2>&1
if %ERRORLEVEL%==0 set PROBE_OK=1
exit /b 0


