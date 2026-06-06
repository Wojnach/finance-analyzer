@echo off
REM ##############################################################
REM  PF-Prophecy — Daily AI price-prediction agent
REM  Scheduled: 10:00 Europe/Stockholm daily via Task Scheduler
REM
REM  FREEZE-SAFE: this .bat calls `claude -p` DIRECTLY and BYPASSES
REM  claude_gate (like the other research .bat files). Therefore its real
REM  kill switch is the SYSTEM_DISABLED sentinel checked below — NOT the
REM  scheduled-task state alone. While data\prophecy_runs\SYSTEM_DISABLED
REM  exists, this exits 0 BEFORE spending any tokens. Ships present (frozen).
REM
REM  Prompt:   docs\prophecy-prompt.md
REM  Pipeline: prep.py (0 tok) -> claude -p (spend) -> publish/outcomes/cost (0 tok)
REM  Outputs:  data\prophecy_runs\{context,raw,run}_<date>.json,
REM            prediction_journal.jsonl, latest.json, accuracy.json, cost_log.jsonl
REM ##############################################################

set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=
cd /d Q:\finance-analyzer

set PY=Q:\finance-analyzer\.venv\Scripts\python.exe
set PDIR=Q:\finance-analyzer\data\prophecy_runs

REM --- guard 1: correct cwd / configured (premortem #6) ---
if not exist Q:\finance-analyzer\config.json (
    echo [prophecy] config.json missing - wrong cwd or unconfigured, aborting
    exit /b 3
)

REM --- guard 2: system freeze sentinel (premortem #3) ---
if exist %PDIR%\SYSTEM_DISABLED (
    echo [prophecy] SYSTEM_DISABLED present - frozen, exiting 0 with zero token spend
    exit /b 0
)

REM --- date (UTC) + start timestamp ---
for /f "tokens=*" %%T in ('powershell -NoProfile -Command "[datetime]::UtcNow.ToString('yyyy-MM-dd')"') do set PDATE=%%T
for /f "tokens=*" %%T in ('powershell -NoProfile -Command "[datetime]::UtcNow.ToString('o')"') do set TS_START=%%T

REM --- reset progress + log start ---
powershell -NoProfile -Command ^
  "Set-Content -Path '%PDIR%\prophecy-progress.json' -Value ('{\"session_started\":\"%TS_START%\",\"date\":\"%PDATE%\",\"current_phase\":\"prep\",\"status\":\"running\"}')"
powershell -NoProfile -Command ^
  "Add-Content -Path '%PDIR%\prophecy-log.jsonl' -Value ('{\"ts\":\"%TS_START%\",\"event\":\"started\",\"date\":\"%PDATE%\"}')"
echo [%TS_START%] Prophecy starting for %PDATE%...

REM --- step 1: prep (zero tokens) ---
"%PY%" -m prophecy.prep --date %PDATE%

REM --- step 2: claude run (the token spend). --max-turns is a runaway backstop,
REM     NOT the cost lever (control cost later by trimming instruments/depth in
REM     prophecy_config.json). --output-format json so cost.py can price it. ---
type Q:\finance-analyzer\docs\prophecy-prompt.md | claude -p --verbose --model claude-opus-4-8 --output-format json --max-turns 600 > %PDIR%\run_%PDATE%.json 2>&1
set EXIT_CODE=%ERRORLEVEL%

REM --- steps 3-5 (zero tokens). Run EVEN IF claude failed so anti-stale guard +
REM     cost/error alerts fire on a bad run instead of failing silently. ---
"%PY%" -m prophecy.publish --date %PDATE%
"%PY%" -m prophecy.outcomes
"%PY%" -m prophecy.cost --date %PDATE%

REM --- end timestamp + duration ---
for /f "tokens=*" %%T in ('powershell -NoProfile -Command "[datetime]::UtcNow.ToString('o')"') do set TS_END=%%T
for /f "tokens=*" %%D in ('powershell -NoProfile -Command ^
  "$s=[datetime]::Parse('%TS_START%'); $e=[datetime]::Parse('%TS_END%'); [int]($e - $s).TotalSeconds"') do set DURATION=%%D

if not %EXIT_CODE%==0 goto :log_failed
powershell -NoProfile -Command ^
  "Add-Content -Path '%PDIR%\prophecy-log.jsonl' -Value ('{\"ts\":\"%TS_END%\",\"event\":\"success\",\"exit_code\":0,\"duration_s\":%DURATION%,\"date\":\"%PDATE%\"}')"
echo [%TS_END%] SUCCESS in %DURATION%s
goto :done

:log_failed
powershell -NoProfile -Command ^
  "Add-Content -Path '%PDIR%\prophecy-log.jsonl' -Value ('{\"ts\":\"%TS_END%\",\"event\":\"failed\",\"exit_code\":%EXIT_CODE%,\"duration_s\":%DURATION%,\"date\":\"%PDATE%\"}')"
echo [%TS_END%] claude exit=%EXIT_CODE% in %DURATION%s (publish/cost still ran; check critical_errors)

:done
exit /b %EXIT_CODE%
