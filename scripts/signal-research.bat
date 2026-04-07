@echo off
REM ============================================================
REM  PF-SignalResearch — Daily AI signal discovery & implementation
REM  Scheduled: 18:30 CET daily via Task Scheduler
REM  Prompt: docs\signal-research-prompt.md
REM  Progress: data\signal-research-progress.json
REM  Logs: data\signal-research-log.jsonl
REM  Output: data\signal_research_out.txt
REM
REM  Produces:
REM    data\signal_research_papers.json    — academic paper findings
REM    data\signal_research_web.json       — web/blog research findings
REM    data\signal_research_ranked.json    — scored & ranked candidates
REM    data\signal_research_summary.json   — context-reset summary
REM    data\signal_research_backlog.jsonl  — append-only candidate backlog
REM ============================================================

set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=
cd /d Q:\finance-analyzer

REM --- Timestamp ---
for /f "tokens=*" %%T in ('powershell -NoProfile -Command "[datetime]::UtcNow.ToString('o')"') do set TS_START=%%T

REM --- Reset progress file ---
powershell -NoProfile -Command ^
  "Set-Content -Path 'Q:\finance-analyzer\data\signal-research-progress.json' -Value ('{\"session_started\":\"%TS_START%\",\"current_phase\":\"starting\",\"status\":\"starting\",\"phases_completed\":[],\"notes\":\"Session launched, waiting for Claude to begin\"}')"

REM --- Log: session starting ---
powershell -NoProfile -Command ^
  "Add-Content -Path 'Q:\finance-analyzer\data\signal-research-log.jsonl' -Value ('{\"ts\":\"%TS_START%\",\"event\":\"started\"}')"

echo [%TS_START%] Signal Research session starting...

REM --- Run claude with the research prompt ---
type Q:\finance-analyzer\docs\signal-research-prompt.md | claude -p --verbose --model claude-opus-4-6 > Q:\finance-analyzer\data\signal_research_out.txt 2>&1
set EXIT_CODE=%ERRORLEVEL%

REM --- Timestamp end ---
for /f "tokens=*" %%T in ('powershell -NoProfile -Command "[datetime]::UtcNow.ToString('o')"') do set TS_END=%%T

REM --- Calculate duration ---
for /f "tokens=*" %%D in ('powershell -NoProfile -Command ^
  "$s=[datetime]::Parse('%TS_START%'); $e=[datetime]::Parse('%TS_END%'); [int]($e - $s).TotalSeconds"') do set DURATION=%%D

REM --- Read last phase from progress file ---
set LAST_PHASE=unknown
if exist Q:\finance-analyzer\data\signal-research-progress.json (
    for /f "tokens=*" %%P in ('powershell -NoProfile -Command ^
      "$j = Get-Content 'Q:\finance-analyzer\data\signal-research-progress.json' -Raw | ConvertFrom-Json; $j.current_phase"') do set LAST_PHASE=%%P
)

REM --- Read phases completed count ---
set PHASES_DONE=0
if exist Q:\finance-analyzer\data\signal-research-progress.json (
    for /f "tokens=*" %%N in ('powershell -NoProfile -Command ^
      "$j = Get-Content 'Q:\finance-analyzer\data\signal-research-progress.json' -Raw | ConvertFrom-Json; $j.phases_completed.Count"') do set PHASES_DONE=%%N
)

REM --- Count previous runs from log ---
set SUCCESS_COUNT=0
set FAIL_COUNT=0
if exist Q:\finance-analyzer\data\signal-research-log.jsonl (
    for /f %%N in ('powershell -NoProfile -Command ^
      "$m = Select-String -Path 'Q:\finance-analyzer\data\signal-research-log.jsonl' -Pattern '\"event\":\"success\"' -SimpleMatch; if($m){$m.Count}else{0}"') do set SUCCESS_COUNT=%%N
    for /f %%N in ('powershell -NoProfile -Command ^
      "$m = Select-String -Path 'Q:\finance-analyzer\data\signal-research-log.jsonl' -Pattern '\"event\":\"failed\"' -SimpleMatch; if($m){$m.Count}else{0}"') do set FAIL_COUNT=%%N
)

REM --- Log result with phase info ---
if not %EXIT_CODE%==0 goto :log_failed

set /a NEW_SUCCESS=%SUCCESS_COUNT%+1
powershell -NoProfile -Command ^
  "Add-Content -Path 'Q:\finance-analyzer\data\signal-research-log.jsonl' -Value ('{\"ts\":\"%TS_END%\",\"event\":\"success\",\"exit_code\":0,\"duration_s\":%DURATION%,\"last_phase\":\"%LAST_PHASE%\",\"phases_done\":%PHASES_DONE%,\"total_success\":%NEW_SUCCESS%,\"total_failed\":%FAIL_COUNT%}')"
echo [%TS_END%] SUCCESS in %DURATION%s ^| last_phase=%LAST_PHASE% ^| phases_done=%PHASES_DONE% ^| (total: %NEW_SUCCESS% ok, %FAIL_COUNT% failed)
goto :done

:log_failed
set /a NEW_FAIL=%FAIL_COUNT%+1
powershell -NoProfile -Command ^
  "Add-Content -Path 'Q:\finance-analyzer\data\signal-research-log.jsonl' -Value ('{\"ts\":\"%TS_END%\",\"event\":\"failed\",\"exit_code\":%EXIT_CODE%,\"duration_s\":%DURATION%,\"last_phase\":\"%LAST_PHASE%\",\"phases_done\":%PHASES_DONE%,\"total_success\":%SUCCESS_COUNT%,\"total_failed\":%NEW_FAIL%}')"
echo [%TS_END%] FAILED exit=%EXIT_CODE% in %DURATION%s ^| last_phase=%LAST_PHASE% ^| phases_done=%PHASES_DONE% ^| (total: %SUCCESS_COUNT% ok, %NEW_FAIL% failed)

:done
exit /b %EXIT_CODE%
