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
REM  2026-06-11 (audit batch 3): the kill switch is now FAIL-CLOSED — if
REM  data\prophecy_runs\ itself is missing/unreadable (data cleanup, disk
REM  restore), the sentinel state is unknown, so we treat that as DISABLED,
REM  recreate the dir WITH the sentinel, write a critical, and exit non-zero.
REM  A missing dir must never mean "enabled".
REM
REM  UNFREEZE CHECKLIST (premortem hook 6 — unfreeze latency bomb): the
REM  fail-closed flip and tool restrictions below shipped while the system was
REM  frozen, i.e. with ZERO production validation. After deleting
REM  SYSTEM_DISABLED, run ONE manual smoke invocation of this .bat and verify
REM  raw_<date>.json + publish/outcomes/cost all succeed BEFORE trusting the
REM  schedule. Do not assume the scheduled run works just because this file
REM  parses.
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

REM --- date (UTC) + start timestamp (needed before guard 2 so the frozen-skip
REM     and fail-closed paths can log events) ---
for /f "tokens=*" %%T in ('powershell -NoProfile -Command "[datetime]::UtcNow.ToString('yyyy-MM-dd')"') do set PDATE=%%T
for /f "tokens=*" %%T in ('powershell -NoProfile -Command "[datetime]::UtcNow.ToString('o')"') do set TS_START=%%T

REM --- guard 2a: kill switch FAIL-CLOSED (audit batch 3, 2026-06-11) ---
REM     If the sentinel's parent dir is gone we cannot know the freeze state.
REM     Recreate it WITH the sentinel so the next run is still frozen, alert,
REM     and exit non-zero. (prophecy.alerts also recreates the dir as a side
REM     effect of its rate-limit state write — that is why the sentinel is
REM     recreated FIRST: an empty recreated dir without the sentinel would
REM     silently flip the system back on at the next scheduled run.)
if exist %PDIR%\ goto :dir_ok
echo [prophecy] %PDIR% MISSING or unreadable - kill-switch state unknown, failing CLOSED
mkdir %PDIR% 2>nul
echo recreated by prophecy-daily.bat fail-closed guard - see critical_errors.jsonl > %PDIR%\SYSTEM_DISABLED
"%PY%" -c "from prophecy.alerts import log_critical; log_critical('prophecy_killswitch', 'data/prophecy_runs/ was missing - kill-switch state unknown, failed CLOSED and recreated SYSTEM_DISABLED', caller='prophecy-daily.bat')"
exit /b 4
:dir_ok

REM --- guard 2b: system freeze sentinel (premortem #3) ---
if not exist %PDIR%\SYSTEM_DISABLED goto :not_frozen
echo [prophecy] SYSTEM_DISABLED present - frozen, exiting 0 with zero token spend
powershell -NoProfile -Command ^
  "Add-Content -Path '%PDIR%\prophecy-log.jsonl' -Value ('{\"ts\":\"%TS_START%\",\"event\":\"frozen_skip\",\"date\":\"%PDATE%\"}')"
exit /b 0
:not_frozen

REM --- reset progress + log start ---
powershell -NoProfile -Command ^
  "Set-Content -Path '%PDIR%\prophecy-progress.json' -Value ('{\"session_started\":\"%TS_START%\",\"date\":\"%PDATE%\",\"current_phase\":\"prep\",\"status\":\"running\"}')"
powershell -NoProfile -Command ^
  "Add-Content -Path '%PDIR%\prophecy-log.jsonl' -Value ('{\"ts\":\"%TS_START%\",\"event\":\"started\",\"date\":\"%PDATE%\"}')"
echo [%TS_START%] Prophecy starting for %PDATE%...

REM --- step 1: prep (zero tokens). Exit code gated (audit batch 3): a broken
REM     prep used to fall straight through to the token spend, producing a
REM     full-cost run with no context, no anti-stale guard and self-claimed
REM     spots. Now: prep failure = no claude run, critical, exit non-zero. ---
"%PY%" -m prophecy.prep --date %PDATE%
if not errorlevel 1 goto :prep_ok
echo [prophecy] prep FAILED - skipping claude invocation (no token spend)
"%PY%" -c "from prophecy.alerts import log_critical; log_critical('prophecy_prep_failed', 'prophecy.prep exited non-zero - claude run skipped, zero tokens spent', caller='prophecy-daily.bat', context={'date': '%PDATE%'})"
powershell -NoProfile -Command ^
  "Add-Content -Path '%PDIR%\prophecy-log.jsonl' -Value ('{\"ts\":\"%TS_START%\",\"event\":\"prep_failed\",\"date\":\"%PDATE%\"}')"
exit /b 5
:prep_ok

REM --- model: read from prophecy_config.json (audit batch 3 — the "model" key
REM     was dead config; the hardcode here silently mislabeled cost/accuracy
REM     attribution on any config-driven model swap). config.model() falls
REM     back to DEFAULT_MODEL when the key is absent; the bat-level fallback
REM     below only fires if the python call itself fails. ---
set PMODEL=
for /f "tokens=*" %%M in ('%PY% -c "from prophecy import config; print(config.model())"') do set PMODEL=%%M
if not "%PMODEL%"=="" goto :model_ok
echo [prophecy] WARNING: could not read model from prophecy_config.json - falling back to claude-opus-4-8
set PMODEL=claude-opus-4-8
:model_ok

REM --- step 2: claude run (the token spend). --max-turns is a runaway backstop,
REM     NOT the cost lever (control cost later by trimming instruments/depth in
REM     prophecy_config.json). --output-format json => stdout is a single JSON
REM     result object cost.py parses; stderr goes to a SEPARATE .log so verbose/
REM     diagnostic lines never corrupt the JSON deliverable (review P2).
REM
REM     SECURITY (audit batch 3, 2026-06-11 — prompt-injection hardening):
REM     this headless agent is INSTRUCTED to fetch untrusted web/forum content
REM     (r/Silverbugs, WallStreetSilver, Flashback, X/TradingView). It used to
REM     inherit the repo's interactive .claude/settings.json permissions —
REM     Bash(*), Write(*), Edit(*) — so an injection payload in any fetched
REM     page was one tool call away from arbitrary shell on the box holding
REM     the live Avanza session and config.json API keys. Layered restriction:
REM       --setting-sources user  : do NOT load the project allow-all rules
REM       --strict-mcp-config     : no MCP servers (no avanza-mcp etc.)
REM       --tools                 : hard-removes Bash/Edit from the toolset
REM       --allowedTools          : auto-approves ONLY what the job needs;
REM                                 Write is path-scoped to data/prophecy_runs/**
REM       --disallowedTools       : belt-and-braces deny on Bash/Edit
REM     Task is included because the prompt fans instruments out to
REM     sub-agents; sub-agents inherit this session's restricted toolset, but
REM     note they still read the same untrusted web content — the Write scope
REM     is what keeps an injected sub-agent contained. Verified against
REM     claude.exe 2.1.168 --help (flags: --tools, --setting-sources,
REM     --allowedTools, --disallowedTools). ---
type Q:\finance-analyzer\docs\prophecy-prompt.md | claude -p --model %PMODEL% --output-format json --max-turns 600 --setting-sources user --strict-mcp-config --tools "Read,Write,Glob,Grep,WebSearch,WebFetch,Task,TodoWrite" --allowedTools "Read,Glob,Grep,WebSearch,WebFetch,Task,TodoWrite,Write(data/prophecy_runs/**)" --disallowedTools "Bash,Edit,NotebookEdit" > %PDIR%\run_%PDATE%.json 2> %PDIR%\run_%PDATE%.log
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
