@echo off
REM ##############################################################
REM  DISABLED 2026-06-05 — PF-AfterHoursResearch task set /DISABLE to cut
REM  Claude token usage (user pivot). This .bat calls `claude -p` DIRECTLY
REM  (opus, full-context) and bypasses claude_gate — the ONLY guard is the
REM  disabled Task Scheduler entry. DO NOT re-enable (schtasks /Change
REM  /ENABLE or re-running install-research-task.ps1) without user sign-off.
REM ##############################################################
REM ============================================================
REM  PF-AfterHoursResearch — Daily research agent after market close
REM  Scheduled: 22:30 CET daily via Task Scheduler
REM  Prompt: docs\after-hours-research-prompt.md
REM  Progress: data\after-hours-research-progress.json
REM  Logs: data\after-hours-research-log.jsonl
REM  Output: data\after-hours-research-out.txt
REM
REM  Produces:
REM    data\daily_research_review.json    — today's trade review
REM    data\daily_research_macro.json     — macro/news research
REM    data\daily_research_quant.json     — quant strategy research
REM    data\daily_research_signal_audit.json — signal performance audit
REM    data\morning_briefing.json         — synthesized morning briefing
REM ============================================================

set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=
cd /d Q:\finance-analyzer

REM --- Timestamp ---
for /f "tokens=*" %%T in ('powershell -NoProfile -Command "[datetime]::UtcNow.ToString('o')"') do set TS_START=%%T

REM --- Reset progress file ---
powershell -NoProfile -Command ^
  "Set-Content -Path 'Q:\finance-analyzer\data\after-hours-research-progress.json' -Value ('{\"session_started\":\"%TS_START%\",\"current_phase\":\"starting\",\"status\":\"starting\",\"phases_completed\":[],\"notes\":\"Session launched, waiting for Claude to begin\"}')"

REM --- Log: session starting ---
powershell -NoProfile -Command ^
  "Add-Content -Path 'Q:\finance-analyzer\data\after-hours-research-log.jsonl' -Value ('{\"ts\":\"%TS_START%\",\"event\":\"started\"}')"

echo [%TS_START%] After-Hours Research session starting...

REM --- Run claude with the research prompt ---
REM 2026-06-11 (audit batch 3, prompt-injection hardening): this agent fetches
REM untrusted web content. Unlike prophecy-daily.bat it CANNOT drop Bash/Edit —
REM the prompt's job is writing code + running tests — so the restriction here
REM is partial: --strict-mcp-config removes MCP servers (avanza-mcp order
REM placement etc.) and the explicit --allowedTools pins the toolset instead
REM of inheriting whatever the project settings allow. Residual risk: an
REM injected page can still reach Bash. Real fix = split research phase
REM (web, no Bash) from build phase (Bash, no web) — out of batch-3 scope.
REM Task is currently DISABLED (2026-06-05 token freeze); revisit before re-enable.
type Q:\finance-analyzer\docs\after-hours-research-prompt.md | claude -p --verbose --model claude-opus-4-6 --strict-mcp-config --allowedTools "Read,Glob,Grep,Edit,Write,Bash,WebSearch,WebFetch,Task,TodoWrite" > Q:\finance-analyzer\data\after-hours-research-out.txt 2>&1
set EXIT_CODE=%ERRORLEVEL%

REM --- Timestamp end ---
for /f "tokens=*" %%T in ('powershell -NoProfile -Command "[datetime]::UtcNow.ToString('o')"') do set TS_END=%%T

REM --- Calculate duration ---
for /f "tokens=*" %%D in ('powershell -NoProfile -Command ^
  "$s=[datetime]::Parse('%TS_START%'); $e=[datetime]::Parse('%TS_END%'); [int]($e - $s).TotalSeconds"') do set DURATION=%%D

REM --- Read last phase from progress file ---
set LAST_PHASE=unknown
if exist Q:\finance-analyzer\data\after-hours-research-progress.json (
    for /f "tokens=*" %%P in ('powershell -NoProfile -Command ^
      "$j = Get-Content 'Q:\finance-analyzer\data\after-hours-research-progress.json' -Raw | ConvertFrom-Json; $j.current_phase"') do set LAST_PHASE=%%P
)

REM --- Read phases completed count ---
set PHASES_DONE=0
if exist Q:\finance-analyzer\data\after-hours-research-progress.json (
    for /f "tokens=*" %%N in ('powershell -NoProfile -Command ^
      "$j = Get-Content 'Q:\finance-analyzer\data\after-hours-research-progress.json' -Raw | ConvertFrom-Json; $j.phases_completed.Count"') do set PHASES_DONE=%%N
)

REM --- Log result ---
if not %EXIT_CODE%==0 goto :log_failed

powershell -NoProfile -Command ^
  "Add-Content -Path 'Q:\finance-analyzer\data\after-hours-research-log.jsonl' -Value ('{\"ts\":\"%TS_END%\",\"event\":\"success\",\"exit_code\":0,\"duration_s\":%DURATION%,\"last_phase\":\"%LAST_PHASE%\",\"phases_done\":%PHASES_DONE%}')"
echo [%TS_END%] SUCCESS in %DURATION%s ^| last_phase=%LAST_PHASE% ^| phases_done=%PHASES_DONE%
goto :done

:log_failed
powershell -NoProfile -Command ^
  "Add-Content -Path 'Q:\finance-analyzer\data\after-hours-research-log.jsonl' -Value ('{\"ts\":\"%TS_END%\",\"event\":\"failed\",\"exit_code\":%EXIT_CODE%,\"duration_s\":%DURATION%,\"last_phase\":\"%LAST_PHASE%\",\"phases_done\":%PHASES_DONE%}')"
echo [%TS_END%] FAILED exit=%EXIT_CODE% in %DURATION%s ^| last_phase=%LAST_PHASE% ^| phases_done=%PHASES_DONE%

:done
exit /b %EXIT_CODE%
