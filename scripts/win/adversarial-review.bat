@echo off
REM ##############################################################
REM  DISABLED 2026-06-05 — PF-AdversarialReview task set /DISABLE to cut
REM  Claude token usage (user pivot to reduce spend). This .bat calls
REM  `claude -p` DIRECTLY and bypasses claude_gate, so the ONLY guard is the
REM  disabled Task Scheduler entry. DO NOT re-enable (schtasks /Change
REM  /ENABLE, or re-running install-adversarial-review-task.ps1) without
REM  explicit user sign-off — this is one of the biggest single-session burns.
REM ##############################################################
REM PF-AdversarialReview — Daily adversarial review (Claude Code subagents only)
REM Runs claude code CLI with the full review prompt.
REM Output: data\adversarial_review_out.txt
REM Switched from dual Codex+Claude to Claude-only 2026-05-17: Codex usage
REM limits hit unpredictably and there's no reliable stall detection on a
REM background subprocess. Run codex manually if you want a second pass.
cd /d Q:\finance-analyzer
set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=

echo [%date% %time%] Starting adversarial review... >> data\adversarial_review_out.txt 2>&1

claude -p "Follow /fgl protocol. Run a full adversarial review of the finance-analyzer codebase: partition into 8 subsystems (signals-core, orchestration, portfolio-risk, metals-core, avanza-api, signals-modules, data-external, infrastructure), create worktree with empty-baseline branches, spawn a fresh Claude Code review subagent (caveman:cavecrew-reviewer for tight diffs, pr-review-toolkit:code-reviewer for broader scope) in background for each subsystem, write your own independent adversarial review pass, collect subagent results, cross-critique findings, write synthesis doc. Commit all docs to main, push via Windows git, clean up worktrees. Do NOT ask for approval — follow /fgl rules. Spend your entire context on this." --allowedTools "Edit,Read,Bash,Write,Glob,Grep" --max-turns 80 >> data\adversarial_review_out.txt 2>&1

echo [%date% %time%] Adversarial review finished (code %ERRORLEVEL%). >> data\adversarial_review_out.txt 2>&1
