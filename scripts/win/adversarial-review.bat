@echo off
REM PF-AdversarialReview — Daily dual adversarial review (Codex + Claude)
REM Runs claude code CLI with the full review prompt.
REM Output: data\adversarial_review_out.txt
cd /d Q:\finance-analyzer
set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=

echo [%date% %time%] Starting adversarial review... >> data\adversarial_review_out.txt 2>&1

claude -p "Follow /fgl protocol. Run a full dual adversarial review of the finance-analyzer codebase: partition into 8 subsystems (signals-core, orchestration, portfolio-risk, metals-core, avanza-api, signals-modules, data-external, infrastructure), create worktree with empty-baseline branches, run /codex:adversarial-review in background for each subsystem, write your own independent adversarial review, collect codex results, cross-critique in both directions, write synthesis doc. Commit all docs to main, push via Windows git, clean up worktrees. Do NOT ask for approval — follow /fgl rules. Spend your entire context on this." --allowedTools "Edit,Read,Bash,Write,Glob,Grep" --max-turns 80 >> data\adversarial_review_out.txt 2>&1

echo [%date% %time%] Adversarial review finished (code %ERRORLEVEL%). >> data\adversarial_review_out.txt 2>&1
