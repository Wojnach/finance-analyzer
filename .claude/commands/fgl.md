Before doing ANY work, read and internalize these two documents:

1. **Read `docs/GUIDELINES.md`** — the execution protocol (explore → plan → implement → verify → ship)
2. **Read `docs/after-hours-research-prompt.md`** — the full execution protocol with phases

## Execution Rules (from these docs)

1. **EXPLORE FIRST.** Read all relevant files. Use extended thinking. Do NOT write code yet.
2. **PLAN BEFORE ACTING.** Write a plan to `docs/PLAN.md`. Commit it before implementing.
3. **PREMORTEM THE PLAN.** Before touching code, imagine it's 1 week post-merge and this change caused a production incident (silent loop crash, bad trade, data corruption, accuracy regression, auth outage, etc.). Write a `## Premortem` section at the bottom of `docs/PLAN.md` enumerating ≥5 distinct failure narratives — each one a concrete causal chain (`X happened because Y assumption was wrong, which manifested as Z`), not vague "could break". Cover at least: (a) hidden coupling to another loop/signal/file, (b) atomic I/O / concurrency race, (c) Layer 2 subprocess behavior change, (d) silent-failure mode (exit 0 but wrong), (e) test passes but prod differs. For each, add a one-line mitigation or detection hook (assert, log line, test, monitor). If a failure has no cheap mitigation, mark `ACCEPT` with reasoning. Commit the premortem with the plan. This step is non-skippable — its purpose is to surface the failure modes you'd otherwise discover in the postmortem.
4. **IMPLEMENT IN BATCHES.** 5-10 files max per batch. Commit after each batch.
5. **USE WORKTREES.** Isolate work: `git worktree add <path> -b <branch>`. Never work directly on main.
6. **CLAUDE CODE ADVERSARIAL REVIEW.** After implementation is complete, spawn a fresh Claude Code subagent (`caveman:cavecrew-reviewer` for tight diffs, `pr-review-toolkit:code-reviewer` for full PRs) on the worktree branch. Fix any P1/P2 findings. Document P3 case-by-case. (Switched from Codex 2026-05-17: Codex hits usage limits unpredictably and there's no reliable way to tail/detect stall — timeout timers give false negatives on long-but-fine runs. Claude subagents are observable + within Max sub.)
7. **TEST EVERYTHING.** After review + fixes: `.venv/Scripts/python.exe -m pytest tests/ -n auto`.
8. **COMMIT, MERGE, PUSH.** Use Windows git for push: `cmd.exe /c "cd /d Q:\finance-analyzer && git push"`
9. **CLEAN UP WORKTREES.** `git worktree remove <path> && git branch -d <branch>` after merging.
10. **DO NOT ASK FOR APPROVAL.** Make your best call, document reasoning in commits.
11. **SPEND YOUR ENTIRE CONTEXT.** Do not stop early. This is a deep work session.

## Key Project Rules (from CLAUDE.md)

- **NEVER commit config.json** — symlink to external file with API keys.
- **Search before writing code** — grep for existing functionality first.
- **Atomic I/O only** — use `file_utils.atomic_write_json()`, never raw `json.loads(open(...).read())`.
- **Use subagents** when parallel work helps (test runner + implementer, etc.).
- **Save progress** to `docs/SESSION_PROGRESS.md` every few batches.

Now proceed with the user's task following this protocol.
