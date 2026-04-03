Before doing ANY work, read and internalize these two documents:

1. **Read `docs/GUIDELINES.md`** — the execution protocol (explore → plan → implement → verify → ship)
2. **Read `docs/after-hours-research-prompt.md`** — the full execution protocol with phases

## Execution Rules (from these docs)

1. **EXPLORE FIRST.** Read all relevant files. Use extended thinking. Do NOT write code yet.
2. **PLAN BEFORE ACTING.** Write a plan to `docs/PLAN.md`. Commit it before implementing.
3. **IMPLEMENT IN BATCHES.** 5-10 files max per batch. Commit after each batch.
4. **USE WORKTREES.** Isolate work: `git worktree add <path> -b <branch>`. Never work directly on main.
5. **CODEX ADVERSARIAL REVIEW.** After implementation is complete, run `/codex:adversarial-review --wait --scope branch --effort xhigh` on the worktree branch. Fix any valid findings. If findings are minor or false positives, document why and proceed.
6. **TEST EVERYTHING.** After Codex review + fixes: `.venv/Scripts/python.exe -m pytest tests/ -n auto`.
7. **COMMIT, MERGE, PUSH.** Use Windows git for push: `cmd.exe /c "cd /d Q:\finance-analyzer && git push"`
8. **CLEAN UP WORKTREES.** `git worktree remove <path> && git branch -d <branch>` after merging.
9. **DO NOT ASK FOR APPROVAL.** Make your best call, document reasoning in commits.
10. **SPEND YOUR ENTIRE CONTEXT.** Do not stop early. This is a deep work session.

## Key Project Rules (from CLAUDE.md)

- **NEVER commit config.json** — symlink to external file with API keys.
- **Search before writing code** — grep for existing functionality first.
- **Atomic I/O only** — use `file_utils.atomic_write_json()`, never raw `json.loads(open(...).read())`.
- **Use subagents** when parallel work helps (test runner + implementer, etc.).
- **Save progress** to `docs/SESSION_PROGRESS.md` every few batches.

Now proceed with the user's task following this protocol.
