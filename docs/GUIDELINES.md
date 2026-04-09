ultrathink

# Execution Guidelines

## HOW TO EXECUTE

1. **EXPLORE FIRST.** Read all relevant files and understand the system before doing anything. Use extended thinking. Do not write code or make changes during this step.

2. **PLAN BEFORE ACTING.** Write a plan to `docs/PLAN.md` with: what you'll do, why, what could break, and execution order. Commit it before proceeding.

3. **IMPLEMENT IN BATCHES.** Work in small batches (5-10 files max). After each batch: run tests, verify nothing broke, commit with a conventional commit message. If tests are missing for code you're changing, write them first.

4. **STAY MODULAR.** Every change should make the system easier to swap, extend, or remove parts. No tight coupling. No god files.

5. **DOCUMENT AS YOU GO.** Update any existing docs that your changes affect. Write findings and decisions into your plan doc.

6. **ADVERSARIAL REVIEW (codex).** After implementation is complete, commit a SHA and run `codex review --commit <SHA>`. Address every P1/P2 finding; decide P3 case-by-case and document reasoning in the follow-up commit. If a finding reveals a half-wired feature (e.g. SHORT entry added but exits still assume LONG), **disable the feature at the gate with a TODO** — never ship half-complete functionality.

7. **TEST.** Run the full suite in parallel: `.venv/Scripts/python.exe -m pytest tests/ -n auto`. For focused work, run only the tests touching your files. If a test fails, run it in isolation (`pytest file::Class::test_name`) first — if it passes alone it's pre-existing state isolation, skip per the known-failure list in `docs/TESTING.md`. Never push red.

8. **MERGE, COMMIT, PUSH.** Review your git log. Merge into main, commit any fix adjustments from steps 6–7, and push via Windows git: `cmd.exe /c "cd /d Q:\finance-analyzer && git push"` (Windows git has the SSH keys, WSL git does not). This is MANDATORY — work that isn't merged and pushed didn't happen.

9. **RESTART LOOPS.** If your change touches code loaded by `data/metals_loop.py` or `portfolio/main.py --loop`, restart them so the new code takes effect: kill orphaned python processes first (the singleton lock is held by the running process, so a bare schtasks restart will hit exit code 11 and the bat wrapper will exit), then `cmd.exe /c "schtasks /run /tn PF-MetalsLoop"` and `cmd.exe /c "schtasks /run /tn PF-DataLoop"`. Tail `data/metals_loop_out.txt` and verify the new `SwingTrader init:` / `#1 Baseline established` lines appear with the expected catalog size.

## TESTING

- **Run tests in parallel when possible.** Use `pytest -n auto` (pytest-xdist) to distribute tests across all CPU cores. This is significantly faster for the 2600+ test suite.
- For targeted test runs (single file or module), sequential is fine: `pytest tests/test_foo.py -v`.
- Run tests in the background (`run_in_background`) when you have other work to do in parallel.

## RULES

- Use a git worktree or branch to isolate work from the live environment. **Clean up worktrees after merging** — delete the worktree directory and remove the branch: `git worktree remove <path> && git branch -d <branch>`. Leftover worktrees waste disk and cause confusion.
- Use subagents and agent teams when parallel work makes sense.
- Save progress to `docs/SESSION_PROGRESS.md` every few batches in case of context refresh. Prefer fresh context over degraded context.
- Do not ask for approval. Make your best call, document reasoning in commits.
- If something is too risky, skip it with a `// TODO: MANUAL REVIEW` comment.
- Spend your entire output context on this. Do not stop early.
- **ALWAYS commit, merge into main, and push when done.** Unpushed work is lost work. Use Windows git for push: `cmd.exe /c "cd /d Q:\finance-analyzer && git push"`
