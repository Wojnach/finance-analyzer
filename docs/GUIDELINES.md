# Execution Guidelines

## HOW TO EXECUTE

1. **EXPLORE FIRST.** Read all relevant files and understand the system before doing anything. Use extended thinking. Do not write code or make changes during this step.

2. **PLAN BEFORE ACTING.** Write a plan to `docs/PLAN.md` with: what you'll do, why, what could break, and execution order. Commit it before proceeding.

3. **IMPLEMENT IN BATCHES.** Work in small batches (5-10 files max). After each batch: run tests, verify nothing broke, commit with a conventional commit message. If tests are missing for code you're changing, write them first.

4. **STAY MODULAR.** Every change should make the system easier to swap, extend, or remove parts. No tight coupling. No god files.

5. **DOCUMENT AS YOU GO.** Update any existing docs that your changes affect. Write findings and decisions into your plan doc.

6. **VERIFY & SHIP.** Run full test suite + linters. Review your git log. Merge, push, restart if needed.

## RULES

- Use a git worktree or branch to isolate work from the live environment.
- Use subagents and agent teams when parallel work makes sense.
- Save progress to `docs/SESSION_PROGRESS.md` every few batches in case of context refresh. Prefer fresh context over degraded context.
- Do not ask for approval. Make your best call, document reasoning in commits.
- If something is too risky, skip it with a `// TODO: MANUAL REVIEW` comment.
- Spend your entire output context on this. Do not stop early.
