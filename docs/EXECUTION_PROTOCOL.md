# Execution Protocol

## Phase 1: Understand

- Read all relevant files before making any changes. Use extended thinking.
- Check for existing docs (DESIGN.md, ARCHITECTURE.md, docs/). Note any drift from reality.
- Do NOT write code during this phase.

## Phase 2: Plan

- Write your plan to `docs/PLAN.md`: what you'll change, why, what could break, execution order in batches of 5-10 files.
- Prioritize: critical bugs > architecture > features > polish.
- Commit the plan before proceeding.

## Phase 3: Implement

- Work in small batches. Only touch files in the current batch.
- If tests are missing for code you're changing, write failing tests FIRST and commit them separately before implementing.
- After each batch: run tests, verify nothing broke, commit with conventional commits. If the test suite is slow, delegate test runs to a background subagent so implementation can continue in parallel — but always verify test results before committing the next batch.
- After every 2-3 batches: save progress to `docs/SESSION_PROGRESS.md`.

## Phase 4: Verify & Ship

- Run full test suite + linters. Everything must pass.
- Update any docs your changes affect.
- Merge, push, restart if applicable. Clean up worktree.

---

## Rules

- Use a git worktree or branch to isolate from live environment.
- Use subagents for parallel work: exploration, test writing, code review. Use a dedicated background subagent for long-running test suites so the main thread isn't blocked. Use agent teams when multiple independent modules need simultaneous changes. Don't force parallelism — only when it helps.
- Prefer fresh context over degraded context. If context is getting long, save state to `docs/SESSION_PROGRESS.md`, commit, then continue from there after refresh.
- Keep changes modular. Every change should make the system easier to extend, swap, or remove parts.
- Keep changes reversible. Prefer additive over destructive. If too risky, skip with `// TODO: MANUAL REVIEW`.
- Do not ask for approval. Make your best call and document reasoning in commits.
- Spend your entire output context on this. Do not stop early.
- Never modify config files containing API keys, credentials, or trading parameters without a `// TODO: MANUAL REVIEW` comment.
- Send a Telegram summary when significant milestones are reached or the session is complete, if Telegram is configured.
- Parallelize across all available CPU cores and threads whenever possible. Use `pytest -n auto` (pytest-xdist) for test suites, multiprocessing/concurrent.futures for batch operations, and parallel subagents for independent workstreams. Maximize hardware utilization — don't leave cores idle when work can be distributed.
