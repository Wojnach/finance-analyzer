# Plan — Full Adversarial Codebase Review (2026-05-19)

## Goal

Spawn 8 parallel review subagents — one per subsystem — against the current `main`
(commit `2daa4fd9`), against an empty-baseline orphan branch so each subsystem
diff equals the whole subsystem (every file appears "added"). Each agent
returns severity-tagged findings. The main thread runs its own independent
adversarial pass focused on cross-cutting / architectural issues that
single-subsystem agents will miss, then cross-critiques and synthesizes.

## Why empty-baseline

Reviewers default to "diff mode" — they only flag changes. Pointing each
review agent at `git diff empty-baseline HEAD -- <subsystem files>` makes
the entire subsystem appear as an enormous new PR, so the agent reviews
*everything*, not just the last week of edits. This is the only way to get
a real adversarial whole-system audit out of diff-oriented agents.

## Subsystem Partition

| # | Subsystem | Reviewer agent | Scope |
|---|-----------|----------------|-------|
| 1 | signals-core | pr-review-toolkit:code-reviewer | signal_engine, registry, accuracy, weights, IC |
| 2 | orchestration | pr-review-toolkit:code-reviewer | main loop, Layer 2 dispatch, triggers, gates |
| 3 | portfolio-risk | pr-review-toolkit:code-reviewer | portfolio mgr, risk, guards, MC, kelly |
| 4 | metals-core | pr-review-toolkit:code-reviewer | metals_loop, grid_fisher, snipe, orb, exit_optimizer |
| 5 | avanza-api | caveman:cavecrew-reviewer | session, orders, control, account, order_lock |
| 6 | signals-modules | pr-review-toolkit:code-reviewer | portfolio/signals/*.py (60 modules) |
| 7 | data-external | pr-review-toolkit:code-reviewer | data_collector, sentiment, futures, onchain, news |
| 8 | infrastructure | caveman:cavecrew-reviewer | file_utils, http_retry, journal, telegram, health |

caveman:cavecrew-reviewer used for the two smallest/tightest subsystems so the
reviewer can fit the whole thing in context with one-line-per-finding output.
The broader subsystems get pr-review-toolkit:code-reviewer which handles
larger scope.

## Execution Order

1. Create worktree `Q:\finance-analyzer-reviews\2026-05-19` off `main` (commit `2daa4fd9`).
2. In worktree, create orphan branch `review-baseline-empty` with zero files.
3. Spawn all 8 review agents in **background** (parallel). Each gets:
   - Working dir = worktree path
   - Explicit file list for its subsystem
   - Empty-baseline framing (treat entire subsystem as new code)
   - Severity scale: P0 (data loss / silent failure / will lose money) → P3
   - Output format: `path:line: <severity>: <problem>. <fix>.`
4. While agents run, the main thread does an **independent adversarial pass**
   focused on cross-cutting concerns: thread safety across files, atomic-I/O
   regressions, signal-engine ↔ Layer 2 coupling, dashboard auth/leaks, env-var
   leaks, retry storms, fee model regressions. Write findings to
   `docs/adversarial_review/main_thread_pass.md`.
5. Collect each agent's output as it completes; save raw output to
   `docs/adversarial_review/agent_<n>_<subsystem>.md`.
6. Cross-critique pass: read all 9 outputs (8 agents + main thread), look for:
   - Duplicate findings (consolidate)
   - Disagreements (decide who's right; document reasoning)
   - Gaps (issues that should have been caught but weren't)
   - False positives (mark, explain dismissal)
7. Write synthesis to `docs/ADVERSARIAL_REVIEW_2026-05-19.md`.

## What This Pass Will NOT Do

- **No code fixes.** This is review-only. Findings feed the next implementation
  session. Filing complete findings beats half-finished fixes.
- **No new tests.** Same reason.
- **No live-config touch.** `config.json` stays untouched.

## Cleanup

After synthesis lands on main and is pushed:
- `git worktree remove Q:\finance-analyzer-reviews\2026-05-19`
- `git branch -D review-baseline-empty`

## Premortem (review-only pass, blast radius ≈ 0)

1. **Agent output rot** — generic "looks ok" with no findings. *Mitigation:*
   any agent producing fewer than 3 findings gets re-spawned with stricter prompt.
2. **Diff truncation** — empty-baseline diff too large for agent context.
   *Mitigation:* fall back to direct file-list review without diff.
3. **Findings are noise** — cross-critique pass filters.
4. **Worktree pollution** — cleanup step is non-skippable.
5. **Synthesis doc never lands** — incremental commits of each agent's raw
   output as it completes.

ACCEPT: no live trading impact, no signal/loop touches, fully reversible (docs only).
