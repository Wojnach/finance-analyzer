# /fgl Dual Adversarial Review — 2026-05-09

Two-agent adversarial review of the finance-analyzer codebase. Two reviewers
attack the same eight subsystems independently, then critique each other's
findings, then everything gets folded into one synthesis document.

## Subsystem partition

The codebase (~250 portfolio modules, ~16K-line metals loop, ~38 signal
modules) is partitioned into eight subsystems with mutually exclusive file
sets. Each subsystem has a worktree branch off `empty-baseline`, so
`codex review --base empty-baseline` sees the entire subsystem as a single
diff to review.

| Subsystem        | Files | Worktree                                      | Branch                                   |
|------------------|-------|-----------------------------------------------|------------------------------------------|
| signals-core     | 13    | `.worktrees/adv-signals-core`                 | `review/2026-05-08-signals-core`         |
| orchestration    | 15    | `.worktrees/adv-orchestration`                | `review/2026-05-08-orchestration`        |
| portfolio-risk   | 13    | `.worktrees/adv-portfolio-risk`               | `review/2026-05-08-portfolio-risk`       |
| metals-core      | 16    | `.worktrees/adv-metals-core`                  | `review/2026-05-08-metals-core`          |
| avanza-api       | 18    | `.worktrees/adv-avanza-api`                   | `review/2026-05-08-avanza-api`           |
| signals-modules  | 49    | `.worktrees/adv-signals-modules`              | `review/2026-05-08-signals-modules`      |
| data-external    | 14    | `.worktrees/adv-data-external`                | `review/2026-05-08-data-external`        |
| infrastructure   | 18    | `.worktrees/adv-infrastructure`               | `review/2026-05-08-infrastructure`       |

Branches were refreshed from current `main` (HEAD `aa493aec`) on
2026-05-09 before the review.

## Two-reviewer protocol

1. **Codex adversarial review** — `codex review --base empty-baseline` per
   subsystem worktree, run in background, output to
   `data/fgl-logs/codex-<sub>.txt`. Codex sees the entire subsystem diff
   against an empty baseline. `gpt-5.4` model, `xhigh` reasoning.
2. **Claude adversarial review** — eight parallel `general-purpose`
   subagents, one per subsystem, each with the file list and a strict
   adversarial prompt (silent failures, concurrency, logic, contracts,
   API hazards, risk math, leaks, gaps). Each agent writes
   `docs/fgl/claude-<sub>.md`.
3. **Cross-critique** — once both reviewers finish, a third pass critiques
   the *other* reviewer's findings:
   - Claude reviews each codex report → `docs/fgl/claude-critique-of-codex-<sub>.md`
   - Codex (executed via `codex review` with the claude report as input) →
     `docs/fgl/codex-critique-of-claude-<sub>.md`
4. **Synthesis** — single `docs/fgl/SYNTHESIS.md` consolidating findings,
   ranked by P0 → P3, with the cross-critique verdict on each finding
   (confirmed / disputed / disagreement).

## Severity scale

Both reviewers use the same scale:

- **P0** — currently shipping a wrong trade, losing real money, corrupting
  state, or a silent failure that hides a P0 from operators.
- **P1** — would lose money on the next bad day; bug exists but hasn't fired.
- **P2** — incorrect under specific conditions; will cause confusion or
  small loss but won't blow up.
- **P3** — correctness issue with no realistic blast radius; or dead code.

## Output products

```
docs/fgl/
├── PLAN.md                          this file
├── claude-<sub>.md                  ×8   claude's adversarial review
├── codex-<sub>.md                   ×8   codex's adversarial review
├── claude-critique-of-codex-<sub>.md ×8  claude vetting codex
├── codex-critique-of-claude-<sub>.md ×8  codex vetting claude
└── SYNTHESIS.md                     master prioritized findings list
```

## What this is NOT

- Not an implementation pass. No code changes are made under `/fgl`.
  The output is a prioritized findings document; fixes happen in
  follow-up branches.
- Not a substitute for `pytest tests/`. The synthesis only reports what
  the two reviewers agree (or disagree) is wrong by reading code.
- Not exhaustive. The claude reviewer sub-samples the
  `signals-modules` subsystem (38 files) because exhaustive line-level
  review of every module isn't feasible; codex reviews the full diff.
