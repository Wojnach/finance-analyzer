# /fgl Dual Adversarial Review — Output Index

This folder holds the output of `/fgl 2026-05-09`, a dual-reviewer
adversarial pass over the finance-analyzer codebase.

## How the review was run

Two independent adversarial reviewers were aimed at the same eight
subsystem partitions of the codebase. Both used the same severity scale,
the same focus areas, and the same instruction to report only bugs
(no praise, no style nits). After both finished, each reviewer was given
the *other* reviewer's report and asked to vet it for confirmed /
disputed / overlooked findings. A single synthesis document then folds
both directions.

```
                 ┌─ codex (gpt-5.4 xhigh, codex review --base empty-baseline)
8 subsystems ────┤
                 └─ claude (Opus 4.7 1M, parallel general-purpose subagents)
                                 │
                                 ▼
                 cross-critique in both directions
                                 │
                                 ▼
                          docs/fgl/SYNTHESIS.md
```

The eight subsystems live on review branches off `empty-baseline` in
`.worktrees/adv-<sub>/`. See `PARTITION.md` for the file lists and
`PLAN.md` for the protocol.

## File layout

```
docs/fgl/
├── README.md                            this index
├── PLAN.md                              protocol + severity scale
├── PARTITION.md                         file-by-subsystem map
├── claude-<sub>.md                      ×8  claude's adversarial review
├── codex-<sub>.md                       ×8  codex's adversarial review
├── claude-critique-of-codex-<sub>.md    ×8  claude vetting codex
├── codex-critique-of-claude-<sub>.md    ×8  codex vetting claude
└── SYNTHESIS.md                         master prioritized findings
```

## Subsystems

| Subsystem        | Files | Owner module                              |
|------------------|------:|-------------------------------------------|
| signals-core     | 13    | `portfolio/signal_engine.py`              |
| orchestration    | 15    | `portfolio/main.py`, `agent_invocation.py`|
| portfolio-risk   | 13    | `portfolio/portfolio_mgr.py`              |
| metals-core      | 16    | `data/metals_loop.py`                     |
| avanza-api       | 18    | `portfolio/avanza/`                       |
| signals-modules  | 49    | `portfolio/signals/*.py`                  |
| data-external    | 14    | `portfolio/data_collector.py`             |
| infrastructure   | 18    | `portfolio/file_utils.py`, `dashboard/`   |

## Severity scale (both reviewers, identical)

- **P0** — currently shipping a wrong trade, losing real money,
  corrupting state, or hiding a P0 from operators (silent failure).
- **P1** — would lose money on the next bad day; bug present but
  hasn't fired.
- **P2** — incorrect under specific conditions; small loss or
  confusion only.
- **P3** — correctness issue with no realistic blast radius, or dead code.

## Reading order

1. `SYNTHESIS.md` — master list, both reviewers' findings reconciled.
2. `claude-<sub>.md` and `codex-<sub>.md` for any subsystem you
   want to dig into.
3. `*-critique-of-*` to see where the two reviewers disagreed and
   which side has the stronger argument.

## What `/fgl` does NOT do

- It does NOT modify source code. Fixes are scheduled into a follow-up
  branch (typically tracked in `docs/IMPROVEMENT_BACKLOG.md`).
- It does NOT run tests. The synthesis only reports what reviewers
  *say* by reading code.
- It does NOT enumerate every signal module — `signals-modules`
  (49 files) is sub-sampled by claude; codex sees the full diff.
