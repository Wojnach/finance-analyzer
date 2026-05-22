# PLAN — Full Adversarial Codebase Review (FGL protocol)

**Date:** 2026-05-22 (Fri)
**Type:** Read-only adversarial review. Ships ZERO production code — only review docs.
**Trigger:** `/fgl` — "full adversarial review of finance-analyzer, partition into 8 subsystems."

## Objective

Adversarially review the entire `finance-analyzer` codebase for bugs, silent-failure
modes, money-math errors, concurrency races, and contract violations. Produce a
synthesis document with severity-ranked findings. No code is changed this session —
findings feed the next implementation session and `docs/IMPROVEMENT_BACKLOG.md`.

## Subsystem partition (8)

| # | Subsystem | Scope | Reviewer agent |
|---|-----------|-------|----------------|
| 1 | signals-core | vote aggregation, accuracy gating, weighting, IC | pr-review-toolkit:code-reviewer |
| 2 | orchestration | main loop, Layer 2 subprocess, triggers, escalation | pr-review-toolkit:code-reviewer |
| 3 | portfolio-risk | portfolio state, sizing, risk mgmt, monte carlo, exits | pr-review-toolkit:code-reviewer |
| 4 | metals-core | metals/oil/crypto loops, warrants, fishing, grid | pr-review-toolkit:code-reviewer |
| 5 | avanza-api | broker client, BankID auth, order flow, locks | caveman:cavecrew-reviewer |
| 6 | signals-modules | 64 signal plugins + LLM signal adapters | caveman:cavecrew-reviewer |
| 7 | data-external | market data fetchers, sentiment, FX, precompute | caveman:cavecrew-reviewer |
| 8 | infrastructure | atomic I/O, locks, health, notifications, dashboard | pr-review-toolkit:code-reviewer |

## Execution

1. **Worktree + empty-baseline branches.** Create one worktree (`Q:/fa-review-wt`).
   In it: orphan branch `review/empty` (empty tree) + 8 branches `review/sub-<name>`,
   each = empty baseline + that subsystem's files. `git diff review/empty review/sub-X`
   then renders the whole subsystem as a reviewable diff for diff-oriented agents.
2. **Spawn 8 background review subagents**, one per subsystem, each told its branch,
   file list, and the project-specific failure modes to hunt (atomic I/O, exit-0
   silent failure, subprocess hangs, money rounding, MINI knockout barriers).
3. **Independent adversarial pass (this agent)** — while subagents run, read the
   highest-risk files directly. Fresh-eyes findings, not anchored to subagent output.
4. **Collect** subagent results.
5. **Cross-critique** — for each subagent finding: confirm / downgrade / reject with
   reasoning. For each independent finding: check if a subagent caught it too
   (corroboration raises confidence). Flag disagreements.
6. **Synthesis doc** → `docs/reviews/2026-05-22-fgl-adversarial-review.md`:
   severity-ranked (P0/P1/P2/P3), per-subsystem, with file:line, plus a top-N
   "fix first" list and a corroboration matrix.
7. **Commit docs to main**, push via Windows git, remove worktree + branches.

## What could break / risks

- **None to production.** Output is docs only; no prod code, config, or data files
  are modified. The worktree is isolated and destroyed at the end.
- **Risk: reviewer false positives.** Mitigation — the cross-critique pass (step 5)
  is the guard; every reported finding gets independently judged before synthesis.
- **Risk: context exhaustion** before synthesis. Mitigation — subagents run in
  background (parallel), output is severity-tagged one-liners (bounded), synthesis
  is written incrementally.

## Premortem

The /fgl protocol mandates an agent-driven premortem for plans that ship code. **This
plan ships zero production code** — it adds review markdown under `docs/` and creates
then destroys a throwaway worktree. The premortem's purpose (surface prod-incident
failure chains: silent loop crash, bad trade, data corruption, accuracy regression,
auth outage) has no surface area here: no loop code, signal weight, threshold, config,
or data file is touched. The single residual risk — a stale worktree or `review/*`
branch left behind — is covered by the explicit cleanup step, verified at the end.
Formal agent-premortem skipped with that reasoning recorded here; the adversarial
review itself IS a system-wide premortem of the existing code.

## Execution order

1. PLAN.md commit (this file).
2. Worktree + 9 branches (1 empty + 8 subsystem).
3. Spawn 8 background reviewers.
4. Independent review pass (parallel with 3).
5. Collect + cross-critique.
6. Synthesis doc.
7. Commit docs, push, cleanup.
