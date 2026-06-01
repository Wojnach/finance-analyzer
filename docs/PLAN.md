# PLAN — Full Adversarial Codebase Review (2026-06-01)

> Supersedes the prior `fix/state-resilience-20260601` plan (merged today as
> `57de1814` / `c5ba4ae0` / `e0617a90`). **That fix recurred:** `portfolio_state_corrupt`
> fired again 15:11–15:19 today with "no backup recovered". The quarantine/fail-loud
> half works (those journal entries ARE the new code firing), but the **backup-creation
> path evidently never produced a `.bak`**, so each cycle still falls through to fresh
> defaults until a human restores. That unfinished half is a primary review target.

## Goal
Adversarial review of the finance-analyzer production code, partitioned into 8
subsystems, each reviewed independently by a fresh Claude Code review subagent in
parallel, plus an independent author (this session) review pass, then
cross-critiqued and synthesized. **Read-only review — no production code is
modified this session.** Deliverable is review docs committed to `main`.

## Why this shape
- **Empty-baseline worktrees:** each subsystem's curated hot-path files are
  committed onto a branch (`fgl/<sub>`) whose parent is an empty tree, so
  `git diff fgl/baseline-empty HEAD` presents the whole subsystem as a clean
  "PR" the reviewer can audit. Full repo remains at `Q:/finance-analyzer` for
  cross-file context reads.
- **Curated scope, not exhaustive:** ~153K LOC across 421 prod files. No reviewer
  can digest that. Each subsystem is scoped to the code that runs the live loops
  and touches money, state, or auth. One-off scripts, backtests, dead modules excluded.
- **Two reviewer types:** `caveman:cavecrew-reviewer` (tight, one-line findings),
  `pr-review-toolkit:code-reviewer` (broad/deep).

## Subsystem partition (curated)
| Subsystem | Reviewer | Files / LOC | Key adversarial focus |
|---|---|---|---|
| signals-core | pr-toolkit | 8 / 9.7K | vote aggregation, accuracy gate inversion, lookahead in backfill |
| orchestration | pr-toolkit | 8 / 10K | L2 subprocess silent-fail (exit 0 but failed), loop crash recovery |
| portfolio-risk | pr-toolkit | 7 / 3.9K | **LIVE corruption bug** state I/O + missing-backup; ATR/DD math |
| metals-core | pr-toolkit | 6 / 15.6K | live order/stop-loss, MINI barrier proximity, EOD-flat |
| avanza-api | caveman | 6 / 3.5K | stop-loss API (Mar 3 incident), BankID session, account filter |
| signals-modules | pr-toolkit | 12 / 5.8K | per-signal lookahead, NaN, vote inversion, confidence range |
| data-external | caveman | 9 / 2.3K | live-prices-first, stale-cache-as-live, retry/timeout |
| infrastructure | pr-toolkit | 10 / 5.2K | **file_utils atomic I/O** (corruption root cause), auth bypass |

## Live context driving focus (startup checks)
- 12 unresolved critical errors today. Recurring `portfolio_state_corrupt` →
  portfolio-risk + infrastructure pointed at it (audit the **backup creation**
  path, not just recovery — why is there no `.bak`?).
- 6–12 signals dropped >15pp below 50% (`accuracy_degradation`) → signals-core
  checks the detector for false positives + baseline staleness.

## Execution order
1. ✅ Inventory + partition (`data/_fgl_inventory.json`, `data/_fgl_manifest.json`)
2. ✅ Empty-baseline branch + 8 worktrees
3. Commit this plan
4. Spawn 8 background review subagents (concurrent)
5. Independent adversarial pass (this session) on highest-risk files while
   subagents run: `portfolio_mgr.py`, `file_utils.py`, `metals_loop` stop-loss,
   `signal_engine` voting, `agent_invocation` subprocess
6. Collect subagent results → `docs/fgl-review/<sub>.md`
7. Cross-critique (dedup, confirm/refute against real lines, severity reconcile)
8. Synthesis `docs/fgl-review/SYNTHESIS.md`
9. Commit docs to main, push via Windows git, clean up worktrees

## Premortem (review-validity failure modes)

This is a **read-only, docs-only** task: no production code, config, or state is
touched, so classic /fgl incident vectors (loop crash, bad trade, data corruption,
auth outage) are **ACCEPT — unreachable by this plan**. The real failure mode is a
review that gives *false confidence* or *misleads* a future session. A dedicated
premortem agent is not spawned — the 8 independent reviews + cross-critique already
supply adversarial multi-perspective pressure. Failure modes considered:

1. **False-clean on the live corruption bug.** Reviewers bless `portfolio_mgr`'s
   try/except recovery and miss that the recovery path *is* the bug (silent
   default-wipe; no working `.bak`). → Manifest names today's incident explicitly;
   my own pass independently audits the same path; both portfolio-risk and
   infrastructure get it.
2. **Curation blind spot.** A real bug lives in an excluded file. → Reviewers told
   full repo is readable for context; synthesis lists OUT-of-scope explicitly so the
   gap is visible. ACCEPT residual (can't review 153K LOC).
3. **Hallucinated findings / wrong line numbers.** → Cross-critique spot-checks each
   P0/P1 against the actual file before synthesis; unverifiable → demoted/flagged.
4. **Severity inflation.** Style nits tagged P1. → Synthesis re-grades on a
   money/state/auth/silent-failure rubric, ignoring subagent self-severity.
5. **Subagent stall / empty return** (the reason /fgl left Codex). → Agents are
   observable; empty return → that subsystem falls back to my pass + synthesis note.
   No silent gap.
6. **Worktree leakage into main.** Leftover `fgl/*` branches / `.worktrees/fgl/`. →
   Explicit cleanup + post-merge verification of `git worktree list` / `git branch`.
