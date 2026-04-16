# Session Progress — Accuracy Gating Reconfiguration (2026-04-16)

**Session start:** ~13:30 CET (after user noticed consensus accuracy dropped)
**Branch:** `fix/accuracy-gating-20260416`
**Worktree:** `/mnt/q/finance-analyzer-accgate/`
**Base SHA:** `95d6823` (main)

## Investigation complete

Traced W15/W16 consensus accuracy collapse (MSTR 1d 21.9% vs +8.4% rally) to
configuration cascade. See memory `project_accuracy_degradation_20260416.md`
and `docs/PLAN.md`.

## Batches shipped on worktree

| # | Commit | Description | Test status |
|---|---|---|---|
| Plan | `b88e3ed` | docs/PLAN.md — 5-batch plan | — |
| 1 | `fd504d4` | Revert recency 0.75→0.70 / 0.95→0.90; widen high-sample min 5000→10000; trim MSTR blacklist 7→2 entries | 319/319 target suites |
| 2 | `04e0ae2` | Voter-count circuit breaker (+ helpers, + pre-condition guard, + 18 tests) | 338/338 target suites |
| 3 | — | Skipped: per-ticker accuracy override already exists (BUG-158 at signal_engine.py:2187-2209); Batch 3 reframed as "already done" in task notes | — |
| 4 | `898c38e` | Horizon-specific per-ticker blacklist `_TICKER_DISABLED_BY_HORIZON`; `_weighted_consensus` takes `ticker` param; 12 new tests | 350/350 target suites |
| 5 | `c3f0916` | `scripts/replay_consensus.py` + `data/consensus_replay_20260416.json` | — |

## Counterfactual replay (14d window, main-workspace signal_log.jsonl)

| Horizon | Actual | Simulated | Δ |
|---|---|---|---|
| 3h | 51.90% | 48.69% | **−3.21pp** (regression — investigate) |
| 1d | 52.73% | 53.59% | +0.86pp |
| 3d | 52.76% | 54.37% | +1.61pp |

Per-ticker at 1d (Tier-1 only):
- MSTR: 49.15 → 54.95 (+5.80pp) ← the core fix
- XAG: 47.15 → 56.70 (+9.55pp)
- XAU: 41.92 → 47.94 (+6.02pp)
- BTC: 49.66 → 52.40 (+2.74pp)
- ETH: 52.64 → 47.54 (**−5.10pp** — investigate)

Tier-1 1d average delta: +3.80pp.

Stored: `data/consensus_replay_20260416.json` (full breakdown per horizon + ticker).

## To do before shipping

1. **Investigate 3h regression** and **ETH 1d regression** in replay.
2. **Run full pytest suite** — `pytest tests/ -n auto`.
3. **Codex adversarial review** on the branch (if usage limit allows).
4. **Merge to main + push + restart PF-DataLoop.**
5. **Clean up worktree.**

## Rollback

Each batch is an independent commit. `git revert <sha>` undoes one without
touching others. Worktree at `Q:\finance-analyzer-accgate\` is isolated —
no live loop impact until merged into main and loop restarted.

## Session context for resume

If this session crashes:
- Branch: `fix/accuracy-gating-20260416` at HEAD `c3f0916`
- All target suite tests pass (350/350 in signal engine suite at Batch 4).
- Next action: investigate ETH 1d + 3h regressions in replay (may be expected
  given the circuit breaker intentionally lets borderline signals through),
  then full pytest, codex review, merge, push, restart loop.
- Key files: `portfolio/signal_engine.py`, `portfolio/accuracy_stats.py`,
  `tests/test_signal_engine_circuit_breaker.py`, `tests/test_horizon_specific_blacklist.py`.
- Data snapshot: `data/consensus_replay_20260416.json` has full replay breakdown.
