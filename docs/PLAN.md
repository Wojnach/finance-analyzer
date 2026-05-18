# PLAN — Widen accuracy-degradation window to 14d to smooth regime flips

Date: 2026-05-18
Branch: `fix-quiet-accuracy-degradation-alerts`
Worktree: `.worktrees/quiet-accuracy-alerts`

## Problem

14 unresolved `critical_errors.jsonl` entries from the past 7 days,
mostly `accuracy_degradation` alerts firing on 5 signals (sentiment,
structure, macro_regime, econ_calendar, crypto_macro). The user reported
"system isn't trading" — investigation found three independent
confidence gates (bull_trap, ensemble_entropy 0.92, per_ticker_consensus
49.4%) correctly crushing weighted_confidence to ~0.10, well below the
0.56 GRID_MIN_SIGNAL_CONFIDENCE floor → grid_fisher skips all
placements.

Root cause confirmed via signal_log.db: market regime flipped around
2026-05-11. 1d pct_up by ticker:
- 5/04-5/11: 79% BTC, 63% ETH, 74% MSTR, 71% XAG, 62% XAU (rally)
- 5/11-5/19: 24% BTC, 20% ETH, 32% MSTR, 43% XAG, 27% XAU (pullback)

The 5 BUY-leaning signals scored 60-70% during the rally and 33-39%
during the pullback. They are not broken — they're directionally biased
and lost on a regime flip. The degradation alert compares recent 7d
to baseline 7d, and 7d windows fit entirely on one side of the flip
→ guaranteed false alarm.

Empirical proof via signal_log.db (14d windows smooth the flip):

| Signal         | 7d_recent  | 14d_recent | 14d_baseline | 14d delta |
|----------------|------------|------------|--------------|-----------|
| sentiment      | 38.0%      | 51.0%      | 48.1%        | +2.9pp    |
| structure      | 37.8%      | 50.2%      | 37.5%        | +12.7pp   |
| macro_regime   | 39.1%      | 49.6%      | 49.2%        | +0.4pp    |
| econ_calendar  | 33.9%      | 53.2%      | 44.8%        | +8.4pp    |
| crypto_macro   | 35.2%      | 55.4%      | 53.8%        | +1.6pp    |

All 14d deltas are positive or below the 15pp drop threshold → alert
would not fire. 7d windows produce 22-37pp false drops.

## Goal

1. Widen the accuracy-degradation recent-window from 7d to 14d so
   single-week regime flips don't trip the alert.
2. Resolve the 14 unresolved critical_errors entries (regime-flip
   artifacts, not real bugs).

This does NOT touch:
- The trade-gate accuracy lookup (`get_or_compute_recent_accuracy`
  in `signal_engine.py` with `days=7`). Trade confidence stays gated
  by the bull_trap + ensemble_entropy + per_ticker_consensus cascade
  which is working correctly.
- `ACCURACY_GATE_THRESHOLD = 0.47`, `GRID_MIN_SIGNAL_CONFIDENCE = 0.56`,
  penalty multipliers. They are catching real risk.

The fix quiets noise. It does NOT cause new trades. Trading will
resume when actual signal accuracy recovers or when the regime
clarifies.

## Design

### Change 1: Widen recent window

`portfolio/accuracy_degradation.py`:

```python
BASELINE_TARGET_DAYS = 14.0      # was 7.0
MIN_SNAPSHOT_AGE_DAYS = 13.0     # was 6.0
```

`save_full_accuracy_snapshot(*, days: int = 7)` → `days: int = 14`.

Rationale:
- `BASELINE_TARGET_DAYS = 14.0` controls both baseline lookup AND the
  current recent-window cutoff in `_diff_against_baseline()`
  (line 449: `cutoff = now - timedelta(days=int(BASELINE_TARGET_DAYS))`).
  Bumping it widens both ends symmetrically.
- `MIN_SNAPSHOT_AGE_DAYS = 13.0` ensures we only compare baselines
  that are old enough to have been written under the new wider window
  (1d slack for snapshot timing jitter).
- Snapshot writer `days=14` ensures future snapshots store 14d data.

### Change 2: Resolve old critical_errors

Append resolution lines to `data/critical_errors.jsonl` for the 14
unresolved entries. Grouped by category:
- 11x `accuracy_degradation` → resolved by this PR (window widened)
- 2x `avanza_account_mismatch` → resolved by user re-running
  `scripts/avanza_login.py` (session now valid through 2026-05-19T13:05)
- 1x `contract_violation` (layer2_journal_activity at 09:32) → resolved
  by subsequent journal entries written within the grace window

Resolution lines follow the format in CLAUDE.md:

```json
{"ts":"<now>","level":"info","category":"resolution",
 "caller":"<same>","resolution":"<what was done>",
 "resolves_ts":"<original ts>","message":"<short>","context":{}}
```

### Change 3: Tests

`tests/test_accuracy_degradation*.py` — verify:
- Constants updated to new values
- `save_full_accuracy_snapshot` default is 14
- Existing tests that may assume `days=7` updated

## Files Touched

1. `portfolio/accuracy_degradation.py` — bump 3 constants
2. `data/critical_errors.jsonl` — append resolution lines (in repo root,
   not worktree — append from main after merge)
3. `tests/test_accuracy_degradation*.py` — verify tests still pass,
   update any hardcoded `days=7` assertions
4. `docs/PLAN.md` — this file (committed before implementation)
5. `docs/SESSION_PROGRESS.md` — end-of-session note

## Risks

1. **Slower real-degradation detection.** A signal that genuinely starts
   underperforming will take ~2x longer to trip the alert under a 14d
   window vs 7d. Acceptable trade-off: real alpha decay persists for
   weeks, the alert catches it eventually. Spurious 1-week regime flips
   are the dominant failure mode in practice (this incident + 2026-04-16
   W15/W16 collapse memory).

2. **Transition window 13-14 days of fewer alerts.** Until 13d+
   baselines accumulate in the snapshot history, MIN_SNAPSHOT_AGE_DAYS
   blocks comparison. Net effect: alerts go quiet for ~13 days, then
   resume. Acceptable — the user is already aware of current degradation
   from this conversation, and the *trade gates* (which DO operate on
   7d) continue catching weak signals during this window.

3. **Critical_errors resolution mass-write.** Atomic-append, each line
   independent. If the loop is mid-write to the file, our appends
   interleave cleanly (jsonl is line-delimited). No risk of corruption.

4. **`days=14` parameter change ripples elsewhere?** Grepped for
   `save_full_accuracy_snapshot` — only called from
   `maybe_save_daily_snapshot(config)` which doesn't override `days`.
   So the default change takes effect everywhere.

## Execution Order

1. Write plan + commit (this step)
2. Premortem via fresh general-purpose agent
3. Implement bump: constants + snapshot writer default
4. Run targeted tests: `pytest tests/test_accuracy_degradation*.py -v`
5. Implement critical_errors resolution lines (append-only to data file
   in main worktree, not branch — done as part of merge step)
6. Run full suite: `pytest tests/ -n auto`
7. Adversarial review via `caveman:cavecrew-reviewer`
8. Address P1/P2 findings, document P3
9. Commit + merge + push
10. Restart PF-DataLoop (degradation tracker runs inside main loop)

## Premortem

(To be filled by fresh agent — see step 2.)
