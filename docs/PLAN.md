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

Fresh agent ran step 2. Findings + mitigations:

### F1 — Silent transition blackout (11-13 days, no comparison runs)
Until baselines aged ≥13d accumulate in the JSONL, `check_degradation`
returns `[]` every cycle. Looks healthy, but the detector is dark.

**Mitigation:** emit an INFO log + structured field
`transition_active=True` in `check_degradation` when baseline lookup
fails due to age. Tracked in `accuracy_snapshot_state.json` so
follow-up observability is straightforward. ACCEPT no Telegram for it
— user is aware from this conversation.

### F2 — Apples-to-oranges baseline (days 13-28 post-merge)
Once a baseline aged ~14d is picked, its `signals_recent` was written
with `days=7` (pre-merge format). Current side is computed with
`days=14`. Diff = spurious alerts.

**Mitigation:** stamp every snapshot with `window_days` in the
top-level dict. `_find_baseline_snapshot` filters to snapshots whose
`window_days == BASELINE_TARGET_DAYS`. Missing field is treated as
old-format and skipped. This makes the transition truly quiet (no
mismatched comparisons) and turns the durable invariant explicit.

### F3 — Cycle budget breach from 2x data size
`_per_ticker_recent` filters `recent_entries` from `load_entries()`.
Doubling to 14d roughly doubles the row count. The Codex P2 review
note flagged 290s of redundant compute risk under 7d. Could nudge
p95 over the 180s MAX_CYCLE_DURATION_S budget.

**Mitigation:** the HOURLY_THROTTLE_S=55min path ALREADY gates this
to ~24 runs/day. Hot-path is once-per-hour, not once-per-cycle. Plus
the entry-sharing optimization (`recent_entries` computed once,
threaded through all four scopes) keeps the scan cost O(1) in the
number of signal names. Doubling the entry list is +O(n), well within
budget. ACCEPT.

### F4 — `cached_forecast_accuracy` cache-key collision
`scripts/audit_accuracy_drops.py:241` still hardcodes `days=7` to
`cached_forecast_accuracy`. The cached_forecast_accuracy memoization
key is `(horizon, days, use_raw_sub_signals)` — different params
means different cache entries, NOT collision/poisoning. The audit
script's 7d view is an intentional separate diagnostic.

**Mitigation:** leave `audit_accuracy_drops.py` at `days=7`. Add a
code comment noting the intentional divergence. The two views co-exist
cleanly in the lru_cache.

### F5 — Telegram daily summary still says "vs prev 7d"
`build_daily_summary` (lines 901, 940, 948, 962) hardcodes "7d" /
"recent-7d". Post-merge the math is 14d but the UI lies.
Existing test at `tests/test_accuracy_degradation.py:714` asserts the
"7d" string and would pass on the bug.

**Mitigation:** replace hardcoded strings with
`f"{int(BASELINE_TARGET_DAYS)}d"`. Update the test to use
the constant symbolically.

### F6 — Snapshot file growth
Snapshots are append-forever; doubling window size doubles per-snapshot
bytes. With ~120d history and growing.

**Mitigation:** ACCEPT for now. Existing `_check_snapshot_freshness`
watches mtime. Add a file-size warning in a separate follow-up if it
becomes material. Not blocking this PR.

### F7 — `ViolationTracker` escalation count reset
The 14 currently-unresolved alerts have escalation history.
Post-merge they'll have different key sets (14d math) and existing
escalation counts effectively reset.

**Mitigation:** This is desired — the existing alerts ARE noise and
we're resolving them explicitly anyway. ACCEPT.

## Updated Files Touched

1. `portfolio/accuracy_degradation.py`:
   - Bump `BASELINE_TARGET_DAYS` 7.0 → 14.0
   - Bump `MIN_SNAPSHOT_AGE_DAYS` 6.0 → 13.0
   - Change `save_full_accuracy_snapshot(*, days: int = 7)` → 14
   - Add `window_days` field to snapshot extras
   - Filter baselines by matching `window_days` in `_find_baseline_snapshot`
   - Emit informational log when transition_active
   - Replace hardcoded "7d" strings in `build_daily_summary`
2. `portfolio/accuracy_stats.py`:
   - Plumb `window_days` extra through `save_accuracy_snapshot`
3. `tests/test_accuracy_degradation*.py`:
   - Update tests asserting "7d" strings or `days=7` defaults
4. `data/critical_errors.jsonl`: append 14 resolution lines (main worktree, post-merge)
5. `docs/PLAN.md`: this file
6. `docs/SESSION_PROGRESS.md`: end-of-session note

