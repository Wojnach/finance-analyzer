# FGL Review — signals-core

Scope: clean PR adding 8 files (`git diff fgl/baseline-empty HEAD`):
`portfolio/signal_engine.py`, `accuracy_stats.py`, `accuracy_degradation.py`,
`outcome_tracker.py`, `signal_registry.py`, `shadow_registry.py`,
`signal_utils.py`, `forecast_accuracy.py`. Reviewed in full against the
project invariants (atomic I/O; no silent-wrong-but-quiet) and the
adversarial focus list (vote aggregation, accuracy gate, recency blend,
MIN_VOTERS, degradation false-positives, lookahead, shadow leakage,
NaN/div0, per-ticker overrides, tier-boost double-count).

No P0 issues found. The hot consensus path is heavily defended
(`_safe_accuracy`/`_safe_sample_count`/`_coerce_sample_count` coercion,
fail-closed accuracy load, atomic I/O throughout, shadow votes correctly
isolated from the consensus tally, outcome backfill correctly gated on
`now_ts < target_ts` so there is no lookahead leak into live votes).

The findings below are correctness/consistency gaps where the system
silently does something other than what its own comments / CLAUDE.md
claim — i.e. the "wrong-but-quiet" failure mode the invariants target.

## Critical (90-100)

(none)

## Important (80-89)

- **[P2→reported, conf 88] signal_engine.py:723-736, 3471** — The
  per-ticker rescue override `("ml", "ETH-USD")` in
  `_DISABLED_SIGNAL_OVERRIDES` is permanently dead, so ETH never gets the
  ML-classifier vote the code claims it does. Causal chain: `ml` is not a
  registered enhanced signal — its vote is hardcoded
  `votes["ml"] = "HOLD"` at line 3471 with no compute path and
  `extra["ml_action"]` is never set (confirmed: only readers exist in
  `outcome_tracker.py:96`, `reporting.py:80`, `main.py:534`). The override
  set is only consulted in `_compute_applicable_count` (1631) and the
  enhanced-dispatch gate (3736), neither of which can resurrect a
  hardcoded-HOLD core vote. Result: the comment "ETH-USD has genuine edge"
  (55.1% @ 3h, 1206 sam) and the CLAUDE.md line "ML Classifier → ETH
  (55.1% 3h)" describe a contribution that silently does not happen — a
  signal the maintainers believe is voting is force-HOLD'd every cycle.
  Low money-impact (one voter, and it would be a *new* voter, not a
  wrong one), but it is a quiet divergence between documented and actual
  behavior. **Fix:** either implement a real `ml` compute path that reads
  the classifier and sets `votes["ml"]`/`extra["ml_action"]` (then the
  override works), or delete the `("ml", "ETH-USD")` entry and the
  CLAUDE.md claim so no future session trusts a phantom voter.

## Low / Maintainability (P3, reported for traceability)

- **[P3, conf 90] signal_engine.py:516-530** — `_accuracy_tier_mult` and
  its `_ACCURACY_TIER_THRESHOLDS` table (the documented 1.25x/1.15x/1.05x
  "walk-forward accuracy tier multiplier") are defined but never called
  anywhere in the repo (grep: only the def at line 525 matches). The
  docstring says "Applied AFTER direction-specific weight assignment," but
  `_weighted_consensus` never invokes it — weight is just `acc` (line
  2727) with no tier amplification. This is the opposite of the
  "double-counting" the focus list worried about: the boost is entirely
  absent, so the system is *more conservative* than CLAUDE.md's
  "Accuracy tier boost" bullet claims. Not a correctness bug (raw accuracy
  is already monotonic in weight), hence P3, but it is dead code masquerading
  as an active mechanic. **Fix:** either wire `weight *= _accuracy_tier_mult(acc)`
  into the weight loop after line 2729 (matching the docstring), or delete
  the function + table + the CLAUDE.md bullet. Decide deliberately — turning
  it on changes live weights.

- **[P3, conf 82] forecast_accuracy.py:142-159** — `actual_change =
  outcome.get("change_pct", 0)` then `actual_up = actual_change > 0`
  has no None-guard, unlike every other scorer in this PR
  (`accuracy_stats._vote_correct` explicitly handles None, per its own
  2026-04-22 comment). Today this is safe because the only writer of
  `outcome[horizon]` is `backfill_forecast_outcomes` in the same module,
  which always writes a numeric `change_pct`. But it is the one outcome
  scorer in the PR that would raise `TypeError: '>' not supported between
  NoneType and int` if a null ever lands in `forecast_predictions.jsonl`
  (e.g. a future writer, or a hand-edited/torn file) — and it sits inside
  `_diff_against_baseline`'s broad `try/except` (accuracy_degradation.py:546),
  so the failure would be swallowed and silently zero out the entire
  forecast degradation scope. Also, the `,0` default scores a missing
  outcome as "down" rather than skipping it. **Fix:** mirror `_vote_correct`:
  `change_pct = outcome.get("change_pct"); if change_pct is None: continue`,
  and skip near-zero moves with the same `_MIN_CHANGE_PCT` neutral band the
  other scorers use.

## Summary

- P0: 0
- P1: 0
- P2 (reported): 1  — dead `("ml","ETH-USD")` override (signal_engine.py:723/3471)
- P3 (reported): 2  — uncalled `_accuracy_tier_mult` dead code; missing None-guard in `forecast_accuracy.compute_forecast_accuracy`

Most important finding: the dead `("ml", "ETH-USD")` rescue override
(signal_engine.py:723-736 vs the hardcoded `votes["ml"]="HOLD"` at 3471).
It is the only place in this subsystem where a signal the maintainers and
CLAUDE.md believe is contributing to ETH consensus is in fact silently
force-HOLD'd every cycle — the "wrong-but-quiet" class the invariants
flag. It carries no immediate money-loss risk (an absent voter, not a
mis-voting one), which is why it stays below Critical, but it should be
either implemented or removed so the documented signal map matches reality.

Positive verifications (no finding): no lookahead in `outcome_tracker.backfill_outcomes`
(scoring gated on `now_ts < target_ts`, outcomes written to a separate
`outcomes` field never fed back into a live vote); shadow votes never enter
the consensus tally (`votes[sig]="HOLD"` for shadow-safe signals; `raw_votes`
merge is logging-only); accuracy gate is force-HOLD with directional rescue,
never inverted; recency blend math (70/30 normal, 90/10 fast, catastrophic
floor) is correct and applies the sample floor to recent-only and directional
keys; MIN_VOTERS counts active BUY+SELL voters post-persistence
(`post_persistence_voters`), not total; degradation tracker uses a 14d
windowed baseline with SE-significance + dual gate + per-key cooldown
(no high-water-mark re-fire); atomic I/O used everywhere
(`atomic_write_json`/`atomic_append_jsonl`/`atomic_write_jsonl`), and the
one raw-`open` site in `backfill_outcomes` is a deliberate
lock-held byte-preserving rewrite, not a state write.
