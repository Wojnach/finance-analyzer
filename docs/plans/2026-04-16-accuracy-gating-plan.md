# Plan — Accuracy Gating Reconfiguration (2026-04-16)

## Problem

Consensus accuracy on Tier-1 tickers dropped W14→W15: 12h went 55%→39%. MSTR cratered to 21.9% at 1d in W16 against an +8.4% rally. Investigation showed cause is **configuration cascade**, not a bug:

1. Per-ticker blacklist (`_TICKER_DISABLED_SIGNALS`, shipped Apr 14-15) was built from **3h** accuracy data but applied at **all horizons**. On MSTR, 5 of 7 blacklisted signals have 66-81% accuracy at the 1d horizon.
2. Aggregate 47% accuracy gate masks per-ticker heterogeneity — fear_greed 100% on XAU/XAG but 0-20% on stocks pulls the aggregate to 25.9%, killing it globally.
3. Recency weights 0.75/0.95 (raised Apr 15) amplify noise on small 7d windows.
4. Gates fire with phase lag — by the time the 7d window shows degradation, the regime causing it is ending.

## Evidence

Per-ticker per-horizon accuracy (last 7d, Tier-1, n>=30):

- MSTR 1d: macro_regime 81.4% / trend 71.2% / sentiment 80.0% / volatility_sig 66.7% / volume 62.3% — all BLACKLISTED right now.
- fear_greed 1d per ticker: XAU 100%, XAG 100%, BTC 68%, ETH 75%, MU 0%, VRT 2% — aggregate 25.9%.
- Tier-1 weekly consensus 12h: W14 55% -> W15 39% -> W16 46.6%.

Full data in memory `project_accuracy_degradation_20260416.md`.

## Fix Batches

### Batch 1 — Constants revert + MSTR blacklist trim (low risk, reversible)

Files: `portfolio/signal_engine.py`, affected tests.

- `_RECENCY_WEIGHT_NORMAL`: 0.75 -> 0.70
- `_RECENCY_WEIGHT_FAST`: 0.95 -> 0.90
- `_ACCURACY_GATE_HIGH_SAMPLE_MIN`: 5000 -> 10000
- `_TICKER_DISABLED_SIGNALS["MSTR"]`: trim to `{claude_fundamental, credit_spread_risk}` (remove macro_regime, trend, volatility_sig, volume, sentiment).
- Update affected test assertions.

Commit: `fix(signals): revert recency weights + trim MSTR horizon-mismatched blacklist`

### Batch 2 — Voter-count circuit breaker

Files: `portfolio/signal_engine.py`, `tests/test_signal_engine_circuit_breaker.py` (new).

When the accuracy gate would leave fewer than `_MIN_ACTIVE_VOTERS_SOFT` (5) active voters, progressively relax the gate by +0.02 until the floor is met or relaxation caps at +0.06. Rationale: losing voter diversity is worse than letting a borderline signal vote. Log the relaxation events.

Commit: `feat(signals): circuit breaker - relax accuracy gate when voter count drops below floor`

### Batch 3 — Per-ticker accuracy gate

Files: `portfolio/accuracy_stats.py`, `portfolio/signal_engine.py`, `tests/test_per_ticker_gate.py` (new).

Extend `get_accuracy_stats` to optionally return per-ticker accuracy keyed by `(signal, ticker, horizon)`. In `_weighted_consensus`, when per-ticker accuracy exists for >=30 samples, gate on per-ticker value; otherwise fall back to aggregate. Un-gates fear_greed on XAU/XAG while keeping it off MSTR/stocks.

Commit: `feat(signals): per-ticker accuracy gate - aggregate gate no longer masks per-ticker wins`

### Batch 4 — Horizon-specific per-ticker blacklist

Files: `portfolio/signal_engine.py`, `tests/test_horizon_specific_blacklist.py` (new).

Replace `_TICKER_DISABLED_SIGNALS` with `_TICKER_DISABLED_BY_HORIZON = {"3h": {...}, "1d": {...}, "_default": {...}}`. Compute-time gate uses the intersection (minimum disable set) to avoid wasted work. Consensus-time gate applies the per-horizon blacklist. Rebuild the entries from 30d per-horizon data.

Commit: `feat(signals): horizon-specific per-ticker blacklist - prevents horizon-mismatch regressions`

### Batch 5 — Recompute accuracy cache + counterfactual replay

Files: `scripts/recompute_accuracy_cache.py`, `scripts/replay_consensus.py`, `data/consensus_replay_20260416.json`.

After all code changes:
1. Regenerate `accuracy_cache.json` with the new per-ticker/per-horizon aggregations so the new gate has the right data to consult on first cycle after restart.
2. Replay signal_log (last 30 days) with the new gating logic; compute consensus accuracy under new rules; save to data/consensus_replay. Validates the fix quantitatively before restart.

Commit: `tools(signals): recompute accuracy cache + counterfactual replay for validation`

## Post-Implementation

1. Full pytest: `.venv/Scripts/python.exe -m pytest tests/ -n auto --timeout=60`
2. Codex adversarial review: `/codex:adversarial-review --wait --scope branch --effort xhigh`
3. Address P1/P2 findings. Document P3.
4. Merge to main, push, restart PF-DataLoop.
5. Monitor consensus accuracy for the next 72 hours via `--accuracy` report.

## Risk Assessment

- Batch 1: Low — pure constant reverts + blacklist trim. Reversible in 1 commit.
- Batch 2: Medium — adds new code path. New test covers it. Can be disabled by setting `_MIN_ACTIVE_VOTERS_SOFT = 0`.
- Batch 3: Medium — plumbing change through accuracy_stats. Fallback to aggregate keeps old behavior for signals with insufficient per-ticker samples.
- Batch 4: Medium — structural refactor of blacklist. Must preserve old behavior for horizons not explicitly listed.
- Batch 5: Low — tools and data only; no runtime behavior change.

## Rollback Plan

Each batch is an independent commit. If any causes issues in production:
1. `git revert <sha>` to undo that batch only.
2. Push. Restart loop. Previous batches stay.
3. Diagnose, revise, re-ship.

## Out of Scope (deferred)

- Rebuilding historical consensus outputs in signal_log.jsonl (they're a faithful record of what happened under the old rules; rewriting them would lose provenance).
- Dynamic (automated) blacklist regeneration from per-horizon accuracy. Requires running a background job that could churn the blacklist — kept manual for now, revisit after we see how well the static per-horizon lists hold up.
- Oscillators selective re-enable — deferred to a follow-up PR to keep this PR focused.
