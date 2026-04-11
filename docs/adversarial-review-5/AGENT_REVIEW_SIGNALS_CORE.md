# Agent Review: signals-core — Round 5 (2026-04-11)

**Agent**: feature-dev:code-reviewer
**Files reviewed**: 13 (signal_engine, signal_registry, signal_utils, signal_weights,
signal_weight_optimizer, signal_history, signal_db, signal_postmortem, accuracy_stats,
ticker_accuracy, outcome_tracker, forecast_accuracy, train_signal_weights)
**Duration**: ~257s

---

## Findings (9 total: 0 P0, 2 P1, 5 P2, 2 P3)

### P1

**SC-R5-1** accuracy_stats.py:291-297 — Cost-adjusted accuracy neutral/cost threshold mismatch
- Pre-filter uses _MIN_CHANGE_PCT=0.05%, cost threshold is 0.10% (10 bps)
- Moves in 0.05-0.10% band inflate total but never reach correct
- Cost-adjusted accuracy systematically deflated

**SC-R5-2** [CRITICAL] signal_engine.py:1929-1932 — Utility boost saturates at max for ALL signals
- avg_return from signal_utility() is in raw % (e.g., 2.5 for 2.5%)
- boost = min(1.0 + 2.5, 1.5) = always 1.5 if avg_return > 0.5%
- Nearly all signals get maximum 1.5x boost → overrides accuracy gate
- A 48% accuracy signal with positive returns → boosted to 72% → passes 47% gate
- Fix: Normalize to fraction: `boost = 1.0 + min(u_score / 100.0, 0.15)`

### P2

**SC-R5-3** accuracy_stats.py:851-879 — Regime/ticker accuracy caches share single timestamp
- Same cross-horizon staleness bug as BUG-133 but in regime and ticker caches
- 3h cache write makes 1d cache appear fresh for up to 1 hour

**SC-R5-4** forecast_accuracy.py:294-301 — backfill_forecast_outcomes truncates predictions file
- `break` on max_entries exits before appending remaining entries
- _write_predictions writes incomplete file → progressive history loss
- Fix: Append remaining entries after break

**SC-R5-5** signal_history.py:53-82 — update_history has no locking under ThreadPoolExecutor
- Read-modify-write without lock, 8 concurrent worker threads
- Lost writes when two tickers update simultaneously

**SC-R5-6** signal_engine.py:1466-1475 — On-chain BTC tied vote behavior undocumented
- 1-BUY/1-SELL/2-HOLD satisfies total >= 2 but silently falls to HOLD
- Sub-signals returning None are skipped (not counted) vs HOLD (counted)

**SC-R5-7** outcome_tracker.py:84-91 — Reconstructed sentiment votes don't apply hysteresis
- Historical entries use 0.40 threshold, live engine uses 0.55 for flips
- Inflates sentiment accuracy for old entries
- Only affects pre-_votes entries (oldest data)

### P3

**SC-R5-8** signal_weights.py — Dead code (MWU weight system never loaded)
**SC-R5-9** accuracy_stats.py:912 — read_text() instead of load_jsonl for accuracy snapshots

---

## Regression Verification
All Round 1-4 fixes confirmed present and correct:
- Directional gate 0.40 ✓, BUG-182 directional weights ✓
- Accuracy gate 0.47 ✓, ADX cache content key ✓
- _accuracy_write_lock ✓
