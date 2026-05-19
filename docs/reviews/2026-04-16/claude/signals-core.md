# Adversarial Review — signals-core (Claude-independent)

## Critical Findings

**P0: Consensus counting inaccuracy in recency blending** (signal_engine.py, accuracy_stats.py:725-777)
- **File:line**: accuracy_stats.py:764
- **What's wrong**: `blend_accuracy_data()` computes blended accuracy but derives `correct = int(round(blended * total))` using `total = max(at_samples, rc_samples)` — taking the max of alltime vs recent samples. When recent has 50 samples (48% accuracy) and alltime has 5000 samples (55%), blending at 70%/30% yields 0.526 accuracy. The code then assigns `total=5000`, deriving `correct=2630`. But neither the 48%-on-50 nor the 55%-on-5000 produced 2630 correct votes. This corrupts downstream gate calculations that check if `signal["total"] >= MIN_SAMPLES`.
- **Why it matters**: A signal incorrectly appears to have 5000 samples when accuracy is recent-window biased, causing the 47% gate to apply to signals that should be evaluated on only 50 recent samples (where 48% might indicate genuine recent degradation requiring intervention). Silent stat corruption feeds into position-sizing decisions and accuracy metrics.
- **Suggested fix**: Change line 764 to `total = rc_samples if rc_samples >= min_recent_samples else at_samples` — use the sample count that actually drove the blended accuracy.

**P0: Double-write desynchronization on concurrent signal_db writes** (signal_db.py, outcome_tracker.py:331-410)
- **File:line**: outcome_tracker.py:163-165
- **What's wrong**: `log_signal_snapshot()` appends to JSONL then opens a *new* SignalDB connection and calls `insert_snapshot()` with the same entry. SQLite has WAL but if the JSONL write succeeds and the SQLite write fails partway (e.g., outcome insert raises but snapshot insert succeeded), the two stores are desynchronized. No transaction wraps both writes; a crash mid-SQLite-write leaves the entry in JSONL but incomplete in SQLite, causing `signal_db.load_entries()` to return incomplete data.
- **Why it matters**: Accuracy stats read from SQLite and find incomplete `ticker_signals` rows without outcomes, silently excluding them from accuracy calculations. Accuracy appears higher than reality because losing outcomes = losing failure cases.
- **Suggested fix**: Wrap both JSONL and SQLite writes in a single atomic operation. If JSONL succeeds and SQLite fails, delete the JSONL line on exception, or vice versa. Alternatively, make SQLite the source of truth and drop JSONL dual-write.

**P1: Circuit breaker voter-floor logic has off-by-one in gate relaxation** (signal_engine.py:219-235)
- **File:line**: signal_engine.py:233-234
- **What's wrong**: The circuit breaker relaxes the accuracy gate from 47% down to 41% in 2pp steps: `threshold = 0.47 - (relax_step * n)` where `relax_step=0.02` and max relax=6pp (3 steps). After 3 relaxations, threshold is 0.41. But the rules.md says "effective floor 41%" and the code comment says "to 41%", implying 41% is the absolute floor. If a 4th voter would be recovered at 40%, the code keeps gating at 41%, leaving the floor unreachable. The gate should be `max(0.47 - 0.06, 0.41)` but is instead `0.47 - (min(3, n) * 0.02)` which caps at 41%. This is a *correctness* issue if the rules intended to recover voters down to 41% exactly.
- **Why it matters**: During regime transitions, voters might cluster just below 41% accuracy and the system leaves them gated when the intent was to recover them at the 41% floor. Consensus voter count stays below MIN_ACTIVE_VOTERS_SOFT unnecessarily.
- **Suggested fix**: Document whether 41% is a hard floor or a "up to 6pp relaxation" behavior. If hard floor, add explicit `max(threshold, 0.41)` clamp. If "up to 6pp", clarify in rules.md.

**P1: Silent exception swallowing in signal load / early return as HOLD** (signal_engine.py, signal_registry.py:93-103)
- **File:line**: signal_registry.py:82-103 and signal_engine.py wherever signals are dispatched
- **What's wrong**: When a signal module fails to import (e.g., typo in enhanced signal list, missing dependency), `load_signal_func()` logs a warning, caches the failure for 300s, and returns None. The caller then treats None as "HOLD" or skips the signal silently. Meanwhile, the error goes to a log file instead of producing a Violation or immediate alert. If a critical signal (e.g., macro_regime, forecast) silently fails to load for 5 minutes, the system is voting without it and the user has no real-time visibility — only a timestamped log line.
- **Why it matters**: In a 60-second loop, a 5-minute cooldown on load failures means 4 full cycles (20 snapshots) where the signal doesn't vote and no alert fires. Position-sizing, tier classification, and consensus accuracy all degrade silently.
- **Suggested fix**: On first import failure, log at ERROR level and immediately emit a Violation (or equivalent high-priority alert). Don't wait for the cooldown; fail fast. Optionally: emit a Violation on every attempt if the 300s cache is still hot (severity=WARNING, throttled).

**P2: Accuracy gate using pre-gated sample count instead of raw votes** (accuracy_stats.py:116-185, signal_engine.py)
- **File:line**: accuracy_stats.py:154-155
- **What's wrong**: When computing `signal_accuracy()`, a signal vote is counted as "total" only if the vote is non-HOLD: `if vote == "HOLD": continue; stats[sig_name]["total"] += 1`. This "total" is the number of non-HOLD votes. Later, gates in signal_engine.py check `if s["total"] >= ACCURACY_GATE_MIN_SAMPLES` to decide whether to apply the 47% gate. But if the signal was already gated at an earlier stage (regime gate, directional gate), the votes recorded in signal_log are post-gate votes (HOLD), not raw votes. The gate samples count shrinks, making the gate appear to have more data than it actually evaluated.
- **Why it matters**: A signal that voted 100 times but was regime-gated 60 times (forced HOLD) has only 40 non-HOLD votes in signal_log. If those 40 votes are 18 correct, accuracy is 45%. But the gate sees "total=40" and checks `40 >= 30` (ACCURACY_GATE_MIN_SAMPLES), so it applies the gate. However, the 40 samples are biased: they're only the signals that survived post-regime gating. The true accuracy should be computed on the original 100 votes before regime gating.
- **Suggested fix**: Track raw votes before any gating (or pre-gate votes in extra["_raw_votes"]), and use raw votes for accuracy calculation. Use post-gate votes only for consensus. Alternatively: in outcome_tracker.py, always log pre-gate votes in signals dict, and compute accuracy on those.

**P2: Per-ticker directional gate using wrong sample size** (accuracy_stats.py:1340-1415 in accuracy_by_ticker_signal, signal_engine.py directional gating)
- **File:line**: accuracy_stats.py:1398 (buy_acc = s["correct_buy"] / s["total_buy"] if s["total_buy"] > 0)
- **What's wrong**: When computing per-ticker per-signal directional accuracy (e.g., "trend BUY accuracy on BTC-USD"), the code divides `correct_buy / total_buy`. But if the signal has only 12 BUY votes for a ticker and 88 SELL votes across all time, the BUY accuracy is computed on 12 samples. The directional gate checks `if buy_acc < 0.40: gate_BUY` without checking if `total_buy >= MIN_SAMPLES_FOR_DIRECTIONAL_GATE`. A signal with 1 correct BUY out of 2 (50% BUY accuracy) is not gated, but a signal with 4 correct BUY out of 10 (40%) is gated at exactly the threshold. The 2-sample signal should never reach the gate logic.
- **Why it matters**: Rare signal+direction+ticker combinations trigger false-positive gates, removing otherwise good signals from consensus because of insufficient sample size.
- **Suggested fix**: Add `if s["total_buy"] < _DIRECTIONAL_GATE_MIN_SAMPLES: continue` before the directional gate check. Set `_DIRECTIONAL_GATE_MIN_SAMPLES` to 20-30.

**P2: Trigger state reset doesn't preserve triggered_consensus baseline** (trigger.py:155-177)
- **File:line**: trigger.py:156-173, specifically line 172
- **What's wrong**: During startup grace period, the code resets the baseline with the *current* consensus: `triggered_consensus[ticker] = sig["action"]` (line 172). If consensus is HOLD at startup (from yesterday's lingering state before the grace period), and then it legitimately becomes BUY 5 minutes later, the trigger records it as "new consensus from HOLD → BUY" and fires correctly. However, if consensus was BUY yesterday and the grace period *during* a BUY state resets the baseline to BUY, then the next BUY→HOLD→BUY flip is detected as "two direction flips" (lines 205-208) instead of "HOLD then new BUY". The triggered_consensus state is not properly isolated from the previous loop's state.
- **Why it matters**: On restart during an active position, the trigger system loses context about whether the current consensus is a continuation or a new signal. This causes spurious T2 (signal analysis) invocations that consume the Tier 2 budget.
- **Suggested fix**: Store triggered_consensus in a separate state-reset path so grace period doesn't overwrite it. Or: clear triggered_consensus at grace-period start instead of seeding it from the current signals.

**P2: Recency weighting divergence threshold doesn't match min_recent_samples** (signal_engine.py:190-193, accuracy_stats.py:726-777)
- **File:line**: signal_engine.py:190-193 and accuracy_stats.py:756
- **What's wrong**: signal_engine.py defines `_RECENCY_MIN_SAMPLES = 30` but blend_accuracy_data() in accuracy_stats.py uses `min_recent_samples=50`. When a signal has 35 recent samples and 1000 all-time samples, signal_engine checks if `rc_samples >= 30`, sees it passes, and enters recency blending. But blend_accuracy_data() checks `rc_samples >= 50` and decides "insufficient recent data, use all-time only". Now two callers interpret "recent window available" differently, leading to inconsistent gate decisions.
- **Why it matters**: A signal can be gated as HOLD by signal_engine (because it misses the divergence check due to blend_accuracy_data falling back to all-time) but the accuracy_stats report shows it as "blended" when actually it was all-time only. Audit inconsistencies.
- **Suggested fix**: Centralize MIN_RECENT_SAMPLES definition in a shared constants module and use the same value in both places.

**P2: SQLite integer division in signal_accuracy (signal_db.py:271)** (signal_db.py:277)
- **File:line**: signal_db.py:277
- **What's wrong**: The SQL query `SELECT ... FROM ... WHERE o.change_pct` loads change_pct and casts it correctly as float in Python (line 271: `if (vote == "BUY" and change_pct > 0)`). However, the signal_db module doesn't import numpy, so NaN comparisons fall back to Python defaults: `change_pct > 0` where change_pct is NaN returns False (NaN is not > 0). But change_pct can be None if the outcome record is incomplete (no price fetch). The code should filter `WHERE o.change_pct IS NOT NULL` before the Python-side comparison, but the SQL doesn't enforce this. If a row has change_pct=None, the Python code silently treats it as non-positive, either excluding it or (if result_val is None) skipping it, creating silent accuracy degradation.
- **Why it matters**: Incomplete outcome records are invisibly excluded from accuracy metrics, inflating observed accuracy by losing failure cases.
- **Suggested fix**: Add `AND o.change_pct IS NOT NULL AND o.price_usd IS NOT NULL` to the WHERE clause in signal_db.py:255-260.

## Reviewed and Looked OK

- **signal_utils.py**: majority_vote(), sma(), ema(), rsi() implementations look sound. The tie-breaking (confidence = 0.0 for HOLD on BUY==SELL) is explicitly documented and correct.
- **signal_weights.py**: MWU weight update logic is correct; (1±eta) multipliers applied correctly, floor at 0.01 respected.
- **signal_weight_optimizer.py**: walk-forward cross-validation structure avoids look-ahead bias; train/test split is clean.
- **ic_computation.py**: Spearman rank correlation implementation looks correct; NaN/zero handling in denominator is defensive.
- **signal_history.py**: Persistence score calculation is correct; flip counting is accurate.
- **trigger.py lines 56-82**: sustained debounce logic (_update_sustained) is correct; count and duration gates OR correctly.
- **accuracy_degradation.py**: Baseline comparison logic (drop_threshold + absolute_floor dual gate) is sound. Throttle replay path (Codex P1#2) is correct.
- **ticker_accuracy.py**: direction_probability() weighting by sqrt(samples) is standard and reasonable.

## Reviewer confidence

0.72
