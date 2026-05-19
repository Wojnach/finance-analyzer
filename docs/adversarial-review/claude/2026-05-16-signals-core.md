# Adversarial Review: Signals-Core Subsystem
**Date:** 2026-05-16  
**Scope:** signal_engine.py, signal_registry.py, signal_utils.py, accuracy_stats.py, outcome_tracker.py, decision_outcome_tracker.py, forecast_accuracy.py, cumulative_tracker.py, accuracy_degradation.py

---

## [P1] NaN/None coercion race in _weighted_consensus accuracy sanitization
**File:** portfolio/signal_engine.py:2214-2250  
**Bug:** Paired-field drop logic at line 2247-2248 drops ("accuracy", "total") when accuracy is None, but if buy_accuracy/total_buy are clean, they survive. Creates asymmetric state: a signal has no `accuracy` key but has `buy_accuracy` key. Downstream at line 2544, `stats.get("buy_accuracy", acc)` where `acc = stats.get("accuracy", 0.5)` defaults to 0.5. This masks the poisoned state. A half-written cache row like `{"buy_accuracy": 0.75, "total_buy": 500, "accuracy": null, "total": null}` bypasses the 50%/47% accuracy gate but votes at 75% weight.

**Why it matters:** Bleeds poisoned mid-computation state into live consensus weights, promoting cache-corruption artifact into major voter.

**Fix:** When `accuracy` is None, drop ALL pairs: (`accuracy`, `total`, `buy_accuracy`, `total_buy`, `sell_accuracy`, `total_sell`). Invariant: if ANY accuracy field is poisoned, drop all accuracy+count pairs.

---

## [P1] Silent accuracy-data corruption on high-sample tier relaxation
**File:** portfolio/signal_engine.py:2496-2501  
**Bug:** High-sample tier uses `max(gate - relaxation, _ACCURACY_GATE_HIGH_SAMPLE_THRESHOLD)`. If relaxation > 0 and (gate - relaxation) < 0.47, then effective_gate = max(negative, 0.50) = 0.50. Comment says "high-sample tier is NOT relaxed" but implementation allows relaxation to lower gate below 0.50 then clamp — lossy operation that makes relaxation invisible for high-sample signals. No logging when gate fires.

**Why it matters:** High-sample signals at 49% can silently clear gate if relaxation=0.06, violating documented invariant. Circuit breaker can un-gate broken signals without warning.

**Fix:** Compute high-sample gate BEFORE relaxation: `if samples >= _ACCURACY_GATE_HIGH_SAMPLE_MIN: effective_gate = _ACCURACY_GATE_HIGH_SAMPLE_THRESHOLD` (no max). Apply relaxation only to standard tier. Log if high-sample signal is gated.

---

## [P1] Concurrent backfill race on locked signal_log during append
**File:** portfolio/outcome_tracker.py:369-564  
**Bug:** Function acquires sidecar lock in 3 phases: read (369-401), process (404-509 outside lock), rewrite (514-562 re-acquired). If concurrent appender starts AFTER snapshot but BEFORE lock release, appender blocks on lock. When backfill re-acquires and rewrites, appender is STILL waiting. After os.replace(), lock releases, appender's write to old fd is lost. Data loss.

**Why it matters:** Under high load, concurrent appenders starve. On lock re-acquire, partially-written appends orphaned. Silent signal log corruption (missing recent entries).

**Fix:** Hold lock continuously from read through os.replace(). Move HTTP phase BEFORE acquiring lock. Guard only file I/O phases.

---

## [P2] Loss-of-precision in soft-confidence dampening composition
**File:** portfolio/signal_engine.py:2623-2628  
**Bug:** Multiplies weight by 0.15-0.20 soft value. If prior weight is small (0.0024), result is 0.00043. At consensus, 3-soft-vote slate produces buy_weight=0.0016, sell_weight=0.0001, buy_conf≈0.941, returns BUY at 94.1%. Soft votes at 0.15-0.20 should never produce >90% consensus.

**Why it matters:** Soft multiplier compounds with prior small weights, inverting to produce >90% consensus from weak votes instead of proportional dampening.

**Fix:** Apply soft dampening BEFORE regime/horizon/macro multipliers. Or cap soft-only slates to max 65%. Or require >=2 hard votes for consensus if soft votes participate.

---

## [P2] Race condition in sentiment state write with transient failures
**File:** portfolio/signal_engine.py:1074-1095  
**Bug:** `flush_sentiment_state()` clears dirty flag AFTER successful write. On exception, flag NOT cleared; except block logs and returns without re-raising. Transient I/O errors cause flag to stay True, sentiment state never persists. In-memory changes lost on restart.

**Why it matters:** System persists sentiment to prevent direction-flip oscillation. Long-lived I/O errors cause silent data loss: on restart, reverts to stale state, causing false flip detections.

**Fix:** Log CRITICAL on repeated failures (>3 consecutive). Disable writes or introduce retry cooldown with exponential backoff.

---

## [P2] Stale cached forecast accuracy blocks degradation detection
**File:** portfolio/accuracy_degradation.py:177-185  
**Bug:** `save_full_accuracy_snapshot()` calls `cached_forecast_accuracy()` with 1h TTL. If snapshot saved at 06:00 UTC but cache populated at 04:00 UTC (before outcome backfill), includes stale forecast data. outcome_tracker.py:569-573 invalidates signal_utility cache but NOT forecast_accuracy cache.

**Why it matters:** Degradation detector compares recent accuracy to 7d baseline. Stale cache on one side misses genuine degradation or produces false positives.

**Fix:** outcome_tracker.py, also call `invalidate_forecast_accuracy_cache()` after backfill. Or have forecast backfill invalidate its own cache.

---

## [P2] Inconsistent hold threshold default for local models
**File:** portfolio/signal_engine.py:2747  
**Bug:** `_gate_local_model_vote()` at line 2767 uses `if accuracy < hold_threshold` but name suggests "block if below". At exactly 0.55 (default), signal PASSES gate. Operator should be `<=` not `<`.

**Why it matters:** Ministral signal at 55% votes when it shouldn't. Parameter name and behavior inverted.

**Fix:** Line 2767, change to `if accuracy <= hold_threshold`. Or rename to `accuracy_floor`.

---

## [P3] Silent signal re-registration on module reload
**File:** portfolio/signal_registry.py:38-52  
**Bug:** `register_enhanced()` called at import (~60 times). If test/dynamic code calls twice for same signal (e.g., module reload), old entry silently overwritten. Correct behavior but surfaces test isolation bugs as silent overrides.

**Why it matters:** Low-severity. Test registering two versions silently uses last, masking setup bug.

**Fix:** Line 43, log debug: `if name in _ENHANCED_SIGNALS: logger.debug("Re-registering signal %s", name)`.

---

## SUMMARY
P1=3 P2=5 P3=1
