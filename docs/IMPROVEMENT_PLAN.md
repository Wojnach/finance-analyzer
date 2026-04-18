# Improvement Plan — Auto Session 2026-04-18

Based on deep exploration of signal_engine.py (3000+ lines), accuracy_stats.py,
ic_computation.py, and research data from the 2026-04-17 after-hours session.

## Priority Order

### Batch 1: IC-Based Weight Multiplier (HIGHEST IMPACT)

**Problem:** Current weighting uses directional accuracy as primary signal weight.
This misses return magnitude prediction. Phantom performers (calendar 58.9% acc,
IC=0.000; econ_calendar 62.6% acc, IC=0.000) get high weight despite zero
predictive power for move size. Genuinely good signals (ministral IC=0.094,
momentum IC=0.063) are underweighted relative to their true value.

**Files:** `portfolio/signal_engine.py` (insert after line 1519 in `_weighted_consensus()`)

**Changes:**
1. Add `_get_ic_data(horizon)` — lazy-cached IC loader using existing `ic_computation.py`
2. Add `_compute_ic_mult(ic, icir, samples)` — clamps to [0.6, 1.5]
3. Apply IC multiplier after accuracy weight, before regime mult
4. Add zero-IC penalty: signals with |IC| < 0.01 and samples > 500 get 0.85x
5. Pass `horizon` and `ticker` to IC lookup for per-ticker IC (Phase 2 data available)

**Impact:**
- ministral IC=0.094 → 1.19x boost
- momentum IC=0.063 → 1.13x boost  
- calendar IC=0.000 → 0.85x penalty (zero-IC)
- econ_calendar IC=0.000 → 0.85x penalty

**Risk:** LOW — IC cache already exists (1h TTL), multiplicative factor won't break
existing consensus logic.

**Tests:** `tests/test_ic_weighting.py` — tests for ic_mult math, stability filter,
integration with `_weighted_consensus()`.

---

### Batch 2: Fix Dynamic Correlation + Add Orphaned Signals (HIGH IMPACT)

**Problem 1:** Dynamic correlation is dead code. Pearson correlation on vote encoding
(BUY=1, HOLD=0, SELL=-1) is diluted by 70-90% HOLD dominance. Max observed Pearson
r=0.538 (ema↔trend), but these signals agree 100% on non-HOLD votes. The 0.7
threshold is unreachable → always falls back to static groups.

**Problem 2:** 10 signal pairs have 90-100% agreement but aren't in any static group:
- fear_greed↔calendar (100%, 501 sam)
- fear_greed↔funding (100%, 543 sam)
- news_event↔econ_calendar (100%, 714 sam)
- funding↔calendar (100%, 104 sam)
- fear_greed↔claude_fundamental (92.7%, 1241 sam)

**Files:** `portfolio/signal_engine.py`

**Changes:**
1. Replace Pearson correlation in `_compute_dynamic_correlation_groups()` with
   agreement rate (only counting non-HOLD pairs)
2. Change threshold from 0.7 (Pearson) to 0.85 (agreement rate)
3. Add orphaned signals to static groups:
   - Expand `macro_external` → `{"fear_greed", "sentiment", "news_event", "calendar",
     "econ_calendar", "funding"}` (6 members)
   - Set penalty to 0.15x (like momentum_cluster) since 6 members is large

**Risk:** MEDIUM — changing correlation groups affects vote weights for all signals.
Static group changes are safe (additive). Dynamic correlation fix needs careful testing.

---

### Batch 3: Disabled Signal Rescue (MEDIUM IMPACT)

**Problem:** Global `DISABLED_SIGNALS` hides per-ticker value.
ML on ETH-USD: 55.1% at 3h (1206 samples) — genuine edge hidden by BTC's 26.4%.

**Files:** `portfolio/signal_engine.py`

**Changes:**
1. Add `_DISABLED_SIGNAL_OVERRIDES` set with (signal, ticker) tuples
2. In `_weighted_consensus()`, check overrides before applying DISABLED_SIGNALS gate
3. Override: `("ml", "ETH-USD")` — 55.1% at 3h with 1206 samples

**Risk:** LOW — signals already computed, just unblocked. Accuracy gate auto-protects.

---

### Batch 4: Confidence Calibration Compression (MEDIUM IMPACT)

**Problem:** Confidence is massively overconfident above 60%:
- 60-69% conf → 49% actual (16pp gap)
- 80-89% conf → 53% actual (32pp gap)

**Files:** `portfolio/signal_engine.py` (in `apply_confidence_penalties`)

**Changes:**
1. Add Stage 7: Confidence Compression after Stage 6
2. Compress: `conf = 0.55 + (conf - 0.55) * 0.3` for conf > 0.55
3. Maps: 60% → 56.5%, 70% → 59.5%, 80% → 62.5%

**Risk:** LOW — only reduces confidence, never increases.

## Dependency Order

Batch 1 → Batch 2 → Batch 3 → Batch 4 (sequential, each committed separately)

## Deferred

- Contrarian signal inversion (ml IC=-0.321): needs backtesting
- Regime-conditioned per-signal weights: regime detection has lag
- credit_spread_risk re-enable: investigate why disabled first
- Per-ticker IC weighting: Phase 2, needs more validation
