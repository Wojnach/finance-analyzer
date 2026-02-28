# Improvement Plan — Auto-Session 2026-02-28

## Priority: Critical Bugs > Architecture > Features > Polish

---

## 1. Bugs & Problems Found

### BUG-13: test_avanza_session.py imports nonexistent function
- **File:** `tests/test_avanza_session.py:15`
- **Severity:** HIGH (prevents test collection, masks 39 tests)
- **Issue:** Imports `create_requests_session` from `portfolio.avanza_session`, but that function doesn't exist.
- **Fix:** Remove the import and any tests that reference it, or stub the function.
- **Impact:** Test-only fix, no production code affected.

### BUG-14: reporting.py has 7 bare `except Exception: pass` blocks
- **File:** `portfolio/reporting.py` lines 143, 181, 198, 207, 246, 305, 365
- **Severity:** MEDIUM (silent data loss — Layer 2 operates without critical context)
- **Issue:** When optional modules fail (macro_context, accuracy_stats, alpha_vantage, futures_data, avanza), the exception is silently swallowed. Layer 2 doesn't know context is missing.
- **Fix:** Replace `pass` with `logger.warning(...)` including the module name and exc_info. Add a `_warnings` list to agent_summary that Layer 2 can check.
- **Impact:** Reporting only. No behavior change for working modules.

### BUG-15: shared_state.py stale cache allows 5x TTL fallback
- **File:** `portfolio/shared_state.py:24` (`_MAX_STALE_FACTOR = 5`)
- **Severity:** MEDIUM-LOW (stale F&G/sentiment data could influence signals)
- **Issue:** If a data source fails, cached data up to 5x the TTL is returned. For a 5-minute TTL, this means 25-minute-old data. For a 1-hour TTL (FX rate), 5 hours.
- **Fix:** Reduce to `_MAX_STALE_FACTOR = 3` and add a `_stale_warnings` counter that's surfaced in health state.
- **Impact:** Slightly more aggressive cache expiry. Functions returning `None` must be handled by callers (already the case).

### BUG-16: trigger.py triggered_consensus has lazy pruning
- **File:** `portfolio/trigger.py:56`
- **Severity:** LOW (memory leak, not functional)
- **Issue:** Orphaned entries for removed tickers accumulate until the dict exceeds current_tickers + 10. Buffer allows up to 10 orphaned entries indefinitely.
- **Fix:** Always prune entries not in `current_tickers` (remove the +10 buffer).
- **Impact:** trigger_state.json stays cleaner. No functional change.

### BUG-17: reporting.py debug-level exception logging (6 blocks)
- **File:** `portfolio/reporting.py` lines 271, 280, 295, 317, 326, 335
- **Severity:** LOW (exceptions logged at debug level, invisible in production)
- **Issue:** Trade guard, risk audit, portfolio metrics, probabilities, cumulative gains, and warrant portfolio failures log at `debug` level. Production log level is `INFO`, so these are invisible.
- **Fix:** Change `logger.debug(...)` to `logger.warning(...)` for these blocks.
- **Impact:** More visible logging only.

---

## 2. Architecture Improvements

### ARCH-1: Surface module failure warnings to Layer 2
- **Files:** `portfolio/reporting.py`
- **Why:** Layer 2 currently has no way to know when critical modules failed. If accuracy_stats fails, weighted consensus silently degrades to raw voting. This is the single largest source of potential hidden failures.
- **Change:** Add a `warnings` list to agent_summary.json (and compact variant). Populate it with module names that failed during report generation.
- **Impact:** reporting.py changes only. Layer 2 sees warnings in its context.
- **Depends on:** BUG-14

### ARCH-2: Reduce stale fallback aggressiveness
- **Files:** `portfolio/shared_state.py`
- **Why:** 5x stale factor is too generous for a system making financial decisions. A 25-minute-old Fear & Greed value could cause wrong signals.
- **Change:** Reduce `_MAX_STALE_FACTOR` from 5 to 3. Track stale-hit count in health state.
- **Impact:** `_cached()` callers. All callers already handle None returns.
- **Depends on:** BUG-15

---

## 3. Useful Features

### FEAT-1: Health endpoint for module status
- **Files:** `portfolio/health.py`, `portfolio/reporting.py`
- **Why:** Currently no way to check which modules failed in the last cycle. The dashboard health tab shows heartbeat but not per-module status.
- **Change:** Track `last_module_failures` dict in health_state.json. Report module names and timestamps of last failure. Surface in `/api/health`.
- **Impact:** health.py + reporting.py. Dashboard reads it passively.

---

## 4. Refactoring TODOs

### REF-5: Fix test_avanza_session.py collection error
- **Files:** `tests/test_avanza_session.py`
- **Why:** Prevents 39 tests from running. Masks potential regressions.
- **Change:** Remove or fix the `create_requests_session` import and associated tests.
- **Impact:** Test-only.
- **Depends on:** BUG-13

### REF-6: Standardize exception logging in reporting.py
- **Files:** `portfolio/reporting.py`
- **Why:** Inconsistent — some blocks use `pass`, some use `logger.debug()`, some use `logger.warning()`. Should be consistently `logger.warning()` with the module name.
- **Change:** All `except Exception` blocks in reporting.py get `logger.warning(f"[reporting] {module_name} failed", exc_info=True)`.
- **Impact:** Logging only.
- **Depends on:** BUG-14, BUG-17

### REF-7: Clean trigger state pruning
- **Files:** `portfolio/trigger.py`
- **Why:** The +10 buffer is unnecessary complexity. Just prune orphans every save.
- **Change:** Remove the `len(tc) > len(current_tickers) + 10` guard.
- **Impact:** Slightly more frequent dict operations on save. Negligible.
- **Depends on:** BUG-16

---

## 5. Dependency/Ordering — Implementation Batches

### Batch 1: Test infrastructure fix (1 file)
**Files:** `tests/test_avanza_session.py`
**Changes:** REF-5 (fix import error)
**Tests:** Run collection to verify 39 tests collected
**Risk:** None — test-only

### Batch 2: Reporting robustness (1 file)
**Files:** `portfolio/reporting.py`
**Changes:** BUG-14, BUG-17, REF-6, ARCH-1
**Tests:** Existing reporting tests + verify warnings list populated on module failure
**Risk:** Low — logging changes only affect reporting output format
**Depends on:** Batch 1 (clean test baseline)

### Batch 3: Cache and state cleanup (2 files)
**Files:** `portfolio/shared_state.py`, `portfolio/trigger.py`
**Changes:** BUG-15, BUG-16, ARCH-2, REF-7
**Tests:** Existing shared_state and trigger tests (142 tests)
**Risk:** Low — stale factor change could cause more None returns, but callers handle it
**Depends on:** Batch 2 (clean reporting)

### Batch 4: Health module status tracking (2 files)
**Files:** `portfolio/health.py`, `portfolio/reporting.py`
**Changes:** FEAT-1
**Tests:** Add tests for module failure tracking
**Risk:** Low — additive feature, no existing behavior changed
**Depends on:** Batch 2 (reporting warnings infrastructure)

---

## Items NOT Planned (Justified)

These were considered during exploration but intentionally excluded:

1. **Moving hardcoded thresholds to config.json** — Too broad, would touch 10+ files. The current values are well-tuned. Save for a dedicated configuration session.
2. **Concurrency safety for sentiment_state.json** — Layer 1 is single-threaded. The theoretical race condition requires concurrent L1/L2 writes, which doesn't happen in practice.
3. **Circuit breaker Telegram alerts** — Nice-to-have but would require notification infrastructure changes. The health module already detects staleness.
4. **Parallel ticker processing in main.py** — Sequential processing is a feature (predictable, debuggable). Parallelizing would add complexity for marginal speed gains on a 60s cycle.
5. **Calendar signal quorum inconsistency** — Quorum=2 is intentional design (documented). Changing it affects signal accuracy baselines.
6. **Sub-signal correlation across composite signals** — Would require a complete redesign of the voting architecture. The current system works well enough; accuracy tracking compensates.
7. **Equity curve Sharpe ratio for crypto (365d)** — Minor accuracy improvement. Not worth the complexity of dual calculation paths.
