# Improvement Plan — 2026-05-15

## Exploration Summary

4 parallel exploration agents + direct code reading covered:
- Signal pipeline (signal_engine.py 4206 lines, 53 signal modules, accuracy_stats, outcome_tracker)
- Data collection (data_collector, fear_greed, sentiment, futures_data, onchain, fx_rates, http_retry)
- Orchestration (main.py, agent_invocation, trigger, portfolio_mgr, risk_management, market_timing)
- Infrastructure (dashboard/auth, file_utils, health, shared_state, grid_fisher, telegram, metals_loop)

Findings prioritized by production impact. Cosmetic issues omitted.

---

## 1. Bugs & Problems Found

### B1 [P0] Portfolio backup rotation before write — crash data loss
**File:** `portfolio/portfolio_mgr.py:108-113`
**Bug:** `_save_state_to()` and `update_state()` call `_rotate_backups(path)` BEFORE
`_atomic_write_json(path, state)`. If write crashes, the original file was already rotated
to `.bak`. On restart, recovery loads the pre-trade `.bak` state — positions revert.
**Fix:** Move `_rotate_backups()` AFTER successful write.

### B2 [P0] ADX cache key uses `id(df)` — stale hits on address reuse
**File:** `portfolio/signal_engine.py:37`
**Bug:** `_adx_cache` uses `(id(df), len(df), last_close)` as key. Python `id()` returns
memory address, reused after GC. New DataFrame at same address gets stale ADX.
**Fix:** Replace `id(df)` with content-based discriminator.

### B3 [P1] Flip cooldown uses `time.time()` — NTP jump vulnerability
**File:** `portfolio/trigger.py:260-282`
**Bug:** Backwards NTP adjustment makes `now - last_flip_ts` negative, suppressing all flip
triggers for up to 30 minutes.
**Fix:** Guard against negative elapsed time (clock skew reset).

### B4 [P1] Alert budget not thread-safe
**File:** `portfolio/alert_budget.py`
**Bug:** `_sent_timestamps` and `_buffer` accessed from ThreadPoolExecutor threads without lock.
**Fix:** Add `threading.Lock()`.

### B5 [P1] Reporting module failure streak not thread-safe
**File:** `portfolio/reporting.py:34-68`
**Bug:** `_module_failure_streaks` dict mutated from ticker threads without synchronization.
**Fix:** Add `threading.Lock()`.

### B6 [P2] Failed timeframe silently dropped from multi-TF analysis
**File:** `portfolio/data_collector.py:296-299`
**Bug:** `compute_indicators()` returning None causes `_fetch_one_timeframe()` to return None,
silently dropping the timeframe from `raw_results`.
**Fix:** Return `(label, {"error": "..."})` instead of None.

---

## 2. Implementation Batches

### Batch 1: Safety & Thread Safety (4 files)
1. `portfolio/portfolio_mgr.py` — B1: backup after write
2. `portfolio/alert_budget.py` — B4: threading lock
3. `portfolio/reporting.py` — B5: threading lock on failure streaks
4. `portfolio/trigger.py` — B3: clock skew guard

### Batch 2: Signal Quality (2 files)
1. `portfolio/signal_engine.py` — B2: ADX cache key fix
2. `portfolio/data_collector.py` — B6: error tuple from failed timeframes

### Batch 3: Housekeeping
1. Resolve stale critical_errors.jsonl entries for disabled signals
2. Update SYSTEM_OVERVIEW.md

---

## 3. Previous Session (2026-05-14) — COMPLETE

6 bugs fixed, 15 tests added across 3 batches. See git log for details.
