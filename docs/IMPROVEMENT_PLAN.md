# Improvement Plan — Auto Session 2026-02-22 (Phase 3)

> Based on deep reading of all ~90 Python files, 1101 passing tests.
> Prior sessions completed: signal_utils extraction, thread-safe cache writes,
> DB connection reuse, cached accuracy, kline dedup, dashboard validation dedup,
> structured logging, bigbet improvements. This plan covers NEW work.

## Priority 1: Bugs & Correctness

### B1. Triple yfinance rate limiter (not coordinating)
**Files:** `shared_state.py:85`, `macro_context.py:18`, `outcome_tracker.py:11`
**Bug:** Three separate `_RateLimiter(30, "yfinance")` instances. Each allows
30 req/min independently, so yfinance could be hit at 90 req/min (3×30).
**Fix:** Delete local instances in macro_context.py and outcome_tracker.py,
import from shared_state.py instead.
**Impact:** Prevents yfinance rate limit violations.

### B2. `_crash_alert()` opens config.json without encoding
**File:** `main.py:305`
**Bug:** `json.load(open(config_path))` — no `encoding="utf-8"`. On Windows,
defaults to system locale, could fail on non-ASCII config values.
**Fix:** Add `encoding="utf-8"` to the open() call.
**Impact:** Trivial fix, prevents crash-handler crash.

### B3. `best_worst_signals()` redundantly calls `signal_accuracy()`
**File:** `accuracy_stats.py:153-164`, called from `reporting.py:161`
**Bug:** `reporting.py` computes `sig_acc = signal_accuracy("1d")` on line ~157,
then calls `best_worst_signals("1d")` which internally calls `signal_accuracy()`
again. Loads and filters the entire signal log twice.
**Fix:** Add optional `acc=None` parameter; if provided, skip internal call.
Update reporting.py to pass pre-computed data.
**Impact:** Avoids redundant full-log scan per cycle.

### B4. Unused `import sys` in analyze.py
**File:** `analyze.py:11`
**Bug:** `import sys` is never used.
**Fix:** Remove the import.
**Impact:** Trivial cleanup.

## Priority 2: Architecture & Deduplication

### A1. Duplicate JSON/JSONL loading helpers (7+ copies)
**Files:** kelly_sizing.py, regime_alerts.py, risk_management.py, stats.py,
weekly_digest.py, dashboard/app.py
**Bug:** ~230 lines of identical `_load_json()` / `_load_jsonl()` functions.
**Fix:** Add `load_json()` and `load_jsonl()` to `file_utils.py`. Update all
modules to import from there.
**Impact:** Reduces duplication, ensures consistent encoding/error handling.

### A2. Binance/Alpaca API URL duplication (4+ files)
**Files:** data_collector.py, iskbets.py, macro_context.py, ml_signal.py,
outcome_tracker.py, data_refresh.py, funding_rate.py
**Bug:** `BINANCE_BASE`, `BINANCE_FAPI_BASE`, `ALPACA_BASE` defined
independently in multiple files.
**Fix:** Define canonical URL constants in `api_utils.py` (already has
config/header helpers). Update all modules to import from there.
**Impact:** Single source of truth for API endpoints.

## Priority 3: Performance

### P1. `copy.deepcopy` in reporting.py
**File:** `reporting.py:247`
**Bug:** `copy.deepcopy(summary)` copies the entire agent_summary dict
(30+ tickers × 7 timeframes). Only top-level fields are modified.
**Fix:** Shallow copy the dict, deep copy only the `signals` sub-dict
(which gets mutated).
**Impact:** Reduces CPU/memory cost per reporting cycle.

### P2. `backfill_outcomes()` loads entire JSONL into memory
**File:** `outcome_tracker.py:262-267`
**Bug:** Reads all entries into a list before processing. File grows
monotonically.
**Fix:** Stream entries and process in-place, or use SQLite query
(SignalDB already has the data).
**Impact:** Reduces memory spikes during daily backfill.

## Priority 4: Robustness

### R1. `_log_trigger()` non-atomic append
**File:** `agent_invocation.py:33`
**Bug:** Regular `open("a")` append. If two processes race, JSONL lines
could interleave and corrupt.
**Fix:** Use `file_utils.atomic_append_jsonl()` (to be added in A1).
**Impact:** Low probability (single writer normally), but cheap to fix.

## Execution Order

### Batch 1: Rate limiter + encoding + accuracy fix (B1, B2, B3, B4)
Files: shared_state.py, macro_context.py, outcome_tracker.py, main.py,
accuracy_stats.py, reporting.py, analyze.py
- Consolidate yfinance rate limiter to shared_state.py
- Fix encoding in _crash_alert
- Add pre-computed data param to best_worst_signals
- Remove unused import

### Batch 2: Deduplication (A1, A2)
Files: file_utils.py, kelly_sizing.py, regime_alerts.py, risk_management.py,
stats.py, weekly_digest.py, dashboard/app.py, api_utils.py, data_collector.py,
iskbets.py, macro_context.py, ml_signal.py, outcome_tracker.py
- Extract shared JSON helpers to file_utils.py
- Centralize API URLs in api_utils.py

### Batch 3: Performance + robustness (P1, P2, R1)
Files: reporting.py, outcome_tracker.py, agent_invocation.py, file_utils.py
- Optimize deepcopy in reporting
- Stream backfill_outcomes
- Add atomic JSONL append to file_utils, use in agent_invocation
