# Improvement Plan — Auto Session 2026-02-22 (Phase 2)

> Previous session: extracted signal_utils.py, added error logging, cache thread-safety,
> performance improvements, 142 new tests (933 → 1075). This plan covers remaining work.

## Priority 1: Bug Fixes

### B1. Division by zero — macro_regime.py
**File:** `portfolio/signals/macro_regime.py:93`
**Bug:** `pct_diff = (current_close - sma_val) / sma_val` — no guard for `sma_val == 0`
**Fix:** Add `if sma_val == 0: return "HOLD", {}` before the division
**Impact:** Low risk, isolated to one sub-indicator in macro_regime

### B2. Division by zero — fibonacci.py
**File:** `portfolio/signals/fibonacci.py` — `_near_level()` function
**Bug:** `abs(price - level) / abs(level)` — ZeroDivisionError when `level == 0`
**Fix:** Add `if level == 0: return False` guard
**Impact:** Low risk, isolated helper function

### B3. Confidence calculation inconsistency across signal modules
**Bug:** mean_reversion.py and momentum_factors.py calculate confidence as `winner / active_voters` (excludes HOLD), while trend.py, momentum.py, oscillators.py etc. use `winner / total` (includes HOLD). Same vote pattern → different confidence values.
**Fix:** Extract `majority_vote()` to `signal_utils.py` and standardize on excluding HOLD (active voters as denominator) — matches how consensus is calculated in signal_engine.
**Files:** 10 signal modules + signal_utils.py
**Impact:** Medium — changes confidence values reported by enhanced signals. Does not change BUY/SELL/HOLD decisions, only the confidence number.

## Priority 2: User-Requested Features

### F1. Big bet cooldown: 4h → 10min + stale notifications
**File:** `portfolio/bigbet.py`
**Current:** `cooldown_hours` defaults to 4 in config.json and bigbet.py
**Changes:**
1. Change default cooldown from 4h to 10min (0.167h or use `cooldown_minutes` key)
2. Track when a big bet condition set becomes active (window opens)
3. Send notification when conditions are no longer met (window closes / bet goes stale)
4. State tracking: store active conditions per ticker, compare each cycle
**Impact:** bigbet.py only, config.json default change. No other modules affected.

### F2. Dashboard: Layer 2 decisions history with filtering
**Files:** `dashboard/app.py`, `dashboard/static/index.html` (or new JS)
**Current:** Dashboard has no Layer 2 decision view. Journal entries (`layer2_journal.jsonl`) contain all decisions but aren't exposed.
**Changes:**
1. New API endpoint: `GET /api/decisions?limit=50&ticker=BTC-USD&action=BUY`
2. Reads `layer2_journal.jsonl`, supports filtering by ticker, action, date range
3. Frontend: scrollable log table with filter dropdowns
**Impact:** Dashboard only, read-only, no risk to core system.

### F3. LoRA accuracy assessment
**Action:** Research only — check accuracy data for LoRA vs base Ministral.
**Current state:** Custom LoRA disabled (20.9% accuracy, 97% SELL bias). Original CryptoTrader-LM LoRA still active via ministral_signal.py.
**Files to check:** accuracy data via `--accuracy`, `data/ab_test_log.jsonl`
**Deliverable:** Summary of LoRA vs Ministral accuracy in SYSTEM_OVERVIEW.md. Recommendation for next steps.

## Priority 3: Code Quality

### Q1. Extract majority_vote to signal_utils.py
**Part of B3** — standardize and deduplicate voting logic across 10 modules.
**Estimated savings:** ~150 lines of duplication removed.

### Q2. Dashboard _read_jsonl loads entire file
**File:** `dashboard/app.py:26-37`
**Bug:** `path.read_text().splitlines()` loads entire JSONL into memory (same issue fixed in journal.py).
**Fix:** Stream line-by-line, collect only the last N entries using a deque.
**Impact:** Dashboard only, performance improvement for large log files.

## Execution Order

### Batch 1: Bug fixes (B1, B2)
- Fix division by zero in macro_regime.py and fibonacci.py
- Add tests for the edge cases
- Files: macro_regime.py, fibonacci.py, test_signals_macro_regime.py, test_signal_utils.py

### Batch 2: Majority vote extraction + confidence fix (B3, Q1)
- Add `majority_vote()` to signal_utils.py
- Update 10 signal modules to use it
- Standardize confidence calculation
- Files: signal_utils.py, 10 signal modules, test_signal_utils.py

### Batch 3: Big bet improvements (F1)
- Change cooldown to 10min default
- Add stale bet tracking and notifications
- Files: bigbet.py, tests/test_bigbet.py

### Batch 4: Dashboard decisions endpoint (F2, Q2)
- Add /api/decisions endpoint with filtering
- Fix _read_jsonl streaming
- Add frontend decisions panel
- Files: dashboard/app.py, dashboard/static/

### Batch 5: LoRA assessment (F3)
- Run accuracy report, analyze data
- Document findings in SYSTEM_OVERVIEW.md
