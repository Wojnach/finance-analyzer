# Improvement Plan — Auto Session 2026-02-24 (Session 2)

> Deep dive following Session 1's bug fixes. Focus: breaking import bug,
> test coverage for critical untested modules, stale data safety, voting
> consistency across signal modules.
>
> **Status: ALL ITEMS COMPLETED** — see commits on `improve/auto-session-2026-02-24`.

## Priority 1: Critical Bugs — DONE

### B1. AGENT_TIMEOUT import breaks main.py ✅
**Commit:** `c98a0a3`
Removed stale `AGENT_TIMEOUT` re-export from `main.py`. Fixed docstring "25-signal" → "27-signal".

### B2. Stale data returned indefinitely on cache errors ✅
**Commit:** `c98a0a3`
Added `_MAX_STALE_FACTOR = 5` guard to `_cached()`. Returns `None` with warning when cached data exceeds 5x TTL.

### B3. Regime detection uses 1.0% EMA gap vs signal engine's 0.5% ✅
**Commit:** `c98a0a3`
Aligned `detect_regime()` to use `>= 0.5%` (was `> 1.0%`), matching signal_engine's EMA deadband.

## Priority 2: Architecture & Safety — DONE

### A1. heikin_ashi.py duplicate majority_vote ✅
**Commit:** `5e9c659`
Replaced local `_majority_vote()` with `signal_utils.majority_vote(count_hold=True)`.

### A2. Stale data missing timestamps ✅
**Commit:** `5e9c659`
Added `stale_since` ISO timestamp to preserved data in `reporting.py`.

### A3. triggered_consensus unbounded growth ✅
**Commit:** `5e9c659`
Added pruning in `_save_state()` for tickers no longer in current signals.

## Priority 3: Test Coverage — DONE

### T1. Trigger core tests ✅
**Commit:** `07f76ff` — 64 tests in `tests/test_trigger_core.py`

### T2. Market timing DST tests ✅
**Commit:** `07f76ff` — 100 tests in `tests/test_market_timing.py`

### T3. Shared state cache tests ✅
**Commit:** `07f76ff` — 49 tests in `tests/test_shared_state.py`

## Summary

| Metric | Before | After |
|--------|--------|-------|
| Tests passing | 1333 | 1546 |
| New test files | 0 | 3 |
| Bugs fixed | 0 | 3 (1 critical) |
| Files modified | 0 | 6 production + 3 test |
