# Improvement Plan — Auto Session 2026-02-24 (Session 2)

> Deep dive following Session 1's bug fixes. Focus: breaking import bug,
> test coverage for critical untested modules, stale data safety, voting
> consistency across signal modules.

## Priority 1: Critical Bugs

### B1. AGENT_TIMEOUT import breaks main.py (CRITICAL — live system affected)
**File:** `portfolio/main.py` line 104
**Bug:** Previous session removed `AGENT_TIMEOUT` from `agent_invocation.py` (replaced with per-invocation `_agent_timeout`) but forgot to remove it from `main.py` re-exports. Any code doing `from portfolio.main import ...` crashes with `ImportError: cannot import name 'AGENT_TIMEOUT'`.
**Impact:** Tests crash on import. Live loop may work only because it doesn't re-import main.py after startup, but any restart will fail. 4 test files cannot even load.
**Fix:** Remove `AGENT_TIMEOUT` from the import on line 104. Also fix the stale "25-signal" docstring on line 10.

### B2. Stale data returned indefinitely on cache errors
**File:** `portfolio/shared_state.py` lines 40-46
**Bug:** When `func()` throws in `_cached()`, it returns old cached data with a 60s retry cooldown. If the function keeps failing (e.g., API down for hours), stale data from hours ago is used for trading decisions with no warning or expiration.
**Impact:** Trading signals computed from hours-old prices during API outages. No log entry indicates staleness.
**Fix:** Add a max staleness guard (e.g., 5x TTL). If cached data is older than that, return None instead of stale data. Log a warning with the age.

### B3. Regime detection uses 1.0% EMA gap vs signal engine's 0.5%
**File:** `portfolio/indicators.py` line 109
**Bug:** `detect_regime()` uses `> 1.0%` EMA gap to classify trending vs ranging. `signal_engine.py` line 163 uses `>= 0.5%` for the EMA signal. A ticker with 0.7% EMA gap would be classified as "ranging" for regime purposes but have an active EMA signal. Regime weights would then penalize the EMA signal (0.5x in "ranging") that just fired.
**Impact:** EMA signals self-suppress in the 0.5-1.0% gap zone. Ranging regime penalizes the very signal that triggered.
**Fix:** Align `detect_regime()` to use the same 0.5% threshold as the EMA signal.

## Priority 2: Architecture & Safety

### A1. heikin_ashi.py reimplements majority_vote instead of importing
**File:** `portfolio/signals/heikin_ashi.py`
**Bug:** Contains a local `_majority_vote()` function (~25 lines) that duplicates `signal_utils.majority_vote()`. Inconsistent behavior if the canonical version is updated.
**Impact:** Maintenance burden; potential vote-counting divergence.
**Fix:** Replace local `_majority_vote()` with import from `signal_utils`.

### A2. Stale data preserved indefinitely in agent_summary.json
**File:** `portfolio/reporting.py` lines 210-223
**Bug:** When instruments are missing from the current cycle (e.g., stocks off-hours), their data from the previous cycle is preserved with `"stale": True` but no timestamp indicating HOW stale. Data from 24+ hours ago looks the same as 5-minute-old data.
**Impact:** Layer 2 may make decisions based on very old data without knowing its age.
**Fix:** Add `"stale_since": ISO-8601` timestamp when marking stale. Layer 2 can then judge.

### A3. triggered_consensus dict grows unbounded
**File:** `portfolio/trigger.py` (inside `_load_state()` / `check_triggers()`)
**Bug:** The `triggered_consensus` dict in `trigger_state.json` records every ticker that ever reached BUY/SELL. Tickers are never removed. Over months this grows without bound.
**Impact:** Trigger state file grows, minor memory/I/O cost. Functional impact is low since checks are dict lookups.
**Fix:** Prune entries older than 7 days during `_save_state()`.

## Priority 3: Test Coverage

### T1. Add tests for trigger detection logic
**File:** `tests/test_trigger_core.py` (new)
**Why:** `trigger.py` has 286 lines of logic handling 7 trigger types, tier classification, and state management, but `test_trigger_edge_cases.py` only covers edge cases. Core trigger logic (each trigger type firing correctly, cooldowns, sustained checks) needs dedicated tests.

### T2. Add tests for market_timing DST calculations
**File:** `tests/test_market_timing.py` (new)
**Why:** DST logic uses complex date arithmetic (2nd Sunday of March, 1st Sunday of November). No tests exist. Edge cases around DST transitions could cause wrong market hours.

### T3. Add tests for shared_state cache behavior
**File:** `tests/test_shared_state.py` (new)
**Why:** `_cached()` has error recovery, stale data return, retry cooldown, and eviction logic — none tested. The B2 fix above needs tests to verify correctness.

## Priority 4: Refactoring

### R1. Remove stale AGENT_TIMEOUT references across codebase
**Files:** `portfolio/main.py`, any test files referencing `AGENT_TIMEOUT`
**Why:** Cleanup from previous session's incomplete migration.

### R2. Fix main.py docstring signal count
**File:** `portfolio/main.py` line 10
**Why:** Says "25-signal" but system has 27 signals.

## Execution Order

### Batch 1: Critical bug fixes (3 files)

| File | Change | Risk |
|------|--------|------|
| `portfolio/main.py` | B1: Remove AGENT_TIMEOUT import + R2: fix docstring | Low — removes broken import |
| `portfolio/shared_state.py` | B2: Add max staleness guard to _cached() | Medium — affects all cached data |
| `portfolio/indicators.py` | B3: Align regime EMA gap to 0.5% | Low — makes regime consistent with signals |

### Batch 2: Tests for Batch 1 changes (3 new files)

| File | Change | Risk |
|------|--------|------|
| `tests/test_shared_state.py` | T3: Cache behavior tests (fresh, stale, error, eviction) | None — new tests |
| `tests/test_market_timing.py` | T2: DST calculations, market state, agent window | None — new tests |
| `tests/test_trigger_core.py` | T1: Core trigger type tests | None — new tests |

### Batch 3: Architecture improvements (3 files)

| File | Change | Risk |
|------|--------|------|
| `portfolio/signals/heikin_ashi.py` | A1: Replace local _majority_vote with signal_utils import | Low — same logic |
| `portfolio/reporting.py` | A2: Add stale_since timestamp to preserved data | Very low — additive field |
| `portfolio/trigger.py` | A3: Prune old triggered_consensus entries | Very low — housekeeping |

**Total files modified:** 6 production + 3 new test files
**Estimated risk:** Low-Medium. Batch 1 fixes genuine bugs. Batch 2 adds safety nets. Batch 3 is defensive.
