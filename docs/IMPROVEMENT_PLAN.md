# Improvement Plan

Updated: 2026-03-11
Branch: worktree-improve-auto-session-2026-03-11

Previous sessions: 2026-03-05 (dashboard hardening), 2026-03-06 (CircuitBreaker, TTL cache, prune fix), 2026-03-07 (digest hardening, outcome tracker, disabled signals), 2026-03-08 (signal logging, file I/O safety, cache TTL, trigger hardening), 2026-03-09 (signal validation, confidence caps, candlestick/fibonacci/structure tests), 2026-03-10 (accuracy tolerance, signal failure tracking, ADX caching, TOCTOU fix, weighted consensus tests).

## Session Results (2026-03-11)

All 3 batches implemented. 5 bugs fixed (BUG-34 through BUG-38), 1 architecture improvement (ARCH-14 partial — high-traffic paths migrated), 2 test coverage gaps filled (TEST-7, TEST-8), 2 refactorings (REF-5, REF-6). 31 new tests (21 portfolio_mgr + 10 inversion cap). All 151 tests pass across modified files.

**Changes:**
- `portfolio/portfolio_mgr.py` — TOCTOU fix + state validation via `_validated_state()`
- `portfolio/health.py` — TOCTOU fix in `load_health()`
- `portfolio/file_utils.py` — TOCTOU fix in `load_jsonl()` and `prune_jsonl()`
- `portfolio/signal_engine.py` — Inversion weight cap at 0.75 + sentiment state `load_json()` migration
- `portfolio/digest.py` — Digest state `load_json()` migration
- `tests/test_portfolio_mgr_core.py` — 21 new tests for corruption, missing keys, type validation
- `tests/test_weighted_consensus.py` — 10 new tests for inversion weight cap behavior

## Session Results (2026-03-10 — previous)

All 3 batches implemented. 6 bugs fixed (BUG-28 through BUG-33), 2 architecture improvements (ARCH-12, ARCH-13), 3 test coverage gaps filled (TEST-4, TEST-5, TEST-6), 1 refactoring (REF-4). 193+ new tests.

## Session Plan (2026-03-11)

### 1) Bugs & Problems Found

#### BUG-34 (P1): `portfolio_mgr.py` TOCTOU race and no error handling in load functions

- **File**: `portfolio/portfolio_mgr.py:27-28, 41-42`
- **Issue**: `load_state()` and `load_bold_state()` use `if STATE_FILE.exists(): json.loads(STATE_FILE.read_text())`.
  This is the same TOCTOU pattern that BUG-30 fixed in `file_utils.load_json()`. Two problems:
  (1) Race condition: file can be deleted between `exists()` and `read_text()` → crash.
  (2) No JSON decode error handling: corrupted file → `json.JSONDecodeError` crash instead of graceful default.
  These are the most critical state files in the system (portfolio cash + holdings). A crash here kills the loop.
- **Fix**: Use `file_utils.load_json(path, default=None)` which handles FileNotFoundError and JSONDecodeError.
  Then merge loaded data with `_DEFAULT_STATE` to fill missing keys.
- **Impact**: High. Prevents loop crash on corrupted or concurrently-written portfolio state.

#### BUG-35 (P1): `portfolio_mgr.py` no validation on loaded state

- **File**: `portfolio/portfolio_mgr.py:27-32`
- **Issue**: After loading portfolio_state.json, the raw dict is returned without validation.
  If a required key is missing (`cash_sek`, `holdings`, `transactions`), downstream code crashes:
  - `reporting.py:78`: `state["cash_sek"]` → KeyError
  - `portfolio_value()`: `state.get("holdings", {})` → safe but `cash_sek` missing → wrong total
  - Layer 2 portfolio edits: writes back incomplete state, losing data
- **Fix**: After loading, merge with `_DEFAULT_STATE` using `{**_DEFAULT_STATE, **loaded}`.
  This fills missing keys with defaults while preserving existing values.
- **Impact**: High. Prevents cascade failures from corrupted state files.

#### BUG-36 (P1): `health.py` TOCTOU race in `load_health()`

- **File**: `portfolio/health.py:42-44`
- **Issue**: Same `if exists(): json.loads(read_text())` TOCTOU pattern. If health_state.json
  is deleted or being atomically rewritten between the check and the read, `read_text()` raises
  `FileNotFoundError`. The function catches `JSONDecodeError` and `OSError` but relies on `exists()`
  preventing `FileNotFoundError`.
- **Fix**: Remove `exists()` check, use try/except with `FileNotFoundError` in the catch chain.
- **Impact**: Medium. Health monitoring is less critical than portfolio state but a crash here
  still kills the loop.

#### BUG-37 (P2): `load_jsonl()` TOCTOU race

- **File**: `portfolio/file_utils.py:56`
- **Issue**: `load_jsonl()` uses `if not path.exists(): return []`. Same TOCTOU concern.
  File can be deleted between check and `open()`. Less critical than JSON loads because JSONL
  files are append-only, but still inconsistent with the `load_json()` fix from BUG-30.
- **Fix**: Remove `exists()` check, use try/except FileNotFoundError.
- **Impact**: Low. Consistency fix; JSONL files rarely disappear.

#### BUG-38 (P2): Inversion weight uncapped in `_weighted_consensus()` — single signal can dominate

- **File**: `portfolio/signal_engine.py:163`
- **Issue**: `weight = (1.0 - acc)` for inverted signals. A signal with 5% accuracy gets
  weight 0.95 after inversion. With regime multiplier 1.5x and activation normalization,
  this single inverted signal can reach effective weight ~1.4, while a genuine 60% accurate
  signal has weight ~0.6. One bad-then-inverted signal can overpower two good signals.
  The 20-sample minimum (line 158) means a signal with only 20 observations at 5% accuracy
  gets this extreme weight — the confidence in the 5% estimate is low with so few samples.
- **Fix**: Cap inverted weight at 0.75. This preserves the contrarian value (75% is still
  a strong weight) while preventing extreme outliers from dominating. Formula becomes:
  `weight = min(0.75, 1.0 - acc) if invert else acc`
- **Impact**: Medium. Prevents consensus distortion by single poorly-estimated inverted signals.
  Most inverted signals have accuracy 30-45% (inverted to 55-70%), so the cap only affects extremes.

### 2) Architecture Improvements

#### ARCH-14: Standardize JSON/JSONL loading across codebase

- **Files**: Multiple modules still use `exists()` + `read_text()` + manual `json.loads()`
  instead of `file_utils.load_json()`. Key examples:
  - `portfolio/digest.py:42` — digest state
  - `portfolio/accuracy_stats.py:355,371,385` — accuracy cache
  - `portfolio/avanza_orders.py:32` — pending orders
  - `portfolio/signal_engine.py:52-60` — sentiment state
  - `portfolio/bigbet.py:38` — bigbet state
  - `portfolio/iskbets.py:73` — iskbets state
- **Fix**: Migrate high-traffic paths to `load_json()`. Low-traffic paths (backup, analyze)
  can stay — they're developer tools, not production-critical.
- **Impact**: Consistency + safety. Each migration prevents one potential TOCTOU crash.
  Not all need to be done this session — prioritize by criticality.

### 3) Test Coverage Gaps

#### TEST-7: Tests for portfolio_mgr validation and TOCTOU safety

- **File**: `tests/test_portfolio_mgr_core.py`
- **Issue**: Existing tests cover load/save/value but don't test:
  - Corrupt JSON file (should return default, not crash)
  - Missing keys in loaded state (should fill defaults)
  - Concurrent file deletion (TOCTOU scenario)
- **Fix**: Add test cases to existing test file.

#### TEST-8: Tests for inversion weight cap

- **File**: `tests/test_weighted_consensus.py`
- **Issue**: The existing 67 weighted consensus tests don't cover the extreme inversion case
  (5-10% accuracy signal dominating consensus). Need tests that verify:
  - Weight is capped at 0.75 for inverted signals
  - Cap doesn't affect non-inverted signals
  - Cap doesn't affect signals with <20 samples (still use 0.5 default)
- **Fix**: Add test cases to existing test file.

### 4) Refactoring TODOs

#### REF-5: Migrate `digest.py` state loading to `load_json()`

- **File**: `portfolio/digest.py:42`
- **Issue**: `state = json.loads(path.read_text(...)) if path.exists() else {}`
  Should use `load_json(path, default={})`.
- **Fix**: One-line change. Safe, no behavioral difference.

#### REF-6: Migrate `signal_engine.py` sentiment state loading to `load_json()`

- **File**: `portfolio/signal_engine.py:52-60`
- **Issue**: `_load_prev_sentiments()` manually reads JSON files with `exists()` + `json.loads()`.
  Should use `load_json()` for consistency.
- **Fix**: Replace manual reads with `load_json()` calls.

### 5) Dependency/Ordering

#### Batch 1: Portfolio & Health Safety (BUG-34, BUG-35, BUG-36, BUG-37, TEST-7)
- Files: `portfolio/portfolio_mgr.py`, `portfolio/health.py`, `portfolio/file_utils.py`
- Tests: Add corruption/TOCTOU tests to existing test files
- Must be done first — these are the most critical data paths

#### Batch 2: Signal Quality (BUG-38, TEST-8)
- Files: `portfolio/signal_engine.py`, `tests/test_weighted_consensus.py`
- Tests: Add inversion cap tests
- Independent of Batch 1

#### Batch 3: Consistency Cleanup (ARCH-14, REF-5, REF-6)
- Files: `portfolio/digest.py`, `portfolio/signal_engine.py`, others
- Migrate remaining high-traffic `exists()` patterns to `load_json()`
- Independent of Batches 1 and 2

## Summary

| Category | Count | Items |
|----------|-------|-------|
| Bugs | 5 | BUG-34 through BUG-38 |
| Architecture | 1 | ARCH-14 |
| Test Coverage | 2 | TEST-7, TEST-8 |
| Refactoring | 2 | REF-5, REF-6 |
| Batches | 3 | Batch 1 critical-first, Batch 2+3 independent |
