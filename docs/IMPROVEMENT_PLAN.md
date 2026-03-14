# Improvement Plan

Updated: 2026-03-14
Branch: improve/auto-session-2026-03-14

Previous sessions: 2026-03-05 through 2026-03-13.

## Session Plan (2026-03-14)

### Theme: Complete IO Safety Sweep

Commit `8246657` ("fix(io-safety): replace raw json.loads with load_json and
atomic_append_jsonl") addressed 3 files (health.py, outcome_tracker.py,
reporting.py). This session completes the sweep across all remaining portfolio
modules — 37 raw `json.loads(path.read_text())` calls and 3 non-atomic file
writes remain.

### 1) Bugs & Problems Found

#### BUG-47 (P2): 37 raw `json.loads(path.read_text())` in portfolio/ modules — TOCTOU risk

- **Files** (20 modules):
  - `accuracy_stats.py:357,373,387` — cache reads
  - `alpha_vantage.py:44` — cache read
  - `analyze.py:75,265,304,363,613` — portfolio/summary/config reads
  - `autonomous.py:676` — compact summary read
  - `avanza_client.py:41` — config read
  - `avanza_orders.py:35` — pending orders read
  - `avanza_session.py:53,83` — session state reads
  - `avanza_tracker.py:37` — config read
  - `bigbet.py:40,91` — state and summary reads
  - `daily_digest.py:204` — bold state read
  - `focus_analysis.py:161,162` — summary and config reads
  - `forecast_signal.py:251` — summary read
  - `iskbets.py:45,75,264` — config, state, summary reads
  - `journal.py:183,432,445,452` — portfolio/config/summary reads
  - `main.py:461,696,697,698` — config and reporting reads
  - `onchain_data.py:66` — cache read
  - `perception_gate.py:96` — config read
  - `prophecy.py:65` — beliefs read
  - `telegram_notifications.py:126` — bold state read
- **Issue**: All use the `json.loads(path.read_text())` pattern which:
  1. Can read partial content during concurrent atomic rewrites (TOCTOU)
  2. Crashes on empty files (JSONDecodeError) instead of returning a default
  3. Duplicates error handling that `load_json()` already provides
- **Fix**: Replace with `load_json(path, default=...)` from `file_utils`.
  Each call needs the right default value (usually `{}` or `None`).
- **Impact**: Medium. Prevents crash-on-corrupt-file and aligns with the
  established TOCTOU-safe pattern.

#### BUG-48 (P2): 3 non-atomic JSON/JSONL writes — corruption risk on crash

- **Files**:
  - `prophecy.py:78` — `save_beliefs()` uses `open("w")` + `json.dump()`
  - `signal_history.py:49` — `_save_history()` uses `open("w")` for JSONL rewrite
  - `forecast_accuracy.py:339` — `_write_predictions()` uses `open("w")` for JSONL rewrite
- **Issue**: If the process crashes or is killed mid-write, the file is left
  partially written (truncated or empty), losing all data.
- **Fix**: Use `atomic_write_json()` for prophecy.py. For the JSONL writers,
  use a tempfile-then-rename pattern (new `atomic_write_jsonl()` helper).
- **Impact**: Medium. prophecy.json is read every Layer 2 invocation.

#### BUG-49 (P2): Manual JSONL reading loops instead of `load_jsonl()`

- **Files**:
  - `analyze.py:53-65` — `_load_journal_for_ticker()` manual JSONL parse
  - `signal_history.py:28-39` — `_load_history()` manual JSONL parse
  - `accuracy_stats.py:37-44` — `load_entries()` JSONL fallback manual parse
  - `equity_curve.py:48-56` — JSONL manual parse
  - `focus_analysis.py:87-95` — JSONL manual parse
- **Issue**: Duplicates the JSONL parsing logic that `load_jsonl()` handles.
- **Fix**: Replace with `load_jsonl()` where possible.
- **Impact**: Low. DRY improvement and consistent error handling.

#### BUG-50 (P3): `signal_history._save_history()` loads then rewrites entire file

- **File**: `portfolio/signal_history.py:42-51,67-95`
- **Issue**: Non-atomic full-file rewrite. Crash during write loses all history.
- **Fix**: Use atomic write pattern from REF-8.
- **Impact**: Low.

### 2) Refactoring

#### REF-8: Add `atomic_write_jsonl()` helper to file_utils.py

- **File**: `portfolio/file_utils.py`
- **Issue**: `prune_jsonl()` has the tempfile pattern for JSONL rewrite but it's
  embedded in pruning logic. Extract reusable `atomic_write_jsonl(path, entries)`.
- **Impact**: Enables safe JSONL rewrites for BUG-48 and BUG-50.

### 3) Test Coverage

#### TEST-11: Verify IO safety of replaced calls

- **File**: `tests/test_io_safety_sweep.py`
- Test that each modified module imports `load_json`/`load_jsonl` and handles
  missing/corrupt files gracefully.

### 4) Dependency/Ordering

#### Batch 1: Add `atomic_write_jsonl()` helper (REF-8)
- Files: `portfolio/file_utils.py`
- Zero-risk — additive only

#### Batch 2: IO safety sweep — core modules (BUG-47 partial, BUG-48, BUG-49, BUG-50)
- Files: `portfolio/accuracy_stats.py`, `portfolio/bigbet.py`,
  `portfolio/prophecy.py`, `portfolio/autonomous.py`,
  `portfolio/daily_digest.py`, `portfolio/forecast_signal.py`,
  `portfolio/onchain_data.py`, `portfolio/perception_gate.py`,
  `portfolio/telegram_notifications.py`, `portfolio/signal_history.py`,
  `portfolio/forecast_accuracy.py`, `portfolio/alpha_vantage.py`
- ~20 replacements

#### Batch 3: IO safety sweep — Avanza + analysis + remaining modules (BUG-47 remainder)
- Files: `portfolio/analyze.py`, `portfolio/avanza_client.py`,
  `portfolio/avanza_orders.py`, `portfolio/avanza_session.py`,
  `portfolio/avanza_tracker.py`, `portfolio/focus_analysis.py`,
  `portfolio/iskbets.py`, `portfolio/journal.py`, `portfolio/main.py`,
  `portfolio/equity_curve.py`, `portfolio/cumulative_tracker.py`,
  `portfolio/local_llm_report.py`
- ~20 replacements

#### Batch 4: Tests (TEST-11)
- Files: `tests/test_io_safety_sweep.py`

## Summary

| Category | Count | Items |
|----------|-------|-------|
| Bugs | 4 | BUG-47 through BUG-50 |
| Refactoring | 1 | REF-8 |
| Test Coverage | 1 | TEST-11 |
| Batches | 4 | Batch 1 helper → Batch 2+3 sweep → Batch 4 tests |
