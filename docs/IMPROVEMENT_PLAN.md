# Improvement Plan — Auto Session 2026-02-22

## Priority 1: Bugs & Consistency Issues

### B1. Remaining inline atomic writes (3 files) -- DONE
**Files:** `signal_engine.py:57-62`, `accuracy_stats.py:18-32`, `outcome_tracker.py:354`
**Fix:** Replaced signal_engine and accuracy_stats with `from portfolio.file_utils import atomic_write_json`. outcome_tracker kept as-is (JSONL format, different pattern).

### B2. `data_refresh.py` uses raw `requests.get` -- DONE
**Fix:** Replaced with `fetch_with_retry`.

### B3. Stale doc claims in system-design.md -- DONE
**Fix:** Corrected test_digest.py claim, trigger system claim, and health monitoring claim.

## Priority 2: Dead Code Removal

### D1. Remove never-imported modules -- PARTIALLY DONE
**Deleted:** `collect.py`, `avanza_watch.py` (confirmed zero imports).
**Kept:** `stats.py` (used by `digest.py` via lazy import), `social_sentiment.py` (used by `signal_engine.py` via lazy import). Initial grep missed these because they're inside function bodies.

### D2. Remove stale data/ Python files -- N/A
Files don't exist in worktree (already cleaned up or runtime-only artifacts).

## Priority 3: Architecture Improvements

### A1. Consolidate atomic writes -- DONE (part of B1)

### A2. signal_engine._set_prev_sentiment -- DONE
Now uses `file_utils.atomic_write_json`.

### A3. Unused `requests` imports -- DONE
Removed from: `funding_rate.py`, `fear_greed.py`, `macro_context.py`, `outcome_tracker.py`, `telegram_poller.py`.

## Priority 4: Test Improvements

### T1. Add tests for new modules -- DONE
- `tests/test_file_utils.py` — 14 tests
- `tests/test_signal_registry.py` — 17 tests

### T2. Verify test stability -- DONE
933 tests passing. No flaky tests identified.

## Bug Found During Testing

### file_utils.py mkdir missing parents=True -- FIXED
`atomic_write_json` used `path.parent.mkdir(exist_ok=True)` which fails for nested dirs. Fixed to `mkdir(parents=True, exist_ok=True)`. Discovered by test_creates_parent_dirs.

## Final Results

| Metric | Before | After |
|--------|--------|-------|
| Tests passing | 902 | 933 (+31) |
| Dead imports removed | 0 | 6 (`requests` from 5 files + data_refresh converted) |
| Dead modules removed | 0 | 2 (collect.py, avanza_watch.py) |
| Inline atomic writes | 3 | 1 (outcome_tracker JSONL intentionally kept) |
| Bugs fixed | 0 | 1 (file_utils mkdir parents) |
| Doc errors fixed | 0 | 3 (system-design.md) |
