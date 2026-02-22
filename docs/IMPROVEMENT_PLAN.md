# Improvement Plan — Auto Session 2026-02-22

## Priority 1: Bugs & Consistency Issues

### B1. Remaining inline atomic writes (3 files)
**Files:** `signal_engine.py:57-62`, `accuracy_stats.py:18-32`, `outcome_tracker.py:354`
**Problem:** The `file_utils.atomic_write_json` was extracted in v2 audit but 3 modules still have inline copies. This is a consistency bug — if the atomic write pattern needs to change, these copies won't get updated.
**Fix:** Replace all 3 with `from portfolio.file_utils import atomic_write_json`.
**Impact:** Low risk. Pure refactor, same behavior.

### B2. `data_refresh.py` uses raw `requests.get` instead of `fetch_with_retry`
**File:** `data_refresh.py:28`
**Problem:** All other API-calling modules use `fetch_with_retry` for retry logic. This one doesn't.
**Fix:** Import and use `fetch_with_retry`.
**Impact:** Low risk. Only affects the manual `--refresh` CLI command.

### B3. Stale doc claims in system-design.md
**File:** `docs/system-design.md:843-845`
**Problem:** Says "test_digest.py — 0 tests" and "Trigger system — No unit tests" — both are wrong. Also says "no proactive health monitoring" (line 756) which contradicts `health.py`'s `check_agent_silence()`.
**Fix:** Correct these claims.
**Impact:** None (doc-only).

## Priority 2: Dead Code Removal

### D1. Remove never-imported modules
**Files:** `portfolio/collect.py`, `portfolio/stats.py`, `portfolio/social_sentiment.py`, `portfolio/avanza_watch.py`
**Evidence:** Grep for all import patterns across entire codebase — zero hits for each.
**Fix:** Delete these files. If any are used by standalone scripts, check `scripts/` first.
**Impact:** Must verify no external script references them. Check `scripts/*.py` imports.

### D2. Remove stale data/ Python files
**Files:** `data/extract_summary.py`, `data/layer2_action.py`, `data/layer2_exec.py`, `data/layer2_invoke.py`
**Evidence:** These appear to be agent-generated temp files, not part of the module system.
**Fix:** Delete after confirming they're not referenced.
**Impact:** Zero — these are runtime artifacts.

## Priority 3: Architecture Improvements

### A1. Consolidate `_atomic_write_json` references
Part of B1 above. After fixing, grep should show zero `mkstemp` calls outside `file_utils.py`.

### A2. Make `signal_engine._set_prev_sentiment` use file_utils
**File:** `signal_engine.py:57-62`
**Current:** Inline `tempfile.mkstemp` + `os.replace` inside a try/except that silently swallows all errors.
**Fix:** `from portfolio.file_utils import atomic_write_json` + use it.
**Concern:** The silent `except` — consider logging the error.

### A3. Unused `requests` imports
**Files:** `funding_rate.py:2`, `fear_greed.py:1`, `macro_context.py:6`, `outcome_tracker.py:6`, `telegram_poller.py:11`
**Problem:** These import `requests` but use `fetch_with_retry` for all actual calls. The `requests` import is dead.
**Fix:** Remove `import requests` from each. Some may need it for exception classes (`requests.exceptions.RequestException`) — check before removing.
**Impact:** Low. Removes unused imports.

## Priority 4: Test Improvements

### T1. Add tests for new iteration 2 modules that lack test depth
- `signal_registry.py` — has no dedicated test file
- `file_utils.py` — has no dedicated test file
**Fix:** Create `tests/test_signal_registry.py` and `tests/test_file_utils.py`.

### T2. Verify test stability
Run full suite, identify any flaky/time-dependent tests, fix or document them.

## Execution Batches

### Batch 1: Atomic write consolidation + dead import cleanup (B1 + A2 + A3)
Files: `signal_engine.py`, `accuracy_stats.py`, `outcome_tracker.py`, `funding_rate.py`, `fear_greed.py`, `macro_context.py`
Depends on: Nothing
Risk: Low

### Batch 2: Dead code removal (D1 + D2)
Files: `collect.py`, `stats.py`, `social_sentiment.py`, `avanza_watch.py`, `data/*.py` temp files
Depends on: Verify no external references
Risk: Low (but verify first)

### Batch 3: New tests (T1)
Files: `tests/test_signal_registry.py`, `tests/test_file_utils.py`
Depends on: Batch 1 (so the code being tested is clean)
Risk: Zero (additive only)

### Batch 4: Doc fixes (B3)
Files: `docs/system-design.md`
Depends on: All code changes done
Risk: Zero

### Batch 5: Final verification (T2)
Run full test suite, fix any issues, commit.
