# Improvement Plan

Updated: 2026-03-08
Branch: improve/auto-session-2026-03-08

Previous sessions: 2026-03-05 (dashboard hardening), 2026-03-06 (CircuitBreaker, TTL cache, prune fix), 2026-03-07 (digest hardening, outcome tracker, disabled signals).

## Session Results (2026-03-07 — previous)

All 4 batches implemented and committed. 28 new tests added (22 digest + 6 outcome_tracker).

| Batch | Items | Commit | Tests |
|-------|-------|--------|-------|
| 1 | BUG-10,11,13,14 + ARCH-6 | `90bc8a4` | 22 (test_4h_digest.py) |
| 2 | BUG-12 + ARCH-7 | `123949b` | 6 (test_outcome_tracker_backfill.py) |
| 3 | ARCH-8 | `d586c89` | 0 (1-line change, no new tests needed) |
| 4 | FEAT-2 | `d21bb51` | 3 (added to test_4h_digest.py) |

## Session Plan (2026-03-08)

### 1) Bugs & Problems Found

#### BUG-15 (P1): Silent ImportError passes in signal_engine.py

- **File**: `portfolio/signal_engine.py` (5 locations: lines ~359, 377, 427, 451, 516)
- **Issue**: Optional module imports (social_sentiment, fear_greed, sentiment, etc.) silently
  `pass` on ImportError. If a dependency is missing, the signal votes "HOLD" with no warning.
  The system operates with incomplete signal coverage without anyone knowing.
- **Fix**: Add `logger.debug()` to each ImportError catch so missing modules are visible in logs.
- **Impact**: Low risk — only adds logging, no behavioral change.

#### BUG-16 (P2): Sentiment state persistence logged at DEBUG level

- **File**: `portfolio/signal_engine.py:62-63, 79-80`
- **Issue**: Sentiment state loading/saving failures logged at DEBUG level. In a trading system,
  losing state persistence is at least a WARNING — it means sentiment hysteresis won't work
  across restarts.
- **Fix**: Upgrade `logger.debug()` to `logger.warning()` for sentiment persist failures.
- **Impact**: No behavioral change, just better visibility.

#### BUG-17 (P2): Silent None returns in _compute_adx() and compute_indicators()

- **File**: `portfolio/signal_engine.py:172-197` and `portfolio/indicators.py:10-12`
- **Issue**: Both functions return None silently on insufficient data or computation failure.
  Callers handle None correctly, but no logging makes debugging impossible when ADX or
  indicators aren't computed.
- **Fix**: Add `logger.debug()` for data insufficiency, `logger.warning()` for computation errors.
- **Impact**: No behavioral change, adds diagnostic logging.

#### BUG-18 (P2): Accuracy stats load failure degrades consensus silently

- **File**: `portfolio/signal_engine.py:681-682`
- **Issue**: If accuracy statistics can't be loaded, weighted consensus falls back to equal
  weights silently. Signals with 30% accuracy carry the same weight as 80% signals. The
  warning is logged but at a level that doesn't trigger alerts.
- **Fix**: Upgrade to `logger.error()` and ensure accuracy data fallback is explicit.
- **Impact**: No behavioral change, better monitoring.

#### BUG-19 (P2): JSONL malformed lines silently skipped

- **File**: `portfolio/file_utils.py:64-65`
- **Issue**: `load_jsonl()` silently skips lines with JSON decode errors. If 50 out of 1000
  lines are corrupt, data is lost without warning. For signal_log and journal files, missing
  entries means missing context.
- **Fix**: Add `logger.debug()` with filename and error details.
- **Impact**: No behavioral change, aids debugging corrupt files.

#### BUG-20 (P2): portfolio_value() has no type validation

- **File**: `portfolio/portfolio_mgr.py:51-56`
- **Issue**: If `shares`, `price`, or `fx_rate` is None or a string (from corrupted JSON),
  the multiplication crashes with TypeError. No validation at the boundary.
- **Fix**: Add type checks for `fx_rate`, `shares`, `price` with safe defaults and logging.
- **Impact**: Prevents crash on corrupt portfolio state; may mask underlying issues.

#### BUG-21 (P3): Cache eviction threshold mismatch in shared_state.py

- **File**: `portfolio/shared_state.py:34-49`
- **Issue**: Cache eviction uses hardcoded 3600s (1 hour) threshold regardless of individual
  TTLs. Entries with TTL=43200 (12h, like onchain data) get evicted after 1h. This wastes
  API calls by re-fetching data that should still be cached.
- **Fix**: Use per-entry TTL-aware expiry check instead of global 3600s threshold.
- **Impact**: Reduces unnecessary API calls; may slightly increase memory usage.

#### BUG-22 (P3): Trigger state not persisted on early Layer 2 crash

- **File**: `portfolio/trigger.py:84-85`
- **Issue**: When portfolio file parsing fails (KeyError, JSONDecodeError), the exception is
  caught and passed silently. If the trigger state was modified before the error, changes are
  lost. Also, trigger_state.json isn't persisted until the next successful check_triggers call.
- **Fix**: Add `logger.warning()` for parse failures. Ensure trigger state is saved even when
  downstream (agent invocation) fails.
- **Impact**: Better diagnostics; prevents stale trigger state on crash.

### 2) Architecture Improvements

#### ARCH-9: Fix signal count inconsistency in CLAUDE.md

- **File**: `CLAUDE.md` line 486
- **Issue**: Section header says "27 Signals (8 Core + 19 Enhanced Composite)" but the actual
  list goes to signal #30 (Forecast, Claude Fundamental, Futures Flow). This inconsistency
  confuses Layer 2 about how many signals exist.
- **Fix**: Update header to "30 Signals (8 Core Active + 3 Disabled + 19 Enhanced Composite)".
- **Impact**: Documentation-only. Layer 2 will correctly understand the signal landscape.

### 3) Useful Features

(No new features this session — focusing on reliability improvements.)

### 4) Refactoring TODOs

#### REF-1 (carried): DRY outcome_tracker price fetchers
#### REF-2 (carried): Align reflection.py and equity_curve.py trade matching

### 5) Dependency/Ordering

#### Batch 1: Signal Engine Logging (BUG-15, BUG-16, BUG-17, BUG-18)
- Files: `portfolio/signal_engine.py`, `portfolio/indicators.py`
- Tests: Add test for _compute_adx edge cases
- No dependencies on other batches

#### Batch 2: File I/O Safety (BUG-19, BUG-20)
- Files: `portfolio/file_utils.py`, `portfolio/portfolio_mgr.py`
- Tests: Add tests for malformed JSONL, corrupt portfolio state
- No dependencies on other batches

#### Batch 3: Cache & Trigger Hardening (BUG-21, BUG-22)
- Files: `portfolio/shared_state.py`, `portfolio/trigger.py`
- Tests: Add tests for TTL-aware eviction, trigger parse failures
- No dependencies on other batches

#### Batch 4: Documentation Fix (ARCH-9)
- Files: `CLAUDE.md`
- No tests needed

## Summary

| Category | Count | Items |
|----------|-------|-------|
| Bugs | 8 | BUG-15 through BUG-22 |
| Architecture | 1 | ARCH-9 |
| Features | 0 | — |
| Refactoring | 2 | REF-1, REF-2 (carried from previous) |
| Batches | 4 | Independent (can run in parallel) |
