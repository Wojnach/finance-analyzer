# Improvement Plan

Updated: 2026-03-09
Branch: improve/auto-session-2026-03-09

Previous sessions: 2026-03-05 (dashboard hardening), 2026-03-06 (CircuitBreaker, TTL cache, prune fix), 2026-03-07 (digest hardening, outcome tracker, disabled signals), 2026-03-08 (signal logging, file I/O safety, cache TTL, trigger hardening).

## Session Results (2026-03-08 — previous)

All 4 batches implemented. BUG-15 through BUG-22 fixed, ARCH-9 done.

## Session Plan (2026-03-09)

### 1) Bugs & Problems Found

#### BUG-23 (P1): Signal return value not validated — None/NaN can enter consensus

- **File**: `portfolio/signal_engine.py:563-569`
- **Issue**: Enhanced signals return `{"action": ..., "confidence": ...}`. The engine does
  `result.get("action", "HOLD")` and `result.get("confidence", 0.0)` but doesn't validate
  the values. If a signal returns `{"action": None}` or `{"confidence": float("nan")}`,
  these corrupt values propagate into `extra_info` and downstream reporting. Specifically:
  - `None` action would be stored in `votes[sig_name]` and counted as neither BUY/SELL/HOLD
  - `NaN` confidence would poison weighted consensus math (`NaN + anything = NaN`)
- **Fix**: Add `_validate_signal_result()` function that normalizes action to a valid
  string and ensures confidence is a finite float in [0, 1].
- **Impact**: Medium. Prevents silent corruption of consensus when a signal misbehaves.

#### BUG-24 (P2): news_event.py crashes on None ticker

- **File**: `portfolio/signals/news_event.py:50`
- **Issue**: `_fetch_headlines()` does `ticker.upper().replace("-USD", "")` on line 50.
  The `context` parameter comes from `signal_engine.py:545` where `ticker` can be None
  (when `generate_signal()` is called without a ticker). If ticker is None, this crashes
  with `AttributeError: 'NoneType' object has no attribute 'upper'`.
- **Fix**: Add early return with HOLD if ticker is None.
- **Impact**: Low risk. ticker is almost always set in practice, but defensive coding matters.

#### BUG-25 (P2): `load_json()` silently swallows OSError

- **File**: `portfolio/file_utils.py:39`
- **Issue**: `load_json()` catches `OSError` which includes permission denied, disk full,
  and network filesystem errors. These are NOT the same as "file doesn't exist" — they
  indicate real problems. Returning `default` silently masks infrastructure issues.
- **Fix**: Only catch `json.JSONDecodeError` and `ValueError`. Let `OSError` propagate
  (except `FileNotFoundError` which is already handled by the `exists()` check).
- **Impact**: Low risk — changes error handling to be more correct. Callers that want
  to swallow errors can catch at their level.

#### BUG-26 (P2): Heartbeat not written during initial run

- **File**: `portfolio/main.py:516-517, 557`
- **Issue**: The heartbeat file is written inside the `while True` loop (line 557) but NOT
  after the initial `run()` call (line 516). If the initial run hangs or takes very long,
  the stale heartbeat detection on the next restart will fire spuriously.
- **Fix**: Write heartbeat after the initial `run()` succeeds.
- **Impact**: Low risk. Fixes edge case on first run after restart.

#### BUG-27 (P3): Redundant `pass` in trigger.py:89

- **File**: `portfolio/trigger.py:88-89`
- **Issue**: After `logger.warning(...)`, the `pass` statement is redundant.
- **Fix**: Remove the `pass`.
- **Impact**: No behavioral change. Code cleanup.

### 2) Architecture Improvements

#### ARCH-10: Signal result validation function

- **Files**: `portfolio/signal_engine.py`
- **Issue**: Each enhanced signal returns a dict, but the engine trusts the format blindly.
  A misbehaving signal can inject None/NaN/invalid strings into the consensus pipeline.
- **Fix**: Add `_validate_signal_result(result)` that:
  - Normalizes `action` to one of ("BUY", "SELL", "HOLD")
  - Clamps `confidence` to [0.0, 1.0] and replaces NaN with 0.0
  - Ensures `sub_signals` is a dict (default to {})
  - Returns a clean dict, always
- **Impact**: Centralizes validation. All 19 enhanced signals benefit automatically.

#### ARCH-11: Confidence cap enforcement in signal registry

- **Files**: `portfolio/signal_registry.py`, `portfolio/signal_engine.py`
- **Issue**: Only 5/19 enhanced signals cap confidence at 0.7 (forecast, news_event,
  econ_calendar, claude_fundamental, futures_flow). The other 14 can return 1.0 confidence
  on majority vote, which over-weights them vs capped signals. CLAUDE.md documents that
  context-aware signals cap at 0.7, but technical signals don't.
- **Fix**: Add `max_confidence` parameter to signal registry entries. Apply it in
  `_validate_signal_result()`. Set 0.7 for context-aware signals, 1.0 for others (explicit).
- **Impact**: Makes confidence caps visible and enforceable from the registry. No behavioral
  change for existing capped signals; documents intent for uncapped ones.

### 3) Test Coverage Gaps

#### TEST-1: Missing tests for candlestick signal

- **File**: `portfolio/signals/candlestick.py` (200+ lines, 0 tests)
- **Issue**: No `tests/test_signals_candlestick.py`. Hammer, engulfing, doji, and star
  pattern detection are completely untested.
- **Fix**: Write tests covering: each pattern type, edge cases (insufficient data, flat
  candles, all-NaN), the extra `patterns_detected` field.

#### TEST-2: Missing tests for fibonacci signal

- **File**: `portfolio/signals/fibonacci.py` (450+ lines, 0 tests)
- **Issue**: No `tests/test_signals_fibonacci.py`. Retracement levels, golden pocket,
  extensions, pivot points, and camarilla calculations are untested.
- **Fix**: Write tests covering: each sub-indicator, edge cases, vote aggregation.

#### TEST-3: Missing tests for structure signal

- **File**: `portfolio/signals/structure.py` (280+ lines, 0 tests)
- **Issue**: No `tests/test_signals_structure.py`. High/low breakout, Donchian 55, RSI
  centerline cross, and MACD zero-line cross are untested.
- **Fix**: Write tests covering: each sub-indicator, edge cases.

### 4) Refactoring TODOs

#### REF-3: Remove `patterns_detected` from candlestick signal return

- **File**: `portfolio/signals/candlestick.py:49,59`
- **Issue**: Returns `patterns_detected` which is non-standard. All other signals return
  only `action`, `confidence`, `sub_signals`, and optionally `indicators`. This extra field
  is never used downstream.
- **Fix**: Move pattern names into `indicators` dict instead.

### 5) Dependency/Ordering

#### Batch 1: Signal Validation (BUG-23, ARCH-10, ARCH-11)
- Files: `portfolio/signal_engine.py`, `portfolio/signal_registry.py`
- Tests: Add tests for `_validate_signal_result()`, NaN/None handling
- Must be done first — other batches may depend on clean signal output

#### Batch 2: Defensive Fixes (BUG-24, BUG-25, BUG-26, BUG-27)
- Files: `portfolio/signals/news_event.py`, `portfolio/file_utils.py`,
  `portfolio/main.py`, `portfolio/trigger.py`
- Tests: Add tests for None ticker in news_event, OSError in load_json
- Independent of Batch 1

#### Batch 3: Test Coverage (TEST-1, TEST-2, TEST-3, REF-3)
- Files: `tests/test_signals_candlestick.py` (new), `tests/test_signals_fibonacci.py` (new),
  `tests/test_signals_structure.py` (new), `portfolio/signals/candlestick.py`
- Independent of Batches 1 and 2

## Summary

| Category | Count | Items |
|----------|-------|-------|
| Bugs | 5 | BUG-23 through BUG-27 |
| Architecture | 2 | ARCH-10, ARCH-11 |
| Test Coverage | 3 | TEST-1, TEST-2, TEST-3 |
| Refactoring | 1 | REF-3 |
| Batches | 3 | Batch 1 depends-first, Batch 2+3 independent |
