# Improvement Plan

Updated: 2026-03-10
Branch: worktree-improve-auto-session-2026-03-10

Previous sessions: 2026-03-05 (dashboard hardening), 2026-03-06 (CircuitBreaker, TTL cache, prune fix), 2026-03-07 (digest hardening, outcome tracker, disabled signals), 2026-03-08 (signal logging, file I/O safety, cache TTL, trigger hardening), 2026-03-09 (signal validation, confidence caps, candlestick/fibonacci/structure tests).

## Session Results (2026-03-09 — previous)

All 3 batches implemented. 5 bugs fixed (BUG-23 through BUG-27), 2 architecture improvements (ARCH-10, ARCH-11), 3 test coverage gaps filled (TEST-1, TEST-2, TEST-3), 1 refactoring (REF-3). 193 new tests.

## Session Plan (2026-03-10)

### 1) Bugs & Problems Found

#### BUG-28 (P1): Enhanced signal failures silently counted as HOLD — degrades consensus quality

- **File**: `portfolio/signal_engine.py:680-682` (approx, in `generate_signal()`)
- **Issue**: When any enhanced signal throws an exception, it's caught and the vote is set
  to HOLD: `votes[sig_name] = "HOLD"`. This is correct for preventing crashes, but there is
  NO tracking of how many signals failed per cycle. If 15 out of 19 enhanced signals crash
  (e.g., due to a missing column in OHLCV data), the consensus operates on only 4 enhanced
  + 8 core = 12 signals. The consensus appears valid but is based on dramatically less data.
  Layer 2 has no way to know signals degraded.
- **Fix**: Track signal failures in `extra_info["_signal_failures"]` list. Log a warning if
  failure count > 3. Surface in `agent_summary_compact.json` so Layer 2 can see it.
- **Impact**: Medium. Prevents silent degradation of consensus quality. Layer 2 can adjust
  confidence when many signals failed.

#### BUG-29 (P1): `_vote_correct()` treats 0% price change as neither correct nor incorrect

- **File**: `portfolio/accuracy_stats.py:48-53`
- **Issue**: `_vote_correct("BUY", 0.0)` returns False (0 > 0 is False). Same for SELL.
  When the price doesn't move at all between snapshot and outcome, BOTH BUY and SELL are
  counted as wrong. This biases accuracy downward for all signals, especially in
  low-volatility periods where many outcomes are ~0%. A flat price isn't evidence the signal
  was wrong — it's ambiguous.
- **Fix**: Add a small tolerance: treat change_pct within ±0.05% as neutral (skip, don't
  count as correct or incorrect). This prevents flat-market periods from diluting accuracy.
- **Impact**: High. Affects all accuracy calculations. In sideways markets, signals appear
  less accurate than they really are, potentially triggering wrong signal inversions.

#### BUG-30 (P2): `load_json()` returns default on `PermissionError` and `IsADirectoryError`

- **File**: `portfolio/file_utils.py:31-39`
- **Issue**: `load_json()` catches `json.JSONDecodeError` and `ValueError`, but NOT
  `PermissionError`, `IsADirectoryError`, or `IOError`. On Windows, `path.read_text()` can
  raise `PermissionError` if another process holds a lock (e.g., during atomic write). This
  would crash the loop. While `OSError` was removed from the catch list in BUG-25 fix
  (correctly), we should catch ONLY `FileNotFoundError` + decode errors, and let other OS
  errors propagate as they indicate real problems. Currently, `path.exists()` check is
  TOCTOU: file can be deleted between `exists()` and `read_text()`.
- **Fix**: Remove TOCTOU pattern. Use try/except with `FileNotFoundError` instead of
  `exists()` check. Keep `json.JSONDecodeError` + `ValueError` catch.
- **Impact**: Low risk but improves correctness. The TOCTOU window is tiny in practice.

#### BUG-31 (P2): `_compute_adx()` called on every confidence penalty check — no caching

- **File**: `portfolio/signal_engine.py:336`
- **Issue**: `apply_confidence_penalties()` calls `_compute_adx(df)` on every invocation.
  ADX computes rolling EWM over the full DataFrame. For the "Now" timeframe (100 bars of
  15m candles), this is fast (~0.5ms). But it's called once per ticker per cycle (19 tickers
  = 19 calls). The result is stored in `extra_info["_adx"]` but never cached across cycles.
  More importantly, `_compute_adx()` uses `replace(0, np.nan)` which creates NaN values
  that propagate through the DI calculations. The final `pd.notna(val)` check at line 296
  catches this, but NaN propagation is unnecessary work.
- **Fix**: (1) Replace `atr_smooth.replace(0, np.nan)` with
  `atr_smooth.clip(lower=1e-10)` to avoid NaN entirely.
  (2) Cache ADX result in `extra_info` so penalty cascade doesn't recompute.
- **Impact**: Low. Performance improvement + cleaner NaN handling.

#### BUG-32 (P2): `main.py` re-exports ~50 private symbols — tight coupling and namespace pollution

- **File**: `portfolio/main.py:36-108`
- **Issue**: main.py re-exports 50+ private symbols (prefixed with `_`) from other modules
  for "backwards compatibility". This couples external code (tests, trigger.py) to main.py's
  import surface. Any module rename or refactor requires updating main.py. Some re-exports
  are genuinely private: `_tool_cache`, `_prev_sentiment`, `_cached`, etc. Tests patching
  `portfolio.main._cached` instead of `portfolio.shared_state._cached` create false
  confidence — they test import wiring, not behavior.
- **Fix**: NOT fixing this session (too risky for live system). Document as ARCH-12 for
  future cleanup. The correct fix is: (1) update all test patches to target source modules,
  (2) remove re-exports from main.py, (3) update any external callers.
- **Impact**: No behavioral change. Documentation only for now.

#### BUG-33 (P2): Trap detection uses wrong timeframe data

- **File**: `portfolio/signal_engine.py:359-376`
- **Issue**: `apply_confidence_penalties()` receives `df` which is the "Now" timeframe
  DataFrame (100 bars of 15m candles). The trap detection looks at the last 5 bars, which
  represents 75 minutes of data. For detecting a bull/bear trap, this is a reasonable
  timeframe for intraday signals. However, the function doesn't know which timeframe `df`
  represents — it could be daily bars (5 bars = 5 days) or weekly bars (5 bars = 5 weeks).
  In practice, `df` is always the Now timeframe from `generate_signal()`, so this is correct.
  But adding a defensive check would prevent future bugs if the calling convention changes.
- **Fix**: Add a comment documenting that `df` must be the Now timeframe (15m candles).
  Not worth adding a timeframe parameter for a hypothetical future change.
- **Impact**: Low. Documentation improvement only.

### 2) Architecture Improvements

#### ARCH-12: Signal failure tracking and surfacing

- **Files**: `portfolio/signal_engine.py`, `portfolio/reporting.py`
- **Issue**: When enhanced signals fail (crash/timeout), the failure is logged at WARNING
  level but not surfaced to Layer 2 or the dashboard. Layer 2 operates on the assumption
  that all 30 signals voted, when some may have silently failed.
- **Fix**: Add `signal_failures: list[str]` to the per-ticker signal data in
  `agent_summary_compact.json`. List signal names that failed (threw exceptions). Also
  add a top-level `signal_health: {"ok": N, "failed": N, "failed_names": [...]}` section.
- **Impact**: Enables Layer 2 to lower confidence when signal coverage is degraded.

#### ARCH-13: Accuracy tolerance for flat markets

- **Files**: `portfolio/accuracy_stats.py`
- **Issue**: The `_vote_correct()` function has a binary correct/incorrect model. A BUY
  signal with 0.01% price increase is "correct" but a BUY with -0.01% is "wrong". This
  creates noise in accuracy statistics. Signals shouldn't be penalized for small absolute
  moves where the direction is essentially random.
- **Fix**: Add `min_change_pct` parameter (default 0.05%) to `_vote_correct()`. Outcomes
  within the tolerance band are skipped (not counted as correct or incorrect). This improves
  accuracy signal-to-noise ratio.
- **Impact**: All accuracy calculations become more meaningful. Prevents flat-market accuracy
  dilution that can trigger wrong signal inversions.

### 3) Test Coverage Gaps

#### TEST-4: Missing tests for `apply_confidence_penalties()`

- **File**: `portfolio/signal_engine.py:302-401`
- **Issue**: The 4-stage confidence penalty cascade has 0 dedicated tests. It's indirectly
  tested through `generate_signal()` tests but not the specific edge cases:
  - Regime penalties (ranging 0.75x, high-vol 0.80x, trend-aligned 1.10x)
  - Volume gate (RVOL < 0.5 → force HOLD)
  - Volume + ADX combined gate
  - Trap detection (bull/bear trap with declining volume)
  - Dynamic MIN_VOTERS per regime
  - Confidence clamping to [0, 1]
  - Disabled mode (config.confidence_penalties.enabled = false)
- **Fix**: Write `tests/test_confidence_penalties.py` covering each stage and edge case.

#### TEST-5: Missing tests for `_weighted_consensus()`

- **File**: `portfolio/signal_engine.py:136-186`
- **Issue**: The weighted consensus function has no dedicated tests. It's the core of the
  signal system. Edge cases to test:
  - All HOLD votes → returns HOLD with 0.0 confidence
  - Unanimous BUY → returns BUY with 1.0 confidence
  - Signal inversion at <50% accuracy
  - Accuracy exactly 50% (boundary case)
  - Regime weight multipliers
  - Activation rate normalization
  - Small sample size (<20) → default 0.5 weight
- **Fix**: Write tests in `tests/test_weighted_consensus.py`.

#### TEST-6: Missing tests for `_compute_adx()`

- **File**: `portfolio/signal_engine.py:268-299`
- **Issue**: ADX computation has no tests. Should verify:
  - Returns None for insufficient data
  - Returns valid float for normal data
  - Handles all-zero ATR gracefully (no division by zero)
  - Returns None for NaN-heavy data
- **Fix**: Add to `tests/test_confidence_penalties.py` (same area of code).

### 4) Refactoring TODOs

#### REF-4: Remove NaN propagation in `_compute_adx()`

- **File**: `portfolio/signal_engine.py:289-290`
- **Issue**: `atr_smooth.replace(0, np.nan)` creates NaN values that propagate through DI
  calculations, only to be caught at the final `pd.notna(val)` check. This is wasteful.
- **Fix**: Use `atr_smooth.clip(lower=1e-10)` instead. Same effect (prevents division by
  zero) without NaN propagation.

### 5) Dependency/Ordering

#### Batch 1: Accuracy & Signal Health (BUG-29, ARCH-13, BUG-28, ARCH-12)
- Files: `portfolio/accuracy_stats.py`, `portfolio/signal_engine.py`, `portfolio/reporting.py`
- Tests: Verify accuracy tolerance, test signal failure tracking
- Must be done first — accuracy change affects signal inversion decisions

#### Batch 2: Code Quality & Safety (BUG-30, BUG-31, REF-4)
- Files: `portfolio/file_utils.py`, `portfolio/signal_engine.py`
- Tests: Test TOCTOU fix, ADX caching, NaN-free ADX
- Independent of Batch 1

#### Batch 3: Test Coverage (TEST-4, TEST-5, TEST-6)
- Files: `tests/test_confidence_penalties.py` (new), `tests/test_weighted_consensus.py` (new)
- Independent of Batches 1 and 2

## Summary

| Category | Count | Items |
|----------|-------|-------|
| Bugs | 6 | BUG-28 through BUG-33 |
| Architecture | 2 | ARCH-12, ARCH-13 |
| Test Coverage | 3 | TEST-4, TEST-5, TEST-6 |
| Refactoring | 1 | REF-4 |
| Batches | 3 | Batch 1 depends-first, Batch 2+3 independent |
