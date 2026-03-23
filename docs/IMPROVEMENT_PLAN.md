# Improvement Plan

Updated: 2026-03-23
Branch: improve/auto-session-2026-03-23

Previous sessions: 2026-03-05 through 2026-03-22.

## Session Plan (2026-03-23)

### Theme: Accuracy Tracking Correctness, Memory Optimization, Signal Robustness

Previous session (2026-03-22) fixed zero-division in digest, thread safety in Alpha Vantage
budget counter, performance in digest signal_log reads, and added 50 tests for reporting.py.

This session addresses verified issues found by deep code audit of the signal system,
outcome tracker, and shared utilities:

1. **Accuracy corruption** — outcome_tracker derives RSI votes with fixed 30/70 thresholds
   while signal_engine uses adaptive percentile thresholds. This corrupts accuracy backfill.
2. **Memory waste** — backfill_outcomes loads entire 68MB signal_log.jsonl into memory to
   process only the last 2,000 entries.
3. **Confidence edge case** — majority_vote returns inconsistent confidence when HOLD wins
   by default (neither BUY nor SELL achieves majority).
4. **Test coverage** — outcome_tracker has complex derivation logic with zero dedicated tests.
5. **Observability** — forecast JSON extraction fallbacks don't log which strategy succeeded.

---

### 1) Bugs & Problems Found

#### BUG-111 (P1): outcome_tracker RSI vote derivation uses fixed thresholds

- **Files**: `portfolio/outcome_tracker.py:24-32`, `portfolio/signal_engine.py:490-498`
- **Issue**: `_derive_signal_vote("rsi", ...)` uses hardcoded `< 30` (BUY) and `> 70`
  (SELL) thresholds. But `signal_engine.generate_signal()` uses adaptive thresholds from
  rolling RSI percentiles: `rsi_p20` (20th percentile, default 30) and `rsi_p80` (80th
  percentile, default 70), clamped to [15, 85]. When RSI is between the adaptive threshold
  and the fixed threshold (e.g., RSI=25 with adaptive lower=22), the outcome tracker records
  a BUY vote that signal_engine never actually cast. This corrupts accuracy tracking for RSI.
- **Mitigation**: The `_votes` dict from signal_engine is passed through `extra["_votes"]`
  in most cases, making `_derive_signal_vote` a fallback. But when `_votes` is missing (e.g.,
  stale log entries, format migration), the derivation kicks in with wrong thresholds.
- **Fix**: Pass `rsi_p20` and `rsi_p80` through the indicators dict. Update
  `_derive_signal_vote("rsi", ...)` to use them with proper defaults. Also store the adaptive
  thresholds in the signal snapshot so historical accuracy can be reconstructed correctly.
- **Impact**: HIGH — RSI accuracy numbers may be inflated or deflated depending on how often
  the adaptive thresholds deviate from 30/70. Affects weighted consensus through accuracy
  weighting.

#### BUG-112 (P2): backfill_outcomes reads entire signal_log.jsonl into memory

- **File**: `portfolio/outcome_tracker.py:265-280`
- **Issue**: `backfill_outcomes()` reads all lines from signal_log.jsonl into a Python list,
  parses every JSON line, then splits into head (preserved) and tail (processed). With 68MB+
  file and 150K+ entries, this loads ~75MB of parsed JSON into memory. The `max_entries=2000`
  parameter limits processing but not loading.
- **Fix**: Use a two-pass approach: (1) count total lines with a fast binary scan, (2) re-read
  only the last max_entries lines for processing, (3) on rewrite, copy the head bytes verbatim
  from the original file and append the modified tail. This reduces memory from 75MB to ~1MB.
- **Impact**: MEDIUM — runs daily via PF-OutcomeCheck. Not a crash risk but wastes memory.

#### BUG-113 (P3): majority_vote confidence calculation when HOLD wins

- **File**: `portfolio/signal_utils.py:122-123`
- **Issue**: When HOLD wins (neither BUY nor SELL has strict majority over HOLD), line 123
  returns `hold / denom` when `count_hold=True`, but `0.0` when `count_hold=False`. The
  `count_hold=False` path is correct (HOLD confidence should be 0.0 since it's the absence
  of a signal). The `count_hold=True` path computes `hold / total`, which can produce
  misleading confidence values. Example: votes=["HOLD","HOLD","BUY"] with `count_hold=True`
  returns `("HOLD", 0.6667)` — this suggests 67% confidence in HOLD, but HOLD really means
  "no signal". The confidence should reflect ambiguity, not HOLD strength.
- **Fix**: Return 0.0 confidence for HOLD regardless of `count_hold` flag. HOLD is the
  default/fallback action, not a directional vote. Confidence should only be non-zero for
  BUY/SELL.
- **Impact**: LOW — only affects callers using `count_hold=True`, which is rare (not used
  in the main signal path).

#### BUG-114 (P3): forecast JSON extraction fallbacks lack observability

- **File**: `portfolio/signals/forecast.py` (in `_extract_json_from_stdout()`)
- **Issue**: Three fallback strategies for extracting JSON from contaminated subprocess
  stdout. When all three fail, returns None silently. When a later fallback succeeds, no
  log indicates which strategy worked. This makes debugging Kronos stdout contamination
  harder.
- **Fix**: Add debug-level logging when a non-first fallback strategy succeeds.
- **Impact**: LOW — observability only, no behavioral change.

#### COVERAGE-2 (P2): outcome_tracker has zero dedicated tests

- **File**: `portfolio/outcome_tracker.py` (407 lines, ~5 functions)
- **Issue**: The outcome tracker manages accuracy backfill — a core data pipeline that feeds
  signal accuracy (which drives weighted consensus). Functions like `_derive_signal_vote()`,
  `log_signal_snapshot()`, and `backfill_outcomes()` have complex logic with no tests.
  Regressions here silently corrupt all accuracy data.
- **Fix**: Write targeted tests for `_derive_signal_vote()` (all 11 signal branches),
  `log_signal_snapshot()` (snapshot structure), and edge cases in backfill.
- **Impact**: HIGH — regressions corrupt accuracy tracking → corrupts weighted consensus →
  corrupts trade decisions.

---

### 2) Implementation Batches

#### Batch 1: Accuracy-Critical Fix + Tests (2 files)

| Bug | File | Change |
|-----|------|--------|
| BUG-111 | outcome_tracker.py | Update `_derive_signal_vote("rsi", ...)` to use adaptive thresholds from indicators |
| COVERAGE-2 | test_outcome_tracker_core.py | New test file: ~30 tests for _derive_signal_vote, log_signal_snapshot |

**Risk**: LOW — _derive_signal_vote is a fallback path. The fix aligns it with signal_engine.
Tests are additive.

**Dependency**: None.

#### Batch 2: Memory Optimization (1 file + tests)

| Bug | File | Change |
|-----|------|--------|
| BUG-112 | outcome_tracker.py | Refactor backfill_outcomes to use streaming head + parsed tail |
| — | test_outcome_tracker_core.py | Add backfill tests with large file simulation |

**Risk**: MEDIUM — modifies the file-rewrite logic in a function that runs daily. Must be
tested thoroughly. The rewrite still uses atomic tempfile+replace pattern.

**Dependency**: Batch 1 (tests exist to validate).

#### Batch 3: Signal Utility Fixes (2 files + tests)

| Bug | File | Change |
|-----|------|--------|
| BUG-113 | signal_utils.py | Return 0.0 confidence for HOLD in majority_vote |
| BUG-114 | forecast.py | Add debug logging for JSON extraction fallback selection |
| — | test_signal_utils.py or existing tests | Add edge case test for all-HOLD votes |

**Risk**: LOW — majority_vote change only affects the count_hold=True path. Forecast logging
is additive.

**Dependency**: None (parallel with Batch 1-2).

---

### 3) What Was NOT Changed (and Why)

- **main.py test coverage**: 851 lines, 0 tests. Loop orchestration is inherently
  integration-level. Unit testing individual functions would require extracting them further.
  Not worth the refactoring risk for this session. Deferred.
- **sentiment.py test coverage**: 608 lines, 0 tests. Too many external dependencies.
  Would need extensive mocking infrastructure. Deferred.
- **Signal registry function signature validation**: Would prevent future bugs but has no
  current manifestation. The existing test suite catches signature mismatches at test time.
  Deferred.
- **forecast.py 50-bar minimum**: Not a bug — 50 bars is adequate for time-series models.
  Adding a higher threshold could reduce forecast availability without clear benefit.
- **ADX cache nuclear eviction**: The `_adx_cache.clear()` at 200 entries is crude but
  functional. With 20 tickers × 7 timeframes = 140 entries per cycle (below 200), it
  rarely triggers. Not worth optimizing.

---

### 4) Results

| ID | Type | Status | Details |
|----|------|--------|---------|
| BUG-111 | Accuracy | PENDING | outcome_tracker RSI adaptive thresholds |
| BUG-112 | Performance | PENDING | backfill_outcomes memory optimization |
| BUG-113 | Logic | PENDING | majority_vote HOLD confidence |
| BUG-114 | Observability | PENDING | forecast JSON extraction logging |
| COVERAGE-2 | Tests | PENDING | outcome_tracker test suite |
