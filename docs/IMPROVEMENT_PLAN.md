# Improvement Plan

Updated: 2026-03-21
Branch: improve/auto-session-2026-03-21

Previous sessions: 2026-03-05 through 2026-03-20.

## Session Plan (2026-03-21)

### Theme: Crash Safety, Thread Safety, Silent Failure Elimination

Previous session (2026-03-20) fixed thread-unsafe sentiment/ADX globals (BUG-85/86),
NaN propagation (BUG-87/88), and agent lifecycle reliability (BUG-91/92/97).

This session addresses six verified issues found by deep code audit:

1. **Crash safety** — sentiment flush sets dirty=False before write; data lost on failure
2. **Thread safety** — forecast circuit breaker globals lack locks in ThreadPoolExecutor
3. **Zero-division** — portfolio P&L calculation crashes on corrupt state
4. **Silent failures** — calendar_seasonal has 9 exception handlers with no logging
5. **Alert routing** — FX rate fallback alerts saved but never sent to user
6. **Memory leak** — forecast prediction dedup cache grows unbounded

---

### 1) Bugs & Problems Found

#### BUG-101 (P1): Sentiment flush not crash-safe — dirty flag cleared before write

- **File**: `portfolio/signal_engine.py:88-104`
- **Issue**: `flush_sentiment_state()` sets `_sentiment_dirty = False` (line 98) inside the
  lock, BEFORE `atomic_write_json()` is called (line 102). If the write fails (disk full,
  permission error, etc.), the dirty flag is already cleared. On next cycle, the function
  returns early without retrying the write. On restart, sentiment hysteresis state is lost.
- **Fix**: Move `_sentiment_dirty = False` to after successful write. Re-acquire lock to
  set it only on success.
- **Impact**: After disk error + restart, sentiment signals may flip incorrectly
  (threshold drops from 0.55 to 0.40), causing false triggers.

#### BUG-102 (P2): Forecast circuit breaker globals not thread-safe

- **File**: `portfolio/signals/forecast.py:82-83, 134-158, 167-189`
- **Issue**: `_kronos_tripped_until` and `_chronos_tripped_until` are plain float globals
  modified by `_trip_kronos()`, `_trip_chronos()`, and `_log_health()` from
  ThreadPoolExecutor worker threads. While Python GIL makes float assignment atomic,
  the read-check-write pattern in `_log_health()` (lines 184-189) is NOT atomic.
  Similarly, `_last_prediction_ts` dict (line 88) is modified without lock.
- **Fix**: Add `_forecast_lock = threading.Lock()` around all circuit breaker and dedup
  cache mutations.
- **Impact**: Race condition could cause circuit breaker to trip/reset incorrectly.

#### BUG-103 (P2): Division by zero in portfolio P&L logging

- **File**: `portfolio/main.py:434`
- **Issue**: `state["initial_value_sek"]` used as divisor without guard. If portfolio state
  is corrupted (e.g., `initial_value_sek: 0` or missing key), the entire run() cycle
  crashes with ZeroDivisionError before triggering, reporting, or agent invocation.
- **Fix**: Guard: `initial = state.get("initial_value_sek") or INITIAL_CASH_SEK`
- **Impact**: Entire cycle lost on state corruption.

#### BUG-104 (P3): Calendar seasonal signal: 9 silent exception handlers

- **File**: `portfolio/signals/calendar_seasonal.py:380-422`
- **Issue**: 9 `except Exception:` blocks catch sub-signal computation errors with zero
  logging. Failures are completely invisible.
- **Fix**: Add `logger.debug()` to each handler.
- **Impact**: Silent accuracy degradation when calendar sub-signals break.

#### BUG-105 (P3): FX rate fallback alerts never reach user

- **File**: `portfolio/fx_rates.py:66`
- **Issue**: `send_or_store(msg, config, category="fx_alert")` uses category `"fx_alert"`
  which is save-only in `message_store.py`. User never receives Telegram notification
  when FX rate goes stale or falls back to hardcoded 10.85 SEK.
- **Fix**: Change category to `"error"` (which IS sent to Telegram).
- **Impact**: Trades with stale FX rate without user awareness.

#### BUG-106 (P3): Forecast prediction dedup cache grows unbounded

- **File**: `portfolio/signals/forecast.py:88`
- **Issue**: `_last_prediction_ts` dict maps ticker -> monotonic timestamp. Entries are
  never evicted. Minor memory leak.
- **Fix**: Evict entries older than 10 minutes during each write.
- **Impact**: Negligible but indicates missing cleanup pattern.

---

### 2) Implementation Batches

#### Batch 1: Crash Safety & Core Fixes (3 files)

| Bug | File | Change |
|-----|------|--------|
| BUG-101 | signal_engine.py | Move `_sentiment_dirty = False` after successful write |
| BUG-103 | main.py | Guard division by `initial_value_sek` |
| BUG-105 | fx_rates.py | Change FX alert category to `"error"` |

**Risk**: LOW — all changes are additive guards.

#### Batch 2: Thread Safety & Silent Failures (2 files)

| Bug | File | Change |
|-----|------|--------|
| BUG-102 | signals/forecast.py | Add threading.Lock around circuit breaker + dedup |
| BUG-106 | signals/forecast.py | Add dedup cache eviction |
| BUG-104 | signals/calendar_seasonal.py | Add logger.debug() to 9 exception handlers |

**Risk**: LOW — lock addition is additive. Calendar logging is observational only.

#### Batch 3: Tests

| Test | Covers |
|------|--------|
| test_sentiment_flush_crash_safe.py | BUG-101 |
| test_forecast_thread_safety.py | BUG-102/106 |
| test_fx_alert_routing.py | BUG-105 |
| test_calendar_exception_logging.py | BUG-104 |

---

### 3) Results

All 6 bugs fixed, all 3 batches implemented, 16 new tests passing.
166 related tests pass (zero regressions). 139 pre-existing test failures
(all pre-date this session — verified by running same tests on main branch).

| Bug | Status | Commit |
|-----|--------|--------|
| BUG-101 | FIXED | sentiment dirty flag after write |
| BUG-102 | FIXED | forecast circuit breaker thread lock |
| BUG-103 | FIXED | zero-division guard on initial_value_sek |
| BUG-104 | FIXED | calendar_seasonal exception logging (9 handlers) |
| BUG-105 | FIXED | FX alert category changed to "error" |
| BUG-106 | FIXED | prediction dedup cache eviction (10min) |

### 4) What Was NOT Changed (and Why)

- **reporting.py test coverage**: 24K lines, 0 tests. Too large for this session.
- **ADX cache key fragility**: Theoretically racy id(df) but practically safe.
- **Portfolio state validation**: Needs dedicated feature, not quick fix.
- **Config reload consistency**: Would touch 10+ files.
