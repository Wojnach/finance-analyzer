# Improvement Plan — Auto Session 2026-02-22 (Fresh)

> Based on deep reading of all ~90 Python files, 1101 passing tests.
> Prior session completed: signal_utils extraction, div-by-zero fixes,
> bigbet cooldown, dashboard decisions, LoRA assessment. This plan covers NEW work.

## Priority 1: Bugs & Correctness Issues

### B1. `collect_timeframes` cache writes are not thread-safe
**File:** `portfolio/data_collector.py:273-290`
**Bug:** Direct `_ss._tool_cache[cache_key] = ...` writes bypass the
`_cache_lock` in `shared_state.py`. The `_cached()` helper properly
locks, but `collect_timeframes()` manages its own cache entries directly.
If the TelegramPoller thread or any future concurrent path touches
the same cache dict, this is a data race.
**Fix:** Use `_cached()` helper or wrap direct writes with `_cache_lock`.
**Impact:** Low probability today (single main thread writes), but a
latent race condition waiting to happen. Low risk fix.

### B2. `outcome_tracker.backfill_outcomes` opens SignalDB per-outcome
**File:** `portfolio/outcome_tracker.py:336-344`
**Bug:** Inside a tight loop over every ticker and every horizon,
the code does `_db = SignalDB(); _db.update_outcome(...); _db.close()`.
This opens and closes the SQLite connection for EVERY outcome write.
For 31 tickers × 4 horizons × N entries, that's thousands of open/close
cycles. Each `close()` forces a WAL checkpoint.
**Fix:** Open SignalDB once before the loop, use it for all writes,
close once after. Same pattern for the `log_signal_snapshot` function
at line 147-152.
**Impact:** Performance improvement for `--check-outcomes`. No logic change.

### B3. `accuracy_stats.signal_accuracy` reloads ALL entries on every call
**File:** `portfolio/accuracy_stats.py:50-84`
**Bug:** `signal_accuracy()` calls `load_entries()` which reads the
full signal log from SQLite or JSONL every time. This function is
called multiple times per cycle: once in `signal_engine.py` (via
`load_cached_accuracy`), once in `reporting.py` (for agent_summary),
and once in `print_accuracy_report()`. The cache in `load_cached_accuracy`
helps for the first case, but `write_agent_summary` calls `signal_accuracy`
directly (line 152), bypassing the cache.
**Fix:** Have `write_agent_summary` use `load_cached_accuracy` too.
**Impact:** Performance — avoids redundant full-log scans. No logic change.

### B4. `health.check_agent_silence` scans entire invocations.jsonl
**File:** `portfolio/health.py:77-93`
**Bug:** When `last_invocation_ts` is not in health_state, the fallback
reads the entire `invocations.jsonl` line by line to find the last
timestamp. This file only grows over time. The fallback iterates
forward and keeps overwriting `last_ts` — it should read from the end.
**Fix:** Read the last non-empty line (seek to end, scan backward).
Or better, ensure `last_invocation_ts` is always written so the
fallback never triggers.
**Impact:** Low — the cached path works. But the fallback is O(n).

### B5. `telegram_poller._poll_loop` uses `print()` instead of logger
**File:** `portfolio/telegram_poller.py:40,126,128`
**Bug:** The poller thread uses `print()` for error output, bypassing
the structured logging system. These messages go to stdout only and
are invisible in the rotating log files.
**Fix:** Replace `print()` with `logger.warning()`.
**Impact:** Trivial fix, improves debuggability.

### B6. Dashboard `api_accuracy` does `sys.path.insert` every request
**File:** `dashboard/app.py:168-170`
**Bug:** Every call to `/api/accuracy` inserts the project root into
`sys.path`. This is redundant (it's already there from app startup)
and modifies global state on every request.
**Fix:** Remove the `sys.path.insert` calls (same at line 482).
The imports work without them since the app runs from the project root.
**Impact:** Trivial cleanup. No behavior change.

## Priority 2: Architecture Improvements

### A1. Duplicate kline-fetching code in iskbets.py
**File:** `portfolio/iskbets.py:125-221`
**Bug:** `_compute_atr_15m_impl` reimplements Binance spot, Binance FAPI,
and Alpaca kline fetching — exact duplicates of `data_collector.py`'s
`binance_klines`, `binance_fapi_klines`, `alpaca_klines`. 100+ lines
of redundant code. Any fix to data_collector (e.g., error handling,
header changes) must be duplicated here.
**Fix:** Replace with calls to `data_collector._fetch_klines()`.
**Impact:** Medium — reduces duplication by ~100 lines. Must verify
ATR computation gives same results.

### A2. Duplicate portfolio validation in dashboard vs portfolio_validator
**File:** `dashboard/app.py:213-305` vs `portfolio/portfolio_validator.py`
**Bug:** The dashboard's `/api/validate-portfolio` endpoint reimplements
portfolio validation from scratch (~90 lines), duplicating the logic
in `portfolio_validator.py` (294 lines, more thorough). The dashboard
version is less complete (missing fee reconciliation, avg_cost checks).
**Fix:** Have the dashboard endpoint call `portfolio_validator.validate_portfolio()`.
**Impact:** Reduces code duplication, dashboard gets more thorough validation.

### A3. Signal vote derivation duplicated in outcome_tracker
**File:** `portfolio/outcome_tracker.py:25-98`
**Bug:** `_derive_signal_vote()` reimplements signal logic from
`signal_engine.py` (RSI thresholds, MACD crossover, EMA deadband, etc.).
Any change to signal logic must be duplicated here. The function exists
because older signal log entries didn't store individual votes.
**Fix:** Since newer entries include `_votes` in extra (line 110-112
already checks for this), the derivation fallback is for legacy data only.
Add a comment clarifying this, and consider deprecating for entries
after a certain date.
**Impact:** Low risk — adding documentation. No behavior change.

## Priority 3: Code Quality

### Q1. Replace print() with logger in telegram_poller.py
Part of B5. Trivial.

### Q2. Remove redundant sys.path.insert in dashboard
Part of B6. Trivial.

### Q3. Add `__all__` exports to key modules
Several modules export internal functions that tests import by path.
Adding `__all__` would clarify the public API surface.
Not urgent — deferring.

## Execution Order

### Batch 1: Thread-safety & performance fixes (B1, B2, B3)
Files: data_collector.py, outcome_tracker.py, reporting.py
- Fix cache writes thread-safety in collect_timeframes
- Open SignalDB once in backfill loop
- Use cached accuracy in reporting.py

### Batch 2: Code deduplication (A1, A2)
Files: iskbets.py, dashboard/app.py
- Replace kline fetching in iskbets with data_collector calls
- Replace dashboard validation with portfolio_validator call

### Batch 3: Logging & cleanup (B5, B6, B4)
Files: telegram_poller.py, dashboard/app.py, health.py
- Replace print() with logger
- Remove sys.path.insert
- Optimize health.py fallback (or ensure cached path always works)
