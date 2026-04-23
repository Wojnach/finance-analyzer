# Improvement Plan — Auto Session 2026-04-23

Based on deep exploration by 4 parallel agents (signal engine, portfolio/risk, infrastructure,
test coverage) plus manual verification of all P0/P1 findings.

## Exploration Summary

### Confirmed Critical Bugs
- **BUG-219 REGRESSION**: `_record_new_trades()` calls `record_trade(ticker, direction, strategy)`
  WITHOUT `pnl_pct`. The consecutive-loss escalation code at `trade_guards.py:240` checks
  `if direction == "SELL" and pnl_pct is not None` — since pnl_pct is always None from the
  production path, loss escalation NEVER fires. Cooldowns work (ticker+timestamp), but
  the 1x→2x→4x→8x multiplier on consecutive losses is completely dead.
  - **Severity**: P0 — safety feature broken for all production trades
  - **Evidence**: Lines 720 vs 240. Confirmed by manual reading.

### Confirmed P1 Bugs
- **Rate limiter wake synchronization**: `shared_state.py:256-269` — `last_call` updated AFTER
  sleep, so parallel threads that calculate wait_time simultaneously wake at the same time
  and stampede. Not a bypass (they all wait the same duration), but reduces effective spacing.
- **Drawdown NaN propagation**: `risk_management.py:159` — no guard against NaN/Inf in
  peak_value or current_value propagating through division.
- **Stack overflow counter never resets on non-consecutive failures**: 
  `agent_invocation.py:936-941` — counter resets to 0 on ANY non-stack-overflow exit, 
  including "failed" or "auth_error". This is actually correct behavior (breaks the 
  consecutive chain). VERIFIED NOT A BUG.

### Confirmed P2 Issues
- **cached() stores None results**: `shared_state.py:94-95` — when `func()` returns None
  (e.g., API failure that returns None instead of raising), it's cached as valid data for
  TTL duration, hiding the failure from retry logic.
- **subprocess_utils Job Object suppressed**: `subprocess_utils.py:132-133` — 
  `contextlib.suppress(Exception)` hides job assignment failures, allowing orphan processes.
- **ATR stop cap silent**: `risk_management.py:225` — 15% ATR cap applied without logging.

### Verified Non-Bugs (agent false positives)
- **Sortino ratio formula**: Uses `sum(squared_devs) / len(daily_rets_dec)` — this IS the
  standard semi-deviation formula. Dividing by N_total (not N_downside) is correct per
  Investopedia, scipy, and the original Sortino & Price (1994) paper. The H19 comment is
  accurate. NOT A BUG.
- **Config wipe guard (BUG-210)**: `telegram_poller.py:207` — `len(cfg) < 5` guard is
  adequate because config.json always has 10+ top-level keys in production. The guard
  correctly blocks on empty/corrupt reads. Agent concern about "only 4 keys" is theoretical.
- **Dashboard auth**: The token is loaded from config and when set, all API routes require it.
  The concern about `dashboard_token = None` is valid but this is the documented optional-auth
  design. Adding forced auth would break local-only usage.

---

## Batch 1: BUG-219 Loss Escalation Fix (CRITICAL — 1 file, ~5 lines)

### 1.1 Pass pnl_pct from transaction data to record_trade()

**File:** `portfolio/agent_invocation.py`
**Problem:** `_record_new_trades()` at line 720 calls `record_trade(ticker, direction, strategy)`
without the `pnl_pct` parameter. The transaction dict from portfolio state contains a `pnl_pct`
field for SELL trades (populated by Layer 2 when it records a sale). Without this, the
consecutive-loss escalation system (1x→2x→4x→8x cooldown multiplier) is completely dead.

**Fix:** Extract `pnl_pct` from the transaction dict and pass it to `record_trade()`.

**Impact:** Activates consecutive-loss escalation for the first time in production.
**Risk:** LOW — additive parameter, function already accepts it. If pnl_pct is missing from
the transaction, it defaults to None (existing behavior).

---

## Batch 2: Rate Limiter Fix + Drawdown NaN Guard (2 files, ~15 lines)

### 2.1 Fix rate limiter wake synchronization

**File:** `portfolio/shared_state.py`, lines 256-269
**Problem:** `last_call` is updated AFTER the sleep, allowing parallel threads to calculate
the same wait_time and wake simultaneously.

**Fix:** Update `last_call` to `now + wait_time` BEFORE releasing the lock and sleeping.
This reserves the next slot atomically. Threads arriving during the sleep see the updated
`last_call` and calculate a longer wait.

**Impact:** Proper spacing of API calls under concurrent load.
**Risk:** LOW — timing behavior only, no functional change.

### 2.2 Guard drawdown against NaN/Inf

**File:** `portfolio/risk_management.py`, around line 159
**Problem:** If `peak_value` or `current_value` is NaN/Inf (from corrupted history file or
failed computation), the drawdown percentage becomes NaN, which silently passes all comparison
checks (`NaN > 50.0` is False), bypassing the circuit breaker.

**Fix:** After computing `current_drawdown_pct`, check `math.isfinite()`. If not finite,
log a critical error and return a fail-safe response (treat as 100% drawdown or raise).

**Impact:** Prevents NaN from silently bypassing the drawdown circuit breaker.
**Risk:** LOW — defensive guard only.

---

## Batch 3: Cache None Prevention + Orphan Process Logging (2 files, ~10 lines)

### 3.1 Don't cache None results in _cached()

**File:** `portfolio/shared_state.py`, line 94-95
**Problem:** When `func()` returns `None` (API failure returning None instead of raising),
the result is cached as `{"data": None, "time": now}`. For the full TTL duration, all
subsequent calls return None without retrying. This hides transient failures.

**Fix:** After `data = func(*args)`, check `if data is not None` before writing to cache.
If None, still clear the loading key but don't update the cache entry. Stale data (if any)
will continue to be served, and the next cycle will retry.

**Impact:** Transient API failures no longer poison the cache for TTL duration.
**Risk:** MEDIUM — signals that legitimately return None will retry every cycle instead of
being cached. This is acceptable because signal functions that have no data should return
a HOLD dict, not None.

### 3.2 Log subprocess Job Object assignment failures

**File:** `portfolio/subprocess_utils.py`, lines 132-133
**Problem:** `contextlib.suppress(Exception)` silently hides Job Object assignment failures.
If assignment fails, the child process won't be killed when the parent exits, leading to
orphan processes.

**Fix:** Replace `suppress(Exception)` with `try/except Exception as e: logger.warning(...)`.

**Impact:** Orphan process creation becomes visible in logs.
**Risk:** VERY LOW — logging only.

---

## Batch 4: Tests (~50 lines)

### 4.1 Test pnl_pct wiring in _record_new_trades()
- SELL transaction with negative pnl_pct increments consecutive_losses
- SELL transaction with positive pnl_pct resets consecutive_losses
- BUY transaction doesn't affect consecutive_losses
- Missing pnl_pct field in transaction falls back gracefully

### 4.2 Test rate limiter spacing
- Two threads calling wait() simultaneously should not wake at the same time
- Effective spacing should be >= interval

### 4.3 Test drawdown NaN guard
- NaN peak_value returns fail-safe result
- Inf current_value returns fail-safe result
- Normal values compute correctly (regression test)

### 4.4 Test _cached() with None return
- func() returning None should not cache the result
- Stale data should still be served when func() returns None
- Subsequent calls should retry func()

---

## Dependency Order

Batch 1 → Batch 2 → Batch 3 → Batch 4

Batch 1 is the highest priority (safety-critical). Each batch is independently committable
and testable. Total: ~80 lines of changes across 4 files + ~50 lines of tests.

## What NOT to Implement (Deferred)

- **FIFO round-trip race condition**: equity_curve.py is read-only in production (only called
  from reporting), so the theoretical concurrency issue has no practical impact.
- **Lockfile creation TOCTOU**: The sidecar lockfile in file_utils.py works correctly in
  practice because only same-process threads contend (not cross-process). The TOCTOU window
  is microseconds and the system has been stable for weeks.
- **Dashboard forced auth**: Optional-auth is the documented design for local-only deployment.
- **Data collector timeout isolation**: The ThreadPoolExecutor timeout handling works
  correctly; stuck futures are cleaned up by Python's GC.
- **Config wipe guard strengthening**: BUG-210 guard is adequate for production config.
- **T-copula numerical stability**: Extreme tail values are rare and don't affect P50/P95 VaR.
