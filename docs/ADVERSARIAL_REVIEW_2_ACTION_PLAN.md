# Adversarial Review Action Plan — 2026-04-08

## Immediate Fixes (< 1 hour each, safe, no production risk)

### 1. Fix ADX cache key (C1)
**File**: `portfolio/signal_engine.py:25`
**What**: Replace `id(df)` with content-based key
**Why**: Python reuses `id()` after GC — wrong ADX values affect trade decisions
**How**: Use `hash((len(df), df["close"].iloc[-1], df["close"].iloc[0]))` as key
**Risk**: None — changes only cache key, same computation
**Tests**: Existing `test_signal_engine.py` ADX tests should still pass

### 2. Add health file locking (H17)
**File**: `portfolio/health.py`
**What**: Add threading lock for read-modify-write in `update_health` and `update_signal_health_batch`
**Why**: 8 ThreadPoolExecutor workers can overwrite each other's changes
**How**: Import `_get_lock` pattern from `portfolio_mgr.py`
**Risk**: Minimal — adds thread safety without changing behavior
**Tests**: Add concurrent write test

### 3. Reset stack overflow counter on agent success (H5)
**File**: `portfolio/agent_invocation.py:36-38`
**What**: Reset `_consecutive_stack_overflows = 0` in `check_agent_completion()` on success
**Why**: Counter only goes up, never down — transient crashes permanently disable Layer 2
**How**: Add one line in `check_agent_completion()` when status == "success"
**Risk**: None — only resets a safety counter
**Tests**: Add unit test for counter reset

### 4. Fix session expiry parse error (H11)
**File**: `portfolio/avanza_session.py:76-77`
**What**: Log warning instead of `pass` on `ValueError`
**Why**: Corrupt expiry timestamp silently treated as "valid session"
**How**: `except ValueError: logger.warning("Invalid expires_at: %s", expires_at)`
**Risk**: None — logging only
**Tests**: Add test with corrupt expiry string

## Short-Term Fixes (This Month)

### 5. Playwright error recovery (H12)
**File**: `portfolio/avanza_session.py:181-287`
**What**: Catch `PlaywrightError` in api_get/api_post/api_delete, call `close_playwright()`
**Why**: Browser crash leaves non-None but unusable context
**How**: Add `except Exception as e: if "Target page" in str(e) or ...: close_playwright()`
**Risk**: Low — adds recovery path, doesn't change happy path
**Tests**: Mock Playwright error, verify cleanup called

### 6. Cap metals price history (H9)
**File**: `data/metals_loop.py`
**What**: Add `maxlen=1440` (24h at 60s) to price history deques
**Why**: Unbounded growth → memory leak over weeks
**How**: `_gold_price_history = deque(maxlen=1440)`
**Risk**: None — deque maxlen is a standard Python pattern
**Tests**: Verify deque doesn't grow beyond maxlen

### 7. Account ID validation (H13)
**File**: `portfolio/avanza_session.py`
**What**: Validate account_id in `api_post` when path contains "order"
**Why**: Prevents accidental trading on pension account 2674244
**How**: `ALLOWED_ACCOUNTS = {"1625505"}; assert str(acct) in ALLOWED_ACCOUNTS`
**Risk**: Low — could block legitimate orders if new account added (but that's the point)
**Tests**: Test that pension account ID raises ValueError

### 8. Signal health quorum (CC1 mitigation)
**File**: `portfolio/signal_engine.py` (end of `generate_signal`)
**What**: Alert when fewer than 5 signals vote non-HOLD
**Why**: "Graceful degradation" can silently reduce signal capacity below useful level
**How**: Check `active_voters` and log warning + optional Telegram alert
**Risk**: None — alerting only, doesn't change consensus logic
**Tests**: Test with mock signals all returning HOLD

## Medium-Term Improvements (This Quarter)

### 9. Begin metals_loop.py decomposition (C2)
**Phase 1**: Extract `StopLossManager` class (lines ~2920-3300)
**Phase 2**: Extract `FishEngine` wrapper (lines ~2100-2300)
**Phase 3**: Extract `MetalsOrderExecutor` (lines ~2800-2900)
**Why**: 6561-line file with global state is the #1 reliability risk
**Risk**: Medium — requires careful testing of each extraction
**Estimated effort**: 3-5 sessions

### 10. Calendar date expiry warning (M10)
**File**: `portfolio/econ_dates.py`, `portfolio/fomc_dates.py`
**What**: Add function to check latest date, warn if < 60 days away
**Why**: Hardcoded dates through 2027 — silent failure in Jan 2028
**How**: `check_calendar_freshness()` called at loop startup
**Risk**: None — warning only
**Tests**: Test with mocked date near expiry

### 11. End-to-end signal pipeline test (CC2)
**File**: New test `tests/test_signal_pipeline_e2e.py`
**What**: One integration test exercising full `generate_signal` with real OHLCV
**Why**: 750-line function with 20+ module dependencies, no integration test
**How**: Use historical BTC-USD data, verify action/confidence are reasonable
**Risk**: None — test only
**Tests**: Self-evident

### 12. Alpha Vantage daily budget (H16)
**File**: `portfolio/shared_state.py`
**What**: Add daily call counter with midnight reset
**Why**: 5/min limiter allows 7200/day but free tier is 25/day
**How**: Atomic counter with `datetime.date.today()` reset check
**Risk**: Low — may throttle fundamentals refresh more aggressively
**Tests**: Test counter reset at midnight
