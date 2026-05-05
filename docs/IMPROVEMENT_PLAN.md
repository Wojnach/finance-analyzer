# Improvement Plan — Auto-Session 2026-05-05

## Methodology

3 parallel exploration agents covering core loop/signals/triggers,
portfolio/risk/dashboard, and metals/infrastructure. Manual verification
of all critical claims against source code. health.py race condition
rejected (correctly locked). 3 confirmed issues remain.

---

## 1. Bugs & Problems Found

### P1 — Critical (production impact, data corruption risk)

**B1: trade_guards.py record_trade() has no threading lock**
- File: `portfolio/trade_guards.py:260,310`
- `record_trade()` does load→mutate→save without holding any lock.
  If two threads record trades simultaneously (ThreadPoolExecutor in
  main.py processes tickers in parallel), the second save overwrites
  the first's mutations. Loss counters, cooldowns, and position rate
  limits can be corrupted.
- Impact: Incorrect cooldown multipliers → overtrading violations;
  incorrect consecutive_losses → wrong escalation level.
- Fix: Add a module-level `threading.Lock()` around the full RMW
  sequence in `record_trade()`. Also protect `check_overtrading_guards()`
  which has the same load-without-lock pattern.

**B2: signal_engine.py _cross_ticker_consensus dict race condition**
- File: `portfolio/signal_engine.py:258,3132-3133,3709`
- `_cross_ticker_consensus` is a module-level dict written at line 3709
  inside `generate_signal()` without a lock. When ThreadPoolExecutor
  processes BTC-USD and MSTR in parallel, MSTR reads BTC's consensus
  at lines 3132-3133 — potentially seeing stale or torn data.
- Comment at line 253 claims "GIL protects dict assignment" — this is
  wrong: membership test + subsequent `.get()` is NOT atomic. More
  importantly, stale reads cause MSTR to use a PREVIOUS cycle's BTC
  consensus, not the current one.
- Impact: MSTR trading decisions based on stale BTC consensus data.
- Fix: Process BTC-USD FIRST (before MSTR) by sorting tickers so
  dependencies are resolved before dependents. Add a lock for the dict
  for safety. Document the ordering requirement.

### P2 — Moderate (silent degradation)

**B3: data_collector.py returns empty DataFrame on Binance empty data**
- File: `portfolio/data_collector.py:90-93`
- When Binance returns 200 OK with empty data (e.g., during maintenance,
  new instrument, or rate limit), `_binance_fetch()` records a circuit
  breaker failure but returns an empty DataFrame instead of raising.
- Impact: Downstream signal computation receives empty data, logs
  "insufficient data" warnings, and produces HOLD votes — but the
  trigger system doesn't know data is missing (it sees a valid HOLD).
  This masks outages silently.
- Fix: Raise `ConnectionError` after recording the failure, so the
  caller's error handling (which already exists) properly classifies
  this as a data fetch failure.

---

## 2. Implementation Batches

### Batch 1: Fix B1 — trade_guards.py thread safety
Files: `portfolio/trade_guards.py`, `tests/test_trade_guards.py`
- Add `_state_lock = threading.Lock()` at module level
- Wrap `record_trade()` full body in `with _state_lock:`
- Wrap `check_overtrading_guards()` read path in `with _state_lock:`
- Write test: concurrent `record_trade()` calls don't lose updates

### Batch 2: Fix B2 — signal_engine cross-ticker consensus ordering
Files: `portfolio/signal_engine.py`, `portfolio/main.py`
- Add `_cross_ticker_lock = threading.Lock()` to signal_engine
- Protect writes to `_cross_ticker_consensus` with the lock
- In main.py, ensure BTC-USD is processed BEFORE MSTR (sort tickers
  so dependencies resolve first, or process BTC in the first batch
  then MSTR in the second)
- Add comment documenting ordering requirement

### Batch 3: Fix B3 — data_collector silent failure
Files: `portfolio/data_collector.py`, `tests/test_data_collector.py`
- Change line 93 from `return pd.DataFrame()` to `raise ConnectionError(...)`
- Verify callers already handle ConnectionError (grep for try/except
  around `binance_klines` and `_binance_fetch`)
- Write test confirming the exception propagates

---

## 3. Impact Assessment

| Change | Risk | Rollback |
|--------|------|----------|
| B1: Lock in trade_guards | LOW — additive, no behavior change on happy path | Remove lock |
| B2: Cross-ticker ordering | MEDIUM — changes ticker processing order. If a signal depends on MSTR being processed first (unlikely), it would break | Revert sort |
| B3: Raise instead of return | MEDIUM — callers that don't handle the exception will see errors. Must verify all callers have try/except | Revert to return pd.DataFrame() |

---

## 4. Not Addressed (future work)

- **signal_engine.py persistence state unbounded memory**: Low severity,
  only affects test/probe scenarios. Deferred.
- **file_utils.py Windows lockfile permanent deadlock on crash**: Real
  issue but rare (requires process crash while holding sidecar lock).
  Requires significant refactor to fix safely. Deferred.
- **agent_invocation.py double-logging on timeout path**: Cosmetic,
  doesn't affect trading decisions. Deferred.
- **dashboard/app.py no timeout on adaptive JSONL read**: Would require
  threading/async changes to Flask. Deferred.
