# Improvement Plan — Auto Session 2026-04-21

Based on deep exploration by 4 parallel agents (signals, metals, infrastructure, tests) plus
the 2026-04-20 dual adversarial review (118 findings, 27 P1).

## Exploration Summary

- **Signal system**: Persistence filter cold-start bug (cycles=2 instead of 1), duplicate ADX implementations
- **Metals subsystem**: `_execute_sell` exception can orphan Avanza positions, warrant trailing never disarms
- **Infrastructure**: `_RateLimiter.wait()` sleeps inside lock (thread bottleneck), `_fx_cache` no thread lock, `_loading_timestamps` leak on success path, `journal.py` non-atomic write
- **Tests**: Persistence filter UNTESTED, ADX has only 2 regime tests
- **Adversarial review (3rd consecutive)**: Drawdown circuit breaker still dead code, no OHLCV price validation, dashboard timing attack, telegram poller config wipe, no max order size

---

## Batch 1: Safety-Critical Core (5 files, ~60 lines)

### 1.1 indicators.py — OHLCV zero/negative price validation (P1, 3rd review)
**File:** `portfolio/indicators.py`, after line 37
**Problem:** NaN was fixed (BUG-87) but zero and negative prices propagate unchecked through all 33 signals. A single zero-price candle produces RSI=50, MACD=0, ATR=0 — poisoning consensus.
**Fix:** After the NaN guards (line 37), add: if any close values are ≤0, log WARNING and return None.
**Impact:** Prevents signal poisoning during Binance maintenance or API glitches.
**Risk:** LOW — rejects bad data early, same pattern as existing NaN guard.

### 1.2 dashboard/app.py — Timing-safe token comparison (P1, 3rd review)
**File:** `dashboard/app.py`, lines 675 and 682
**Problem:** `token == expected` is vulnerable to timing attacks. Combined with wildcard CORS, token is brute-forceable from LAN.
**Fix:** Replace `==` with `hmac.compare_digest()` (2 lines).
**Impact:** Prevents timing-based token extraction.
**Risk:** VERY LOW — drop-in replacement, same semantics.

### 1.3 telegram_poller.py — Config wipe guard (P1)
**File:** `portfolio/telegram_poller.py`, lines 198-208
**Problem:** If config.json is momentarily unreadable, `cfg = {}`, then `atomic_write_json` overwrites with nearly-empty config. All API keys destroyed.
**Fix:** After loading cfg, guard: `if len(cfg) < 5: logger.error(...); return`. Refuse to write suspiciously small configs.
**Impact:** Prevents catastrophic config loss from transient file access issues.
**Risk:** VERY LOW — only affects the write path, adds a guard.

### 1.4 avanza_session.py — Maximum order size limit (P1)
**File:** `portfolio/avanza_session.py`, line 592 (after minimum check)
**Problem:** Minimum 1000 SEK enforced, no maximum. A malformed call could commit entire account in one trade.
**Fix:** Add `MAX_ORDER_TOTAL_SEK = 50_000` constant and guard: `if order_total > MAX_ORDER_TOTAL_SEK: raise ValueError(...)`. 50K is ~25% of a 200K ISK account, generous enough for metals warrants but prevents full-account bets.
**Impact:** Prevents total account exposure from single bug.
**Risk:** LOW — only affects order placement. May need adjustment if user wants larger single trades.

### 1.5 shared_state.py — Rate limiter sleep-outside-lock + _loading_timestamps leak (P2)
**File:** `portfolio/shared_state.py`
**Problem A (line 255-262):** `_RateLimiter.wait()` sleeps while holding `self._lock`. With 8-worker ThreadPoolExecutor, all threads serialize on the same rate limiter.
**Fix:** Calculate wait time inside lock, release lock, sleep, re-acquire to update `last_call`.
**Problem B (line 96):** `_cached()` success path pops `_loading_keys` but not `_loading_timestamps`. Keys leak until 120s eviction.
**Fix:** Add `_loading_timestamps.pop(key, None)` on success path.
**Impact:** (A) Reduces thread contention; (B) Prevents memory leak of timestamps.
**Risk:** LOW — (A) changes timing slightly but preserves rate-limiting semantics; (B) cleanup only.

---

## Batch 2: Drawdown Circuit Breaker + I/O Safety (5 files, ~50 lines)

### 2.1 agent_invocation.py — Wire check_drawdown() into production (P1, 3rd review)
**File:** `portfolio/agent_invocation.py`, in `invoke_agent()` before prompt building
**Problem:** `check_drawdown()` has 16 passing tests but zero production callers. No automated risk enforcement on the primary trading path.
**Fix:** Call `check_drawdown()` for both Patient and Bold portfolios before invoking Layer 2. Include drawdown data in the agent prompt. Hard-block new trade invocations only when drawdown > 50% (respecting user's high risk tolerance per `memory/feedback_risk_tolerance.md`). Log WARNING at >20%.
**Impact:** Adds first-ever automated drawdown awareness to the trading path.
**Risk:** MEDIUM — adds a gate to the invocation path. Advisory-only below 50%, so normal trading unaffected. Hard-block at 50% is consistent with user's stated risk preferences.

### 2.2 fx_rates.py — Thread-safe cache (P2)
**File:** `portfolio/fx_rates.py`, line 16
**Problem:** `_fx_cache` is a plain dict accessed from 8-worker ThreadPoolExecutor with no lock. Concurrent reads/writes can produce inconsistent state.
**Fix:** Add `threading.Lock` and wrap cache reads/writes.
**Impact:** Prevents inconsistent FX rate reads.
**Risk:** LOW — adds synchronization to existing code.

### 2.3 journal.py — Atomic file write (P2)
**File:** `portfolio/journal.py`, lines 568 and 580
**Problem:** `CONTEXT_FILE.write_text()` is truncate-then-write. If process crashes mid-write, Layer 2 reads a partial context file.
**Fix:** Replace with `file_utils.atomic_write_text()` or write-to-temp-then-rename pattern.
**Impact:** Prevents corrupted context file.
**Risk:** LOW — same content, safer write path.

### 2.4 monte_carlo.py, monte_carlo_risk.py, data/metals_risk.py — Remove seed=42 (P2)
**Files:** 3 files
**Problem:** `seed=42` produces identical simulation paths every run. Risk metrics (VaR, CVaR) never change regardless of market conditions. Risk metrics are theater.
**Fix:** Use `seed=None` (system entropy) for production runs. Keep `seed=42` only in tests.
**Impact:** Risk simulations actually reflect current conditions.
**Risk:** LOW — Monte Carlo results change per run (correct behavior). Downstream consumers only use these as informational data in agent summaries.

---

## Batch 3: Signal + Metals Fixes (3 files, ~30 lines)

### 3.1 signal_engine.py — Persistence filter cold-start fix
**File:** `portfolio/signal_engine.py`, line 260
**Problem:** On cold-start, non-HOLD votes initialize with `cycles=_PERSISTENCE_MIN_CYCLES` (2). This means the signal passes through on cycle 2 without actually being persistent. Should initialize at 1 so the signal needs 1 more cycle to qualify.
**Fix:** Change `cycles=_PERSISTENCE_MIN_CYCLES` to `cycles=1` for cold-start initialization.
**Impact:** Prevents noisy first-cycle signals from bypassing the persistence filter.
**Risk:** LOW — cold-start only, makes the filter stricter (correct behavior).

### 3.2 metals_swing_trader.py — _execute_sell exception safety
**File:** `data/metals_swing_trader.py`, around line 2839
**Problem:** If `_execute_sell()` raises before setting `sell_failed_at`, the position is added to `to_remove` and deleted from state — but the position may still exist on Avanza. Orphaned holding.
**Fix:** Wrap `_execute_sell()` call in try/except. On exception, set `pos["sell_failed_at"]` and log WARNING. Don't add to `to_remove`.
**Impact:** Prevents orphaned Avanza positions.
**Risk:** MEDIUM — changes exit flow, but the fix is strictly additive (adds safety net).

### 3.3 econ_calendar.py — Force-HOLD (compromise fix for SELL-only bias)
**File:** `portfolio/signals/econ_calendar.py`
**Problem:** All 4 sub-signals can only produce SELL or HOLD, never BUY. Permanent SELL-biased voter.
**Fix:** Add `econ_calendar` to `DISABLED_SIGNALS` in signal config (force-HOLD). A proper fix (adding BUY capability) requires research into what economic events are bullish. Force-HOLD is the safe compromise.
**Impact:** Removes systematic SELL bias from consensus.
**Risk:** LOW — disabling a biased signal is safer than leaving it active.

---

## Batch 4: Tests (test files only)

### 4.1 test_persistence_filter.py — New test file
- Cold-start: first cycle should NOT pass persistence filter
- Normal operation: signal must persist 2+ cycles
- Direction flip: resets counter
- HOLD transitions: HOLD votes pass through
- Thread safety: concurrent calls don't corrupt state

### 4.2 test_safety_guards.py — New test file
- OHLCV zero/negative price rejection
- Dashboard hmac comparison
- Config wipe guard (small config rejection)
- Max order size limit enforcement
- Drawdown check integration (advisory at 20%, block at 50%)

### 4.3 Update existing tests
- `test_risk_management.py` — test drawdown block threshold
- `test_shared_state.py` — rate limiter concurrency, _loading_timestamps cleanup

---

## Dependency Order

Batch 1 (safety-critical) → Batch 2 (drawdown + I/O) → Batch 3 (signal + metals) → Batch 4 (tests)

Batches 1-3 items are internally independent (can be committed together).
Batch 4 depends on all implementation batches.

## What NOT to Implement (Deferred)

- **Agent invocation through claude_gate**: Too complex — requires refactoring subprocess lifecycle. Manual review needed.
- **Browser recovery idempotency**: Requires Avanza API research for idempotency keys. Skip with TODO.
- **Per-ticker accuracy filtering in ticker_accuracy.py/SignalDB**: Research priority, affects accuracy display, not trading.
- **funding_rate threshold symmetry**: Needs statistical analysis of optimal thresholds.
- **Metals loop split (7634 lines)**: Too large for auto-improvement.
- **DST hardcoding fixes**: Requires testing across DST transitions. Skip with TODO.
- **calendar_seasonal/network_momentum signal bias**: Needs research.
