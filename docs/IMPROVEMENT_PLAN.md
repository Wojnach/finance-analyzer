# Improvement Plan — Auto-Session 2026-04-08

Updated: 2026-04-08
Branch: improve/auto-session-2026-04-08
Status: **COMPLETE** — all items implemented, tested, and documented

**Source:** Deep exploration of all core modules, signal engine, portfolio management,
risk management, data collection, infrastructure, and test suite. Previous sessions
fixed BUG-80 through BUG-174 + REF-16 through REF-48. This session addresses critical
calculation bugs, reliability gaps, and cache efficiency.

---

## 1. Bugs & Problems Found

### BUG-176 (P1): Concentration check uses cash-only allocation, ignoring total portfolio value
- **File**: `portfolio/risk_management.py:584-585`
- **Problem**: `proposed_alloc = cash * alloc_pct` calculates the proposed position size as a
  percentage of *available cash* rather than *total portfolio value*. If a portfolio has 100K
  cash and 400K in existing positions, `alloc_pct=0.15` yields 15K (3% of total) instead of
  75K (15% of total). More critically, the concentration check compares
  `(existing_value + proposed_alloc) / total_value`, but the proposed_alloc is too small
  relative to a cash-heavy portfolio and too large relative to a fully-invested one.
- **Fix**: Change `proposed_alloc = cash * alloc_pct` to `proposed_alloc = total_value * alloc_pct`,
  capped at available cash. This makes the allocation proportional to total portfolio value.
- **Impact**: Affects Layer 2 trade sizing. All existing callers pass through; the function
  signature is unchanged. The concentration *warning* threshold (40%) still applies.
- **Risk**: Low — existing behavior allowed inconsistent sizing. The fix is more accurate.

### BUG-177 (P2): Sortino ratio unit mismatch makes downside detection ineffective
- **File**: `portfolio/equity_curve.py:244`
- **Problem**: `downside_returns = [r / 100 - daily_rf for r in daily_rets if r / 100 < daily_rf]`
  — `daily_rets` are in percentage units (e.g., 1.5 for +1.5%), and `r/100 = 0.015`.
  `daily_rf = 3.5% / 252 = 0.0001389`. The condition `0.015 < 0.0001389` is almost never
  true, so `downside_returns` is nearly always empty, and Sortino is never computed.
- **Fix**: Use `daily_rets_dec` (already computed on line 231) consistently for both Sharpe
  and Sortino. Replace the filter with: `[r - daily_rf for r in daily_rets_dec if r < daily_rf]`.
- **Impact**: Sortino ratio will now appear in equity curve reports (was silently missing).
- **Risk**: Zero — Sortino was effectively dead code before.

### BUG-178 (P1): No timeout on ThreadPoolExecutor.as_completed() in main loop
- **File**: `portfolio/main.py:514`
- **Problem**: `for future in as_completed(futures):` has no timeout. If any ticker's signal
  computation hangs, the entire 60s cycle blocks indefinitely.
- **Fix**: Add `timeout=120` to `as_completed()` (2x the normal cycle time). Catch
  `TimeoutError`, log which futures didn't complete, cancel them, and continue the cycle
  with partial results.
- **Impact**: Prevents indefinite hangs. Partial results already handled.
- **Risk**: Low — a 120s timeout is generous. Timed-out tickers get no signal that cycle.

### BUG-179 (P1): No timeout on ThreadPoolExecutor.as_completed() in data_collector
- **File**: `portfolio/data_collector.py:325`
- **Problem**: Same as BUG-178 but for per-ticker timeframe collection.
- **Fix**: Add `timeout=60` to `as_completed()`. Catch `TimeoutError`, log, return partial.
- **Impact**: Prevents per-ticker hangs from cascading to the main loop timeout.
- **Risk**: Zero — partial timeframe data is already supported downstream.

### BUG-180 (P2): ADX cache eviction clears all 200 entries instead of LRU
- **File**: `portfolio/signal_engine.py:981-982`
- **Problem**: When `_adx_cache` reaches 200 entries, `_adx_cache.clear()` removes ALL
  entries. With 20 tickers x 7 timeframes = 140 entries per cycle, overflow triggers often.
- **Fix**: Replace with LRU-style eviction: evict oldest 50% of entries (Python 3.7+ dict
  preserves insertion order). This keeps the most recent entries warm.
- **Impact**: Reduces redundant ADX computation by ~50% when cache overflows.
- **Risk**: Zero — cache is purely an optimization.

---

## 2. Architecture Improvements

### ARCH-29: Trade guards add should_block_trade() helper
- **File**: `portfolio/trade_guards.py`
- **Problem**: `check_overtrading_guards()` returns warnings but has no convenience function
  for go/no-go decisions. The C4 diagnostic warns guards are "NON-FUNCTIONAL".
- **Fix**: Add `should_block_trade(warnings)` that returns True if any warning has
  `severity="block"`. Purely additive — existing behavior unchanged.
- **Risk**: Zero — additive function.

---

## 3. Dependency/Ordering

### Batch 1: Critical calculation bugs (2 files + tests)
1. `portfolio/risk_management.py` — BUG-176
2. `portfolio/equity_curve.py` — BUG-177

### Batch 2: Reliability timeouts (2 files + tests)
1. `portfolio/main.py` — BUG-178
2. `portfolio/data_collector.py` — BUG-179

### Batch 3: Cache + trade guards (2 files + tests)
1. `portfolio/signal_engine.py` — BUG-180
2. `portfolio/trade_guards.py` — ARCH-29

### Batch 4: Documentation
1. `docs/SYSTEM_OVERVIEW.md` — Update with new bug IDs
2. `docs/CHANGELOG.md` — Add session entries
