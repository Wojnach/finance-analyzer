# Improvement Plan — Auto-Session #3 (2026-02-28)

## Priority: Critical Bugs > Architecture > Features > Polish

Previous sessions fixed BUG-13 through BUG-17, ARCH-1/2, FEAT-1, REF-5/6/7.
This session continues from BUG-18 onward.

---

## 1. Bugs & Problems Found

### BUG-18: futures_flow.py majority_vote() called with dict — signal NEVER works
- **File:** `portfolio/signals/futures_flow.py:~265`
- **Severity:** CRITICAL (signal #30 always returns HOLD with 0.0 confidence)
- **Issue:** `majority_vote(sub)` is called where `sub` is a dict mapping names→votes. `majority_vote()` iterates keys (strings like `"oi_trend"`), not values (`"BUY"`/`"SELL"`). No key equals "BUY"/"SELL"/"HOLD", so the function always returns `("HOLD", 0.0)`.
- **Fix:** Change to `majority_vote(list(sub.values()))`.
- **Impact:** Futures flow signal will start producing real votes for BTC-USD and ETH-USD. This changes consensus math for crypto tickers.

### BUG-19: momentum.py variable shadowing — RSI Divergence and StochRSI never work
- **File:** `portfolio/signals/momentum.py:~46, ~127`
- **Severity:** CRITICAL (2 of 8 sub-signals silently produce HOLD always)
- **Issue:** `rsi = rsi(close)` assigns to a local variable named `rsi`, which shadows the imported `rsi` function. Python sees the local assignment and treats `rsi` as a local variable throughout the function body — so `rsi(close)` on the right side raises `UnboundLocalError` (trying to read before assignment). The `try/except Exception` wrapper silently returns HOLD.
- **Fix:** Rename local variable to `rsi_values = rsi(close)`.
- **Impact:** RSI Divergence and StochRSI sub-signals will start voting in momentum composite signal.

### BUG-20: momentum_factors.py 500-bar requirement — two sub-signals permanently HOLD
- **File:** `portfolio/signals/momentum_factors.py:~123, ~140`
- **Severity:** HIGH (2 of 7 sub-signals permanently inactive)
- **Issue:** `_high_proximity` and `_low_reversal` require `len(close) >= 500` bars. Even the 6mo timeframe typically has ~180 daily bars. These sub-signals never generate BUY/SELL votes.
- **Fix:** Reduce to 252 (1 trading year) for stocks, keep 365 for crypto. Use `len(close)` as the lookback when data is shorter.
- **Impact:** High/low proximity sub-signals will start voting when sufficient data exists.

### BUG-21: heikin_ashi.py count_hold=True — systematically lower confidence
- **File:** `portfolio/signals/heikin_ashi.py:~480`
- **Severity:** HIGH (weighted consensus treats heikin_ashi as less confident than it actually is)
- **Issue:** Uses `majority_vote(signals, count_hold=True)` while ALL other signal modules use the default `count_hold=False`. With `count_hold=True`, HOLD votes are in the denominator: 4B/1S/2H → confidence 4/7=0.57 instead of 4/5=0.80. This makes heikin_ashi signal confidence systematically 20-40% lower than equivalent conviction in other signals.
- **Fix:** Remove `count_hold=True` to match all other modules.
- **Impact:** Heikin-ashi confidence values increase. If heikin-ashi has good accuracy, this strengthens its influence in weighted consensus.

### BUG-22: trend.py `is np.nan` identity comparison
- **File:** `portfolio/signals/trend.py:45`
- **Severity:** MEDIUM (can silently skip MA200 sub-signal when data is valid)
- **Issue:** `sma50.iloc[-1] is np.nan` uses Python identity check. NaN objects from pandas operations are not guaranteed to be the same object as `np.nan`. The check can fail to detect NaN or false-positive on valid values.
- **Fix:** Replace with `pd.isna(sma50.iloc[-1])`.
- **Impact:** MA200 sub-signal reliability improves.

### BUG-23: fx_rates.py staleness check unreachable
- **File:** `portfolio/fx_rates.py:~19`
- **Severity:** MEDIUM (stale FX data never triggers a warning)
- **Issue:** The stale check (`age_secs > _FX_STALE_THRESHOLD`) is inside the TTL guard (`now - _fx_cache["time"] < 3600`). Data within the 1h TTL cannot be >2h stale, so the warning never fires.
- **Fix:** Move stale check outside the TTL guard, check on every return of cached data.
- **Impact:** Users get warned when FX data is stale. No functional change.

### BUG-24: health.py uptime_seconds inherits previous session
- **File:** `portfolio/health.py:22`
- **Severity:** MEDIUM (uptime reported incorrectly after restart)
- **Issue:** `state["uptime_seconds"] = time.time() - state.get("start_time", time.time())`. If the loop restarts, `start_time` from the previous session is inherited, making uptime appear continuous.
- **Fix:** Reset `start_time` at loop startup (in `main.py` or `health.py` init).
- **Impact:** Health reporting accuracy. No functional impact.

### BUG-25: reporting.py KeyError on old portfolio state files
- **File:** `portfolio/reporting.py:50`
- **Severity:** MEDIUM (crash if portfolio state was saved before `initial_value_sek` was added)
- **Issue:** `state["initial_value_sek"]` accessed without `.get()` guard. Old state files may not have this field.
- **Fix:** Use `state.get("initial_value_sek", 500000)` (matches pattern on line 300).
- **Impact:** Prevents crash on legacy state files.

### BUG-26: trigger.py hardcoded market open hour
- **File:** `portfolio/trigger.py:298`
- **Severity:** LOW-MEDIUM (silent drift if market_timing.py is updated)
- **Issue:** `7 <= now_utc.hour < close_hour` hardcodes 7 instead of importing `MARKET_OPEN_HOUR` from `market_timing.py`.
- **Fix:** Import and use the constant.
- **Impact:** trigger.py stays in sync with market_timing.py.

### BUG-27: trend.py Ichimoku dead code
- **File:** `portfolio/signals/trend.py:319-323`
- **Severity:** LOW (wasted computation, no incorrect behavior)
- **Issue:** `tenkan = _midline(close, 9)` computed on line 319, then immediately overwritten on line 322. Same for `kijun`.
- **Fix:** Remove the dead `_midline()` calls.
- **Impact:** Minor performance improvement. No behavioral change.

### BUG-28: signal_engine.py CORE_SIGNAL_NAMES recreated on every call
- **File:** `portfolio/signal_engine.py:571-574`
- **Severity:** LOW (unnecessary allocation per signal computation)
- **Issue:** `CORE_SIGNAL_NAMES` set is defined inside `generate_signal()` body, recreated thousands of times per day.
- **Fix:** Hoist to module-level constant.
- **Impact:** Minor memory/CPU improvement.

### BUG-29: smart_money.py O(n²) FVG scan without break
- **File:** `portfolio/signals/smart_money.py:~290`
- **Severity:** LOW (performance only, no incorrect behavior)
- **Issue:** Inner FVG fill-check loop continues scanning after gap is filled (no `break`).
- **Fix:** Add `break` after `filled = True`.
- **Impact:** Faster FVG computation for large datasets.

---

## 2. Architecture Improvements

### ARCH-3: Eliminate redundant disk I/O in trigger path
- **Files:** `portfolio/trigger.py`, `portfolio/main.py`
- **Why:** trigger_state.json is read 3x and written 2x per triggered cycle (check_triggers, classify_tier, update_tier_state). Wasteful and adds latency.
- **Change:** Return state dict from `check_triggers()`, pass to `classify_tier()` and `update_tier_state()`. Add `state` parameter to `update_tier_state()`.
- **Impact:** trigger.py, main.py. Saves 2 disk reads per triggered cycle.

### ARCH-4: Deduplicate reporting.py constants
- **Files:** `portfolio/reporting.py`
- **Why:** `KEEP_EXTRA_FULL` set defined identically on lines 419 and 693. If one is updated without the other, tiered summary generation diverges.
- **Change:** Extract to module-level `_KEEP_EXTRA_FULL` constant.
- **Impact:** reporting.py only.

### ARCH-5: Extract post-run hooks in main.py
- **Files:** `portfolio/main.py`
- **Why:** Digest/hook code is copy-pasted in two `loop()` paths (~15 lines each). DRY violation.
- **Change:** Extract into `_post_run_hooks(config)` helper.
- **Impact:** main.py only.

---

## 3. Refactoring TODOs

### REF-8: Fix candlestick.py unused import
- **File:** `portfolio/signals/candlestick.py`
- **Why:** `majority_vote` imported but never called (module uses custom confidence formula).
- **Change:** Remove unused import.
- **Impact:** candlestick.py only.

### REF-9: Remove redundant defaultdict import in equity_curve.py
- **File:** `portfolio/equity_curve.py:~344`
- **Why:** `from collections import defaultdict` appears both at module level and inside `_pair_round_trips`.
- **Change:** Remove the inner import.
- **Impact:** equity_curve.py only.

### REF-10: Remove redundant double-import in heikin_ashi.py
- **File:** `portfolio/signals/heikin_ashi.py:28,45`
- **Why:** Two separate import lines from `signal_utils`.
- **Change:** Merge into single import statement.
- **Impact:** heikin_ashi.py only.

---

## 4. Items NOT Planned (Justified)

1. **journal_index.py 4h half-life** — Changing this affects journal retrieval for all Layer 2 invocations. Needs analysis of how Layer 2 actually uses retrieved entries. Too risky without validation data.
2. **outcome_tracker.py yfinance daily close for stocks** — The intraday resolution issue (3h/1d horizons identical) is real but requires a new data source. yfinance free tier doesn't support intraday historical. Would need Alpaca historical bars API.
3. **calendar_seasonal.py holiday date accuracy** — Approximated holidays are off by 1-2 days. Fixing requires a proper holiday calendar library (e.g., `holidays` package). Not worth adding a dependency for ±1 day accuracy.
4. **trade_guards.py disk I/O per check** — 54 disk reads per cycle is wasteful but the state file is tiny (<1KB). Adding an in-memory cache adds complexity for negligible wall-clock benefit.
5. **smart_money.py supply/demand margin formula** — The `proximity_pct / 0.005` factor is a no-op at current config but would break if config changes. Documenting rather than fixing because the current behavior is correct.
6. **digest.py signal_log format mismatch** — The digest reads `signals` key but newer entries use `tickers`. Fixing requires understanding the full schema evolution. Documenting for manual review.
7. **forecast.py Kronos candle fallback** — The fallback path never works as designed. But Kronos typically receives candles from the primary path. Low-priority dead-code fix.

---

## 5. Dependency/Ordering — Implementation Batches

### Batch 1: Critical signal fixes (3 files)
**Files:** `portfolio/signals/futures_flow.py`, `portfolio/signals/momentum.py`, `portfolio/signals/momentum_factors.py`
**Changes:** BUG-18, BUG-19, BUG-20
**Tests needed:** Verify each signal produces non-HOLD votes with valid data
**Risk:** LOW — these signals are currently producing no useful data; fixing them can only improve accuracy

### Batch 2: Signal confidence/correctness (3 files)
**Files:** `portfolio/signals/heikin_ashi.py`, `portfolio/signals/trend.py`, `portfolio/signals/smart_money.py`
**Changes:** BUG-21, BUG-22, BUG-27, BUG-29, REF-10
**Tests needed:** heikin_ashi confidence matches expected pattern; trend NaN handling
**Risk:** LOW — heikin_ashi confidence change is most impactful (weighted consensus)

### Batch 3: Core infrastructure fixes (4 files)
**Files:** `portfolio/fx_rates.py`, `portfolio/health.py`, `portfolio/reporting.py`, `portfolio/signal_engine.py`
**Changes:** BUG-23, BUG-24, BUG-25, BUG-28
**Tests needed:** fx_rates staleness warning fires; health uptime resets; reporting handles missing initial_value_sek
**Risk:** LOW — all fixes are defensive guards or performance improvements

### Batch 4: Architecture cleanup (3 files)
**Files:** `portfolio/trigger.py`, `portfolio/main.py`, `portfolio/reporting.py`
**Changes:** BUG-26, ARCH-3, ARCH-4, ARCH-5
**Tests needed:** trigger uses MARKET_OPEN_HOUR; state threading doesn't break existing trigger tests
**Risk:** MEDIUM — main.py and trigger.py changes touch the hot path; must verify all trigger tests pass

### Batch 5: Minor cleanup (2 files)
**Files:** `portfolio/signals/candlestick.py`, `portfolio/equity_curve.py`
**Changes:** REF-8, REF-9
**Tests needed:** None (import-only changes)
**Risk:** NONE
