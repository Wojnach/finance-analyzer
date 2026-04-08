# Independent Adversarial Review — Claude Direct (2026-04-08 Round 2)

Reviewer: Claude Opus 4.6 (direct analysis, independent from subagent reviews)
Date: 2026-04-08
Scope: All 8 subsystems of finance-analyzer

---

## 1. signals-core

### CRITICAL

**C1. ADX cache keyed by `id(df)` — stale data from reused memory addresses**
- File: `portfolio/signal_engine.py:25`
- `_adx_cache` uses `id(df)` as the cache key. Python's `id()` returns the object's
  memory address, which is reused after garbage collection. A new DataFrame allocated
  at the same address as a previously-cached (and now GC'd) one will return the old
  ADX value. This can produce silently wrong ADX readings that affect Stage 2
  volume/ADX gating (line 1040), potentially allowing or blocking trades incorrectly.
- Impact: Wrong ADX → wrong volume/ADX gate → wrong trade decisions.
- The `_ADX_CACHE_MAX=200` cap + eviction reduces probability but doesn't eliminate it.
- Fix: Use a content-based hash (e.g., hash of df shape + last few values) instead of
  `id(df)`.

### HIGH

**H1. Flush sentiment state TOCTOU race between snapshot and dirty-flag clear**
- File: `portfolio/signal_engine.py:127-148`
- `flush_sentiment_state()` takes a snapshot under `_sentiment_lock`, writes to disk
  OUTSIDE the lock, then re-acquires the lock to clear `_sentiment_dirty`. Between
  the snapshot and the second lock acquisition, `_set_prev_sentiment()` can set
  `_sentiment_dirty=True` and mutate `_prev_sentiment`. The subsequent
  `_sentiment_dirty=False` clears the flag even though the new mutation wasn't
  persisted.
- Impact: Sentiment hysteresis state can be silently lost on the next cycle, causing
  redundant direction flips.

**H2. Dynamic correlation group merging can create degenerate mega-groups**
- File: `portfolio/signal_engine.py:508-591`
- The greedy union-find clustering merges any two groups that share a signal with
  r > 0.7. Transitive chains (A corr B > 0.7, B corr C > 0.7, but A corr C < 0.5)
  can create large groups where 4-5 signals get 0.3x penalty while only one gets
  full weight. This concentrates voting power in one signal per mega-group.
- Impact: Reduced effective signal count → more HOLD outcomes → missed trades.

**H3. Utility boost can exceed accuracy cap of 0.95**
- File: `portfolio/signal_engine.py:1694`
- `boosted_acc = min(accuracy_data[sig_name]["accuracy"] * boost, 0.95)` caps at 0.95.
  But `boost = min(1.0 + u_score, 1.5)` and `accuracy` can already be high (e.g., 0.8).
  0.8 * 1.5 = 1.2, capped to 0.95. This means signals with 64%+ accuracy AND positive
  utility all get boosted to the same 0.95 ceiling, destroying their relative ranking
  in weighted consensus.

### MEDIUM

**M1. Per-ticker accuracy override may use stale cached data**
- File: `portfolio/signal_engine.py:1655-1663`
- Per-ticker accuracy data (`_ticker_acc_data`) is populated once from
  `accuracy_by_ticker_signal_cached()` and overrides global accuracy. But this cache
  has a TTL (typically 3600s). During a fast-moving regime change, per-ticker data
  from the previous hour is used, potentially gating signals that have already
  recovered.

**M2. `_weighted_consensus` — gated signals not returned to caller**
- File: `portfolio/signal_engine.py:806-807`
- `gated_signals` list is logged at DEBUG level but not returned. The caller has no
  way to know which signals were gated or why. This makes debugging consensus
  decisions require log analysis.

### LOW

**L1. `CORE_SIGNAL_NAMES` includes `claude_fundamental` but it's often unavailable**
- File: `portfolio/signal_engine.py:78-81`
- `claude_fundamental` is in `CORE_SIGNAL_NAMES` but requires Claude API calls that
  frequently time out or are rate-limited. When it votes HOLD (timeout), it doesn't
  count as a core voter. This is correct behavior but may reduce core_active count
  below expectations.

---

## 2. orchestration

### HIGH

**H4. Agent subprocess not killed on Windows when kill fails**
- File: `portfolio/agent_invocation.py:168-201`
- If `taskkill` fails (rc != 0), `kill_ok = False` and no new agent is spawned. But
  the old `_agent_proc` is set to `None` (line 202), losing the reference. On the
  next trigger, `_agent_proc is None` → a new agent IS spawned (line 164 check
  passes). This contradicts the intent of "don't spawn if old process may still be
  running."
- Actually, line 200 sets `_agent_proc = None` only inside the `if not kill_ok` branch
  and returns False. So the reference is lost. Next call: `_agent_proc` is None →
  `_agent_proc and _agent_proc.poll()` is False → new agent spawns regardless.
- Impact: Potential duplicate Layer 2 agents running simultaneously.

**H5. Stack overflow counter never resets**
- File: `portfolio/agent_invocation.py:36-38`
- `_consecutive_stack_overflows` counts up but is never decremented on success. Once
  it hits `_MAX_STACK_OVERFLOWS=5`, Layer 2 is permanently disabled until the loop
  process restarts. A transient Node.js issue could permanently disable the trading
  agent.

### MEDIUM

**M3. Trigger state persisted with `set()` type that's not JSON-serializable**
- File: `portfolio/trigger.py:100`
- `state["_current_tickers"] = set(signals.keys())` assigns a Python `set` to the
  state dict. This field is removed before save (`_save_state` pops it at line 62),
  but if `_save_state` throws before reaching the pop, `atomic_write_json` would
  fail on the set (not JSON-serializable), and trigger state would not be saved.

**M4. Classify_tier market-open detection may be wrong during DST transition week**
- File: `portfolio/trigger.py:300-301`
- `now_utc.hour` is compared to `eu_open` and `close_hour`, but the comparison is
  integer-only. A DST transition at 01:00 UTC could cause the hour boundary to shift
  mid-day in ways that `_eu_market_open_hour_utc` doesn't account for if the DST
  detection function caches the result.

### LOW

**L2. Re-import of `logging` inside `check_triggers`**
- File: `portfolio/trigger.py:108`
- `import logging` is repeated inside the function body (already imported at module
  level). Harmless but unnecessary.

---

## 3. portfolio-risk

### HIGH

**H6. Drawdown check silent on corrupt portfolio file**
- File: `portfolio/risk_management.py:74`
- `load_json(portfolio_path, default={})` returns empty dict on corruption. With no
  holdings and no `cash_sek` key, `current_value = portfolio.get("cash_sek", initial_value)`
  returns 500K (the default initial_value). Drawdown = 0% — circuit breaker never fires.
- Impact: A corrupt portfolio state file silently disables the drawdown circuit breaker.

**H7. `_compute_portfolio_value` uses fallback avg_cost_usd when no live price**
- File: `portfolio/risk_management.py:44-46`
- If a ticker isn't in `agent_summary.signals`, the function uses `avg_cost_usd` from
  holdings as the price. This means the portfolio value never shows losses for tickers
  that aren't in the current summary — effectively hiding drawdowns for stocks after
  hours or during data outages.

### MEDIUM

**M5. Stop-loss floor at 1% of entry makes stops meaningless for expensive assets**
- File: `portfolio/risk_management.py:184-186`
- When `2 * atr_pct > 100` (ATR > 50%), stop is floored to 1% of entry. This
  creates an essentially useless stop. ATR > 50% itself should trigger an alert —
  it means the data is likely wrong (few assets have 50% 14-period ATR).

**M6. Monte Carlo simulation uses `n_paths=2000` — may be insufficient for tail risk**
- File: `portfolio/risk_management.py:283-289`
- 2000 paths gives ~95% confidence intervals but poor tail risk estimates (VaR at
  99% needs ~10K+ paths for stability). The `stop_hit_prob` at 2000 paths has
  ±2.2% sampling error at the 5% probability level.

---

## 4. metals-core

### CRITICAL

**C2. metals_loop.py: 6561-line god file with global state**
- File: `data/metals_loop.py`
- This file manages real money through a single global Playwright page (`_loop_page`),
  global price histories, global fish engine state, and global stop-order tracking.
  Any uncaught exception in any section can corrupt shared state. The file is too
  large to reason about safely and too entangled to test in isolation.
- Impact: This is the highest systemic risk in the codebase. Real money positions can
  be left unprotected if any component fails.

### HIGH

**H8. Fish engine sell path: Playwright page can silently die**
- File: `data/metals_loop.py:2158`
- `if _loop_page is None` check doesn't detect a detached/crashed Playwright page.
  The page object can become unusable (browser crashed, connection dropped) while
  still being non-None. Subsequent API calls will throw, and the exception handler
  sends a Telegram but doesn't attempt recovery.

**H9. Price history deques have no explicit max length**
- File: `data/metals_loop.py:2246-2248`
- `_gold_price_history` and `_silver_price_history_fish` are appended to every cycle
  (60s). Without a maxlen constraint, after 24 hours that's 1440 entries. After a
  week, 10080. Over months of continuous operation, this is a memory leak.

**H10. Fish engine BUY uses Kelly but falls back to fixed 1500 SEK**
- File: `data/metals_loop.py:2110`
- When Kelly says "no edge" (negative or zero Kelly fraction), the code falls back
  to fixed sizing capped at 1500 SEK. The comment says "Fishing is contrarian; Kelly's
  historical win rate may not reflect the current oversold/overbought setup." This
  overrides Kelly's risk signal, which exists specifically to prevent taking positions
  without edge.

### MEDIUM

**M7. ORB predictor calls `predictor._parse_klines(raw)` — private method**
- File: `data/metals_loop.py:2293`
- Calling a private method (`_parse_klines`) from outside the class creates a fragile
  dependency. If `ORBPredictor` refactors its internal parsing, metals_loop breaks.

---

## 5. avanza-api

### HIGH

**H11. Session expiry ValueError silently caught — treats corrupt session as valid**
- File: `portfolio/avanza_session.py:76-77`
- `except ValueError: pass` on ISO format parsing means a corrupted `expires_at` field
  (e.g., "2026-04-32T00:00:00" or "null") won't trigger expiry detection. The session
  is treated as valid indefinitely.
- Impact: An expired session could be used for trading, causing 401 errors on order
  placement.

**H12. Playwright context lazy-init not resilient to browser crash**
- File: `portfolio/avanza_session.py:116-135`
- `_get_playwright_context()` creates the browser once and reuses it. If the browser
  process dies (OOM, GPU crash), `_pw_context` is still non-None but unusable. The
  next API call will throw an exception. `close_playwright()` is called on 401, but
  not on arbitrary Playwright errors.

**H13. No account ID validation on order placement**
- File: `portfolio/avanza_session.py:32`
- `DEFAULT_ACCOUNT_ID = "1625505"` is correct, but `api_post` doesn't validate that
  the payload's `accountId` matches the allowed account. If a bug passes the pension
  account ID (2674244), the order goes through.

### MEDIUM

**M8. `api_delete` generic 404-as-success masks unrelated 404s**
- File: `portfolio/avanza_session.py:287`
- The function is used for stop-loss deletion where 404 (already deleted) is correct.
  But as a generic DELETE helper, it could hide bugs when called on non-existent
  resources.

---

## 6. signals-modules

### HIGH

**H14. Swing detection can't detect swings in last `lookback` bars**
- File: `portfolio/signals/smart_money.py:46-79`
- `_find_swing_highs` requires `lookback` bars on EACH side of a potential swing point.
  The last 3 bars (with default `_SWING_LOOKBACK=3`) can never be identified as swings.
  For intraday trading with 15m candles, this means the most recent 45 minutes of
  structure is invisible.

### MEDIUM

**M9. Signal modules silently return HOLD on data shorter than threshold**
- All signal modules in `portfolio/signals/` check `len(df) < MIN_ROWS` and return
  HOLD. This means if data collection partially fails (returning only 20 candles instead
  of 100), ALL enhanced signals vote HOLD, making consensus impossible. The failure is
  not surfaced to the signal engine — it looks like a legitimate HOLD.

**M10. Calendar/seasonal signal has hardcoded FOMC dates**
- File: `portfolio/signals/econ_calendar.py`
- FOMC dates for 2026-2027 are hardcoded. After 2027, the signal will stop detecting
  FOMC proximity and will always return HOLD for that sub-indicator.

**M11. Fibonacci levels assume trend detection works correctly**
- File: `portfolio/signals/fibonacci.py`
- Fibonacci retracement levels are calculated from swing high/low detection. If the
  underlying swing detection (from smart_money or structure modules) is wrong, Fibonacci
  levels are wrong. There's no cross-validation between the two modules' swing detection
  implementations.

---

## 7. data-external

### HIGH

**H15. `_cached` key collisions possible with user-constructed keys**
- File: `portfolio/shared_state.py:34`
- The cache is keyed by string. Keys are constructed as `f"fear_greed_{ticker}"`,
  `f"sentiment_{ticker}"`, etc. If a ticker name contains an underscore that matches
  another cache key pattern, collision occurs. E.g., a hypothetical ticker "USD_RATE"
  would make `f"fear_greed_USD_RATE"` — unlikely but the pattern is fragile.

**H16. Alpha Vantage daily budget not enforced**
- File: `portfolio/shared_state.py:186`
- `_alpha_vantage_limiter = _RateLimiter(5, "alpha_vantage")` enforces 5/min but the
  API's true limit is 25/day (free tier). The per-minute limiter allows up to
  `5 * 60 * 24 = 7200` calls/day. After hitting the daily limit, all Alpha Vantage
  calls fail until midnight, and the fundamentals cache goes stale.

### MEDIUM

**M12. Circuit breaker recovery timeout (60s) may be too short for persistent outages**
- File: `portfolio/data_collector.py:21-23`
- `CircuitBreaker("binance_spot", failure_threshold=5, recovery_timeout=60)` opens
  after 5 failures and retries after 60s. During a sustained Binance outage (common
  during high-volume events), the circuit breaker oscillates between open and half-open,
  generating warnings every 60 seconds forever.

**M13. yfinance fallback uses stale data for stocks after hours**
- File: `portfolio/data_collector.py:36-42`
- `_YF_INTERVAL_MAP["15m"]` fetches 5 days of 15m data. After US market close, this
  data is stale (no new candles). The system continues generating signals from this
  stale data without marking it as potentially outdated.

---

## 8. infrastructure

### HIGH

**H17. `update_health` read-modify-write race condition**
- File: `portfolio/health.py:16-36`
- `update_health()` calls `load_health()` (reads file), modifies the dict in-memory,
  then calls `atomic_write_json()`. With 8 ThreadPoolExecutor workers potentially
  triggering health updates via `update_signal_health_batch`, concurrent calls can
  overwrite each other's changes (lost update).
- Impact: Signal health statistics may be inconsistently tracked, and error counts
  may be lower than actual.

### MEDIUM

**M14. `atomic_append_jsonl` — partial writes on crash**
- File: `portfolio/file_utils.py:155-168`
- Opens file in append mode and writes. If process crashes mid-write, a partial JSON
  line is left. Subsequent reads (via `load_jsonl`) skip it, but the file grows with
  garbage. Over months, this could accumulate.

**M15. `load_jsonl_tail` skips first valid line when offset lands on boundary**
- File: `portfolio/file_utils.py:135`
- When `offset > 0`, the first line is always skipped (assumed truncated). If `offset`
  exactly aligns with a line boundary, a valid entry is lost.

**M16. Telegram rate limiting relies on caller discipline**
- File: `portfolio/telegram_notifications.py`
- Telegram has strict rate limits (30 messages/second, ~20 messages/minute to same
  chat). The notification module doesn't enforce this internally — it relies on callers
  to use `message_throttle.py`. If a new caller bypasses the throttle, Telegram will
  temporarily ban the bot.

### LOW

**L3. `prune_jsonl` reads entire file into memory**
- File: `portfolio/file_utils.py:232-276`
- For large JSONL files (signal_log.jsonl can grow to 100K+ entries), `prune_jsonl`
  reads ALL lines into a list, then rewrites. For a 100K-line file at ~500 bytes/line,
  that's ~50MB in memory.

---

## Cross-Cutting Issues

### CRITICAL

**CC1. Pervasive `except Exception: pass` pattern masks real bugs**
- Throughout the codebase (signal_engine.py, agent_invocation.py, reporting.py, etc.),
  broad exception handlers with `pass` or `logger.debug()` are used for "graceful
  degradation." While individually defensible, the cumulative effect is that:
  - Bugs in signal modules are silently suppressed → wrong HOLD votes
  - Accuracy computation errors are hidden → wrong weights
  - Risk checks can be bypassed if their imports/computations fail
  - The system appears healthy while operating in a degraded state

### HIGH

**CC2. No end-to-end integration test for the signal pipeline**
- `generate_signal()` (signal_engine.py:1119-1869) is 750 lines and calls into 20+
  modules. There's no integration test that exercises the full pipeline with realistic
  data. Unit tests per-module can't catch interaction bugs (e.g., accuracy data format
  changes breaking weighted consensus).

**CC3. State file proliferation without garbage collection**
- The system writes to 30+ JSON/JSONL files in `data/`. There's no unified garbage
  collection or rotation for many of them (e.g., `invocations.jsonl`,
  `telegram_messages.jsonl`, `after-hours-research-log.jsonl`). Over months of
  operation, these files grow unbounded.

### MEDIUM

**CC4. Thread safety relies on per-module locking — no system-wide coordination**
- `signal_engine.py` has `_adx_lock` and `_sentiment_lock`.
- `portfolio_mgr.py` has per-file `_state_locks`.
- `shared_state.py` has `_cache_lock`.
- `health.py` has NO lock.
- There's no system-wide lock ordering convention, creating theoretical deadlock risk
  if two modules acquire each other's locks (currently unlikely due to call graph, but
  fragile under refactoring).

---

## Summary Statistics

| Severity | Count |
|----------|-------|
| CRITICAL | 3     |
| HIGH     | 17    |
| MEDIUM   | 16    |
| LOW      | 3     |
| **Total**| **39**|

## Top 5 Recommendations (Priority Order)

1. **Refactor metals_loop.py** (CC2/C2): Extract into 5-6 modules (data fetcher,
   signal processor, order manager, stop-loss manager, fish engine, telegram). This
   is the single highest-impact improvement for system reliability.

2. **Fix ADX cache key** (C1): Replace `id(df)` with a content hash. Low effort,
   high impact on decision quality.

3. **Add health file locking** (H17): Use the same `_get_lock()` pattern from
   `portfolio_mgr.py` in `health.py`.

4. **Add session recovery on Playwright errors** (H12): Catch Playwright-specific
   exceptions and call `close_playwright()` + retry, not just on 401.

5. **Add integration test for generate_signal** (CC2): One test with real OHLCV data
   that exercises the full pipeline from raw data to final action.
