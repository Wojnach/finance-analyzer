# Independent Adversarial Review — Finance Analyzer
**Reviewer**: Claude Opus 4.6 (Independent Pass)
**Date**: 2026-04-07
**Scope**: Full codebase, 8 subsystems, ~58K lines

---

## 1. SIGNALS-CORE

### CRITICAL: Accuracy data mutation can corrupt shared cache
**File**: `portfolio/signal_engine.py:1554`
**Impact**: Financial — wrong signal weights → wrong trade decisions

The utility boost loop modifies `accuracy_data` entries in-place. BUG-136 fix creates
new dicts, but the accuracy_data dict itself may be a reference to the cached alltime/recent
data from `load_cached_accuracy()`. If `blend_accuracy_data()` returns shallow copies,
per-ticker overrides (line 1518) mutate the shared cache entry for ALL subsequent callers
in the same cycle.

```python
accuracy_data[sig_name] = {  # line 1518 — overwrites blended entry
    "accuracy": t_stats["accuracy"],
    ...
}
```

**Fix**: Deep-copy accuracy_data before any mutation, or ensure `blend_accuracy_data()`
returns independent dicts.

### HIGH: _adx_cache keyed by `id(df)` — false positives after GC
**File**: `portfolio/signal_engine.py:25`
**Impact**: Stale ADX values served to wrong tickers

`id(df)` returns the memory address. After a DataFrame is garbage-collected, Python may
reuse that address for a different DataFrame. The cache eviction only happens when
`_ADX_CACHE_MAX` is reached (200 entries), but within a single cycle, up to 20 tickers
produce DataFrames. If any are GC'd before the cycle ends (e.g., due to memory pressure),
a new DataFrame could get the same `id()` and receive a cached ADX from a different ticker.

**Fix**: Key by `(id(df), len(df), hash of first/last rows)` or use weakref callbacks.

### HIGH: Regime gating exemption race with accuracy cache
**File**: `portfolio/signal_engine.py:1376-1393`
**Impact**: Inconsistent gating within the same cycle

`accuracy_by_ticker_signal_cached()` loads from disk cache. If the cache file is being
written by another process (e.g., outcome_tracker running in parallel), the loaded data
could be mid-write (despite atomic I/O, there's a window where the old data is already
replaced but the new data hasn't been fully read). More importantly, the function name
says "cached" but the cache freshness is uncontrolled — it could serve data from a
different accuracy horizon.

### MEDIUM: `_SENTIMENT_STATE_FILE` not included in rotating backups
**File**: `portfolio/signal_engine.py:81`
**Impact**: Sentiment hysteresis state lost on corruption

The sentiment state file uses `atomic_write_json` (good) but has no backup rotation
like portfolio_state.json. If the file corrupts, the hysteresis system resets, causing
a burst of rapid sentiment flip signals.

### MEDIUM: Group leader gate threshold hardcoded inside function body
**File**: `portfolio/signal_engine.py:596`
**Impact**: Maintenance risk — `_GROUP_LEADER_GATE_THRESHOLD = 0.46` is buried deep
inside `_weighted_consensus()`, making it hard to audit alongside other thresholds.

### LOW: `flush_sentiment_state()` dirty flag race
**File**: `portfolio/signal_engine.py:126-140`
**Impact**: Minimal — sentiment state written slightly more often than needed

The dirty flag is read inside the lock, but new sentiment updates could arrive between
the snapshot copy and the write completion. This is benign (the write includes latest data),
but the dirty flag could be set again immediately after being cleared.

---

## 2. ORCHESTRATION

### CRITICAL: `_process_ticker` exceptions swallowed without propagation
**File**: `portfolio/main.py:503`
**Impact**: Silent data loss — failed tickers produce no signals

When `_process_ticker` catches an exception, it returns `(name, None)`. The main loop
counts this as `signals_failed += 1` but continues processing. If a systemic error
(e.g., Binance API down) affects ALL tickers, the system produces zero signals and
writes an empty `agent_summary.json`. Layer 2 would then see no actionable signals
and do nothing — but the trigger system still runs with `signals = {}`, potentially
not triggering when it should.

**Fix**: Add a circuit breaker — if >50% of tickers fail, abort the cycle and alert.

### HIGH: ThreadPoolExecutor workers share `_regime_cache` unsafely
**File**: `portfolio/shared_state.py:151-153`
**Impact**: Race condition — regime cache could serve wrong regime for a ticker

`_regime_cache` and `_regime_cache_cycle` are protected by `_regime_lock`, but
`detect_regime()` in `indicators.py` may not consistently acquire this lock when
reading/writing the cache. If one thread writes a regime for BTC-USD while another
reads it for ETH-USD, the cache could serve BTC's regime for ETH.

### HIGH: Singleton lock file not robust on Windows
**File**: `portfolio/main.py:73`
**Impact**: Duplicate main loops running simultaneously

The singleton lock uses `msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)` which
locks exactly 1 byte. If the file handle leaks (e.g., process crash without cleanup),
the lock persists until the handle is GC'd or the process terminates. On Windows,
file handles are sometimes inherited by child processes (e.g., Claude CLI subprocess),
which could prevent restart.

**Fix**: Write PID to lock file and validate on acquisition that the PID is still alive.

### MEDIUM: Post-cycle tasks run synchronously after signal loop
**File**: `portfolio/main.py:262-371`
**Impact**: Cycle time bloat — 15+ post-cycle tasks run sequentially

Each post-cycle task (digest, daily_digest, alpha_vantage, metals_precompute, etc.)
runs synchronously. If any of them block (e.g., alpha_vantage rate limit sleep), the
entire 60s cycle is delayed. Some tasks (like metals_precompute) could take 30+ seconds.

**Fix**: Run post-cycle tasks in a separate thread or with a timeout.

### MEDIUM: `_extract_triggered_tickers` regex doesn't match all patterns
**File**: `portfolio/main.py:243-258`
**Impact**: Ticker extraction fails for some trigger reasons, defaulting to empty set

The regex `r'^([A-Z][A-Z0-9]*(?:-[A-Z]+)?)\s+(?:consensus|moved|flipped)'` requires
the ticker to be at the start of the string. Trigger reasons like "post-trade reassessment"
or "F&G crossed 20 (extreme fear)" won't match, causing `triggered_tickers` to be empty.
This affects `write_tiered_summary()` which uses triggered_tickers to filter the summary.

### LOW: `_startup_grace_active` is a module-level global
**File**: `portfolio/trigger.py:40`
**Impact**: If trigger.py is re-imported (hot reload), grace period resets

---

## 3. PORTFOLIO-RISK

### HIGH: `check_drawdown` reads `portfolio_value_history.jsonl` line by line
**File**: `portfolio/risk_management.py:97-110`
**Impact**: Performance — O(n) scan of unbounded history file every cycle

The history file grows indefinitely (one entry per cycle = ~1440/day). After a year,
this is 500K+ entries scanned every time `check_drawdown` is called. The file is read
with raw `open()` + `json.loads()` per line, not using `file_utils.load_jsonl_tail()`.

**Fix**: Track peak_value in health_state.json (written once per cycle) instead of
re-scanning the entire history.

### HIGH: `_compute_portfolio_value` uses `avg_cost_usd` as fallback price
**File**: `portfolio/risk_management.py:46`
**Impact**: Overstated portfolio value when live prices are unavailable

When `agent_summary` doesn't contain a ticker's price, the function falls back to
`avg_cost_usd` from the holdings record. For a stock that has dropped 50% since
purchase, this would report the portfolio as 2x its actual value, masking a drawdown
that should trigger the circuit breaker.

**Fix**: Log a WARNING and use 0 (conservative) or mark position as "stale-priced".

### MEDIUM: `check_concentration_risk` uses cash-based allocation, not portfolio-based
**File**: `portfolio/risk_management.py:585`
**Impact**: Under-reports concentration when cash is low

`proposed_alloc = cash * alloc_pct` — when most of the portfolio is invested, cash
is low, so the "proposed" allocation is small. But the existing position might already
be 35% of the portfolio. Adding even a small amount could push it over 40%.

### MEDIUM: Trade guards use `severity: "warning"` — never "block"
**File**: `portfolio/trade_guards.py:93,120,153`
**Impact**: All guards are advisory, never blocking

The `check_overtrading_guards` function only ever returns `severity: "warning"`. The
consuming code in Layer 2 may log these but never actually prevent the trade. The
LOSS_ESCALATION dict goes up to 8x cooldown, but even this is just a warning.

### LOW: `record_trade` has no locking — concurrent calls can lose state
**File**: `portfolio/trade_guards.py:183`
**Impact**: Minimal — concurrent trades on same strategy could reset loss counter

---

## 4. METALS-CORE

### CRITICAL: metals_loop.py is 5366 lines — god module
**File**: `data/metals_loop.py`
**Impact**: Maintenance nightmare, hidden bugs, untestable

A single file with 5K+ lines handling: price fetching, signal generation, stop-loss
management, order placement, Telegram notifications, local LLM inference, Monte Carlo
simulation, fast-tick monitoring, and more. This makes it nearly impossible to unit test,
and changes in one area can silently break another.

### HIGH: metals_loop imports Playwright at module level
**File**: `data/metals_loop.py:54`
**Impact**: Import failure crashes the entire metals loop

`from playwright.sync_api import sync_playwright` is a top-level import. If Playwright
is not installed or has a version mismatch, the entire metals loop fails to start.
All other modules use lazy imports for optional dependencies.

### HIGH: Separate signal systems for metals vs main loop
**Impact**: Divergent signal logic — metals_loop has its own signal computation
that doesn't use `signal_engine.py`. This means fixes to signal_engine (accuracy
gating, regime gating, correlation groups) don't apply to metals signals.

### MEDIUM: exit_optimizer uses `np.random` without seed control
**File**: `portfolio/exit_optimizer.py`
**Impact**: Non-reproducible Monte Carlo results

The Monte Carlo path simulation uses numpy's global random state. Concurrent calls
from different threads could interleave random draws, producing different results
for the same inputs across runs.

### MEDIUM: `microstructure_state.py` uses module-level mutable state
**File**: `portfolio/microstructure_state.py`
**Impact**: State leaks between tests; not thread-safe

---

## 5. AVANZA-API

### CRITICAL: No account ID validation on order placement
**File**: `portfolio/avanza_session.py:371-379`
**Impact**: Order placed on wrong account (pension vs ISK)

The `_place_order` function defaults to `DEFAULT_ACCOUNT_ID = "1625505"` (ISK), but
any caller can pass a different `account_id`. There's no validation that the account
is the ISK account (1625505) vs the pension account (2674244). A bug in calling code
or a config change could route orders to the pension account.

**Fix**: Add an allowlist check: `assert account_id in ALLOWED_ACCOUNTS`.

### HIGH: Playwright context not refreshed on session expiry
**File**: `portfolio/avanza_session.py:115-134`
**Impact**: All Avanza operations fail after session expires

`_get_playwright_context()` validates the session on first call, but the context is
cached as `_pw_context`. If the session expires mid-operation (BankID sessions last ~24h),
subsequent calls get a stale context. The 401 handling in `api_get`/`api_post` calls
`close_playwright()`, but the caller then gets an exception — the retry/re-auth logic
is not automatic.

### HIGH: `api_delete` considers 404 as "ok"
**File**: `portfolio/avanza_session.py:286`
**Impact**: Silently succeeds when deleting a non-existent stop-loss

`"ok": 200 <= resp.status < 300 or resp.status == 404` — deleting a stop-loss that
doesn't exist returns ok=True. Calling code may think the stop-loss was removed when
it was never there (or was already triggered).

### MEDIUM: CSRF token extraction has no retry
**File**: `portfolio/avanza_session.py:207-212`
**Impact**: Transient cookie state → order fails

`_get_csrf()` reads cookies from the Playwright context. If cookies haven't been
set yet (race condition on first request), it raises `AvanzaSessionError`. There's
no retry or wait.

### MEDIUM: `avanza_orders.py` confirms orders FIFO, not by ID
**File**: `portfolio/avanza_orders.py:127-132`
**Impact**: CONFIRM could match wrong pending order

`check_pending_orders` iterates pending orders sequentially. A single CONFIRM reply
always confirms the first pending order, not the one the user intended. If two orders
are pending, the first one gets confirmed regardless.

---

## 6. SIGNALS-MODULES (portfolio/signals/*.py)

### HIGH: NaN propagation in momentum.py
**Impact**: NaN values in indicators cascade to NaN confidence, which gets
normalized to 0.0 by `_validate_signal_result()`, effectively silencing the signal.
This is safe (HOLD) but means the signal silently stops contributing during
data quality issues.

### HIGH: `claude_fundamental.py` makes synchronous API calls
**Impact**: Each call to Claude API blocks the thread for 5-30 seconds.
In the ThreadPoolExecutor with 8 workers, 20 tickers × claude_fundamental
could consume all worker threads and stall the cycle.

### MEDIUM: `econ_calendar.py` uses hardcoded FOMC dates
**Impact**: Dates are hardcoded for 2026-2027. After 2027, the signal produces
no risk-off recommendations, silently degrading.

### MEDIUM: `futures_flow.py` crypto-only but doesn't validate ticker
**Impact**: If called with a stock ticker, it queries Binance FAPI which returns
an error. The error is caught and HOLD is returned, but the API call wastes quota.

### MEDIUM: `forecast.py` depends on external model files at `Q:/models/`
**Impact**: Path is hardcoded. If the external drive is unmounted or renamed,
the signal silently fails to HOLD.

### LOW: Signal module interface inconsistency
Some modules accept `(df)`, others `(df, context=None)`, others `(df, macro=None)`.
The `signal_registry.py` handles this via metadata flags, but the inconsistency
makes it hard to reason about the signal pipeline.

---

## 7. DATA-EXTERNAL

### HIGH: `data_collector.py` yfinance fallback creates silent data quality issues
**Impact**: yfinance returns daily bars when intraday is requested. The system
may compute 15m RSI on daily data, producing fundamentally different signals
without any warning flag.

### HIGH: `fear_greed.py` returns cached data on API failure
**Impact**: The Fear & Greed index could be hours or days stale. The signal
engine uses it for voting without checking staleness beyond TTL.

### MEDIUM: `sentiment.py` handles API key exhaustion silently
**Impact**: When NewsAPI daily budget (90 calls) is exhausted, sentiment falls
back to Yahoo-only, which has lower quality. No flag is set in the signal output
to indicate degraded sentiment quality.

### MEDIUM: `fx_rates.py` is only 77 lines with minimal error handling
**Impact**: USD/SEK is critical for portfolio valuation. If the FX API fails,
the cached rate is used. If the cached rate is stale (e.g., from yesterday),
portfolio values could be off by 1-2%, potentially masking or triggering
drawdown circuit breakers.

### LOW: `onchain_data.py` BGeometrics API limit (15/day)
**Impact**: With 15 calls/day and 2 crypto tickers, each gets ~7 updates/day.
On-chain data changes slowly so this is adequate, but there's no tracking of
remaining quota.

---

## 8. INFRASTRUCTURE

### HIGH: `atomic_append_jsonl` is NOT truly atomic
**File**: `portfolio/file_utils.py:155-167`
**Impact**: Data loss on crash during append

The function opens the file in append mode, writes one line, and fsyncs. But
`open("a")` + `write()` is not atomic on any OS — if the process crashes between
`write()` and `fsync()`, the data may be in the OS buffer but not on disk. More
critically, if two processes append simultaneously, their writes can interleave
(partial lines), corrupting the JSONL file.

**Fix**: Use advisory file locking (fcntl/msvcrt) around the write, or write to
a temp file and append atomically via a separate step.

### HIGH: `health.py` reads and writes `health_state.json` without locking
**File**: `portfolio/health.py:16-36`
**Impact**: Concurrent health updates (main loop + signal health batch) can clobber

`update_health()` does `load_health()` → modify → `atomic_write_json()`. If
`update_signal_health_batch()` runs concurrently (e.g., from a different thread),
the second writer's `load_health()` might read the pre-update state, and its write
would overwrite the first writer's changes.

### MEDIUM: `shared_state._cached()` dogpile prevention has a leak path
**File**: `portfolio/shared_state.py:77-105`
**Impact**: If `func(*args)` raises AND the except handler also raises (e.g., due to
`_tool_cache[key]` KeyError), `_loading_keys` is never cleaned up. The key stays in
`_loading_keys` forever, causing all subsequent calls to return None.

### MEDIUM: Rate limiters use `time.sleep()` — blocks the calling thread
**File**: `portfolio/shared_state.py:167-174`
**Impact**: In ThreadPoolExecutor, rate limiter sleep blocks a worker thread.
With 8 workers processing 20 tickers, binance_limiter (600/min) and alpaca_limiter
(150/min) could cause significant thread starvation.

### MEDIUM: `telegram_notifications.py` — only 142 lines suggests thin error handling
**Impact**: Network errors during Telegram sends could propagate up and crash
the calling loop if not caught at call sites.

### LOW: `prune_jsonl` reads entire file into memory
**File**: `portfolio/file_utils.py:232-276`
**Impact**: For large JSONL files (signal_log.jsonl at 68MB), this allocates
significant memory. Should use streaming approach.

---

## CROSS-CUTTING CONCERNS

### Architecture: Two parallel signal systems
The main loop (`signal_engine.py`) and metals loop (`metals_loop.py`) have independent
signal computation. Bug fixes and accuracy improvements in one don't propagate to the other.
This is the single biggest architectural risk.

### Concurrency: ThreadPoolExecutor(8) + module-level mutable state
Multiple modules use module-level dicts/lists (sentiment cache, ADX cache, regime cache)
protected by individual locks. The lock granularity is fine-grained but the overall
concurrency model is hard to reason about. A deadlock between any two locks would freeze
the main loop.

### Testing: 3168 tests but coverage gaps in critical paths
The test suite is extensive but focuses on unit tests. Integration tests for the full
signal pipeline (fetch → compute → vote → consensus) are sparse. The most dangerous code
paths (order placement, stop-loss management) are the least tested because they require
real API connections.

### File I/O: Extensive use of JSONL for append-only logs
The JSONL append pattern (`atomic_append_jsonl`) is used for 10+ log files. None of these
appends are protected by file locks, creating interleaving risk when multiple processes
(main loop + metals loop + outcome tracker) write to the same file.

### Configuration: External symlink
`config.json` is a symlink to an external file. If the symlink target is modified while
the system is running, `_load_config()` will pick up mid-edit partial JSON, causing a
crash or misconfiguration.
