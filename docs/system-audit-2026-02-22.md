# System Audit -- 2026-02-22

## 1. Executive Summary

This audit covers the full codebase of the finance-analyzer autonomous trading system
as of 2026-02-22. The system comprises a two-layer architecture: a Python fast loop
(Layer 1) that collects data, computes 25 signals across 7 timeframes for 31 tickers,
and detects triggers; and a Claude Code agent (Layer 2) that is invoked on triggers to
make trade decisions for two simulated portfolios.

**Scope:** All modules under `portfolio/`, `dashboard/`, `tests/`, and supporting
scripts. The audit examined code quality, duplication, correctness, test coverage, and
operational robustness.

**Key findings:**
- 8 bugs identified and fixed (config validation, stale references to disabled signal,
  code duplication across 3 categories, non-atomic file writes, per-cycle import overhead)
- 10 remaining issues catalogued (mostly cosmetic or low-risk)
- Total codebase: ~18,988 lines in `portfolio/` (43 modules + 14 signal sub-modules),
  ~12,340 lines in `tests/` (28 test files)
- Test coverage is strong for core signal logic but has gaps in trigger, journal, and
  data collection modules

---

## 2. Architecture Overview

### Two-Layer Architecture

```
+----------------------------------------------------------------------+
|                        Windows 11 Pro Host                           |
|                                                                      |
|  Task Scheduler                                                      |
|  +-----------------------------+   +------------------------------+  |
|  | PF-DataLoop (at logon)      |   | PF-Dashboard (at logon)      |  |
|  | pf-loop.bat (auto-restart)  |   | dashboard/app.py :5000       |  |
|  +-----------------------------+   +------------------------------+  |
|              |                                                       |
|              v                                                       |
|  +-----------------------------+                                     |
|  | Layer 1: main.py --loop     |                                     |
|  | 60s (open) / 300s (closed)  |                                     |
|  | 600s (weekends)             |                                     |
|  |                             |                                     |
|  | 1. Fetch prices (31 tkrs)   |                                     |
|  | 2. Compute 25 signals x 7TF |                                     |
|  | 3. Check triggers           +--> NO TRIGGER: sleep & loop         |
|  | 4. Write agent_summary      |                                     |
|  | 5. Invoke Layer 2 ----------+------+                              |
|  +-----------------------------+      |                              |
|                                       v                              |
|  +----------------------------------------------+                   |
|  | Layer 2: claude -p (Claude Code agent)        |                   |
|  | pf-agent.bat -> Claude Opus 4                 |                   |
|  |                                               |                   |
|  | 1. Read layer2_context.md (memory)            |                   |
|  | 2. Read agent_summary_compact.json (~18K tok)  |                   |
|  | 3. Read both portfolio states                 |                   |
|  | 4. Analyze signals + macro + regime           |                   |
|  | 5. Decide: Patient HOLD/BUY/SELL              |                   |
|  | 6. Decide: Bold HOLD/BUY/SELL                 |                   |
|  | 7. Edit portfolio_state*.json (if trade)      |                   |
|  | 8. Append layer2_journal.jsonl                |                   |
|  | 9. Send Telegram message                      |                   |
|  +----------------------------------------------+                   |
|              |                                                       |
|  +-----------------------------+   +------------------------------+  |
|  | PF-OutcomeCheck (daily 18h) |   | PF-MLRetrain (weekly)        |  |
|  | --check-outcomes            |   | --retrain                    |  |
|  +-----------------------------+   +------------------------------+  |
+----------------------------------------------------------------------+
```

**Key invariant:** Layer 1 NEVER trades or sends Telegram messages. Layer 2 is the sole
authority on both.

### Module Dependency Graph

Data flows left to right through the pipeline:

```
APIs (Binance, Alpaca, yfinance, alternative.me, Ollama)
  |
  v
data_collector.py -----> indicators.py -----> signal_engine.py
  |                         |                    |
  | (OHLCV klines)          | (RSI, MACD, etc)  | (25 votes + consensus)
  |                         |                    |
  v                         v                    v
fx_rates.py             macro_context.py      trigger.py
  |                       |                    |
  |  (USD/SEK)            |  (DXY, yields)     |  (should we invoke L2?)
  |                       |                    |
  +-------+-------+-------+                    v
          |                              reporting.py
          v                                |
    portfolio_mgr.py                       | (agent_summary.json)
          |                                |
          v                                v
    main.py (orchestrator) <---------  agent_invocation.py
          |                                |
          v                                v
    digest.py                         journal.py
          |                                |
          v                                v
    telegram_notifications.py         Layer 2 (Claude Code)
```

**Cross-cutting imports (shared by many modules):**

| Module                  | Imported by                                                  |
|-------------------------|--------------------------------------------------------------|
| `tickers.py`            | signal_engine, accuracy_stats, signal_history, journal, main, macro_context, outcome_tracker |
| `shared_state.py`       | signal_engine, macro_context, data_collector, main           |
| `http_retry.py`         | main, data_collector, macro_context, reporting               |
| `api_utils.py`          | main, data_collector, macro_context                          |
| `indicators.py`         | signal_engine, data_collector                                |

---

## 3. Module Inventory

### Core Pipeline (`portfolio/`)

| Module                    | Lines | Purpose                                                        | Status      |
|---------------------------|------:|----------------------------------------------------------------|-------------|
| `main.py`                 |   455 | Thin orchestrator: loop(), run(), CLI entry                    | OK          |
| `shared_state.py`         |    82 | Mutable globals, _tool_cache, rate limiters, TTL constants     | OK          |
| `market_timing.py`        |    81 | DST-aware market hours, agent window, get_market_state()       | OK          |
| `fx_rates.py`             |    67 | USD/SEK exchange rate with caching                             | OK          |
| `indicators.py`           |   140 | compute_indicators(), detect_regime(), technical_signal()      | OK          |
| `data_collector.py`       |   271 | Binance/Alpaca/yfinance kline fetchers, multi-TF collector     | OK          |
| `signal_engine.py`        |   569 | 25-signal voting, weighted consensus, generate_signal()        | FIXED       |
| `portfolio_mgr.py`        |    51 | Portfolio state load/save/value, atomic writes                 | OK          |
| `reporting.py`            |   264 | agent_summary.json builder, compact summary                    | OK          |
| `telegram_notifications.py` |  124 | Telegram send/escape/alert                                   | OK          |
| `digest.py`               |   150 | 4-hour digest builder                                          | OK          |
| `agent_invocation.py`     |   108 | Layer 2 Claude Code subprocess invocation                      | OK          |
| `logging_config.py`       |    47 | Structured logging with RotatingFileHandler                    | OK          |
| `trigger.py`              |   202 | Trigger detection (consensus, price, F&G, cooldown)            | OK          |
| `journal.py`              |   407 | Layer 2 memory/context builder                                 | FIXED       |
| `tickers.py`              |   143 | Single source of truth: ticker lists, symbol maps, SIGNAL_NAMES | OK        |

### Signal & Accuracy Tracking

| Module                    | Lines | Purpose                                                        | Status      |
|---------------------------|------:|----------------------------------------------------------------|-------------|
| `accuracy_stats.py`       |   549 | Signal accuracy computation, cache                             | OK          |
| `outcome_tracker.py`      |   382 | Outcome backfilling for signal_log                             | OK          |
| `signal_db.py`            |   334 | SQLite storage for signal snapshots                            | OK          |
| `signal_history.py`       |   213 | Historical signal data, flip-flop detection                    | FIXED       |
| `migrate_signal_log.py`   |    54 | One-time JSONL to SQLite migration                             | OK          |

### External Signal Providers

| Module                    | Lines | Purpose                                                        | Status      |
|---------------------------|------:|----------------------------------------------------------------|-------------|
| `fear_greed.py`           |    80 | Fear & Greed index fetcher                                     | OK          |
| `sentiment.py`            |   260 | CryptoBERT / Trading-Hero-LLM sentiment                       | NEEDS_WORK  |
| `social_sentiment.py`     |   139 | Reddit scraper for social posts                                | OK          |
| `ministral_signal.py`     |    40 | Ministral-8B inference wrapper                                 | OK          |
| `ministral_trader.py`     |    96 | CryptoTrader-LM LoRA trading logic                             | OK          |
| `ml_signal.py`            |   163 | ML classifier inference                                        | NEEDS_WORK  |
| `ml_trainer.py`           |   210 | HistGradientBoosting training pipeline                         | OK          |
| `funding_rate.py`         |    59 | Binance perpetual funding rate                                 | OK          |
| `macro_context.py`        |   278 | DXY, treasury yields, yield curve, FOMC proximity              | NEEDS_WORK  |
| `fomc_dates.py`           |    57 | FOMC calendar (static, through 2027)                           | OK          |

### Infrastructure & Utilities

| Module                    | Lines | Purpose                                                        | Status      |
|---------------------------|------:|----------------------------------------------------------------|-------------|
| `http_retry.py`           |    56 | HTTP retry with exponential backoff                            | OK          |
| `health.py`               |   123 | Health monitoring: heartbeat, error tracking, agent silence    | FIXED       |
| `api_utils.py`            |    55 | Config loading, shared API utilities                           | OK          |
| `config_validator.py`     |    55 | Config schema validation                                       | FIXED       |
| `avanza_client.py`        |   148 | Avanza API client (Tier 2/3 Nordic stocks)                     | OK          |
| `avanza_tracker.py`       |   102 | Avanza position tracking                                       | NEEDS_WORK  |
| `avanza_watch.py`         |   106 | Avanza watchlist monitoring                                    | OK          |
| `backup.py`               |    93 | Data backup utilities                                          | OK          |
| `data_refresh.py`         |    86 | Manual data refresh tools                                      | OK          |
| `equity_curve.py`         |   355 | Equity curve computation                                       | OK          |
| `kelly_sizing.py`         |   355 | Kelly criterion position sizing                                | OK          |
| `log_rotation.py`         |   469 | Log file rotation                                              | OK          |
| `portfolio_validator.py`  |   294 | Portfolio state validation                                     | OK          |
| `regime_alerts.py`        |   263 | Regime change alerting                                         | OK          |
| `risk_management.py`      |   428 | Risk checks and limits                                         | OK          |
| `stats.py`                |   115 | Portfolio statistics                                           | OK          |
| `telegram_poller.py`      |   127 | Incoming Telegram command handler                              | NEEDS_WORK  |
| `weekly_digest.py`        |   332 | Weekly performance digest                                      | OK          |
| `iskbets.py`              | 1,022 | ISKBETS game integration                                       | NEEDS_WORK  |
| `bigbet.py`               |   475 | Big-bet analysis                                               | NEEDS_WORK  |
| `analyze.py`              |   862 | Detailed ticker analysis                                       | OK          |
| `collect.py`              |    53 | Collection helper                                              | OK          |

### Enhanced Signal Sub-Modules (`portfolio/signals/`)

| Module                    | Lines | Purpose                                     |
|---------------------------|------:|---------------------------------------------|
| `trend.py`                |   644 | Golden/Death Cross, MA Ribbon, Supertrend, SAR, Ichimoku, ADX |
| `heikin_ashi.py`          |   724 | HA Trend/Doji/Color, Hull MA, Alligator, Elder Impulse, TTM |
| `oscillators.py`          |   666 | Awesome Oscillator, Aroon, Vortex, Chande, KST, Schaff, TRIX, Coppock |
| `smart_money.py`          |   577 | BOS, CHoCH, Fair Value Gaps, Liquidity Sweeps, Supply/Demand |
| `fibonacci.py`            |   551 | Retracement, Golden Pocket, Extensions, Pivots, Camarilla |
| `mean_reversion.py`       |   524 | RSI(2), RSI(3), IBS, Consecutive Days, Gap Fade, BB %B |
| `calendar_seasonal.py`    |   476 | Day-of-Week, Turnaround Tuesday, Month-End, FOMC Drift |
| `momentum.py`             |   477 | RSI Divergence, Stochastic, StochRSI, CCI, Williams %R, ROC, PPO |
| `momentum_factors.py`     |   464 | Time-Series Momentum, ROC-20, 52-Week High/Low, Acceleration |
| `macro_regime.py`         |   419 | Adaptive SMA filter, DXY, Yield Curve, FOMC proximity |
| `volatility.py`           |   405 | BB Squeeze/Breakout, ATR Expansion, Keltner, Historical Vol, Donchian |
| `candlestick.py`          |   401 | Hammer/Shooting Star, Engulfing, Doji, Morning/Evening Star |
| `volume_flow.py`          |   324 | OBV, VWAP, A/D Line, CMF, MFI, Volume RSI |
| `structure.py`            |   283 | High/Low Breakout, Donchian 55, RSI Centerline, MACD Zero-Line |
| `__init__.py`             |     4 | Package init                                |

### Dashboard

| Module                    | Lines | Purpose                                     | Status      |
|---------------------------|------:|---------------------------------------------|-------------|
| `dashboard/app.py`        |  ~387 | Flask API + static frontend serving          | FIXED       |

### Tests (`tests/`)

| Test File                           | Lines | Covers                               |
|-------------------------------------|------:|--------------------------------------|
| `test_portfolio.py`                 |   797 | Portfolio math, state management      |
| `test_iskbets.py`                   |   825 | ISKBETS game integration              |
| `test_http_retry.py`               |   753 | HTTP retry logic                      |
| `test_signals_mean_reversion.py`    |   572 | Mean reversion signal module          |
| `test_dashboard.py`                 |   570 | Dashboard API endpoints               |
| `test_analyze.py`                   |   538 | Ticker analysis                       |
| `test_signals_momentum_factors.py`  |   535 | Momentum factors signal module        |
| `test_enhanced_signals.py`          |   533 | Enhanced signal modules (general)     |
| `test_trigger_edge_cases.py`        |   529 | Trigger edge cases                    |
| `test_signals_calendar.py`          |   506 | Calendar seasonal signal module       |
| `test_signal_improvements.py`       |   471 | Signal improvement features           |
| `test_signals_macro_regime.py`      |   470 | Macro regime signal module            |
| `test_signal_pipeline.py`          |   465 | End-to-end signal pipeline            |
| `test_portfolio_math.py`           |   447 | Portfolio calculation math             |
| `test_telegram_formatting.py`       |   437 | Telegram message formatting           |
| `test_journal_features.py`         |   412 | Journal feature tests                 |
| `test_macro_regime_integration.py`  |   407 | Macro regime integration              |
| `test_lora_pipeline.py`            |   342 | LoRA pipeline (disabled signal)       |
| `integration/test_strategy.py`     |   343 | Strategy integration tests            |
| `test_consensus.py`                |   320 | Consensus engine logic                |
| `test_health.py`                   |   318 | Health monitoring                     |
| `test_signal_db.py`                |   273 | Signal database (SQLite)              |
| `test_digest.py`                   |   262 | Digest builder                        |
| `test_avanza.py`                   |   257 | Avanza client/tracker                 |
| `test_metals.py`                   |   248 | Metals (XAU/XAG) handling             |
| `unit/test_indicators.py`          |   225 | Indicator computation                 |
| `test_journal.py`                  |   175 | Journal read/write                    |
| `unit/test_signal_logic.py`        |   161 | Signal logic unit tests               |
| `test_bigbet.py`                   |   138 | Big-bet analysis                      |

**Totals:** 18,988 lines across 57 Python modules in `portfolio/`; 12,340 lines across
28 test files in `tests/`.

---

## 4. Bugs Found and Fixed

### Bug 1: config_validator.py -- Wrong Alpaca key names

**File:** `Q:/finance-analyzer/portfolio/config_validator.py`
**Severity:** Medium (config validation would pass with wrong keys, or fail to detect missing correct keys)

**What was wrong:** The `OPTIONAL_KEYS` dict specified `{"key": str, "secret": str}` for the
Alpaca section, but the actual `config.json` uses `{"key_id": str, "secret_key": str}` (matching
Alpaca's SDK conventions). The validator would never flag missing Alpaca credentials because it
checked for the wrong key names.

**Fix:** Updated key names to `key_id` and `secret_key` to match actual config schema.
**Status:** FIXED

### Bug 2: dashboard/app.py -- Disabled custom_lora in core_signals list

**File:** `Q:/finance-analyzer/dashboard/app.py`
**Severity:** Low (cosmetic -- dashboard would show a perpetually empty signal)

**What was wrong:** The `core_signals` list used for dashboard rendering still included
`"custom_lora"`, which was disabled in the signal pipeline on Feb 20. The dashboard would
display an empty/null entry for this signal, potentially confusing the user.

**Fix:** Removed `custom_lora` from the core_signals list in dashboard/app.py.
**Status:** FIXED

### Bug 3: signal_history.py -- SIGNAL_NAMES included disabled custom_lora

**File:** `Q:/finance-analyzer/portfolio/signal_history.py`
**Severity:** Low (would track history for a signal that never votes)

**What was wrong:** `signal_history.py` maintained its own `SIGNAL_NAMES` list that still
included `custom_lora`. This caused the history tracker to allocate storage and track
flip-flop stats for a signal that is permanently HOLD/absent.

**Fix:** Replaced local `SIGNAL_NAMES` with an import from `portfolio.tickers`, which is
the canonical source of truth (where `custom_lora` was already removed).
**Status:** FIXED

### Bug 4: _RateLimiter duplicated across 3 files

**Files:** `shared_state.py`, `macro_context.py`, `outcome_tracker.py`
**Severity:** Medium (code duplication -- divergent behavior if one copy is updated)

**What was wrong:** The `_RateLimiter` class (a token-bucket rate limiter with threading
lock) was defined independently in three modules. The canonical version in `shared_state.py`
(lines 50-65) was the most complete. The copies in `macro_context.py` and `outcome_tracker.py`
were functionally identical but would diverge if the canonical version received improvements
(e.g., burst allowance, logging).

**Fix:** `macro_context.py` and `outcome_tracker.py` now import `_RateLimiter` from
`portfolio.shared_state` instead of defining their own copies.
**Status:** FIXED

### Bug 5: SIGNAL_NAMES duplicated across 3 files

**Files:** `tickers.py` (canonical), `accuracy_stats.py`, `signal_history.py`
**Severity:** Medium (same list in 3 places -- easy to get out of sync)

**What was wrong:** The authoritative list of 24 active signal names was duplicated:
- `tickers.py` lines 116-143 (canonical, 24 signals, custom_lora removed)
- `accuracy_stats.py` had its own copy
- `signal_history.py` had its own copy

When `custom_lora` was disabled on Feb 20, the `tickers.py` copy was updated but the
others could fall out of sync.

**Fix:** Both `accuracy_stats.py` and `signal_history.py` now import `SIGNAL_NAMES` from
`portfolio.tickers` instead of maintaining local copies.
**Status:** FIXED

### Bug 6: health.py -- Non-atomic write pattern

**File:** `Q:/finance-analyzer/portfolio/health.py`
**Severity:** Low (race condition -- health file could be corrupted on crash during write)

**What was wrong:** The `update_health()` function wrote directly to `health_state.json`
without an atomic write pattern. If the process crashed mid-write (e.g., during the
`json.dumps` or `write_text` call), the health file would be left in a corrupted state,
causing `load_health()` to fall back to defaults on the next read.

**Fix:** Changed to a write-to-temp-then-rename pattern: writes to `health_state.tmp` first,
then uses `Path.replace()` for an atomic rename. This is the same pattern used by
`portfolio_mgr.py` and `trigger.py`.
**Status:** FIXED

### Bug 7: signal_engine.py -- importlib.import_module called every cycle

**File:** `Q:/finance-analyzer/portfolio/signal_engine.py`
**Severity:** Low (performance -- unnecessary repeated dynamic imports)

**What was wrong:** The enhanced signal modules (14 modules in `portfolio/signals/`) were
loaded via `importlib.import_module()` on every call to `generate_signal()`, which runs
once per ticker per cycle (31 tickers x every 60 seconds = ~31 dynamic imports per minute).
While Python's import system caches modules internally, the `importlib.import_module()` call
still incurs overhead for module lookup, attribute resolution, and dict access on each
invocation.

**Fix:** Added a module-level cache dict that stores references to imported signal modules
after the first import. Subsequent calls use the cached reference directly, bypassing
`importlib` entirely.
**Status:** FIXED

### Bug 8: journal.py -- Duplicate TICKERS list

**File:** `Q:/finance-analyzer/portfolio/journal.py`
**Severity:** Low (duplication of ticker list; could drift from canonical source)

**What was wrong:** `journal.py` defined its own `TICKERS` list (lines 13-19) with all 29
Tier 1 tickers, duplicating the list already maintained in `tickers.py` as `ALL_TICKERS`.
If a ticker were added or removed, journal.py could be missed.

**Fix:** Replaced the local `TICKERS` list with an import from `portfolio.tickers`, using
the `ALL_TICKERS` set (converted to a sorted list where ordering matters).
**Status:** FIXED

---

## 5. Remaining Issues (Not Fixed This Audit)

### 5.1 Bare `requests.get()` without `fetch_with_retry`

**Files:** `ml_signal.py`, `sentiment.py`, `iskbets.py`, `telegram_poller.py`, `bigbet.py`
**Priority:** LOW

These modules make HTTP requests using bare `requests.get()` or `requests.post()` without
the `fetch_with_retry()` wrapper from `http_retry.py`. A transient network error (DNS
timeout, 502, connection reset) would cause an unhandled exception instead of a graceful
retry.

**Mitigating factor:** These are all called infrequently (sentiment/ML every 15 min with
TTL caching, iskbets/bigbet on demand, telegram_poller on a slow poll loop). The risk of
a transient failure during the brief call window is low.

### 5.2 `iskbets.py` has its own `_send_telegram()`

**File:** `iskbets.py`
**Priority:** COSMETIC

The module defines its own `_send_telegram()` helper instead of using the shared
`telegram_notifications.send_telegram()`. Functionally identical, but creates a maintenance
burden if the Telegram sending logic changes (e.g., rate limiting, error handling).

### 5.3 `avanza_tracker.py` uses `print()` instead of `logging`

**File:** `avanza_tracker.py`
**Priority:** COSMETIC

Debug output uses `print()` statements instead of the structured logging configured via
`logging_config.py`. This means Avanza tracker output does not appear in rotated log files
and cannot be filtered by level.

### 5.4 `sentiment.py` uses `urllib` instead of `requests`

**File:** `sentiment.py`
**Priority:** COSMETIC

The module uses `urllib.request` for HTTP calls instead of the `requests` library used
everywhere else in the codebase. This is a style inconsistency; `urllib` has different
error handling patterns and does not integrate with `fetch_with_retry()`.

### 5.5 `macro_context.py` uses hardcoded `time.sleep()`

**File:** `macro_context.py`
**Priority:** LOW RISK

The module has hardcoded `time.sleep()` calls for rate limiting instead of using the shared
`_RateLimiter` instances from `shared_state.py`. The rate limiter import was already added
(for deduplication of the class), but the existing sleep calls were not replaced.

### 5.6 `FEE_CRYPTO` / `FEE_STOCK` constants unused in Python

**File:** `main.py` lines 88-89
**Priority:** COSMETIC

```python
FEE_CRYPTO = 0.0005
FEE_STOCK = 0.001
```

These constants are defined but never used by any Python code. The fee logic is entirely
in CLAUDE.md for Layer 2 (the agent computes fees when editing portfolio state). The
constants exist for documentation purposes but could mislead developers into thinking
Python-side fee calculation exists.

### 5.7 `_tool_cache` grows unboundedly

**File:** `shared_state.py` line 14
**Priority:** LOW RISK

The `_tool_cache` dict accumulates entries over time. While the TTL mechanism prevents
stale data from being *served*, expired entries are never *evicted* from the dict. Over
very long uptimes (weeks), the cache could accumulate dead entries for tools that were
called once and never again.

**Mitigating factor:** The cache holds at most ~6 unique keys (Fear & Greed, Sentiment,
Ministral, ML, Funding, Volume) per ticker. With 31 tickers, the maximum cache size is
~186 entries -- negligible memory impact even without eviction.

### 5.8 `backfill_outcomes()` loads entire JSONL into memory

**File:** `outcome_tracker.py` line 252
**Priority:** SCALABILITY

The function reads all of `signal_log.jsonl` into memory to find entries needing outcome
backfill. At current scale (~1,000 entries, ~500KB), this is fine. At 10,000+ entries
(~6 months of continuous operation), it could cause noticeable memory spikes during the
daily 18:00 outcome check.

**Future fix:** Consider reading only the last N lines, or migrating fully to the SQLite
`signal_db.py` backend.

### 5.9 `_is_agent_window()` blocks weekends entirely including crypto

**File:** `market_timing.py` lines 52-68
**Priority:** KNOWN DESIGN DECISION

The agent window function returns `False` for all of Saturday and Sunday (weekday >= 5),
which means Layer 2 is never invoked on weekends. Crypto trades 24/7, so a weekend flash
crash or rally would not trigger a Layer 2 analysis until Monday 06:00 UTC.

This is an intentional design choice (documented in MEMORY.md) to reduce API costs and
agent invocations during low-liquidity periods. Layer 1 still runs on weekends at 600s
intervals and would log signal changes.

### 5.10 `fomc_dates.py` has hardcoded dates through 2027

**File:** `fomc_dates.py`
**Priority:** LOW (needs update in late 2027)

The FOMC meeting dates are statically defined through December 2027. The Fed typically
publishes the next year's calendar in June of the current year. This file will need an
update when the 2028 FOMC calendar is published (expected June 2027).

---

## 6. Test Coverage Assessment

### Well-Tested Areas

| Area                        | Test File(s)                              | Lines | Notes                                 |
|-----------------------------|-------------------------------------------|------:|---------------------------------------|
| Signal consensus engine     | `test_consensus.py`                       |   320 | MIN_VOTERS, weighted, hysteresis      |
| Signal pipeline (end-to-end)| `test_signal_pipeline.py`, `test_signal_improvements.py` | 936 | Full generate_signal flow    |
| Health monitoring           | `test_health.py`                          |   318 | Heartbeat, staleness, agent silence   |
| HTTP retry logic            | `test_http_retry.py`                      |   753 | Thorough: backoff, jitter, status codes |
| Dashboard API               | `test_dashboard.py`                       |   570 | All endpoints, auth, error cases      |
| Signal database (SQLite)    | `test_signal_db.py`                       |   273 | CRUD, queries, migrations             |
| Portfolio math              | `test_portfolio_math.py`, `test_portfolio.py` | 1,244 | BUY/SELL execution, fee calc, state |
| Telegram formatting         | `test_telegram_formatting.py`             |   437 | Message structure, escaping, length   |
| Enhanced signal modules     | `test_enhanced_signals.py` + 4 dedicated  | 2,616 | Mean reversion, calendar, macro, momentum factors |
| Trigger edge cases          | `test_trigger_edge_cases.py`              |   529 | Cooldown, sustained flips, post-trade |
| Journal features            | `test_journal.py`, `test_journal_features.py` | 587 | Context builder, recent entries    |

### Coverage Gaps

| Area                    | Current State                           | Risk   | Recommendation            |
|-------------------------|-----------------------------------------|--------|---------------------------|
| `trigger.py` core logic | Only edge-case tests; no unit tests for `check_triggers()` main flow | HIGH | Add unit tests for each trigger type |
| `data_collector.py`     | No tests (API calls are hard to mock)   | MEDIUM | Mock Binance/Alpaca, test error paths |
| `fx_rates.py`           | No tests                                | LOW    | Simple module; add basic mock test   |
| `journal.py` write_context | `test_journal.py` exists (175 lines) but light coverage | MEDIUM | Add tests for edge cases in context building |
| `macro_context.py`      | Tested via integration (`test_macro_regime_integration.py`) but no unit tests | MEDIUM | Mock yfinance calls, test DXY/yields parsing |
| `agent_invocation.py`   | No tests (subprocess invocation)        | LOW    | Hard to unit test; integration test via mock |
| Enhanced signal modules | 5 of 14 modules have dedicated tests    | MEDIUM | Add tests for remaining 9 modules    |
| `outcome_tracker.py`    | No dedicated tests                      | MEDIUM | Test backfill logic with fixture data |
| `digest.py`             | `test_digest.py` exists (262 lines)     | LOW    | Recently rewritten; appears adequate  |

### Test Infrastructure

- **Framework:** pytest
- **Total test count:** ~750 passing (as of Feb 21 audit)
- **CI/CD:** None (manual runs only)
- **Fixtures:** `conftest.py` (9 lines) -- minimal shared fixtures
- **Mocking:** Tests use `unittest.mock.patch` extensively for API isolation

---

## 7. Recommendations

### P0 -- High Priority (should fix soon)

1. **Wire `fetch_with_retry` into remaining bare-requests modules.**
   Files: `ml_signal.py`, `sentiment.py`, `iskbets.py`, `telegram_poller.py`, `bigbet.py`.
   The HTTP retry wrapper exists and is proven (753 lines of tests). Integrating it into
   these 5 modules is straightforward (change `requests.get()` to `fetch_with_retry()`)
   and eliminates the most common failure mode (transient network errors).

### P1 -- Medium Priority (next audit cycle)

2. **Add unit tests for `trigger.py` core logic.**
   The trigger system gates all Layer 2 invocations. A bug here means either missed
   triggers (silent agent) or excessive triggers (wasted API budget). The
   `test_trigger_edge_cases.py` file covers edge cases but not the happy path of each
   trigger type (consensus change, price move, F&G threshold, cooldown expiry).

3. **Add unit tests for `outcome_tracker.py` backfill logic.**
   The accuracy tracking system depends on correct outcome backfilling. A subtle bug
   (wrong price, wrong horizon) would corrupt accuracy stats, which feed into weighted
   consensus and Layer 2 reasoning.

4. **Consolidate `iskbets.py` `_send_telegram()` to shared module.**
   Replace with import from `telegram_notifications.py`. While cosmetic, this is a
   one-line fix that prevents future divergence.

### P2 -- Low Priority (backlog)

5. **Add `SignalDB` context manager support.**
   The `signal_db.py` SQLite module currently requires explicit connection management.
   Adding `__enter__`/`__exit__` methods would make it safer to use in `with` blocks,
   ensuring connections are always closed even on exceptions.

6. **Consider splitting `iskbets.py` (1,022 lines).**
   This is the second-largest module after `analyze.py` (862 lines). It combines game
   logic, API interaction, state management, and Telegram messaging. Splitting into
   `iskbets_game.py` (logic), `iskbets_api.py` (HTTP), and reusing shared Telegram
   would improve maintainability.

7. **Add eviction to `_tool_cache`.**
   While low risk at current scale (max ~186 entries), adding a simple max-size check
   with LRU eviction would be good hygiene. A 3-line addition to the `_cached()` function.

8. **Migrate `sentiment.py` from `urllib` to `requests`.**
   Standardize on the `requests` library for consistency. This would also allow
   `fetch_with_retry()` integration (P0 item) without major refactoring.

9. **Replace `print()` with `logging` in `avanza_tracker.py`.**
   Simple find-and-replace to use the structured logging infrastructure.

10. **Add weekend crypto trigger support (optional).**
    If weekend trading becomes desirable, modify `_is_agent_window()` to allow
    crypto-only triggers on weekends with a longer cooldown (e.g., 4 hours). This
    would require also gating stock tickers out of the trigger evaluation.

### P3 -- Future Considerations

11. **Signal log rotation or migration.**
    `signal_log.jsonl` grows unboundedly. The SQLite `signal_db.py` backend exists
    but is not yet the primary store. Full migration would solve both the memory issue
    in `backfill_outcomes()` and enable faster queries.

12. **CI/CD pipeline.**
    The 750-test suite runs manually. Adding a pre-commit hook or GitHub Actions
    workflow would catch regressions before they reach production.

13. **Update `fomc_dates.py` when 2028 calendar is published.**
    Expected: June 2027. Set a calendar reminder.

---

*Audit performed: 2026-02-22. Next recommended audit: 2026-03-22 or after next major
feature addition.*
