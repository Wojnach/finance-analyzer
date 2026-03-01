# Portfolio Intelligence System — System Overview

> **Updated:** 2026-03-01 (auto-improvement session #4)
> **Canonical architecture doc:** docs/architecture-plan.md
> **Layer 2 instructions:** CLAUDE.md

## Architecture Summary

Two-layer autonomous trading system managing 19 instruments (2 crypto, 2 metals, 15 US equities)
with 30 signals across 7 timeframes, dual simulated portfolios (Patient + Bold, 500K SEK each).

| Layer | Role | Runs | Decides |
|-------|------|------|---------|
| **Layer 1** (Python) | Data collection, signal computation, trigger detection | Every 60s (market), 5min (closed), 10min (weekend) | When to invoke Layer 2 |
| **Layer 2** (Claude Code) | Analysis, trade execution, Telegram notifications | On trigger (~20-50/day) | What to trade, what to say |

## Module Dependency Graph

```
main.py (orchestrator — loop, run, CLI dispatch)  [572 lines]
├── shared_state.py        (global caches, rate limiters, locks)  [114 lines]
├── market_timing.py       (DST-aware market hours, agent window)  [79 lines]
├── fx_rates.py            (USD/SEK caching via Frankfurter API)  [67 lines]
├── indicators.py          (RSI, MACD, EMA, BB, ATR, regime detection)  [166 lines]
├── data_collector.py      (kline fetching + multi-timeframe collector)  [259 lines]
│   ├── http_retry.py      (retry with exponential backoff)  [65 lines]
│   ├── circuit_breaker.py (per-source failure tracking)
│   └── api_utils.py       (config loading, Alpaca headers, API URLs)
├── signal_engine.py       (30-signal voting + weighted consensus)  [718 lines]
│   ├── signal_registry.py (enhanced signal plugin registry)  [130 lines]
│   ├── signal_utils.py    (shared helpers: SMA, EMA, RSI, majority_vote)  [130 lines]
│   ├── macro_context.py   (DXY, yields, FOMC, volume signal)
│   ├── accuracy_stats.py  (signal performance tracking, SQLite)  [559 lines]
│   └── signals/           (19 enhanced signal modules, ~8,400 lines total)
├── portfolio_mgr.py       (state load/save, portfolio_value, bold state)  [~60 lines]
│   └── file_utils.py      (atomic JSON I/O)
├── reporting.py           (agent_summary.json, tiered context)  [759 lines]
│   ├── equity_curve.py    (FIFO trade metrics, profit factor)  [596 lines]
│   ├── trade_guards.py    (cooldown, consecutive-loss escalation)  [266 lines]
│   ├── risk_management.py (concentration, correlation, ATR stops)  [704 lines]
│   ├── monte_carlo.py    (GBM price simulation, antithetic variates)  [401 lines]
│   ├── monte_carlo_risk.py (t-copula portfolio VaR/CVaR)  [350 lines]
│   ├── journal_index.py   (BM25 journal retrieval)  [399 lines]
│   ├── futures_data.py    (Binance FAPI OI/LS data)  [240 lines]
│   ├── avanza_tracker.py  (Nordic equity price tracking)
│   └── alpha_vantage.py   (stock fundamentals)  [329 lines]
├── trigger.py             (6 trigger conditions, tier classification)  [326 lines]
├── agent_invocation.py    (Claude subprocess management)  [198 lines]
│   ├── perception_gate.py (pre-invocation signal filter)  [98 lines]
│   ├── message_store.py   (save-only notifications)
│   └── telegram_notifications.py (Telegram sends)  [136 lines]
├── digest.py              (4-hour summary builder)  [201 lines]
├── reflection.py          (periodic strategy metrics)  [242 lines]
└── health.py              (heartbeat + error tracking)  [143 lines]
```

**Total:** ~97 Python modules, ~15,000+ lines of production code.

## Signal Architecture (30 total: 11 core + 19 enhanced)

**Core signals** (in signal_engine.py):
1. RSI(14), 2. MACD(12,26,9), 3. EMA(9,21), 4. BB(20,2), 5. Fear & Greed,
6. Sentiment, 7. CryptoTrader-LM (Ministral-8B), 8. Volume Confirmation
- Disabled: ML Classifier (#8, 28.2%), Funding Rate (#9, 27.0%), Custom LoRA (#11, 20.9%)

**Enhanced composite signals** (portfolio/signals/, each 4-8 sub-indicators via majority vote):
12-trend, 13-momentum, 14-volume_flow, 15-volatility, 16-candlestick, 17-structure,
18-fibonacci, 19-smart_money, 20-oscillators, 21-heikin_ashi, 22-mean_reversion,
23-calendar, 24-macro_regime, 25-momentum_factors, 26-news_event, 27-econ_calendar,
28-forecast (Kronos/Chronos), 29-claude_fundamental (LLM cascade),
30-futures_flow (Binance FAPI OI/LS, crypto only)

**Signal applicability:**
- Crypto (BTC, ETH): 27 signals (8 active core + 19 enhanced)
- Metals (XAU, XAG): 25 signals (7 core + 18 enhanced; no futures_flow)
- Stocks (15 tickers): 25 signals (7 core + 18 enhanced; no ministral, no futures_flow)

**Signal consensus flow:**
1. Each of 30 signals votes BUY/SELL/HOLD per ticker
2. Disabled signals (<50% accuracy) auto-inverted in weighted consensus
3. Confidence penalty cascade: regime→volume gate→trap detection→dynamic MIN_VOTERS
4. Raw consensus + weighted consensus both reported to Layer 2
5. Layer 2 uses both as inputs to independent judgment

## Global State Inventory

| Location | Variable | Purpose | Thread-Safe |
|----------|----------|---------|-------------|
| shared_state.py | `_tool_cache` | Per-cycle data cache (256 max) | Yes (`_cache_lock`) |
| shared_state.py | `_run_cycle_id` | Cycle counter (regime cache invalidation) | No (single-threaded loop) |
| shared_state.py | `_current_market_state` | "open"/"closed"/"weekend" | No (written by main loop only) |
| shared_state.py | `_regime_cache` | Per-cycle regime detection cache | No (invalidated by cycle_id) |
| shared_state.py | `_*_limiter` | Rate limiters (Alpaca, Binance, yfinance, AV) | Yes (internal Lock) |
| signal_engine.py | `_prev_sentiment` | Sentiment hysteresis per ticker | No (single-threaded) |
| fx_rates.py | `_fx_cache` | FX rate + timestamp | No (single-threaded) |
| agent_invocation.py | `_agent_proc` | Current subprocess handle | No (managed by main loop) |
| api_utils.py | `_config_cache` | Parsed config.json | Yes (`_config_lock`) |
| data_collector.py | `*_cb` | Circuit breakers | Yes (internal) |

## External Dependencies

| Dependency | Purpose | Rate Limit |
|------------|---------|------------|
| Binance Spot API | Crypto OHLCV (BTC, ETH) | 600/min |
| Binance FAPI | Metals OHLCV + futures data | 600/min |
| Alpaca IEX v2 | US stock OHLCV (27 tickers) | 150/min |
| yfinance | Stock fallback (extended hours) | 30/min |
| Alternative.me | Crypto Fear & Greed | 5min cache |
| Frankfurter API | USD/SEK exchange rate | 1h cache |
| Alpha Vantage | Stock fundamentals (P/E, revenue) | 5/min, 25/day |
| Telegram Bot API | Notifications | Unbounded |
| Ministral-8B (local GPU) | CryptoTrader-LM signal | 15min cache |
| Claude Code CLI | Layer 2 agent | On trigger |
| Kronos/Chronos (local) | Forecast signal #28 | 5min cache |
| Anthropic API | Claude Fundamental signal #29 | Per config cooldowns |

## Data Files

| File | Written By | Read By | Purpose |
|------|-----------|---------|---------|
| agent_summary.json | L1 | L2 (T3), Dashboard | Full 30-signal snapshot (~64KB) |
| agent_summary_compact.json | L1 | L2 (T1/T2) | Tiered compaction (~15KB) |
| agent_context_t1.json | L1 | L2 (T1) | Held positions + macro (~200 lines) |
| agent_context_t2.json | L1 | L2 (T2) | Triggered tickers + detail (~600 lines) |
| portfolio_state.json | L2 | L1, Dashboard | Patient holdings + transactions |
| portfolio_state_bold.json | L2 | L1, Dashboard | Bold holdings + transactions |
| trigger_state.json | L1 | L1 | Consensus, sustained counts, sentiment |
| sentiment_state.json | L1 | L1 | Sentiment hysteresis state |
| layer2_journal.jsonl | L2 | L1 (digest), Dashboard | Decisions, theses, reflections |
| layer2_context.md | L1 | L2 | Memory built from journal entries |
| signal_log.jsonl + SQLite | L1 | Accuracy tracker | All signal votes per cycle |
| health_state.json | L1 | Dashboard, monitors | Heartbeat, error log |
| invocations.jsonl | L1 | Dashboard, health | Layer 2 invocation history |
| telegram_messages.jsonl | L2 | Dashboard | All sent Telegram messages |
| fundamentals_cache.json | L1 | L1 (enrichment) | Alpha Vantage stock data |
| trade_guard_state.json | L1 | L1 | Per-ticker cooldowns, loss tracking |

## Test Suite

- **~79 test files**, ~2,750+ tests passing, 18 pre-existing failures
- Pre-existing failures: 15 integration (missing `ta_base_strategy`), 2 trigger tests, 1 subprocess test
- Collection error fixed: `test_avanza_session.py` rewritten for Playwright-based auth (31 tests)
- Coverage is excellent across all core modules (signal_engine, trigger, data_collector, reporting)
- Test configuration: pytest + pyproject.toml, ruff linting (line length 120)

## Monte Carlo Simulation (Session #5, 2026-03-01)

- **Core GBM engine** (`portfolio/monte_carlo.py`): Geometric Brownian Motion with antithetic variates (50-75% variance reduction). Converts directional probability P(up) into drift via inverse normal CDF. Computes price quantile bands (p5/p25/p50/p75/p95), stop-loss hit probability, and expected return distribution (mean/std/skew). Horizons: 1d, 3d.
- **Portfolio VaR** (`portfolio/monte_carlo_risk.py`): Student-t copula (df=4) for correlated multi-position simulation. Captures tail dependence (lambda ~0.18) that Gaussian copula misses. VaR/CVaR at 95%/99%, drawdown probability. Correlation from empirical returns or hardcoded priors from CORRELATED_PAIRS.
- **Reporting integration**: `monte_carlo` and `portfolio_var` sections in `agent_summary_compact.json`. Config: `monte_carlo.enabled` (default true), `n_paths` (10K), `horizons` ([1,3]). Graceful degradation on failure.
- **71 tests**, all passing in ~6s. Covers GBM statistics, antithetic variance reduction, quantile ordering, probability boundary conditions, t-copula correlation preservation, fat tails, VaR/CVaR ordering, correlated crash scenarios, diversification benefit, edge cases, and performance.

## Recent Improvements (Session #4, 2026-03-01)

- **6 bugs fixed (BUG-30 to BUG-35):** Dashboard heatmap missing 3 signals (forecast, claude_fundamental, futures_flow); digest.py reading wrong key from signal_log; http_retry returning response instead of None on retryable exhaust; message_store SEND_CATEGORIES including "invocation"; journal_index XAG price buckets capped at $35 (expanded to $120); alpha_vantage importing from portfolio_mgr instead of file_utils.
- **Bold portfolio loader centralized (ARCH-6):** `load_bold_state()`/`save_bold_state()` added to `portfolio_mgr.py`. Direct JSON reads across 4+ modules can now use the canonical loader.
- **Held tickers cache (BUG-36):** `_get_held_tickers()` in reporting.py cached per cycle via `_run_cycle_id`, saving 4 redundant disk reads per triggered cycle.
- **Heikin-ashi refactored (REF-11):** Removed unnecessary `_majority_vote` wrapper; direct `majority_vote()` call from signal_utils.
- **334 new signal module tests:** Dedicated test files for volume_flow (96), oscillators (70), smart_money (68), heikin_ashi (100). Plus 18 regression tests for bug fixes.
- **Ticker cleanup:** Removed 12 instruments (MSTR, BABA, GRRR, IONQ, TEM, UPST, VERI, QQQ, K33, H100, BTCAP-B, BULL-NDX3X). Added INVE-B. Tier 1: 27→19 (15 stocks + 2 crypto + 2 metals).

## Session #3 (2026-02-28)

- **3 broken signal modules repaired:** futures_flow (#30) passed dict to majority_vote (always HOLD); momentum RSI Divergence/StochRSI had variable shadowing (always HOLD); momentum_factors high/low proximity required 500 bars (unreachable, always HOLD). All now produce real votes.
- **Heikin-ashi confidence corrected:** Used `count_hold=True` unlike all other signals, making confidence 20-40% lower. Now matches standard behavior.
- **Trend signal NaN check fixed:** `is np.nan` identity comparison replaced with `pd.isna()`.
- **Trend Ichimoku dead code removed:** Duplicate tenkan/kijun computation cleaned up.
- **FX staleness guard fixed:** Unreachable stale-data warning now fires correctly.
- **Health uptime reset:** `reset_session_start()` prevents uptime from inheriting previous session.
- **Reporting KeyError guard:** `initial_value_sek` accessed with `.get()` default.
- **Signal engine constant hoisted:** `CORE_SIGNAL_NAMES` moved to module-level frozenset.
- **Trigger market hour constant:** Hardcoded `7` replaced with `MARKET_OPEN_HOUR` import.
- **Trigger state threading:** `update_tier_state()` accepts optional `state` param (saves disk read).
- **Reporting constants deduplicated:** `KEEP_EXTRA_FULL` extracted to module-level `_KEEP_EXTRA_FULL` frozenset.
- **11 new regression tests** in `tests/test_signal_bug_fixes.py`.

### Session #2 (earlier same day)

- **Reporting robustness:** 13 silent `except: pass` blocks replaced with `logger.warning()` + `_module_warnings` list surfaced to Layer 2
- **Stale cache reduced:** `_MAX_STALE_FACTOR` 5→3 (max 3x TTL fallback for failed data sources)
- **Trigger state cleanup:** Orphaned ticker entries pruned every save cycle (was lazy +10 buffer)
- **Health module failures:** `update_module_failures()` tracks which modules failed per cycle in `health_state.json`

## Deployment

- **Machine:** Windows 11 (herc2), RTX 3080 10GB
- **Python:** 3.12, two venvs (.venv for main, .venv-llm for GPU inference)
- **Scheduler:** Windows Task Scheduler (PF-DataLoop, PF-Dashboard, PF-OutcomeCheck, PF-MLRetrain)
- **Auto-restart:** pf-loop.bat with :restart loop + 30s delay
- **Dashboard:** Flask on port 5055

## Discrepancies vs Architecture Doc

1. **Architecture doc lists MRVL, PONY, RXRX** as tickers; CLAUDE.md does not. Possible ticker rotation.
2. **Architecture doc says "Cooldown expired"** as a trigger type; cooldown was REMOVED Feb 27 per MEMORY.md.
3. **Architecture doc says 11 core signals** but 3 are disabled; effective core count is 8.
4. **Architecture doc file tree** omits several modules added since (perception_gate, reflection, vector_memory, trade_guards, equity_curve, journal_index, futures_flow signal).
5. **Scheduled tasks**: PF-ForceSleep/PF-WakeUp/PF-AutoImprove not in arch doc.
