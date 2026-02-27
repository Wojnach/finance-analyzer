# Portfolio Intelligence System — System Overview

> **Updated:** 2026-02-27 (auto-improvement session)
> **Canonical architecture doc:** docs/architecture-plan.md
> **Layer 2 instructions:** CLAUDE.md

## Architecture Summary

Two-layer autonomous trading system managing 31 instruments (2 crypto, 2 metals, 27 US equities)
with 30 signals across 7 timeframes, dual simulated portfolios (Patient + Bold, 500K SEK each).

| Layer | Role | Runs | Decides |
|-------|------|------|---------|
| **Layer 1** (Python) | Data collection, signal computation, trigger detection | Every 60s (market), 5min (closed), 10min (weekend) | When to invoke Layer 2 |
| **Layer 2** (Claude Code) | Analysis, trade execution, Telegram notifications | On trigger (~20-50/day) | What to trade, what to say |

## Module Dependency Graph

```
main.py (orchestrator — loop, run, CLI dispatch)
├── shared_state.py        (global caches, rate limiters, locks)
├── market_timing.py       (DST-aware market hours, agent window)
├── fx_rates.py            (USD/SEK caching via Frankfurter API)
├── indicators.py          (RSI, MACD, EMA, BB, ATR, regime detection)
├── data_collector.py      (kline fetching + multi-timeframe collector)
│   ├── http_retry.py      (retry with exponential backoff)
│   ├── circuit_breaker.py (per-source failure tracking)
│   └── api_utils.py       (config loading, Alpaca headers, API URLs)
├── signal_engine.py       (30-signal voting + weighted consensus)
│   ├── signal_registry.py (enhanced signal plugin registry)
│   ├── signal_utils.py    (shared helpers: SMA, EMA, RSI, majority_vote)
│   ├── macro_context.py   (DXY, yields, FOMC, volume signal)
│   ├── accuracy_stats.py  (signal performance tracking, SQLite)
│   └── signals/           (19 enhanced signal modules)
├── portfolio_mgr.py       (state load/save, portfolio_value)
│   └── file_utils.py      (atomic JSON I/O)
├── reporting.py           (agent_summary.json, tiered context)
│   ├── equity_curve.py    (FIFO trade metrics, profit factor)
│   ├── trade_guards.py    (cooldown, consecutive-loss escalation)
│   ├── risk_management.py (concentration, correlation, ATR stops)
│   ├── journal_index.py   (BM25 journal retrieval)
│   ├── futures_data.py    (Binance FAPI OI/LS data)
│   ├── avanza_tracker.py  (Nordic equity price tracking)
│   └── alpha_vantage.py   (stock fundamentals)
├── trigger.py             (6 trigger conditions, tier classification)
├── agent_invocation.py    (Claude subprocess management)
│   ├── message_store.py   (save-only notifications)
│   └── telegram_notifications.py (Telegram sends)
├── digest.py              (4-hour summary builder)
└── health.py              (heartbeat + error tracking)
```

## Signal Modules (30 total: 11 core + 19 enhanced)

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
- Stocks (27 tickers): 25 signals (7 core + 18 enhanced; no ministral, no futures_flow)

## Global State Inventory

| Location | Variable | Purpose | Thread-Safe |
|----------|----------|---------|-------------|
| shared_state.py | `_tool_cache` | Per-cycle data cache | Yes (`_cache_lock`) |
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
| sentiment_state.json | L1 | L1 | Sentiment hysteresis state (extracted from trigger_state) |
| layer2_journal.jsonl | L2 | L1 (digest), Dashboard | Decisions, theses, reflections |
| layer2_context.md | L1 | L2 | Memory built from journal entries |
| signal_log.jsonl + SQLite | L1 | Accuracy tracker | All signal votes per cycle |
| health_state.json | L1 | Dashboard, monitors | Heartbeat, error log |
| invocations.jsonl | L1 | Dashboard, health | Layer 2 invocation history |
| telegram_messages.jsonl | L2 | Dashboard | All sent Telegram messages |
| fundamentals_cache.json | L1 | L1 (enrichment) | Alpha Vantage stock data |
| trade_guard_state.json | L1 | L1 | Per-ticker cooldowns, loss tracking |

## Deployment

- **Machine:** Windows 11 (herc2), RTX 3080 10GB
- **Python:** 3.12, two venvs (.venv for main, .venv-llm for GPU inference)
- **Scheduler:** Windows Task Scheduler (PF-DataLoop, PF-Dashboard, PF-OutcomeCheck, PF-MLRetrain)
- **Auto-restart:** pf-loop.bat with :restart loop + 30s delay
- **Dashboard:** Flask on port 5055

## Discrepancies vs Architecture Doc

1. **Signal count drift**: Architecture doc says 29 signals, code has 30 (futures_flow #30 added Feb 26).
   Applicable counts: crypto=27 (not 26), stocks/metals=25.
2. **CLAUDE.md says "27 signals"** in several places but refers to 30 signal modules — the number
   refers to applicable per-asset-class, not total.
3. **Scheduled tasks**: PF-ForceSleep/PF-WakeUp/PF-AutoImprove not in arch doc.

## Known Issues

### Fixed (previous sessions)
- atomic_append_jsonl fsync, sentiment state race, agent log fd leak, stale ticker accumulation
- Portfolio snapshot now includes holdings value, zero-division guard in indicators
- accuracy_stats DRY refactored, test coverage added for indicators/signal_engine/portfolio_mgr/trigger

### Remaining
- Sentiment hysteresis doesn't persist neutral direction (false threshold elevation possible)
- No test coverage for `data_collector.py` (data fetching core)
- No test coverage for `agent_invocation.py` (Layer 2 subprocess)
- Architecture doc signal count stale (29 → should be 30)
- Config example missing keys for 5+ features (perception_gate, reflection, etc.)
- `_cached()` in shared_state.py allows duplicate concurrent calls for same key (inefficient, not unsafe)
