# System Overview — Portfolio Intelligence Trading Agent

> **Generated:** 2026-02-23 by autonomous improvement session
> **Codebase snapshot:** commit `881c084` (main)

## 1. Architecture Summary

A two-layer event-driven trading system:

- **Layer 1 (Python fast loop):** Runs every 60s, collects market data from Binance/Alpaca/yfinance, computes 27 signals across 7 timeframes for 30+ instruments, detects meaningful changes via a trigger system, and invokes Layer 2 when something matters.
- **Layer 2 (Claude Code agent):** Invoked as a subprocess when triggers fire. Reads signal snapshots, portfolio state, and its own memory. Makes independent BUY/SELL/HOLD decisions for two simulated portfolios (Patient & Bold, 500K SEK each). Sends analysis via Telegram.

### Data Flow

```
Market APIs (Binance, Alpaca, yfinance)
    │
    ▼
data_collector.collect_timeframes()  ──→  indicators.compute_indicators()
    │                                           │
    ▼                                           ▼
signal_engine.generate_signal()  ←──  27 signal modules (11 core + 16 enhanced)
    │
    ▼
reporting.write_agent_summary()  ──→  data/agent_summary.json
    │
    ▼
trigger.check_triggers()  ──→  7 trigger types
    │
    ├─ No trigger  ──→  continue looping
    │
    └─ Trigger fires:
        ├─ trigger.classify_tier()  ──→  T1 (quick) / T2 (signal) / T3 (full)
        ├─ reporting.write_tiered_summary()  ──→  compact JSON per tier
        ├─ journal.write_context()  ──→  data/layer2_context.md
        └─ agent_invocation.invoke_agent()  ──→  Claude Code subprocess
                │
                ▼
            Layer 2 reads context, analyzes, decides, trades, sends Telegram
```

## 2. Module Inventory (76+ Python files)

### Core Pipeline (12 modules)

| Module | Lines | Purpose |
|--------|-------|---------|
| `main.py` | ~435 | Orchestrator, CLI entry point, re-export hub |
| `shared_state.py` | ~180 | Thread-safe caching (`_cached()`), rate limiters, cycle counter |
| `market_timing.py` | ~120 | DST-aware US market hours, agent invocation window |
| `fx_rates.py` | ~80 | USD/SEK fetching via frankfurter.app, 2h staleness alerts |
| `indicators.py` | ~200 | RSI, MACD, EMA, BB, ATR computation + regime detection |
| `data_collector.py` | ~350 | Multi-source kline fetching (Binance spot/FAPI, Alpaca, yfinance) |
| `signal_engine.py` | ~500 | 27-signal voting engine, weighted consensus, sentiment hysteresis |
| `portfolio_mgr.py` | ~100 | Portfolio state load/save, value calculation |
| `reporting.py` | ~400 | agent_summary.json builder, tiered compact summaries |
| `trigger.py` | ~286 | 7 trigger types, tier classification, state persistence |
| `agent_invocation.py` | ~200 | Layer 2 subprocess launcher, tier-specific timeouts/turns |
| `logging_config.py` | ~50 | RotatingFileHandler (10MB x 3 backups) + stream handler |

### Signal Infrastructure (5 modules)

| Module | Lines | Purpose |
|--------|-------|---------|
| `signal_registry.py` | ~121 | Plugin system for lazy-loading signal modules |
| `signal_utils.py` | ~130 | Shared TA helpers: sma, ema, rsi, true_range, majority_vote |
| `signal_db.py` | ~335 | SQLite WAL-mode signal logging, dual-write with JSONL |
| `accuracy_stats.py` | ~533 | Per-signal accuracy, activation rates, accuracy snapshots |
| `signal_history.py` | ~100 | Signal history queries |

### Enhanced Composite Signals (16 modules in `portfolio/signals/`)

Each module runs 4-8 sub-indicators and produces one BUY/SELL/HOLD via majority vote.

| # | Signal | Module | Sub-indicators |
|---|--------|--------|----------------|
| 12 | Trend | `trend.py` | 7 |
| 13 | Momentum | `momentum.py` | 8 |
| 14 | Volume Flow | `volume_flow.py` | 6 |
| 15 | Volatility | `volatility.py` | 6 |
| 16 | Candlestick | `candlestick.py` | 5 |
| 17 | Structure | `structure.py` | 5 |
| 18 | Fibonacci | `fibonacci.py` | 5 |
| 19 | Smart Money | `smart_money.py` | 5 |
| 20 | Oscillators | `oscillators.py` | 8 |
| 21 | Heikin-Ashi | `heikin_ashi.py` | 7 |
| 22 | Mean Reversion | `mean_reversion.py` | 7 |
| 23 | Calendar | `calendar_seasonal.py` | 8 |
| 24 | Macro Regime | `macro_regime.py` | 6 |
| 25 | Momentum Factors | `momentum_factors.py` | 7 |
| 26 | News Event | `news_event.py` | 5 |
| 27 | Econ Calendar | `econ_calendar.py` | 4 |

### Support Modules (20+ modules)

| Module | Purpose |
|--------|---------|
| `file_utils.py` | Atomic JSON/JSONL I/O (tempfile + os.replace) |
| `circuit_breaker.py` | API failure protection (CLOSED -> OPEN -> HALF_OPEN) |
| `http_retry.py` | Exponential backoff retry (3 retries, 2x backoff) |
| `config_validator.py` | Startup config.json validation |
| `portfolio_validator.py` | Portfolio state integrity checks (8 reconciliation checks) |
| `telegram_notifications.py` | Telegram send + Markdown escaping + fallback |
| `digest.py` | 4-hour aggregated digest messages |
| `weekly_digest.py` | Weekly performance summary |
| `journal.py` | Layer 2 memory management (layer2_context.md) |
| `health.py` | Heartbeat, error tracking, agent silence detection |
| `macro_context.py` | DXY, treasury yields, Fed calendar + volume signal |
| `tickers.py` | Single source of truth for symbols and signal names |
| `avanza_client.py` / `avanza_tracker.py` / `avanza_orders.py` | Avanza integration |
| `bigbet.py` | Extreme setup alerts |
| `iskbets.py` | ISKBETS monitoring |
| `forecast_signal.py` | Prophet + Chronos GPU forecast |
| `outcome_tracker.py` | Backfill price outcomes at 1d/3d/5d/10d horizons |
| `kelly_sizing.py` | Kelly criterion position sizing |
| `risk_management.py` | Risk metrics and limits |
| `equity_curve.py` | Portfolio equity history |

### Dashboard

- `dashboard/app.py` — Flask API server (port 5055)
- `dashboard/static/index.html` — Web frontend (Chart.js for equity/accuracy charts)

## 3. Configuration

`config.json` (gitignored): Telegram bot, Alpaca API, Avanza, ISKBETS, Mistral API keys.

## 4. Test Infrastructure

- **Framework:** pytest
- **Baseline:** 1332 passing, 3 known pre-existing failures
- **47 test files** across `tests/`, `tests/unit/`, `tests/integration/`

## 5. Discrepancies vs Existing Documentation

`docs/architecture-plan.md` (last updated 2026-02-20) is stale:

1. **Signal count:** Says 25 signals, actual is **27** (news_event + econ_calendar added Feb 23)
2. **MIN_VOTERS:** Says crypto=3, stocks=2; actual is **3 for all asset classes**
3. **Cooldown:** Says 1min; actual is **10min** (`COOLDOWN_SECONDS = 600`)
4. **Applicable signals:** Says crypto=25, stocks=21; actual is **crypto=27, stocks=23**
5. **File layout:** Missing ~15 newer modules
6. **Stale test:** `test_market_hours_cooldown_is_1_min` asserts 60, actual is 600
