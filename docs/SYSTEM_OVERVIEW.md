# System Overview — Portfolio Intelligence Trading Agent

> **Generated:** 2026-02-24 by autonomous improvement session
> **Codebase snapshot:** branch `improve/auto-session-2026-02-24` (from `472424c`)

## 1. Architecture Summary

A two-layer event-driven trading system:

- **Layer 1 (Python fast loop):** Runs every 60s, collects market data from Binance/Alpaca/yfinance, computes 27 signals across 7 timeframes for 31+ instruments, detects meaningful changes via a trigger system, and invokes Layer 2 when something matters.
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
| 27 | Econ Calendar | `econ_calendar.py` | 5 |

### Support Modules (20+ modules)

| Module | Purpose |
|--------|---------|
| `file_utils.py` | Atomic JSON/JSONL I/O (tempfile + os.replace) |
| `circuit_breaker.py` | API failure protection (CLOSED -> OPEN -> HALF_OPEN) |
| `http_retry.py` | Exponential backoff retry (3 retries, 2x backoff, 10% jitter) |
| `config_validator.py` | Startup config.json validation |
| `portfolio_validator.py` | Portfolio state integrity checks (8 reconciliation checks) |
| `telegram_notifications.py` | Telegram send + Markdown escaping + fallback |
| `digest.py` | 4-hour aggregated digest messages |
| `weekly_digest.py` | Weekly performance summary |
| `journal.py` | Layer 2 memory management (layer2_context.md) |
| `health.py` | Heartbeat, error tracking, agent silence detection |
| `macro_context.py` | DXY, treasury yields, Fed calendar + volume signal |
| `tickers.py` | Single source of truth for symbols and signal names |
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
- **47 test files** across `tests/`, `tests/unit/`, `tests/integration/`

## 5. Discrepancies Found (vs existing docs and code)

1. **Dashboard heatmap missing 2 signals:** `app.py` signal-heatmap endpoint lists only 24 signals (missing `news_event` and `econ_calendar` added in recent PRs).
2. **Agent tier timeout unused:** `agent_invocation.py` computes `AGENT_TIMEOUT_DYNAMIC` per tier but the actual timeout check always uses the global `AGENT_TIMEOUT = 900`.
3. **Stale comments:** `signal_engine.py` line 20 says "25-signal" but system has 27. `trigger.py` docstring says "1 min cooldown" but code uses 600s (10 min).
4. **FX fallback outdated:** Hardcoded 10.50 SEK in `fx_rates.py` — actual rate is ~10.8-11.0 as of Feb 2026.
5. **Logger formatting:** `agent_invocation.py` lines 109, 162-164 still use f-string loggers instead of %-style (project convention established in prior session).
6. **BB NaN edge case:** `indicators.py` — if all prices in the 20-period window are identical, `bb_std` is 0 (not NaN), but `bb_upper == bb_lower == bb_mid == close`, so `price_vs_bb` will always be "inside". Not a crash bug but worth guarding.
