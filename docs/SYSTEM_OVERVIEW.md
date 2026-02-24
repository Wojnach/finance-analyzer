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

## 5. Changes Made (auto-improve session 2026-02-24)

All discrepancies found during exploration have been resolved:

1. **Dashboard heatmap:** Added `news_event` and `econ_calendar` to signal heatmap endpoint (was 24 signals, now 26 enhanced = complete).
2. **Agent tier timeout:** Replaced global `AGENT_TIMEOUT = 900` with per-invocation `_agent_timeout` set from tier config (T1=120s, T2=300s, T3=900s). Removed unused `AGENT_TIMEOUT_DYNAMIC`.
3. **Stale comments:** Fixed "25-signal" → "27-signal" in `signal_engine.py` and "1 min cooldown" → "10 min" in `trigger.py`.
4. **FX fallback:** Updated hardcoded fallback from 10.50 → 10.85 SEK in `fx_rates.py`.
5. **Logger formatting:** Converted remaining f-string loggers in `agent_invocation.py` to %-style.
6. **Telegram truncation:** Added 4096-char message length guard in `send_telegram()` to prevent silent HTTP 400 failures.

### Session 2 changes (2026-02-24)

7. **AGENT_TIMEOUT import:** Removed stale `AGENT_TIMEOUT` re-export from `main.py` that crashed on import (left over from Session 1's per-tier timeout migration).
8. **Cache staleness guard:** `_cached()` now returns `None` when data exceeds 5x TTL during errors, preventing hours-old prices from being used.
9. **Regime/EMA alignment:** `detect_regime()` EMA gap threshold lowered from 1.0% to 0.5% to match signal engine's EMA deadband.
10. **Heikin-Ashi voting:** Replaced local `_majority_vote()` with canonical `signal_utils.majority_vote()`.
11. **Stale data timestamps:** Preserved data in agent_summary.json now includes `stale_since` ISO timestamp.
12. **Trigger state pruning:** `triggered_consensus` entries older than 7 days are auto-pruned.
13. **New tests:** `test_shared_state.py`, `test_market_timing.py`, `test_trigger_core.py` — 40+ new tests covering cache, DST, triggers.

### Known non-blocking items (deferred)
- **BB NaN edge case:** `indicators.py` — if all prices in the 20-period window are identical, `bb_std` is 0, so `price_vs_bb` will always be "inside". Not a crash bug, effectively a HOLD signal, which is correct behavior for a flat market.
