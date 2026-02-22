# System Overview — Portfolio Intelligence Trading Agent

> Updated: 2026-02-22 | Auto-improvement session (Phase 2)
> See also: `docs/system-design.md` (detailed reference), `docs/architecture-plan.md` (canonical architecture)

## Architecture

Two-layer system: Python fast loop (Layer 1) collects data every 60s, computes 25 trading signals across 7 timeframes for 31 instruments. When something meaningful changes (trigger fires), it invokes Claude Code (Layer 2) via subprocess. Layer 2 analyzes all context, decides whether to trade for two independent portfolios (Patient + Bold), edits portfolio JSON files, and sends Telegram notifications.

**Key invariant:** Layer 1 NEVER trades or sends Telegram messages. Layer 2 is the sole authority.

## Module Map (54 Python files)

### Core Orchestration
| Module | Lines | Purpose |
|--------|-------|---------|
| `main.py` | 453 | Thin orchestrator: `loop()` → `run()` → fetch/signals/trigger/invoke |
| `shared_state.py` | ~80 | Global cache dict, TTL constants, rate limiters (token-bucket), threading locks |
| `trigger.py` | 202 | 6 trigger conditions: consensus, flip, price, F&G, sentiment, cooldown |
| `agent_invocation.py` | ~100 | Spawns Layer 2 subprocess, 600s timeout, env var stripping |
| `journal.py` | 407 | Builds `layer2_context.md` from recent journal entries for Layer 2 memory |
| `health.py` | 133 | Heartbeat, crash detection, agent silence monitoring |
| `market_timing.py` | ~120 | DST-aware NYSE hours, market state (open/closed/weekend) |
| `logging_config.py` | ~100 | RotatingFileHandler (10MB, 3 backups) |

### Data Collection
| Module | Lines | Purpose |
|--------|-------|---------|
| `data_collector.py` | 271 | Binance spot/FAPI + Alpaca + yfinance kline fetchers, circuit breakers |
| `http_retry.py` | ~70 | Exponential backoff, max 5 retries, no retry on 4xx |
| `circuit_breaker.py` | 91 | CLOSED/OPEN/HALF_OPEN per data source (Binance, Alpaca) |
| `fx_rates.py` | ~60 | USD/SEK via ECB, cached 1hr |
| `api_utils.py` | ~100 | Config loading, Alpaca headers, canonical API URLs |
| `config_validator.py` | ~76 | Startup config.json validation |
| `tickers.py` | 113 | Single source of truth: SYMBOLS dict → all derived mappings |

### Signal Pipeline
| Module | Lines | Purpose |
|--------|-------|---------|
| `signal_engine.py` | 564 | 25-signal voting, weighted consensus, sentiment hysteresis |
| `signal_registry.py` | 113 | Plugin-style registration for enhanced signal modules |
| `signal_utils.py` | 94 | Shared helpers: sma, ema, rsi, true_range, safe_float, rma, wma, roc |
| `indicators.py` | 140 | RSI, MACD, EMA, BB, ATR, regime detection |
| `signals/*.py` (14) | ~6700 | Enhanced composite signals (trend, momentum, etc.) |
| `fear_greed.py` | ~60 | alternative.me API for crypto F&G |
| `sentiment.py` | 268 | CryptoBERT + Trading-Hero-LLM inference |
| `ministral_signal.py` | ~100 | Ministral-8B + LoRA wrapper |
| `ml_signal.py` | 166 | HistGradientBoosting classifier (crypto only) |
| `funding_rate.py` | ~100 | Binance perpetual funding rate |
| `macro_context.py` | 277 | DXY, treasury yields, yield curve, FOMC |

### Portfolio & Trading
| Module | Lines | Purpose |
|--------|-------|---------|
| `portfolio_mgr.py` | ~100 | Load/save portfolio state, atomic writes |
| `portfolio_validator.py` | 294 | Holdings/cash integrity checks |
| `risk_management.py` | 428 | Position sizing, concentration, leverage |
| `kelly_sizing.py` | 355 | Kelly criterion calculations |
| `equity_curve.py` | 355 | P&L tracking, drawdown |

### Notification & Reporting
| Module | Lines | Purpose |
|--------|-------|---------|
| `telegram_notifications.py` | ~150 | Send with Markdown fallback, escaping |
| `reporting.py` | 264 | agent_summary.json + compact version |
| `digest.py` | 150 | 4-hour periodic digest |
| `weekly_digest.py` | 327 | 7-day performance digest |
| `accuracy_stats.py` | 549 | Per-signal hit rates at 1d/3d/5d/10d |
| `outcome_tracker.py` | 372 | Backfill outcomes for forward tracking |
| `signal_db.py` | 334 | SQLite WAL-mode storage (dual-write with JSONL) |

### Specialized Subsystems
| Module | Lines | Purpose |
|--------|-------|---------|
| `bigbet.py` | 462 | High-confidence alert system |
| `iskbets.py` | 1026 | Intraday quick-entry mode (ATR exits, stage ladder) |
| `analyze.py` | 846 | Deep per-ticker analysis |
| `avanza_client.py` | 148 | Nordic stocks API (Tier 2/3) |
| `avanza_tracker.py` | ~100 | Avanza position tracking |
| `regime_alerts.py` | 258 | Regime change alerts |
| `telegram_poller.py` | ~100 | Incoming Telegram command handler |

### Removed Dead Code
| Module | Status |
|--------|--------|
| `collect.py` | Deleted — old data collector, confirmed zero imports |
| `avanza_watch.py` | Deleted — confirmed zero imports |

### Initially Flagged but Active (lazy imports)
| Module | Used By |
|--------|---------|
| `stats.py` | `digest.py` — `from portfolio.stats import load_jsonl` (lazy import) |
| `social_sentiment.py` | `signal_engine.py` — `from portfolio.social_sentiment import get_reddit_posts` (lazy import) |

## Data Flow

```
Binance/Alpaca/yfinance APIs
    → data_collector.py (klines, 7 timeframes)
    → indicators.py (RSI, MACD, EMA, BB, ATR, regime)
    → signal_engine.py (25 votes → weighted consensus)
    → trigger.py (fire if meaningful change)
    → agent_invocation.py (spawn Claude Code subprocess)
    → Layer 2 reads agent_summary.json, decides, edits portfolio, sends Telegram
```

## Recent Improvements (Feb 22)

### Phase 3 (latest session)
- **Bug fix**: Consolidated 3 independent `_yfinance_limiter` instances → single shared instance in `shared_state.py`. macro_context.py and outcome_tracker.py now import instead of creating their own (was allowing 90 req/min instead of 30).
- **Bug fix**: `_crash_alert()` in main.py now opens config.json with `encoding="utf-8"` (prevented crash-handler crash on Windows).
- **Performance**: `best_worst_signals()` accepts pre-computed accuracy data; `reporting.py` passes it to avoid redundant full-log scan.
- **Deduplication**: Extracted `load_json()`, `load_jsonl()`, `atomic_append_jsonl()` to `file_utils.py`. Updated 7 modules (kelly_sizing, regime_alerts, stats, weekly_digest, dashboard/app) to delegate to shared helpers (~230 lines removed).
- **Deduplication**: Centralized `BINANCE_BASE`, `BINANCE_FAPI_BASE`, `ALPACA_BASE` in `api_utils.py`. Updated 7 modules (data_collector, iskbets, ml_signal, funding_rate, data_refresh, outcome_tracker, macro_context) to import from canonical source.
- **Performance**: `_write_compact_summary()` in reporting.py uses selective dict comprehensions instead of `copy.deepcopy(summary)` — avoids copying entire 30+ ticker × 7 timeframe dict.
- **Robustness**: `agent_invocation._log_trigger()` uses `atomic_append_jsonl()` instead of raw `open("a")` append.
- **Cleanup**: Removed unused `import sys` from analyze.py, unused `import copy` from reporting.py.
- **Tests**: Added 21 tests for `load_json`, `load_jsonl`, `atomic_append_jsonl` (1164 total passing).

### Phase 2 (prior session)
- **Thread-safety**: `collect_timeframes()` cache reads/writes now use `_cache_lock`
- **Performance**: `backfill_outcomes()` opens SignalDB once instead of per-outcome
- **Performance**: `write_agent_summary()` uses cached accuracy stats (avoids redundant log scans)
- **Deduplication**: iskbets `_compute_atr_15m_impl` now calls `data_collector._fetch_klines()` (-100 lines)
- **Deduplication**: dashboard `/api/validate-portfolio` delegates to `portfolio_validator` (-70 lines)
- **Logging**: telegram_poller.py uses structured `logger.warning()` instead of `print()`
- **Cleanup**: Removed redundant `sys.path.insert` calls from dashboard endpoints
- **Performance**: health.py reads last 4KB of invocations.jsonl instead of full scan

### Phase 1 (initial session)
- Extracted `signal_utils.py` with 8 shared helpers from 10+ signal modules (~230 lines removed)
- Added error logging to 5 bare `except: pass` blocks in signal_engine.py
- Added threading lock to shared_state.py cache eviction
- Fixed MACD edge case in bigbet.py (missing macd_hist_prev)
- Fixed technical_signal() thresholds in indicators.py (RSI 30/70, EMA deadband)
- Streaming reads in journal.py, cached invocation time in health.py, ATR caching in iskbets.py
- Added 142 new tests (933 → 1075): signal_utils, risk_management, kelly_sizing, equity_curve

### Known Issues
1. **architecture-plan.md**: Still references "Custom LoRA" as signal #11 but it's disabled
2. **outcome_tracker._derive_signal_vote**: Duplicates signal logic for legacy entries (documented, no fix needed)
3. **Enhanced signal coverage**: 10/14 enhanced signal modules lack dedicated unit tests
4. **backfill_outcomes()**: Still loads entire JSONL into memory for read-modify-write. Future: migrate to SQLite-only reads.

## LoRA Status (assessed Feb 22)

### Accuracy Data (1,743 A/B entries, Feb 12-20)

| Model | 1d Accuracy | Action Distribution | Status |
|-------|-------------|---------------------|--------|
| **CryptoTrader-LM LoRA** | 44.0% (1,632 samples) | 62% SELL, 29% HOLD, 9% BUY | Active but underperforming |
| **Custom LoRA** | 20.9% (disabled) | 77% SELL, 15% BUY, 8% HOLD | Disabled Feb 20 |
| Agreement rate | 60.4% | Both heavily SELL-biased | — |

### Analysis

Both LoRA adapters have a severe SELL bias. The original CryptoTrader-LM LoRA at **44% accuracy is worse than a coin flip** — it's anti-predictive. At 3d horizon it drops to 40.4%. The model outputs SELL 62% of the time regardless of market conditions.

For comparison, the best signals are: funding (96.4%), fear_greed (67.7%), ML classifier (66.2%), calendar (64.4%). The LoRA is the 4th worst signal overall.

### Recommendation

1. **Short-term:** Consider disabling the original LoRA too, or inverting its signal (if it says SELL, treat as weak BUY evidence)
2. **Medium-term:** Run base Ministral without any LoRA adapter to establish a baseline — the LoRA may be making the base model worse
3. **Long-term:** If retraining, need a much more balanced training set (current training data is likely SELL-heavy, producing SELL-biased outputs)

## Test Coverage

**1164 tests passing** (15 integration tests deselected: ta_base_strategy import).

### Well-covered modules
- signal_utils.py, signal_engine.py, accuracy_stats.py, risk_management.py, kelly_sizing.py
- equity_curve.py, portfolio_mgr.py, http_retry.py, trigger.py, file_utils.py, signal_registry.py

### Coverage gaps (no dedicated tests)
- 10/14 enhanced signal modules (trend, momentum, volume_flow, volatility, candlestick, structure, fibonacci, smart_money, oscillators, heikin_ashi)
- dashboard/app.py (48 tests in test_dashboard.py but no sub-indicator edge cases)
- bigbet.py, iskbets.py, journal.py context formatting
