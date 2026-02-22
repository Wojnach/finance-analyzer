# System Overview — Portfolio Intelligence Trading Agent

> Generated: 2026-02-22 | Auto-improvement session
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
| `api_utils.py` | ~100 | Config loading, Alpaca header helper |
| `config_validator.py` | ~76 | Startup config.json validation |
| `tickers.py` | 113 | Single source of truth: SYMBOLS dict → all derived mappings |

### Signal Pipeline
| Module | Lines | Purpose |
|--------|-------|---------|
| `signal_engine.py` | 564 | 25-signal voting, weighted consensus, sentiment hysteresis |
| `signal_registry.py` | 113 | Plugin-style registration for enhanced signal modules |
| `indicators.py` | 140 | RSI, MACD, EMA, BB, ATR, regime detection |
| `signals/*.py` (14) | ~2500 | Enhanced composite signals (trend, momentum, etc.) |
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

### Likely Dead Code (never imported)
| Module | Evidence |
|--------|---------|
| `collect.py` | Old data collector, no imports found anywhere |
| `stats.py` | Unused stats utility, no imports found |
| `social_sentiment.py` | No imports found |
| `avanza_watch.py` | No imports found |

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

## Key Discrepancies Found vs Existing Docs

1. **system-design.md line 843**: Says "test_digest.py — 0 tests" but `tests/test_digest.py` exists with tests
2. **system-design.md line 845**: Says "Trigger system — No unit tests" but `tests/test_trigger_edge_cases.py` has 40+ tests
3. **architecture-plan.md**: Still references "Custom LoRA" as signal #11 but it's disabled
4. **system-design.md line 756**: Says "no proactive health monitoring that alerts on silence" but `health.py` has `check_agent_silence()`
5. **accuracy_stats.py**: Still has local `_atomic_write_json` instead of shared `file_utils`
6. **signal_engine.py**: Inline tempfile/os atomic write in `_set_prev_sentiment()` instead of `file_utils`
7. **outcome_tracker.py line 354**: Inline mkstemp instead of `file_utils`
