# System Overview

Updated: 2026-03-10
Branch: metals-state-store

## 1) Architecture Summary

Two-layer autonomous trading system with 30 signals, 19 instruments, and dual-strategy portfolio management.

- **Layer 1** (`portfolio/main.py`): Continuous 60s loop — data collection, signal generation, trigger detection, summary writing.
- **Layer 2** (`portfolio/agent_invocation.py`): Claude subprocess — reads summaries, makes trade decisions, writes journals, sends Telegram.
- **Autonomous mode** (`portfolio/autonomous.py`): Fallback when Layer 2 disabled — signal-based decisions without LLM.
- **Dashboard** (`dashboard/app.py`): Flask REST API over `data/` files, port 5055.
- **Metals subsystem** (`data/metals_loop.py`): Separate autonomous warrant trading loop via Avanza API, using worktree-relative repo paths and atomic shared state files.

## 2) Entry Points

| Surface | Entry | Notes |
|---------|-------|-------|
| Main loop | `portfolio/main.py --loop` | Via `scripts/win/pf-loop.bat`, auto-restart |
| One-shot | `--report`, `--accuracy`, `--check-outcomes` | Developer/cron tools |
| Dashboard | `dashboard/app.py` | Port 5055, optional token auth |
| Metals loop | `data/metals_loop.py` | Separate process, warrant trading |
| Agent | `scripts/win/pf-agent.bat` | Spawns Claude CLI for Layer 2 |

## 3) Module Map (~95 portfolio modules)

### Orchestration (5 modules)
- `main.py` (642 lines): Loop lifecycle, crash backoff (10s→5min), health heartbeat, module orchestration
- `agent_invocation.py` (206 lines): Layer 2 subprocess lifecycle, tiered prompts (T1/T2/T3), timeout killing
- `trigger.py` (326 lines): Change detection — consensus flip, price >2%, F&G threshold, sentiment reversal, post-trade
- `market_timing.py` (80 lines): DST-aware US market hours, agent invocation window
- `config_validator.py`: Startup config validation

### Signal System (30 signals: 8 core + 19 enhanced + 3 disabled)
- `signal_engine.py` (727 lines): 30-signal voting, weighted consensus, accuracy inversion, confidence penalties
- `signal_registry.py` (130 lines): Plugin-based signal discovery via importlib
- `signal_utils.py` (130 lines): Shared helpers — SMA, EMA, RSI, majority_vote
- `signals/*.py` (19 modules): Enhanced composite signals, each with 4-8 sub-indicators
- `accuracy_stats.py` (636 lines): Per-signal hit rate tracking, accuracy cache, activation rates
- `outcome_tracker.py` (391 lines): Signal snapshot logging, price backfill for accuracy

### Data Collection (3 modules)
- `data_collector.py` (259 lines): Binance spot/FAPI, Alpaca, yfinance; circuit breakers; 7 timeframes
- `indicators.py` (167 lines): RSI, MACD, EMA, BB, ATR, regime detection (cache per cycle)
- `shared_state.py` (124 lines): Thread-safe cache (TTL + stale fallback), rate limiters

### Portfolio & Risk (7 modules)
- `portfolio_mgr.py` (57 lines): State load/save via atomic_write_json
- `trade_guards.py` (267 lines): Per-ticker cooldown, consecutive-loss escalation, position rate limit
- `risk_management.py` (710 lines): Drawdown circuit breaker (-15%), ATR stops, concentration risk, correlation pairs
- `equity_curve.py` (599 lines): FIFO round-trip matching, Sharpe/Sortino, max drawdown, calmar ratio
- `monte_carlo.py` (401 lines): GBM with antithetic variates, probability-driven drift
- `monte_carlo_risk.py` (504 lines): Student-t copula VaR/CVaR, correlation priors
- `kelly_sizing.py`: Kelly criterion position sizing

### Reporting & Analysis (6 modules)
- `reporting.py` (962 lines): agent_summary.json (full/compact/tiered), three-tier compaction
- `journal.py`: Layer 2 journal JSONL streaming
- `journal_index.py` (400 lines): BM25 relevance ranking, importance scoring
- `reflection.py` (243 lines): Periodic strategy metrics (win rate, avg PnL)
- `prophecy.py`: Macro belief system (silver_bull_2026, etc.)
- `focus_analysis.py`: Mode B probability format for focus instruments

### External Data (11 modules)
- `fear_greed.py`, `sentiment.py`, `social_sentiment.py`, `onchain_data.py`
- `funding_rate.py`, `alpha_vantage.py`, `fx_rates.py`, `futures_data.py`
- `ministral_signal.py`, `ministral_trader.py`, `ml_signal.py` (disabled)

### Notification (5 modules)
- `telegram_notifications.py` (138 lines): Send with Markdown escaping, 4096 char limit, fallback
- `telegram_poller.py`: Incoming message polling, command handling
- `message_store.py`: Transaction/notification logging
- `message_throttle.py`: Analysis message rate limiting
- `digest.py` (206 lines): 4-hour periodic digest with invocation stats

### Infrastructure (6 modules)
- `file_utils.py` (129 lines): atomic_write_json, load_json/jsonl, prune_jsonl
- `http_retry.py` (66 lines): Exponential backoff (3 retries, 1s base, 2x factor)
- `circuit_breaker.py` (97 lines): Thread-safe state machine (CLOSED→OPEN→HALF_OPEN)
- `health.py` (187 lines): Heartbeat, error ring buffer, module failure tracking
- `logging_config.py` (48 lines): RotatingFileHandler (10MB, 3 backups)
- `signal_db.py`: WAL-mode SQLite dual-write with JSONL fallback

## 4) Data Flow

```
main.loop()
  → market_timing.get_market_state() → select active instruments
  → for each ticker:
      data_collector.collect_timeframes() → 7 OHLCV DataFrames
      indicators.compute_indicators() → RSI, MACD, EMA, BB, ATR
      signal_engine.generate_signal() → 30 votes → consensus action
  → trigger.check_triggers() → compare vs persistent baseline
  → if triggered:
      reporting.write_agent_summary() → agent_summary.json
      reporting.write_tiered_summary() → T1/T2/T3 context files
      outcome_tracker.log_signal_snapshot() → signal_log.jsonl + SQLite
      agent_invocation.invoke_agent() → spawns Claude CLI subprocess
        OR autonomous.autonomous_decision() → fallback when L2 disabled
  → post-cycle:
      digest._maybe_send_digest() → 4h Telegram digest
      health.update_health() → heartbeat + error tracking
      reflection.maybe_reflect() → periodic strategy metrics
      file_utils.prune_jsonl() → keep last 5000 entries per file
```

## 5) Signal Architecture

### Consensus Formula
- Active voters = signals that voted BUY or SELL (not HOLD)
- MIN_VOTERS = 3 (all asset classes)
- Core gate: at least 1 core signal must be active for non-HOLD
- Confidence = active_voters_in_direction / total_active_voters
- Sub-50% accuracy signals auto-inverted (30% BUY → 70% SELL)
- Recency-weighted: 70% recent (7d) + 30% all-time

### Weighted Consensus
- Weight = accuracy_weight × regime_multiplier × activation_frequency_normalization
- Regime weights: trending → trust EMA/MACD more; ranging → trust RSI/BB more
- Activation rates: rare, balanced signals get bonus; noisy/biased get penalty

### Signal Inventory (30 total)
- **Core active (8)**: RSI, MACD, EMA, BB, Fear&Greed, Sentiment, Ministral-8B, Volume
- **Core disabled (3)**: ML Classifier (28.2%), Funding Rate (27.0%), Custom LoRA (20.9%)
- **Enhanced composite (19)**: Trend, Momentum, Volume Flow, Volatility, Candlestick, Structure, Fibonacci, Smart Money, Oscillators, Heikin-Ashi, Mean Reversion, Calendar, Macro Regime, Momentum Factors, News Event, Econ Calendar, Forecast, Claude Fundamental, Futures Flow

## 6) Configuration

Primary config: `config.json` (not in repo). Key domains:
- `telegram.token`, `telegram.chat_id`, `telegram.layer1_messages`
- `alpaca.api_key`, `alpaca.api_secret`
- `layer2.enabled`, `layer2.max_turns`, `layer2.timeout`
- `notification.mode` ("signals" | "probability"), `notification.focus_tickers`
- Feature flags: `trade_guards.enabled`, `risk_audit.enabled`, `reflection.enabled`
- `forecast.kronos_enabled`, `claude_fundamental.enabled`
- `bigbet.enabled`, `iskbets.*`, `dashboard_token`

### External API Integrations (all configured as of 2026-03-11)

| Service | Config Key | Tier | Purpose |
|---------|-----------|------|---------|
| Binance | `exchange.key/secret` | Free | Crypto spot + FAPI futures (BTC, ETH, XAU, XAG) |
| Alpaca | `alpaca.key/secret` | Paper | US stock OHLCV data (15 NASDAQ/NYSE tickers) |
| Telegram | `telegram.token/chat_id` | Free | All Layer 2 notifications + digest |
| Alpha Vantage | `alpha_vantage.api_key` | Free (25/day) | Stock fundamentals (P/E, revenue, analyst targets) |
| NewsAPI | `newsapi_key` | Free (100/day) | Stock headlines for sentiment + news_event signal |
| FRED | `golddigger.fred_api_key` | Free | US 10Y Treasury yield (DGS10) for GoldDigger |
| BGeometrics | `bgeometrics.api_token` | Free (15/day) | Bitcoin on-chain (MVRV, SOPR, NUPL, netflow) |
| Claude (CLI) | via Max subscription | Max sub | Claude Fundamental signal #29 (Haiku/Sonnet/Opus cascade) |
| CryptoCompare | Public (no key) | Free | Crypto news headlines |
| Yahoo Finance | Public (no key) | Free | Stock prices, VIX, news fallback |
| Alternative.me | Public (no key) | Free | Crypto Fear & Greed Index |
| Avanza | Playwright BankID session | Manual | Warrant + Nordic equity trading (manual login ~24h) |

**Avanza auth:** Manual BankID login every ~24h via `scripts/avanza_login.py` (Playwright).
Session stored in `data/avanza_storage_state.json`. Config `avanza.username/password/totp_secret`
are empty — credentials not yet automated. Plan: add TOTP-based auto-renewal.

## 7) Test Surface

- ~3,168 tests across 105+ test files
- Sequential: ~16 min; Parallel (`-n auto`): ~5.5 min
- 26 pre-existing failures (integration/strategy, consensus thresholds)
- Config: `pyproject.toml` → `[tool.pytest.ini_options]`
- Linter: ruff (line-length=120, target py311)
- 7 untested utility modules: telegram_poller, data_refresh, backup, log_rotation, social_sentiment, stats, regime_alerts

## 8) Key Design Patterns

- **Atomic writes**: `file_utils.atomic_write_json()` prevents corrupt state files
- **Metals shared state**: positions, stop orders, trade queue, and guard/spike state now use atomic JSON writes with explicit corrupt-file logging
- **Circuit breakers**: Per-API failure tracking with auto-recovery
- **Cache-through**: TTL cache with stale-data fallback (shared_state._cached)
- **Tiered invocation**: T1 (quick, 70%), T2 (signal, 25%), T3 (full, 5%)
- **Three-tier compaction**: Held → full votes; triggered → vote_detail string; HOLD → minimal
- **Crash protection**: Exponential backoff (10s→5min), alert suppression after 5 crashes
- **Graceful degradation**: Each signal/module wrapped in try/except, module warnings surfaced

## 9) Known Issues (as of 2026-03-10)

- BUG-15 through BUG-22: Fixed in 2026-03-08 session
- BUG-23 through BUG-27: Fixed in 2026-03-09 session (signal validation, None ticker, OSError, heartbeat, pass cleanup)
- BUG-28: Enhanced signal failures silently count as HOLD — no tracking or surfacing
- BUG-29: `_vote_correct()` treats 0% change as incorrect — biases accuracy downward in flat markets
- BUG-30: `load_json()` has TOCTOU race between `exists()` and `read_text()`
- BUG-31: `_compute_adx()` not cached, uses NaN propagation via `replace(0, np.nan)`
- BUG-32: main.py re-exports ~50 private symbols (documentation only, not fixing this session)
- BUG-33: Trap detection relies on undocumented assumption about `df` timeframe
- ARCH-10: Signal result validation centralized in `_validate_signal_result()`
- ARCH-11: Confidence caps enforced via `max_confidence` in signal registry
- ARCH-12: Signal failure tracking and surfacing (planned)
- ARCH-13: Accuracy tolerance for flat markets (planned)
- REF-3: Candlestick `patterns_detected` moved from top-level to `indicators` dict
- TEST coverage: candlestick (57 tests), fibonacci (51 tests), structure (32 tests) — formerly zero
- ~3,310 tests across 110+ test files
