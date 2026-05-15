# System Overview

Updated: 2026-05-15
Branch: improve/auto-session-2026-05-15

## 1) Architecture Summary

Two-layer autonomous trading system with 52 signals (33 active, 19 disabled), 5 Tier-1 instruments, and dual-strategy portfolio management.

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

## 3) Module Map (~152 portfolio modules)

### Orchestration (5 modules)
- `main.py` (909 lines): Loop lifecycle, crash backoff (10s→5min), health heartbeat, parallel ticker processing via ThreadPoolExecutor(8)
- `agent_invocation.py` (489 lines): Layer 2 subprocess lifecycle, tiered prompts (T1/T2/T3), timeout killing, completion tracking, stack overflow auto-disable. Spawn-vs-watchdog race fixed (2026-05-12): metadata set before Popen so watchdog never sees stale start time.
- `trigger.py` (330 lines): Change detection — consensus flip (with clock-skew guard), price >2%, F&G threshold, sentiment reversal, post-trade
- `market_timing.py` (141 lines): DST-aware US market hours, agent invocation window, market state (open/closed/weekend)
- `config_validator.py`: Startup config validation

### Signal System (52 signals: 12 core + 40 enhanced, 33 active + 19 disabled)
- `signal_engine.py` (~4,200 lines): 52-signal voting, weighted consensus, accuracy gating, 8-stage confidence penalties, correlation groups, horizon-aware regime gating, dynamic horizon weights, thread-safe sentiment + content-keyed ADX cache
- `signal_registry.py` (~300 lines): Plugin-based signal discovery via importlib, lazy loading. All signals registered as "enhanced" via `register_enhanced()`. 5-min import-failure cooldown.
- `signal_utils.py` (130 lines): Shared helpers — SMA, EMA, RSI, majority_vote
- `signals/*.py` (24 modules): Enhanced composite signals, each with 4-8 sub-indicators
- `accuracy_stats.py` (636 lines): Per-signal hit rate tracking, accuracy cache, activation rates
- `outcome_tracker.py` (391 lines): Signal snapshot logging, price backfill for accuracy

### Data Collection (3 modules)
- `data_collector.py` (299 lines): Binance spot/FAPI, Alpaca, yfinance; circuit breakers; 7 timeframes
- `indicators.py` (167 lines): RSI, MACD, EMA, BB, ATR, regime detection (cache per cycle)
- `shared_state.py` (206 lines): Thread-safe cache (TTL + stale fallback), rate limiters, NewsAPI quota

### Portfolio & Risk (7 modules)
- `portfolio_mgr.py` (181 lines): State load/save with rolling backups (C7), per-file locks (C8), atomic read-modify-write, corruption recovery
- `trade_guards.py` (267 lines): Per-ticker cooldown, consecutive-loss escalation, position rate limit
- `risk_management.py` (710 lines): Drawdown circuit breaker (-15%), ATR stops, concentration risk, correlation pairs
- `equity_curve.py` (599 lines): FIFO round-trip matching, Sharpe/Sortino, max drawdown, calmar ratio
- `monte_carlo.py` (401 lines): GBM with antithetic variates, probability-driven drift
- `monte_carlo_risk.py` (504 lines): Student-t copula VaR/CVaR, correlation priors
- `kelly_sizing.py`: Kelly criterion position sizing

### Reporting & Analysis (6 modules)
- `reporting.py` (962 lines): agent_summary.json (full/compact/tiered), three-tier compaction, thread-safe module failure tracking
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

### Infrastructure (10 modules)
- `file_utils.py` (~250 lines): atomic_write_json, load_json/jsonl, prune_jsonl, atomic_append_jsonl, load_jsonl_tail, last_jsonl_entry
- `http_retry.py` (66 lines): Exponential backoff (3 retries, 1s base, 2x factor)
- `circuit_breaker.py` (97 lines): Thread-safe state machine (CLOSED→OPEN→HALF_OPEN)
- `health.py` (~340 lines): Heartbeat, error ring buffer, module failure tracking, signal health, dead signal detection
- `logging_config.py` (48 lines): RotatingFileHandler (10MB, 3 backups)
- `signal_db.py`: WAL-mode SQLite dual-write with JSONL fallback. `load_entries()` optimized from O(n²) per-snapshot SELECTs to 3 bulk queries + dict reassembly (2026-05-12).
- `process_lock.py` (101 lines): Cross-platform non-blocking file locks (msvcrt/fcntl)
- `subprocess_utils.py` (260 lines): Windows Job Object subprocess protection, orphan reaper. WMIC→PowerShell migration (2026-05-12): `kill_orphaned_by_cmdline()` uses Get-CimInstance for Win11 compat.
- `notification_text.py` (65 lines): Shared text helpers for human-readable notifications
- `llama_server.py` (309 lines): Unified persistent llama-server manager, cross-process model swap with file lock, query-scoped locking (BUG-165)

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
- Accuracy gate: signals below 47% accuracy (30+ samples) are force-HOLD (not inverted). 50% for 10K+ samples.
- Recency-weighted: 70% recent (7d) + 30% all-time; fast blend (90/10) on 15%+ divergence
- Global confidence cap: 0.80 (70-80% bracket has best actual accuracy at 57-59%)

### Weighted Consensus
- Weight = directional_accuracy_weight × regime_mult × horizon_mult × activation_norm × activity_cap × correlation_penalty
- **Directional accuracy** (BUG-182, 2026-04-11): BUY votes weighted by `buy_accuracy`, SELL by `sell_accuracy`. Falls back to overall accuracy when directional samples < 20. Prevents signals with asymmetric accuracy (e.g., qwen3 BUY 30% vs SELL 74%) from being over-weighted in their weak direction.
- Regime weights: trending → trust EMA/MACD more; ranging → trust RSI/BB more
- Regime gating: horizon-aware — some signals gated in certain regimes only for specific prediction horizons
- Horizon weights: dynamic (computed from accuracy cache ratio this_horizon/cross_horizon) with static fallback
- Correlation groups: within groups of correlated signals, only the best-accuracy signal gets full weight; others get 0.3x
- Activity rate cap: signals with >70% activation rate get 0.5x penalty
- Activation rates: rare, balanced signals get bonus; noisy/biased get penalty

### 5-Stage Confidence Penalty Cascade
1. Regime penalty: ranging 0.75x, high-vol 0.80x, trend-aligned +10%
2. Volume/ADX gate: RVOL <0.5 → force HOLD; RVOL <0.8 + ADX <20 + conf <65% → force HOLD
3. Trap detection: price up + volume declining → 0.5x (bull trap); same for bear traps
4. Dynamic MIN_VOTERS: trending=3, high-vol=4, ranging=5
5. Unanimity penalty: 90%+ agreement → 0.6x, 80-90% → 0.75x (high unanimity = already priced in)

### Signal Inventory (52 total: 33 active, 19 disabled)
- **Core active (10)**: RSI, MACD, EMA, BB, Fear&Greed, Sentiment, Ministral-8B, Qwen3-8B, Volume, Funding Rate (3h-only, 74.2%)
- **Core active BTC-only (1)**: On-Chain BTC (MVRV, SOPR, NUPL, Netflow)
- **Core disabled (1)**: ML Classifier (41.7%)
- **Enhanced active (22)**: Trend, Momentum, Volume Flow, Volatility, Candlestick, Structure, Fibonacci, Smart Money, Heikin-Ashi, Mean Reversion, Calendar, Macro Regime, Momentum Factors, News Event, Econ Calendar, Forecast, Claude Fundamental, Futures Flow, Metals Cross-Asset, DXY Cross-Asset, COT Positioning, Credit Spread Risk
- **Enhanced disabled (19)**: ML, Oscillators (below 45%), Orderbook Flow (51.1%), Smart Money (2026-04-24), Mahalanobis Turbulence, Crypto EVRP, Futures Basis, Hurst Regime, Shannon Entropy, VIX Term Structure, Gold Real Yield Paradox, Cross-Asset TSMOM, Copper/Gold Ratio, Statistical Jump Regime, Network Momentum, OVX Metals Spillover, XTrend Equity Spillover, Complexity Gap Regime, Realized Skewness (all pending live validation, added Apr 2026)

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
| Alpaca | `alpaca.key/secret` | Paper | US stock OHLCV data (MSTR only, reduced from 15 tickers on Apr 9) |
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

- ~5,994 tests across 242 test files
- Sequential: ~16 min; Parallel (`-n auto`): ~5.5 min (2.9x speedup on 8 workers)
- 26 pre-existing failures (integration/strategy, consensus thresholds, forecast config)
- Config: `pyproject.toml` → `[tool.pytest.ini_options]`
- Linter: ruff (line-length=120, target py311)
- Fixtures: `conftest.py` provides `make_indicators()`, `make_candles()`, `make_ohlcv_df()`, `sample_config`, `config_file`, `tmp_data_dir`
- 76 tests use `tmp_path` isolation for parallel-safe file I/O
- 7 untested utility modules: telegram_poller, data_refresh, backup, log_rotation, social_sentiment, stats, regime_alerts

## 8) Key Design Patterns

- **Atomic writes**: `file_utils.atomic_write_json()` and `atomic_write_jsonl()` prevent corrupt state files. All portfolio modules (including golddigger/elongir subsystems) use `load_json()`/`load_jsonl()` — zero raw `json.loads(path.read_text())` calls remain (enforced by `test_io_safety_sweep.py`)
- **Metals shared state**: positions, stop orders, trade queue, and guard/spike state now use atomic JSON writes with explicit corrupt-file logging
- **Circuit breakers**: Per-API failure tracking with auto-recovery
- **Cache-through**: TTL cache with stale-data fallback (shared_state._cached)
- **Tiered invocation**: T1 (quick, 70%), T2 (signal, 25%), T3 (full, 5%)
- **Three-tier compaction**: Held → full votes; triggered → vote_detail string; HOLD → minimal
- **Crash protection**: Exponential backoff (10s→5min), alert suppression after 5 crashes
- **Graceful degradation**: Each signal/module wrapped in try/except, module warnings surfaced

## 9) Known Issues (as of 2026-05-04)

**244 bugs fixed** across 70+ sessions (BUG-15 through BUG-244).
Full history: [docs/RESOLVED_BUGS.md](RESOLVED_BUGS.md).

### Open Issues

- ARCH-17: main.py re-exports 100+ symbols (obscures module boundaries)
- ARCH-18/BUG-162: metals_loop.py is 7,699-line monolith — highest bug density, hardest to maintain
- ARCH-19: No CI/CD pipeline — all testing is manual
- ARCH-20: No type checking (mypy)
- BUG-132: orb_predictor.py fetches 5000+ candles uncached
- BUG-149: meta_learner orphaned (predict() never called from production)
- TEST-1: GPU gate (`gpu_gate.py`) has zero test coverage
- TEST-3: 26+ pre-existing test failures (integration, config, state isolation)

### Findings from 2026-05-04 Auto Session

- **B1 (fixed)**: Equity curve annualized with 252 days (stock convention) but portfolio runs 24/7 → crypto volatility understated 17%. Changed to 365.
- **B2 (fixed)**: Contract violation dedup wrote critical_errors.jsonl BEFORE dedup marker → duplicate entries on marker write failure. Swapped order.
- **B3 (fixed)**: Monte Carlo ATR fallback was generic 2.0% for all assets. Now per-asset-class: crypto=3.5%, metals=4.0%, stocks=2.0%.
- **B4 (fixed)**: Stuck loading key eviction in shared_state.py logged at DEBUG. Elevated to WARNING.
- ~6,000+ tests across 242 test files

### Findings from 2026-05-12 Auto Session

- **B1 (fixed)**: Agent spawn race — watchdog could kill freshly spawned process due to stale `_agent_start`. Fix: set metadata before Popen.
- **B2 (fixed)**: WMIC deprecated in Win11 — `kill_orphaned_by_cmdline()` migrated to PowerShell Get-CimInstance.
- **B3 (fixed)**: `signal_db.load_entries()` O(n²) per-snapshot queries → 3 bulk queries + dict reassembly.
- **B4 (skipped)**: health.py fromisoformat already guarded at all 3 call sites.
- **B5 (partial)**: `__import__("json")` → standard import in metals_cross_asset.py. oscillator_trend correlation group verified NOT dead code.
- **False positives rejected**: risk_management concentration min() is correct; grid_fisher ORDER_FILLED prevents double-count.

### Findings from 2026-05-15 Auto Session

- **B1 (false positive)**: Portfolio backup rotation before write looked like a crash-data-loss risk, but `_atomic_write_json` uses `os.replace()` which is atomic — if write crashes before replace, original file stays intact. No fix needed.
- **B2 (fixed)**: ADX cache key used `id(df)` (memory address) — GC address reuse could return stale ADX. Replaced with content-based key `(len, first_close, last_close)`.
- **B3 (fixed)**: Flip cooldown in `trigger.py` used `time.time()` — backward NTP adjustments made elapsed time negative, suppressing all flip triggers for up to 30 min. Added clock-skew reset guard.
- **B4 (fixed)**: `alert_budget.py` `AlertBudget` class had no thread safety. Added `threading.Lock()` around all public methods.
- **B5 (fixed)**: `reporting.py` `_module_failure_streaks` dict and `_module_escalated` set mutated from ThreadPoolExecutor threads without synchronization. Added `_module_lock`; escalation I/O moved outside lock.
- **B6 (fixed)**: `data_collector._fetch_one_timeframe()` returned `None` on `compute_indicators()` failure, silently dropping timeframes. Now returns `(label, {"error": ...})` for explicit error visibility.
