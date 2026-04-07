# System Overview

Updated: 2026-04-07
Branch: improve/auto-session-2026-04-07

## 1) Architecture Summary

Two-layer autonomous trading system with 32 signals (all tracked), 20 instruments, and dual-strategy portfolio management.

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

## 3) Module Map (~142 portfolio modules)

### Orchestration (5 modules)
- `main.py` (909 lines): Loop lifecycle, crash backoff (10s→5min), health heartbeat, parallel ticker processing via ThreadPoolExecutor(8)
- `agent_invocation.py` (489 lines): Layer 2 subprocess lifecycle, tiered prompts (T1/T2/T3), timeout killing, completion tracking, stack overflow auto-disable
- `trigger.py` (330 lines): Change detection — consensus flip, price >2%, F&G threshold, sentiment reversal, post-trade
- `market_timing.py` (141 lines): DST-aware US market hours, agent invocation window, market state (open/closed/weekend)
- `config_validator.py`: Startup config validation

### Signal System (32 signals: 9 core + 22 enhanced + 2 disabled)
- `signal_engine.py` (1,538 lines): 32-signal voting, weighted consensus, accuracy gating, 5-stage confidence penalties, correlation groups, horizon-aware regime gating, dynamic horizon weights, thread-safe sentiment + ADX cache
- `signal_registry.py` (135 lines): Plugin-based signal discovery via importlib, lazy loading
- `signal_utils.py` (130 lines): Shared helpers — SMA, EMA, RSI, majority_vote
- `signals/*.py` (19 modules): Enhanced composite signals, each with 4-8 sub-indicators
- `accuracy_stats.py` (636 lines): Per-signal hit rate tracking, accuracy cache, activation rates
- `outcome_tracker.py` (391 lines): Signal snapshot logging, price backfill for accuracy

### Data Collection (3 modules)
- `data_collector.py` (299 lines): Binance spot/FAPI, Alpaca, yfinance; circuit breakers; 7 timeframes
- `indicators.py` (167 lines): RSI, MACD, EMA, BB, ATR, regime detection (cache per cycle)
- `shared_state.py` (206 lines): Thread-safe cache (TTL + stale fallback), rate limiters, NewsAPI quota

### Portfolio & Risk (7 modules)
- `portfolio_mgr.py` (77 lines): State load/save via atomic_write_json, TOCTOU-safe via load_json
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

### Infrastructure (10 modules)
- `file_utils.py` (~250 lines): atomic_write_json, load_json/jsonl, prune_jsonl, atomic_append_jsonl, load_jsonl_tail, last_jsonl_entry
- `http_retry.py` (66 lines): Exponential backoff (3 retries, 1s base, 2x factor)
- `circuit_breaker.py` (97 lines): Thread-safe state machine (CLOSED→OPEN→HALF_OPEN)
- `health.py` (~340 lines): Heartbeat, error ring buffer, module failure tracking, signal health, dead signal detection
- `logging_config.py` (48 lines): RotatingFileHandler (10MB, 3 backups)
- `signal_db.py`: WAL-mode SQLite dual-write with JSONL fallback
- `process_lock.py` (101 lines): Cross-platform non-blocking file locks (msvcrt/fcntl)
- `subprocess_utils.py` (233 lines): Windows Job Object subprocess protection, orphan reaper
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
- Accuracy gate: signals below 45% accuracy (30+ samples) are force-HOLD (not inverted)
- Recency-weighted: 70% recent (7d) + 30% all-time; fast blend (90/10) on 15%+ divergence
- Global confidence cap: 0.80 (70-80% bracket has best actual accuracy at 57-59%)

### Weighted Consensus
- Weight = accuracy_weight × regime_mult × horizon_mult × activation_norm × activity_cap × correlation_penalty
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

### Signal Inventory (32 total: all tracked)
- **Core active (9)**: RSI, MACD, EMA, BB, Fear&Greed, Sentiment, Ministral-8B, Qwen3-8B, Volume
- **Core disabled (2)**: ML Classifier (28.2%), Funding Rate (27.0%)
- **Enhanced composite (21)**: Trend, Momentum, Volume Flow, Volatility, Candlestick, Structure, Fibonacci, Smart Money, Oscillators, Heikin-Ashi, Mean Reversion, Calendar, Macro Regime, Momentum Factors, News Event, Econ Calendar, Forecast, Claude Fundamental, Futures Flow, Crypto Macro, Orderbook Flow, Metals Cross-Asset
- **Note**: BUG-160 fixed (2026-04-01) — all 3 previously untracked signals (crypto_macro, orderbook_flow, metals_cross_asset) added to SIGNAL_NAMES

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

- ~5,994 tests across 159 test files (156 main + 1 integration + 2 unit)
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

## 9) Known Issues (as of 2026-04-04)

- BUG-15 through BUG-22: Fixed in 2026-03-08 session
- BUG-23 through BUG-27: Fixed in 2026-03-09 session (signal validation, None ticker, OSError, heartbeat, pass cleanup)
- BUG-28: Enhanced signal failures silently count as HOLD — no tracking or surfacing (addressed by ARCH-12)
- BUG-29: `_vote_correct()` treats 0% change as incorrect — fixed with `_MIN_CHANGE_PCT` threshold
- BUG-30: `load_json()` TOCTOU race — fixed in 2026-03-10 session
- BUG-31: `_compute_adx()` not cached, uses NaN propagation via `replace(0, np.nan)` — fixed as BUG-84 (2026-03-19)
- BUG-32: main.py re-exports ~50 private symbols (documentation only)
- BUG-33: Trap detection relies on undocumented assumption about `df` timeframe
- BUG-34 through BUG-38: Fixed in 2026-03-11 session (portfolio_mgr TOCTOU, health TOCTOU, inversion weight cap)
- BUG-39 through BUG-46: Fixed in 2026-03-12 session (agent completion wiring, digest/daily_digest state isolation, TOCTOU-safe I/O in reporting+trigger, JSONL tail-read optimization, digest load limits, config dedup)
- BUG-47 through BUG-50: Fixed in 2026-03-14 session (IO safety sweep: raw json.loads, non-atomic writes, manual JSONL loops)
- BUG-51 (P1): Signal failure tracking ephemeral — no persistent record of degradation (2026-03-16)
- BUG-52 (P2): `total_applicable` hardcoded at 27/25, doesn't account for disabled/failing signals (2026-03-16)
- BUG-53 (P2): 7 modules use non-atomic JSONL appends instead of `atomic_append_jsonl()` (2026-03-16)
- BUG-54 (P3): `_compute_adx()` not cached — fixed as BUG-84 (2026-03-19)
- BUG-55 (P3): `fin_evolve.py` has dead ImportError fallback wrappers for file_utils (2026-03-16)
- BUG-80: Duplicate `"could"` in sentiment.py stopwords set — fixed 2026-03-19 (auto-fix)
- BUG-81: Missing `raise ... from None` in avanza_client.py — fixed 2026-03-19
- BUG-82: Unused imports in claude_gate.py — fixed 2026-03-19 (auto-fix)
- BUG-83: 5 remaining silent `except Exception: pass` handlers — fixed 2026-03-19
- BUG-84: `_compute_adx()` not cached per ticker — fixed 2026-03-19 (id(df) cache)
- ARCH-10: Signal result validation centralized in `_validate_signal_result()`
- ARCH-11: Confidence caps enforced via `max_confidence` in signal registry
- ARCH-12: Signal failure tracking and health surfacing (in progress, 2026-03-16)
- ARCH-13: Accuracy tolerance for flat markets (planned)
- ARCH-14: Dynamic `total_applicable` computation (in progress, 2026-03-16)
- ARCH-15: Centralized JSONL tail-read utility `last_jsonl_entry()` in `file_utils.py` (done 2026-03-12)
- REF-3: Candlestick `patterns_detected` moved from top-level to `indicators` dict
- REF-7: Removed legacy `trigger_state.json` migration in `signal_engine.py` (done 2026-03-12)
- REF-9: Consolidate 7 raw JSONL appends to `atomic_append_jsonl()` (in progress, 2026-03-16)
- REF-10: Remove dead `fin_evolve.py` fallback wrappers (in progress, 2026-03-16)
- REF-16: 1,910 ruff auto-fix violations (datetime.UTC, PEP 604, PEP 585, import sorting) — fixed 2026-03-19
- REF-17: 28 manual ruff fixes (B007 unused loop vars, B904 raise-from, SIM103 needless bool) — fixed 2026-03-19
- FEAT-2: Signal failure rate in accuracy reports (in progress, 2026-03-16)
- TEST coverage: candlestick (57 tests), fibonacci (51 tests), structure (32 tests) — formerly zero
- BUG-71/73: Golddigger/elongir config loading used raw `json.load()` — fixed 2026-03-18
- BUG-72: Golddigger sent Telegram directly bypassing message_store — fixed 2026-03-18
- BUG-74: Golddigger data_provider used raw `json.load()` for cached data — fixed 2026-03-18
- BUG-75/76/77: Dead variables in signal_engine, trigger, telegram_poller — fixed 2026-03-18
- BUG-79: avanza_tracker silent exception — fixed 2026-03-18
- REF-13: 112 ruff lint violations (unused imports, f-strings, reimports) — fixed 2026-03-18
- REF-14: 15 dead variable assignments across 13 modules — fixed 2026-03-18
- ARCH-16: Golddigger/elongir duplicated config loading (deferred — localized, may diverge)
- ~5,994 tests across 159 test files
- BUG-85 (P1): Thread-unsafe `_prev_sentiment` + per-ticker serialization data loss in signal_engine.py — fixed 2026-03-20
- BUG-86 (P2): Thread-unsafe `_adx_cache` in signal_engine.py — fixed 2026-03-20
- BUG-87 (P1): NaN propagation from compute_indicators into JSON and signals — fixed 2026-03-20
- BUG-88 (P1): Tier 1 votes string always shows 0 HOLD in reporting.py — fixed 2026-03-20
- BUG-89 (P1): `update_module_failures` crash prevents all summary output — fixed 2026-03-20
- BUG-90 (P2): Confidence penalty cascade amplifies above 1.0 before gates — fixed 2026-03-20
- BUG-91 (P1): Timed-out agent completion never logged — fixed 2026-03-20
- BUG-92 (P1): taskkill failure not checked; potential concurrent agents — fixed 2026-03-20
- BUG-93 (P2): Circuit breaker HALF_OPEN allows unlimited concurrent requests — fixed 2026-03-20
- BUG-95 (P2): Stack overflow counter not reset on non-overflow failures — fixed 2026-03-20
- BUG-97 (P2): Exception in check_agent_completion leaves agent state dirty — fixed 2026-03-20
- BUG-99 (P3): ZeroDivisionError if initial_value_sek is 0 — fixed 2026-03-20
- BUG-100 (P2): Empty Binance response recorded as circuit breaker success — fixed 2026-03-20
- BUG-101 through BUG-106: Crash safety, thread safety, zero-division, silent failures, alert routing, memory leak — fixed 2026-03-21
- BUG-107 (P2): Zero-division in digest P&L calculations — fixed 2026-03-22
- BUG-108 (P3): Alpha Vantage budget counter not thread-safe — fixed 2026-03-22
- BUG-109 (P3): Digest reads entire 68MB signal_log.jsonl — fixed 2026-03-22
- BUG-110 (P3): Stale import path in digest.py — fixed 2026-03-22
- BUG-111 (P1): outcome_tracker RSI vote derivation uses fixed 30/70, not adaptive thresholds — fixed 2026-03-23
- BUG-112 (P2): backfill_outcomes reads entire signal_log.jsonl into memory — fixed 2026-03-23 (streaming optimization: 75MB → 2MB)
- BUG-113 (P3): majority_vote HOLD confidence inconsistency — fixed 2026-03-23
- BUG-114 (P3): forecast JSON extraction fallbacks lack observability — fixed 2026-03-23
- BUG-115 (P1): `structure.py` signal module had no logging — failures silently returned HOLD — **fixed 2026-03-24** (added logger + exception logging)
- BUG-116 (P2): Trigger state pruning silently dropped tickers — **fixed 2026-03-24** (added logger.info on prune)
- BUG-117 (P2): `fx_rates.py` hardcoded fallback 10.85 SEK/USD — **fixed 2026-03-24** (bounds check 7-15, ERROR level logging)
- BUG-118 (P3): FOMC/econ dates hardcoded for 2026-2027 — **mitigated 2026-03-24** (staleness warnings in econ_calendar.py, calendar_seasonal.py)
- BUG-119 (P2): CLAUDE.md T2=25 turns but code=40 — **fixed 2026-03-24** (docs updated to 600s/40 turns)
- BUG-120 (P3): `_safe_series()` forward-fills silently — **fixed 2026-03-24** (warning when >5% values filled)
- BUG-121 (P3): Sector representative mapping in news_event.py is hardcoded dict — not extensible (deferred)
- Also added logging to `macro_regime.py` exception handlers (had no `import logging` at all) — **fixed 2026-03-24**
- BUG-122 (P1): health.py reads entire 68MB signal_log.jsonl (x2) — **fixed 2026-03-25** (load_jsonl_tail)
- BUG-123 (P1): Untracked files break worktrees (metals_ladder, process_lock, subprocess_utils, notification_text) — **fixed 2026-03-25** (tracked in git)
- BUG-124 (P2): fin_snipe_manager raw config.json read — **fixed 2026-03-25** (load_json)
- BUG-125 (P2): onchain_data non-atomic cache write — **fixed 2026-03-25** (atomic_write_json)
- BUG-126 (P3): main.py silent Telegram exception handlers — **fixed 2026-03-25** (logger.debug)
- BUG-127 (P3): crypto_scheduler silent exception handler — **fixed 2026-03-25** (logger.debug)
- REF-9: Raw JSONL append consolidation (5 files) — **fixed 2026-03-25** (atomic_append_jsonl)
- REF-10: fin_evolve.py aliased imports — **fixed 2026-03-25** (direct imports)
- Also synced 5 modified portfolio files + 5 test files from main working tree — **fixed 2026-03-25**
- ARCH-17: main.py re-exports 100+ symbols from submodules — obscures true module boundaries (deferred)
- ARCH-18: metals_loop.py is 1000+ lines monolith — should be split into modules (deferred)
- ARCH-19: No CI/CD pipeline — all testing is manual (deferred)
- ARCH-20: No type checking (mypy) configured (deferred)
- TEST-1: GPU gate module (`gpu_gate.py`) has zero test coverage
- TEST-2: Health module (`health.py`) had only 3 tests — **extended 2026-03-24** (24 new tests covering staleness, signal health, dead signals)
- TEST-3: 26 pre-existing test failures still unaddressed (integration, config, state isolation)
- BUG-128 (P2): `avanza_orders.py` non-atomic offset file — **fixed 2026-03-26** (atomic_write_json + legacy read compat)
- BUG-129 (P3): `avanza_session.py` Playwright globals not thread-safe — **fixed 2026-03-26** (threading.Lock)
- BUG-130 (P3): Dashboard no caching — **fixed 2026-03-26** (TTL cache, 5s default / 60s config)
- BUG-131 (P3): Telegram truncation breaks Markdown — **fixed 2026-03-26** (newline-boundary truncation)
- BUG-132 (P3): `orb_predictor.py` fetches 5000+ candles from Binance on every call with no caching (deferred)
- ARCH-21: `autonomous.py` has 500+ line functions (e.g., `_build_telegram_mode_a`) — should decompose for maintainability (deferred)
- ARCH-22: `agent_invocation.py` uses module-level globals for process lifecycle — could be a class for clarity (deferred)
- BUG-133 (P1): Accuracy cache shared timestamp causes cross-horizon staleness — **fixed 2026-03-27** (per-horizon timestamps)
- BUG-134 (P1): Regime accuracy hardcoded to "1d" regardless of prediction horizon — **fixed 2026-03-27** (uses acc_horizon)
- BUG-135 (P1): Signal utility always evaluated at "1d" horizon — **fixed 2026-03-27** (uses acc_horizon)
- BUG-136 (P2): Utility boost mutates accuracy_data dict in-place — **fixed 2026-03-27** (dict copy)
- BUG-137 (P2): SQLite DB resource leak in load_entries() — **fixed 2026-03-27** (try/finally)
- BUG-138 (P2): Backtester duplicates accuracy blending logic — **fixed 2026-03-27** (shared blend_accuracy_data)
- BUG-139 (P2): load_json() crashes on PermissionError — **fixed 2026-03-27** (OSError catch)
- ARCH-23: Extract accuracy blending into reusable function — **done 2026-03-27**
- ARCH-24: Parameterize accuracy functions with pre-loaded entries — **done 2026-03-27**
- BUG-143 (P1): Unanimity penalty uses pre-gated vote counts (regime-gated signals counted) — **fixed 2026-03-29** (gating applied before counts)
- BUG-144 (P1): Forecast regime discount is dead code (regime never passed in context_data) — **fixed 2026-03-29** (regime added to context)
- ARCH-25: Pass regime through context_data to enhanced signals — **done 2026-03-29**
- ARCH-26: Post-gated vote counts for penalty stages — **done 2026-03-29**
- BUG-145 (P2): meta_learner SQLite connection leak on exception — **fixed 2026-03-29** (try/finally)
- BUG-146 (P2): meta_learner old datetime import style — **fixed 2026-03-29** (UTC modernization)
- BUG-147 (P2): meta_learner duplicates SIGNAL_NAMES list — **fixed 2026-03-29** (import from tickers)
- BUG-148 (P2): meta_learner.predict() loads model from disk on every call — **fixed 2026-03-29** (module-level cache with mtime)
- ARCH-27: Meta-learner model caching for predict() — **done 2026-03-29**
- BUG-149 (P3): meta_learner orphaned — predict() never called from production (deferred — document or integrate as signal #31)
- BUG-157 (P1): `analyze.py:434` loop variable capture in closure (B023) — **fixed 2026-03-31** (default param binding)
- BUG-158 (P1): `test_signal_improvements.py:402,411` undefined `datetime` in lambda (F821) — **fixed 2026-03-31** (removed dead code)
- BUG-159 (P2): `avanza_session.py:255` raise without exception chaining (B904) — **fixed 2026-03-31** (`from None`)
- REF-21: 4 unused imports in portfolio/ (F401) — **fixed 2026-03-31** (auto-fix)
- REF-22: 3 unused variables in portfolio/ (F841) — **fixed 2026-03-31** (removed)
- REF-23: 2 f-strings without placeholders (F541) — **fixed 2026-03-31** (auto-fix)
- REF-24..28: Import sorting, PEP604 annotations, deprecated imports, redundant open modes — **fixed 2026-03-31** (auto-fix)
- REF-29: Unregistered `slow` pytest mark — **fixed 2026-03-31** (added to pyproject.toml)
- REF-30: pyproject.toml description "29-signal" → "30-signal" — **fixed 2026-03-31**
- REF-31: 78 unused imports + 65 unsorted imports in test files — **fixed 2026-03-31** (auto-fix)
- ARCH-30: 14 `try/except/pass` → `contextlib.suppress` conversions — **done 2026-03-31**
- BUG-160 (P1): 3 signals missing from SIGNAL_NAMES — **fixed 2026-04-01** (added crypto_macro, orderbook_flow, metals_cross_asset)
- BUG-161 (P2): metals_loop.py raw JSONL appends — **fixed 2026-04-01** (atomic_append_jsonl)
- BUG-162 (P3): `metals_loop.py` is 4,553-line monolith — hardest module to maintain, highest bug density (deferred)
- BUG-163 (P3): `exit_optimizer.py` antithetic variate breaks on odd n_paths (deferred)
- BUG-164 (P3): `orb_predictor.py` hardcodes UTC hours 8-10 for morning range (deferred)
- BUG-165 (P1): llama_server.py model swap race condition — **fixed 2026-04-02** (locks held during entire query operation)
- BUG-166 (P2): `shared_state._cached()` thundering herd on TTL expiry — **fixed 2026-04-02** (dogpile prevention via per-key loading flag)
- BUG-167 (P2): Dead `_CORE_SIGNAL_SET` in signal_engine.py — **fixed 2026-04-02** (removed)
- REF-33..37: 18 unused imports, 5 SIM105, 3 F541, 1 F841 in portfolio/ — **fixed 2026-04-02** (ruff auto-fix)
- BUG-168 (P3): `llama_server.py` `_ensure_model()` missing `global _local_model` — **fixed 2026-04-04** (removed dead assignment)
- BUG-169 (P3): `indicators.py` `_regime_cache` accessed without lock from 8 concurrent threads — **fixed 2026-04-04** (added `_regime_lock`)
- BUG-170 (P3): `fear_greed.py` streak file non-atomic write — **fixed 2026-04-04** (atomic_write_json + load_json)
- REF-39: I001/UP015/UP017 auto-fixes — **fixed 2026-04-04** (ruff auto-fix)
- REF-40: Unused loop variables in `llm_batch.py` — **fixed 2026-04-04** (prefixed with `_`)
- REF-41: Ambiguous variable `l` in 3 modules — **fixed 2026-04-04** (renamed to `line`/`lo`)
- REF-42: Duplicate isinstance in `calendar_seasonal.py` — **fixed 2026-04-04** (merged)
- REF-43: Lambda assignments in `avanza_control.py`, `avanza_session.py` — **fixed 2026-04-04** (converted to def)
- REF-44: `except Exception: pass` in `main.py` — **fixed 2026-04-04** (contextlib.suppress)
- ~6,036 tests across 212+ test files (as of 2026-04-04), 79 pre-existing failures
- BUG-171 (P2): ~14 remaining `except Exception: pass` silent swallowers — **fixed 2026-04-07** (contextlib.suppress + logger.debug)
- BUG-172 (P3): `fin_fish.py` datetime.timezone.utc → datetime.UTC — **fixed 2026-04-07** (UP017)
- BUG-173 (P3): `orchestrator.py` typing.Callable → collections.abc.Callable — **fixed 2026-04-07** (UP035)
- BUG-174 (P3): Unused Path import in golddigger_strategy.py — **fixed 2026-04-07** (F401)
- REF-45: 9 collapsible nested `if` statements (SIM102) — **fixed 2026-04-07**
- REF-46: 3 if-with-same-arms (SIM114) — **fixed 2026-04-07** (combined with `or`)
- REF-47: 2 suppressible-exception (SIM105) in bot runners — **fixed 2026-04-07** (contextlib.suppress)
- REF-48: 91 ruff auto-fixes in test files (F401, I001, UP017, SIM300) — **fixed 2026-04-07**
- REF-49: SIM103 needless bool in daily_digest.py — **fixed 2026-04-07**
- Portfolio ruff violations: 74 → 56 (remaining: 51 E402 intentional + 5 SIM115 intentional)
- Test ruff violations: 253 → 170 (remaining: F841 test vars, SIM117 style, E741 test naming)
- ~6,138 tests, 85 pre-existing failures (as of 2026-04-07)
