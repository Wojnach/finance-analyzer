# System Overview

Updated: 2026-06-01
Branch: improve/auto-session-2026-06-01

## 1) Architecture Summary

Two-layer autonomous trading system with 89 tracked signals (15 active, 76 disabled; 79 files in `portfolio/signals/`, 70 registered), 5 Tier-1 instruments, and dual-strategy portfolio management. (Counts reconciled 2026-06-11.)

- **Layer 1** (`portfolio/main.py`): Continuous 600s loop (bumped from 60s 2026-04-09) — data collection, signal generation, trigger detection, summary writing.
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

## 3) Module Map (167 top-level `portfolio/*.py`, 300 incl. subpackages; 2026-06-11)

### Orchestration (5 modules)
- `main.py` (1532 lines): Loop lifecycle, crash backoff (10s→5min), health heartbeat, parallel ticker processing via ThreadPoolExecutor(8), post-cycle housekeeping (15+ tasks), heartbeat keepalive for Layer 2
- `agent_invocation.py` (1724 lines): Layer 2 subprocess lifecycle, tiered prompts (T1/T2/T3), timeout killing, completion watchdog (30s daemon), stack overflow auto-disable, drawdown circuit breaker, trade guards gate, multi-agent mode. Spawn-vs-watchdog race fixed (2026-05-12): metadata set before Popen.
- `trigger.py` (651 lines): 5-section change detection — consensus flip (with clock-skew guard), sustained flip (OR-debounce), price >2%, F&G threshold, sentiment reversal, post-trade. Tier classification (T1/T2/T3), density gate, flip cooldown (30 min), ranging dampening, confidence/ATR floors via claude_budget config.
- `market_timing.py` (342 lines): DST-aware US/EU market hours, NYSE + Swedish holiday calendars (Easter-based), agent invocation window, GPU signal gating
- `config_validator.py` (87 lines): Startup config validation

### Signal System (89 tracked names: 15 active + 76 disabled; 79 files, 70 registered)
- `signal_engine.py`: weighted-consensus voting (15 active), accuracy gating, 8-stage confidence penalties, correlation groups, horizon-aware regime gating, dynamic horizon weights, thread-safe sentiment + content-keyed ADX cache, dead-zone soft votes, per-phase timing diagnostics
- `signal_registry.py` (399 lines): Plugin-based signal discovery via importlib, lazy loading. 70 enhanced signals via `register_enhanced()`. 5-min import-failure cooldown. Shadow enrollment for LLM models.
- `signal_utils.py` (132 lines): Shared helpers — SMA, EMA, RSI, majority_vote
- `signals/*.py` (79 files): Enhanced composite signals, each with 4-8 sub-indicators
- `accuracy_stats.py` (2070 lines): Per-signal hit rate tracking, accuracy cache, activation rates, thundering-herd lock, degradation detection
- `outcome_tracker.py` (391 lines): Signal snapshot logging, price backfill for accuracy

### Data Collection (3 modules)
- `data_collector.py` (344 lines): Binance spot/FAPI, Alpaca, yfinance; circuit breakers; 7 timeframes; 60s per-frame timeout
- `indicators.py` (253 lines): RSI, MACD, EMA, BB, ATR, regime detection (cache per cycle)
- `shared_state.py` (388 lines): Thread-safe cache (TTL + stale fallback + dogpile prevention), rate limiters, NewsAPI quota, LLM batch rotation

### Portfolio & Risk (8 modules)
- `portfolio_mgr.py` (180 lines): State load/save with rolling backups (C7), per-file locks (C8), atomic read-modify-write, corruption recovery
- `trade_guards.py` (406 lines): Per-ticker cooldown, consecutive-loss escalation (0→8x), position rate limit, time-decay
- `risk_management.py` (988 lines): Drawdown circuit breaker, peak value tracking (streaming JSONL with byte-offset cache), ATR stops, concentration risk, FX fallback chain
- `equity_curve.py` (600 lines): FIFO round-trip matching, Sharpe/Sortino, max drawdown, calmar ratio
- `monte_carlo.py` (422 lines): GBM with antithetic variates, probability-driven drift
- `correlation_priors.py` (31 lines): Single source of truth for asset correlation strengths (BTC↔ETH: 0.75, XAG↔XAU: 0.85)
- `monte_carlo_risk.py` (492 lines): Student-t copula VaR/CVaR, imports correlation priors
- `kelly_sizing.py` (389 lines): Kelly criterion position sizing

### Reporting & Analysis (6 modules)
- `reporting.py` (1330 lines): agent_summary.json (full/compact/tiered), three-tier compaction, thread-safe module failure tracking with escalation
- `journal.py` (583 lines): Layer 2 journal JSONL streaming, context markdown generation
- `journal_index.py` (399 lines): BM25 relevance ranking, importance scoring
- `reflection.py` (241 lines): Periodic strategy metrics (win rate, avg PnL)
- `prophecy.py` (392 lines): Macro belief system (silver_bull_2026, etc.)
- `focus_analysis.py` (236 lines): Mode B probability format for focus instruments

### External Data (11 modules)
- `fear_greed.py`, `sentiment.py`, `social_sentiment.py`, `onchain_data.py`
- `funding_rate.py`, `alpha_vantage.py`, `fx_rates.py`, `futures_data.py`
- `ministral_signal.py`, `ministral_trader.py`, `ml_signal.py` (disabled)

### Notification (5 modules)
- `telegram_notifications.py` (142 lines): Send with Markdown escaping, 4096 char limit, fallback
- `telegram_poller.py`: Incoming message polling, command handling
- `message_store.py`: Transaction/notification logging
- `message_throttle.py`: Analysis message rate limiting
- `digest.py` (271 lines): 4-hour periodic digest with invocation stats

### Infrastructure (10 modules)
- `file_utils.py` (423 lines): atomic_write_json, load_json/jsonl, prune_jsonl, atomic_append_jsonl, load_jsonl_tail, last_jsonl_entry, sidecar locking (msvcrt/fcntl)
- `http_retry.py` (99 lines): Exponential backoff (3 retries, 1s base, 2x factor, full-delay jitter), secret redaction
- `circuit_breaker.py` (134 lines): Thread-safe state machine (CLOSED→OPEN→HALF_OPEN)
- `health.py` (452 lines): Heartbeat, error ring buffer, module failure tracking, signal health, dead signal detection, outcome staleness
- `logging_config.py` (47 lines): RotatingFileHandler (10MB, 3 backups)
- `signal_db.py` (405 lines): WAL-mode SQLite dual-write with JSONL fallback. `load_entries()` optimized from O(n²) per-snapshot SELECTs to 3 bulk queries + dict reassembly (2026-05-12).
- `process_lock.py` (107 lines): Cross-platform non-blocking file locks (msvcrt/fcntl)
- `subprocess_utils.py` (337 lines): Windows Job Object subprocess protection, orphan reaper. WMIC→PowerShell migration (2026-05-12): `kill_orphaned_by_cmdline()` uses Get-CimInstance for Win11 compat.
- `notification_text.py` (64 lines): Shared text helpers for human-readable notifications
- `llama_server.py` (658 lines): Unified persistent llama-server manager, cross-process model swap with file lock, query-scoped locking (BUG-165)

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

### Signal Inventory (89 tracked: 15 active, 76 disabled; reconciled 2026-06-11)
- **Active (15)**: RSI, BB, Fear&Greed, Ministral-8B, Qwen3-8B, Momentum, Mean Reversion, News Event, Econ Calendar, Crypto Macro, COT Positioning, On-Chain BTC, Statistical Jump Regime, Drift Regime Gate, Amihud Illiquidity Regime
- **Per-ticker overrides** (globally disabled, active on one ticker via `_DISABLED_SIGNAL_OVERRIDES`): Realized Skewness (XAU), ML Classifier (ETH). (Williams VIX Fix override removed 2026-05-31; Credit Spread Risk override removed 2026-05-26.)
- Recently disabled (collapsed accuracy): Metals Cross-Asset (2026-06-06), Crypto EVRP (2026-05-26), ADX Regime Switch (2026-06-01), Choppiness Regime Gate, BOCPD Regime Switch, Vol Ratio Regime.
- **Disabled (54)**: ML Classifier, MACD, EMA, Volume, Funding Rate, Sentiment, Forecast, Claude Fundamental, Fibonacci, Trend, Volume Flow, Volatility, Candlestick, Structure, Heikin-Ashi, Calendar, Macro Regime, Smart Money, Oscillators, Orderbook Flow, Futures Flow, DXY Cross-Asset, Momentum Factors, BTC Proxy, Credit Spread Risk, Crypto EVRP, plus 28 pending-validation signals added Apr-May 2026 (Futures Basis, Hurst Regime, Shannon Entropy, VIX Term Structure, Kalman Trend Momentum, etc.)

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

- ~10,300+ tests across 418 test files
- Sequential: ~16 min; Parallel (`-n auto`): ~2 min (8x speedup on 8 workers)
- 26 pre-existing failures (13 Freqtrade integration + 13 xdist-flaky/state-coupled)
- Config: `pyproject.toml` → `[tool.pytest.ini_options]`
- Linter: ruff (line-length=120, target py311)
- Fixtures: `conftest.py` provides `make_indicators()`, `make_candles()`, `make_ohlcv_df()`, `sample_config`, `config_file`, `tmp_data_dir`
- 76 tests use `tmp_path` isolation for parallel-safe file I/O
- 6 untested utility modules: telegram_poller, data_refresh, backup, social_sentiment, stats, regime_alerts
- log_rotation: 36 tests (age archival, size pruning, text rotation, unmanaged file detection)

## 8) Key Design Patterns

- **Atomic writes**: `file_utils.atomic_write_json()` and `atomic_write_jsonl()` prevent corrupt state files. All portfolio modules (including golddigger/elongir subsystems) use `load_json()`/`load_jsonl()` — zero raw `json.loads(path.read_text())` calls remain (enforced by `test_io_safety_sweep.py`)
- **Metals shared state**: positions, stop orders, trade queue, and guard/spike state now use atomic JSON writes with explicit corrupt-file logging
- **Circuit breakers**: Per-API failure tracking with auto-recovery
- **Cache-through**: TTL cache with stale-data fallback (shared_state._cached)
- **Tiered invocation**: T1 (quick, 70%), T2 (signal, 25%), T3 (full, 5%)
- **Three-tier compaction**: Held → full votes; triggered → vote_detail string; HOLD → minimal
- **Crash protection**: Exponential backoff (10s→5min), alert suppression after 5 crashes
- **Graceful degradation**: Each signal/module wrapped in try/except, module warnings surfaced

## 9) Known Issues (as of 2026-06-01)

**263+ bugs fixed** across 80+ sessions (BUG-15 through BUG-263).
Full history: [docs/RESOLVED_BUGS.md](RESOLVED_BUGS.md).

### Open Issues

- ARCH-17: main.py re-exports 100+ symbols (obscures module boundaries)
- ARCH-18/BUG-162: metals_loop.py is 7,880-line monolith — highest bug density, hardest to maintain
- ARCH-19: No CI/CD pipeline — all testing is manual
- ARCH-20: No type checking (mypy)
- BUG-132: orb_predictor.py fetches 5000+ candles uncached
- BUG-149: meta_learner orphaned (predict() never called from production)
- TEST-1: GPU gate (`gpu_gate.py`) has zero test coverage
- TEST-3: 26+ pre-existing test failures (integration, config, state isolation)
- P0-B (unfixed): grid_fisher reconciles against ALL Avanza accounts (no account_id filter)
- P1: 3d/5d/10d horizons collapse to 1d accuracy in signal gating (TODO at signal_engine:4175)
- P1: claude_gate._count_today_invocations full JSONL scan on every call (perf degradation)

### 2026-06-01 Fixes (auto-session)

8 bugs fixed (5 P0 + 3 P1):
- **P0-B1**: outcome_tracker._fetch_historical_price 1h→1m interval + open price (eliminates 59-min forward shift biasing all short-horizon accuracy)
- **P0-B2**: agent_invocation auth_error journal stub (auth outages now leave journal record like failed/incomplete)
- **P0-B3**: signal_engine cross-ticker consensus cache keyed by (ticker, horizon) not ticker alone (prevents MSTR btc_proxy horizon mismatch)
- **P0-B4**: fx_rates sanity-check explicit early return (no more silent stale fallthrough)
- **P0-B5**: metals_loop SILVER_ALERT_LEVELS TypeError (float subscript crash on active silver positions)
- **P1-B6**: risk_management._CORRELATED_PAIRS sentinel + retry (transient import failure no longer permanently disables correlation risk)
- **P1-B7**: dashboard signal heatmap now dynamic from SIGNAL_NAMES (was hardcoded stale 30-signal list)
- **P1-B8**: check_critical_errors auto-resolve for stale categories (stops fix-agent budget burn on 31 phantom entries)

### 2026-05-30 Fixes

11 bugs fixed from FGL adversarial review (2026-05-29):
- **Tier 0**: autonomous.py failure journal stub + loop_contract autonomous status handling — stops 22+ false CRITICALs/week
- **P0-1**: warrant_portfolio over-sell validation (refuse SELL of non-existent, clamp over-sell)
- **P0-2**: avanza_orders orderId="?" placeholder rejection (mark error, alert via Telegram)
- **P0-3**: avanza_session orderbook_id validation (non-empty + numeric before POST)
- **P0-4**: multi_agent_layer2 tree-kill + invocation journaling (claude_gate protections)
- **Theme B**: choppiness_regime_gate tie-breaker removal + engine-level REGIME_GATE_ONLY_SIGNALS mechanism
- **Theme F**: loop_health status="ok" hardcode → now uses `ok` parameter
- **Theme A**: http_retry fatal-vs-transient typing (401/403/404 not retried)

### Findings from 2026-05-04 Auto Session

- **B1 (fixed)**: Equity curve annualized with 252 days (stock convention) but portfolio runs 24/7 → crypto volatility understated 17%. Changed to 365.
- **B2 (fixed)**: Contract violation dedup wrote critical_errors.jsonl BEFORE dedup marker → duplicate entries on marker write failure. Swapped order.
- **B3 (fixed)**: Monte Carlo ATR fallback was generic 2.0% for all assets. Now per-asset-class: crypto=3.5%, metals=4.0%, stocks=2.0%.
- **B4 (fixed)**: Stuck loading key eviction in shared_state.py logged at DEBUG. Elevated to WARNING.
- Test suite grew to ~7,730+ tests across 430 files by 2026-05-21.

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

### Findings from 2026-05-19 Auto Session

- 7 P1 fixes implemented and merged (ConnorsRSI signal, ADX dual-regime meta-signal, trigger density gate with dampening, EXIT_LOCK_CONFLICT propagation in golddigger, MSTR loop singleton + auto-restart, and more).
- Signal count: 65→67 modules, 17→17 active (two added, two disabled).

### Findings from 2026-05-20 Auto Session

- **B0 (CRITICAL, fixed)**: `agent_invocation.py` — `_journal_count_before` and `_telegram_count_before` were local vars in `invoke_agent()` but referenced as globals in `_check_agent_completion_locked()`. Every agent completion since commit 28af2f73 (May 17) silently raised NameError, swallowed by watchdog `except Exception`. Completion detection, invocation logging, and new-trade detection all broken for 3 days.
- **B1 (fixed)**: Consensus test flakes — production accuracy cache files leaked into tests. `_null_cached` mock didn't intercept `accuracy_stats.get_or_compute_accuracy()` called directly inside `generate_signal()`. Fixed by redirecting all 4 accuracy cache file paths to session-scoped tmp dirs.
- **B2 (fixed)**: `http_retry.py` jitter was 10% of delay — marginal for thundering herd. Bumped to full-delay uniform distribution. Also fixed 429 path that replaced jittered wait with raw `retry_after`.
- **B4 (fixed)**: `shared_state.py` newsapi TTL used hardcoded UTC offsets (07:00-21:00 UTC). During CEST (summer), 08:00 CET = 06:00 UTC, shifting the active window by 1h. Replaced with timezone-aware `Europe/Stockholm` check.
- Signal count: 67→70 modules, 17→18 active, 50→52 disabled.

### Findings from 2026-05-21 Auto Session

- **BUG-A (fixed)**: `avanza_orders.py` — `place_buy_order()`/`place_sell_order()` can return `None` on Playwright errors. Line 367 called `.get()` on the result without None guard. Added explicit check with clear "API returned no response" diagnostic.
- **BUG-B (fixed)**: `dashboard/app.py` — `/api/mstr_loop` endpoint read JSONL files via raw `open()` line-by-line iteration. Replaced with `last_jsonl_entry()` (4KB tail seek, O(1) instead of O(n)).
- **DOC-A (fixed)**: SYSTEM_OVERVIEW.md line counts for 15+ modules were wrong by 30-200% (e.g., agent_invocation.py listed 489, actual 1644). Full accuracy pass on all section 3 numbers.
- Signal count: 70 modules (7 core + 63 enhanced), 18 active, 52 disabled.

### Findings from 2026-05-25 Auto Session

- **BUG-A (fixed)**: `portfolio/tickers.py` — 5 shadow signals (connors_rsi2, adx_regime_switch, choppiness_regime_gate, bocpd_regime_switch, gold_overnight_bias) were in DISABLED_SIGNALS but missing from SIGNAL_NAMES. outcome_tracker silently dropped their votes, preventing accuracy accumulation. Also added 3 LLM rotation signals (cryptotrader_lm, finance_llama, meta_trader) to both lists.
- **BUG-C (fixed)**: `dashboard/app.py` — Missing security headers (X-Frame-Options, X-Content-Type-Options, HSTS, CSP frame-ancestors). Added to after_request handler.
- **BUG-B (fixed)**: `docs/SYSTEM_HEALTH_CONTRACT.md` — Referenced 20 instruments (only 5 Tier-1 remain).
- **CLEANUP-A (fixed)**: `signal_registry.py` — Dead `register_signal()` decorator removed (never used, all 66 modules use `register_enhanced()`).
- Signal count: 66 modules (66 enhanced), 16 active, 50 disabled.
