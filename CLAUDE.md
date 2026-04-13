# Portfolio Intelligence — Finance Analyzer

## ⚠ STARTUP CHECK — READ FIRST

Before doing anything else in this session:

```
.venv/Scripts/python.exe scripts/check_critical_errors.py
```

If the script exits non-zero, it has found unresolved critical errors in
`data/critical_errors.jsonl` from the last 7 days. These represent
failure modes the loop can't recover from on its own (e.g. the ~3-week
silent Layer 2 auth outage of March–April 2026 that went undetected
because `claude -p --bare` exited 0 while printing "Not logged in").

When unresolved errors exist:

1. Surface the list to the user verbatim before continuing.
2. Ask whether to address them first, or proceed with the user's request.
3. Do not silently ignore them.

To resolve an entry, append a follow-up line to `data/critical_errors.jsonl`:

```json
{"ts":"<now>","level":"info","category":"resolution","caller":"<same>","resolution":"<what you did>","resolves_ts":"<original ts>","message":"<short>","context":{}}
```

The journal is append-only. Never mutate earlier entries.

### Auto-spawn fix agent

The `PF-FixAgentDispatcher` scheduled task (every 10 min) spawns a Claude
fix agent when unresolved critical entries exist, with per-category
cooldown + exponential backoff (30m → 2h → 12h → effectively disabled).
Tool allow-list: `Read,Edit,Bash` — the agent never commits or pushes.

If you see repeated `fix_agent_failed` entries in the journal, the
dispatcher has given up on that category — manual investigation is
required. If you need to stop auto-spawn entirely (e.g. during an
outage or experimentation):

```
touch data/fix_agent.disabled
```

Remove the file to re-enable. See
`docs/plans/2026-04-13-auto-spawn-fix-agent.md` for the full design.

## Overview

Autonomous two-layer trading system. Layer 1 (Python, 60s loop) collects market data, computes
32 active signals (36 modules registered, 4 force-HOLD) across 7 timeframes for 5 Tier-1
instruments, and detects meaningful triggers. Layer 2 (Claude CLI subprocess) is invoked on
triggers to make trade decisions for two simulated portfolios (Patient & Bold, each starting
500K SEK). A separate metals subsystem trades Avanza warrants independently.

The system tracks crypto (BTC, ETH), metals (XAU, XAG), and MSTR via Binance, Alpaca,
and Avanza. All decisions are logged to journals, accuracy is tracked, and notifications go to
Telegram. A Flask dashboard serves real-time data on port 5055.

## Architecture

### Layer 1: Data Loop (`portfolio/main.py`)
- 60s cycle: fetch OHLCV → compute indicators → run 32 active signals → detect triggers → write summaries
- Parallel ticker processing (ThreadPoolExecutor, 8 workers)
- Crash recovery: exponential backoff (10s→5min), Telegram alerts (first 5 only)
- Entry: `.venv/Scripts/python.exe -u portfolio/main.py --loop` (via `scripts/win/pf-loop.bat`)

### Layer 2: Decision Engine (`portfolio/agent_invocation.py`)
- Claude CLI subprocess (`claude -p "..."`) invoked by Layer 1 on trigger events
- Tiered: T1 Quick (120s/15 turns), T2 Signal (600s/40 turns), T3 Full (900s/40 turns)
- Reads signal summaries → makes trade decisions → writes journal → sends Telegram
- Full trading playbook: **`docs/TRADING_PLAYBOOK.md`**

### Layer 3: Autonomous Fallback (`portfolio/autonomous.py`)
- Replaces Layer 2 when `config.layer2.enabled = false`
- Signal-based decision rules without LLM. Recommendations only, no execution.

### Metals Subsystem (`data/metals_loop.py`)
- Separate process for XAG/XAU warrant trading via Avanza API
- 60s cycle with embedded 10s silver fast-tick monitor
- Local LLM inference (Ministral-8B, Chronos-2, Qwen3-8B)
- Entry: `scripts/win/metals-loop.bat`

### Dashboard (`dashboard/app.py`)
- Flask REST API on port 5055 (optional token auth)
- Key endpoints: `/api/portfolio`, `/api/summary`, `/api/accuracy`, `/api/trades`,
  `/api/decisions`, `/api/health`, `/api/metals`, `/api/forecast`, `/api/prophecy`

### Trading Bots
- **GoldDigger** (`portfolio/golddigger/`): Gold certificate trading (dry-run/live via Avanza)
- **Elongir** (`portfolio/elongir/`): Equity trading bot (separate signal system)

## Signal System (36 Modules · 32 Active)

### Core Active (10)
1. RSI(14) — Oversold <30 BUY, overbought >70 SELL
2. MACD(12,26,9) — Histogram crossover
3. EMA(9,21) — Trend following, 0.5% deadband
4. BB(20,2) — Bollinger Band breakout
5. Fear & Greed — Contrarian (≤20 BUY, ≥80 SELL)
6. Sentiment — CryptoBERT (crypto) / Trading-Hero-LLM (stocks), keyword-weighted
7. Ministral-8B — Local LLM reasoning via llama-cpp-python
8. Qwen3-8B — Local LLM reasoning, 61.8% at 3h (added 2026-03-29, GPU-gated)
9. Volume Confirmation — Spike >1.5x avg confirms direction
10. Funding Rate — 74.2% at 3h (535 sam), horizon-gated (active 3h/4h only, gated at 1d). Crypto-only.

### Core Disabled (1)
11. ML Classifier (28.2%) — worse than coin flip

### Core Active (BTC-only) (1)
12. On-Chain BTC — MVRV Z-Score, SOPR, NUPL, Exchange Netflow. Sub-metric majority vote. BTC-only, 12h cache.

### Enhanced Active (21 modules in `portfolio/signals/`)
13. Trend — Golden/Death Cross, Supertrend, Ichimoku, ADX
14. Momentum — Stochastic, StochRSI, CCI, Williams %R, ROC, PPO
15. Volume Flow — OBV, VWAP, A/D, CMF, MFI
16. Volatility — BB Squeeze, ATR Expansion, Keltner, Donchian
17. Candlestick — Hammer, Engulfing, Doji, Morning/Evening Star
18. Structure — High/Low Breakout, Donchian 55, RSI/MACD centerline
19. Fibonacci — Retracement, Golden Pocket, Extensions, Pivots
20. Smart Money — BOS, CHoCH, FVG, Liquidity Sweeps, Supply/Demand
21. Oscillators — Awesome, Aroon, Vortex, KST, Schaff, TRIX, Coppock
22. Heikin-Ashi — HA Trend, Hull MA, Alligator, Elder Impulse, TTM Squeeze
23. Mean Reversion — RSI(2/3), IBS, Gap Fade, BB %B
24. Calendar — Day-of-Week, Turnaround Tue, FOMC Drift, Month-End
25. Macro Regime — 200-SMA, DXY vs Risk, Yield Curve, FOMC proximity
26. Momentum Factors — Time-Series Mom, ROC-20, 52W High/Low
27. News Event — Headline velocity, keyword severity, source credibility
28. Econ Calendar — FOMC/CPI/NFP proximity risk-off (hardcoded 2026-2027)
29. Forecast — Kronos + Chronos time-series foundation models
30. Claude Fundamental — Haiku/Sonnet/Opus cascade (quality, valuation, catalysts)
31. Futures Flow — Binance FAPI (crypto only): OI, LS Ratio, Funding Trend
32. Orderbook Flow — Depth imbalance, trade flow, VPIN, OFI, spread health (metals+crypto)
33. Metals Cross-Asset — Copper, GVZ, Gold/Silver ratio velocity, SPY, Oil (metals only)

### Enhanced Disabled (3)
34. Crypto Macro — Options max pain, gold-BTC rotation, exchange reserves (crypto only) — registered but force-HOLD via DISABLED_SIGNALS
35. COT Positioning — CFTC speculative/commercial positioning, contrarian (metals only) — registered but force-HOLD pending live validation
36. Credit Spread Risk — Credit spread monitoring — registered but force-HOLD pending live validation

### Signal Mechanics
- **MIN_VOTERS = 3** (all asset classes). Consensus = active voters (BUY+SELL), not total.
- **Accuracy gate**: signals below 45% accuracy (30+ samples) are force-HOLD (not inverted — inversion causes whiplash)
- **Recency-weighted**: 70% recent (7d) + 30% all-time
- **Regime penalties**: ranging 0.75x, high-vol 0.80x confidence multipliers
- **Volume/ADX gates**: RVOL <0.5 forces HOLD
- **Applicable signals**: crypto=31, stocks=26, metals=28

## Instruments

### Tier 1: Full signals (32 active × 7 timeframes)
| Asset Class | Tickers | Source |
|-------------|---------|--------|
| Crypto 24/7 | BTC-USD, ETH-USD | Binance spot |
| Metals 24/7 | XAU-USD, XAG-USD | Binance FAPI |
| US Stocks | MSTR | Alpaca |

(Removed Mar 15: AMD, GOOGL, AMZN, AAPL, AVGO, META, SOUN, LMT)
(Removed Apr 09: PLTR, NVDA, MU, SMCI, TSM, TTWO, VRT — cycle p50 reduction)

### Tier 2: Avanza price-only (no signals)
SAAB-B, SEB-C, INVE-B

### Tier 3: Warrants (Avanza price + underlying's signals)
XBT-TRACKER (→BTC), ETH-TRACKER (→ETH), MINI-SILVER (→XAG 5x)

## Key Modules

### Orchestration
`main.py` (loop lifecycle), `agent_invocation.py` (Layer 2 subprocess),
`trigger.py` (change detection), `market_timing.py` (DST-aware hours)

### Signal Pipeline
`signal_engine.py` (36-signal voting, 32 active), `signal_registry.py` (plugin discovery),
`signals/*.py` (24 enhanced modules), `accuracy_stats.py` (hit rates),
`outcome_tracker.py` (backfill), `forecast_accuracy.py` (model health)

### Data & External
`data_collector.py` (Binance/Alpaca/yfinance), `fear_greed.py`, `sentiment.py`,
`alpha_vantage.py` (fundamentals), `futures_data.py` (Binance FAPI),
`onchain_data.py` (BTC MVRV/SOPR), `fx_rates.py` (USD/SEK)

### Microstructure & Cross-Asset
`metals_orderbook.py` (Binance FAPI depth+trades), `microstructure.py` (OFI/VPIN/depth imbalance),
`microstructure_state.py` (snapshot accumulator, persisted rolling OFI),
`metals_cross_assets.py` (copper/GVZ/SPY/G-S ratio via yfinance)

### Portfolio & Risk
`portfolio_mgr.py` (atomic state I/O), `trade_guards.py` (cooldowns/escalation),
`risk_management.py` (drawdown circuit breaker, ATR stops, concentration),
`equity_curve.py` (Sharpe/Sortino, round-trip P&L),
`monte_carlo.py` + `monte_carlo_risk.py` (GBM simulation, t-copula VaR/CVaR)

### Metals & Avanza
`avanza_session.py` (Playwright BankID auth), `avanza_orders.py` (order flow),
`exit_optimizer.py` (probabilistic exit), `price_targets.py` (structural levels),
`orb_predictor.py` (Opening Range Breakout), `iskbets.py` (intraday quick-gamble),
`fin_snipe.py` (metals bid/exit ladder)

### Reporting & Notification
`reporting.py` (agent_summary generation), `journal.py` (decision JSONL),
`prophecy.py` (macro beliefs), `telegram_notifications.py` (sending),
`message_store.py` (logging + delivery), `digest.py` (4h periodic),
`daily_digest.py` (morning), `telegram_poller.py` (incoming /mode commands)

### Infrastructure
`file_utils.py` (atomic JSON/JSONL I/O), `http_retry.py` (backoff),
`health.py` (heartbeat, module failures), `claude_gate.py` (CLI gate),
`gpu_gate.py` (GPU lock), `shared_state.py` (thread-safe cache, rate limiters)

Full module map (142 modules): `docs/SYSTEM_OVERVIEW.md`

## Key Data Files

| File | Purpose |
|------|---------|
| `data/agent_summary.json` | Full signal report (all tickers, ~64K tokens) |
| `data/agent_summary_compact.json` | Tiered compaction for Layer 2 (~1400 lines) |
| `data/agent_context_t1.json` | Tier 1 quick-check context (~200 lines) |
| `data/agent_context_t2.json` | Tier 2 signal context (~600 lines) |
| `data/portfolio_state.json` | Patient strategy: cash, holdings, transactions |
| `data/portfolio_state_bold.json` | Bold strategy: cash, holdings, transactions |
| `data/portfolio_state_warrants.json` | Warrant holdings with leverage |
| `data/layer2_journal.jsonl` | Layer 2 decision log |
| `data/signal_log.jsonl` | Every signal snapshot (+ `signal_log.db` SQLite) |
| `data/prophecy.json` | Macro beliefs (silver_bull, btc_range, eth_follows_btc) |
| `data/trigger_state.json` | Trigger detection baseline |
| `data/health_state.json` | System health (heartbeat, errors, module failures) |
| `data/telegram_messages.jsonl` | All sent Telegram messages |
| `data/fundamentals_cache.json` | Alpha Vantage stock data (daily refresh) |
| `data/accuracy_cache.json` | Signal accuracy (1d/3d/5d/10d horizons) |

## CLI Commands

```bash
# Main loop (production)
.venv/Scripts/python.exe -u portfolio/main.py --loop

# One-shot signal report
.venv/Scripts/python.exe -u portfolio/main.py --report

# Signal accuracy report
.venv/Scripts/python.exe -u portfolio/main.py --accuracy

# Backfill price outcomes
.venv/Scripts/python.exe -u portfolio/main.py --check-outcomes

# Forecast model health
.venv/Scripts/python.exe -u portfolio/main.py --forecast-accuracy

# Prophecy belief review
.venv/Scripts/python.exe -u portfolio/main.py --prophecy-review

# GoldDigger bot
.venv/Scripts/python.exe -m portfolio.golddigger [--live|--dry-run]

# Dashboard (port 5055)
.venv/Scripts/python.exe dashboard/app.py
```

## Testing

```bash
# All tests (~5,994 tests across 242 files, ~16 min sequential)
.venv/Scripts/python.exe -m pytest tests/

# Parallel (~5.5 min, 8 workers)
.venv/Scripts/python.exe -m pytest tests/ -n auto

# Specific file
.venv/Scripts/python.exe -m pytest tests/test_signal_engine.py -v
```

Tests using module-level file paths must patch to `tmp_path` for xdist safety.
26 pre-existing failures (integration, config, state isolation). See `docs/TESTING.md`.

## Environment

- **OS**: Windows 11 Pro. Shell is Git Bash (set via `CLAUDE_CODE_GIT_BASH_PATH`).
- **Python**: `.venv/Scripts/python.exe` — always use forward slashes, full path
- **GPU**: RTX 3080 10GB, CUDA 13.1. LLM inference (Ministral-8B, Chronos-2, Qwen3-8B) runs
  in separate venv at `Q:/models/.venv-llm`. GPU lock: `Q:/models/gpu_lock.py`.
- **Config**: Symlink `config.json` → `C:\Users\Herc2\.config\finance-analyzer\config.json`
  (OUTSIDE repo). **NEVER commit config.json** — exposed API keys on Mar 15, 2026.
- **Timezone**: User is CET (UTC+1). Market hours are DST-dependent (see `memory/market_hours.md`).
- **Scheduled Tasks**: PF-DataLoop (main loop, logon + auto-restart), PF-Dashboard (logon),
  PF-OutcomeCheck (daily 18:00), PF-MLRetrain (weekly).

## Critical Rules

1. **NEVER commit config.json.** It's a symlink to external file with API keys.
2. **Search before writing code.** Grep for existing functionality first. Reuse:
   `avanza_session.py`, `avanza_orders.py`, `file_utils.py`, `signal_utils.py`.
3. **Live prices first.** Never base analysis on cached/precomputed data. Hit live APIs.
4. **Atomic I/O only.** Use `file_utils.atomic_write_json()`, `load_json()`,
   `atomic_append_jsonl()`. Never raw `json.loads(open(...).read())`.
5. **Stop-loss API.** Use `/_api/trading/stoploss/new`, NOT regular order API
   (causes instant fill at bad price — Mar 3 incident).
6. **Git workflow.** Always use worktrees for changes, merge into main, commit and push.

## External APIs (all configured as of Mar 11)

Binance (crypto spot+FAPI), Alpaca (US stocks), Telegram (notifications), Alpha Vantage
(fundamentals, 25/day), NewsAPI (headlines, 100/day), FRED (treasury yields), BGeometrics
(on-chain BTC, 15/day), Avanza (manual BankID ~24h), Claude CLI (Max subscription — NOT API keys).

## Available Skills

- `/fin` — Project status report
- `/fin-crypto` — Deep BTC + ETH + MSTR analysis with live data
- `/fin-gold` — Deep XAU-USD analysis with live data
- `/fin-silver` — Deep XAG-USD analysis with live data
- `/fin-oil` — Deep WTI + Brent analysis with live data

## Layer 2 Trading Agent

The Layer 2 automated trading agent follows the playbook in **`docs/TRADING_PLAYBOOK.md`**.
That document contains: dual strategy personalities (Patient & Bold), execution math,
journal format, Telegram notification format, and all decision rules.

Layer 2 sessions automatically read this CLAUDE.md for project context. The playbook
provides the specific operational instructions for trade decisions.

## Key Principles

- **Data-driven, not speculative.** Every decision backed by signals.
- **Two strategies, one analysis.** Patient (conservative) and Bold (aggressive) decisions each invocation.
- **Log everything.** Every trade gets a reason in the transaction record.
- **The user trades real money elsewhere based on your signals.** Be clear about confidence.
- **System reliability is #1.** The loop must run 100% of the time. Features are secondary.
