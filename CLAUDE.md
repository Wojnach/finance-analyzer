# Portfolio Intelligence — Finance Analyzer

## ⚠ STARTUP CHECK — READ FIRST

Before doing anything else in this session:

```
# Steam Deck (Linux):  .venv/bin/python scripts/check_critical_errors.py
# herc2 (Windows):     .venv/Scripts/python.exe scripts/check_critical_errors.py
```

If the script exits non-zero, it has found unresolved critical errors in
`data/critical_errors.jsonl` from the last 7 days. These represent
failure modes the loop can't recover from on its own (e.g. the ~3-week
silent Layer 2 auth outage of March–April 2026 that went undetected
because `claude -p --bare` exited 0 while printing "Not logged in").

**Before the steps below — are you running headless?**
If the `PF_HEADLESS_AGENT` environment variable is set to `"1"`, you are
running as a Layer 2 / bigbet / iskbets / analyze / multi-agent-specialist
subprocess with NO interactive stdin. The interactive protocol below does
not apply. Instead:

1. Log the unresolved critical errors as a one-line summary inside your
   journal entry / trade rationale so they stay surfaced.
2. Proceed with the trigger task — do NOT ask "how would you like to proceed?".
   Any prompt that blocks on user input will hang the subprocess until the
   tier timeout fires with zero work done (exact failure pattern in
   `data/agent.log` on 2026-04-16, commits `877221a` / `7c9cf36e` /
   `08d6ea3b`).
3. If the unresolved errors are directly on the task you were spawned to
   handle (e.g. a trigger that's already failed once), note it in your
   decision but still execute.

Interactive sessions (no `PF_HEADLESS_AGENT` env var set) follow these three
steps when unresolved errors exist:

1. Surface the list to the user verbatim before continuing.
2. Ask whether to address them first, or proceed with the user's request.
3. Do not silently ignore them.

To resolve an entry, append a follow-up line to `data/critical_errors.jsonl`:

```json
{
  "ts": "<now>",
  "level": "info",
  "category": "resolution",
  "caller": "<same>",
  "resolution": "<what you did>",
  "resolves_ts": "<original ts>",
  "message": "<short>",
  "context": {}
}
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

> **Status 2026-06-10:** the audit (`docs/IMPROVEMENT_AUDIT_2026-06-10.md`)
> found the `PF-FixAgentDispatcher` scheduled task is NOT installed in
> production — auto-spawn as described above has never been live on this
> machine. Additionally, `data/fix_agent.disabled` is in place during the
> Claude token freeze (see memory, 2026-06-06). To re-enable: remove the
> flag file AND run `scripts/win/install-fix-agent-task.ps1` as admin.

## ⚠ STARTUP CHECK 2 — read the bottle (pending pickups)

Run this immediately after the critical-errors check:

```
# Steam Deck (Linux):  .venv/bin/python scripts/session_start_bottle.py
# herc2 (Windows):     .venv/Scripts/python.exe scripts/session_start_bottle.py
```

The script reads `data/pending_pickups.json` and prints any scheduled
verification work that is overdue, due today, or due in the next 2
weeks. These are "bottle from the ocean" messages — a prior session
flagged work for a future session (often days later when neither
human nor AI would otherwise remember). Printing is silent when
nothing is pending or recently completed.

If output shows `[OVERDUE]` or `[DUE TODAY]`, surface the entry to the
user verbatim BEFORE doing anything else, then propose either:

- let the cron path run on its own schedule — `PF-PendingPickups`
  daily 08:00 CET runs `scripts/process_pending_pickups.py`
  automatically; OR
- force-run now:
  `.venv/Scripts/python.exe scripts/process_pending_pickups.py --force <ID>`

If output shows recently completed pickups (last 48h), read the
latest history entry in `data/pending_pickups.json` for that pickup
and skim the top of `docs/SESSION_PROGRESS.md` — the verdict and
summary land there. Then decide whether the verdict requires human
action (e.g. a `promote` recommendation needs a human run of
`scripts/review_shadow_signals.py --promote`; a `retire` verdict
needs a `data/shadow_registry.json` status flip).

Adding a new pickup (so a future session picks up some work for you):

1. Append an entry to `data/pending_pickups.json` with `id`, `title`,
   `due_ts`, `handler`, and a `context` block with the decision
   thresholds the handler needs.
2. Add the handler module under `scripts/pickups/<handler>.py`
   exposing `run(pickup, repo_root) -> dict`. For work only a human can do,
   reuse the existing `manual_reminder` handler instead of writing a new one.
3. Whitelist the handler in `scripts/process_pending_pickups.py`
   `_HANDLERS` dict. This whitelist is the CWE-706 guard — never
   dynamic-import handler names from JSON.
   **A handler named in the JSON but absent from `_HANDLERS` fails every
   single daily run forever** and reds `pf-pickups.service` — this is exactly
   what `manual_reminder` did from 2026-07-31 to 2026-08-21.
   `tests/test_pickup_manual_reminder.py` now asserts the two stay in sync.

Handler verdicts and what the dispatcher does with them:

| verdict                   | status after                             | notes                                          |
| ------------------------- | ---------------------------------------- | ---------------------------------------------- |
| `defer`                   | stays `pending`                          | retries next run; use for human-only reminders |
| anything else non-`error` | `completed`                              | one-shot; there is no recurrence mechanism     |
| `error`                   | `pending`, `attempts+1`; `error` after 3 | exits 1, writes `critical_errors.jsonl`        |

4. The dashboard tile at More → Pickups (`/api/pickups`) surfaces
   pending + completed pickups for visual review.

Dispatcher source: `scripts/process_pending_pickups.py`. Cron install
(admin, one-time): `scripts/win/install-pending-pickups-task.ps1`.
Backlog reference: `docs/IMPROVEMENT_BACKLOG.md`.

## Overview

Autonomous two-layer trading system. Layer 1 (Python, 600s loop) collects market data, computes
13 active signals (90 tracked names, 77 disabled, 80 modules in portfolio/signals/) across 7
timeframes for 5 Tier-1 instruments, and detects meaningful triggers. Layer 2 (Claude CLI
subprocess) is invoked on triggers to make trade decisions for two simulated portfolios
(Patient & Bold, each starting
500K SEK). A separate metals subsystem trades Avanza warrants independently.

The system tracks crypto (BTC, ETH), metals (XAU, XAG), and MSTR via Binance, Alpaca,
and Avanza. All decisions are logged to journals, accuracy is tracked, and notifications go to
Telegram. A Flask dashboard serves real-time data on port 5055.

## Architecture

### Layer 1: Data Loop (`portfolio/main.py`)

- 600s cycle (bumped from 60s on 2026-04-09): fetch OHLCV → compute indicators → run 13 active signals → detect triggers → write summaries
- Parallel ticker processing (ThreadPoolExecutor, 8 workers)
- Crash recovery: exponential backoff (10s→5min), Telegram alerts (first 5 only)
- Entry: `.venv/Scripts/python.exe -u portfolio/main.py --loop` (via `scripts/win/pf-loop.bat`)

### Layer 2: Decision Engine (`portfolio/agent_invocation.py`)

- Claude CLI subprocess (`claude -p "..."`) invoked by Layer 1 on trigger events
- Tiered: T1 Quick (180s/15 turns), T2 Signal (600s/40 turns), T3 Full (900s/40 turns)
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
- **Grid market-maker** (`portfolio/grid_fisher.py`, added 2026-05-11):
  Marja Folcke-style limit ladder running inside the metals loop.
  Places 2-tier buy ladders (1200 SEK/leg, 6500 SEK global cap) on the
  with-signal direction of BULL/BEAR certs for XAG, XAU, OIL-USD. Oil
  signal comes from `portfolio/oil_grid_signal.py` (Brent BZ=F RSI+EMA,
  5-min cache). Rotates fills into sell+stop, EOD-flat. Runbook:
  `docs/GRID_FISHER.md`. Endpoint: `/api/grid-fisher`. Probe script:
  `scripts/grid_fisher_probe.py`.

### Dashboard (`dashboard/app.py`)

- Flask REST API on port 5055, dual-stack IPv4+IPv6 bind (token-cookie auth)
- Auth: `?token=<dashboard_token>` once, then 1-year rolling cookie. Bearer
  header for CLI clients. Cloudflare Access header bypasses local auth.
- 50 routes in app.py + 11 in house_blueprint.py (61 total). Last reconciled
  2026-06-11 — re-grep `@app.route` / `@bp.route` if this list looks stale.

  **Health & ops:** `/api/health`, `/api/loop_health`, `/api/lora-status`,
  `/api/market-health`

  **Portfolio & trading:** `/api/portfolio`, `/api/portfolio-bold`,
  `/api/trades`, `/api/decisions`, `/api/invocations`, `/api/risk`,
  `/api/triggers`, `/api/equity-curve`, `/api/warrants`,
  `/api/validate-portfolio` (POST), `/api/grid-fisher`

  **Per-instrument:** `/api/btc`, `/api/eth`, `/api/mstr`,
  `/api/mstr_loop`, `/api/oil`, `/api/metals`, `/api/crypto`,
  `/api/golddigger`

  **Signals & accuracy:** `/api/summary`, `/api/signals`,
  `/api/signal-log`, `/api/signal-heatmap`, `/api/accuracy`,
  `/api/accuracy-history`, `/api/metals-accuracy`

  **Other:** `/api/iskbets`, `/api/local-llm-trends`, `/api/telegrams`

### Multi-asset swing loops (paper-mode by default)

- **Crypto loop** (`data/crypto_loop.py`): BTC + ETH 60s cycle, DRY_RUN=True.
  Install: `scripts/win/install-crypto-loop-task.ps1` → `PF-CryptoLoop`.
- **MSTR loop** (`portfolio/mstr_loop/`): shadow phase. Install:
  `scripts/win/install-mstr-loop-task.ps1` → `PF-MstrLoop`. Phase A
  (live) requires 90 days shadow + manual approval per
  `docs/MSTR_LOOP_NOTES.md`.
- **Oil loop** (`data/oil_loop.py`): WTI 60s cycle, DRY_RUN=True. Install:
  `scripts/win/install-oil-loop-task.ps1` → `PF-OilLoop`. Notes:
  `docs/OIL_LOOP_NOTES.md`. Routes prices via `portfolio.price_source`
  (CL=F/BZ=F → yfinance; oil has no Binance FAPI perp in the symbol map,
  so it falls through to the yfinance last-resort path — reconciled 2026-06-11).

### Swedbank book (`portfolio/swedbank/`, monitoring only)

Tracks three externally-custodied share accounts (26 instruments) that are NOT
held at Avanza. **Monitoring and valuation only — it can never place an order.**
The operator executes every trade by hand; `tests/test_swedbank_no_trading.py`
enforces this by AST-inspecting the package for order imports, order-function
references and any `api_post`/`api_delete`.

- Prices: Avanza is primary for all 26 (`get_quote` is real-time for equities,
  certificates and warrants alike via the `/stock/` path; `/_api/price-chart/
stock/<ob>` serves OHLCV from 1-minute to monthly). Alpaca is the automatic
  fallback for the 19 US names, flagged degraded since it carries no bid/ask.
  The 7 Stockholm instruments have no fallback and degrade to last-good price
  with a visible age stamp.
- Sweeps are **sequential, never concurrent** — the real-money metals loop
  shares this Avanza session. A full 26-instrument sweep is ~1.5s (2.5% duty
  cycle at 60s). This loop never invokes browser recovery.
- Marking: `last` when inside the bid/ask, otherwise bid/ask **mid** with a
  `stale_last` flag. Thin instruments carry hours-stale prints.
- Orderbook IDs are **pinned** in `instruments.py`, never resolved by runtime
  search — search returned the wrong share class for 2 of 19 US names, and the
  ID is both the pricing key and the manual-order deep link.
- Asset class is carried on the instrument, NOT via `tickers.py` sets. Adding
  these names to `STOCK_SYMBOLS` would exhaust the 25/day Alpha Vantage quota
  (`alpha_vantage.py:238`) and stale Tier-1 fundamentals; omitting them makes
  `_compute_applicable_count` read equities as non-stocks. Zero Tier-1 coupling.
- **`data/swedbank_*.json` holds REAL positions and is gitignored — this repo is
  public.** Guarded in `.gitignore` and `.git/info/exclude`. Tests use synthetic
  books only.
- Entry: `.venv/bin/python -m portfolio.swedbank {show,quotes,sync}` ·
  loop `data/swedbank_loop.py --loop` · install
  `scripts/deck/install-swedbank-loop.sh` (unit `pf-swedbank`, not auto-enabled)
- Endpoint: `/api/swedbank` (serves the loop's snapshot; never prices on demand,
  so a page refresh cannot contend for the Avanza session). Tab: More → Swedbank.
- **Signals and trajectories ARE wired** (corrected 2026-08-21 — this line read
  "NOT yet wired" long after the fact). `portfolio/swedbank/signals.py` exposes
  `evaluate()` / `evaluate_universe()` / `log_snapshot()`; the loop calls them
  every `SIGNAL_EVERY_N_CYCLES` and the snapshot carries `signals` +
  `signals_computed_at`. `_UNINFORMATIVE` drops the signal families that cannot
  mean anything per-equity (e.g. COT positioning has no per-equity series).
- **Stockholm-listed names mark at the 17:30 close** and the quote carries a
  `note: "quote is <n>s old"` with `degraded=True` — that flag means _stale_
  here, not _missing bid/ask_. Both XBT crypto trackers are in this group, so
  the crypto sleeve (~22% of book value) is **frozen at Friday's close all
  weekend while BTC/ETH trade 24/7**. Monday's first sweep gaps. Don't read a
  weekend book P&L as live.
- `data/swedbank_signal_log.jsonl` logs actions + confidence only — **no prices,
  no values.** Book P&L history is therefore not reconstructable from it; an
  equity curve needs the loop to start logging marks.

### Trading Bots

- **GoldDigger** (`portfolio/golddigger/`): Gold certificate trading (dry-run/live via Avanza)
- **Elongir** (`portfolio/elongir/`): Equity trading bot (separate signal system)

## Signal System (90 Tracked · 13 Active · 77 Disabled)

> Counts reconciled to code 2026-08-21: `SIGNAL_NAMES`=90, `DISABLED_SIGNALS`=77,
> active=13, 80 files in `portfolio/signals/`, 69 `register_enhanced()` calls in
> `signal_registry.py`. Derive the live active set with:
> `python -c "from portfolio.tickers import SIGNAL_NAMES, DISABLED_SIGNALS; print([s for s in SIGNAL_NAMES if s not in DISABLED_SIGNALS])"`

### Active (13 globally + per-ticker overrides)

Accuracy figures below are as-documented and go stale; the live source of truth
is `data/accuracy_cache.json` via `python -m portfolio.main --accuracy`.

1. RSI(14) — Oversold <30 BUY, overbought >70 SELL
2. BB(20,2) — Bollinger Band breakout
3. Fear & Greed — Contrarian (≤20 BUY, ≥80 SELL)
4. Momentum — Stochastic, StochRSI, CCI, Williams %R, ROC, PPO
5. Mean Reversion — RSI(2/3), IBS, Gap Fade, BB %B
6. News Event — Headline velocity, keyword severity, source credibility
7. Econ Calendar — FOMC/CPI/NFP proximity risk-off + post_event_relief BUY
8. Crypto Macro — DeFi TVL, staking yields, protocol revenue (crypto only)
9. COT Positioning — CFTC speculative/commercial positioning
10. On-Chain BTC — MVRV Z-Score, SOPR, NUPL, Exchange Netflow (BTC-only)
11. Statistical Jump Regime — Jump detection for regime changes
12. Drift Regime Gate — Regime detection via drift analysis
13. Amihud Illiquidity Regime — Illiquidity ratio regime detection

**NO local-LLM signal is globally active.** Ministral-8B and Qwen3-8B were
listed here as #4 and #5 until 2026-08-21 but were globally disabled on
2026-07-17 (`a953f63b`, "Bench ministral + qwen3 — focus on phi4_mini"). The
only LLM voter is `phi4_mini`, and only via the per-ticker overrides below.

Per-ticker overrides (globally disabled, re-enabled for specific tickers via
`_DISABLED_SIGNAL_OVERRIDES` in signal_engine.py):

- `phi4_mini` → BTC-USD, ETH-USD, XAU-USD, XAG-USD (promoted 2026-07-13,
  `5c654545`; deliberately NOT on MSTR — untested there). Runs via **remote**
  inference on herc2, so it votes even though `data/local_llm.disabled` is in
  place — that flag gates LOCAL inference only.
- `realized_skewness` → XAU-USD (60.3%, 572 sam)
- `ml` → ETH-USD (55.1% 3h, 1206 sam)

(Williams VIX Fix override removed 2026-05-31 — recent accuracy collapsed to
30.5%; Credit Spread Risk override removed 2026-05-26 — was re-enabling a
broken signal.)

### Disabled (77 — force-HOLD via DISABLED_SIGNALS)

Core disabled: ML Classifier, MACD, EMA, Volume Confirmation, Funding Rate,
Sentiment, Forecast (Chronos — fully disabled 2026-05-12), Kronos (retired
2026-04-21), Claude Fundamental, Fibonacci, Ministral-8B + Qwen3-8B (both
benched 2026-07-17), phi4_mini (globally off; per-ticker override only)

Recently disabled (collapsed accuracy, 2026-05/06): Metals Cross-Asset
(disabled 2026-06-06), Crypto EVRP (re-disabled 2026-05-26), ADX Regime
Switch (re-disabled 2026-06-01, 49.0% on 492 sam), Choppiness Regime Gate,
BOCPD Regime Switch, Vol Ratio Regime

Enhanced disabled: Trend, Volume Flow, Volatility, Candlestick, Structure,
Heikin-Ashi, Calendar, Macro Regime, Smart Money, Oscillators, Orderbook Flow,
Futures Flow, DXY Cross-Asset

Pending validation (added Apr-May 2026): Futures Basis, Hurst Regime,
Shannon Entropy, VIX Term Structure, Gold Real Yield Paradox, Cross-Asset
TSMOM, Copper/Gold Ratio, Network Momentum, OVX Metals Spillover, XTrend
Equity Spillover, Complexity Gap Regime, Realized Skewness, Mahalanobis
Turbulence, Crypto EVRP, Hash Ribbons, Residual Pair Reversion, Williams
VIX Fix, Treasury Risk Rotation, Intraday Seasonality, Cubic Trend
Persistence, VWAP Z-Score MR, Gold Overnight Bias, Metals VRP, Breakeven
Inflation Momentum, Trend Slope Momentum

### Signal Mechanics

- **MIN_VOTERS = 3** (crypto/stocks), **2** (metals, since 2026-05-11 — MIN_VOTERS=3 produced 0 metals trades in 20 days). Consensus = active voters (BUY+SELL), not total.
- **Accuracy gate**: signals below 47% accuracy (30+ samples) are force-HOLD (not inverted — inversion causes whiplash). Tiered: 50% for 7K+ sample signals.
- **Recency-weighted**: 70% recent (7d) + 30% all-time
- **Regime penalties**: ranging 0.75x, high-vol 0.80x confidence multipliers
- **Volume/ADX gates**: RVOL <0.5 forces HOLD
- **Accuracy tier boost**: 1.25x for 65%+ accuracy, 1.15x for 60%+, 1.05x for 55%+
- **Applicable signals** (via `_compute_applicable_count`, reconciled 2026-08-21):
  crypto=**14** (BTC-USD, ETH-USD), stocks=**10** (MSTR), metals=**12** (XAU-USD, XAG-USD).
  NOTE: this count is not constant. `GPU_SIGNALS` (ministral, qwen3, forecast,
  phi4_mini) are dropped when remote inference is unreachable, so crypto and
  metals fall by 1 (crypto 14→13) whenever herc2 is asleep. If an applicable-count
  test fails, check herc2 reachability before assuming a code change.

## Instruments

### Tier 1: Full signals (13 active × 7 timeframes, +per-ticker overrides)

| Asset Class | Tickers          | Source       |
| ----------- | ---------------- | ------------ |
| Crypto 24/7 | BTC-USD, ETH-USD | Binance spot |
| Metals 24/7 | XAU-USD, XAG-USD | Binance FAPI |
| US Stocks   | MSTR             | Alpaca       |

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

`signal_engine.py` (consensus voting, 13 active), `signal_registry.py` (plugin discovery),
`signals/*.py` (80 enhanced modules, 69 register_enhanced calls), `accuracy_stats.py` (hit rates),
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
`gpu_gate.py` (GPU lock), `shared_state.py` (thread-safe cache, rate limiters),
`local_llm_gate.py` (master pause switch for ALL local model inference —
GGUF/llama-server, BERT sentiment, Chronos, metals fallbacks.
**Pause:** `touch data/local_llm.disabled` (or config `local_llm.enabled=false`).
**Resume:** `rm data/local_llm.disabled` (and/or set config key back to true).
Live-togglable, no loop restart needed. Flag file wins over config.
**Status: PAUSED since 2026-07-02** — flag file is in place at user request.
**This does NOT mean no LLM signal votes.** The gate covers LOCAL inference
only; `phi4_mini` runs via REMOTE inference on herc2 and still votes on
BTC/ETH/XAU/XAG through `_DISABLED_SIGNAL_OVERRIDES`. It also means the
per-asset applicable-signal count silently drops by 1 whenever herc2 is
unreachable.)

Full module map (173 top-level `portfolio/*.py`, 300 incl. subpackages): `docs/SYSTEM_OVERVIEW.md`

## Key Data Files

| File                                 | Purpose                                                 |
| ------------------------------------ | ------------------------------------------------------- |
| `data/agent_summary.json`            | Full signal report (all tickers, ~64K tokens)           |
| `data/agent_summary_compact.json`    | Tiered compaction for Layer 2 (~1400 lines)             |
| `data/agent_context_t1.json`         | Tier 1 quick-check context (~200 lines)                 |
| `data/agent_context_t2.json`         | Tier 2 signal context (~600 lines)                      |
| `data/portfolio_state.json`          | Patient strategy: cash, holdings, transactions          |
| `data/portfolio_state_bold.json`     | Bold strategy: cash, holdings, transactions             |
| `data/portfolio_state_warrants.json` | Warrant holdings with leverage                          |
| `data/layer2_journal.jsonl`          | Layer 2 decision log                                    |
| `data/signal_log.jsonl`              | Every signal snapshot (+ `signal_log.db` SQLite)        |
| `data/prophecy.json`                 | Macro beliefs (silver_bull, btc_range, eth_follows_btc) |
| `data/trigger_state.json`            | Trigger detection baseline                              |
| `data/health_state.json`             | System health (heartbeat, errors, module failures)      |
| `data/telegram_messages.jsonl`       | All sent Telegram messages                              |
| `data/fundamentals_cache.json`       | Alpha Vantage stock data (daily refresh)                |
| `data/accuracy_cache.json`           | Signal accuracy (1d/3d/5d/10d horizons)                 |

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
# All tests (~11,700 tests across 478 files; reconciled 2026-08-21 via pytest --collect-only -q)
.venv/Scripts/python.exe -m pytest tests/

# Parallel (~5.5 min, 8 workers)
.venv/Scripts/python.exe -m pytest tests/ -n auto

# Specific file
.venv/Scripts/python.exe -m pytest tests/test_signal_engine.py -v
```

Tests using module-level file paths must patch to `tmp_path` for xdist safety.
Pre-existing failures (integration, config, state isolation) exist — see `docs/TESTING.md`
for the current count and triage.

## Environment

**Two hosts run this repo. Check which one you are on before running anything —
the Python path differs and the wrong one fails immediately.**

```bash
uname -s   # Linux => Steam Deck    MINGW*/MSYS* => herc2 (Windows)
```

### Steam Deck (Linux) — where the loop runs today

- **Python**: `.venv/bin/python` (3.13.5). Every command in this file written as
  `.venv/Scripts/python.exe` is the herc2 form — substitute `.venv/bin/python`.
- **Config**: symlink `config.json` → `/home/deck/.config/finance-analyzer/config.json`
- **Process management is systemd --user, NOT Windows scheduled tasks.**
  Enabled: `pf-loop` (Layer 1), `pf-dashboard`, `pf-swedbank`, plus timers
  `pf-databackup`, `pf-outcomecheck` (01:00), `pf-pickups` (08:00).
  Inspect with `systemctl --user list-units 'pf-*'` / `journalctl --user -u pf-loop`.
  Installed-but-disabled: `pf-metalsloop`, `pf-cryptoloop`, `pf-oilloop`,
  `pf-mstrloop`, `pf-golddigger`.
- **No GPU.** All local model inference is unavailable here; `phi4_mini` votes
  via _remote_ inference on herc2.
- **This host suspends often** — roughly half of wall-clock uptime. Wall-clock
  file staleness is therefore meaningless on its own; see the loop-health
  readers, which anchor on awake time (CLOCK_MONOTONIC) plus a boot id.

### herc2 (Windows 11 Pro) — GPU / LLM host

- Shell is Git Bash (set via `CLAUDE_CODE_GIT_BASH_PATH`).
- **Python**: `.venv/Scripts/python.exe` — forward slashes, full path.
- **GPU**: RTX 3080 10GB, CUDA 13.1. LLM inference runs in a separate venv at
  `Q:/models/.venv-llm`. GPU lock: `Q:/models/gpu_lock.py`.
- **Config**: symlink `config.json` → `C:\Users\Herc2\.config\finance-analyzer\config.json`
- **Scheduled Tasks**: PF-DataLoop, PF-Dashboard, PF-OutcomeCheck, PF-MLRetrain.

### Both

- **NEVER commit config.json** — it is a symlink to a file outside the repo and
  holds live API keys. Exposed on Mar 15 2026; those credentials are still
  unrotated and the blob is still publicly fetchable. Verify with
  `python scripts/check_leaked_creds_rotated.py` (exits 0 when clean).
- **Timezone**: operator is CET (UTC+1, CEST in summer). Market hours are
  DST-dependent (see `memory/market_hours.md`).
- **Live trading is gated off.** `portfolio/trading_gate.py` is the master kill
  switch and it fails CLOSED. Re-arming deliberately needs BOTH
  `rm data/trading.disabled` AND `trading.live_enabled=true` in config.json.
  The operator places every order by hand.

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
