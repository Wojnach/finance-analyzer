# Current Dashboard Inventory — pre-mobile-redesign

Snapshot date: 2026-05-03
Scope: `dashboard/app.py` (1445 lines, 33 routes) + `dashboard/static/index.html` (3211 lines).
Plus `dashboard/house_blueprint.py` (separate Househunting blueprint, 11 routes — listed for completeness, not part of trader UI).

Tabs are listed in the order they appear in the nav bar
(`index.html:531-541`). Default tab on load is **Accuracy** (`active` on
line 691 + `index.html:691`). Header (always visible) is at
`index.html:490-528`.

---

## 0. Always-visible chrome (above tabs)

Tab summary: persistent header + nav, refresh control.

Widgets:
- **`tvl` Patient total** — Patient strategy value in SEK. From `/api/summary` → `signals.portfolio.total_sek`. 60s auto. Density: low. Mobile fitness: `keep-as-is` — single KPI.
- **`pnl` P&L %** — Patient pnl_pct, color coded. From `/api/summary`. 60s auto. Density: low. Fitness: `keep-as-is`.
- **`csh` Cash** — Patient cash_sek. From `/api/summary`. 60s auto. Density: low. Fitness: `keep-as-is`.
- **`boldSum` Bold total** — Bold strategy value + pnl%. Hidden until data present. Computed client-side from `/api/summary` portfolio_bold. 60s auto. Density: low. Fitness: `keep-as-is`.
- **`holdingsSum` Holdings list** — Comma-separated open positions across both strategies (`index.html:3120`). Computed client-side. 60s auto. Density: medium (truncated). Fitness: `redesign` — current text-only ellipsis is hard to scan on a phone.
- **`rfd` + `cdv` countdown** — Auto-refresh tick visualization (`index.html:521-523`). Density: low. Fitness: `keep-as-is`.
- **`pauseBtn` pause** — Toggles auto-refresh (`index.html:524`, JS at `949-963`). Fitness: `keep-as-is`.
- **`themeBtn` theme toggle** — Light/dark (`index.html:525`). Fitness: `keep-as-is`.
- **`navTabs`** — 10 nav tabs. Fitness: `redesign` — 10 horizontally-scrolling tab pills do not fit a phone width; needs hamburger / bottom-nav / scroll snap.

---

## 1. Tab × widget inventory

### Tab: Accuracy (default) — `tab-accuracy` (`index.html:691-717`)

Tab summary: signal accuracy stats for both the portfolio loop and the metals loop, with a trend chart and per-horizon breakdowns.

Sub-tab switcher (`accView-portfolio` / `accView-metals` chips at `index.html:693-696`).

Widgets:
- **`accChart` Signal Accuracy Trend chart** (`index.html:701-703`, JS `loadAccuracyHistory` at `2526`). Top-8-by-samples signals as Chart.js lines + 50% baseline. API: `/api/accuracy-history`. Cadence: on tab switch (lazy). Density: high (8 series + dates). Fitness: `redesign` — long legend bottom + tight axes do not survive narrow viewport; would compress to 2-3 series + sparkline.
- **`accC` Signal Accuracy panel** (`index.html:704-707`, JS `loadAccuracy`/`renderAccuracy` at `1595-1662`). Per-horizon (1d/3d/5d/10d) tables with: consensus pct, signals list w/ progress bars, per-ticker bars. Sortable (`pct_desc`/`asc`/`samples_desc`/`name_asc`). API: `/api/accuracy`. Cadence: on tab switch. Density: high (~30+ signals × 4 horizons). Fitness: `redesign` — tables of progress bars are dense; mobile should default to 1 horizon, expandable.
- **`metalsAccC` Metals Loop accuracy** (`index.html:711-716`, JS `loadMetalsAccuracy` at `2625`). 1h/3h horizons × {LLM, Chronos 1h, Chronos 3h, Ministral} × {XAG, XAU} matrix + extras. API: `/api/metals-accuracy`. Cadence: on sub-tab switch (memoized via `_metalsAccLoaded`). Density: high. Fitness: `redesign` — 2D matrix rendering needs reorientation on phone.

### Tab: Overview — `tab-overview` (`index.html:550-601`)

Tab summary: at-a-glance signal cards per ticker, multi-timeframe heatmap, market context, trades, warrants, risk.

Widgets:
- **`sCards` Ticker signal cards** (`index.html:551`, JS `rCards` at `1041`). One card per ticker: action badge, regime, vote tallies (B/S/H), weighted-confidence bar, RSI/MACD/F&G/ATR%, signal dots, inline 7-timeframe alignment row (Now/12h/2d/7d/1mo/3mo/6mo). API: `/api/summary`→signals. Cadence: 60s auto. Density: high. Fitness: `redesign` — perfect candidate for vertical card stack on mobile, but inline 7-tf row needs to wrap.
- **`hmT` Multi-Timeframe Heatmap** (`index.html:556-559`, JS `rHeat` at `1132`). Rows = tickers, cols = 7 timeframes, cells = B/S/H + confidence. API: `/api/summary`→timeframes. Cadence: 60s. Density: high. Fitness: `redesign` — already wrapped in `overflow-x:auto`; on mobile should pivot to ticker → vertical strip.
- **`mCtx` Market Context** (`index.html:566-573`). Container with:
  - **`fgR` Fear & Greed dual** (crypto + stocks F&G) — JS `rMkt:1166-1174`. From `/api/summary`. Density: low. Fitness: `keep-as-is`.
  - **`dxB` DXY** (`rMkt:1176-1181`). value/trend/implication. Density: low. Fitness: `keep-as-is`.
  - **`trB` Treasury yields + 2s10s** (`rMkt:1184-1196`). Density: medium. Fitness: `keep-as-is`.
  - **`fedB` FOMC proximity** (`rMkt:1198-1205`). Density: low. Fitness: `keep-as-is`.
  - **`sG` Sentiment per ticker** (`rMkt:1208-1213`). Density: medium. Fitness: `keep-as-is`.
- **`tH` Trade History panel** (`index.html:576-578`, JS `rTrades` at `1219`). Combined Patient+Bold transactions (last 50) in a table with Time/Strat/Ticker/Action/Shares/Price/SEK/Reason. API: `/api/summary`→portfolio + portfolio_bold transactions. Cadence: 60s. Density: high (8 columns). Fitness: `redesign` — 8-column table needs vertical card list on phone.
- **`warrantC` Warrant Portfolio** (`index.html:584-585`, JS `loadWarrants` at `3035`). Holdings table (Warrant/Units/Entry/Underlying/Leverage). API: `/api/warrants`. Cadence: every refresh (called from `refresh()` `3181`). Density: medium. Fitness: `redesign` — 5-col table.
- **`riskC` Risk Overview (Monte Carlo + VaR)** (`index.html:588-589`, JS `loadRisk` at `3059`). Per-strategy VaR95/CVaR95 + per-ticker price bands (p5/p50/p95) + stop probability. API: `/api/risk`. Cadence: every refresh (`3182`). Density: high. Fitness: `redesign` — dense numeric grid.
- **`lorS` LoRA Training Status** (collapsible, `index.html:593-596`, JS `rLora` at `1273`). Pipeline step list + training progress bar; hidden when no state. API: `/api/lora-status`. Cadence: every refresh (`3178`). Density: medium. Fitness: `drop` (mobile) — training-runtime monitoring is a desktop concern.
- **`telC` Recent Telegram Messages** (collapsible, `index.html:597-600`, JS `rTel` at `1298`). Last 10 messages newest-first. API: `/api/summary`→telegrams. Cadence: 60s. Density: medium. Fitness: `redesign` (mobile-only-candidate as standalone tab) — duplicates Messages tab; can drop here on mobile.

### Tab: Signal Heatmap — `tab-signals` (`index.html:606-611`)

Tab summary: full 30-signal × all-tickers grid (B/S/H per cell) for a comprehensive view of the voter system.

Widgets:
- **`sigHeatWrap` 30-Signal Heatmap** (`index.html:608-610`, JS `loadSignalHeatmap` at `1312`). Two grouped sections (Core 1-11, Enhanced 12-30) × N tickers. Single-letter B/S/H cells. API: `/api/signal-heatmap`. Cadence: on tab switch. Density: very high (30 rows × 5+ cols). Fitness: `redesign` — needs orientation flip or horizontal scroll with sticky first column.

### Tab: Equity Curve — `tab-equity` (`index.html:616-622`)

Tab summary: P&L equity curve over time with BUY/SELL trade annotations.

Widgets:
- **`equityChart` P&L Equity Curve** (`index.html:619`, JS `loadEquityCurve` at `1387`). Chart.js line: Patient + Bold series in SEK + scatter overlay of BUY (triangle) / SELL (rect) annotations. API: `/api/equity-curve` + `/api/trades`. Cadence: on tab switch. Density: high (2 series + N annotations). Fitness: `redesign` — chart aspect ratio + dense annotations need touch-tooltip rework.
- **`eqNoData` empty-state** (`index.html:620`). Static. Fitness: `keep-as-is`.

### Tab: Trigger Timeline — `tab-triggers` (`index.html:627-634`)

Tab summary: log of recent trigger/invocation events (what nudged Layer 2 to wake up).

Widgets:
- **`trigList` Recent Trigger Activity** (`index.html:630-632`, JS `loadTriggers` at `1559`). Reverse-chronological list of {timestamp, reasons[], action} pills. API: `/api/triggers`. Cadence: on tab switch. Density: medium (50 items, multiline). Fitness: `keep-as-is` — vertical-list pattern already mobile-friendly.

### Tab: Decisions — `tab-decisions` (`index.html:639-686`)

Tab summary: Layer 2 decision history with filterable table + drill-down detail panel.

Widgets:
- **`decFilterTicker` / `decFilterAction` / `decFilterStrategy` filters** (`index.html:643-662`). Three `<select>` dropdowns. Trigger reload via `loadDecisions()`. Density: low. Fitness: `redesign` — selects fine, layout horizontal needs to wrap.
- **`decTable` Decision table** (`index.html:665-680`, JS `loadDecisions`/`renderDecisions` at `2730`/`2771`). 6-column table: Time/Trigger/Regime/Patient/Bold/Reasoning. Click row → detail. Last 100. API: `/api/decisions?limit=100&...`. Cadence: on tab switch + filter change. Density: very high. Fitness: `redesign` — 6 cols won't fit; needs cards.
- **`decDetail` Decision detail panel** (`index.html:682-685`, JS `showDecDetail` at `2808`). Two-column metadata + per-strategy decision blocks + ticker outlooks grid + watchlist. Triggered by row click. Density: very high. Fitness: `redesign` — currently 2-col grid, must collapse to single column on phone.

### Tab: Messages — `tab-messages` (`index.html:722-741`)

Tab summary: searchable, filterable feed of all Telegram messages (sent or stored).

Widgets:
- **`msgFilters` Category chips** (`index.html:724-734`). 9 chips: All/Trade/Analysis/ISKBETS/Big Bet/Digest/Invocation/Regime/Error. JS `filterMsgs` at `1670`. Density: low. Fitness: `redesign` — 9 chips wrap awkwardly; keep but stack.
- **`msgSearch` Search box** (`index.html:736`). 300ms debounce (`searchMsgsDebounce` at `1678`). Density: low. Fitness: `keep-as-is`.
- **`msgC` Message list** (`index.html:738-740`, JS `loadMessages` at `1683`). One block per message: category badge + timestamp + sent/saved tag + body. API: `/api/telegrams?limit=200&category=&search=`. Cadence: on tab switch + filter/search change. Density: high (200 items, multiline each). Fitness: `keep-as-is` — list of cards is already mobile-friendly.

### Tab: Health — `tab-health` (`index.html:743-748`)

Tab summary: system health snapshot: loop heartbeat, agent activity, errors, module failures.

Widgets:
- **`healthC` System Health panel** (`index.html:745-747`, JS `loadHealth` at `1718`). 4 KPI cards (Loop Status, Agent Status, Cycles, Errors) + Module Failures chip list + Recent Errors list. API: `/api/health`. Cadence: on tab switch. Density: medium. Fitness: `keep-as-is` — 4 KPI cards already responsive grid (`auto-fill,minmax(200px,1fr)` at `1727`).

### Tab: Metals — `tab-metals` (`index.html:753-789`)

Tab summary: live metals subsystem monitor — positions, P&L, risk, signals, technicals, decisions, intraday chart.

Widgets:
- **`metalsSummary` Portfolio Summary Banner** (`index.html:755-757`, JS `renderMetalsSummary` at `1788`). Flex row of: P&L %, Value/Invested SEK, Gold USD, Silver USD, Session check/invoke counts + hours-to-close. API: `/api/metals`→context.totals/underlying. Cadence: on tab switch (no auto-refresh). Density: medium. Fitness: `redesign` — 5-cell flex row needs wrap.
- **`metalsPositions` Position cards** (`index.html:760`, JS `renderMetalsPositionCards` at `1832`). One card per warrant: name + leverage badge, P&L %, P&L SEK, bid/entry/value, stop-distance bar, peak/barrier/day metrics. API: `/api/metals`→context.positions. Density: high per card. Fitness: `redesign` — already card-shaped but min-width 220 horizontal flex; should stack.
- **`metalsRisk` Risk & Signals** (`index.html:764-767`, JS `renderMetalsRisk` at `1879`). Drawdown badge + Monte Carlo stop-prob table (3 rows × 3 cols) + Pipeline signals (XAG/XAU action) + LLM Consensus + Trade Guards status. API: `/api/metals`→context.risk + signals + llm_predictions. Density: very high. Fitness: `redesign` — small inner table will not fit phone.
- **`metalsTechnicals` Silver Technicals** (`index.html:768-771`, JS `renderMetalsTechnicals` at `1965`). Header price + session range + warrant change %, then 4-timeframe (1m/5m/15m/1h) × 5-indicator (RSI/MACD/BB/Vol/Chg%) table. API: `/api/metals`→technicals. Density: very high. Fitness: `redesign`.
- **`metalsDecisions` Recent Decisions** (`index.html:774-780`, JS `renderMetalsDecisions` at `2019`). 6-col table (Time/Tier/Gold/Silver79/Silver301/Prediction). API: `/api/metals`→decisions. Density: very high. Fitness: `redesign`.
- **`metalsPriceChart` Intraday Prices** (`index.html:782-788`, JS `renderMetalsPriceChart` at `2077`). Chart.js line chart, dual-axis (SEK warrant bids left, USD underlyings right), from `context.price_history_recent`. Density: high. Fitness: `redesign` — dual-y-axis chart on phone is a known anti-pattern.

### Tab: GoldDigger — `tab-golddigger` (`index.html:794-827`)

Tab summary: GoldDigger gold-certificate trading bot status — composite signal, position, score history, trades.

Widgets:
- **`gdSummary` Banner** (`index.html:796-798`, JS `renderGdSummary` at `2194`). Composite S(t), Mode (POSITIONED/SCANNING), Session OPEN/CLOSED, XAU/USD price, USD/SEK, Confirms count, last update. API: `/api/golddigger`→state. Cadence: on tab switch. Density: medium. Fitness: `redesign` — flex row of 7 KPIs.
- **`gdSignal` Composite Signal** (`index.html:802-805`, JS `renderGdSignal` at `2251`). Z-score bidirectional bars (gold/fx/yield) with weights + thresholds + signal gates. API: `/api/golddigger`→state. Density: high. Fitness: `redesign` — bars already vertical-friendly but gates section is dense.
- **`gdPosition` Position & Risk** (`index.html:806-809`, JS `renderGdPosition` at `2318`). Current quantity, entry, current, P&L %, stop, daily counters. Density: medium. Fitness: `redesign`.
- **`gdScoreChart` Composite Score (S_t) History** (`index.html:813-818`, JS `renderGdScoreChart` at `2367`). Chart.js line of S(t) with theta_in/theta_out reference lines. API: `/api/golddigger`→log. Density: medium. Fitness: `redesign` — chart fine; height 300 may be cramped on phone.
- **`gdTrades` Trade History** (`index.html:820-826`, JS `renderGdTrades` at `2486`). Trade list table. API: `/api/golddigger`→trades. Density: high. Fitness: `redesign`.

---

## 2. API endpoint catalogue (`dashboard/app.py`)

All endpoints require `pf_dashboard_token` cookie auth via `require_auth` (`auth.py`). All return JSON. TTL cache 5s default for file reads (`app.py:79`).

| Path | Method | Response (key fields) | Upstream cadence | Tabs that consume |
|------|--------|-----------------------|------------------|-------------------|
| `/` | GET | `static/index.html` | static | all (entry) |
| `/api/summary` | GET | `{signals, portfolio, portfolio_bold, telegrams}` (`app.py:704-717`) | 60s loop | header + Overview (auto-refresh) |
| `/api/signals` | GET | full `agent_summary.json` (`app.py:720`) | 60s loop | none currently bound (legacy; superseded by `/api/summary`) |
| `/api/portfolio` | GET | `portfolio_state.json` Patient (`app.py:729`) | on-trade | none currently bound (legacy) |
| `/api/portfolio-bold` | GET | `portfolio_state_bold.json` Bold (`app.py:738`) | on-trade | none currently bound (legacy) |
| `/api/mstr_loop` | GET | `{state, scorecard, last_poll, last_trade}` MSTR loop (`app.py:747`) | 60s MSTR loop | none in HTML |
| `/api/invocations` | GET | last 50 from `invocations.jsonl` (`app.py:794`) | per-trigger | none directly (replaced by `/api/triggers`) |
| `/api/telegrams` | GET | filtered telegram messages (`app.py:801`); query: `limit` (≤2000), `category`, `search` | per message-send | Messages |
| `/api/signal-log` | GET | last 50 entries from `signal_log.jsonl` (`app.py:828`) | 60s loop | none in HTML |
| `/api/accuracy` | GET | `{1d, 3d, 5d, 10d: {signals, consensus, per_ticker}}` (`app.py:835`) | daily backfill (PF-OutcomeCheck 18:00) | Accuracy |
| `/api/iskbets` | GET | `{config, state}` (`app.py:862`) | per iskbets cycle | none in HTML |
| `/api/lora-status` | GET | `{state, training_progress}` (`app.py:870`) | training run | Overview (collapsible LoRA) |
| `/api/validate-portfolio` | POST | `{valid, errors[]}` validates a posted portfolio (`app.py:882`) | one-shot | none in HTML (admin) |
| `/api/equity-curve` | GET | last 5000 from `portfolio_value_history.jsonl` (`app.py:910`) | per cycle | Equity Curve |
| `/api/signal-heatmap` | GET | `{tickers, signals, core_signals, enhanced_signals, heatmap}` 30×N grid (`app.py:925`) | 60s loop | Signal Heatmap |
| `/api/triggers` | GET | last 50 from `invocations.jsonl` (`app.py:980`) | per-trigger | Trigger Timeline |
| `/api/accuracy-history` | GET | last 500 from `accuracy_snapshots.jsonl` (`app.py:988`) | daily | Accuracy (chart) |
| `/api/local-llm-trends` | GET | `{ticker, latest, series}` (`app.py:996`); query: `limit` (≤366), `ticker` | daily local-LLM report | none in HTML |
| `/api/metals-accuracy` | GET | `{stats, resolved_snapshots, ts}` (`app.py:1021`) | metals loop | Accuracy (Metals sub-tab) |
| `/api/trades` | GET | merged Patient+Bold transactions sorted ts (`app.py:1031`) | on-trade | Equity Curve |
| `/api/decisions` | GET | last entries from `layer2_journal.jsonl` (`app.py:1062`); query: `limit` (≤500), `ticker`, `action`, `strategy` | per Layer-2 invocation | Decisions |
| `/api/health` | GET | `{status, heartbeat_age_seconds, agent_silent, agent_silence_seconds, cycle_count, signals_ok, signals_failed, error_count, recent_errors[], module_failures, last_trigger}` (`app.py:1105`) | 60s loop | Health |
| `/api/warrants` | GET | `{holdings, transactions}` (`app.py:1121`) | on-trade | Overview |
| `/api/risk` | GET | `{monte_carlo, portfolio_var}` (`app.py:1138`) | 60s loop | Overview |
| `/api/metals` | GET | `{context, decisions, history, technicals}` (`app.py:1158`) | 60s metals loop | Metals |
| `/api/crypto` | GET | `{state, context, warrant_catalog, risk, decisions, trades}` (`app.py:1215`) | 60s crypto loop | none in HTML |
| `/api/btc` | GET | per-ticker BTC slice of crypto state (`app.py:1246`) | 60s crypto loop | none in HTML |
| `/api/eth` | GET | per-ticker ETH slice (`app.py:1266`) | 60s crypto loop | none in HTML |
| `/api/loop_health` | GET | per-loop heartbeat rollup (`app.py:1286`) | 60s | none in HTML |
| `/api/oil` | GET | `{state, context, warrant_catalog, risk, decisions, trades, heartbeat}` (`app.py:1304`) | 60s oil loop | none in HTML |
| `/api/mstr` | GET | `{ticker, deep_context, loop_state, scorecard}` (`app.py:1342`) | 60s MSTR loop | none in HTML |
| `/api/golddigger` | GET | `{state, log, trades}` (`app.py:1368`) | 5s GoldDigger poll | GoldDigger |
| `/api/market-health` | GET | `{market_health, exposure_recommendation, earnings_proximity}` (`app.py:1392`) | 60s loop | none in HTML |

### House blueprint (`dashboard/house_blueprint.py`) — Househunting viewer (separate project, mounted at `/house`)

11 routes (`/house/`, `/house/runs`, `/house/runs/<id>`, `/house/runs/<id>/_manifest.json`, `/house/runs/<id>/<slug>`, `/house/runs/<id>/<slug>/raw`, `/house/heatmap`, `/house/api/runs`, `/house/api/runs/<id>`, `/house/api/runs/<id>/<slug>`). Out of scope for the trader-dashboard mobile redesign; listed for completeness only.

---

## 3. Cadence map

### Polled every 60s (driven by `refresh()` `index.html:3147`)
Triggered by `setInterval` countdown in `startCd()` (`index.html:3188-3201`). Pause-able via `togglePause()`.

- `/api/summary` (`refresh:3148` — single combined fetch)
- `/api/lora-status` (`refresh:3178`)
- `/api/warrants` (`loadWarrants()` called from `refresh:3181`)
- `/api/risk` (`loadRisk()` called from `refresh:3182`)

That is, **every 60s the dashboard makes 4 HTTP calls** while the user is on any tab. The tab whose UI is currently visible determines what gets re-rendered, but all 4 fetches happen regardless.

### Lazy-loaded on tab switch (one-shot per visit, no auto-refresh)
From `switchTab()` (`index.html:979-996`):

- Signal Heatmap → `/api/signal-heatmap`
- Equity Curve → `/api/equity-curve` + `/api/trades`
- Trigger Timeline → `/api/triggers`
- Decisions → `/api/decisions?limit=100&...` (re-fired on filter change)
- Accuracy → `/api/accuracy` + `/api/accuracy-history` (Metals sub-tab additionally one-shot fetches `/api/metals-accuracy` via `_metalsAccLoaded` memo flag)
- Messages → `/api/telegrams?limit=200&...` (re-fired on chip click + 300ms debounce on search input)
- Health → `/api/health`
- Metals → `/api/metals`
- GoldDigger → `/api/golddigger`

There is no auto-refresh on tab content. Users must switch tabs (or back-and-forth) to refresh non-Overview data.

### One-shot / no UI binding
Endpoints that exist server-side but no frontend call:
- `/api/signals`, `/api/portfolio`, `/api/portfolio-bold` (legacy — `/api/summary` covers them)
- `/api/invocations` (replaced by `/api/triggers`)
- `/api/signal-log`
- `/api/iskbets`
- `/api/local-llm-trends`
- `/api/crypto`, `/api/btc`, `/api/eth`, `/api/oil`, `/api/mstr`, `/api/mstr_loop`
- `/api/loop_health`
- `/api/market-health`
- `/api/validate-portfolio` (POST)

These are server-implemented but not consumed by the current `index.html`. They are likely available for external/CLI use or a future tab.

### TTL cache layer
Server-side, all `_read_json()` / `_read_jsonl()` use 5s TTL (`app.py:79`). Config reads use 60s TTL (`app.py:110`). So 4 simultaneous browser tabs polling at 60s cause at most 1 disk read per file per 5s, not 4× per minute.
