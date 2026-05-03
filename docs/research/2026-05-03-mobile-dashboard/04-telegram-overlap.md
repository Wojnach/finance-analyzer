# 04 — Telegram Overlap Analysis (Mobile Dashboard Redesign)

**Date:** 2026-05-03
**Premise:** Telegram is the primary push channel. The phone dashboard's job is to show what Telegram cannot — drill-down history, rich charts, on-demand state. Anything fully covered by Telegram is low-priority on a phone.

This document maps every Telegram message type to the dashboard widget(s) that overlap it, then assigns each widget DUPLICATE / COMPLEMENT / UNIQUE.

---

## Section 1 — Telegram Message Taxonomy

Sources: `portfolio/message_store.py`, `portfolio/telegram_notifications.py`, `portfolio/digest.py`,
`portfolio/daily_digest.py`, `portfolio/agent_invocation.py`, `portfolio/bigbet.py`,
`portfolio/iskbets.py`, `portfolio/golddigger/runner.py`, `portfolio/elongir/runner.py`,
`portfolio/regime_alerts.py`, `portfolio/avanza_orders.py`, `portfolio/main.py`,
`portfolio/loop_contract.py`, `portfolio/accuracy_degradation.py`, `portfolio/crypto_scheduler.py`,
`portfolio/weekly_digest.py`, `portfolio/mstr_loop/telegram_report.py`, `data/metals_loop.py`,
`data/crypto_loop.py`.

Routing: `message_store.send_or_store(msg, config, category=...)` at `portfolio/message_store.py:170`.
`SEND_CATEGORIES` whitelist at `message_store.py:52` controls which categories actually hit
Telegram (vs. log-only).

### Frequency from `data/telegram_messages.jsonl` (5,003 entries, 2026-03-26 → 2026-05-03)

| Category | Count | Sent to TG | Notes |
|---|---:|---:|---|
| error | 1,893 | mostly yes | LOOP ERRORS, LOOP CONTRACT, FX WARNING, L2 FAILED |
| invocation | 1,439 | **no** (saved-only) | "Layer 2 T# (TIER): trigger" tag |
| analysis | 769 | **yes** | Layer 2 HOLD/BUY/SELL summaries; AUTO PROBABILITY |
| golddigger | 475 | mixed | bot lifecycle + signal-only BUY/SELL |
| digest | 177 | yes | 4-hour digest |
| crypto_report | 97 | usually no | hourly BTC/ETH/MSTR reports |
| health | 73 | mostly no | Pre-US / Post-US / Full health probes |
| daily_digest | 46 | yes | morning digest + ACCURACY DAILY |
| bigbet | 24 | yes | mean-reversion alerts |
| trade | 8 | yes | actual BUY/SELL execution headlines |
| research | 1 | no | overnight research summaries |

The (uncategorized) tail also contains hundreds of `data/metals_loop.py` raw `send_telegram()`
calls for FISH BUY/EXIT, L3 SELL, AVANZA SESSION, TRADE FILLED — these go directly via
`requests.post` (`metals_loop.py:959`) and bypass `send_or_store`.

### Message type catalog (one row per template)

| # | Trigger | Category | Code site | One-line content |
|---:|---|---|---|---|
| 1 | 4-hour boundary, periodic | digest | `portfolio/digest.py:267` | Claude Code activity counts, L1 trigger reason histogram, consensus B/S/H, both portfolios SEK + holdings, system health one-liner |
| 2 | Daily ~09:00 local | daily_digest | `portfolio/daily_digest.py:274` | Focus instruments (XAG/BTC) prices + 7d change + 1d direction prob + accuracy, market health score, exposure ceiling, movers, both portfolios, accuracy 7d window |
| 3 | Daily accuracy snapshot | daily_digest | `portfolio/accuracy_degradation.py:874` | "ACCURACY DAILY" — recent 7d consensus + per-LLM accuracy + signal counts |
| 4 | Layer 1 trigger fires (T1/T2/T3) | invocation | `portfolio/agent_invocation.py:824` | Saved-only: `_Layer 2 T# (TIER): trigger reason_` |
| 5 | Layer 2 produces actionable HOLD/BUY/SELL | analysis | Layer 2 subprocess → `requests.post` directly | Multi-line *HOLD*/*BUY*/*SELL* summary with per-ticker rows, F&G, VIX, FOMC days, both portfolios SEK+pnl, free-text reasoning paragraph |
| 6 | Layer 2 trade execution | trade | Layer 2 subprocess | `*PATIENT BUY ETH* $X · X SEK · F&G X` followed by ticker matrix and reasoning |
| 7 | Layer 2 fails | error | `portfolio/agent_invocation.py:1154` | `*L2 FAILED* T# exit=X (Xs) journal=False tg=False` |
| 8 | Auto-disabled stack overflow | alert | `portfolio/agent_invocation.py:1179` | `*ALERT* Layer 2 auto-disabled after N consecutive stack overflows` |
| 9 | Big Bet alert (3+/6 conditions sustained) | bigbet | `portfolio/bigbet.py:599` (formatter line 318) | `🔵 *BIG BET: BULL ETH-USD*` + conditions + price + F&G + Claude probability |
| 10 | Big Bet window closed/expired | bigbet | `portfolio/bigbet.py:456` (formatter line 367) | `⚪ *BIG BET CLOSED: BULL ETH-USD*` with entry→current price |
| 11 | ISKBETS entry triggered | iskbets | `portfolio/iskbets.py:699` (formatter line 449) | `🟡 *ISKBETS BUY {short}* @ $X` with signal votes, RSI/MACD/BB, conditions, stop/target levels, "Bought? Reply:" cmd |
| 12 | ISKBETS exit (stop/target/timeout) | iskbets | `portfolio/iskbets.py:655` (formatter line 521) | Exit type, P&L SEK + %, duration |
| 13 | Regime shift | regime | `portfolio/regime_alerts.py:181` | Saved-only: `*REGIME SHIFT: ticker* old → new` + 7d distribution |
| 14 | Crypto hourly report | crypto_report | `portfolio/crypto_scheduler.py:367` | BTC/ETH/MSTR signal+RSI+options+futures+prophecy progress |
| 15 | Loop crash recovery | error | `portfolio/main.py:898/910` | "_LOOP RESTARTED_" / outcome staleness / dead signals safeguards |
| 16 | Loop ticker errors | error | `portfolio/main.py:983/995/1118` | `*LOOP ERRORS* (Xs cycle)` ticker(s) failed |
| 17 | Loop contract violation | error | `portfolio/loop_contract.py:1695` (cat="error") | `*LOOP CONTRACT (main/metals/golddigger)* — N critical violation(s)` + per-invariant lines |
| 18 | FX rate stale/fallback | error | `portfolio/fx_rates.py:85` | `*FX WARNING* USD/SEK rate is Xh stale` |
| 19 | Health check (Pre-US, Post-US, Full) | health | `scripts/health_check_*.py` (saved-only) | Categorized OK / auto-fixed / needs attention |
| 20 | Avanza order placed/filled (portfolio swing loop) | analysis | `portfolio/avanza_orders.py:389,396,414` | `AVANZA BUY/SELL` confirmation; `AVANZA ORDER ERROR`; `AVANZA ORDER EXPIRED` |
| 21 | Metals trade-queue filled | (raw, no category) | `data/metals_loop.py:4690` | `*TRADE FILLED* {action} {warrant}` volume/price/total/reasoning |
| 22 | Metals trade-queue failure modes | (raw) | `data/metals_loop.py:4521,4551,4599,4628,4652,4700` | Session expired, expired-by-age, slippage rejection, stop-cancel-failed, place_order raised, TRADE FAILED |
| 23 | Metals fish engine BUY | (raw) | `data/metals_loop.py:3034` | `FISH BUY: {nm} {volume}u@{ask}` |
| 24 | Metals fish engine EXIT/SELL | (raw) | `data/metals_loop.py:3101,3105,3116` | `FISH EXIT … P&L:+X SEK`; `FISH SELL BLOCKED`; `FISH SELL ERROR` |
| 25 | Metals fish entry-momentum spike | (raw) | `data/metals_loop.py:1290` | `*ENTRY MOMENTUM: XAG +X% in Ys*` |
| 26 | Metals L3 emergency sell flow | (raw) | `data/metals_loop.py:3732,3743,3813,3869,3876,3892,3905` | `*L3 EMERGENCY SELL*`; `*L3 SELL OK*`; `*L3 SELL FAILED*`; `*L3 WARNING*`; `*L3* {pos}: no longer held` |
| 27 | Hardware trailing stop placement | (raw) | `data/metals_loop.py:4788,4792` | `Trailing stop placed: {name} {pct}% trail, {vol}u`; `*WARNING* HW trailing stop failed` |
| 28 | Avanza session lifecycle | (raw) | `data/metals_loop.py:4396,4407,4426` | `*METALS SESSION* recovered`; `*AVANZA SESSION EXPIRED*`; `*AVANZA SESSION WARNING* {age}h old` |
| 29 | Stop-cancel snapshot/reconcile failures | (raw) | `data/metals_loop.py:4063,4154,4204` | `*STOP SNAPSHOT FAILED*`; `*STOP RECONCILE FAILED*`; `*STOP CANCEL FAILED*` |
| 30 | Fish startup / EOD flush / reconcile | (raw) | `data/metals_loop.py:705,2065,2111` | `*FISH WARNING* positions API unavailable`; `*FISH EOD SELL*`; `*FISH RECONCILE*` |
| 31 | GoldDigger lifecycle (start/stop/digest) | golddigger | `portfolio/golddigger/runner.py:244,404,407` | `*GOLDDIGGER STARTED* mode/equity/poll`; daily digest; `*GOLDDIGGER STOPPED*` |
| 32 | GoldDigger BUY/SELL signal | golddigger | `portfolio/golddigger/runner.py:164,343` | `*GOLDDIGGER BUY/SELL* qty@price gold/S=composite reason` |
| 33 | GoldDigger session/SL errors | golddigger | `portfolio/golddigger/runner.py:189,192,280-321,378` | session expired / recovered / reloaded / dead; SL placement failure; halted |
| 34 | Elongir lifecycle (start/stop/hourly) | elongir | `portfolio/elongir/runner.py:181,211,217,253,270` | `*ELONGIR STARTED* equity/stop/TP/trail`; trade messages; hourly report |
| 35 | Crypto loop trades/alerts | (direct) | `data/crypto_loop.py:400` | dry-run shadow loop alerts |
| 36 | MSTR loop reports | analysis | `portfolio/mstr_loop/telegram_report.py:144` | shadow phase periodic |
| 37 | Weekly digest (manual / cron) | analysis | `portfolio/weekly_digest.py:291` | trades this week, P&L delta, best/worst signals, regime distribution |
| 38 | Layer 1 broad ALERT (alert function) | analysis | `portfolio/telegram_notifications.py:139` | `*ALERT: {headline}*` per-ticker action grid (currently disabled — `layer1_messages: false`) |
| 39 | Snipe instrument plan failure | error | `portfolio/fin_snipe_manager.py:104` | `*SNIPE ALERT* plan_instrument failed for {name}` |
| 40 | Overnight research complete | research | (manual) | `*OVERNIGHT RESEARCH COMPLETE*` shipped signal list |

### Sample messages (last 30 days, ground-truthed from `telegram_messages.jsonl`)

These are the actual delivered messages on the user's phone:

```
[2026-05-03 18:33] analysis (sent)
*HOLD ALL* · BTC $78.7K BB ETH $2.33K HH XAG $75.94 H · F&G 47/68
`BTC  $78727  HOLD  4B/0S/26H  ··BBBS·`
`ETH  $2332   HOLD  3B/2S/24H  ···BBSS`
`XAG  $75.94  HOLD  3B/2S/18H  ·SBSS`
_P:495K(-1%) ETH+XAG · B:466K(-7%) BTC · VIX 17↑ · FOMC 44d_
ETH SELL pressure cleared (sustained flip to HOLD). All positions ranging…
```

```
[2026-04-27 16:47] trade (sent)
*PATIENT SELL XAG* $75.25 · 17K SEK · FOMC tmrw
`XAG  $75.3  ↓47% 3h  ↓47% 1d  ↓36% 3d`
`  acc: 44% 1d (3411 sam) | 7d: -5.7%`
`  SOLD 50%: 24.6 oz @ 690.8 SEK/oz`
`BTC  $76.8K  ↑53% 3h  ↑55% 1d  ↑56% 3d`
`  -> Bold: 0.201 BTC @ +3.0% ✓`
…XAG stop breach (-1.7x ATR) forced 50% trim…
```

```
[2026-05-03 16:08] digest (sent)
*4H DIGEST*
_12:08 - 16:08 UTC (May 03)_
*Claude Code Activity*
`Invoked:    0`
`Succeeded:  0`
*L1 Triggers: 0*
_Patient: 498,020 SEK (-0.4%) · ETH-USD, XAG-USD_
_Bold: 489,500 SEK (-2.1%) · BTC-USD_
_Health: loop OK · signals 0ok/4fail · agent N/A_
```

```
[2026-04-15 ~] daily_digest (sent)
*DAILY* · XAU +0% 7d · XAG +3% 7d · BTC -4% 7d
*Focus Instruments*
`XAU  $4,497  +0.0% 7d  ↑55% 1d (acc 50%)`
`XAG  $69.8  +2.9% 7d  ↑100% 1d (acc 49%)`
`BTC  $66K  -3.9% 7d  ↑54% 1d (acc 50%)`
*Other Movers*  *Portfolio*  *Accuracy (7d window)*
```

```
[2026-03-27 11:12] bigbet (sent)
🔵 *BIG BET: BULL BTC-USD*
Setup: 3/6 conditions met
• RSI 19 (oversold) on 15m + next TF (16)
• Below lower BB on Now, 12h, 2d
• Volume 3.3x avg (capitulation)
BTC-USD $66,408.56 | F&G: 13
Confidence: MODERATE (3/6)
_Expected bounce: 5-15% ($69,729–$76,370)_  _Hold: 3-5h max_
```

```
[2026-04-30 ~] analysis (sent, T1)
⚡ T1 Quick Check | ETH sentiment flip
Trigger: ETH-USD sentiment neg→pos
📊 Held positions OK:
• Patient ETH 0.43sh: $2,256 (-3.2%) — ATR stop $2,237 ✓
• Patient XAG 12.3sh: $73.33 (-3.8%) — ATR stop $72.70 ✓ ⚠ below $75 support
• Bold BTC 0.20sh: $76,290 (+2.3%) — ATR stop $75,802 ✓
Macro: F&G 29/62 · VIX 18.2↓ · FOMC 47d
All ranging. No action. 3×HOLD.
```

```
[2026-05-03 13:25] health (saved-only)
*Health Check (Pre-US) — 2026-05-03 15:25 CET*
*AUTO-FIXED:*
[4] LLM Inference: Restarted PF-MetalsLoop (missing Chronos for [...])
*NEEDS ATTENTION:*
[2] Heartbeat & Cycles: Heartbeat 7m old (>5m)
[6] Telegram Delivery: 14 unsent msgs >30m old; Daily digest 54h old
3 OK / 1 fixed / 2 failed
```

```
[2026-04-08 ~] crypto_report (saved-only — usually muted)
*CRYPTO REPORT — 2026-03-26 13:02 CET*
📊 F&G ? | DXY 99.72 | VIX 27.44 | FOMC 33d
*BTC — $69,256*  Signal: BUY (3B/4S/21H) wConf 64% | RSI 28.0
Options: MaxPain $75,000 (-7.7%) | PCR 0.621
Futures: Funding 0.0014% | L/S 1.762
Prophecy: $100,000 target, 5.1% progress
*ETH — $...* …
```

```
[2026-05-03 13:48] error (sent)
*LOOP CONTRACT (main)* — 1 critical violation(s)
• min_success_rate: Signal success rate 0% below 50% threshold. 4/4 tickers failed.
```

```
[2026-04-19 ~] golddigger (mostly saved-only)
*GOLDDIGGER BUY* (signal-only)
9950248x @ 0.00 SEK
Gold: $4382 | S=2.35
_ENTRY: S=2.35 >= 0.7 (confirmed 4x), z_gold=2.35 | vol=HOLD…_
```

---

## Section 2 — Dashboard Widget Categorization

Tab inventory (from `dashboard/static/index.html:531-541`):
**Accuracy** (default), **Overview**, **Signal Heatmap**, **Equity Curve**, **Trigger Timeline**,
**Decisions**, **Messages**, **Health**, **Metals**, **GoldDigger**.

For each major widget I assign **DUPLICATE** (Telegram already pushes the same info on update),
**COMPLEMENT** (Telegram alerts on change; dashboard shows depth/history/context Telegram can't fit),
or **UNIQUE** (only available on dashboard).

### TAB: Overview (`index.html:550`)

| Widget | Category | Rationale |
|---|---|---|
| `#sCards` ticker cards (price, action, B/S/H, indicators, TF strip) | **COMPLEMENT** | Latest action+price is in every digest and analysis msg. The **per-ticker indicator row + TF strip** is denser than any TG message and updates in real time without spamming. |
| Multi-Timeframe Heatmap `#hmT` | **COMPLEMENT** | Telegram analysis msgs show a string like `BBSSSB` (`telegram_messages.jsonl` samples). The heatmap renders the full TF×ticker grid with color intensity — much faster to read on a phone than parsing dot strings. |
| Market Context (F&G, DXY, treasuries, FOMC, sentiment grid) | **DUPLICATE** | Every analysis msg footers `_F&G X · VIX Y · FOMC Zd_`. Daily digest leads with these. |
| Trade History `#tH` | **COMPLEMENT** | Trade execution messages exist (cat=`trade`), but only ~8 in 5K msgs. Dashboard shows the **filterable historical record** with running P&L — TG cannot. |
| Warrant Portfolio `#warrantC` | **UNIQUE** | No periodic Telegram message lists current warrant positions with leverage/barrier/spread. Daily digest only mentions warrants if the focus ticker is held. |
| Risk Overview / Monte Carlo `#riskC` | **UNIQUE** | VaR/CVaR/Sharpe/drawdown surfaces never appear in Telegram. |
| LoRA Training Status (collapsible) | **UNIQUE** | No TG channel for training progress. |
| Recent Telegram Messages (collapsible) | **DUPLICATE** | Literally re-displays Telegram. |

### TAB: Signal Heatmap (`index.html:606`)

| Widget | Category | Rationale |
|---|---|---|
| 30-Signal × ticker grid (BUY/SELL/HOLD per signal per ticker) | **UNIQUE** | The fullest you ever see in TG is a one-line `B·SSSS·` per ticker in analysis messages — the **per-signal vote-by-vote breakdown does not exist anywhere on Telegram**. This is the dashboard's deepest unique value. |

### TAB: Equity Curve (`index.html:616`)

| Widget | Category | Rationale |
|---|---|---|
| Equity curve canvas (Patient + Bold over time) | **UNIQUE** | Digests and analysis messages show **point-in-time** SEK values (`P:495K(-1%)`). The continuous curve / drawdown shape only exists on the dashboard. |

### TAB: Trigger Timeline (`index.html:627`)

| Widget | Category | Rationale |
|---|---|---|
| Trigger list (last 50, with reason) | **COMPLEMENT** | Each trigger fires a saved-only `invocation` Telegram (1,439 of these in 30d, all `sent=False` — see `agent_invocation.py:824`). The 4h `digest` aggregates them into reason histograms (`digest.py:194-196`). The dashboard shows the **chronological raw stream**, which TG never emits. |

### TAB: Decisions (`index.html:639`)

| Widget | Category | Rationale |
|---|---|---|
| Decision table (filterable by ticker/action/strategy) | **COMPLEMENT** | Each Layer 2 decision sends an analysis/trade Telegram (cat=`analysis` or `trade`). Telegram is **one decision per message in chronological sequence**. The dashboard adds **filtering by ticker/action/strategy** which TG cannot do. |
| Decision detail panel (`showDecDetail`, `index.html:2808`) | **COMPLEMENT** | Telegram message shows the headline + summary paragraph. The detail panel reads `layer2_journal.jsonl` and exposes the full structured fields (regime, watchlist, conviction per ticker, thesis) — much richer than the SMS-shaped TG message. |

### TAB: Accuracy (`index.html:691`) — DEFAULT TAB

| Widget | Category | Rationale |
|---|---|---|
| Signal accuracy current (per-signal × horizon table) | **UNIQUE** | Daily digest shows accuracy **only for focus tickers at 1d horizon** (`daily_digest.py:248-257`). The full per-signal × horizon matrix is dashboard-only. |
| Signal Accuracy Trend chart | **UNIQUE** | Time series of accuracy is never on Telegram. |
| Per-ticker accuracy section | **UNIQUE** | Same. |
| Metals Loop Accuracy (sub-tab, 1h/3h horizons) | **UNIQUE** | Metals accuracy at sub-day horizons is dashboard-only. The `daily_digest` accuracy line covers daily horizons. |

### TAB: Messages (`index.html:722`)

| Widget | Category | Rationale |
|---|---|---|
| Filterable message log with category chips + search (`loadMessages` at line 1683) | **DUPLICATE-ish but COMPLEMENT** | This **is** the Telegram archive. It duplicates content but adds **search + category filter + view of saved-only categories** (`invocation`, `crypto_report`, most `health`, most `golddigger`). For categories that NEVER reach the phone (1,439 invocations + 73 health + 97 crypto_report + 475 golddigger = ~2,084 of 5,003 messages, **42%** of the log), the dashboard is the **only** way to see them. So even though the format is identical, this is functionally **COMPLEMENT** for the saved-only half and **DUPLICATE** for the sent half. |

### TAB: Health (`index.html:743`)

| Widget | Category | Rationale |
|---|---|---|
| Loop Status / Agent Status / Cycles / Errors cards | **COMPLEMENT** | The 4h `digest.py:253` ends with `_Health: loop OK · signals Nok/Mfail · agent X/Y_`. The pre-US/post-US health probes go to Telegram as cat=`health` (mostly saved-only — only 73 in 30d, and a single one-liner per probe). The dashboard shows **continuously updating heartbeat age + module-level failure list** which TG never carries. |
| Module failures pills | **UNIQUE** | Per-module last-failure timestamp is dashboard-only. |
| Recent Errors list | **COMPLEMENT** | Errors are pushed to TG (`*LOOP ERRORS*`, `*L2 FAILED*`, `*LOOP CONTRACT*`) but the dashboard shows the **rolling window of all of them with timestamps**. |

### TAB: Metals (`index.html:753`)

| Widget | Category | Rationale |
|---|---|---|
| Portfolio Summary banner | **UNIQUE** | Live SEK total + P&L for warrants is not in any periodic TG. |
| Position cards (held warrants with leverage, entry, stop, P&L) | **UNIQUE** | Telegram fires per-event (`*TRADE FILLED*`, `FISH BUY`, `*L3 SELL*`) — dashboard shows the **current live state of every held warrant**. |
| Risk & Signals panel (drawdown, ATR stops, signal mix) | **UNIQUE** | Only mentioned obliquely in trade Telegrams ("ATR stop $2,237"). Aggregated panel is dashboard-only. |
| Silver Technicals panel | **COMPLEMENT** | Silver price/RSI sometimes appears in analysis msgs and crypto_report has a metals proxy — but full structural levels (BB, ATR, support/resistance) are dashboard-only. |
| Recent Decisions panel | **COMPLEMENT** | Per-decision the analysis msg covers it; the panel is the **scrollable history with filtering**. |
| Intraday Price Chart | **UNIQUE** | No chart on Telegram. |

### TAB: GoldDigger (`index.html:794`)

| Widget | Category | Rationale |
|---|---|---|
| Summary banner (equity, daily/total P&L, daily trades) | **DUPLICATE-ish** | The end-of-session `_build_daily_digest` sends almost exactly this content (`golddigger/runner.py:131-145`). When the bot is running, the same numbers are visible. |
| Composite Signal panel (z-scores, S_t composite, regime) | **COMPLEMENT** | Each `*GOLDDIGGER BUY/SELL*` message embeds the S value (`S=2.35 >= 0.7 (confirmed 4x), z_gold=2.35`). The dashboard panel shows **continuous evolution + sub-component breakdown** without spamming the channel. |
| Position & Risk panel | **UNIQUE** | Continuous live position + stop distance is dashboard-only. |
| S_t History chart | **UNIQUE** | Time-series of the composite is never on Telegram. |
| Trade history table | **COMPLEMENT** | Each fill produces a Telegram; the table is the **filterable historical record**. |

---

## Section 3 — Mobile Priority Recommendation

### Top 5 widgets that should be HOME-SCREEN on mobile

Ranked by `UNIQUE` value × frequency-of-need × poor-Telegram-coverage:

1. **Equity Curve (Patient + Bold)** — `tab-equity`, `index.html:616`.
   Rationale: UNIQUE. Telegram only carries point values; the curve answers "how am I doing
   right now" in one glance. This is the single most important question a phone user has and
   it's simply not answerable from the TG channel alone.

2. **Position Cards across all asset classes** — combine:
   - Patient + Bold holdings from Overview `#sCards` portion + Trade History,
   - Warrant Portfolio `#warrantC` (Overview),
   - Metals position cards `#metalsPositions` (`tab-metals`),
   - GoldDigger position panel.

   Rationale: UNIQUE. TG messages mention current holdings only as a one-line tail
   (`P:495K(-1%) ETH+XAG · B:466K(-7%) BTC`). On phone the user wants to **see entry, stop,
   leverage, distance-to-stop, $/SEK P&L per holding** at a glance. This is dashboard-only.

3. **Signal Heatmap (30 signals × tickers)** — `tab-signals`, `index.html:606`.
   Rationale: UNIQUE. The deepest "why" lives here. TG analysis messages compress this to a
   `BBSSSB` string per ticker, omitting which signal voted which way. On a phone, tapping into
   this answers "what's the current consensus and which voters disagree" in seconds.

4. **Health summary card** — `tab-health`, `index.html:743`.
   Rationale: COMPLEMENT but critical. The 4h digest mentions one health line; pre/post-US
   probes are saved-only (cat=`health`, only 73/5003 = 1.5% of log). On a phone you want a
   **persistent green/red dot** showing the loop is alive and last-heartbeat age, not a one-shot
   Telegram from 3 hours ago. Recent CLAUDE.md `STARTUP CHECK` and `critical_errors.jsonl`
   architecture make this prominence essential.

5. **Trigger Timeline + Decisions (combined feed)** — `tab-triggers` + `tab-decisions`.
   Rationale: COMPLEMENT. The 1,439 saved-only `invocation` messages and the analysis/trade
   stream together tell the story of "what just happened and why". On a phone, the merged
   chronological feed with **filterable by ticker** is the most useful drill-down — Telegram
   sends the events but cannot filter or scroll.

### Widgets that can be deprioritized / collapsed / put behind a tap (DUPLICATE)

These should NOT be home-screen because Telegram already pushes the same info on every change:

- **Market Context** (F&G, DXY, treasuries, sentiment grid) — every analysis msg embeds the
  F&G/VIX/FOMC tail. Deprioritize behind a tap. (Code: `index.html:563-574`.)
- **Recent Telegram Messages** collapsible on Overview — direct re-display of TG. Drop on
  mobile entirely; route the user to the Messages tab instead. (Code: `index.html:597-600`.)
- **GoldDigger Summary banner** — virtually identical to the bot's own daily digest TG. Keep
  as a small status pill, not a full banner.
- **Daily Digest content (focus instrument prices + 7d change)** — fully covered by the
  morning daily_digest TG. Don't surface this as a separate widget on mobile.

### Things Telegram covers so well that the dashboard widget could be dropped on mobile entirely

- **Bigbet alerts panel** (if one were ever to be added): the `*BIG BET:*` Telegram is
  self-contained with conditions, F&G, expected bounce range, and Claude probability. Adding
  a dashboard mirror buys nothing on a phone. (Currently no dedicated tab — good.)
- **ISKBETS entry/exit alerts**: same — the message contains stop/target/levels/conditions and
  even the `bought TICKER PRICE AMOUNT` reply command. A phone user reads, taps, replies.
  Dashboard mirror would only add latency. (No dedicated tab — keep it that way.)
- **Daily digest content** rendered as its own widget: the Telegram message itself is already
  the canonical rendering of focus prices + 7d changes + accuracy + portfolio totals. Don't
  reproduce it as a tile on mobile — link to the latest digest in Messages instead.
- **Crypto Report 4h summary** (cat=`crypto_report`): currently saved-only and showing
  "?" placeholders for many fields (recent samples — `crypto_report` payload broken since
  end of April, see sample on 2026-05-03). On mobile this should not be a tab; surface only
  if/when the data quality is fixed and the user actually wants it pushed.

---

## Surprising findings

1. **42% of `telegram_messages.jsonl` is never delivered to the phone.** Categories
   `invocation` (1,439), `health` (73), `crypto_report` (97), and most `golddigger` (475)
   are saved-only. The dashboard's Messages tab is the **sole interface** to roughly two
   thousand log entries the user has never seen. This shifts the mobile priority of the
   Messages tab from "duplicate" to "complement-with-search-only".

2. **Layer 2 analysis messages (`category=analysis`) are sent directly by the Layer 2
   subprocess via `requests.post`, not through `send_or_store`.** That's why `agent_invocation.py`
   uses the `_telegram_ts_before` / `telegram_ts_after` diff trick (`agent_invocation.py:1056-1073`)
   to detect a Telegram was sent. This means the actual analysis content lives in the JSONL
   only via the subprocess writing it after the fact — not in Python code we can grep for
   `_format_analysis`. The phone's Messages tab is therefore the only audit trail of what
   Layer 2 actually said.

3. **`metals_loop.py` has its own raw `send_telegram()` at line 959** that bypasses
   `send_or_store` entirely. That's ~30+ message templates (FISH BUY/EXIT, L3 SELL flow,
   AVANZA SESSION lifecycle, STOP CANCEL flows, TRADE FILLED) that **do not get categorized
   or sent through the muting/blocklist gates** of `message_store.py`. On mobile,
   uncategorized metals messages appear in TG but cannot be filtered in the Messages tab
   because they have no `category` field. This is a coverage hole worth flagging.

4. **`SEND_CATEGORIES` whitelist quietly grew to include `error`**
   (`message_store.py:52`), so every loop ticker failure becomes a phone notification.
   Recent log shows 1,893 error messages in 38 days = ~50/day. The most recent `LOOP CONTRACT`
   spam (every 8min on 2026-05-03) suggests this is currently noisier than useful — relevant
   for the mobile UX because phone users need a **digest-style error rollup**, not 50
   individual buzzes per day. The Health tab already aggregates these; surfacing a
   "errors-since-last-open" badge on the dashboard tab icon is the right mobile pattern.

5. **The Accuracy tab is the dashboard's default tab**, yet is the LEAST covered by Telegram
   (only the 7d focus-ticker accuracy reaches TG, and only via the morning digest).
   This means the current dashboard default already implicitly aligns with the
   "show what TG can't" principle for mobile redesign. Keep Accuracy prominent — but it's
   a sit-down-and-think tab, not a glance tab. For mobile home screen, prefer Equity Curve
   + Position Cards (more actionable at-a-glance) and demote Accuracy to a secondary tab.

6. **`telegram.layer1_messages: false`** in config means
   `portfolio/telegram_notifications.py:35-47` (the legacy `send_telegram` and the
   `_maybe_send_alert` function) are effectively dead paths on this deployment. All
   user-visible TG goes through `send_or_store`. The `*ALERT: {headline}*` per-ticker action
   grid in `telegram_notifications.py:91-137` exists in code but never fires — the equivalent
   is now the cat=`analysis` Layer 2 message, which is richer.
