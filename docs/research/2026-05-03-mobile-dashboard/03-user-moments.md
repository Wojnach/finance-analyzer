# Track 3 — User-Moment Mapping (Inferred)

**Date:** 2026-05-03
**Author:** main agent (inferred from CLAUDE.md, memory, layer2_journal, morning_briefing,
recent commits — no user interview per "continue without asking question" directive)
**Status:** hypothesis-grade; should be validated via opt-in usage log (Track 2 recommendation)

## Premise
The user is a single trader-developer on CET (Stockholm), running a fully autonomous
trading system. Telegram already pushes alerts, decisions, digests, and errors. The
phone dashboard's role is to provide what Telegram cannot: visual depth, drill-down,
historical scroll, and on-demand state.

This document enumerates the moments at which the user is likely to open the
dashboard on a phone, and what each moment needs.

## The seven moments

### M1. Morning Anchor — 07:00-09:00 CET, coffee or commute
**Question being answered:** "What happened overnight, where do we start the day?"

**Triggered by:** habitual; user wakes, phone in hand. Possibly the morning_briefing
TG message arriving (`docs/after-hours-research-prompt.md` Phase 4 sends one).

**Data needed:**
- Net P&L delta since last close (Patient + Bold + Warrants combined)
- Open positions with overnight movement (BTC, ETH, MSTR, XAG, XAU)
- Top 3-5 lines from `data/morning_briefing.json` (today's market_outlook, key levels)
- Last 5 Layer 2 decisions with timestamps (was the bot active overnight?)
- Loop health rollup (everything green?)
- "Watchlist" items the previous evening's reflection flagged (e.g., "BTC breakout
  above 80K" from journal)

**Telegram coverage:** morning briefing is partially in TG already. Dashboard's
add-value: visual scan, position-by-position cards, quick tap to drill in.

**Frequency:** daily, ~5 minutes.

### M2. Trigger Drill-Down — ad-hoc during the day
**Question being answered:** "TG just fired about XAG flipping SELL→BUY. Why does
the bot think that, and is it worth my attention?"

**Triggered by:** a TG alert. User taps notification, but TG is summary-only —
they want the full picture.

**Data needed:**
- Latest Layer 2 decision detail (full reasoning, both Patient + Bold)
- Signal vote breakdown for the trigger ticker (which signals voted what, weighted
  by accuracy)
- Recent price chart for the ticker (1h/4h/1d) with the trigger marked
- Accuracy of the dissenting signals (do they have a track record?)
- Recent decisions on the same ticker (what's the agent's own pattern?) — note
  this is exactly what the new `_build_decision_feedback` (`commit ef486cb4`)
  injects into Layer 2; the user wants the same context.

**Telegram coverage:** the trigger alert itself + the Layer 2 reasoning paragraph.
Dashboard's add-value: signal-vote *visual* (red/green dots, accuracy %), price
*chart*, scrollable history.

**Frequency:** 5-15 times/day during volatile periods, near zero on quiet days.
This is the highest-value mobile moment.

### M3. Position Check During Volatility
**Question being answered:** "Things are moving — am I bleeding? Are stops about
to trigger?"

**Triggered by:** user notices a price move (their own scanning, news, or just
intuition). Not necessarily a TG alert.

**Data needed:**
- Live P&L per position (real-time, not 60s-stale)
- Distance to stop-loss as % and SEK
- ATR-relative price velocity (is this a normal range or breakout?)
- Stop-loss order status (active? primed? expired?)
- Any recent Avanza order activity

**Telegram coverage:** stop-loss firings are TG-pushed. Live unrealized P&L is NOT
in TG (would be too noisy). Dashboard-unique.

**Frequency:** during 1-5 high-vol windows per week, possibly multi-minute sessions.

**Memory cite:** `feedback_live_price_every_query.md` — user wants Binance FAPI
live, not yfinance lag. Phone dashboard inherits this requirement.

### M4. System Pulse — background curiosity
**Question being answered:** "Is the system still running? Did anything crash?"

**Triggered by:** habitual quick check between tasks; or absence of expected TG
messages ("haven't heard anything in 4h, suspicious").

**Data needed:**
- PF-DataLoop heartbeat age, last cycle timestamp
- PF-MetalsLoop, PF-CryptoLoop, PF-OilLoop, MSTR-loop, GoldDigger heartbeats
- Recent critical_errors count (resolved vs unresolved)
- Module failure count (signals_failed)
- LLM health (Ministral, Chronos, Qwen3 — are they responding?)
- Last successful Layer 2 invocation

**Telegram coverage:** errors push, but "everything fine" doesn't push. Dashboard-
unique for positive confirmation.

**Frequency:** several times a day, 10-20s sessions.

**Memory cite:** `feedback_restart_loops.md` — user keenly aware loops can desync
after merges. Pulse view should also show "loop-version vs HEAD-of-main mismatch"
if that's detectable.

### M5. Decision Backlog Review — evening / weekend
**Question being answered:** "Show me the day. Did the agent do well? Any patterns?"

**Triggered by:** post-work reflection or weekend review.

**Data needed:**
- Layer 2 decisions table, filterable by ticker / action / strategy / time
- Per-decision drill: trigger, regime, both strategies' reasoning, signal context
- Accuracy snapshot (1d/3d/5d/10d outcomes if backfilled)
- Pattern hits: e.g., "today the bot rejected 3 BUYs based on volume confirmation"

**Telegram coverage:** every decision goes to TG, but TG scroll-back is awful.
Dashboard wins on retrospection.

**Frequency:** daily evening (~10 min), weekend deep-dive (~30-60 min).

### M6. Research / Improvement Mode — weekend, late night
**Question being answered:** "Which signals are pulling weight? Where's the bias
hurting us? What should I blacklist?"

**Triggered by:** the user's standing research backlog (`memory/quant_research_priorities.md`,
session goals).

**Data needed:**
- Signal heatmap: signal × ticker × timeframe accuracy matrix
- Correlation clusters (which signals always agree)
- Regime-conditional accuracy breakdown
- Recent calibration drift (auto-blacklist recommendations)
- ML/LLM model health (Chronos, Qwen3, Ministral hit rates)
- Comparison: this week vs last week (deltas)

**Telegram coverage:** none meaningfully. Dashboard-native.

**Frequency:** weekly, 30-90 min sessions. Mobile-acceptable but desktop
preferable. Mobile design should be functional but not optimize for this.

### M7. Trade Execution Confirmation — post-execute
**Question being answered:** "Did the order fill? At what price? Updated state?"

**Triggered by:** user just placed an Avanza order manually (the user actively
trades metals warrants per `feedback_trading_rules.md`), or the GoldDigger /
metals_loop did. TG confirmation may have arrived but seeing the *position state*
is different.

**Data needed:**
- Most recent transaction (ticker, side, qty, fill price, fee)
- Updated position size + unrealized P&L
- Stop-loss order placement confirmation
- Distance to break-even / target

**Telegram coverage:** trade fills are TG-pushed by both metals_loop and Layer 2.
Dashboard adds: position state context, ladder of next-actions.

**Frequency:** 0-10 per day depending on activity.

## Prioritization for the home screen

The home screen on phone has space for ~5 cards above the fold (assuming a
390×844 iPhone-style viewport with 80px header + 80px bottom nav, leaving ~684px
content height = roughly 5 cards at 130px each, or 4 cards at 170px each).

**Above-the-fold priorities (in order):**

1. **Net P&L card** — three numbers (Patient / Bold / Warrants) + delta vs
   yesterday's close. Most-asked question. (M1, M3)
2. **Open positions strip** — horizontal scroll of position cards: ticker, side,
   P&L%, sparkline, distance-to-stop. Tap = M2/M3 drill-down. (M3)
3. **Latest decision card** — single most recent Layer 2 decision: action chip,
   ticker, 1-sentence reason, age. Tap = full decision detail. (M2, M5)
4. **System pulse strip** — loop heartbeats as colored dots + signal_failed
   count. (M4)
5. **Active triggers / consensus row** — for each ticker: BUY/SELL/HOLD chip
   with vote count "5B/3S/22H". Tap = drill-down. (M2)

**Below-the-fold (scroll):**

6. Today's market summary (from morning_briefing) — collapsible.
7. Watchlist items — what the agent is monitoring for entry/exit.
8. Recent Telegram messages — collapsible history (already in dashboard).

**Other tabs (bottom-nav second-tier, not home):**

- *Decisions* (M5 — full filterable table)
- *Signals* (M6 — heatmap, signal drill)
- *Health* (M4 expanded — every loop, every module, every error)
- *Settings* (theme, refresh cadence, pause auto-refresh, logout)

## Implications for design

- **Bottom nav with 4 items** seems right: Home / Decisions / Signals / More.
  "More" hosts Health, Messages, Metals, GoldDigger, Settings. Minimal,
  thumb-reachable.
- **Home is scroll, not tabs.** All 5 cards scroll vertically; no horizontal
  swipe between sections.
- **Pull-to-refresh** anywhere triggers full re-fetch (M1, M3, M4 are all
  polling-tolerant).
- **Tap-to-drill** consistently from each card → detail view.
- **Live polling** stays on Home (M3 needs it). Other tabs lazy-load on entry,
  poll only while active.
- **Charts on M2 drill-down** must be touch-friendly: pinch zoom, tap-on-bar
  for tooltip. (Track 6 will decide library.)
- **Don't duplicate Telegram digest** — the morning briefing card on home should
  *summarize and link out*, not re-render the full TG message.

## Open questions (for the next research iteration)

- Does the user actually use M6 (research mode) on phone, or only desktop?
  Implication: do we deprioritize signal heatmap mobile-fitness?
- Are the loop-health watchdog (`scripts/loop_health_watchdog.py`) alerts
  intrusive enough that M4 system pulse is "background only" rather than
  "actively check"?
- M7 fill-confirmation: would a TG-link-to-dashboard pattern (TG message
  contains `/decision/<id>` deep-link) be valuable? Could be implemented later.

These don't block the design — they refine future iterations.
