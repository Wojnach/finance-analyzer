# Silver Fishing Patterns — Lessons Learned

## Source Data
Analysis of March 2026 (21 trading days, 15-min candle data from SI=F futures).
Last updated: 2026-03-30.

## The Core Pattern: Silver Dumps During US Session

Silver consistently peaks in the EU morning and sells off into/during the US session.

### Timing (CEST — Swedish Summer Time, UTC+2)

**IMPORTANT:** These times shift by 1 hour when Sweden is on CET (winter time, UTC+1).
Always anchor to UTC or US market open (9:30 EDT = 13:30 UTC) as the reference.

| Phase | CEST (summer) | CET (winter) | UTC | What happens |
|-------|--------------|--------------|-----|-------------|
| Peak formation | 01:00-10:00 | 00:00-09:00 | 23:00-08:00 | Asia/early EU. Silver typically hits daily high |
| Pre-positioning | 13:00-14:00 | 12:00-13:00 | 11:00-12:00 | Selling starts, pre-US |
| US market open | 15:30 | 14:30 | 13:30 | Selling accelerates |
| TROUGH ZONE | 17:00-22:00 | 16:00-21:00 | 15:00-20:00 | 82% of dips bottom here |
| Avg trough time | ~17:00 | ~16:00 | ~15:00 | Best time to buy BULL |
| Recovery/bounce | 17:30-22:00 | 16:30-21:00 | 15:30-20:00 | Avg +2.7% in 1h after trough |

### Probability of Dips (March 2026)
- Dip >= 3% from daily high: **95%** (20 of 21 days)
- Dip >= 5%: **76%** (16 of 21 days)
- Dip >= 7%: **38%** (8 of 21 days)
- Average daily drawdown (all sessions): **-5.7%**
- Average US-session drawdown: **-7.3%**

### Dip Shape
- **73% gradual** (slow grind, avg 12.3h peak-to-trough)
- **18% grind-stay** (drops and stays low)
- **9% V-bounce** (sharp drop, quick recovery)
- Silver stays near trough for **avg 50 min** (median 30 min)
- Plenty of time to exit — these are NOT flash crashes.

### US Open Dump Speed (on dump days)
The dump builds gradually after US open:

| Time after open | Avg silver drop | Bear cert (5x) gain |
|-----------------|----------------|---------------------|
| 15 min | -0.7% | +3% |
| 30 min | -1.3% | +6% |
| 1h | -2.0% | +10% |
| 2h | -2.7% | +13% |
| 3h | -3.2% | +16% |

### Post-Trough Bounce (for BULL timing)
After the trough, silver bounces:

| Window after trough | Avg bounce | 5x cert gain |
|---------------------|-----------|-------------|
| 30 min | +1.9% | +9.5% |
| 1h | +2.7% | +13% |
| 2h | +3.2% | +16% |
| 3h | +3.6% | +18% |

Stabilization takes 15-60 min (median 30 min). Wait for confirmation
(price holds +0.5% above trough for 2 consecutive 15-min bars) before buying BULL.

## Strategy Comparison (Backtested on March 2026)

### BEAR now (buy bear at ~12:30 CEST, ride the dump)
- **Win rate: 71%** (15W / 6L)
- Avg P&L: **+5.6%** per trade
- Cumulative: **+118%** over 21 days
- Max loss: -15% (SL hit on rally days)

### BULL fish at -3% dip (limit buy, catch bounce)
- **Win rate: 26%** (5W / 14L)
- Avg P&L: **-5.3%** per trade
- Cumulative: **-111%** over 21 days
- **WARNING: Loses money in a trending-down market.** Silver was in a downtrend
  in March 2026. The -3% dip often kept going to -7% or -10%, hitting the SL.
  Only works in ranging/uptrending markets.

### Optimal: BEAR then BULL rotation
1. Buy BEAR before US open
2. Sell BEAR at trough (~17:00-20:00 CEST)
3. Wait 30-45 min for stabilization
4. Buy BULL, ride the bounce (+13-18% cert in 1-3h)
5. Exit everything by 21:00 CEST

## Instrument Selection

### Best Avanza Certificates (ranked by spread)

**Silver BULL (5x, AVA issuer):**
| Name | OB ID | Spread | Use when |
|------|-------|--------|----------|
| BULL SILVER X5 AVA 4 | 1650161 | 0.15% | **Best for fishing** (tight spread, 6.50 SEK/unit) |
| BULL SILVER X5 AVA 3 | 1069606 | 0.98% | Avoid — 6x wider spread than AVA 4 |

**Silver BEAR (5x, AVA issuer):**
| Name | OB ID | Spread | Use when |
|------|-------|--------|----------|
| BEAR SILVER X5 AVA 13 | 2304634 | 0.17% | **Best for bear fishing** (tight spread) |
| BEAR SILVER X5 AVA 12 | 2286417 | 0.32% | OK fallback |

**Gold BULL (5x, AVA issuer):**
| Name | OB ID | Spread | Use when |
|------|-------|--------|----------|
| BULL GULD X5 AVA 4 | 2044213 | 0.08% | **Best gold bull** |
| BULL GULD X5 AVA 3 | 1000679 | 0.08% | Same spread, higher price |

**Gold BEAR (5x, AVA issuer):**
| Name | OB ID | Spread | Use when |
|------|-------|--------|----------|
| BEAR GULD X5 AVA 7 | 2295081 | 0.11% | **Best gold bear** (5x AVA exists!) |

### Key Rules
- **AVA issuer preferred** — zero commission, tight spreads (0.1-0.2%).
- **Avoid SG/VT** — 0.5-2.5% spreads eat profits.
- Check `data/avanza_instruments_live.json` for full live catalog (34 instruments).
- `fin_fish_config.py` may be outdated — verify OB IDs against live data.

## Risk Management

### Stop-Loss Rules
- **-3% underlying minimum** for 5x certs (-15% cert). Never tighter.
- Previous -1% underlying / -5% cert stops triggered on normal intraday wicks repeatedly.
- **Sell + stop-loss volume must NOT exceed position size** (Avanza blocks as short-selling).
- Distribute volume: e.g., 40% TP1 + 35% TP2 + 25% SL = 100%.

### Trailing Stop Schedule
Ratchet SL upward as position moves in favor:

| Cert gain | Move SL to | Locks in |
|-----------|-----------|----------|
| +5% | -3% from entry | Small loss cap |
| +8% | Breakeven | Free trade |
| +10% | +4% | Profit protected |
| +13% | +7% | Strong gain locked |
| +15% | +10% | Let remainder ride |

### Volatile Window Management (US Open)
- **Remove SL** 15 min before US open (15:15 CEST summer / 14:15 CET winter).
- Silver can spike **+2.2%** in first 15 min before dumping — enough to trigger tight SLs.
- **Re-set SL** after initial volatility settles (~30-45 min after open).
- During this window, rely on TP sell orders + manual monitoring (15-30s checks).

### Time Management
- **Max hold: intraday only.** Daily certs reset overnight.
- **No new entries after 18:55 CEST** (summer) / 17:55 CET (winter).
- **Force exit by 21:00 CEST** (summer) / 20:00 CET (winter).
- Tighten SL to breakeven after 3h hold.

## Signal Timeframe Reference

**CRITICAL: Know what timeframe each signal predicts before using it for fishing.**

| Signal / Tool | Predicts | Relevant for intraday fishing? |
|--------------|----------|-------------------------------|
| **3h focus probability** | Next 3 hours | **YES — primary intraday signal** |
| **Chronos 1h** | Next 1 hour | **YES** |
| **Chronos 24h** | Next 24 hours | Partially — extends past close |
| **Now heatmap** | Instant | Too short, noisy |
| **12h heatmap** | Next 12 hours | Extends to next session — NO |
| **1d focus probability** | Next 24 hours | For overnight MINI holds only |
| **3d focus probability** | Next 3 days | For swing trades only — NO |
| **Monte Carlo P(up)** | 1-3 day drift | **NO** — do NOT use for intraday direction |
| **Signal consensus (BUY/SELL)** | Ambiguous timeframe | Use with caution — may reflect 1d+ view |
| **RSI level** | Current state | YES — overbought/oversold is instant |
| **BB position** | Current state | YES — above/below bands is instant |
| **news_event** | 3h forward | **YES** (72% accuracy on silver) |
| **econ_calendar** | Event proximity | **YES** — fires near scheduled events |

**Lesson from 2026-03-31:** Monte Carlo showed 78.5% P(up) and consensus was 95% BUY.
We panicked and adjusted TPs downward. But those signals predicted the NEXT DAY, not the
next 3 hours. The 3h focus probability (59% DOWN) was the correct signal for our intraday
BEAR trade. Always match the signal timeframe to the trade timeframe.

## Signal Filters (from 3,900 XAG-USD snapshots, Feb-Mar 2026)

### Dip Predictors — check before entering BEAR
These signals predict silver drops with above-average accuracy (3h forward):

| Signal | SELL accuracy | Samples | Use |
|--------|-------------|---------|-----|
| sentiment SELL | 88% | 24 | Strongest but small sample |
| econ_calendar SELL | 73% | 402 | **Best reliable predictor** — fires near scheduled events |
| news_event SELL | 72% | 488 | **Strong** — headline-driven selling |
| momentum SELL | 66% | 59 | Confirms direction |
| bb SELL | 62% | 164 | Bollinger breakout |
| rsi SELL | 58% | 810 | Largest sample, slight edge |
| qwen3 SELL | 57% | 288 | Local LLM, decent |
| macro_regime SELL | 55% | 916 | Weak but huge sample |

**Rule:** Prefer BEAR entry when econ_calendar=SELL AND (news_event=SELL OR momentum=SELL).

### Rally Predictors — check before entering BULL
These signals predict silver bounces after dips:

| Signal | BUY accuracy | Samples | Use |
|--------|-------------|---------|-----|
| smart_money BUY | 75% | 177 | **Best bounce detector** — BOS/CHoCH/FVG |
| fibonacci BUY | 75% | 118 | Catches structural support |
| momentum_factors BUY | 69% | 485 | **Large sample, reliable** |
| structure BUY | 68% | 376 | High/low breakout levels |
| mean_reversion BUY | 65% | 282 | Catches oversold conditions |
| calendar BUY | 63% | 315 | Day-of-week/seasonal patterns |

**Rule:** Enter BULL after dip when smart_money=BUY OR (fibonacci=BUY AND structure=BUY).

### Useless Signals for XAG (skip these)
| Signal | Issue |
|--------|-------|
| ema | 41% BUY accuracy — worse than coin flip |
| trend | 40% BUY — trend-following broken on silver |
| oscillators | 43% BUY, 13% SELL — broken both ways |
| claude_fundamental | 49% BUY — not useful short-term |
| volume_flow | 46% both ways — no edge |
| heikin_ashi | 52% BUY, 45% SELL — marginal |

### Vote Count Patterns
The number of BUY vs SELL voters predicts dip probability:

| Pattern | Dip rate (3h) | Notes |
|---------|--------------|-------|
| 0B/4S | **31%** | Strongest dip signal — go BEAR |
| 0B/3S | 19% (avg -2.0%) | Bearish |
| 1B/2S | **26%** | More sellers than buyers |
| 4B/2S | **27%** | Conflicting signals — uncertainty precedes dips |
| **5B/0S** | **0%** | Strong consensus — do NOT go BEAR |
| **6B/0S** | **0%** | Very safe — no dips ever followed |

**Rule:** Never enter BEAR when buy_count >= 5 and sell_count == 0.
Best BEAR entries: sell_count >= 3 AND buy_count <= 1.

## Macro Cross-Checks

Before entering a position, check:
- **Gold direction** — if gold is rallying, silver bear is riskier.
- **DXY** — strong dollar = headwind for metals (supports bear thesis).
- **Oil/Hormuz** — oil spike = risk-off = metals can dump.
- **RSI** — below 45 favors BULL fishing, above 65 favors BEAR fishing.
- **Chronos 24h forecast** — negative = supports bear, positive = supports bull.

## Lessons Learned

1. **Go WITH the trend.** In a downtrending market (March 2026), the bear play
   wins 71% vs 26% for bull fishing. Don't fight the direction.

2. **Spread matters more than you think.** BULL SILVER X5 AVA 4 (0.15%) vs
   AVA 3 (0.98%) saves ~0.83% per round-trip. On 1,500 SEK that's 62 SEK.
   Always check for newer certificates with tighter spreads.

3. **DST shifts the entire pattern by 1 hour.** Anchor to UTC or US market
   open time, not Swedish wall clock. Recalculate when clocks change
   (last Sunday of March, last Sunday of October).

4. **The dump is gradual, not a flash crash.** Average 12.3h selloff with
   50 min near trough. You have time to manage positions.

5. **Don't set tight stop-losses during US open.** Silver can spike +2.2%
   in 15 min before dumping. Remove SL, rely on TP orders + monitoring.

6. **Sell orders + SL share volume budget.** Plan exit as a "volume budget":
   e.g., 40% TP1 + 35% TP2 + 25% SL = 100% of position.

7. **Fish with limit orders, never hit the ask.** Place limits at bid or
   computed dip levels. Patience saves 0.5-1% per entry.

8. **The sequential play (bear then bull) has the highest EV.** Ride the
   dump with bear cert, sell at trough, wait 30 min, switch to bull for
   the bounce. Requires active management but ~2x the profit.

9. **Monitor dynamically.** Cruise at 2 min, speed up to 30s when near
   targets or SL. The `fish_monitor.py` script handles this.

10. **Today's pattern may not match history.** Always re-evaluate with live
    signals (gold direction, RSI, volatility) before committing. If gold
    is rallying hard, the silver bear thesis weakens.

11. **RSI oversold is not an instant exit signal.** On 2026-03-30, RSI hit
    43.8 and we exited the bear. Silver then dropped another -1.18% over
    the next 2 hours. RSI can stay oversold for extended periods during
    strong selloffs. Consider using RSI < 35 (not 45) as the exit threshold,
    or combine with price stabilization (2 consecutive green 15m bars)
    before exiting. The late-session dump (20:00-21:30 CEST) is real.

12. **Place fishing orders early — don't wait for the "right time."**
    On 2026-03-31, placed limit buy on BEAR at 23.80 during EU morning.
    Silver spiked to ~$73.40 and the order filled automatically while we
    waited. Better entry than buying at market at 13:00.

13. **The EU afternoon is dead zone chop (13:30-15:15 CEST).** Silver
    oscillates ±2-3% on the BEAR cert with no direction. Don't panic sell
    on dips or chase spikes. The real move comes with US volume after 15:30.
    On 2026-03-31: BEAR swung between 23.39-25.75 in this window, crossing
    entry price 10+ times before the directional move started.

14. **Monitoring beats stop-losses for intraday fishing.** Formal SLs on
    Avanza trigger on any wick regardless of context. Active monitoring
    with pre-calculated zones (CRUISE/APPROACH/DANGER) lets you distinguish
    noise from reversals. Example: at 14:10 BEAR dipped -1.7% on a
    liquidity grab, then snapped +76 SEK in 5 min. A -5% SL would have
    been triggered and cost us the trade.

15. **3h vs 1d probability split validates bear-then-bull.** On 2026-03-31:
    3h focus probability was 60% DOWN (supporting intraday bear play) while
    1d probability was 60% UP (supporting overnight MINI L hold). Use
    these probabilities to confirm the sequential strategy.

16. **The US open spike is smaller than documented.** Analysis of 25 days
    shows avg spike is +0.43% (not +2.2%). Only 6/25 days (24%) had >1%
    spikes, and they often DON'T reverse — some are breakouts. Don't try
    to trade the spike as a double-dip; hold through it.

17. **The double-dip strategy doesn't work.** Tested: buy BEAR for pre-US
    dip, sell before open, re-buy after spike. Data shows 1/27 days (4%)
    had the pattern. The dump is gradual and spread across the afternoon,
    not concentrated at open. Single BEAR hold is superior.

18. **MINI L SILVER SG spread is 0.08%, not 2.5%.** Config data was wrong.
    Live spread on 2026-03-31: bid 39.53, ask 39.56. Makes MINI warrants
    viable for overnight holds with minimal friction. ~1.75x leverage,
    barrier ~$31 (57% below current, essentially zero knockout risk).

19. **Chronos 24h prediction can shift dramatically intraday.** Went from
    -0.77% at 10:00 to -2.50% at 15:15 on 2026-03-31. Always re-check
    signals before key decision points, not just at session start.

20. **REGIME DETECTION IS THE #1 PRIORITY.** On 2026-03-31, we applied the
    March downtrend dump pattern (95% of days dump) to a BOUNCE day. Silver
    had rallied +6% overnight ($69.06→$73.57) — that's not a normal dump
    setup. The overnight move direction + magnitude should be the FIRST check:
    - Overnight +3%+ rally = DON'T go bear (momentum shifted)
    - Overnight flat/down = dump pattern likely applies
    - Check multi-day trend, not just intraday pattern

21. **When you're +8%, TAKE SOME PROFIT.** On 2026-03-31, BEAR hit +8.2%
    at 12:40 but our TPs were set at +13%/+23%/+33%. We gave back ALL gains
    and ended deep red. Always have a TP0 at +5% for 20-30% of position.
    A bird in the hand. Especially on days with uncertain regime.

22. **Monte Carlo and 1d signals were RIGHT — we dismissed them.** MC said
    78.5% P(up), 1d focus said 63% UP. We called these "wrong timeframe"
    for intraday, but they correctly predicted the DIRECTION. When MC and
    1d signals agree on direction, that's a strong counter-signal to intraday
    patterns. Don't dismiss longer-timeframe signals — they set the backdrop.

23. **Use mean-reversion math for exit targets, not hope.** Z-score, half-life,
    Bollinger Bands, and SMA20 give calculated reversion targets. On 2026-03-31:
    z-score 1.80, half-life 37 min, SMA20 target $73.73. These are more reliable
    than "pattern says it should dump 5%." Calculate, don't guess.

24. **Don't sell because price looks nice — predict WHERE it's going.** Set sell
    orders at calculated targets (mean-reversion SMA, z-score levels, Chronos
    predictions), not at "looks good" prices. Use yfinance + pandas-ta for
    live technical data when making exit decisions.

25. **News headlines can confirm or contradict.** "Gold and Silver Crashing"
    (Motley Fool, 2026-03-31) published same-day as our trade. Quarter-end
    (March 31 = Q1 close) can cause rebalancing flows. Always check news
    before AND during a trade, not just at entry.

26. **Pre-flight module is ESSENTIAL.** Build fish_preflight.py that aggregates:
    - Overnight move direction + magnitude (regime check)
    - Monte Carlo P(up) (directional backdrop)
    - 1d focus probability (medium-term direction)
    - RSI level at entry (overbought/oversold)
    - Signal consensus (BUY/SELL/HOLD)
    - News severity keywords
    Output: GO/NO-GO with confidence score and recommended direction.
    Today's score would have been: BEAR score 35/100, BULL score 72/100 = NO-GO bear.

27. **Don't invert bad signals — gate them.** Ministral at 18.9% accuracy on
    silver is noise, not inverse alpha. Only trust signals with >70% accuracy
    AND >100 samples. Everything else is excluded from the decision.

28. **Dynamic sell targets based on live data.** When the original thesis breaks,
    use live mean-reversion analysis (z-score, half-life, BB) to compute new
    targets instead of holding to dead TPs. Today: adjusted from 26.79→24.20→22.50→22.00
    as data changed. Each adjustment was data-driven, not emotional.

29. **Signals change throughout the day — don't anchor on entry signals.** On
    2026-03-31: entered on SELL consensus + RSI 65.5 overbought. By 16:50 signals
    had flipped to BUY (4B/1S) with 94.9% weighted confidence. RSI dropped to 58.
    The market you entered is NOT the market you're in 4 hours later. Re-evaluate
    at least every 2 hours with fresh signal data.

30. **Avanza order settlement is NOT instant.** When you sell an instrument, the
    buying power doesn't update immediately. On 2026-03-31 at 21:44 we tried to
    sell MINI S AVA 409 and immediately buy MINI S AVA 405 — the buy failed with
    "insufficient buying power" (2.91 SEK) because the sell hadn't settled. Plan
    swaps with time buffer, or do them sequentially with a wait.

31. **MINI S Silver instruments exist on Avanza.** Found 30+ variants via Playwright
    browser search. Key ones:
    - MINI S SILVER AVA 409 (2374804): ~10x leverage, barrier $80.97
    - MINI S SILVER AVA 405 (2367822): ~6x leverage, barrier $85.94
    - MINI S SILVER AVA 414 (2374783): higher leverage, closer barrier
    Use Playwright search (type in search box), NOT API endpoints, to find them.
    API search returns 404 but the browser search works perfectly.

32. **Choose barrier distance based on today's volatility.** If silver moved +8%
    today, an instrument with 8.2% barrier is RISKY for overnight hold. Match
    barrier distance to at least 1.5x the day's range. AVA 405 (14.3% barrier)
    is safer than AVA 409 (8.2%) after a volatile day. General rule:
    - Low vol day (<3%): 8-10% barrier OK
    - Normal day (3-5%): 12-15% barrier preferred
    - High vol day (>5%): 15%+ barrier or skip overnight

33. **When time is critical, ACT on existing data.** Don't search for news or
    additional confirmation when you have 10 minutes to market close. The signals
    said 4-0 bearish — that's enough to act. Web searching cost us the swap to
    a safer instrument. Pre-compute decisions BEFORE the exit window.

34. **The late session (18:45-21:55 CEST) is mostly flat.** Data from 29 days:
    - 48% flat, 28% up, 24% down
    - Average move: -0.07% (essentially zero)
    - Biggest crash: -2.78% (Mar 20)
    - Don't expect heroic recoveries in the last 3 hours. If you're underwater
      at 18:00, you'll likely exit underwater.

35. **Day results tracking.** On 2026-03-31:
    - BEAR SILVER X5 AVA 13: bought 23.80, sold 19.83 = **-254 SEK (-16.7%)**
    - Overnight: MINI S SILVER AVA 409 (180u @ 7.20, short, ~10x leverage)
    - Root cause: regime detection failure (applied dump pattern to rally day)
    - What would have saved it: TP0 at +5% (would have netted +76 SEK on 20%
      of position at 12:40) + faster recognition that thesis was broken

36. **Signal credibility ranking for silver (updated 2026-03-31):**
    - TRUST (>70% accuracy, >100 samples): econ_calendar (94.7%), fear_greed
      (89.1%), claude_fundamental (81.7%), momentum_factors (76.5%), structure (76.2%)
    - IGNORE (<50% or <100 samples): oscillators (40.1%), ministral (18.9%),
      sentiment (4.5%). These are noise, not signals.
    - FOR INTRADAY: only 3h focus probability and Chronos 1h are timeframe-relevant.
      Everything else predicts 1d+ horizons.

37. **The overnight MINI play is the second leg.** When intraday goes wrong, the
    overnight MINI (L or S, based on direction signals) can recover some losses.
    On 2026-03-31: 3d signals were 74% DOWN, so MINI S was chosen. The sequential
    play is: intraday cert (high leverage, exit same day) + overnight MINI (low
    leverage, ride the trend). Don't skip the second leg because the first leg lost.

38. **Avanza instrument search: use Playwright, not API.** The API search endpoints
    return 404 for warrants. But typing in the browser search box (keyboard '/')
    returns results immediately. Pattern:
    ```python
    page.keyboard.press('/')
    page.keyboard.type('MINI S SILVER AVA', delay=50)
    time.sleep(3)
    # Parse results from page.content()
    ```
    This found 30+ instruments in seconds. Document all IDs for future use.

39. **MC whipsaws in ranging markets.** P(up) swings 80%→26% in one 60s cycle when
    signal votes are close (5B/4S vs 4B/5S = one vote flip changes drift direction).
    Fix: require MC to hold threshold for 2 consecutive checks before acting.

40. **Auto-fish needs heartbeat kill-switch.** If monitoring agent disconnects, script
    runs blind. Heartbeat file touched every 30s; if stale >120s, emergency action.
    Rule: sell if profitable, hold if negative (don't force-sell at a loss on disconnect).

41. **4-gate entry rule for ranging markets.** Main loop action + metals loop action +
    MC confirms (>75%/<25% stable 2x) + RSI confirms (<50 for LONG, >50 for SHORT).
    All 4 must agree. Single-gate entries (just MC) cause whipsaw losses.

42. **Read ALL data sources in monitor.** Main loop: 30-signal consensus, MC, focus
    probs. Metals loop: Chronos/Ministral LLM predictions. Full summary: Fibonacci,
    pivots, Keltner levels. Reading everything gives the full picture for intelligent
    entry/exit decisions instead of blind price watching.

43. **Document lessons incrementally.** Don't wait until end of session. If agent dies
    or context resets, all session knowledge is lost. Write to FISHING_PATTERNS.md and
    session JSON files after each significant event.

44-55. See `data/fishing_session_20260401.json` for lessons on: combined RSI+MC exit
    (#45), RSI entry gate removal (#46), TURBO instruments (#47), faster flip after
    high-conviction exit (#48), better BULL cert (#49), news/events must be checked
    (#50-54), timeframe-matched signal usage (#55).

56. **FOLLOW YOUR OWN PLAYBOOK.** We wrote Liberation Day rules (no overnight, wider
    stops) and ignored them. Cost: -620 SEK. When a dated playbook exists for a known
    event, it overrides signal consensus. Signals don't know about events.

57. **Crash days close near the low.** Only 22% average recovery from intraday low
    across 24 crash days (>3% drop). Don't buy the dip on the crash day itself.
    67% bounce probability on the NEXT day, avg +1.7%.

58. **Thursday is silver's worst day.** Avg -0.88%, down 55% of the time. Friday
    after Thursday crash bounces 63%, avg +2.5%. Tradeable pattern with half-size.

59. **Liberation Day exception.** Apr 2025: Thu -7.7% then Fri -8.6%. If the event
    has new information (tariff announcement), the crash continues. Check for actual
    new announcements vs just anniversary noise before trading the Friday bounce.

60. **No TURBO/MINI with barriers <15% on event days.** TURBO L AVA 491 (barrier $68)
    got knocked out when silver dropped to $69.66. Use no-knockout BULL/BEAR X5 only.

61. **No loop limits on monitor.** Run until session end (21:45). Use `while True` with
    session-end check, not `while ck < 300`. If we disconnect, heartbeat detects it.

62. **Use script files, not inline `-c` code.** Background inline scripts die silently
    (stdout buffering, signal 144). Use proper script files that log to disk and start
    via Windows `Start-Process` to survive shell kills.
