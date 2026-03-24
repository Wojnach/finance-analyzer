Combined deep metals analysis — XAU-USD (gold) AND XAG-USD (silver) in a single pass with shared data reads.

Both metals share macro context, Avanza positions, G/S ratio, and real yield analysis. This command reads all shared data ONCE, then produces per-metal analysis sections. More efficient and more insightful than running /fin-gold and /fin-silver separately because cross-asset correlations are analyzed together.

## Prerequisites
- Read `memory/trading_rules.md` FIRST — check per-ticker accuracy rules for BOTH XAU-USD and XAG-USD before any recommendation.

## Steps

1. **Check the time** — Run `powershell.exe -NoProfile -Command "Get-Date -Format 'dddd yyyy-MM-dd HH:mm:ss'"`
   Record this as `start_time` for execution timing.
   Avanza commodity warrant hours: **08:15-21:55 CET** (NOT 17:25).
   Gold and silver trade 24/7 on spot markets. COMEX: 00:00-22:00 CET with 1h break.

1b. **Recall prior verdicts** — Read `data/fin_command_log.jsonl` and extract the last 3 entries covering XAU-USD or XAG-USD:
   ```python
   import json
   entries = []
   with open("data/fin_command_log.jsonl", encoding="utf-8") as f:
       for line in f:
           line = line.strip()
           if not line: continue
           try:
               e = json.loads(line)
               tickers = e.get("tickers", [])
               ticker = e.get("ticker", "")
               if ticker in ("XAU-USD", "XAG-USD") or "XAU-USD" in tickers or "XAG-USD" in tickers:
                   entries.append(e)
           except: pass
   for e in entries[-3:]:
       print(json.dumps(e, indent=2))
   ```
   For each prior entry, compute reflection by comparing prior prices against current:
   - For single-ticker entries (fin-gold/fin-silver): compare `price_usd` vs current price
   - For multi-ticker entries (fin-goldsilver): compare `gold.price_usd` and `silver.price_usd` vs current
   - If `verdict_correct_1d` or `outcome_1d_pct` exists, note the scored result
   - Include a **Prior Verdict Reflection** section early in your output:
     ```
     ### Prior Verdict Reflection
     {date} ({command}): Gold — said {verdict} at ${price} (conf {conf}) → now ${current} ({pct}%) — {RIGHT/WRONG}
     {date} ({command}): Silver — said {verdict} at ${price} (conf {conf}) → now ${current} ({pct}%) — {RIGHT/WRONG}
     ```
   - If `verdict_correct_1d` is populated, append `(scored: {correct/incorrect} at 1d)`
   - Use this reflection to calibrate current verdicts — if you've been consistently wrong on a metal, note it and adjust confidence down
   - If no prior entries exist, skip this section

2. **Ensure fresh precomputed context** — Check BOTH `data/gold_deep_context.json` AND `data/silver_deep_context.json`:
   - Read both files and parse their `generated_at` timestamps
   - Compute age for each: `(now - generated_at)` in hours
   - **If EITHER file is missing OR older than 2 hours**, refresh BOTH by running:
     ```
     .venv/Scripts/python.exe portfolio/metals_precompute.py
     ```
     This takes ~15 seconds and fetches fresh futures prices, ETF data, COT, FRED yields for both metals.
     After it completes, re-read BOTH `data/gold_deep_context.json` and `data/silver_deep_context.json`.
   - **If both files are fresh (< 2 hours old)**, use them directly — no refresh needed.
   - Print the data age for each: `"Gold context: {age} min old | Silver context: {age} min old"` (or `"freshly generated"` if just refreshed)

   Gold context contains: analyst targets, central bank buying, real yields, COT positioning,
   GLD ETF flows, G/S ratio history, signal accuracy, price trajectory.

   Silver context contains: analyst targets, supply/demand, COT positioning (with weekly trend),
   FRED real yields, G/S ratio history, prophecy snapshot, signal accuracy, price trajectory.

2b. **Ensure fresh learned lessons** — Check `data/system_lessons.json`:
   - If the file is missing OR older than 2 hours, refresh by running:
     ```
     .venv/Scripts/python.exe portfolio/fin_evolve.py
     ```
     This scores past verdicts against actual outcomes and generates calibration advice.
   - If fresh, use directly.

   This file is auto-generated every 2h by the evolution engine.
   It contains unified accuracy from ALL prediction sources (Layer 2 journal + /fin commands):
   - Your accuracy by source (layer2 vs fin-silver vs fin-gold vs fin-goldsilver), regime, confidence level
   - Per-ticker accuracy across ALL sources (BTC-USD, ETH-USD, XAG-USD, XAU-USD, etc.)
   - Anti-patterns (conditions where past verdicts were wrong)
   - Confirmed patterns (conditions where past verdicts were right)
   - Cross-asset patterns (e.g., "when gold bullish, silver follows X%")
   - Calibration advice (are you over/underconfident?)

   **Apply these adjustments to BOTH metals:**
   - If calibration says OVERCONFIDENT, reduce all verdict confidence by the suggested amount
   - If an anti-pattern matches current conditions, note it in the bear case and reduce confidence
   - If a confirmed pattern matches, note it in the bull case and increase confidence
   - If accuracy for current regime is < 0.5, add a disclaimer: "Note: past verdicts in {regime} regime have been weak ({accuracy}%)"
   - Check cross-asset patterns specifically — these are most valuable in a combined analysis

   If the file doesn't exist or has < 5 verdicts, skip this step (not enough data yet).

3. **Read live data** (parallel, ONCE for both metals):
   - `data/agent_summary_compact.json` — BOTH XAU-USD and XAG-USD sections: signals, prices,
     probabilities, regime, cumulative gains, forecast signals, Monte Carlo, signal_reliability.
     Also read the `macro` section (DXY, yields, FOMC) — shared between both metals.
   - `data/prophecy.json` — check for gold beliefs AND `silver_bull_2026` belief (target, checkpoints, conviction)
   - `data/silver_research.md` — ongoing silver thesis and catalysts (if exists)
   - `data/silver_analysis.json` — current silver technical snapshot (if exists)
   - `data/metals_positions_state.json` — position state for ALL metals
   - Last 5 entries from `data/layer2_journal.jsonl` that mention XAU-USD OR XAG-USD
     (check `tickers` and `prices` keys — a single journal entry may mention both)

4. **Fetch LIVE Avanza positions** (ONCE — returns ALL metals) — Run:
   ```
   .venv/Scripts/python.exe scripts/avanza_metals_check.py
   ```
   This hits the live Avanza API and returns all metals-related positions with:
   - name, units, current value (SEK), acquired value, P&L (SEK + %), today's change
   - Separates `active` positions from `knocked_out` ones
   - Covers BOTH gold and silver warrants in a single API call

   **If Avanza fails** (session expired, API error, script crashes):
   - Print: "Avanza unavailable — using fallback data"
   - Fall back to `data/metals_positions_state.json` for position info (units, entry price)
   - Use the precomputed `futures_context` from gold/silver_deep_context.json for current underlying prices
   - Estimate warrant P&L from underlying price change: `pnl_pct = (current_underlying / entry_underlying - 1) * leverage * 100`
   - Note clearly that P&L is estimated, not live
   - Tell user: "For live data, run `python scripts/avanza_login.py` to refresh session"

5. **Compute shared derived metrics** (ONCE, used by both metals):
   - **Gold/silver ratio:** XAU price / XAG price.
     Compare to: historical avg (60-70 modern era), 2011 extreme low (32), current bull market range, 2020 COVID peak (124).
     If ratio > 80: "Silver deeply undervalued vs gold — mean-reversion setup."
     If ratio 60-80: "Normal range — watch for directional compression."
     If ratio < 50: "Silver catching up fast — gold premium shrinking."
     Compute direction: is ratio expanding (gold outperforming = flight to safety) or compressing (silver catching up = risk-on)?
   - **Real yield proxy:** 10Y yield - headline CPI (from macro section or FRED data in precompute).
     Negative real yields = bullish for both metals (gold more sensitive).
     Rising real yields = headwind for both (gold more sensitive).
   - **DXY inverse correlation:** DXY up = metals headwind. DXY down = metals tailwind.
     Compute current direction and 5d change.
   - **FOMC proximity:** Days until next FOMC. Flag if within 4 days.
   - **Reflection:** Compare previous journal XAU and XAG prices vs current. Were the theses right?

6. **Per-metal analysis** — For EACH metal (gold first, then silver), compute:
   - **Signal consensus and heatmap** from agent_summary_compact.json
   - **Fibonacci levels** from price trajectory (3mo high/low for gold, 7d high/low for silver)
   - **Signal accuracy ranking** from signal_reliability for the specific ticker
   - **Chronos forecast** direction and confidence
   - **Monte Carlo** probability of up and confidence intervals

   **Gold-specific analysis:**
   - Cross-reference with analyst targets (GS, JPM, BofA)
   - Central bank buying pace — structural floor
   - GLD ETF flows — confirming or diverging from price action?
   - COT positioning — crowded or room to add?
   - Real yield is the #1 gold driver — always compute and display
   - Distance from ATH — discovery mode or pullback?

   **Silver-specific analysis:**
   - Prophecy progress: distance to $120 target, checkpoints triggered
   - Supply deficit narrative — confirmed by price action?
   - Industrial demand catalysts (solar, EVs, electronics)
   - SLV ETF flows and physical premiums (Tokyo, Dubai)
   - COT positioning (specs at lows = room, crowded = risk)
   - CME margin hike risk if price near recent highs ($100+)
   - COMEX registered inventory trend

7. **Run adversarial debate** — MANDATORY for EACH metal separately:

   **Gold debate:**
   - **Bull:** Central bank buying, de-dollarization, fiscal deficits, geopolitical risk, real yield decline
   - **Bear:** DXY strength, rising real yields, profit-taking from ATH, crowded positioning, Fed hawkish
   - **Synthesis:** Which side dominates, at each horizon?

   **Silver debate:**
   - **Bull:** Signal consensus, supply deficit, industrial demand growth, prophecy alignment, G/S ratio compression
   - **Bear:** CME margin risk, DXY headwinds, exhaustion signals, overbought RSI, industrial slowdown
   - **Synthesis:** Which side dominates, at each horizon?

8. **Produce combined output** in this exact format:

```
# METALS DEEP ANALYSIS — {date} {time} CET

## Shared Context

**Macro:** DXY {val} ({5d_chg}%) | 10Y {yield}% ({trend}) | Real yield ~{pct}% | Fear & Greed {val} | FOMC {days}d
**G/S Ratio:** {current}:1 ({compressing/expanding}, {direction} vs 30d avg) — historical avg 60-70, 2011 low 32
  {if expanding: "Gold outperforming — flight to safety mode"}
  {if compressing: "Silver catching up — risk-on, G/S mean-reversion in play"}
**Real Yield:** {10Y - CPI}% ({falling/rising}) — {bullish/bearish} for metals ({gold more/less sensitive})
**DXY:** {val} ({direction}) — {tailwind/headwind} for metals

### Avanza Positions (ALL metals, LIVE)
{for each active position (gold AND silver):}
  {NAME}
  {units} units | {value_sek} SEK | P&L: {profit_sek} SEK ({profit_pct}%)
  Today: {change_today_pct}% | Price: {last_price} SEK
  {if barrier known: Barrier: ${barrier} ({distance}% away)}
{if no positions: No active metals positions on Avanza.}

### Learned Lessons (if available)
{if system_lessons.json has relevant data:}
- Calibration: {over/underconfident by X}
- XAU accuracy: {pct}% ({n} verdicts) | XAG accuracy: {pct}% ({n} verdicts)
- Matching patterns: {any anti-patterns or confirmed patterns matching current conditions}
- Cross-asset: {any cross-asset pattern, e.g., "gold bullish -> silver follows 80% of time"}

---

## GOLD — XAU-USD ${price}

### Signals
{consensus} ({XB/YS/ZH}) | wConf {pct}% | Regime: {regime}
Heatmap: {Now->6mo B/S/. chars}
RSI {rsi} | MACD {macd} ({trend}) | BB {position} | Vol {ratio}x | ATR {atr_pct}%
Chronos 24h: {pct}% ({accuracy}% acc, {n} sam)
Prob: up{3h}% 3h | up{1d}% 1d | up{3d}% 3d
Monte Carlo: P(up)={pct}% | 1d: ${lo}-${hi} | 3d: ${lo}-${hi}

### External Context (cached {gold_age})
- Analyst targets: GS ${target}, JPM ${target}, BofA ${target}
- Central bank buying: {tonnes/yr} tonnes ({trend} vs prior year)
- GLD ETF: ${price}, {1mo change}%, volume trend: {rising/stable/falling}
- COT: Non-commercial net {contracts} ({WoW change}), managed money net {contracts}
- Real yield impact: {10Y - CPI}% — {bullish/bearish interpretation for gold}
- Distance from ATH: ${ath} — {pct}% {above/below}

### Adversarial Debate
**Bull:** {1-2 sentences — central banks, de-dollarization, fiscal deficits, safe haven, real yield decline}
**Bear:** {1-2 sentences — DXY strength, real yields rising, profit-taking, crowded longs, Fed hawkish}
**Synthesis:** {1-2 sentences — which side dominates, at what horizon}

### Verdict
| Horizon | Bias | Confidence | Note |
|---------|------|------------|------|
| 1-3d | {bullish/bearish/neutral} | {0.0-1.0} | {reason} |
| 1-4w | {bullish/bearish/neutral} | {0.0-1.0} | {reason} |
| 1-3m | {bullish/bearish/neutral} | {0.0-1.0} | {reason} |
| 6-12m | {bullish/bearish/neutral} | {0.0-1.0} | {reason} |

### Key Levels
Support: ${s1}, ${s2} | Resistance: ${r1}, ${r2}
Fibonacci: 23.6% ${f1} | 38.2% ${f2} | 50% ${f3} | 61.8% ${f4}

---

## SILVER — XAG-USD ${price}

### Prophecy: $120 Target
Progress: {pct}% | Checkpoints: {n}/5
{checkpoint status summary — which triggered, which pending, with dates}

### Signals
{consensus} ({XB/YS/ZH}) | wConf {pct}% | Regime: {regime}
Heatmap: {Now->6mo B/S/. chars}
RSI {rsi} | MACD {macd} ({trend}) | BB {position} | Vol {ratio}x | ATR {atr_pct}%
Chronos 24h: {pct}% ({accuracy}% acc, {n} sam)
Prob: up{3h}% 3h | up{1d}% 1d | up{3d}% 3d
Monte Carlo: P(up)={pct}% | 1d: ${lo}-${hi} | 3d: ${lo}-${hi}

### External Context (cached {silver_age})
- Analyst consensus: ${lo}-${hi} (Citi ${target}, GS ${target}, JPM ${target})
- Supply deficit: {Moz} Moz (6th consecutive year)
- COMEX registered: {Moz} Moz (down {pct}% from 2020)
- COT: Specs at {n}-year low, managed funds {net_long}K contracts
- Physical premiums: Tokyo {pct}%, Dubai {pct}%
- Industrial demand: solar {trend}, EV {trend}, electronics {trend}
- CME margin risk: {current margin level, recent hike history if near highs}
- G/S ratio: {current} vs {historical_avg} historical -> implies ${target} at current gold price

### Adversarial Debate
**Bull:** {1-2 sentences — signal consensus, supply deficit, industrial demand, prophecy alignment, G/S compression}
**Bear:** {1-2 sentences — CME margins, DXY, exhaustion, RSI overbought, industrial slowdown}
**Synthesis:** {1-2 sentences — which side dominates, at what horizon}

### Verdict
| Horizon | Bias | Confidence | Note |
|---------|------|------------|------|
| 1-3d | {bullish/bearish/neutral} | {0.0-1.0} | {reason} |
| 1-4w | {bullish/bearish/neutral} | {0.0-1.0} | {reason} |
| 1-3m | {bullish/bearish/neutral} | {0.0-1.0} | {reason} |
| 6-12m | {bullish/bearish/neutral} | {0.0-1.0} | {reason} |

### Key Levels
Support: ${s1}, ${s2} | Resistance: ${r1}, ${r2} | Prophecy target: $120
Fibonacci: 23.6% ${f1} | 38.2% ${f2} | 50% ${f3} | 61.8% ${f4}

---

## Cross-Asset Summary

**G/S Ratio Implications:**
- Current: {ratio}:1 | Direction: {compressing/expanding}
- If gold at ${gold_price} and ratio compresses to 50:1 -> silver = ${implied_silver_at_50}
- If gold at ${gold_price} and ratio compresses to 60:1 -> silver = ${implied_silver_at_60}
- G/S ratio verdict: {which metal has more relative upside right now}

**Which metal looks better right now:**
- Short-term (1-3d): {gold/silver/equal} — {reason}
- Medium-term (1-4w): {gold/silver/equal} — {reason}
- Structural (1-12m): {gold/silver/equal} — {reason}

**Combined Warrant P&L:**
- Gold warrants: {total_value} SEK, P&L: {total_pnl} SEK ({pct}%)
- Silver warrants: {total_value} SEK, P&L: {total_pnl} SEK ({pct}%)
- Total metals: {combined_value} SEK, P&L: {combined_pnl} SEK ({combined_pct}%)
{if no positions: No active warrant positions.}

**Reflection:**
{1-2 sentences comparing your last journal theses on XAU and XAG vs what actually happened}
```

## 9. Log the invocation

After producing the output, append a SINGLE log entry to `data/fin_command_log.jsonl` with data for both metals:
```python
import json, datetime, pathlib
entry = {
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "command": "fin-goldsilver",
    "tickers": ["XAU-USD", "XAG-USD"],
    "gs_ratio": <computed G/S ratio>,
    "real_yield": <10Y - CPI from FRED>,
    "dxy": <DXY value>,
    "gold": {
        "price_usd": <current XAU price>,
        "signal_consensus": "<BUY/SELL/HOLD>",
        "vote_breakdown": "<XB/YS/ZH>",
        "weighted_confidence": <0.0-1.0>,
        "regime": "<regime>",
        "rsi": <rsi value>,
        "chronos_24h_pct": <chronos prediction %>,
        "chronos_accuracy": <chronos accuracy for XAU>,
        "prob_3h": <directional probability 3h>,
        "prob_1d": <directional probability 1d>,
        "prob_3d": <directional probability 3d>,
        "monte_carlo_p_up": <probability of up>,
        "verdict_1_3d": "<bullish/bearish/neutral>",
        "verdict_1_3d_conf": <0.0-1.0>,
        "verdict_1_4w": "<bullish/bearish/neutral>",
        "verdict_1_4w_conf": <0.0-1.0>,
    },
    "silver": {
        "price_usd": <current XAG price>,
        "signal_consensus": "<BUY/SELL/HOLD>",
        "vote_breakdown": "<XB/YS/ZH>",
        "weighted_confidence": <0.0-1.0>,
        "regime": "<regime>",
        "rsi": <rsi value>,
        "chronos_24h_pct": <chronos prediction %>,
        "chronos_accuracy": <chronos accuracy for XAG>,
        "prob_3h": <directional probability 3h>,
        "prob_1d": <directional probability 1d>,
        "prob_3d": <directional probability 3d>,
        "monte_carlo_p_up": <probability of up>,
        "verdict_1_3d": "<bullish/bearish/neutral>",
        "verdict_1_3d_conf": <0.0-1.0>,
        "verdict_1_4w": "<bullish/bearish/neutral>",
        "verdict_1_4w_conf": <0.0-1.0>,
    },
    "precompute_age_hours": {
        "gold": <hours since gold_deep_context.json generated>,
        "silver": <hours since silver_deep_context.json generated>,
    },
    "avanza_available": <true/false>,
    "data_sources_used": ["agent_summary", "precompute_gold", "precompute_silver", "prophecy", ...],
    "execution_time_sec": <wall clock seconds from step 1 to this step>,
}
with open("data/fin_command_log.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

**Timing:** Record the start time at step 1 (when you check the clock). Compute `execution_time_sec`
as the difference between now and step 1 start time.

**Why log:** Builds a dataset of every combined analysis for tracking:
- Verdict accuracy for BOTH metals (compare with actual price moves)
- G/S ratio prediction accuracy over time
- Which metal the "looks better" call was right about
- Data availability patterns
- Execution time vs running /fin-gold + /fin-silver separately

## Critical rules

### Shared rules (apply to both metals)
- **Adversarial debate is MANDATORY** for EACH metal separately — never skip, even when signals are unanimous.
- **G/S ratio analysis is MANDATORY** — computed once, referenced in both metals' sections and the cross-asset summary.
- **LIVE Avanza positions are source of truth** — ignore state files if they conflict.
- **User prefers 5x leverage, not 10x.** Does NOT want to hold warrants overnight.
- **Warrant scalps:** 3-5h max hold, +2% underlying take-profit, -2% hard stop.
- **FOMC within 4 days:** Flag prominently in the shared context header. Reduce short-term (1-3d) conviction by at least 0.2 for BOTH metals. Note: "FOMC proximity — reduce position sizing, widen stops."
- **DXY inverse correlation is strong** for both metals. Always compute direction and implication.
- **Trust signals with >70% accuracy** for each ticker. Treat <50% accuracy signals as inverted.
- If precomputed context is missing for either metal, note it and proceed with live data only.

### Gold-specific rules
- **Gold is the macro anchor.** Its behavior tells you about risk appetite, dollar confidence, and central bank policy direction. Analyze it as a macro instrument, not just a commodity.
- **Real yields are the #1 gold driver.** When 10Y yield minus CPI is falling, gold is bullish. When rising, gold is bearish. Always compute and display this.
- **Central bank buying is structural.** Record pace since 2022. This is a multi-year floor, not a cyclical trade. Note it in the bull case every time.
- **Distance from ATH matters** — is gold in price discovery mode or pulling back from highs?

### Silver-specific rules
- **Separate long-term thesis from short-term signal noise** — ALWAYS. The $120 prophecy is a multi-month view. Short-term signals may contradict it without invalidating it.
- **Silver is the PRIMARY instrument.** This is the user's highest-conviction trade ($120 target, 0.8 conviction).
- **CME margin hike risk:** If silver price is near recent highs ($100+), always mention margin hike risk in the bear case. Historical: Jan 2026 +47%, Feb 2026 +15-18%, caused a 31% crash from $121.
- **Industrial demand is silver's unique driver** (unlike gold). Solar panel demand, EV wiring, electronics — structural growth catalysts.
- **Silver is more volatile than gold.** ATR% is typically 1.5-2x gold's. Position sizing should reflect this.

### Cross-asset rules
- **If gold is strong but silver is lagging**, that is often bullish for silver catch-up (G/S ratio compression).
- **If silver is outperforming gold** (G/S ratio compressing), that signals risk-on appetite — the move may have legs.
- **If both metals are weak**, check DXY — dollar strength is likely the common cause.
- **If gold is weak but silver holds**, check industrial demand catalysts — silver may be decoupling on its own fundamentals.
- **Always state which metal looks better** at each time horizon in the cross-asset summary. The user needs to allocate between gold and silver warrants.
