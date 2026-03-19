Deep oil analysis — WTI (CL=F) and Brent (BZ=F) with precomputed quant signals, term structure, COT positioning, event calendar, and multi-horizon verdict.

Oil is a term-structure market driven by OPEC+ policy, inventory cycles, and macro (DXY/yields). Not currently a Tier 1 tracked instrument — this skill provides standalone deep analysis.

## Prerequisites
- Read `memory/trading_rules.md` FIRST — check per-ticker accuracy rules before any recommendation.

## Steps

1. **Check the time** — Run `powershell.exe -NoProfile -Command "Get-Date -Format 'dddd yyyy-MM-dd HH:mm:ss'"`
   WTI electronic trading: nearly 24h (Sun 5PM - Fri 4PM CT). NYMEX pit: 9:00-14:30 ET.
   Avanza oil warrants (if any): **08:15-21:55 CET**.

1b. **Recall prior verdicts** — Read `data/fin_command_log.jsonl` and extract the last 3 entries covering CL-USD:
   ```python
   import json
   entries = []
   with open("data/fin_command_log.jsonl", encoding="utf-8") as f:
       for line in f:
           line = line.strip()
           if not line: continue
           try:
               e = json.loads(line)
               if e.get("ticker") == "CL-USD" or "CL-USD" in e.get("tickers", []):
                   entries.append(e)
           except: pass
   for e in entries[-3:]:
       print(json.dumps(e, indent=2))
   ```
   For each prior entry, compare prior price against current:
   - Use `price_usd` for WTI price, `brent_price_usd` for Brent
   - If `verdict_correct_1d` or `outcome_1d_pct` exists, note the scored result
   - Include a **Prior Verdict Reflection** section early in your output:
     ```
     ### Prior Verdict Reflection
     {date} ({command}): Said {verdict} at ${price} (conf {conf}) → now ${current} ({pct}%) — {RIGHT/WRONG}
     ```
   - Use this to calibrate current verdicts — if you've been consistently wrong, adjust confidence down
   - If no prior entries exist, skip this section

2. **Ensure fresh precomputed context** — Check `data/oil_deep_context.json`:
   - Read the file and parse `generated_at` timestamp
   - Compute age: `(now - generated_at)` in hours
   - **If the file is missing OR older than 2 hours**, refresh it by running:
     ```
     .venv/Scripts/python.exe portfolio/oil_precompute.py
     ```
     This takes ~20 seconds and fetches WTI/Brent futures, OVX, COT, FRED, USO, crack spread.
     After it completes, re-read `data/oil_deep_context.json` for the fresh data.
   - **If the file is fresh (< 2 hours old)**, use it directly — no refresh needed.
   - Print the data age: `"Precomputed context: {age} min old"` (or `"freshly generated"` if just refreshed)

   The file contains: TSMOM signals, Donchian channels, MA crossovers, realised vol, OVX,
   IV-RV spread, Brent-WTI spread, curve slope, crack spread, COT positioning, FRED macro,
   event calendar, seed research (supply/demand, OPEC, seasonal patterns).

2b. **Ensure fresh learned lessons** — Check `data/system_lessons.json`:
   - If the file is missing OR older than 2 hours, refresh by running:
     ```
     .venv/Scripts/python.exe portfolio/fin_evolve.py
     ```
     This scores past verdicts against actual outcomes and generates calibration advice.
   - If fresh, use directly.

   This file is auto-generated every 2h by the evolution engine.
   It contains unified accuracy from ALL prediction sources:
   - Your accuracy by source, regime, confidence level
   - Per-ticker accuracy (including CL-USD if prior verdicts exist)
   - Anti-patterns and confirmed patterns
   - Calibration advice

   **Apply these adjustments:**
   - If calibration says OVERCONFIDENT, reduce all verdict confidence by the suggested amount
   - If an anti-pattern matches current conditions, note it and reduce confidence
   - If accuracy for current regime is < 0.5, add a disclaimer
   - If the file doesn't exist or has < 5 verdicts, skip this step

3. **Read live data** (parallel):
   - `data/agent_summary_compact.json` — macro section: DXY, yields, FOMC calendar, VIX
   - `data/oil_deep_context.json` — the full precomputed oil context

4. **Avanza positions** — Oil warrants are not currently tracked. Skip this step.
   If the user adds oil warrants in the future, follow the same Avanza check pattern as fin-gold/fin-silver.

5. **Compute derived metrics** from precomputed data:
   - **TSMOM aggregate**: Count signals across 21d/63d/126d horizons. Majority positive = bullish, majority negative = bearish.
   - **Regime**: Classify from OVX + RV:
     - OVX > 40 OR RV_20d > 40: "crisis"
     - OVX > 30 OR RV_20d > 30: "high-vol"
     - OVX < 20 AND RV_20d < 20: "low-vol"
     - else: "normal"
   - **Event proximity risk**: If EIA within 2 days, FOMC/OPEC within 4 days → flag and reduce 1-3d confidence by 0.2
   - **Vol targeting context**: If RV_20d > 35%, note "high volatility — reduce position sizing"
   - **Crowding risk**: COT z-score > 1.5 or < -1.5 → contrarian flag
   - **Crack spread regime**: Widening (>$25) = refinery demand pull; narrowing (<$10) = demand weakness
   - **Curve structure**: Brent premium > $5 = supply tightness; negative spread = unusual
   - **Donchian confluence**: Both 20d and 55d signalling same direction = strong trend confirmation

6. **Cross-reference** — Oil-specific questions to answer:
   - Is OPEC+ maintaining discipline? (compliance vs quotas from seed research)
   - Is US shale responding to prices? (production trends)
   - Are inventories building or drawing? (EIA data if available)
   - Is DXY confirming or contradicting the oil move?
   - Is geopolitical risk premium justified or fading?
   - Is the driving season approaching? (seasonal bullish May-Sep)
   - What does the crack spread say about refining demand?
   - Is the curve in backwardation (supply tight) or contango (oversupply)?

7. **Run adversarial debate** — MANDATORY for every invocation:
   - **Bull case:** OPEC+ discipline, geopolitical risk, seasonal demand, inventory draws, weaker DXY, backwardation
   - **Bear case:** Demand destruction, OPEC cheating, US production surge, DXY strength, recession risk, contango
   - **Synthesis:** Weigh both sides — which dominates at each horizon?
   - **Event haircut:** If EIA/FOMC/OPEC within 4 days, note it and reduce 1-3d confidence by 0.2

8. **Produce output** in this exact format:

```
# OIL DEEP ANALYSIS — {date} {time} CET
**WTI: ${price} | Brent: ${brent} | Spread: ${spread} | OVX: {val} ({regime}) | FOMC: {days}d**

### Prior Verdict Reflection
{date}: Said {verdict} at ${price} (conf {conf}) → now ${current} ({pct}%) — {RIGHT/WRONG}
{If no prior entries: "No prior oil verdicts — first analysis."}

## Quant Signals (precomputed {age} ago)
TSMOM: 21d={sign} | 63d={sign} | 126d={sign} — aggregate: {bullish/bearish/neutral}
Donchian: 20d={signal} | 55d={signal}
MA Cross: 20/50={val}% | 50/200={val}%
RSI(14): {val} | RV(20d): {pct}% | RV(10d): {pct}% | RV(60d): {pct}%
OVX: {val} ({regime}) | IV-RV spread: {val} ({fear premium / complacency / neutral})
Volume ratio (5d/20d): {val}x

## Term Structure & Carry
Brent-WTI: ${spread} ({note}) — {implication}
Curve slope (log): {val} ({backwardation/contango proxy})
Carry (annualised): {val} ({note})
Crack spread: ${val}/bbl ({trend}) — {demand signal}

## Positioning (CFTC COT)
Report date: {date}
Non-commercial net: {contracts} (WoW change: {change})
Net/OI ratio: {pct}% | Z-score: {val} ({crowding assessment})
Managed money net: {contracts}
Trend (3wk): Non-comm {direction} | Managed money {direction}
Crowding: {low/medium/high} — {implication}

## Fundamentals (seed research)
OPEC+ production: {mbd} vs quota (compliance: {pct}%)
US production: {mbd} | SPR: {mb} barrels
Global demand: {mbd} ({growth note})
Supply-demand balance: {note}

## Event Calendar
Next EIA: {date} ({days}d, {time} ET) | Next OPEC: {date} ({days}d) | FOMC: {date} ({days}d)
{If any within 4 days: "**EVENT PROXIMITY WARNING** — reduce short-term confidence by 0.2"}

## USO ETF Flows
${price} | 1mo: {change}% | Volume trend: {rising/stable/falling}

## FRED Macro
WTI spot (FRED): ${val} | Brent spot (FRED): ${val}
10Y yield: {val}% ({direction}) | DXY: {val} ({chg}% 5d)

## Adversarial Debate
**Bull:** {2-3 sentences — OPEC discipline, geopolitical risk, seasonal demand, inventory draws, curve structure}
**Bear:** {2-3 sentences — demand destruction, shale surge, OPEC cheating, strong dollar, recession}
**Synthesis:** {2-3 sentences — which dominates at each horizon, event risks, vol context}

## Verdict
| Horizon | Bias | Confidence | Key Driver |
|---------|------|------------|------------|
| 1-3d | {bullish/bearish/neutral} | {0.0-1.0} | {TSMOM/event/momentum} |
| 1-4w | {bullish/bearish/neutral} | {0.0-1.0} | {carry/inventories/OPEC} |
| 1-3m | {bullish/bearish/neutral} | {0.0-1.0} | {term structure/demand} |
| 6-12m | {bullish/bearish/neutral} | {0.0-1.0} | {macro/structural} |

## Key Levels (WTI)
Support: ${s1}, ${s2} | Resistance: ${r1}, ${r2}
Fibonacci: 23.6% ${f1} | 38.2% ${f2} | 50% ${f3} | 61.8% ${f4}

## Risk Flags
{List any triggered:}
- {OVX > 40: "CRISIS VOL — extreme caution"}
- {RV > 35%: "High realised vol — reduce position sizing"}
- {COT z-score extreme: "Crowded positioning — contrarian risk"}
- {Data >24h stale: "STALE DATA WARNING — source X aged Yh"}
- {Event within 2d: "EVENT PROXIMITY — EIA/FOMC/OPEC imminent"}
- {Brent-WTI spread inverted: "Unusual spread inversion — investigate"}
- {Crack spread < $10: "Weak crack spread — demand concern"}
```

## 9. Log the invocation

After producing the output, append a log entry to `data/fin_command_log.jsonl`:
```python
import json, datetime, pathlib
entry = {
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "command": "fin-oil",
    "ticker": "CL-USD",
    "tickers": ["CL-USD", "BZ-USD"],
    "price_usd": <current WTI price>,
    "brent_price_usd": <current Brent price>,
    "brent_wti_spread": <spread>,
    "ovx": <OVX value>,
    "ovx_regime": "<crisis/high_vol/normal/low_vol>",
    "signal_tsmom_agg": "<bullish/bearish/neutral>",
    "donchian_20d": <signal>,
    "donchian_55d": <signal>,
    "rsi_14": <rsi value>,
    "rv_20d": <realised vol 20d>,
    "iv_rv_spread": <spread>,
    "crack_spread": <crack per bbl>,
    "cot_nc_net": <non-commercial net>,
    "cot_nc_zscore": <z-score>,
    "cot_crowding": "<low/medium/high>",
    "regime": "<crisis/high-vol/normal/low-vol>",
    "dxy": <DXY value from macro>,
    "treasury_10y": <10Y yield>,
    "verdict_1_3d": "<bullish/bearish/neutral>",
    "verdict_1_3d_conf": <0.0-1.0>,
    "verdict_1_4w": "<bullish/bearish/neutral>",
    "verdict_1_4w_conf": <0.0-1.0>,
    "verdict_1_3m": "<bullish/bearish/neutral>",
    "verdict_1_3m_conf": <0.0-1.0>,
    "precompute_age_hours": <hours since oil_deep_context.json was generated>,
    "data_sources_used": ["oil_precompute", "agent_summary_macro", ...],
    "execution_time_sec": <wall clock seconds from step 1 to this step>,
}
with open("data/fin_command_log.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

**Timing:** Record the start time at step 1 (when you check the clock). Compute `execution_time_sec`
as the difference between now and step 1 start time.

## Critical rules
- **Oil is a term structure market** — always analyze curve shape (Brent-WTI spread, carry),
  not just spot price. Backwardation = supply tight. Contango = oversupply.
- **EIA Wednesday 10:30 ET is the primary short-term catalyst.** Flag proximity prominently.
  EIA inventory surprises move WTI 1-3% within minutes.
- **OPEC decisions can move price 5-10% in hours.** Always check next meeting date and
  current compliance. OPEC+ is THE marginal supply controller.
- **April 2020 reminder:** Extreme storage/logistics caused negative WTI prices (-$37.63).
  Flag any curve stress or storage concerns. This is the tail risk.
- **DXY inverse correlation:** Oil is priced in USD; strong dollar = headwind.
  Always note DXY direction and implication.
- **Seasonal patterns are real:**
  - Driving season (May-Sep): bullish — US gasoline demand peaks
  - Refinery maintenance (Feb-Apr): bearish — lower crude processing
  - Hurricane season (Jun-Nov): supply disruption risk for Gulf production
- **Vol targeting context:** If RV_20d > 35%, note "high volatility — reduce position sizing."
  If OVX > 40, flag "crisis volatility."
- **Event haircuts:** Reduce 1-3d confidence by 0.2 within 4 days of EIA/FOMC/OPEC.
- **Circuit breaker flags:**
  - OVX > 40 → "CRISIS"
  - RV_20d > 99th percentile → "EXTREME VOL"
  - Data >24h stale → "STALE DATA"
  - Crack spread < $5 → "DEMAND DESTRUCTION SIGNAL"
- **Adversarial debate is MANDATORY** — never skip it, even when signals are unanimous.
- **TSMOM is the primary quant signal.** Multi-horizon momentum (21d/63d/126d) from
  the research document. Majority agreement across horizons = strong signal.
- **COT crowding is contrarian.** Extreme positioning (z-score > 1.5) often precedes reversals.
- Oil is NOT a tracked Tier 1 instrument — no signal_reliability or forecast data exists.
  Rely on the precomputed quant signals and fundamental analysis.
- If precomputed context is missing entirely, note it and proceed with macro data only.
