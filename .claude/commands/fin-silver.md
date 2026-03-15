Deep silver analysis — XAG-USD with precomputed external research, live signals, and multi-horizon verdict.

This is the PRIMARY instrument. Silver is the user's highest-conviction trade ($120 target).

## Prerequisites
- Read `memory/trading_rules.md` FIRST — check per-ticker accuracy rules before any recommendation.

## Steps

1. **Check the time** — Run `powershell.exe -NoProfile -Command "Get-Date -Format 'dddd yyyy-MM-dd HH:mm:ss'"`
   Avanza commodity warrant hours: **08:15-21:55 CET** (NOT 17:25).
   US futures (COMEX): 00:00-22:00 CET with 1h break. Spot XAG trades 24/7.

2. **Ensure fresh precomputed context** — Check `data/silver_deep_context.json`:
   - Read the file and parse `generated_at` timestamp
   - Compute age: `(now - generated_at)` in hours
   - **If the file is missing OR older than 2 hours**, refresh it by running:
     ```
     .venv/Scripts/python.exe portfolio/metals_precompute.py
     ```
     This takes ~15 seconds and fetches fresh futures prices, ETF data, COT, FRED yields.
     After it completes, re-read `data/silver_deep_context.json` for the fresh data.
   - **If the file is fresh (< 2 hours old)**, use it directly — no refresh needed.
   - Print the data age: `"Precomputed context: {age} min old"` (or `"freshly generated"` if just refreshed)

   The file contains: analyst targets, supply/demand, COT positioning (with weekly trend),
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
   - Your accuracy by source (layer2 vs fin-silver vs fin-gold), regime, confidence level
   - Per-ticker accuracy across ALL sources (BTC-USD, ETH-USD, XAG-USD, XAU-USD, etc.)
   - Anti-patterns (conditions where past verdicts were wrong)
   - Confirmed patterns (conditions where past verdicts were right)
   - Cross-asset patterns (e.g., "when gold bullish, silver follows X%")
   - Calibration advice (are you over/underconfident?)

   **Apply these adjustments:**
   - If calibration says OVERCONFIDENT, reduce all verdict confidence by the suggested amount
   - If an anti-pattern matches current conditions, note it in the bear case and reduce confidence
   - If a confirmed pattern matches, note it in the bull case and increase confidence
   - If accuracy for current regime is < 0.5, add a disclaimer: "Note: past verdicts in {regime} regime have been weak ({accuracy}%)"

   If the file doesn't exist or has < 5 verdicts, skip this step (not enough data yet).

3. **Read live data** (parallel):
   - `data/agent_summary_compact.json` — XAG-USD and XAU-USD sections: signals, prices, probabilities,
     regime, cumulative gains, forecast signals, Monte Carlo, signal_reliability
   - `data/prophecy.json` — `silver_bull_2026` belief (target, checkpoints, conviction)
   - `data/silver_research.md` — ongoing thesis and catalysts
   - `data/silver_analysis.json` — current technical snapshot (if exists, may not)
   - `data/metals_positions_state.json` — position state
   - Last 5 entries from `data/layer2_journal.jsonl` that mention XAG-USD (check `tickers` and `prices` keys)

4. **Fetch LIVE Avanza positions** — Run:
   ```
   .venv/Scripts/python.exe scripts/avanza_metals_check.py
   ```
   This hits the live Avanza API and returns all metals-related positions with:
   - name, units, current value (SEK), acquired value, P&L (SEK + %), today's change
   - Separates `active` positions from `knocked_out` ones

   **If Avanza fails** (session expired, API error, script crashes):
   - Print: "Avanza unavailable — using fallback data"
   - Fall back to `data/metals_positions_state.json` for position info (units, entry price)
   - Use the precomputed `futures_context` from `silver_deep_context.json` for current underlying price
   - Estimate warrant P&L from underlying price change: `pnl_pct ≈ (current_underlying / entry_underlying - 1) * leverage * 100`
   - Note clearly that P&L is estimated, not live
   - Tell user: "For live data, run `python scripts/avanza_login.py` to refresh session"

5. **Compute derived metrics** (from live data):
   - **Gold/silver ratio:** XAU price / XAG price (from agent_summary_compact.json prices).
     Compare to historical avg (30-35), current bull market range (60-70), and 2011 low (32).
     If ratio > 80: "Silver deeply undervalued vs gold". If ratio < 50: "Silver catching up fast".
   - **Fibonacci levels:** From 7d high/low in price trajectory (or from agent_summary indicators).
     Compute: 0% (low), 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100% (high) retracement levels.
   - **Distance to prophecy target:** (target - current) / current * 100
   - **Signal accuracy ranking:** From signal_reliability section for XAG-USD — which signals
     are most/least reliable for silver specifically
   - **Reflection:** Compare previous journal XAG prices vs current. Was the thesis right?

6. **Cross-reference precomputed external research with current technicals:**
   - Do analyst targets align with current momentum direction?
   - Is the supply deficit narrative being confirmed by price action?
   - Are physical market premiums (Tokyo, Dubai) widening or narrowing?
   - Is COT positioning bullish (specs at lows = room to add) or bearish (crowded)?
   - Is the gold/silver ratio compressing (bullish) or expanding (bearish)?

7. **Run adversarial debate** — MANDATORY for every invocation:
   - **Bull case:** Signal consensus, external research support, supply deficit, prophecy alignment
   - **Bear case:** CME margin risk, DXY headwinds, exhaustion signals, short-term overbought
   - **Synthesis:** Weigh both sides — which dominates at each time horizon?

8. **Produce output** in this exact format:

```
# SILVER DEEP ANALYSIS — {date} {time} CET
**XAG-USD: ${price} | G/S Ratio: {ratio}:1 | FOMC: {days}d**

## Signals
{consensus} ({XB/YS/ZH}) | wConf {pct}% | Regime: {regime}
Heatmap: {Now->6mo B/S/. chars}
RSI {rsi} | MACD {macd} ({trend}) | BB {position} | Vol {ratio}x | ATR {atr_pct}%
Chronos 24h: {pct}% ({accuracy}% acc, {n} sam)
Prob: up{3h}% 3h | up{1d}% 1d | up{3d}% 3d
Monte Carlo: P(up)={pct}% | 1d: ${lo}-${hi} | 3d: ${lo}-${hi}

## Prophecy: $120 Target
Progress: {pct}% | Checkpoints: {n}/5
{checkpoint status summary — which triggered, which pending, with dates}

## External Context (cached {age})
- Analyst consensus: ${lo}-${hi} (Citi ${target}, GS ${target}, JPM ${target})
- Supply deficit: {Moz} Moz (6th consecutive year)
- COMEX registered: {Moz} Moz (down {pct}% from 2020)
- COT: Specs at {n}-year low, managed funds {net_long}K contracts
- Physical premiums: Tokyo {pct}%, Dubai {pct}%
- G/S ratio: {current} vs {historical_avg} historical -> implies ${target} at current gold price

## Avanza Positions (LIVE)
{for each active position:}
  {NAME}
  {units} units | {value_sek} SEK | P&L: {profit_sek} SEK ({profit_pct}%)
  Today: {change_today_pct}% | Price: {last_price} SEK
  {if barrier known: Barrier: ${barrier} ({distance}% away)}
{if no positions: No active silver positions on Avanza.}

## Macro
DXY {val} ({chg}%) | 10Y {yield}% ({trend}) | F&G {val} | FOMC {days}d

## Adversarial Debate
**Bull:** {1-2 sentences — signal consensus, supply deficit, external research, prophecy}
**Bear:** {1-2 sentences — CME margins, DXY, exhaustion, RSI overbought, substitution}
**Synthesis:** {1-2 sentences — which side dominates, at what horizon}

## Verdict
| Horizon | Bias | Confidence | Note |
|---------|------|------------|------|
| 1-3d | {bullish/bearish/neutral} | {0.0-1.0} | {reason} |
| 1-4w | {bullish/bearish/neutral} | {0.0-1.0} | {reason} |
| 1-3m | {bullish/bearish/neutral} | {0.0-1.0} | {reason} |
| 6-12m | {bullish/bearish/neutral} | {0.0-1.0} | {reason} |

## Key Levels
Support: ${s1}, ${s2} | Resistance: ${r1}, ${r2} | Prophecy target: $120
Fibonacci: 23.6% ${f1} | 38.2% ${f2} | 50% ${f3} | 61.8% ${f4}
```

## 9. Log the invocation

After producing the output, append a log entry to `data/fin_command_log.jsonl`:
```python
import json, datetime, pathlib
entry = {
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "command": "fin-silver",
    "ticker": "XAG-USD",
    "price_usd": <current XAG price from agent_summary_compact>,
    "gs_ratio": <computed G/S ratio>,
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
    "precompute_age_hours": <hours since silver_deep_context.json was generated>,
    "avanza_available": <true/false>,
    "data_sources_used": ["agent_summary", "precompute", "prophecy", ...],
    "execution_time_sec": <wall clock seconds from step 1 to this step>,
}
with open("data/fin_command_log.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

**Timing:** Record the start time at step 1 (when you check the clock). Compute `execution_time_sec`
as the difference between now and step 1 start time. This tracks how long the full analysis takes.

**Why log:** Over time, this builds a dataset of every analysis — you can track:
- How often each verdict was correct (compare verdict_1_3d with actual price move 3 days later)
- Which data sources were available vs missing
- Whether precompute staleness correlates with analysis quality
- Average execution time and whether it's improving

## Critical rules
- **Separate long-term thesis from short-term signal noise** — ALWAYS. The $120 prophecy is a
  multi-month view. Short-term signals may contradict it without invalidating it.
- **Trust XAG signals with >70% accuracy** (historically 71-83%). Treat <50% accuracy signals as inverted.
- **FOMC within 4 days:** Flag it prominently in the output header. Reduce short-term (1-3d)
  conviction by at least 0.2. Note: "FOMC proximity — reduce position sizing, widen stops."
- **Gold/silver ratio comparison is MANDATORY** — compute from live XAU and XAG prices.
  Include both current ratio and historical context.
- **Adversarial debate is MANDATORY** — never skip it, even when signals are unanimous.
- **CME margin hike risk:** If price is near recent highs ($100+), always mention margin hike
  risk in the bear case. Jan 2026: +47%, Feb 2026: +15-18%, caused a 31% crash from $121.
- **LIVE Avanza positions are source of truth** — ignore state files if they conflict.
- User prefers 5x leverage, not 10x. Does NOT want to hold warrants overnight.
- Warrant scalps: 3-5h max hold, +2% underlying take-profit, -2% hard stop.
- If precomputed context is missing entirely, note it and proceed with live data only.
