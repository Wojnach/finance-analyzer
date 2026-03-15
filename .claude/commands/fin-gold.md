Deep gold analysis — XAU-USD with precomputed external research, live signals, and multi-horizon verdict.

Gold is the anchor asset — central bank reserve, inflation hedge, safe haven. Silver follows gold.

## Prerequisites
- Read `memory/trading_rules.md` FIRST — check per-ticker accuracy rules before any recommendation.

## Steps

1. **Check the time** — Run `powershell.exe -NoProfile -Command "Get-Date -Format 'dddd yyyy-MM-dd HH:mm:ss'"`
   Avanza commodity warrant hours: **08:15-21:55 CET** (NOT 17:25).
   Gold trades 24/7 on spot markets. COMEX: 00:00-22:00 CET with 1h break.

2. **Read precomputed context** — `data/gold_deep_context.json`
   Contains cached external research (analyst targets, central bank buying, real yields context,
   COT positioning, ETF flows, price trajectory, signal accuracy).
   Check `generated_at` — if older than 7 days, print:
   `WARNING: Precomputed context is STALE ({age} days old) — run: .venv/Scripts/python.exe portfolio/metals_precompute.py`

3. **Read live data** (parallel):
   - `data/agent_summary_compact.json` — XAU-USD and XAG-USD sections: signals, prices, probabilities,
     regime, cumulative gains, forecast signals, Monte Carlo, signal_reliability
   - `data/prophecy.json` — check for any gold beliefs
   - `data/metals_positions_state.json` — position state (gold warrants)
   - Last 5 entries from `data/layer2_journal.jsonl` that mention XAU-USD

4. **Fetch LIVE Avanza positions** — Run:
   ```
   .venv/Scripts/python.exe scripts/avanza_metals_check.py
   ```
   Same as fin-silver — returns all metals positions. Filter for gold-related ones.

   **If Avanza fails** (session expired, API error, script crashes):
   - Print: "Avanza unavailable — using fallback data"
   - Fall back to `data/metals_positions_state.json` for position info (units, entry price)
   - Use the precomputed `futures_context` from `gold_deep_context.json` for current underlying price
   - Estimate warrant P&L from underlying price change
   - Note clearly that P&L is estimated, not live
   - Tell user: "For live data, run `python scripts/avanza_login.py` to refresh session"

5. **Compute derived metrics** (from live data):
   - **Gold/silver ratio:** XAU price / XAG price. If ratio compressing → silver catching up (bullish for silver, neutral for gold). If expanding → gold outperforming (flight to safety).
   - **Real yield proxy:** 10Y yield - headline CPI (from macro section). Negative real yields = bullish for gold. Rising real yields = headwind.
   - **DXY inverse correlation:** DXY up = gold headwind. DXY down = gold tailwind. Compute current direction.
   - **Fibonacci levels:** From 3mo high/low in precomputed futures_context.
   - **Signal accuracy ranking:** From signal_reliability section for XAU-USD
   - **Distance from ATH:** Current vs recent high — is gold in discovery or pulling back?

6. **Cross-reference precomputed external research with current technicals:**
   - Are central banks still buying at record pace?
   - Is DXY direction confirming or contradicting the gold trend?
   - Are real yields falling (bullish) or rising (bearish)?
   - Is COT positioning crowded (contrarian risk) or light (room to add)?
   - Are GLD ETF flows confirming the move (inflows) or diverging (outflows)?
   - Is gold leading silver or vice versa? What does the G/S ratio trend say?

7. **Run adversarial debate** — MANDATORY for every invocation:
   - **Bull case:** Central bank buying, de-dollarization, fiscal deficits, geopolitical risk, real yield decline
   - **Bear case:** DXY strength, rising real yields, profit-taking from ATH, crowded positioning, Fed hawkish
   - **Synthesis:** Weigh both sides — which dominates at each time horizon?

8. **Produce output** in this exact format:

```
# GOLD DEEP ANALYSIS — {date} {time} CET
**XAU-USD: ${price} | G/S Ratio: {ratio}:1 | FOMC: {days}d**

## Signals
{consensus} ({XB/YS/ZH}) | wConf {pct}% | Regime: {regime}
Heatmap: {Now->6mo B/S/. chars}
RSI {rsi} | MACD {macd} ({trend}) | BB {position} | Vol {ratio}x | ATR {atr_pct}%
Chronos 24h: {pct}% ({accuracy}% acc, {n} sam)
Prob: up{3h}% 3h | up{1d}% 1d | up{3d}% 3d
Monte Carlo: P(up)={pct}% | 1d: ${lo}-${hi} | 3d: ${lo}-${hi}

## External Context (cached {age})
- Analyst targets: GS ${target}, JPM ${target}, BofA ${target}
- Central bank buying: {tonnes/yr} tonnes ({trend} vs prior year)
- GLD ETF: ${price}, {1mo change}%, volume trend: {rising/stable/falling}
- COT: Non-commercial net {contracts} ({WoW change}), managed money net {contracts}
- Real yield proxy: {10Y - CPI}% ({bullish/bearish} for gold)
- G/S ratio: {current}:1 — gold {leading/lagging} silver

## Avanza Positions (LIVE)
{for each active gold position:}
  {NAME}
  {units} units | {value_sek} SEK | P&L: {profit_sek} SEK ({profit_pct}%)
  Today: {change_today_pct}%
{if no positions: No active gold positions on Avanza.}

## Macro
DXY {val} ({chg}%) | 10Y {yield}% ({trend}) | Real yield ~{pct}% | F&G {val} | FOMC {days}d

## Adversarial Debate
**Bull:** {1-2 sentences — central banks, de-dollarization, fiscal deficits, safe haven}
**Bear:** {1-2 sentences — DXY strength, real yields rising, profit-taking, crowded longs}
**Synthesis:** {1-2 sentences — which side dominates, at what horizon}

## Verdict
| Horizon | Bias | Confidence | Note |
|---------|------|------------|------|
| 1-3d | {bullish/bearish/neutral} | {0.0-1.0} | {reason} |
| 1-4w | {bullish/bearish/neutral} | {0.0-1.0} | {reason} |
| 1-3m | {bullish/bearish/neutral} | {0.0-1.0} | {reason} |
| 6-12m | {bullish/bearish/neutral} | {0.0-1.0} | {reason} |

## Key Levels
Support: ${s1}, ${s2} | Resistance: ${r1}, ${r2}
Fibonacci: 23.6% ${f1} | 38.2% ${f2} | 50% ${f3} | 61.8% ${f4}

## Silver Cross-Reference
G/S ratio {current}:1 ({compressing/expanding} — {implication for silver})
If gold at ${gold_price} and ratio compresses to 50:1 → silver = ${implied}
```

## Critical rules
- **Gold is the macro anchor.** Its behavior tells you about risk appetite, dollar confidence,
  and central bank policy direction. Analyze it as a macro instrument, not just a commodity.
- **Real yields are the #1 driver.** When 10Y yield minus CPI is falling → gold bullish.
  When rising → gold bearish. Always compute and display this.
- **DXY inverse correlation is strong.** Always note DXY direction and its implication.
- **Central bank buying is structural.** Record pace since 2022. This is a multi-year floor,
  not a cyclical trade. Note it in the bull case every time.
- **Trust XAU signals with >70% accuracy.** Check signal_reliability for XAU-USD specifically.
- **FOMC within 4 days:** Flag prominently. Gold is hyper-sensitive to Fed communication.
- **Adversarial debate is MANDATORY.**
- **Always include G/S ratio analysis** — gold's relationship to silver matters for the user's
  silver thesis. If gold is strong but silver is lagging, that's bullish for silver catch-up.
- **LIVE Avanza positions are source of truth** — ignore state files if they conflict.
- User prefers 5x leverage, not 10x. Does NOT want to hold warrants overnight.
- If precomputed context is missing entirely, note it and proceed with live data only.
