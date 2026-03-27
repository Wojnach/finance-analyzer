Deep gold analysis — XAU-USD with precomputed external research, live signals, and multi-horizon verdict.

Gold is the anchor asset — central bank reserve, inflation hedge, safe haven. Silver follows gold.

## Prerequisites
- Read `memory/trading_rules.md` FIRST — check per-ticker accuracy rules before any recommendation.

## Steps

1. **Check the time** — Run `powershell.exe -NoProfile -Command "Get-Date -Format 'dddd yyyy-MM-dd HH:mm:ss'"`
   Avanza commodity warrant hours: **08:15-21:55 CET** (NOT 17:25).
   Gold trades 24/7 on spot markets. COMEX: 00:00-22:00 CET with 1h break.

1b. **Recall prior verdicts** — Read `data/fin_command_log.jsonl` and extract the last 3 entries covering XAU-USD:
   ```python
   import json
   entries = []
   with open("data/fin_command_log.jsonl", encoding="utf-8") as f:
       for line in f:
           line = line.strip()
           if not line: continue
           try:
               e = json.loads(line)
               if e.get("ticker") == "XAU-USD" or "XAU-USD" in e.get("tickers", []):
                   entries.append(e)
           except: pass
   for e in entries[-3:]:
       print(json.dumps(e, indent=2))
   ```
   For each prior entry, compare prior price against current:
   - Single-ticker entries: use `price_usd`
   - Multi-ticker entries (fin-goldsilver): use `gold.price_usd`
   - If `verdict_correct_1d` or `outcome_1d_pct` exists, note the scored result
   - Include a **Prior Verdict Reflection** section early in your output:
     ```
     ### Prior Verdict Reflection
     {date} ({command}): Said {verdict} at ${price} (conf {conf}) → now ${current} ({pct}%) — {RIGHT/WRONG}
     ```
   - Use this to calibrate current verdicts — if you've been consistently wrong, adjust confidence down
   - If no prior entries exist, skip this section

2. **Ensure fresh precomputed context** — Check `data/gold_deep_context.json`:
   - Read the file and parse `generated_at` timestamp
   - Compute age: `(now - generated_at)` in hours
   - **If the file is missing OR older than 2 hours**, refresh it by running:
     ```
     .venv/Scripts/python.exe portfolio/metals_precompute.py
     ```
     This takes ~15 seconds and fetches fresh futures prices, ETF data, COT, FRED yields.
     After it completes, re-read `data/gold_deep_context.json` for the fresh data.
   - **If the file is fresh (< 2 hours old)**, use it directly — no refresh needed.
   - Print the data age: `"Precomputed context: {age} min old"` (or `"freshly generated"` if just refreshed)

   The file contains: analyst targets, central bank buying, real yields, COT positioning,
   GLD ETF flows, G/S ratio history, signal accuracy, price trajectory.

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

## 9. Log the invocation

After producing the output, append a log entry to `data/fin_command_log.jsonl`:
```python
import json, datetime, pathlib
entry = {
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "command": "fin-gold",
    "ticker": "XAU-USD",
    "price_usd": <current XAU price from agent_summary_compact>,
    "gs_ratio": <computed G/S ratio>,
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
    "real_yield": <10Y - CPI from FRED>,
    "dxy": <DXY value>,
    "verdict_1_3d": "<bullish/bearish/neutral>",
    "verdict_1_3d_conf": <0.0-1.0>,
    "verdict_1_4w": "<bullish/bearish/neutral>",
    "verdict_1_4w_conf": <0.0-1.0>,
    "precompute_age_hours": <hours since gold_deep_context.json was generated>,
    "avanza_available": <true/false>,
    "data_sources_used": ["agent_summary", "precompute", ...],
    "execution_time_sec": <wall clock seconds from step 1 to this step>,
}
with open("data/fin_command_log.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

**Timing:** Record the start time at step 1 (when you check the clock). Compute `execution_time_sec`
as the difference between now and step 1 start time.

**Why log:** Builds a dataset of every analysis for tracking:
- Verdict accuracy (compare with actual price moves)
- Data availability patterns
- Execution time trends
- Signal accuracy at invocation time vs outcome

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


## Avanza Trading API

When the user asks to place orders, check positions, or manage trades, use these functions from `portfolio.avanza_session`:

```python
from portfolio.avanza_session import (
    get_quote,           # get_quote("1069606") -> {buy, sell, last, changePercent}
    get_buying_power,    # get_buying_power() -> {buying_power, total_value, own_capital}
    get_positions,       # get_positions() -> [{name, volume, value, account_id, ...}]
    place_buy_order,     # place_buy_order("1069606", price=0.86, volume=5000) -> {orderRequestStatus, orderId}
    place_sell_order,    # place_sell_order("1069606", price=1.05, volume=5000) -> same
    cancel_order,        # cancel_order("865451335") -> {orderRequestStatus}
    api_get,             # api_get("/_api/trading/rest/orders") -> list open orders
)
# Stop-losses: api_get("/_api/trading/stoploss") to list
# Open orders: api_get("/_api/trading/rest/orders") to list
```

**Key rules:**
- Default account: `1625505` (ISK). Only use available cash.
- Sell + stop-loss volume must NOT exceed position size (Avanza blocks it as short-selling).
- Cancel orders uses POST not DELETE: `cancel_order(order_id)`.
- Stop-loss payload is nested — see `data/metals_avanza_helpers.py:place_stop_loss()` for format.
- Also works without Playwright via `portfolio.avanza_client` when TOTP credentials are configured.
