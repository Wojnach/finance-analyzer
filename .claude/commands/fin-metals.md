Quick metals focus — XAU-USD (gold) and XAG-USD (silver) with LIVE Avanza positions and probabilities.

These are the PRIMARY instruments. The user holds leveraged warrants on Avanza that change over time.

## Steps

1. **Check the time** — Run `powershell.exe -NoProfile -Command "Get-Date -Format 'dddd yyyy-MM-dd HH:mm:ss'"`
   Avanza commodity warrant hours: **08:15-21:55 CET** (NOT 17:25).

2. **Fetch LIVE Avanza positions** — Run:
   ```
   .venv/Scripts/python.exe scripts/avanza_metals_check.py
   ```
   This hits the live Avanza API and returns all metals-related positions (silver, gold, warrants, MINIs, certificates) with:
   - name, units, current value (SEK), acquired value, P&L (SEK + %), today's change
   - This is the REAL position data — never rely on state files for what's held

   The output separates `active` positions from `knocked_out` ones:
   - **active**: real positions you can trade
   - **knocked_out**: dead warrants (barrier hit, -85%+ loss, price <1.50 SEK) — mention briefly but don't analyze

   If the session is expired, tell the user: "Avanza session expired — run `python scripts/avanza_login.py`"

3. **Read signal data** (parallel with step 2 if possible):
   - `data/metals_agent_summary.json` — metals-only extract (~10 KB vs 92 KB full):
     XAG-USD + XAU-USD signals, prices, probabilities, regime, indicators, cumulative gains,
     macro context, prophecy (silver_bull_2026 + gold beliefs included)
     If this file is missing, fall back to `data/agent_summary_compact.json` (XAG-USD + XAU-USD sections only)
   - `data/metals_swing_config.py` — warrant catalog with leverage, barriers (for barrier-distance calculation)
   - Last 3 entries from `data/layer2_journal.jsonl` — previous metals theses and prices

4. **For EACH metal (XAG-USD and XAU-USD), extract from metals_agent_summary.json:**
   - Current USD price + cumulative gains (1d/3d/7d)
   - Signal consensus + vote breakdown (XB/YS/ZH)
   - Weighted confidence vs raw confidence
   - Timeframe heatmap (7 chars: Now→6mo, B/S/· format)
   - RSI, MACD (value + trend), BB position, volume ratio
   - Regime: trending-up/down, range-bound, high-vol, breakout, capitulation
   - Focus probabilities: directional % at 3h, 1d, 3d (from `focus_probabilities`)
   - Forecast accuracy: Chronos accuracy + sample count for this ticker

5. **For each LIVE Avanza position** (from step 2):
   - Match to underlying (silver→XAG-USD, gold→XAU-USD) using name keywords
   - Look up leverage and barrier from `metals_swing_config.py` warrant catalog (match by name or ob_id)
   - If barrier known: calculate distance % from current underlying price
   - Show live P&L from Avanza (this is the real number, not estimated)
   - Show today's change %
   - Check: is Avanza open right now for this instrument?

6. **Macro context for metals:**
   - DXY: value + 5d change (strong dollar = headwind)
   - 10Y yield + trend (falling yields = tailwind)
   - F&G (extreme fear = safe haven bullish)
   - FOMC proximity

7. **Compare with previous thesis** from journal — was the outlook correct?

## Output format

```
## Metals — {day} {date} {time} CET
Avanza: {OPEN/CLOSED} (08:15-21:55 CET)

### XAG-USD (Silver) — ${price}
{1d}% 1d | {3d}% 3d | {7d}% 7d
Prob: ↑{pct}% 3h | ↑{pct}% 1d | ↑{pct}% 3d (acc: {pct}%, {N} sam)
{BUY/SELL/HOLD} ({XB/YS/ZH}) conf {raw}%/w{weighted}% | {B·S·BBB}
Regime: {regime} | RSI {rsi} | MACD {macd} | Vol {ratio}x
Prophecy: ${current}→$120 ({progress}%) — {checkpoint status}

### XAU-USD (Gold) — ${price}
{same format}

### Avanza Positions (LIVE)
{for each position from the API:}
  {NAME}
  {units} units | {value_sek} SEK | P&L: {profit_sek} SEK ({profit_pct}%)
  Today: {change_today_pct}% | Price: {last_price} SEK
  {if barrier known: Barrier: ${barrier} ({distance}% away)}

{if knocked_out positions exist:}
### Knocked Out
  {NAME} — {units} units, paid {acquired_sek} SEK, residual {value_sek} SEK. Dead.

{if no active metals positions found:}
  No active metals positions on Avanza.

### Macro
DXY {value} ({change}%) | 10Y {yield}% ({trend}) | F&G {val} | FOMC {days}d

### Assessment
{2-4 sentences: silver/gold setup, warrant recommendation, key risk}
```

## Critical rules
- **LIVE positions from API are the source of truth** — ignore state files if they conflict
- XAG-USD signal accuracy historically 71-83% — trust the signals
- NEVER recommend stop-loss within 3% of current bid
- User prefers 5x leverage, not 10x
- Warrant scalps: 3-5h max hold, +2% underlying take-profit, -2% hard stop
- Does NOT want to hold warrants overnight
- Separate prophecy (long-term) from scalp (short-term) — don't mix
