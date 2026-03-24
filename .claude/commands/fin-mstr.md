Deep MSTR analysis — MicroStrategy as leveraged BTC proxy with NAV premium tracking, live signals, and multi-horizon verdict.

MSTR is a NASDAQ stock that holds ~450K+ BTC on its balance sheet, making it a leveraged BTC proxy.
The key metric is NAV premium/discount — the spread between MSTR market cap and its BTC holdings value.
MSTR trades US hours ONLY (15:30-22:00 CET / 9:30-16:00 ET). After-hours volatility is NOT actionable.
MSTR signal accuracy is historically 21-61% — highly variable, treat with extra caution.

## Prerequisites
- Read `memory/trading_rules.md` FIRST — check per-ticker accuracy rules before any recommendation.

## Steps

1. **Check the time** — Run `powershell.exe -NoProfile -Command "Get-Date -Format 'dddd yyyy-MM-dd HH:mm:ss'"`
   US market hours: **15:30-22:00 CET** (09:30-16:00 ET). MSTR trades ONLY during these hours.
   Pre-market: 10:00-15:30 CET. After-hours: 22:00-02:00 CET.
   **If US market is CLOSED**, note prominently at the top — signals are stale, analysis is
   pre-market watch only. Do NOT recommend trades on stale signals.
   **If <1h to close (after 21:00 CET)**, flag prominently — do NOT open new positions per trading rules.

1b. **Recall prior verdicts** — Read `data/fin_command_log.jsonl` and extract the last 3 entries covering MSTR:
   ```python
   import json
   entries = []
   with open("data/fin_command_log.jsonl", encoding="utf-8") as f:
       for line in f:
           line = line.strip()
           if not line: continue
           try:
               e = json.loads(line)
               if e.get("ticker") == "MSTR" or "MSTR" in e.get("tickers", []):
                   entries.append(e)
           except: pass
   for e in entries[-3:]:
       print(json.dumps(e, indent=2))
   ```
   For each prior entry, compare prior price against current:
   - Single-ticker entries: use `price_usd`
   - Multi-ticker entries (fin-crypto): use `mstr.price_usd`
   - If `verdict_correct_1d` or `outcome_1d_pct` exists, note the scored result
   - Include a **Prior Verdict Reflection** section early in your output:
     ```
     ### Prior Verdict Reflection
     {date} ({command}): Said {verdict} at ${price} (conf {conf}) → now ${current} ({pct}%) — {RIGHT/WRONG}
     ```
   - Use this to calibrate current verdicts — if you've been consistently wrong, adjust confidence down
   - If no prior entries exist, skip this section

2. **Ensure fresh learned lessons** — Check `data/system_lessons.json`:
   - If the file is missing OR older than 2 hours, refresh by running:
     ```
     .venv/Scripts/python.exe portfolio/fin_evolve.py
     ```
     This scores past verdicts against actual outcomes and generates calibration advice.
   - If fresh, use directly.

   This file is auto-generated every 2h by the evolution engine.
   It contains unified accuracy from ALL prediction sources (Layer 2 journal + /fin commands):
   - Your accuracy by source (layer2 vs fin-mstr vs fin-btc, etc.), regime, confidence level
   - Per-ticker accuracy across ALL sources (MSTR, BTC-USD, etc.)
   - Anti-patterns (conditions where past verdicts were wrong)
   - Confirmed patterns (conditions where past verdicts were right)
   - Cross-asset patterns (e.g., "when BTC bullish, MSTR follows X%")
   - Calibration advice (are you over/underconfident?)

   **Apply these adjustments:**
   - If calibration says OVERCONFIDENT, reduce all verdict confidence by the suggested amount
   - If an anti-pattern matches current conditions, note it in the bear case and reduce confidence
   - If a confirmed pattern matches, note it in the bull case and increase confidence
   - If accuracy for current regime is < 0.5, add a disclaimer: "Note: past verdicts in {regime} regime have been weak ({accuracy}%)"

   If the file doesn't exist or has < 5 verdicts, skip this step (not enough data yet).

3. **Read live data** (parallel):
   - `data/agent_summary_compact.json` — MSTR section: signals, prices, probabilities,
     regime, cumulative gains, forecast signals, Monte Carlo, signal_reliability.
     Also read: BTC-USD section (MSTR is a BTC proxy — BTC signals are the primary driver),
     `macro` section (DXY, yields, FOMC),
     `fundamentals` section for MSTR (Alpha Vantage: P/E, revenue, analyst targets, market cap),
     `focus_probabilities` (if MSTR is in focus tickers),
     `fear_greed` (stock F&G for market sentiment)
   - `data/prophecy.json` — check for `btc_bull_2026` or `btc_range_2026` belief. No direct
     MSTR prophecy exists, but the BTC prophecy ($100K target, 0.7 conviction) directly implies
     MSTR upside — MSTR typically has 1.5-2.5x beta to BTC.
   - `data/portfolio_state.json` — Patient strategy: check for MSTR holdings
   - `data/portfolio_state_bold.json` — Bold strategy: check for MSTR holdings
   - Last 5 entries from `data/layer2_journal.jsonl` that mention MSTR (check `tickers` and `prices` keys)

   **No precomputed context file** — unlike gold/silver, MSTR has no external research cache.
   All data comes from `agent_summary_compact.json`, Alpha Vantage fundamentals, and the signal pipeline.

4. **Compute derived metrics** (from live data):
   - **MSTR/BTC NAV ratio** — THE key MSTR-specific metric:
     - Compute: MSTR market cap / (BTC holdings * BTC price)
     - Use Alpha Vantage `MarketCapitalization` for MSTR market cap
     - Use ~450K BTC as holdings estimate (check recent filings if available in fundamentals)
     - Use current BTC-USD price from agent_summary_compact
     - Premium if ratio > 1.0 (market values MSTR above its BTC). Discount if < 1.0.
     - **Premium expanding** = speculation/FOMO, institutional demand for BTC exposure via stock
     - **Premium contracting** = value play if BTC thesis intact, or market pricing in dilution risk
     - **At discount** = rare, potentially deep value if you believe BTC thesis, but also signals
       market concern about Saylor's leverage/dilution strategy
   - **MSTR beta to BTC:** Historically 1.5-2.5x. If MSTR is moving 2x BTC, that's normal.
     If MSTR is moving less than BTC (beta < 1.5), premium is compressing. If more (beta > 2.5),
     speculation is overheating.
   - **BTC correlation assessment:** How closely is MSTR tracking BTC today?
     - If BTC up and MSTR lagging: potential catch-up trade (if premium is compressing rationally)
     - If BTC up and MSTR leading: premium expansion, FOMO driving
     - If BTC down and MSTR down more: leverage working against — normal
     - If BTC down and MSTR flat: premium expansion in the face of BTC weakness — very bullish signal
     - If BTC flat and MSTR moving: stock-specific catalyst (earnings, offering, inclusion)
   - **Fundamental snapshot** (from Alpha Vantage):
     - P/E ratio (MSTR's software business is small; P/E is mostly noise — BTC NAV is what matters)
     - Revenue trend (flat/declining software revenue is expected — watch for surprises)
     - Analyst target price (reflects institutional view on BTC exposure premium)
     - Market cap (for NAV ratio computation)
   - **Volume analysis:** Is smart money accumulating or distributing?
     - Vol > 1.5x 20-period avg + price up = accumulation (institutional buying)
     - Vol > 1.5x + price down = distribution (selling into liquidity)
     - Low volume = disinterest — wait for catalyst
   - **Signal accuracy ranking:** From signal_reliability section for MSTR specifically.
     MSTR accuracy is 21-61% — highly variable. Identify which specific signals are above coin-flip.
   - **Reflection:** Compare previous journal MSTR prices vs current. Was the thesis right?

5. **Cross-reference BTC signals with MSTR:**
   - **BTC signal consensus** is the primary driver. If BTC is BUY, MSTR should follow (with leverage).
     If BTC signal is noise, MSTR signal is double-noise.
   - Is MSTR premium expanding (speculation, FOMO, institutional demand) or contracting (de-risking)?
   - Are stock-specific signals (news_event, claude_fundamental, econ_calendar) adding information
     beyond BTC? Stock signals can capture: earnings, convertible note announcements, index
     inclusion/exclusion, Saylor purchases, regulatory developments.
   - **MSTR vs direct BTC:** If BTC consensus is strong, is MSTR a better vehicle (leverage + stock
     market liquidity) or worse (after-hours gaps, dilution risk, premium compression)?
   - **Cross-reference macro:** DXY direction affects both BTC and stocks. Rising yields are double
     headwind for MSTR (hurts both BTC and growth/speculative stocks). FOMC proximity amplified.

6. **Run adversarial debate** — MANDATORY for every invocation:
   - **Bull case:** BTC breakout proxy with built-in leverage (1.5-2.5x beta), institutional BTC
     exposure via stock (accessible to funds that cannot hold BTC directly), premium justified by
     BTC accumulation strategy (Saylor continuously buying), analyst targets, halving cycle
     alignment (Q2/Q3 2026 peak window), convertible note structure creates asymmetric upside.
   - **Bear case:** Premium to NAV is fragile and can compress rapidly, dilution risk from
     convertible notes and ATM offerings, BTC correlation means no diversification benefit,
     stock-specific execution risk (software business declining), after-hours gaps create overnight
     risk that spot BTC avoids, single-company concentration (management key-man risk with Saylor),
     margin calls at lower BTC prices, regulatory risk (SEC crypto scrutiny).
   - **Synthesis:** Weigh both sides — is the premium justified at current levels? Which time
     horizon favors the bull vs bear case?

7. **Produce output** in this exact format:

```
# MSTR DEEP ANALYSIS — {date} {time} CET
**MSTR: ${price} | BTC: ${btc_price} | NAV Premium: {pct}% | Market: {OPEN/CLOSED}**

## Market Status
{if closed: "US market CLOSED — next open {next_open_time} CET. Signals are stale. Pre-market watch only."}
{if open: "US market OPEN — signals are live."}
{if <1h to close: "**NEAR CLOSE (<1h)** — do NOT open new positions per trading rules. Exit decisions only."}
{if weekend: "Weekend — market closed until Monday 15:30 CET. BTC trades 24/7 — watch for BTC moves that MSTR will gap to on Monday open."}

## Signals
{consensus} ({XB/YS/ZH}) | wConf {pct}% | Regime: {regime}
Heatmap: {Now->6mo B/S/. chars}
RSI {rsi} | MACD {macd} ({trend}) | BB {position} | Vol {ratio}x | ATR {atr_pct}%
Prob: up{3h}% 3h | up{1d}% 1d | up{3d}% 3d
Monte Carlo: P(up)={pct}% | 1d: ${lo}-${hi} | 3d: ${lo}-${hi}

**Signal accuracy warning:** MSTR signal accuracy is 21-61% (highly variable).
Best signals for MSTR: {top 3 by accuracy with %}.
Worst signals for MSTR: {bottom 3 by accuracy with %}.
25 applicable signals (7 core + 18 enhanced, no futures_flow — stocks only).

## Fundamentals (Alpha Vantage)
P/E: {val} (mostly noise — BTC NAV is what matters)
Revenue: ${val} ({growth}% YoY) — software business {trend}
Analyst target: ${target} | Current: ${price} | Upside/downside: {pct}%
Market cap: ${mcap}

## NAV Analysis — THE Key Metric
BTC holdings: ~{n}K BTC (estimated from latest filings)
BTC NAV: {n} BTC x ${btc_price} = ${nav_total}
MSTR market cap: ${mcap}
**NAV premium: {pct}%** ({expanding/stable/contracting} — {interpretation})
{if premium > 100%: "Premium is extreme — market is pricing in significant future BTC appreciation or Saylor premium."}
{if premium 50-100%: "Premium is elevated — reflects institutional demand for leveraged BTC exposure via stock."}
{if premium 0-50%: "Premium is moderate — reasonable for a BTC accumulation vehicle."}
{if premium < 0%: "TRADING AT DISCOUNT TO NAV — rare event. Deep value if BTC thesis intact, or market pricing in existential risk."}

## BTC Correlation
BTC consensus: {BUY/SELL/HOLD} ({XB/YS/ZH}) | BTC RSI: {val} | BTC regime: {regime}
MSTR beta to BTC: ~{val}x (typical: 1.5-2.5x)
Today's correlation: {strong/weak/diverging}
{if BTC bullish + MSTR lagging: "**Potential catch-up trade** — MSTR has not yet priced in BTC move."}
{if BTC bearish: "**MSTR will follow BTC lower** — leverage amplifies downside. Caution."}
{if BTC neutral + MSTR moving: "**Stock-specific catalyst** — check news_event signal for MSTR-specific driver."}
{if BTC bullish + MSTR leading: "Premium expanding — FOMO-driven. May overshoot."}

BTC prophecy: ${target} target by {timeline} | Conviction: {val}
MSTR implied at BTC ${target}: ${mstr_implied} ({upside_pct}% from current, assuming {premium_pct}% premium maintained)

## Macro
DXY {val} ({chg}% 5d) | 10Y {yield}% ({trend}) | Fear & Greed Stock {val} | FOMC {days}d
{if FOMC within 4 days: "**WARNING: FOMC in {days}d — double headwind: BTC-sensitive + stock-sensitive. Reduce conviction, widen stops.**"}
{Note: MSTR is doubly sensitive to macro — it's affected both as a BTC proxy AND as a growth/speculative stock.}

## Portfolio Exposure
Patient: {MSTR holdings detail or "no MSTR position"}
Bold: {MSTR holdings detail or "no MSTR position"}
{if either has BTC-USD position: "Also holding BTC-USD directly: {details} — consider combined BTC exposure."}

## Adversarial Debate
**Bull:** {1-2 sentences — BTC leverage, institutional access, Saylor accumulation, halving cycle, analyst targets}
**Bear:** {1-2 sentences — NAV premium fragility, dilution risk, after-hours gaps, BTC correlation = no diversification, Saylor key-man risk}
**Synthesis:** {1-2 sentences — which side dominates, at what horizon, is premium justified}

## Verdict
| Horizon | Bias | Confidence | Note |
|---------|------|------------|------|
| 1-3d | {bullish/bearish/neutral} | {0.0-1.0} | {reason — note if market closed} |
| 1-4w | {bullish/bearish/neutral} | {0.0-1.0} | {reason} |
| 1-3m | {bullish/bearish/neutral} | {0.0-1.0} | {reason} |
| 6-12m | {bullish/bearish/neutral} | {0.0-1.0} | {reason — align with BTC prophecy} |

## Key Levels
Support: ${s1}, ${s2} | Resistance: ${r1}, ${r2}
{if near ATH: ATH: ${ath} — {distance}% away}

## BTC Cross-Reference
BTC-USD: ${price} ({consensus}) — {brief BTC assessment}
BTC/MSTR divergence: {if any — flag as opportunity or warning}
{if BTC breaking out: "BTC breakout in progress — MSTR should follow with 1.5-2.5x beta. Watch for premium expansion."}
{if BTC range-bound: "BTC ranging — MSTR will chop. No edge from BTC direction."}
```

## 8. Log the invocation

After producing the output, append a log entry to `data/fin_command_log.jsonl`:
```python
import json, datetime, pathlib
entry = {
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "command": "fin-mstr",
    "ticker": "MSTR",
    "price_usd": <current MSTR price from agent_summary_compact>,
    "btc_price_usd": <current BTC price from agent_summary_compact>,
    "nav_premium_pct": <computed NAV premium percentage>,
    "signal_consensus": "<BUY/SELL/HOLD>",
    "vote_breakdown": "<XB/YS/ZH>",
    "weighted_confidence": <0.0-1.0>,
    "regime": "<regime>",
    "rsi": <rsi value>,
    "btc_consensus": "<BUY/SELL/HOLD>",
    "btc_rsi": <BTC RSI value>,
    "mstr_beta_to_btc": <estimated beta>,
    "prob_3h": <directional probability 3h>,
    "prob_1d": <directional probability 1d>,
    "prob_3d": <directional probability 3d>,
    "monte_carlo_p_up": <probability of up>,
    "pe_ratio": <P/E from Alpha Vantage>,
    "analyst_target": <analyst target price>,
    "market_cap": <market cap>,
    "fear_greed_stock": <stock F&G value>,
    "dxy": <DXY value>,
    "market_open": <true/false>,
    "verdict_1_3d": "<bullish/bearish/neutral>",
    "verdict_1_3d_conf": <0.0-1.0>,
    "verdict_1_4w": "<bullish/bearish/neutral>",
    "verdict_1_4w_conf": <0.0-1.0>,
    "data_sources_used": ["agent_summary", "fundamentals", "prophecy", "portfolio_patient", "portfolio_bold", "journal"],
    "execution_time_sec": <wall clock seconds from step 1 to this step>,
}
with open("data/fin_command_log.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

**Timing:** Record the start time at step 1 (when you check the clock). Compute `execution_time_sec`
as the difference between now and step 1 start time.

**Why log:** Builds a dataset of every analysis for tracking:
- Verdict accuracy (compare with actual price moves)
- NAV premium tracking over time (is premium expanding or contracting?)
- BTC correlation strength vs MSTR moves
- Signal accuracy at invocation time vs outcome
- Market status (open/closed) correlation with signal reliability

## Critical rules

- **MSTR is a STOCK — US hours ONLY (15:30-22:00 CET).** Always check market status first.
  If market is closed, signals are stale. Say so prominently. Do NOT recommend trades on stale data.
- **MSTR after-hours volatility spikes are NOT actionable.** Wait for the regular session.
  After-hours moves often reverse or gap differently at open. Never trade on after-hours momentum.
- **Near close (<1h to 22:00 CET): do NOT open new positions** per trading rules. Exit decisions only.
- **MSTR is a leveraged BTC proxy — ALWAYS check BTC first.** BTC signal consensus is the primary
  driver for MSTR direction. If BTC signal is noise (44-54% accuracy), MSTR signal is double-noise.
  Never recommend MSTR based solely on MSTR-specific signals without checking BTC alignment.
- **MSTR historical accuracy is 21-61% — highly variable.** Treat all signal consensus with caution.
  Independent judgment, BTC correlation, and NAV premium analysis matter more than vote counting.
- **NAV premium/discount is THE key MSTR-specific metric.** Always compute and display prominently.
  The premium reflects market sentiment toward leveraged BTC exposure. Track whether it is expanding
  or contracting — this is more predictive than technical signals for MSTR.
- **Stock reasoning requirement:** Per CLAUDE.md, briefly state WHY you are holding or trading MSTR.
  Do not just cite signal counts — explain the thesis (BTC proxy, premium/discount, catalyst).
- **Adversarial debate is MANDATORY** — never skip it, even when signals are unanimous.
- **Fee drag: 0.10% stocks round-trip** (vs 0.05% crypto). This means MSTR must move more than
  BTC to be worth trading vs buying BTC directly. Factor this into break-even analysis.
- **MSTR is mapped to "crypto" sector in news_keywords.** The news_event signal will pick up both
  MicroStrategy-specific news AND broader crypto news. This is intentional — MSTR reacts to both.
- **No futures_flow signal for MSTR** — futures_flow (#30) is crypto-only (BTC, ETH). MSTR has
  25 applicable signals (7 core + 18 enhanced), not 27.
- **Dilution risk is always present.** Saylor regularly does convertible note offerings and ATM
  equity raises to buy more BTC. Each offering dilutes existing shareholders. This is structural —
  it is both the bull case (more BTC per share long-term if BTC goes up) and the bear case
  (dilution if BTC goes down or sideways). Always mention in the bear case.
- **Weekend analysis:** On weekends, BTC trades but MSTR does not. If BTC makes a significant move
  over the weekend, MSTR will gap at Monday open. Note this risk/opportunity if analyzing on
  Sat/Sun. BTC weekend moves are a leading indicator for Monday MSTR open.
- **BTC prophecy ($100K target) implies MSTR upside.** With 1.5-2.5x beta and current premium,
  estimate MSTR implied price at BTC $100K for the long-term horizon verdict.
- **No precomputed external context** — unlike gold/silver, there is no `mstr_deep_context.json`.
  All data comes from the signal pipeline and Alpha Vantage fundamentals. If key data is missing
  from agent_summary_compact, note what is unavailable rather than guessing.
