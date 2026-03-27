Deep BTC analysis — BTC-USD with live signals, futures flow, and multi-horizon verdict.

Bitcoin is the crypto anchor — leads the entire crypto market. ETH and MSTR follow BTC.
BTC trades 24/7 on Binance (BTCUSDT). Signal accuracy for BTC is historically 44-54% —
near coin-flip territory. Independent judgment matters more than signal counting here.

## Prerequisites
- Read `memory/trading_rules.md` FIRST — check per-ticker accuracy rules before any recommendation.

## Steps

1. **Check the time** — Run `powershell.exe -NoProfile -Command "Get-Date -Format 'dddd yyyy-MM-dd HH:mm:ss'"`
   BTC trades 24/7 on spot markets. Always actionable, no market hours restriction.
   Avanza XBT Tracker trades during EU hours: **09:00-17:25 CET**.

1b. **Recall prior verdicts** — Read `data/fin_command_log.jsonl` and extract the last 3 entries covering BTC-USD:
   ```python
   import json
   entries = []
   with open("data/fin_command_log.jsonl", encoding="utf-8") as f:
       for line in f:
           line = line.strip()
           if not line: continue
           try:
               e = json.loads(line)
               if e.get("ticker") == "BTC-USD" or "BTC-USD" in e.get("tickers", []):
                   entries.append(e)
           except: pass
   for e in entries[-3:]:
       print(json.dumps(e, indent=2))
   ```
   For each prior entry, compare prior price against current:
   - Single-ticker entries: use `price_usd`
   - Multi-ticker entries (fin-crypto): use `btc.price_usd`
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
   - Your accuracy by source (layer2 vs fin-btc vs fin-silver vs fin-gold), regime, confidence level
   - Per-ticker accuracy across ALL sources (BTC-USD, ETH-USD, XAG-USD, XAU-USD, etc.)
   - Anti-patterns (conditions where past verdicts were wrong)
   - Confirmed patterns (conditions where past verdicts were right)
   - Cross-asset patterns (e.g., "when BTC bullish, ETH follows X%")
   - Calibration advice (are you over/underconfident?)

   **Apply these adjustments:**
   - If calibration says OVERCONFIDENT, reduce all verdict confidence by the suggested amount
   - If an anti-pattern matches current conditions, note it in the bear case and reduce confidence
   - If a confirmed pattern matches, note it in the bull case and increase confidence
   - If accuracy for current regime is < 0.5, add a disclaimer: "Note: past verdicts in {regime} regime have been weak ({accuracy}%)"

   If the file doesn't exist or has < 5 verdicts, skip this step (not enough data yet).

3. **Read live data** (parallel):
   - `data/agent_summary_compact.json` — BTC-USD section: signals, prices, probabilities,
     regime, cumulative gains, forecast signals, Monte Carlo, signal_reliability.
     Also read: ETH-USD section (for BTC/ETH ratio), MSTR section (for correlation),
     `macro` section (DXY, yields, FOMC), `futures_data` (OI, funding rate, LS ratio),
     `focus_probabilities` (BTC directional probabilities at 3h/1d/3d),
     `fear_greed` (crypto-specific F&G from Alternative.me)
   - `data/prophecy.json` — `btc_bull_2026` or `btc_range_2026` belief (target, conviction, checkpoints)
   - `data/portfolio_state.json` — Patient strategy: check for BTC-USD holdings
   - `data/portfolio_state_bold.json` — Bold strategy: check for BTC-USD holdings
   - Last 5 entries from `data/layer2_journal.jsonl` that mention BTC-USD (check `tickers` and `prices` keys)

   **No precomputed context file** — unlike gold/silver, BTC has no external research cache.
   All data comes from `agent_summary_compact.json` and the signal pipeline.

4. **Compute derived metrics** (from live data):
   - **BTC/ETH ratio:** BTC price / ETH price. Rising ratio = BTC outperforming (risk-off within crypto).
     Falling ratio = ETH catching up (risk-on altcoin rotation). Historical context: typical range 15-25.
   - **Distance to prophecy target:** ($100K - current) / current * 100. Track weekly progress.
   - **Funding rate analysis** (from `futures_data`):
     - Positive funding (>0): Longs paying shorts — bullish sentiment but crowded longs.
     - Negative funding (<0): Shorts paying longs — bearish sentiment but potential squeeze.
     - **Extreme positive (>0.1%):** Overleveraged longs — contrarian bearish. Correction risk high.
     - **Extreme negative (<-0.05%):** Overleveraged shorts — contrarian bullish. Short squeeze setup.
     - Neutral (-0.01% to +0.03%): Balanced — no contrarian signal.
   - **Futures OI trend** (from `futures_data`):
     - Rising OI + rising price = new longs entering (trend confirmation)
     - Rising OI + falling price = new shorts entering (bearish pressure building)
     - Falling OI + rising price = short covering rally (weak, may fade)
     - Falling OI + falling price = long liquidation (capitulation, may be near bottom)
   - **Long/Short ratio** (from `futures_data`):
     - Crowd positioning: >1.0 = more longs, <1.0 = more shorts
     - Top trader divergence from crowd = smart money signal
   - **Fear & Greed** (crypto-specific from Alternative.me):
     - 0-20: Extreme Fear (contrarian buy zone)
     - 21-40: Fear
     - 41-60: Neutral
     - 61-80: Greed
     - 81-100: Extreme Greed (contrarian sell zone)
   - **MSTR correlation:** MSTR as leveraged BTC proxy — check if MSTR is leading or lagging BTC.
     MSTR premium/discount to NAV is a sentiment indicator for institutional BTC demand.
   - **Signal accuracy ranking:** From signal_reliability section for BTC-USD specifically.
     BTC accuracy is 44-54% overall — identify which specific signals are above/below coin-flip.
   - **Reflection:** Compare previous journal BTC prices vs current. Was the thesis right?

5. **Cross-reference with macro:**
   - **DXY direction:** DXY up = risk-off headwind for BTC. DXY down = liquidity tailwind.
   - **Treasury yields:** Rising yields = tighter liquidity = BTC headwind. Falling yields = tailwind.
   - **Yield curve:** Inverted = recession risk (BTC mixed: flight to safety vs risk-off).
     Normal/steepening = growth expectation (BTC bullish as risk-on asset).
   - **FOMC proximity:** BTC is hyper-sensitive to liquidity expectations. Within 4 days = flag prominently.
   - **Crypto Fear & Greed vs Stock Fear & Greed:** Divergence = interesting signal.
     If crypto fear while stocks greed = potential crypto catch-up.
   - **Global liquidity:** M2 money supply trends (if available in macro context).

6. **Run adversarial debate** — MANDATORY for every invocation:
   - **Bull case:** Halving cycle (Apr 2024 halving, historically 12-18mo to peak), institutional
     adoption (ETF inflows, corporate treasuries), prophecy alignment ($100K target), macro
     liquidity (Fed pivot expectations), network fundamentals (hash rate, adoption metrics),
     de-correlation from tradfi.
   - **Bear case:** Regulatory risk (SEC, global regulation), DXY strength (liquidity drain),
     funding rate overheated (overleveraged longs), whale distribution (on-chain data if available),
     FOMC hawkish surprise, Mt. Gox/Genesis distributions, miner selling pressure,
     ETF outflow risk, macro risk-off.
   - **Synthesis:** Weigh both sides — which dominates at each time horizon?

7. **Produce output** in this exact format:

```
# BTC DEEP ANALYSIS — {date} {time} CET
**BTC-USD: ${price} | BTC/ETH: {ratio} | Fear & Greed: {val} | FOMC: {days}d**

## Signals
{consensus} ({XB/YS/ZH}) | wConf {pct}% | Regime: {regime}
Heatmap: {Now->6mo B/S/. chars}
RSI {rsi} | MACD {macd} ({trend}) | BB {position} | Vol {ratio}x | ATR {atr_pct}%
Chronos 24h: {pct}% ({accuracy}% acc, {n} sam)
Prob: up{3h}% 3h | up{1d}% 1d | up{3d}% 3d
Monte Carlo: P(up)={pct}% | 1d: ${lo}-${hi} | 3d: ${lo}-${hi}

**Signal accuracy warning:** BTC signal accuracy is 44-54% (coin-flip territory).
Best signals for BTC: {top 3 by accuracy with %}.
Worst signals for BTC: {bottom 3 by accuracy with %}.
Weighted consensus is unreliable — use independent judgment.

## Prophecy: $100K Target
Progress: {pct}% | Conviction: 0.7 | Timeline: Q2/Q3 2026
Distance: ${gap} ({pct}% to go)
{checkpoint status if available — which triggered, which pending}

## Futures Flow
Funding rate: {rate}% ({interpretation}: bullish/bearish/neutral/overleveraged longs/shorts squeeze setup)
OI trend: {rising/falling} + price {rising/falling} = {interpretation}
Long/Short ratio: {ratio} (crowd: {more longs/more shorts/balanced})
Top trader vs crowd: {divergence or alignment — if available}
{if extreme funding or OI divergence, flag prominently}

## Macro
DXY {val} ({chg}% 5d) | 10Y {yield}% ({trend}) | Fear & Greed Crypto {val} | Fear & Greed Stock {val} | FOMC {days}d
{if FOMC within 4 days: **WARNING: FOMC in {days}d — BTC hyper-sensitive to Fed communication. Reduce position sizing, widen stops.**}

## Portfolio Exposure
Patient: {BTC-USD holdings detail or "no BTC position"}
Bold: {BTC-USD holdings detail or "no BTC position"}
XBT Tracker (Avanza): {if held in warrant portfolio, show P&L — otherwise "not held"}

## Adversarial Debate
**Bull:** {1-2 sentences — halving cycle, institutional adoption, ETF inflows, prophecy, macro liquidity}
**Bear:** {1-2 sentences — regulatory risk, DXY strength, funding rate overheated, whale distribution, FOMC hawkish}
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
{if near ATH: ATH: ${ath} — {distance}% away}

## ETH/MSTR Cross-Reference
BTC/ETH ratio: {current} ({rising/falling} — ETH {leading/lagging/tracking} BTC)
ETH-USD: ${price} ({consensus}) — {brief ETH assessment}
MSTR: ${price} ({consensus}) — {MSTR as BTC proxy: premium/discount, correlation note}
{if BTC and ETH diverging: flag — "BTC/ETH divergence: {interpretation}"}
```

## 8. Log the invocation

After producing the output, append a log entry to `data/fin_command_log.jsonl`:
```python
import json, datetime, pathlib
entry = {
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "command": "fin-btc",
    "ticker": "BTC-USD",
    "price_usd": <current BTC price from agent_summary_compact>,
    "btc_eth_ratio": <computed BTC/ETH ratio>,
    "signal_consensus": "<BUY/SELL/HOLD>",
    "vote_breakdown": "<XB/YS/ZH>",
    "weighted_confidence": <0.0-1.0>,
    "regime": "<regime>",
    "rsi": <rsi value>,
    "chronos_24h_pct": <chronos prediction %>,
    "chronos_accuracy": <chronos accuracy for BTC>,
    "prob_3h": <directional probability 3h>,
    "prob_1d": <directional probability 1d>,
    "prob_3d": <directional probability 3d>,
    "monte_carlo_p_up": <probability of up>,
    "funding_rate": <current funding rate>,
    "oi_trend": "<rising/falling>",
    "ls_ratio": <long/short ratio>,
    "fear_greed_crypto": <crypto F&G value>,
    "dxy": <DXY value>,
    "prophecy_distance_pct": <% to $100K target>,
    "verdict_1_3d": "<bullish/bearish/neutral>",
    "verdict_1_3d_conf": <0.0-1.0>,
    "verdict_1_4w": "<bullish/bearish/neutral>",
    "verdict_1_4w_conf": <0.0-1.0>,
    "data_sources_used": ["agent_summary", "prophecy", "portfolio_patient", "portfolio_bold", "journal"],
    "execution_time_sec": <wall clock seconds from step 1 to this step>,
}
with open("data/fin_command_log.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

**Timing:** Record the start time at step 1 (when you check the clock). Compute `execution_time_sec`
as the difference between now and step 1 start time.

**Why log:** Builds a dataset of every analysis for tracking:
- Verdict accuracy (compare with actual price moves)
- Futures flow predictive value (was OI divergence a leading indicator?)
- Funding rate extremes vs subsequent moves
- BTC/ETH ratio changes and cross-asset leads
- Signal accuracy at invocation time vs outcome

## Critical rules

- **BTC signal accuracy is 44-54% — always note this prominently.** Do not present signal
  consensus as reliable. Independent judgment, futures flow, and macro context matter more
  than vote counting for BTC.
- **BTC 12h BUY phantom is proven noise.** Appeared 20+ times, always fades, was confirmed dead
  and resurrected multiple times. Ignore single-timeframe BUY on 12h.
- **BTC Now TF flips rapidly.** SELL/HOLD oscillates each check-in. Never trust single-check
  reads on the Now timeframe. Need 3+ consecutive checks in the same direction.
- **Funding rate is a contrarian indicator at extremes.** >0.1% = overleveraged longs (bearish).
  <-0.05% = overleveraged shorts (bullish squeeze setup). Flag these prominently.
- **FOMC within 4 days:** Flag prominently in the header and macro section. BTC is hyper-sensitive
  to liquidity expectations. Reduce short-term (1-3d) conviction by at least 0.2.
  Note: "FOMC proximity — expect volatility, reduce sizing."
- **User prophecy is medium-term ($100K by Q2/Q3 2026).** Short-term noise does not invalidate it.
  Always frame short-term analysis in context of the medium-term bullish thesis.
- **BTC leads crypto.** If BTC is moving and ETH is not, ETH will likely follow with a lag.
  If ETH is moving and BTC is not, the move may be ETH-specific and less durable.
- **MSTR is leveraged BTC.** MSTR premium expansion = institutional FOMO. MSTR discount = institutional skepticism. Track this as a sentiment indicator.
- **Adversarial debate is MANDATORY** — never skip it, even when signals are unanimous.
- **Fee drag:** 0.05% crypto round-trip. Don't trade on marginal signals where expected move < fee.
- **XBT Tracker on Avanza** trades EU hours only (09:00-17:25 CET). If recommending warrant
  action outside those hours, note that execution must wait for EU open.
- **No precomputed external context** — unlike gold/silver, there is no `btc_deep_context.json`.
  All data comes from the signal pipeline. If key data is missing from agent_summary_compact,
  note what's unavailable rather than guessing.
- **Halving cycle context:** April 2024 halving. Historically, BTC peaks 12-18 months post-halving
  (Oct 2025 - Oct 2026 window). The user's Q2/Q3 2026 prophecy aligns with this cycle.


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
