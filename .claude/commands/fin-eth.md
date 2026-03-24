Deep ETH-USD analysis — Ethereum with live signals, futures flow, BTC cross-reference, and multi-horizon verdict.

ETH is a secondary instrument. User prophecy: bullish, $4K target Q2/Q3 2026, 0.6 conviction.
ETH follows BTC — always check BTC status first. If BTC is bearish, ETH bullish signals are likely noise.

## Prerequisites
- Read `memory/trading_rules.md` FIRST — check per-ticker accuracy rules before any recommendation.

## Steps

1. **Check the time** — Run `powershell.exe -NoProfile -Command "Get-Date -Format 'dddd yyyy-MM-dd HH:mm:ss'"`
   ETH trades 24/7 on Binance (ETHUSDT). No market hours restriction.
   Avanza ETH Tracker (CoinShares) trades during EU hours: **09:00-17:25 CET**.

1b. **Recall prior verdicts** — Read `data/fin_command_log.jsonl` and extract the last 3 entries covering ETH-USD:
   ```python
   import json
   entries = []
   with open("data/fin_command_log.jsonl", encoding="utf-8") as f:
       for line in f:
           line = line.strip()
           if not line: continue
           try:
               e = json.loads(line)
               if e.get("ticker") == "ETH-USD" or "ETH-USD" in e.get("tickers", []):
                   entries.append(e)
           except: pass
   for e in entries[-3:]:
       print(json.dumps(e, indent=2))
   ```
   For each prior entry, compare prior price against current:
   - Single-ticker entries: use `price_usd`
   - Multi-ticker entries (fin-crypto): use `eth.price_usd`
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
   - Your accuracy by source (layer2 vs fin-eth vs fin-silver vs fin-gold), regime, confidence level
   - Per-ticker accuracy across ALL sources (BTC-USD, ETH-USD, XAG-USD, XAU-USD, etc.)
   - Anti-patterns (conditions where past verdicts were wrong)
   - Confirmed patterns (conditions where past verdicts were right)
   - Cross-asset patterns (e.g., "when BTC breaks out, ETH follows X%")
   - Calibration advice (are you over/underconfident?)

   **Apply these adjustments:**
   - If calibration says OVERCONFIDENT, reduce all verdict confidence by the suggested amount
   - If an anti-pattern matches current conditions, note it in the bear case and reduce confidence
   - If a confirmed pattern matches, note it in the bull case and increase confidence
   - If accuracy for current regime is < 0.5, add a disclaimer: "Note: past verdicts in {regime} regime have been weak ({accuracy}%)"

   If the file doesn't exist or has < 5 verdicts, skip this step (not enough data yet).

3. **Read live data** (parallel):
   - `data/agent_summary_compact.json` — ETH-USD AND BTC-USD sections: signals, prices, probabilities,
     regime, cumulative gains, forecast signals, Monte Carlo, signal_reliability, futures_data.
     Also read the `macro` section (DXY, yields, FOMC, Fear & Greed).
   - `data/prophecy.json` — `eth_follows_btc` belief (and `btc_range_2026` for BTC dependency context)
   - `data/portfolio_state.json` — Patient strategy (check for ETH holdings)
   - `data/portfolio_state_bold.json` — Bold strategy (check for ETH holdings — note Bold previously
     bought ETH at $1,978 and sold in stages at $1,963, $1,928, $1,906, fully liquidated)
   - `data/portfolio_state_warrants.json` — ETH Tracker warrant position (if any)
   - Last 5 entries from `data/layer2_journal.jsonl` that mention ETH-USD (check `tickers` and `prices` keys)

   **There is NO precomputed context file for ETH.** Everything comes from agent_summary_compact.json.

4. **Compute derived metrics** (from live data):
   - **ETH/BTC ratio:** ETH price / BTC price. Track direction — is ETH gaining or losing vs BTC?
     Historical context: ETH/BTC ranged 0.015-0.08 in recent years. Below 0.03 = deeply undervalued vs BTC.
     Above 0.05 = ETH outperforming ("alt season"). Current ratio direction matters more than absolute value.
   - **BTC correlation check:** Compare ETH and BTC signal consensus + regime. If BTC is bearish,
     ETH bullish signals are suspect. If BTC is breaking out, ETH likely follows with a lag.
   - **Futures flow analysis:** From agent_summary_compact `futures_data` for ETH-USD:
     - Funding rate: positive = longs paying shorts (crowded longs, contrarian bearish at extremes).
       Negative = shorts paying longs (contrarian bullish at extremes). Neutral = balanced.
     - Open Interest trend: rising OI + rising price = new longs entering (bullish confirmation).
       Rising OI + falling price = new shorts entering (bearish confirmation).
       Falling OI = position unwinding (trend exhaustion).
     - Long/Short ratio: top trader vs retail positioning.
   - **Distance to prophecy target:** ($4,000 - current) / current * 100. Note BTC dependency.
   - **Fear & Greed (crypto):** From macro section. Extreme fear (<20) = contrarian bullish.
     Extreme greed (>80) = contrarian bearish.
   - **ETH staking yield context:** If available in agent_summary or fundamentals, note current
     staking APR (~3-5%). Acts as a yield floor that makes ETH attractive vs bonds when rates fall.
   - **Signal accuracy ranking:** From signal_reliability section for ETH-USD specifically.
     ETH signals historically 44-54% accurate — near coin-flip. Weight judgment heavily over signals.
   - **Reflection:** Compare previous journal ETH prices vs current. Was the thesis right?

5. **Cross-reference with macro:**
   - DXY direction: strong dollar = headwind for crypto (including ETH). Weak dollar = tailwind.
   - Treasury yields: falling yields = bullish for risk assets (ETH benefits). Rising = headwind.
   - FOMC proximity: within 4 days = flag prominently. Crypto is hyper-sensitive to Fed communication.
   - Fear & Greed: extreme readings are contrarian signals.
   - BTC status is the single most important cross-reference for ETH.

6. **Run adversarial debate** — MANDATORY for every invocation:
   - **Bull case:** L2 adoption growing (Arbitrum, Optimism, Base), DeFi TVL recovering, staking yield
     attractive vs falling rates, prophecy alignment ($4K target), follows BTC breakout, ETF narrative,
     supply burn (EIP-1559), institutional adoption.
   - **Bear case:** ETH/BTC ratio declining (ETH losing vs BTC), L2s cannibalizing mainnet fees,
     regulatory risk (SEC staking scrutiny), DXY headwind, ETH signal noise proven unreliable
     (SELL->BUY flips in one check-in), Bold strategy already lost money on ETH (bought $1,978,
     sold $1,906-$1,963), Solana/competitor narrative stealing market share.
   - **Synthesis:** Weigh both sides — which dominates at each time horizon? Is this a BTC-dependent
     trade or does ETH have its own catalyst?

7. **Produce output** in this exact format:

```
# ETH DEEP ANALYSIS — {date} {time} CET
**ETH-USD: ${price} | ETH/BTC: {ratio} | Fear & Greed: {val} | FOMC: {days}d**

## Signals
{consensus} ({XB/YS/ZH}) | wConf {pct}% | Regime: {regime}
Heatmap: {Now->6mo B/S/. chars}
RSI {rsi} | MACD {macd} ({trend}) | BB {position} | Vol {ratio}x | ATR {atr_pct}%
Chronos 24h: {pct}% ({accuracy}% acc, {n} sam)
Prob: up{3h}% 3h | up{1d}% 1d | up{3d}% 3d
Monte Carlo: P(up)={pct}% | 1d: ${lo}-${hi} | 3d: ${lo}-${hi}

**WARNING: ETH signal chaos is CONFIRMED.** SELL->BUY flips in one check-in.
Never trust single-check reads. Require 3+ consecutive checks for any ETH signal to be actionable.

## Prophecy: $4K Target
Progress: {pct}% | Conviction: 0.6 | Timeline: Q2/Q3 2026
Distance: ${gap} ({pct}% to go)
**BTC dependency:** ETH $4K requires BTC to break out first. BTC status: ${btc_price} ({btc_consensus})
BTC prophecy: {btc_prophecy_status — on track / stalling / invalidated}

## Futures Flow
Funding rate: {rate}% ({interpretation — neutral/crowded longs/crowded shorts})
OI trend: {rising/falling/flat} + price {rising/falling} = {interpretation}
Long/Short ratio: {ratio} ({top traders vs crowd positioning})
{If extreme funding or OI divergence: FLAG prominently}

## Macro
DXY {val} ({chg}%) | 10Y {yield}% ({trend}) | Fear & Greed Crypto {val} | FOMC {days}d
{If FOMC within 4 days: **FOMC IMMINENT — reduce short-term conviction, widen stops**}

## Portfolio Exposure
Patient: {ETH holdings or "No ETH position"}
Bold: {ETH holdings or "No ETH position (fully liquidated at check-in #299: bought $1,978, sold $1,906-$1,963, lost ~$72/ETH)"}
ETH Tracker (Avanza): {position details if held, or "Not held"}

## Adversarial Debate
**Bull:** {1-2 sentences — L2 adoption, DeFi TVL, staking yield, follows BTC breakout, ETF narrative}
**Bear:** {1-2 sentences — ETH/BTC ratio declining, L2 fee cannibalization, signal noise, regulatory risk, Bold lost money}
**Synthesis:** {1-2 sentences — which side dominates, BTC dependency, actionability}

## Verdict
| Horizon | Bias | Confidence | Note |
|---------|------|------------|------|
| 1-3d | {bullish/bearish/neutral} | {0.0-1.0} | {reason — always note ETH signal unreliability} |
| 1-4w | {bullish/bearish/neutral} | {0.0-1.0} | {reason — BTC dependency} |
| 1-3m | {bullish/bearish/neutral} | {0.0-1.0} | {reason — prophecy alignment} |
| 6-12m | {bullish/bearish/neutral} | {0.0-1.0} | {reason — structural thesis} |

## Key Levels
Support: ${s1}, ${s2} | Resistance: ${r1}, ${r2}
{If identifiable from signals/indicators, include Fibonacci levels}

## BTC Cross-Reference
BTC: ${btc_price} ({btc_consensus} — {XB/YS/ZH})
BTC regime: {regime} | BTC RSI: {rsi} | BTC MACD: {macd}
ETH/BTC: {ratio} ({rising/falling} — ETH {gaining/losing} vs BTC)
If BTC breaks $100K, ETH implied target: ${implied_eth} (based on current ETH/BTC ratio)
If BTC stays range-bound ($60K-$75K), ETH likely capped at: ${implied_cap}
```

## 8. Log the invocation

After producing the output, append a log entry to `data/fin_command_log.jsonl`:
```python
import json, datetime, pathlib
entry = {
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "command": "fin-eth",
    "ticker": "ETH-USD",
    "price_usd": <current ETH price from agent_summary_compact>,
    "btc_price_usd": <current BTC price>,
    "eth_btc_ratio": <ETH price / BTC price>,
    "signal_consensus": "<BUY/SELL/HOLD>",
    "vote_breakdown": "<XB/YS/ZH>",
    "weighted_confidence": <0.0-1.0>,
    "regime": "<regime>",
    "rsi": <rsi value>,
    "macd": <macd value>,
    "chronos_24h_pct": <chronos prediction %>,
    "chronos_accuracy": <chronos accuracy for ETH>,
    "prob_3h": <directional probability 3h>,
    "prob_1d": <directional probability 1d>,
    "prob_3d": <directional probability 3d>,
    "monte_carlo_p_up": <probability of up>,
    "funding_rate": <funding rate %>,
    "oi_trend": "<rising/falling/flat>",
    "fear_greed_crypto": <F&G value>,
    "dxy": <DXY value>,
    "btc_consensus": "<BUY/SELL/HOLD>",
    "verdict_1_3d": "<bullish/bearish/neutral>",
    "verdict_1_3d_conf": <0.0-1.0>,
    "verdict_1_4w": "<bullish/bearish/neutral>",
    "verdict_1_4w_conf": <0.0-1.0>,
    "data_sources_used": ["agent_summary", "prophecy", "journal", ...],
    "execution_time_sec": <wall clock seconds from step 1 to this step>,
}
with open("data/fin_command_log.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

**Timing:** Record the start time at step 1 (when you check the clock). Compute `execution_time_sec`
as the difference between now and step 1 start time. This tracks how long the full analysis takes.

**Why log:** Over time, this builds a dataset of every analysis — you can track:
- How often each verdict was correct (compare verdict_1_3d with actual price move 3 days later)
- ETH/BTC ratio trends over time
- Whether futures flow signals (funding rate, OI) correlated with correct verdicts
- Whether BTC status at invocation time predicted ETH outcome
- Average execution time and whether it's improving

## Critical rules

- **ETH signal chaos is CONFIRMED.** SELL 5/9 to BUY 5/9 in one check-in. Now TF consensus
  for ETH is pure noise. NEVER trust single-check-in reads. Require 3+ consecutive checks
  in the same direction before treating any ETH signal as meaningful. This is not theoretical —
  it has been observed repeatedly.
- **ETH follows BTC.** Always check BTC status first. If BTC is bearish, ETH bullish signals
  are likely noise. If BTC is breaking out, ETH will likely follow with a lag. The ETH prophecy
  ($4K) depends on BTC breaking out first — always note this dependency.
- **Bold strategy already lost money on ETH.** Bought at $1,978, sold in stages at $1,963,
  $1,928, $1,906. Total loss ~$72/ETH. This history is relevant context — be wary of repeating
  the same mistake (buying on weak signals, selling into capitulation).
- **ETH signal accuracy is near coin-flip (44-54%).** Weight your own judgment and macro context
  much more heavily than raw signal consensus for ETH. A 6-signal consensus on ETH means less
  than a 4-signal consensus on XAG (71-83% accuracy).
- **Funding rate is contrarian when extreme.** Very positive funding = crowded longs = contrarian
  bearish. Very negative funding = crowded shorts = contrarian bullish. Moderate funding = neutral.
- **Futures flow signal (#30) is ETH-specific data.** Use OI trend, funding rate trend, and
  long/short ratio as additional context. These are real market structure signals, not derived
  from price alone.
- **FOMC within 4 days:** Flag prominently in header and verdict. Reduce short-term (1-3d)
  conviction by at least 0.2. Crypto is hyper-sensitive to Fed communication.
- **ETH trades 24/7.** No market hours restriction for the underlying. But the Avanza ETH Tracker
  only trades 09:00-17:25 CET — note this if recommending warrant trades.
- **Adversarial debate is MANDATORY** — never skip it, even when signals are unanimous.
- **27 applicable signals** for ETH (8 core + 19 enhanced, including futures_flow #30).
- **ETH/BTC ratio is the key relative value metric.** If ETH/BTC is declining while both are
  rising in USD, ETH is underperforming — this matters for portfolio allocation decisions.
- **Staking yield context:** ETH staking (~3-5% APR) provides a yield floor that becomes more
  attractive as traditional rates fall. Note this in the bull case when relevant.
- **Do NOT use precomputed context files** — ETH has none. All data comes from
  `agent_summary_compact.json` and the live data sources listed in step 3.
