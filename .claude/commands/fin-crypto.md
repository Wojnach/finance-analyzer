Combined deep crypto analysis — BTC-USD (bitcoin), ETH-USD (ethereum), AND MSTR (MicroStrategy) in a single pass with shared data reads.

All three share macro context (DXY, yields, FOMC, Fear & Greed). BTC and ETH share futures flow data (funding, OI, L/S ratio). MSTR is a leveraged BTC proxy traded as a NASDAQ stock. This command reads all shared data ONCE, then produces per-asset analysis sections. More efficient and more insightful than running separate analyses because cross-asset correlations (BTC leadership, ETH follow-through, MSTR premium dynamics) are analyzed together.

## Prerequisites
- Read `memory/trading_rules.md` FIRST — check per-ticker accuracy rules for ALL THREE tickers before any recommendation.

## Steps

1. **Check the time** — Run `powershell.exe -NoProfile -Command "Get-Date -Format 'dddd yyyy-MM-dd HH:mm:ss'"`
   Record this as `start_time` for execution timing.
   - BTC-USD and ETH-USD: trade 24/7 on crypto exchanges.
   - MSTR: NASDAQ stock, US market hours only **15:30-22:00 CET**. If outside these hours, mark MSTR signals as STALE.
   - Near close (<1h to 22:00 CET): do NOT recommend new MSTR positions. Flag that close is imminent.

1b. **Recall prior verdicts** — Read `data/fin_command_log.jsonl` and extract the last 3 entries covering BTC-USD, ETH-USD, or MSTR:
   ```python
   import json
   tgt = {"BTC-USD", "ETH-USD", "MSTR"}
   entries = []
   with open("data/fin_command_log.jsonl", encoding="utf-8") as f:
       for line in f:
           line = line.strip()
           if not line: continue
           try:
               e = json.loads(line)
               tickers = set(e.get("tickers", []))
               ticker = e.get("ticker", "")
               if ticker in tgt or tickers & tgt:
                   entries.append(e)
           except: pass
   for e in entries[-3:]:
       print(json.dumps(e, indent=2))
   ```
   For each prior entry, compare prior prices against current:
   - Single-ticker entries (fin-btc/fin-eth/fin-mstr): use `price_usd`
   - Multi-ticker entries (fin-crypto): use `btc.price_usd`, `eth.price_usd`, `mstr.price_usd`
   - If `verdict_correct_1d` or `outcome_1d_pct` exists, note the scored result
   - Include a **Prior Verdict Reflection** section early in your output:
     ```
     ### Prior Verdict Reflection
     {date} ({command}): BTC — said {verdict} at ${price} (conf {conf}) → now ${current} ({pct}%) — {RIGHT/WRONG}
     {date} ({command}): ETH — said {verdict} at ${price} (conf {conf}) → now ${current} ({pct}%) — {RIGHT/WRONG}
     ```
   - Use this to calibrate current verdicts — if you've been consistently wrong on a ticker, adjust confidence down
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
   - Your accuracy by source (layer2 vs fin-crypto), regime, confidence level
   - Per-ticker accuracy across ALL sources (BTC-USD, ETH-USD, MSTR, etc.)
   - Anti-patterns (conditions where past verdicts were wrong)
   - Confirmed patterns (conditions where past verdicts were right)
   - Cross-asset patterns (e.g., "when BTC bullish, ETH follows X%")
   - Calibration advice (are you over/underconfident?)

   **Apply these adjustments to ALL THREE assets:**
   - If calibration says OVERCONFIDENT, reduce all verdict confidence by the suggested amount
   - If an anti-pattern matches current conditions, note it in the bear case and reduce confidence
   - If a confirmed pattern matches, note it in the bull case and increase confidence
   - If accuracy for current regime is < 0.5, add a disclaimer: "Note: past verdicts in {regime} regime have been weak ({accuracy}%)"
   - Check cross-asset patterns specifically — these are most valuable in a combined analysis

   If the file doesn't exist or has < 5 verdicts, skip this step (not enough data yet).

3. **Read live data** (parallel, ONCE for all three assets):
   - `data/agent_summary_compact.json` — ALL THREE ticker sections (BTC-USD, ETH-USD, MSTR):
     signals, prices, probabilities, regime, cumulative gains, forecast signals, Monte Carlo,
     signal_reliability. Also read: `macro` section (DXY, yields, FOMC), `futures_data` section
     (BTC + ETH funding rates, OI, L/S ratios), `focus_probabilities` (if BTC is a focus ticker),
     `fundamentals` section (MSTR Alpha Vantage data: P/E, revenue, analyst targets, BTC holdings).
   - `data/system_lessons.json` — learned lessons (if freshened in step 2)
   - `data/prophecy.json` — check `btc_range_2026` / BTC $100K belief AND `eth_follows_btc` / ETH $4K belief
   - `data/portfolio_state.json` — Patient strategy: check BTC-USD, ETH-USD, MSTR holdings
   - `data/portfolio_state_bold.json` — Bold strategy: check BTC-USD, ETH-USD, MSTR holdings
   - `data/portfolio_state_warrants.json` — check XBT Tracker and ETH Tracker positions
   - Last 5 entries from `data/layer2_journal.jsonl` that mention BTC-USD, ETH-USD, or MSTR
     (check `tickers` and `prices` keys — a single journal entry may mention multiple tickers)

3b. **Fetch recent news headlines** — Use WebSearch to get context on what’s driving prices:
   - Search: "bitcoin ethereum crypto market today {date}"
   - Search: "MSTR MicroStrategy stock news today {date}" (only if US market was open in last 24h)
   - Extract the top 3-5 most relevant headlines that explain current price action
   - Include a **News Context** section in the output (after Shared Context, before per-asset analysis):
     ```
     ### News Context
     - {headline 1} ({source}) — {1 sentence on market impact}
     - {headline 2} ({source}) — {1 sentence on market impact}
     - {headline 3} ({source}) — {1 sentence on market impact}
     ```
   - Use these headlines to inform the adversarial debates — news often explains WHY signals are firing
   - If no significant news, note "No major crypto-specific catalysts — price action is technical"

4. **Compute shared derived metrics** (ONCE, used by all three assets):
   - **Crypto Fear & Greed:** Current value from agent_summary. Extreme fear (<20) = contrarian buy signal.
     Extreme greed (>80) = contrarian sell signal.
   - **Stock Fear & Greed:** Separate value — matters for MSTR as a NASDAQ stock.
   - **DXY direction and implication:** DXY up = headwind for crypto. DXY down = tailwind.
     Compute current value, 5d change, direction.
   - **FOMC proximity:** Days until next FOMC. Flag prominently if within 4 days.
   - **BTC/ETH ratio:** Current ETH price / BTC price. Is ETH gaining or losing ground vs BTC?
     Compare to recent history. Rising ratio = ETH outperforming = risk-on within crypto.
     Falling ratio = BTC dominance rising = flight to quality within crypto.
   - **BTC dominance trend:** If available in macro data, note whether BTC dominance is rising
     (risk-off within crypto, altcoins underperforming) or falling (altcoin season beginning).
   - **Overall crypto market regime:** trending-up / trending-down / ranging / high-vol / breakout / capitulation.
     Combine BTC and ETH regime assessments. BTC regime is the primary driver.
   - **Reflection:** Compare previous journal BTC, ETH, and MSTR prices vs current for all three.
     Were the previous theses correct? What happened since?

5. **Per-asset analysis** — For EACH asset (BTC first as leader, then ETH, then MSTR), compute:

   **BTC-USD (always first — BTC leads):**
   - Signal consensus and heatmap from agent_summary_compact.json
   - Futures flow: funding rate, OI trend, L/S ratio, top trader alignment
   - Chronos forecast direction and confidence (note: BTC accuracy is 44-54%, coin-flip territory)
   - Prophecy progress: distance to $100K target, conviction 0.7
   - Monte Carlo probability of up and confidence intervals
   - Key levels: support/resistance, Fibonacci
   - **Proven noise patterns:** BTC 12h BUY phantom (ignore), Now TF rapid flips (never trust single read)
   - **Funding rate:** If extreme (>0.1% or <-0.05%), note contrarian implication

   **ETH-USD (always check BTC status first):**
   - Signal consensus and heatmap from agent_summary_compact.json
   - Futures flow: funding rate, OI trend, L/S ratio
   - Chronos forecast direction and confidence
   - Prophecy progress: distance to $4K target, conviction 0.6. Note: ETH prophecy depends on BTC breaking out first.
   - BTC dependency check: Is ETH leading, following, or diverging from BTC?
   - Monte Carlo probability
   - **Proven noise patterns:** ETH signal chaos (SELL->BUY flips in one check-in), ETH Now TF is pure noise
   - **History:** Bold already lost money on ETH — bought high, sold into capitulation. Note this.

   **MSTR (last — depends on BTC analysis):**
   - Signal consensus and heatmap (if market open; if closed, mark as STALE)
   - Fundamentals from Alpha Vantage: P/E, revenue, revenue growth, analyst price target
   - BTC holdings: approximate number of BTC held by MicroStrategy
   - NAV premium/discount: key metric — is MSTR trading above or below the value of its BTC holdings?
   - BTC correlation: is MSTR tracking BTC or diverging? Beta estimate (~1.5-2.5x)
   - Market status: OPEN or CLOSED. If closed, after-hours vol spikes are NOT actionable.
   - **Fee drag:** 0.10% stocks vs 0.05% crypto — 2x the friction
   - **Leveraged noise:** If BTC signal accuracy is coin-flip (44-54%), MSTR is double-noise

6. **Run adversarial debate** — MANDATORY for EACH asset separately:

   **BTC debate:**
   - **Bull:** Signal consensus, macro tailwinds (DXY weak, yields falling), institutional adoption,
     halving cycle, $100K prophecy momentum, ETF flows
   - **Bear:** DXY strength, rising yields, FOMC proximity, overbought signals, low volume,
     funding rate overleveraged, signal accuracy is coin-flip (44-54%)
   - **Synthesis:** Which side dominates, at each horizon?

   **ETH debate:**
   - **Bull:** BTC breakout follow-through, staking yield, DeFi/L2 catalysts, ETH/BTC ratio
     compression opportunity, $4K prophecy alignment
   - **Bear:** BTC dependency (no independent breakout), signal chaos, competition (SOL/etc),
     Bold lost money on ETH, ETH underperformance trend
   - **Synthesis:** Which side dominates, at each horizon?

   **MSTR debate:**
   - **Bull:** Leveraged BTC upside, NAV discount (if applicable), institutional BTC proxy,
     Michael Saylor conviction, catch-up potential when BTC momentum is strong
   - **Bear:** Stock market risk (NASDAQ correlation), NAV premium (if applicable),
     after-hours noise, double the fee drag, dilution risk, regulatory risk,
     if BTC signal is noise then MSTR is double-noise
   - **Synthesis:** Which side dominates, at each horizon?

7. **Produce combined output** in this exact format:

```
# CRYPTO DEEP ANALYSIS — {date} {time} CET

## Shared Context
**Macro:** DXY {val} ({5d_chg}%) | 10Y {yield}% ({trend}) | Fear & Greed Crypto {val} | Fear & Greed Stock {val} | FOMC {days}d
**Crypto Regime:** {trending-up/down/ranging/high-vol}
**BTC/ETH Ratio:** {current} ({rising/falling} — BTC {leading/lagging})
**US Market:** {OPEN/CLOSED} (MSTR signals {live/stale})

### Portfolio Exposure
Patient: {list any BTC/ETH/MSTR holdings or "no crypto positions"}
Bold: {list any BTC/ETH/MSTR holdings or "no crypto positions"}
Avanza: XBT Tracker {if held, show units + P&L} | ETH Tracker {if held, show units + P&L}

### Learned Lessons (if available)
- Calibration: {over/underconfident advice}
- BTC accuracy: {pct}% ({n} verdicts) | ETH accuracy: {pct}% ({n} verdicts) | MSTR accuracy: {pct}% ({n} verdicts)
- Matching patterns: {any anti-patterns or confirmed patterns matching current conditions}
- Cross-asset: {any cross-asset pattern, e.g., "when BTC breaks out, ETH follows within 24-72h"}

### News Context
{headlines with sources and impact assessment}

---

## BTC — ${price}

### Signals
{consensus} ({XB/YS/ZH}) | wConf {pct}% | Regime: {regime}
Heatmap: {Now->6mo B/S/. chars}
RSI {rsi} | MACD {macd} ({trend}) | BB {position} | Vol {ratio}x | ATR {atr_pct}%
Chronos 24h: {pct}% ({accuracy}% acc, {n} sam) — NOTE: BTC accuracy 44-54% (coin-flip)
Prob: up{3h}% 3h | up{1d}% 1d | up{3d}% 3d
Monte Carlo: P(up)={pct}% | 1d: ${lo}-${hi} | 3d: ${lo}-${hi}

### Prophecy: $100K Target
Progress: {pct}% | Distance: ${gap} | Conviction: 0.7
{checkpoint status summary — which triggered, which pending}

### Futures Flow
Funding: {rate}% ({normal/extreme — if >0.1% or <-0.05%: contrarian flag}) | OI: {trend} | L/S: {ratio} | Top traders: {aligned/diverging}
{if funding extreme: "Funding rate extreme — contrarian signal active"}

### Debate
**Bull:** {1-2 sentences}
**Bear:** {1-2 sentences}
**Synthesis:** {1-2 sentences}

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

## ETH — ${price}

### Signals
{consensus} ({XB/YS/ZH}) | wConf {pct}% | Regime: {regime}
Heatmap: {Now->6mo B/S/. chars}
RSI {rsi} | MACD {macd} ({trend}) | BB {position} | Vol {ratio}x | ATR {atr_pct}%
Chronos 24h: {pct}% ({accuracy}% acc, {n} sam)
Prob: up{3h}% 3h | up{1d}% 1d | up{3d}% 3d
Monte Carlo: P(up)={pct}% | 1d: ${lo}-${hi} | 3d: ${lo}-${hi}

### Prophecy: $4K Target
Progress: {pct}% | Distance: ${gap} | Conviction: 0.6
BTC dependency: {BTC must break out first — is BTC breaking out? YES/NO}

### Futures Flow
Funding: {rate}% | OI: {trend} | L/S: {ratio}
{if BTC and ETH funding rates diverge significantly: note the divergence and implication}

### BTC Dependency Check
ETH/BTC ratio: {current} ({direction} over {period})
BTC status: {bullish/bearish/ranging} — ETH expected to {follow/lag/diverge}
Bold history: lost money on ETH — bought high at $1,978, sold into capitulation at $1,906-$1,962

### Debate
**Bull:** {1-2 sentences}
**Bear:** {1-2 sentences}
**Synthesis:** {1-2 sentences}

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

## MSTR — ${price} {OPEN/CLOSED}

### Signals {if market open, else "STALE — US market closed"}
{consensus} ({XB/YS/ZH}) | wConf {pct}% | Regime: {regime}
Heatmap: {Now->6mo B/S/. chars}
RSI {rsi} | MACD {macd} ({trend}) | BB {position}
{if market closed: "Signals from last close — may gap on open. After-hours vol spikes NOT actionable."}
{if near close (<1h): "NEAR CLOSE — do NOT open new positions"}

### Fundamentals
P/E: {val} | Revenue: ${val} ({growth}%) | Analyst target: ${target}
BTC holdings: ~{n}K BTC | Avg cost/BTC: ~${cost}
NAV premium: {pct}% ({premium is high/normal/discount})
{if NAV discount: "Trading below BTC value — potential catch-up opportunity"}
{if NAV premium >50%: "High premium — paying 50%+ above BTC value, risky"}

### BTC Correlation
Beta: ~{val}x (MSTR typically moves {val}x BTC's daily move)
Correlation today: {strong/weak/diverging}
{if BTC bullish + MSTR cheap (NAV discount): "Catch-up opportunity"}
{if BTC noise + MSTR volatile: "Double-noise — BTC signal accuracy is coin-flip, MSTR amplifies the noise"}

### Debate
**Bull:** {1-2 sentences}
**Bear:** {1-2 sentences}
**Synthesis:** {1-2 sentences}

### Verdict
| Horizon | Bias | Confidence | Note |
|---------|------|------------|------|
| 1-3d | {bullish/bearish/neutral} | {0.0-1.0} | {reason} |
| 1-4w | {bullish/bearish/neutral} | {0.0-1.0} | {reason} |
| 1-3m | {bullish/bearish/neutral} | {0.0-1.0} | {reason} |
| 6-12m | {bullish/bearish/neutral} | {0.0-1.0} | {reason} |

---

## Cross-Asset Summary

**Leader/Follower:**
- BTC: {leading/lagging} — {1 sentence on current BTC stance}
- ETH: {following BTC / diverging / lagging} — {1 sentence on ETH's relative behavior}
- MSTR: {tracking BTC / premium shifting / decoupled} — {1 sentence on MSTR's premium dynamics}

**Which looks best right now:**
- Short-term (1-3d): {BTC/ETH/MSTR} — {reason}
- Medium-term (1-4w): {BTC/ETH/MSTR} — {reason}
- Structural (1-12m): {BTC/ETH/MSTR} — {reason}

**If BTC breaks $100K:**
- ETH implied: ${implied} (at current ETH/BTC ratio of {ratio})
- MSTR implied: ${implied} (at current beta of ~{beta}x)

**Funding Rate Divergence:**
{if BTC and ETH funding rates diverge: explain which is overleveraged and implication}
{if both normal: "No funding rate divergence — no contrarian signal"}

**Reflection:**
{1-2 sentences comparing previous journal theses on BTC, ETH, MSTR vs what actually happened}
```

8. **Log the invocation** — After producing the output, append a SINGLE log entry to `data/fin_command_log.jsonl` with data for all three assets:

```python
import json, datetime, pathlib
entry = {
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "command": "fin-crypto",
    "tickers": ["BTC-USD", "ETH-USD", "MSTR"],
    "crypto_fear_greed": <crypto F&G value>,
    "stock_fear_greed": <stock F&G value>,
    "dxy": <DXY value>,
    "fomc_days": <days until FOMC>,
    "eth_btc_ratio": <ETH price / BTC price>,
    "crypto_regime": "<trending-up/down/ranging/high-vol>",
    "us_market_open": <true/false>,
    "btc": {
        "price_usd": <current BTC price>,
        "signal_consensus": "<BUY/SELL/HOLD>",
        "vote_breakdown": "<XB/YS/ZH>",
        "weighted_confidence": <0.0-1.0>,
        "regime": "<regime>",
        "rsi": <rsi value>,
        "macd": <macd value>,
        "funding_rate": <funding rate %>,
        "oi_trend": "<rising/falling/flat>",
        "chronos_24h_pct": <chronos prediction %>,
        "chronos_accuracy": <chronos accuracy for BTC>,
        "prob_3h": <directional probability 3h>,
        "prob_1d": <directional probability 1d>,
        "prob_3d": <directional probability 3d>,
        "monte_carlo_p_up": <probability of up>,
        "prophecy_progress_pct": <% toward $100K>,
        "verdict_1_3d": "<bullish/bearish/neutral>",
        "verdict_1_3d_conf": <0.0-1.0>,
        "verdict_1_4w": "<bullish/bearish/neutral>",
        "verdict_1_4w_conf": <0.0-1.0>,
    },
    "eth": {
        "price_usd": <current ETH price>,
        "signal_consensus": "<BUY/SELL/HOLD>",
        "vote_breakdown": "<XB/YS/ZH>",
        "weighted_confidence": <0.0-1.0>,
        "regime": "<regime>",
        "rsi": <rsi value>,
        "macd": <macd value>,
        "funding_rate": <funding rate %>,
        "oi_trend": "<rising/falling/flat>",
        "chronos_24h_pct": <chronos prediction %>,
        "chronos_accuracy": <chronos accuracy for ETH>,
        "prob_3h": <directional probability 3h>,
        "prob_1d": <directional probability 1d>,
        "prob_3d": <directional probability 3d>,
        "monte_carlo_p_up": <probability of up>,
        "prophecy_progress_pct": <% toward $4K>,
        "eth_btc_ratio": <ETH/BTC ratio>,
        "verdict_1_3d": "<bullish/bearish/neutral>",
        "verdict_1_3d_conf": <0.0-1.0>,
        "verdict_1_4w": "<bullish/bearish/neutral>",
        "verdict_1_4w_conf": <0.0-1.0>,
    },
    "mstr": {
        "price_usd": <current MSTR price>,
        "signal_consensus": "<BUY/SELL/HOLD>",
        "vote_breakdown": "<XB/YS/ZH>",
        "weighted_confidence": <0.0-1.0>,
        "regime": "<regime>",
        "rsi": <rsi value>,
        "macd": <macd value>,
        "nav_premium_pct": <NAV premium/discount %>,
        "btc_beta": <estimated beta vs BTC>,
        "pe_ratio": <P/E ratio from fundamentals>,
        "analyst_target": <analyst consensus target $>,
        "signals_stale": <true if market closed, false if live>,
        "verdict_1_3d": "<bullish/bearish/neutral>",
        "verdict_1_3d_conf": <0.0-1.0>,
        "verdict_1_4w": "<bullish/bearish/neutral>",
        "verdict_1_4w_conf": <0.0-1.0>,
    },
    "cross_asset": {
        "best_short_term": "<BTC/ETH/MSTR>",
        "best_medium_term": "<BTC/ETH/MSTR>",
        "best_structural": "<BTC/ETH/MSTR>",
        "btc_100k_eth_implied": <implied ETH price>,
        "btc_100k_mstr_implied": <implied MSTR price>,
    },
    "data_sources_used": ["agent_summary", "prophecy", "portfolio_patient", "portfolio_bold", "warrants", "journal", "system_lessons", ...],
    "execution_time_sec": <wall clock seconds from step 1 to this step>,
}
with open("data/fin_command_log.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

**Timing:** Record the start time at step 1 (when you check the clock). Compute `execution_time_sec`
as the difference between now and step 1 start time.

**Why log:** Builds a dataset of every combined crypto analysis for tracking:
- Verdict accuracy for ALL THREE assets (compare with actual price moves)
- BTC/ETH ratio prediction accuracy over time
- MSTR NAV premium tracking
- Which asset the "looks best" call was right about
- BTC leadership timing — does ETH really follow within 24-72h?
- Data availability patterns (MSTR stale when market closed)
- Execution time vs running three separate analyses

## Critical rules

### Shared rules (apply to all three assets)
- **Adversarial debate is MANDATORY** for EACH asset separately — never skip, even when signals are unanimous.
- **DXY inverse correlation applies to all three.** DXY up = headwind. DXY down = tailwind.
- **FOMC within 4 days:** Flag prominently in the shared context header. Reduce short-term (1-3d) conviction by at least 0.2 for ALL THREE assets. Note: "FOMC proximity — reduce position sizing, expect volatility."
- **Trust signals with >70% accuracy** for each ticker. BTC is 44-54% (coin-flip) — always note this.
- **Read data ONCE.** agent_summary_compact.json is read a single time and all three ticker sections plus macro plus futures_data are extracted from that single read.
- **Crypto Fear & Greed applies to BTC and ETH.** Stock Fear & Greed applies to MSTR.

### BTC-specific rules
- **BTC leads — always analyze first.** BTC sets the direction for ETH and MSTR.
- **BTC 12h BUY phantom is proven noise.** Has appeared 20+ times, always fades. Ignore it.
- **BTC Now TF flips rapidly** — never trust single-check reads. Confirmed: BTC/ETH Now TFs swap each check.
- **Funding rate contrarian when extreme:** >0.1% = overleveraged longs (bearish). <-0.05% = overleveraged shorts (bullish).
- **Signal accuracy 44-54% — independent judgment matters more than vote counting.** Always note this in the output. Per-ticker accuracy check is essential.
- **Ministral/ML flips unreliable** — treat as noise unless sustained 3+ consecutive check-ins.

### ETH-specific rules
- **ETH follows BTC — always check BTC status first.** If BTC is bearish, ETH bull case is weak regardless of ETH-specific signals.
- **ETH signal chaos confirmed:** SELL->BUY flips in one check-in. ETH Now TF consensus is pure noise — never trust single-check-in reads.
- **ETH prophecy ($4K) depends on BTC breaking out first.** Always state this dependency.
- **Bold already lost money on ETH.** Bought 5.68 ETH @ $1,978, sold down to $1,906-$1,962. Total ETH loss was part of the -7.09% drawdown. Note this history when considering ETH positions.
- **ETH/BTC ratio direction matters more than absolute ETH price** for relative value assessment.

### MSTR-specific rules
- **STOCK: US hours only (15:30-22:00 CET).** If outside these hours, mark signals as STALE prominently.
- **Near close (<1h to 22:00 CET):** Do NOT recommend new MSTR positions.
- **After-hours vol spikes are NOT actionable** — wait for market session. This is a proven pattern.
- **NAV premium/discount is the key MSTR metric.** A NAV discount means you're buying BTC exposure below spot. A high NAV premium means you're overpaying for BTC exposure.
- **Leveraged BTC proxy — if BTC signal is noise, MSTR is double-noise.** BTC accuracy is coin-flip (44-54%). MSTR amplifies both signal and noise. Always note this.
- **Fee drag 0.10%** for stocks vs 0.05% crypto — double the round-trip friction.
- **Fundamentals matter for MSTR** unlike pure crypto. Check P/E, revenue, analyst targets from Alpha Vantage. MSTR is not just a BTC wrapper — it has a software business too (even if small relative to BTC holdings).

### Cross-asset rules
- **BTC breakout -> ETH follows** (historically with 24-72h lag). If BTC is breaking out and ETH hasn't moved, ETH may be the better short-term play.
- **MSTR premium expands when BTC momentum is strong**, contracts when BTC ranges. Premium direction is a sentiment indicator.
- **If BTC and ETH diverge, check funding rates** — one may be overleveraged. The one with extreme funding is more likely to mean-revert.
- **Always state which of the three looks best** at each horizon in the cross-asset summary. The user needs to allocate between BTC (direct), ETH (direct), MSTR (stock proxy), XBT Tracker (Avanza), and ETH Tracker (Avanza).
- **If all three are ranging with no edge:** Say so clearly. "No crypto setup right now. BTC accuracy is coin-flip, ETH follows BTC, MSTR is double-noise in a range. Wait for structural breakout."
