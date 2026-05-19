# Oil Price Quant Technical Analysis and Event-Aware Forecasting System Spec

## Purpose, scope, and what “quants” actually trade

This document specifies a **local-first** research and implementation blueprint for an oil-price forecasting and trading agent that combines (a) systematic/quant technical analysis, (b) commodity-specific signals (term structure, inventories, spreads), and (c) event/news awareness (FOMC, OPEC decisions, wars/sanctions/shipping disruptions). It is written so you can hand it to Codex/Claude Code to implement a working system end-to-end.

Oil is typically traded and risk-managed in finance via **futures** (and options on futures), not just spot. For example, NYMEX **WTI (CL)** is a benchmark, highly liquid crude oil futures contract; CME’s education materials summarise key mechanics such as trading hours and contract size (1,000 barrels, $0.01 tick size). citeturn1search3turn1search31 The global benchmark **ICE Brent** futures are likewise standardised with contract specs (including a 1,000 barrel contract size). citeturn2search2turn2search6turn2search14

**Critical implication for modelling:** for commodities, price behaviour depends not only on the “spot-like” move, but also on the **futures curve** (contango/backwardation) and the economics of inventories and storage. In futures, return can be usefully decomposed into components commonly discussed as “spot/price return” and “roll yield,” and roll yield is tightly linked to the slope of the term structure. citeturn4search6turn0search13

You asked “how financial systems / quants do technical analysis on oil.” In practice, the consistently recurring “quant oil TA” building blocks are:

A. **Trend / time-series momentum** across multiple horizons (days → months), because it generalises well and is robust in futures. Academic evidence documents statistically significant “time series momentum” across futures (including commodities) with return persistence at 1–12 month horizons and partial reversal at longer horizons. citeturn0search8turn0search4  
B. **Carry / term-structure signals** (backwardation/contango, curve slope, calendar spreads), because they encode inventory tightness and the economics of storage and convenience yield; empirical work links basis/backwardation and prior returns to inventory states and risk premia. citeturn4search2turn0search5  
C. **Event-aware volatility and risk controls** (vol targeting, drawdown constraints, option-implied vol proxies like OVX), because oil regularly experiences regime shifts and event shocks. OVX is explicitly defined as an estimate of expected 30-day volatility derived from USO options using VIX-style methodology. citeturn1search6turn1search14turn1search10  
D. **Fundamental “nowcasting” features** (EIA inventories, rig counts, refining margins) because high-frequency supply/demand proxies often drive short-horizon repricing, especially around scheduled releases. citeturn9search7turn3search2turn2search15turn2search3

This spec is therefore built around a modern quant reality: **oil forecasting is a multi-signal, multi-horizon, regime-dependent problem**. Your system should output **probabilistic forecasts** (distributions or quantiles), not only point estimates, and should separate **direction/return** from **volatility/regime**.

## Data you need locally

Treat “getting the data right” as a first-class problem. Most failed oil trading agents fail here: bad continuous contracts, look-ahead leakage around events, mixing spot with futures, or ignoring term structure and inventories.

### Market data

You want a local store of:

WTI (CL) futures: daily OHLC, settlement, volume, open interest for multiple maturities (front through at least 12 months). WTI is the go-to “screen oil price” benchmark in finance and is commonly used to express views/hedge; CME emphasises its role as the most liquid crude oil contract with broad use for hedging/speculation. citeturn1search3turn2search31

Brent futures: daily OHLC/settlement, volume/OI, multiple maturities for global benchmark context and for the Brent–WTI spread. ICE provides Brent futures product pages and contract spec documents. citeturn2search2turn2search6turn2search14

Options / implied volatility: if you cannot source full implied vol surfaces, you can at least ingest **OVX** as a market-implied volatility proxy. OVX is an estimate of expected 30-day volatility of crude oil as priced via options on USO. citeturn1search6turn1search14turn1search10

### Energy fundamentals and flows

EIA “Weekly Petroleum Status Report” (WPSR) tables (crude inventories, Cushing stocks, refinery inputs/utilisation, imports/exports). EIA explicitly publishes a release schedule: summary PDFs and key tables are released after **10:30 a.m. Eastern on Wednesdays** (with holiday exceptions). citeturn9search7turn9search11turn9search15

EIA API for programmatic ingestion. EIA’s Open Data/API documentation describes the API and its organisation. citeturn2search0turn2search4turn2search16  
Pragmatically you will want the weekly series for total crude stocks (ex-SPR) and Cushing stocks; EIA’s series browser pages expose those weekly series identifiers. citeturn2search12turn2search8

Baker Hughes rig count: released weekly; Baker Hughes describes the rig count as a weekly census and notes the release cadence (North America rig count released weekly). citeturn3search2turn3search14

Refining margins / crack spreads: crack spreads approximate the theoretical refining margin (difference between refined products and crude input). CME explains the crack spread concept, and CME’s calculator defines it as a theoretical refining margin. citeturn2search3turn2search15  
Even if you don’t trade crack spreads, you should **use them as features** (refinery demand/pull for crude often shows up here).

### Positioning and market participants

CFTC Commitments of Traders (COT): weekly breakdowns of positions by trader category; CFTC provides the report hub and petroleum-specific reports. citeturn0search2turn0search6turn0search14  
For oil-specific insight, CFTC-commissioned research has examined relationships between price and positions using CFTC data (e.g., price/position lead-lag tests). citeturn7search0

### Macro and scheduled event data

FOMC meeting calendar: the Federal Reserve publishes official meeting calendars. citeturn1search0turn1search12  
For “surprise” measurement, use a monetary-policy event-study dataset: the San Francisco Fed provides a US Monetary Policy Event-Study Database of high-frequency changes around FOMC communication events. citeturn6search10  
If you implement event-study features, note that modern econometrics cautions that narrow windows alone do not guarantee clean identification; be disciplined about confounds and event-window design. citeturn6search6turn6search18

OPEC and OPEC+ decisions: ingest official press releases and (if possible) meeting calendars from OPEC sources (these often move the curve). OPEC’s press releases document decisions such as production adjustments and monitoring commitments. citeturn1search13turn1search5  
Options-implied “market pricing of OPEC outcomes”: CME’s OPEC Watch uses WTI option prices to compute probabilities of meeting outcomes. citeturn9search2turn9search6turn9search18  
You can’t always replicate it perfectly without full options data, but it’s a useful reference design: **extract event expectations from options markets** when you can.

### Geopolitics, uncertainty, and “war risk” proxies

News-based geopolitical risk index (GPR): Caldara & Iacoviello construct a news-based measure of adverse geopolitical events and associated risks (AER 2022), and publish data. citeturn3search4turn3search9turn3search16  
Central-bank analysis explicitly links the GPR index to oil prices in descriptive comparisons (e.g., ECB analysis plotting global GPR vs Brent). citeturn3search1

Economic Policy Uncertainty (EPU) indices: policyuncertainty.com hosts EPU indices and explains the measurement concept; it also hosts related uncertainty indices (including oil-related uncertainty pages). citeturn9search0turn9search1turn9search12

### Long-horizon reference forecasts (optional, for benchmarking/ensembling)

EIA Short-Term Energy Outlook (STEO): EIA provides oil market commentary and short-term forecasts. citeturn6search7turn6search31  
EIA also publishes a methodology note describing how STEO forecasts crude prices: Brent monthly average forecasts are formulated and WTI is forecast via a Brent–WTI spread forecast. citeturn6search15  
Treat this as a **benchmark** to beat or as an ensemble component; don’t blindly copy it.

## Signal library for oil

This is the “most important” section for Codex/Claude Code: it defines the signals worth building, their equations, and the reasoning behind their priority.

### Core notation

Let \(P_t\) be a price (prefer settlement) for the target instrument (e.g., continuous front-month CL). Use log returns:

\[
r_{t,h} = \ln(P_t) - \ln(P_{t-h})
\]

Let \(F_t(T)\) be the futures price at time \(t\) for maturity \(T\), and \(F_t^{(k)}\) denote the \(k\)-th listed maturity (front-month = \(k=1\), next = \(k=2\), etc.).

### Signals quants consistently prioritise

#### Trend and time-series momentum

Why: broad futures evidence and practical CTA usage. Time-series momentum is documented across futures (including commodities), typically with lookbacks 1–12 months, and shows strong cross-asset robustness. citeturn0search8turn0search4turn0search12

Implement multi-horizon trend features:

- **TSMOM sign signal** (classic):  
  \[
  s^{\text{tsmom}}_{t,L} = \operatorname{sign}(r_{t,L})
  \]
  where \(L\) could be 21d, 63d, 126d, 252d. You then volatility-scale positions (see risk section).

- **Moving-average slope / crossover**:  
  \[
  \text{MA}_L(t)=\frac{1}{L}\sum_{i=0}^{L-1} P_{t-i}
  \]
  Features: \(\text{MA}_{L_1}-\text{MA}_{L_2}\), MA slope, distance of price to MA in stdev units.

- **Breakout / trading range** (Donchian):  
  Use highest high / lowest low over \(L\) days for breakout strength.  
  Classic technical rules (moving averages, trading range breaks) have long been studied in finance, and a key “quant lesson” is to **define them computationally and test statistically**, not by eyeballing charts. citeturn8search1turn8search0

**Patterns to “look for” (quant framing):** not subjective shapes, but measurable regimes such as: persistent drift (trend), post-shock overshoot and partial reversal, and volatility clustering. The “foundations” approach is to turn chart patterns into algorithmic detection and inference. citeturn8search1turn8search5

#### Carry, basis, and term structure

Why: for commodities, curve shape is often an information-rich proxy for inventory tightness and convenience yield; empirical commodity research links basis/backwardation and prior returns to inventory states and risk premia. citeturn4search2turn0search5  
CME’s educational material ties roll yield to term structure slope and clarifies misconceptions about “rolling.” citeturn4search6turn0search13

Implement (choose 3–6 variants; don’t overdo it):

- **Simple curve slope** (front vs deferred):  
  \[
  \text{Slope}_{t} = \ln\left(\frac{F_t^{(k_{\text{far}})}}{F_t^{(k_{\text{near}})}}\right)
  \]
  often \(k_{\text{near}}=1\), \(k_{\text{far}}=3\) or \(6\).

- **Annualised carry proxy**:
  \[
  \text{Carry}_{t} = \frac{\ln(F_t^{(far)} / F_t^{(near)})}{\Delta T}\;\;\;(\Delta T \text{ in years})
  \]

- **Calendar spread levels and changes**:  
  \(\text{CS}_{t}^{(1,2)} = F_t^{(2)}-F_t^{(1)}\) and \(\Delta \text{CS}\).  
  These spreads frequently react to inventory news and supply shocks.

- **Roll-yield proxy**: depending on your continuous contract roll method, approximate roll pressure from the curve slope at roll times; pair with open interest “roll” measures when you can.

#### Inventory and flow “surprise” features

Why: oil reprices around scheduled inventory releases; EIA provides a known release cadence (important for correct timestamping). citeturn9search7turn9search15

You want both **levels** and **changes**, and ideally a **surprise** component.

- Levels: total US crude inventories ex-SPR; Cushing stocks (WTI delivery nexus). citeturn2search8turn2search12  
- Changes: weekly deltas and seasonally-aware z-scores (relative to 5-year seasonal norms if you compute them locally).  
- Surprise: \(\text{Actual} - \text{Consensus}\) if you can source consensus; otherwise proxy with an internal nowcast model (e.g., use API weekly bulletin as an informational lead, or use rolling autoregressive forecasts of inventory change).

#### Volatility and tail-risk signals

Why: oil volatility is regime-based and event-driven; implied volatility indices encode market expectations. OVX is explicitly defined as a 30-day expected volatility estimate derived from USO options. citeturn1search6turn1search14turn1search10  
Academic work in oil markets compares OVX with GARCH-family models for volatility prediction. citeturn8search3

Implement:

- **Realised volatility**:  
  \[
  \sigma^{\text{rv}}_{t,L} = \sqrt{\frac{252}{L}\sum_{i=1}^{L}(r_{t-i,1})^2}
  \]
  (adapt constants for daily vs intraday).

- **Range-based volatility** (uses OHLC, robust for noisy intraday).

- **Implied vol level**: OVX (if available). Also compute implied-realised spread: \( \text{IV} - \text{RV}\).

Use volatility for: (a) **forecasting**, (b) **position sizing**, (c) **regime switching**.

#### Cross-market relative value and “oil complex” spreads

Why: oil is not one market. Quants exploit related instruments for confirmation, regime detection, and sometimes predictive power.

- **Brent–WTI spread**: global vs US constraints and logistics; include both level and momentum.

- **Crack spreads**: proxies for refinery demand and margins. CME describes crack spreads as the spread between refined products and crude oil and ties them to the refining process; CME’s crack spread calculator frames it as a theoretical refining margin. citeturn2search3turn2search15

- **Refined products (RBOB, ULSD) momentum/carry** as features: often help detect demand-side regimes.

#### Positioning and “crowding” features

Use CFTC COT positioning deltas and z-scores (commercials vs non-commercials), scaled by open interest. CFTC provides the COT framework and petroleum reports. citeturn0search2turn0search6turn0search14  
Oil-specific research using CFTC data has tested lead/lag relationships between positions and prices, reinforcing that positioning is a meaningful feature class (even if not always a clean causal driver). citeturn7search0

Recommended features:

- Net spec position: \( \text{NonCommLong}-\text{NonCommShort} \)  
- Weekly change in net position  
- Positioning percentile over trailing 3–5y  
- “Crowding risk” flag: extreme percentile + rising vol + trend exhaustion

#### Event and news signals

You explicitly want the agent to account for: FOMC meetings, wars/sanctions/shipping disruptions, and major OPEC decisions.

The quant approach is: **represent event risk as a structured feature set**:

Prescheduled events (known calendars):  
- FOMC meeting dates from the Federal Reserve. citeturn1search0turn1search12  
- EIA inventory release schedule (10:30 ET Wednesdays). citeturn9search7turn9search15  
- OPEC meeting dates/decision windows (ingest from official comms; optionally use CME OPEC Watch for market-implied “event distribution” design inspiration). citeturn1search13turn9search6turn9search2  

Unscheduled events (news-driven spikes):  
- Use a geopolitical risk proxy (GPR) as a slow-moving “background risk” feature and combine it with your own real-time news classifier. citeturn3search4turn3search16turn3search1  
- Use EPU indices as broader uncertainty regime features. citeturn9search0turn9search12

NLP implementation primitives:  
- **FinBERT** (financial-text sentiment via BERT) can be used as a strong baseline for sentiment labelling, and is explicitly proposed and evaluated for financial sentiment tasks. citeturn6search0turn6search16turn6search20  
- **Loughran–McDonald financial sentiment lexicons** are widely used finance-specific dictionaries; Notre Dame hosts the master dictionary and sentiment lists (updated through 2024). citeturn6search1

Practical “event-to-feature” mapping (minimum viable):

- `event_flag[t]` for each event type (EIA, FOMC, OPEC, major geopolitics).  
- `time_to_event` (days/hours) and `time_since_event`.  
- `surprise` where measurable (rate surprise around FOMC using high-frequency event-study data; inventory surprise). citeturn6search10turn6search6  
- `news_sentiment_score` (FinBERT + lexicon ensemble) and `news_topic_probs` (supply disruption / demand shock / sanctions / shipping / macro tightening). citeturn6search0turn6search1turn3search4

## Modelling stack by horizon

A good oil system is **not one model**. It is a **model stack** with: baseline(s) + short-horizon forecaster + long-horizon forecaster + regime/volatility model + event overlay.

### Targets: predict returns and distribution, not just price

Short-horizon: predict \(r_{t,h}\) for \(h \in \{1,5,10,20\}\) trading days and/or direction \( \mathbb{1}[r_{t,h}>0]\).  
Long-horizon: predict monthly returns (1–12 months) and/or price levels via integrated return forecasts.

Also forecast volatility (next 5–20d realised vol) because position sizing depends on it and oil risk is regime-based. Using OVX as an implied-vol input is a well-defined approach. citeturn1search6turn1search14

### Baselines (must implement; used to detect overfitting)

- Random walk / “no change” for price, or zero-mean for returns.
- Seasonal naive on monthly horizon (oil/energy can have seasonality through demand/refining cycles; encode seasonality explicitly instead of letting ML hallucinate it).
- AR/ARMA on returns.
- Simple trend + carry linear model.

### Short-term (days to weeks): what works in practice

Use supervised ML on engineered features (trend/carry/vol/inventory/events/news). Models: gradient-boosted trees (fast, robust), regularised linear/logistic models (interpretable), and optionally a small neural model if you have enough data.

Event overlays:

- **EIA inventory days**: train a specialised “inventory reaction” model only on EIA release windows; because the distribution of returns and the feature importances differ on release days (regime shift). EIA’s fixed release schedule makes it feasible to define clean event windows. citeturn9search7turn9search15  
- **FOMC days**: incorporate `rate_surprise`/`path_surprise` from event-study data (USMPD), and/or treat FOMC as a volatility regime change. citeturn6search10turn6search6

### Long-term (months): structure matters more

For long horizons, oil returns are profoundly shaped by macro and supply/demand shocks. A key macro insight: “not all oil price shocks are alike” (supply vs aggregate demand vs oil-specific demand), and structural VAR approaches have been used to disentangle them. citeturn5search6turn5search2  
Related macro literature analyses the causes and consequences of major oil shocks (e.g., 2007–08), reinforcing that long-horizon oil moves often reflect underlying demand and production constraints, not just chart patterns. citeturn5search7turn5search3

Implementation option sets:

- **Structural VAR module** (monthly): replicate Kilian-style decomposition to generate demand/supply shock features and feed them into a forecasting layer. citeturn5search6turn5search2  
- **Macro-feature ML module**: regress monthly oil returns on term-structure carry, inventory z-scores, rig count changes, EPU/GPR, and global growth proxies.
- **Benchmark overlay**: ingest STEO forecasts and/or forecast deltas relative to STEO; EIA documents how STEO forecasts crude prices (Brent first, then WTI spread). citeturn6search15turn6search31

### What “successful approaches” do differently

Successful systematic oil and commodity programmes tend to:

- Prioritise **robust signals** (trend/time-series momentum + term structure/carry) that generalise and have economic intuition, rather than brittle chart patterns. Time-series momentum robustness in futures is documented extensively. citeturn0search8turn0search4  
- Treat commodities as **term-structure markets**, explicitly modelling backwardation/contango and roll yield mechanics rather than modelling a single spot series. CME explicitly connects roll yield to term structure slope and shows how roll-yield thinking can be used in strategy design. citeturn4search6turn0search13  
- Use **strict risk management and volatility scaling**, because oil can gap and experience storage/logistics-driven discontinuities; EIA’s analysis of the April 2020 negative WTI episode highlights how storage and liquidity constraints can dominate price formation in extreme states. citeturn7search3  
- Use **event-aware risk controls** around scheduled releases (EIA, FOMC, OPEC) and explicit geopolitical/uncertainty proxies (GPR/EPU) for regime identification. citeturn9search7turn1search0turn3search4turn9search0  
- Rely on extensive out-of-sample testing and do not “continuously tweak” rules to recent regimes.

## Backtesting and risk management rules you should hard-code

### Continuous futures, rolling, and label correctness

You cannot backtest oil futures as if they were a single perpetual ticker without a roll method. Use either:

- A transparent “rolling index” methodology (roll schedule and weights), or
- A liquid-contract stitching approach (roll when next contract volume/OI surpasses front), with explicit price adjustments only for chart continuity.

CME publishes methodology for rolling futures indices (continuous rolling performance representation). citeturn4search9turn4search16  
Separately, CME’s education also covers futures expiration/roll concepts at a high level (useful for avoiding accidental physical delivery assumptions). citeturn4search1turn4search28

**Rule:** store both (a) back-adjusted “continuous price” for indicator calculation and (b) tradeable “contract-level” series for realistic PnL, roll slippage, and liquidity checks.

### Futures return decomposition and why it matters

Use the concept that futures total return can be understood via price movement plus term-structure effects (“roll yield”), and that roll yield is directly related to term structure. This is essential when your model uses carry/basis features and when you test “long-only” vs “long/short” strategies. citeturn4search6turn0search13

### Position sizing and constraints

At minimum implement:

- **Volatility targeting**:  
  \[
  w_t = \text{clip}\left(\frac{\sigma^*}{\hat{\sigma}_{t}},\; w_{\min},\; w_{\max}\right)
  \]
  where \(\hat{\sigma}_t\) is forecast vol (RV/GARCH/OVX-informed). citeturn8search3turn1search14

- **Event risk haircut**: reduce max leverage on:
  - EIA release window (inventory day)
  - FOMC announcement day
  - OPEC meeting window  
  (defined by your calendars). citeturn9search7turn1search0turn9search6

- **Circuit breakers**:
  - Stop trading (or go to minimal exposure) if realised vol > X percentile and liquidity proxies deteriorate.
  - Drawdown-based de-risking.
  - “Curve stress” flags: extreme contango/backwardation regimes.

### Evaluation: separate forecast skill from trading performance

Forecast metrics:
- RMSE/MAE on returns and price
- Directional accuracy
- Calibration (reliability of predicted probabilities)
- Quantile loss / pinball loss if you output quantiles

Trading metrics:
- Net Sharpe, max drawdown, tail losses
- Performance split by regime: high vs low vol, backwardation vs contango, high vs low GPR/EPU.

Always benchmark against:
- Naive baseline
- Pure trend system (multi-horizon TSMOM)
- Pure carry system  
Because time-series momentum and term structure are strong “simple” baselines in futures. citeturn0search8turn4search2turn4search6

## Local-first system architecture blueprint and agent profile

### Repository structure (recommended)

```text
oil-agent/
  data/
    raw/
      futures/
      options/
      eia/
      cot/
      macro/
      geopolitics_uncertainty/
      news/
    processed/
    feature_store/
  notebooks/                      # optional research
  src/
    config/
    ingestion/
      futures_ingest.py
      eia_ingest.py
      cot_ingest.py
      macro_ingest.py
      news_ingest.py
    contracts/
      continuous_futures.py
      roll_calendar.py
    features/
      trend.py
      carry_term_structure.py
      inventory.py
      spreads.py
      positioning.py
      volatility.py
      events.py
      nlp_news.py
    models/
      baselines.py
      short_horizon.py
      long_horizon.py
      regime.py
      ensemble.py
    backtest/
      pnl_engine.py
      cost_model.py
      risk.py
      reports.py
    agent/
      decision_policy.py
      prompts/
        news_analyst.md
        trading_policy.md
  runs/
  tests/
```

### Data ingestion instructions (implementation-level)

Market data:
- Prefer settlement-based daily bars for modelling; optional intraday for event windows.
- Store the full futures curve (multiple maturities) so carry/slope/spreads are computable.
- Validate contract specs and calendars (WTI CL basics are described by CME). citeturn1search31turn1search3

EIA:
- Pull WPSR tables weekly and store as time-stamped data. Use EIA’s release schedule for correct time alignment. citeturn9search7turn9search15  
- Use EIA API for series pulls; build a mapping of the series IDs you need using EIA’s documentation/pages. citeturn2search0turn2search12turn2search8

COT:
- Download weekly petroleum COT data from CFTC; normalise and create time series features. citeturn0search2turn0search6

Macro/events:
- Parse the FOMC calendar from the Fed. citeturn1search0turn1search12  
- Ingest the SF Fed USMPD event-study dataset for FOMC “surprise” features. citeturn6search10  
- Scrape/ingest OPEC press releases and key meeting dates. citeturn1search13turn1search5

Uncertainty/geopolitics:
- Ingest GPR index time series. citeturn3search16turn3search4  
- Ingest EPU indices (US and global). citeturn9search12turn9search20

### News ingestion and NLP “agent” design

Because you will run everything locally, you should separate **news collection** from **news interpretation**:

News collection:
- Use licensed feeds where possible, or RSS from public sources you are permitted to store.
- Store raw text + timestamp + source + dedup hash.
- Maintain a symbol/topic mapping (WTI/Brent, OPEC, “Strait of Hormuz”, “sanctions”, “pipeline”, “refinery outage”, “SPR”, “OPEC+”, etc.).

News interpretation (NLP features):
- Use a sentiment baseline (FinBERT) and optionally lexicon scoring (Loughran–McDonald) as a second view. citeturn6search0turn6search1  
- Train a topic classifier to produce probabilities for:
  - supply disruption
  - demand shock / recession risk
  - OPEC policy / quotas
  - shipping / chokepoints
  - sanctions / trade restrictions
  - monetary policy tightening / easing
  - inventory/storage constraints  

Geopolitical proxy overlay:
- Blend real-time news scores with GPR (slow-moving risk regime). citeturn3search4turn3search1

### Trading decision policy (forecast-to-trade)

A robust “agent policy” in oil typically does:

- Uses ensemble forecasts for each horizon (direction + magnitude + vol).
- Converts forecast to a target position using volatility targeting and event haircuts.
- Enforces hard risk constraints and disables trading in broken data states.

Minimal policy sketch:

```text
Inputs at time t:
  - short_term_return_forecast (1d, 5d, 10d)
  - long_term_return_forecast (1m, 3m, 6m)
  - vol_forecast (next 10d)
  - regime_probs (trend regime / mean-revert regime / crisis regime)
  - event_flags (EIA, FOMC, OPEC, high GPR/EPU)
Output:
  - target_position in CL (and optionally Brent, spreads)
Rules:
  - base_position = f(forecast_mean, forecast_uncertainty)
  - position_size scaled by vol targeting
  - haircut positions around event windows
  - if “crisis regime” prob high -> reduce risk / widen stops / prefer options hedges if supported
```

### Safety and realism constraints (do not skip)

This system is for research/engineering; it is **not guaranteed to predict oil** and should not be treated as investment advice.

Implement explicit “reality checks” because oil markets can move due to logistics and storage constraints and can behave non-intuitively in extreme regimes (e.g., the April 2020 negative WTI front-month event was linked to low liquidity and limited storage availability). citeturn7search3

Finally, if you incorporate “war / invasion / takeover” narratives, treat them as **uncertain claims** until verified by high-quality sources; your agent should score credibility and avoid acting on single-source sensational headlines. A robust alternative is to rely on diversified sources plus systematic proxies like GPR/EPU and observable market-implied risk (vol, skew, curve dislocations). citeturn3search4turn9search12turn1search14