# On the Absence of Exploitable Directional Predictability in Cryptocurrency and Precious-Metal Markets: A Multi-Model, Multi-Modality Walk-Forward Study

**Finance-Analyzer Research Note · July 2026**

---

## Abstract

We conducted a systematic evaluation of whether short- to medium-horizon price _direction_ (up/down) is predictable from publicly available data for four liquid instruments — Bitcoin (BTC), Ethereum (ETH), gold (XAU), and silver (XAG) — using a common walk-forward backtesting harness. Eight predictive systems spanning five distinct model families and four input modalities were evaluated on identical time grids with identical outcome labelling: large language models (a reasoning LLM, an instruct LLM, and, in progress, a 35-billion-parameter mixture-of-experts model), gradient-boosted decision trees, a time-series quantile foundation model, a candlestick foundation model, and hand-engineered order-flow and positioning feature sets. Across approximately 135,000 verified out-of-sample predictions, **no model, on any instrument, at any horizon, produced directional accuracy whose 95% confidence interval clears the 60% deployment threshold; 0 of 153 scoreable cells cleared the bar.** Pooled accuracy for every system falls between 48.7% and 50.6%, statistically indistinguishable from a coin toss. We further show that a cost-aware confidence filter — the most commonly cited rescue for weak classifiers — does not produce positive after-cost expectancy on our data. We interpret this as a strong, empirically grounded null result: for these instruments and horizons, no tested combination of model capacity and price-derived input contains exploitable directional signal. We document the methodology in detail, including three measurement artifacts that produced _false positive_ accuracy in prior work and were corrected here, and we derive the break-even accuracy thresholds that any deployable strategy must exceed after transaction costs.

---

## 1. Introduction

The finance-analyzer project maintains a live, autonomous signal-generation system trading simulated and small real positions across cryptocurrency and precious-metal instruments. A recurring question in its development has been whether increasingly capable predictive models — in particular large language models — can forecast price direction well enough to drive profitable entries.

An earlier internal result had attributed 66.7% directional accuracy to a small reasoning LLM, which motivated its promotion to a live voting signal. Subsequent investigation (Section 5) revealed this figure to be a measurement artifact rather than genuine skill. This finding prompted a complete, methodologically hardened re-evaluation: rather than testing one model against a possibly-flawed pipeline, we built a single certified harness and ran a matrix of qualitatively different models and inputs through it, so that any genuine signal would appear as separation from a common baseline, and any measurement error would be shared and therefore visible.

This paper reports the results of that matrix.

### 1.1 Research question

> For instrument _i_ ∈ {BTC, ETH, XAU, XAG}, candle interval _c_ ∈ {1h, 4h, 1d}, and forecast horizon _h_, does any tested predictive system produce directional accuracy whose Wilson 95% confidence lower bound exceeds 60%, on strictly out-of-sample data, after accounting for the transaction costs specific to the tradeable instrument?

The 60% CI-lower-bound criterion is the project's pre-registered deployment gate. It is deliberately stricter than "accuracy > 50%" for reasons developed in Section 6.

---

## 2. Materials

### 2.1 Instruments and data sources

| Instrument | Symbol  | Source                      | History available               |
| ---------- | ------- | --------------------------- | ------------------------------- |
| Bitcoin    | BTC-USD | Binance spot klines         | Full window (2025-08 → 2026-07) |
| Ethereum   | ETH-USD | Binance spot klines         | Full window                     |
| Gold       | XAU-USD | Binance FAPI (XAUUSDT perp) | From listing 2025-12-11         |
| Silver     | XAG-USD | Binance FAPI (XAGUSDT perp) | From listing 2026-01-07         |

Cross-asset covariates (US Dollar Index, S&P 500 proxy) were drawn from yfinance hourly bars. The Fear & Greed index was drawn from alternative.me. All price outcomes were computed against exchange candle closes.

### 2.2 Predictive systems evaluated

| Family                       | System                       | Parameters            | Input modality                                               |
| ---------------------------- | ---------------------------- | --------------------- | ------------------------------------------------------------ |
| Reasoning LLM                | Phi-4-mini-reasoning         | 3.8B                  | Technical-indicator prompt                                   |
| Instruct LLM                 | Phi-4-mini-instruct          | 3.8B                  | Technical-indicator prompt                                   |
| MoE LLM (in progress)        | Qwen3.6-35B-A3B              | 35B total / 3B active | Technical-indicator prompt                                   |
| Gradient trees               | XGBoost (per-horizon binary) | —                     | Technical indicators                                         |
| TS foundation model          | Chronos-Bolt                 | 120M                  | Raw close-price series                                       |
| Candlestick foundation model | Kronos-base                  | 102M                  | Raw OHLCV tokens                                             |
| Engineered — flow            | XGBoost + order-flow         | —                     | Kline taker-buy microstructure                               |
| Engineered — positioning     | XGBoost + funding/OI         | —                     | Perp funding, open interest, trader positioning, cross-asset |

Additional LLMs were certified and staged (Fin-R1, Qwen3-8B, Ministral-3-8B, CryptoTrader-LM LoRA, finance-Llama-8B, Fin-o1-8B, Gemma-3/4) but not run to completion once the null result stabilised across the first five families; their inclusion was projected to require 400–700 GPU-hours for no expected change in conclusion.

---

## 3. Methods

### 3.1 Walk-forward protocol

For each (instrument, interval) pair, prediction timestamps were placed on a fixed grid. At each timestamp _t_:

1. **Inputs** were computed using only candle data with open time ≤ _t_. For models requiring training (all XGBoost variants), the training set comprised only feature rows whose outcome horizon had _fully resolved_ before _t_ — an expanding window with strict temporal separation between training labels and the prediction point.
2. **Outcomes** were computed forward: the signed percentage change from the close of the candle containing _t_ to the close of the first candle at or beyond _t + h_, for each horizon _h_ in the interval's horizon set.
3. **Scoring**: a BUY vote was scored correct when the realised return was positive; a SELL vote when negative. HOLD and ABSTAIN votes were excluded from the accuracy denominator, so reported accuracy is conditional on the model committing to a direction.

Horizon sets were matched to the candle interval: 1h → {1h, 3h, 24h}; 4h → {4h, 12h, 72h}; 1d → {24h, 72h, 168h}.

### 3.2 Certification gate

Before any large run, every LLM passed a three-part certification, introduced specifically to prevent the class of error described in Section 5:

- **Template verification**: the GGUF model file's embedded chat template was extracted and compared byte-for-byte against the official tokenizer configuration published by the model authors. All models routed through the `/v1/chat/completions` endpoint so that the inference server applied each model's own verified template, eliminating hand-built prompt-markup drift.
- **Sampling verification**: temperature, top-p, and top-k were set to each model's officially recommended values, confirmed against published generation configs.
- **Output inspection**: a small "V0" micro-run retained raw model outputs, which were read individually to confirm coherent reasoning, correct handling of chain-of-thought delimiters, and agreement between the parsed vote and the model's stated conclusion.

### 3.3 Outcome-computation validation

The outcome-labelling code was independently reimplemented and cross-checked against freshly downloaded price data. All **2,799 of 2,799** sampled outcome cells matched to three-decimal precision, confirming that the dependent variable was computed correctly and free of look-ahead contamination.

### 3.4 Statistics

Directional accuracy is reported with Wilson score 95% confidence intervals. The deployment criterion is the _lower bound_ of this interval exceeding 60%, which controls for small-sample optimism in low-vote cells. After-cost expectancy is reported in basis points per trade.

---

## 4. Results

### 4.1 Primary result: five model families, all null

Pooled over all instruments, intervals, and horizons (clean grids only):

| Family               | Architecture           | Out-of-sample votes | Pooled accuracy | Wilson CI-low |
| -------------------- | ---------------------- | ------------------: | --------------: | ------------: |
| Phi-4-mini-reasoning | reasoning LLM 3.8B     |               5,427 |           48.7% |         47.4% |
| Phi-4-mini-instruct  | instruct LLM 3.8B      |               2,205 |           50.1% |         48.0% |
| XGBoost              | boosted trees          |              19,325 |           50.1% |         49.4% |
| Chronos-Bolt         | TS transformer 120M    |              14,567 |           50.2% |         49.4% |
| Kronos-base          | OHLCV token model 102M |              15,788 |           50.3% |         49.5% |

**Across all three generated per-cell tables, 0 of 153 scoreable cells (n ≥ 30) produced a Wilson lower bound ≥ 60%.** No instrument, interval, or horizon was exceptional; the best single cells (e.g. gold 4h at 4h horizon, 56.4%, CI-low 50.3%) remain below the bar and are consistent with sampling noise given the number of cells examined.

### 4.2 Input-modality pivot: order-flow and positioning add nothing

Having exhausted model _capacity_ as a lever, we tested model _input_. A key discovery was that the taker-buy volume fields — a direct order-flow imbalance proxy — had been fetched in every candle request throughout the project's history but never read by any model. We built three feature sets and compared them on identical grids:

| Feature set               | Inputs                                                                             | Crypto pooled | Metals pooled |
| ------------------------- | ---------------------------------------------------------------------------------- | ------------: | ------------: |
| Flow only                 | Kline taker-buy imbalance, trade intensity, realised-vol structure                 |         50.2% |         50.1% |
| Flow + indicators         | Above + RSI/MACD/EMA/BB/F&G                                                        |         50.6% |         50.1% |
| Positioning + cross-asset | Funding rate, open interest, trader long/short ratios, DXY, SPY, gold/silver ratio |         48.8% |         49.8% |

Order-flow features did not separate from the indicator baseline. Perp-market positioning and cross-asset features — information genuinely absent from price series — performed _below_ coinflip on the pooled crypto set. After-cost expectancy was negative across all feature sets and horizons.

### 4.3 The confidence-filter hypothesis fails

A widely cited method for salvaging a weak directional classifier is to trade only its high-confidence predictions. We swept the XGBoost predicted-probability threshold across all ~19,000 predictions:

| Confidence cutoff | Votes retained | Accuracy | After-cost expectancy |
| ----------------- | -------------: | -------: | --------------------: |
| ≥ 0.50            |         19,325 |    49.7% |                −14 bp |
| ≥ 0.60            |          6,376 |    50.1% |                −19 bp |
| ≥ 0.70            |          2,162 |    50.8% |                −12 bp |

Accuracy remained flat and expectancy remained negative at every conviction level. On our data, gradient-tree probability magnitude carries no selectivity signal.

### 4.4 A note on abstention

The instruct LLM voted on only ~12% of prompts (88% HOLD). This selective behaviour initially appeared promising — early flawed-grid measurements showed 61–72% on the voted subset. On clean grids with correct horizon alignment, the voted-subset accuracy collapsed to 50.1% (n = 2,205). The apparent "selectivity edge" was an artifact of the same grid flaw described in Section 5.2. We note, however, that abstention-driven selectivity remains the _only_ mechanism in the study that ever produced above-bar point estimates, and distinguishing genuine contextual abstention from statistical luck would require substantially more voted-subset data than any single model produced.

---

## 5. Measurement artifacts corrected (methodological contribution)

A central lesson of this study is that **retirement and promotion decisions in the project's history were repeatedly driven by broken measurement rather than model quality.** We document three, because the corrections are more broadly instructive than the individual verdicts.

### 5.1 The 66.7% artifact (prompt-template contamination)

The reasoning LLM's headline 66.7% accuracy was produced by a pipeline that (a) fed every model a _different_ model's hand-built chat template, (b) truncated the reasoning chain-of-thought at an insufficient token limit, and (c) ran the inference server with a context window too small to contain the reasoning trace. The result was ~87–99% abstention on a biased ~11% subsample, whose accuracy was not representative of the model's behaviour under correct conditions. With a certified template, adequate context (16,384 tokens), and correct sampling, the same model votes on nearly all prompts and scores 48.7%.

### 5.2 Horizon mislabelling (grid–outcome misalignment)

An early crypto grid used daily candles stepped every 8 hours, producing three identical prompts per day whose outcomes were scored at inconsistent true horizons (a "24h" label spanning 24h for one stamp but 48h for another). This both inflated the effective sample size through non-independent resampling and mismatched predictions to outcomes. Correct grids align stamps to the candle boundary so that each prediction is scored at its true horizon.

### 5.3 Fallback masquerading as model output

The retired time-series "forecast" signal had, since a code change, substituted a simple moving-average-slope vote whenever the underlying model abstained (~87% of cycles) — and logged that fallback _under the model's name_. The accuracy attributed to the model was therefore largely measuring a moving-average rule. A separate candlestick model's driver contained a similar trap where a statistical fallback could be recorded as genuine model output; our re-implementation explicitly abstains rather than emit fallback predictions. When re-tested cleanly, both time-series models scored ~50%, confirming their retirement — but now on valid evidence.

**Corrective principle:** a measurement that cannot distinguish the model from its fallback, or that scores a prediction against a horizon it did not forecast, is not evidence about the model. Every driver in this study emits an explicit ABSTAIN rather than a substitute, and scores only at matched horizons.

---

## 6. The deployment threshold is not 50%

Raw directional accuracy above 50% is insufficient for profitability because transaction costs do not shrink with the forecast horizon while the expected move does. For a symmetric directional trade, the break-even accuracy is

> _p_\* = ½ · (1 + _c_ / _m_)

where _c_ is the round-trip cost and _m_ is the median absolute move over the horizon. Using measured median moves from our outcome data:

| Horizon | Median \|move\| (crypto) | Break-even (crypto, 10 bp) | Break-even (metals warrant, 5× leverage, 0.2% spread) |
| ------- | -----------------------: | -------------------------: | ----------------------------------------------------: |
| 1h      |                    0.26% |                      68.9% |                 ≈ unwinnable unleveraged; 63.7% at 5× |
| 4h      |                    0.46% |                      60.8% |                                           57.2% at 5× |
| 24h     |                    1.54% |                      53.2% |                                           52.7% at 5× |
| 72h     |                    2.85% |                      51.8% |                                           51.1% at 5× |
| 168h    |                    3.97% |                      51.3% |                                           50.7% at 5× |

Two consequences follow. First, **sub-daily horizons are structurally hostile**: a 1h crypto strategy must be right ~69% of the time merely to break even, a bar no tested model approached. Second, **leverage on the zero-commission Avanza MINI warrants materially lowers the metals bar** — the 0.2% spread is diluted roughly in proportion to leverage — making 24h+ metal horizons the least demanding cells in the study. Even there, no model reached the threshold. The correct deployment gate is therefore _after-cost expectancy_ combined with calibration and drawdown robustness, never raw accuracy; the project's 60% CI-lower-bound rule is a conservative proxy for this and was not met anywhere.

---

## 7. Limitations

- **Input scope.** We tested price-, flow-, and positioning-derived inputs. We did _not_ test limit-order-book depth history (not publicly replayable for the window), news/sentiment event features at scale, or on-chain data — all of which the live system's better-performing signals (e.g. an on-chain BTC signal, an illiquidity-regime signal) draw upon. The null result is bounded to the input modalities tested.
- **Horizon scope.** Sub-minute and 15-minute horizons were deprioritised on cost grounds (Section 6) and are not covered.
- **Metals history.** Binance perp data for gold and silver begins only at their late-2025/early-2026 listing dates, yielding smaller samples and only ~7 months of regime coverage for metals versus a full year for crypto.
- **MoE model (resolved).** The 35B mixture-of-experts probe (Qwen3.6-35B-A3B, ~3B active) completed to n = 122 voted crypto predictions before early-stopping. It scored 47.5% / 47.5% / 50.8% at the 4h / 12h / 72h horizons (Wilson CI-low 38.9–42.1%), confirming the null: the largest model in the study lands exactly where the 102M-parameter candlestick model did. This closes the sixth model family with no reversal, as anticipated.
- **Efficient-market interpretation.** The result is consistent with weak-form market efficiency for these instruments and horizons. It does not establish that _no_ signal exists — only that none was recoverable by the tested methods from the tested inputs.

---

## 8. Conclusions

1. **Model capacity is not the bottleneck.** A 102-million-parameter candlestick model, a 3.8-billion-parameter reasoning LLM, gradient-boosted trees, and a 35-billion-parameter mixture-of-experts model all produce the same ~50% directional accuracy. Six model families spanning nearly three orders of magnitude in parameter count converge on the same coinflip. Scaling the model does not help.
2. **Price-derived input is not the bottleneck's solution either.** Neither engineered order-flow features nor perp-positioning and cross-asset features separated from a coinflip. The information tested is not predictive of direction at these horizons.
3. **Confidence filtering does not rescue a null classifier** on this data; conviction magnitude is uncorrelated with correctness.
4. **Most prior "signal" was measurement error.** Three distinct artifact classes — template contamination, horizon mislabelling, and fallback masquerading — each produced spuriously high accuracy that vanished under a certified, horizon-matched, fallback-free harness. This is the study's most transferable finding: _the harness must be validated before the model is judged._
5. **The economically correct bar is after-cost expectancy, not accuracy.** Break-even analysis shows sub-daily horizons require accuracy far beyond anything observed, while leveraged multi-day metal horizons are the most forgiving cells — yet none was met.

The practical recommendation for the live system is to **not** allocate compute to further price-directional model roulette, and to concentrate research on the input modalities excluded here (order-book microstructure, on-chain state, event/news features) where the system's existing best signals already reside. The exhaustively measured null delivered by this campaign is itself the deliverable: it closes a large, expensive question with high confidence, so it need not be reopened by the next capable model that appears.

---

## Appendix A. Reproduction

All drivers emit a common JSON-lines schema and share resume-keying, so results from any model coexist in one scoreable corpus.

```
# Score any collection of result files into the comparison matrix
python scripts/llm_matrix_report.py <files...> --horizon-match --min-votes 30

# Drivers (each --start --end --interval --step-hours --tickers --out)
scripts/llm_backtest.py     # LLMs, unified chat-completions path
scripts/xgb_backtest.py     # gradient trees, indicator features
scripts/micro_backtest.py   # order-flow (kline/full) and positioning (stage1) feature sets
scripts/kronos_backtest.py  # candlestick foundation model
scripts/chronos_backtest.py # time-series quantile foundation model
```

## Appendix B. Verified prediction counts

| Corpus                                       |         Rows |
| -------------------------------------------- | -----------: |
| LLM A/B (phi4 pair, 1d)                      |        1,924 |
| LLM matrix (1h / 4h / 1d)                    |       ~8,400 |
| XGBoost indicator matrix                     |       19,461 |
| Chronos + Kronos re-trial                    |       38,922 |
| Microstructure (flow / positioning)          |      ~56,000 |
| **Total verified out-of-sample predictions** | **~135,000** |

---

_Prepared by the finance-analyzer research process. Data, drivers, and per-cell result tables are version-controlled in the project repository._
