# Model-Matrix Campaign — Final Report

**Date:** 2026-07-20
**Campaign:** GRAND LLM MATRIX (2026-07-17 → 2026-07-19), pilot-first strategy per user decision 2026-07-18
**Verdict bar:** Wilson 95% CI lower bound ≥ 60% directional accuracy per (model, interval, ticker, horizon) cell — user policy "≥60% or no GPU"
**All tables below regenerated 2026-07-20 with `scripts/llm_matrix_report.py` from raw jsonl. Not copied from prose.**

---

## 1. Executive summary

Five model families — Phi-4-mini-reasoning (reasoning LLM), Phi-4-mini-instruct (instruct LLM), XGBoost (gradient-boosted trees, retrained per stamp), Chronos-bolt (time-series transformer), and Kronos-base (102M OHLCV-token foundation model) — were run through identical walk-forward grids over Feb–Jul 2026 (4h grid: Jul 2025–Jul 2026) on BTC, ETH, XAU, XAG at 1h/4h/1d intervals, ~58k scored rows total. **Every family pooled to 48.7–50.3% directional accuracy; 0 of 153 scoreable cells cleared the CI-low ≥ 60% bar; the single best cell anywhere was CI-low 52.0%** (xgboost XAG 1h, still below its own cost break-even). Every earlier "edge" (phi4 66.7%, instruct 61–72% teasers, the original Chronos/Kronos retirement numbers) traced to measurement defects, not model skill. The constant across all five failures is the input diet (indicator/price context), not the architecture — which is where the surviving hypothesis points (section 5). The planned 400–700 GPU-hour full 11-model matrix was cancelled on this evidence.

| family        | architecture                     | clean-grid votes | pooled acc | pooled CI-low | best cell (n≥30)                      | verdict  |
| ------------- | -------------------------------- | ---------------: | ---------: | ------------: | ------------------------------------- | -------- |
| phi4_mini     | reasoning LLM 3.8B               |            5,427 |      48.7% |         47.4% | BTC 1h@1h 49.9% (n=919)               | coinflip |
| phi4_instruct | instruct LLM 3.8B                |            2,205 |      50.1% |         48.0% | ETH 4h@72h 61.4% (n=88, CI-low 50.9)  | coinflip |
| xgboost       | boosted trees, per-stamp retrain |           19,325 |      50.1% |         49.4% | XAG 1h@1h 55.2% (n=961, CI-low 52.0)  | coinflip |
| chronos-bolt  | TS transformer                   |           14,567 |      50.2% |         49.4% | XAU 4h@72h 56.1% (n=355, CI-low 50.9) | coinflip |
| kronos        | OHLCV token model 102M           |           15,788 |      50.3% |         49.5% | XAU 1h@24h 53.3% (n=886, CI-low 50.0) | coinflip |

Stars (CI-low ≥ 60%): **zero, in any table, anywhere.**

---

## 2. Methodology

### Walk-forward grids

| grid | window                  | step | horizons         | cases (4 tickers) |
| ---- | ----------------------- | ---- | ---------------- | ----------------: |
| 1h   | 2026-02-01 → 2026-07-11 | 4h   | 1h / 3h / 24h    |             3,844 |
| 4h   | 2025-07-15 → 2026-07-11 | 12h  | 4h / 12h / 72h   |             2,160 |
| 1d   | 2026-02-01 → 2026-07-11 | 24h  | 24h / 72h / 168h |               483 |

Stamps align to the candle grid (horizons exact, every stamp gets fresh candle context). XAU/XAG use Binance FAPI klines (XAUUSDT/XAGUSDT) with shorter history — metals cells have smaller n by construction. Scoring: directional accuracy on BUY/SELL votes only; HOLD and ABSTAIN excluded from the denominator but reported as rates.

The **old 1d grid (llm_ab_final.jsonl, step 8h on 1d candles) is kept for reference but is flawed**: 3 identical prompts per day (temp-0.8 resamples, unanimous only 57% → effective n ≈ 1/3 nominal) and "24h" outcomes only span 24h for 00:00 stamps. All verdict grids replaced it with candle-aligned steps.

### Certification gate (before any mass GPU burn)

- **Template:** every GGUF's embedded chat template dumped (`scripts/dump_gguf_templates.py`) and byte-compared against the official HF `tokenizer_config` template; llama-server `/props` used as ground truth. All models served via `/v1/chat/completions` so the server applies the GGUF's own template — no hand-built markup (this exact defect class produced the phi4 artifact, see section 3).
- **Sampling:** per-model official parameters in `MODEL_SAMPLING` (e.g. phi4 0.8/0.95/40 per Microsoft; qwen3 0.6/0.95/20 + min_p=0; ministral temp 0.15). Certification pass wf_f869a46e-892, shipped ~commit `1af95994`.
- **V0 gate:** ~40-inference raw-output inspection per model (KEEP_RAW=1) before any full run.

### Outcome integrity

- **Outcome replication 2799/2799:** every outcome cell in the A/B file recomputed from independently fetched Binance klines — exact match, 0 duplicates, 0 gaps.
- **No-leakage guards:** LLM/TS context windows end at the as-of candle (`df.index <= at`, `candles_payload` guard in kronos_backtest.py); XGBoost trains on an expanding window of labels strictly resolvable before `at`; outcome fields are scoring labels written to output rows only, never fed to any model; Kronos/Chronos weights frozen (nothing fitted on outcomes).
- **Fallback attribution guards:** kronos driver rejects any response with `method != "kronos"` → ABSTAIN (see section 3); chronos driver has no gates/EMA-fallback/composite — raw model direction only, explicit `--neutral-band 0.001`.

### Cost model and break-even table

`BE% = 50 · (1 + c / (L·m))` — c = round-trip cost, L = leverage (1 for spot), m = median |move| over the horizon. Costs: crypto exchange ≈ 10 bp round-trip; **Avanza MINI warrants = 0 courtage for orders ≥ 1000 SEK (hence MIN_TRADE_SEK=1000), ~0.2% spread = true round-trip cost, diluted by leverage** (user-corrected 2026-07-18). Median |move| computed from the actual grid outcomes (xgb_matrix files, full coverage):

| ticker  | horizon | med \|move\| % | BE spot/10bp | BE warrant 5x/20bp |
| ------- | ------: | -------------: | -----------: | -----------------: |
| BTC-USD |      1h |          0.251 |        69.9% |              58.0% |
| BTC-USD |     24h |          1.381 |        53.6% |              51.4% |
| BTC-USD |     72h |          2.359 |        52.1% |              50.8% |
| BTC-USD |    168h |          3.703 |        51.4% |              50.5% |
| ETH-USD |      1h |          0.293 |        67.1% |              56.8% |
| ETH-USD |     24h |          1.745 |        52.9% |              51.1% |
| ETH-USD |     72h |          3.410 |        51.5% |              50.6% |
| XAG-USD |      1h |          0.213 |        73.5% |              59.4% |
| XAG-USD |     24h |          1.485 |        53.4% |              51.3% |
| XAG-USD |     72h |          3.494 |        51.4% |              50.6% |
| XAU-USD |      1h |          0.108 |        96.3% |              68.5% |
| XAU-USD |     24h |          0.714 |        57.0% |              52.8% |
| XAU-USD |     72h |          1.450 |        53.4% |              51.4% |
| XAU-USD |    168h |          2.056 |        52.4% |              51.0% |

Reading: leverage rescues long-horizon metals (5x XAU 24h needs 52.8%, not 57%), but 1h is near-unwinnable everywhere and 15m was deprioritized as unwinnable without leverage. Leverage caveats: knockout barriers (asymmetric wipe), overnight financing on multi-day holds, 5x variance — the deployment gate is expectancy + drawdown, never hit-rate alone. **No measured cell beats its own break-even:** best cell overall (XAG 1h@1h, 55.2%, CI-low 52.0) sits under both the 73.5% spot bar and the 59.4% warrant bar for that cell.

---

## 3. Per-model verdicts

### phi4_mini (Phi-4-mini-reasoning) — coinflip; the 66.7% was an artifact

- **Invalid measurement:** the original 66.7% (n=108) that promoted phi4_mini to live voter was produced by three stacked input defects (fixed commit `be86d6bf`): (1) qwen3's prompt builder + parser used for all models — the parser fabricates a bare-regex BUY/SELL from truncated `<think>` prose; (2) `n_predict=2048` truncated the CoT; (3) server `-c 4096` context-starved the model. The 86–99% "abstention" was starvation, and the scored "votes" were a biased ~11% subsample plus parser fabrications.
- **Clean result:** honest full-vote accuracy ≈ coinflip. Flawed-1d-grid file, 00:00-stamps-only (true horizons): pooled 24h 50.2% (n=313, CI-low 44.7). Clean 1h grid: pooled 48.7% (n=5,427 votes, CI-low 47.4); best cell BTC 1h@1h 49.9% (n=919). Votes hard (SELL 56% / BUY 40%), abstain 0.2%.
- Kept in the paper loop as free forward-shadow validation (user decision 2026-07-18); moot while loops are off.

### phi4_instruct (Phi-4-mini-instruct) — coinflip; selectivity hypothesis dead

- **The pilot model** (chosen 2026-07-18: certified, 61% teaser on the flawed grid, cheapest at 3.3 s/inf). Hypothesis under test: HOLD-heavy contextual abstention (~85–91% HOLD) selects a high-quality voted subset — the flawed grid showed 61.4% @24h (n=114) and this report's regenerated flawed-grid table shows BTC 1d@72h 67.5% (n=83, CI-low 56.8).
- **Clean grids (1h full, 4h full-year, 1d step-24, 4 tickers): pooled 50.1% (n=2,205 votes, CI-low 48.0).** No cell near the bar in any interval/ticker/horizon; best clean cell ETH 4h@72h 61.4% (n=88, CI-low 50.9). The teaser cells were flawed-grid artifacts. **Selectivity-correlates-with-quality is dead.**

### XGBoost — coinflip; probability magnitude carries no selectivity either

- Expanding-window walk-forward retrain at every stamp, same feature set as the LLM context, 19,325 votes: **pooled 50.1% (CI-low 49.4)**; best cell XAG 1h@1h 55.2% (n=961, CI-low 52.0) — the best cell of the whole campaign, still under its break-even.
- **Confidence-gating tested on 19k rows: no edge.** Accuracy flat at 50–51% for every probability cutoff 0.5 → 0.7; expectancy after 10 bp cost negative everywhere (−11 to −18 bp/trade). Tree probability magnitude is not a selectivity signal on these features (refutes the external advisor's "trees + conf-filter" thesis).

### Chronos-bolt — retirement confirmed, now with valid evidence

- **Invalid original measurement (audit wf_e52a19f9):** since 2026-05-11 an EMA-slope dead-zone fallback was logged AS `forecast` in ~87% of cycles — the production accuracy table was scoring a moving-average rule, not the model. Also: composite graded at 3h/1d while 1h-dominated (no 1h horizon in outcome_tracker); backfill truncated predictions at a 500/day cap (data destroyed); 1h labels used ±2h cross-source snapshot tolerance with no flat band; the live model was chronos-2, not bolt; silent 15m-candle fallback mislabeled as 1h. Attribution fix shipped to production (merge `7b27485f`): model HOLD stays HOLD for `forecast`.
- **Clean re-trial (`scripts/chronos_backtest.py`, bolt, explicit 0.1% neutral band, 19,461 rows): 48–53% everywhere.** Pooled 50.2% (n=14,567, CI-low 49.4); best cells XAU 4h@4h 56.4% (CI-low 50.3) and XAU 4h@72h 56.1% (CI-low 50.9); worst XAU 1d@168h 23.9% (n=88). The old conclusion was accidentally right; now it is evidenced.

### Kronos-base — coinflip; the statistical-fallback trap was guarded, not triggered

- The one different-input-modality candidate: raw OHLCV token sequences (480-candle context), no indicators.
- **The trap:** `kronos_infer.py` silently degrades to a linear-regression `statistical_fallback` path when the torch model can't load, and reports it in-band via a `method` field. A naive driver would have scored a linear regression under Kronos's name — exactly the Chronos EMA-attribution bug class. The driver hard-rejects `method != "kronos"` → ABSTAIN (`scripts/kronos_backtest.py:260`). In the final run the guard never fired: **0 error rows, 0 fallback abstains — all 19,461 rows are genuine Kronos.** (Kronos's magnitude-heuristic "confidence" is deliberately written as `conf: null` — it is not a probability.)
- **Result: pooled 50.3% (n=15,788, CI-low 49.5)**; best cell XAU 1h@24h 53.3% (n=886, CI-low 50.0). Different input modality, same coinflip.

---

## 4. Measured costs and the extrapolation that was avoided

Measured on the RTX 3080 (herc2), from `secs` fields in the result rows:

| class                              | s/inference (mean) | full 3-grid coverage (6,487 calls) |
| ---------------------------------- | -----------------: | ---------------------------------: |
| instruct-class LLM (phi4_instruct) |                3.3 |                           ~8 h GPU |
| reasoning-class LLM (phi4_mini)    | 22.0 (median 19.9) |                          ~51 h GPU |
| kronos (subprocess reload/call)    |                4.3 |                           ~8 h GPU |
| chronos-bolt (in-process)          |               <0.1 |                            minutes |
| xgboost (retrain+infer, CPU)       |              ~0.03 |                            minutes |

The remaining roster (qwen3, fin_r1, ministral3, ministral8_lora, finance-llama-8b + 4 staged successors: qwen3.5-9B, deepseek-r1-0528-qwen3-8B, fin-o1-8B, gemma4-12B, plus gemma3-12b) is mostly reasoning-class. **Full 11-model matrix ≈ 400–700 GPU-hours ≈ 3–4 weeks rolling.** After the pilot (one instruct model completed fully, ~9 h) plus the TS re-trials (~12 h) and the free XGBoost baseline all pooled to coinflip on the same input diet, that spend was rejected: the prior that any remaining same-diet LLM clears CI-low ≥ 60% is too low to justify it. Actual campaign GPU spend: roughly 40–50 h total (A/B + pilot chain + TS re-trials) versus the 400–700 h avoided.

---

## 5. What survives

1. **The input-side hypothesis.** Five architectures failed identically on the indicator/price diet; the constant across failures is the input, not the model. Falsifiable next step: same harness, different inputs — order-flow/microstructure, cross-asset, on-chain, news-derived features. (Caveat already measured: raw Finnhub headlines made phi4 _worse_ on equities, 57.4% → 54.1% — "different inputs" means engineered features, not prompt-stuffed text.)
2. **Live signals on different input classes are the existing counter-evidence:** Amihud Illiquidity Regime 68.0% 1d (n=225) and On-Chain BTC (MVRV-Z/SOPR/NUPL/netflow) 60.0% 1d are the top live voters, and neither eats the indicator diet — one reads liquidity microstructure, the other chain data. Small n; but directionally consistent with (1).
3. **qwen3.5 generation-check probe, pending.** Cheap probe (not a matrix run) to test whether a Feb-2026-generation model behaves differently on the same diet before closing the LLM question entirely. Model staged on herc2 (unsloth UD-Q4_K_XL 5.97GB) alongside deepseek-r1-0528, fin-o1, gemma4 QAT; **blocked on a llama.cpp build update on herc2 (new architectures — fetch latest ggml-org win-cuda zip to a NEW dir).**
4. **phi4 forward shadow:** stays a paper-loop voter on BTC/ETH/XAU/XAG (free forward validation of the backtest verdict) whenever herc2 serves; dormant while loops are off.
5. **Harness + report tooling** (`llm_backtest.py` unified chat-path harness, certification workflow, `kronos_backtest.py`/`chronos_backtest.py`/`xgb_backtest.py` drivers, `llm_matrix_report.py`) are input-agnostic and reusable for the input-side experiments.

---

## 6. Reproduction

Data files: herc2 `Q:\finance-analyzer\data\` (llm*ab*_.jsonl, llm*matrix_1d.jsonl, ts_matrix*_.jsonl) with Deck scratchpad copies; XGBoost files in repo `data/xgb_matrix_{1h,4h,1d}.jsonl`.

```bash
# Headline tables (this report) — plain for LLM files, --horizon-match for per-horizon model names
.venv/bin/python scripts/llm_matrix_report.py \
    llm_ab_final.jsonl llm_ab_crypto_1h.jsonl llm_ab_crypto_4h.jsonl llm_matrix_1d.jsonl
.venv/bin/python scripts/llm_matrix_report.py --horizon-match \
    ts_matrix_1h.jsonl ts_matrix_4h.jsonl ts_matrix_1d.jsonl
.venv/bin/python scripts/llm_matrix_report.py --horizon-match \
    data/xgb_matrix_1h.jsonl data/xgb_matrix_4h.jsonl data/xgb_matrix_1d.jsonl

# LLM legs (herc2 GPU, resumable; env-driven wrapper auto-pulls repo)
MODELS=phi4_instruct INTERVAL=1h STEP_HOURS=4 START=2026-02-01 END=2026-07-11 \
    TICKERS=BTC-USD,ETH-USD,XAU-USD,XAG-USD OUT='data\llm_ab_crypto_1h.jsonl' \
    scripts/deck/run-llm-backtest-on-herc.sh          # 4h grid: INTERVAL=4h STEP_HOURS=12 START=2025-07-15
                                                      # 1d grid: INTERVAL=1d STEP_HOURS=24 OUT=llm_matrix_1d.jsonl

# TS legs (herc2, wrapper scripts/win/ts-backtest-run.ps1 ran both drivers x 3 intervals)
python scripts/kronos_backtest.py  --start 2026-02-01 --end 2026-07-11 --interval 1h \
    --tickers BTC-USD,ETH-USD,XAU-USD,XAG-USD --out data/ts_matrix_1h.jsonl
python scripts/chronos_backtest.py --start 2026-02-01 --end 2026-07-11 --interval 1h \
    --neutral-band 0.001 --tickers BTC-USD,ETH-USD,XAU-USD,XAG-USD --out data/ts_matrix_1h.jsonl

# XGBoost baseline (CPU — run on herc2 per compute-placement policy, NOT the Deck)
python scripts/xgb_backtest.py --start 2026-02-01 --end 2026-07-11 --interval 1h \
    --tickers BTC-USD,ETH-USD,XAU-USD,XAG-USD --out data/xgb_matrix_1h.jsonl
```

Key commits/workflows: harness unification `cd480a1f`; phi4 input fix `be86d6bf`; certification wf_f869a46e-892; chronos audit wf_e52a19f9; forecast-attribution production fix `7b27485f`; drivers commit `f03a6ab5`.

---

## Appendix — full per-cell tables (generated 2026-07-20)

`*` would mark CI-low ≥ 60%; none earned it.

### A. LLM files (plain)

```
model                      iv   ticker      hor  votes   acc%  CIlow  hold%  abst%
----------------------------------------------------------------------------------
phi4_instruct              1d   BTC-USD     24h     83   60.2   49.5   87.1    0.0
phi4_instruct              1d   BTC-USD     72h     83   67.5   56.8   87.1    0.0
phi4_instruct              1d   BTC-USD    168h     83   59.0   48.3   87.1    0.0
phi4_instruct              1d   ETH-USD     24h     67   55.2   43.4   89.6    0.0
phi4_instruct              1d   ETH-USD     72h     67   53.7   41.9   89.6    0.0
phi4_instruct              1d   ETH-USD    168h     67   50.7   39.1   89.6    0.0
phi4_instruct              1d   XAG-USD     24h      7      -      -   89.6    0.0
phi4_instruct              1d   XAG-USD     72h      7      -      -   89.6    0.0
phi4_instruct              1d   XAG-USD    168h      7      -      -   89.6    0.0
phi4_instruct              1d   XAU-USD     24h     15   40.0   19.8   84.0    0.0
phi4_instruct              1d   XAU-USD     72h     15   46.7   24.8   84.0    0.0
phi4_instruct              1d   XAU-USD    168h     15   26.7   10.9   84.0    0.0
phi4_instruct              1h   BTC-USD      1h    139   48.2   40.1   85.5    0.0
phi4_instruct              1h   BTC-USD      3h    139   48.9   40.8   85.5    0.0
phi4_instruct              1h   BTC-USD     24h    139   55.4   47.1   85.5    0.0
phi4_instruct              1h   ETH-USD      1h     92   54.3   44.2   90.4    0.0
phi4_instruct              1h   ETH-USD      3h     92   55.4   45.3   90.4    0.0
phi4_instruct              1h   ETH-USD     24h     92   43.5   33.8   90.4    0.0
phi4_instruct              1h   XAG-USD      1h     83   43.4   33.2   91.4    0.0
phi4_instruct              1h   XAG-USD      3h     83   54.2   43.5   91.4    0.0
phi4_instruct              1h   XAG-USD     24h     83   54.2   43.5   91.4    0.0
phi4_instruct              1h   XAU-USD      1h     83   41.0   31.0   91.4    0.0
phi4_instruct              1h   XAU-USD      3h     83   44.6   34.4   91.4    0.0
phi4_instruct              1h   XAU-USD     24h     83   47.0   36.6   91.4    0.0
phi4_instruct              4h   BTC-USD      4h    106   50.9   41.6   85.3    0.0
phi4_instruct              4h   BTC-USD     12h    106   50.0   40.6   85.3    0.0
phi4_instruct              4h   BTC-USD     72h    106   39.6   30.8   85.3    0.0
phi4_instruct              4h   ETH-USD      4h     88   59.1   48.6   87.8    0.0
phi4_instruct              4h   ETH-USD     12h     88   46.6   36.5   87.8    0.0
phi4_instruct              4h   ETH-USD     72h     88   61.4   50.9   87.8    0.0
phi4_instruct              4h   XAG-USD      4h     36   58.3   42.2   89.1    0.0
phi4_instruct              4h   XAG-USD     12h     36   52.8   37.0   89.1    0.0
phi4_instruct              4h   XAG-USD     72h     36   61.1   44.9   89.1    0.0
phi4_instruct              4h   XAU-USD      4h     50   40.0   27.6   87.0    0.0
phi4_instruct              4h   XAU-USD     12h     50   46.0   33.0   87.0    0.0
phi4_instruct              4h   XAU-USD     72h     50   52.0   38.5   87.0    0.0
phi4_mini                  1d   BTC-USD     24h    465   51.8   47.3    3.1    0.2
phi4_mini                  1d   BTC-USD     72h    465   53.5   49.0    3.1    0.2
phi4_mini                  1d   BTC-USD    168h    465   52.9   48.4    3.1    0.2
phi4_mini                  1d   ETH-USD     24h    463   50.3   45.8    3.5    0.2
phi4_mini                  1d   ETH-USD     72h    463   52.1   47.5    3.5    0.2
phi4_mini                  1d   ETH-USD    168h    463   46.4   41.9    3.5    0.2
phi4_mini                  1h   BTC-USD      1h    919   49.9   46.7    3.9    0.5
phi4_mini                  1h   BTC-USD      3h    919   49.7   46.5    3.9    0.5
phi4_mini                  1h   BTC-USD     24h    919   49.7   46.5    3.9    0.5
phi4_mini                  1h   ETH-USD      1h    890   48.8   45.5    7.4    0.0
phi4_mini                  1h   ETH-USD      3h    890   46.2   42.9    7.4    0.0
phi4_mini                  1h   ETH-USD     24h    890   47.9   44.6    7.4    0.0
```

(phi4_mini 1d rows above are the flawed 8h-step grid — effective n ≈ 1/3 nominal; see section 2.)

### B. Time-series models (--horizon-match)

```
model                      iv   ticker      hor  votes   acc%  CIlow  hold%  abst%
----------------------------------------------------------------------------------
chronos-12h                4h   BTC-USD     12h    586   50.7   46.6   18.9    0.0
chronos-12h                4h   ETH-USD     12h    618   52.4   48.5   14.5    0.0
chronos-12h                4h   XAG-USD     12h    281   51.2   45.4   14.8    0.0
chronos-12h                4h   XAU-USD     12h    281   48.0   42.3   26.8    0.0
chronos-168h               1d   BTC-USD    168h    158   41.8   34.4    1.9    0.0
chronos-168h               1d   ETH-USD    168h    156   40.4   33.0    3.1    0.0
chronos-168h               1d   XAG-USD    168h     62   59.7   47.3    7.5    0.0
chronos-168h               1d   XAU-USD    168h     88   23.9   16.2    6.4    0.0
chronos-1h                 1h   BTC-USD      1h    546   53.3   49.1   43.2    0.0
chronos-1h                 1h   ETH-USD      1h    625   50.4   46.5   35.0    0.0
chronos-1h                 1h   XAG-USD      1h    659   50.4   46.6   31.4    0.0
chronos-1h                 1h   XAU-USD      1h    332   51.8   46.4   65.5    0.0
chronos-24h                1d   BTC-USD     24h    154   48.1   40.3    4.3    0.0
chronos-24h                1d   ETH-USD     24h    150   52.0   44.1    6.8    0.0
chronos-24h                1d   XAG-USD     24h     64   54.7   42.6    4.5    0.0
chronos-24h                1d   XAU-USD     24h     71   49.3   38.0   24.5    0.0
chronos-24h                1h   BTC-USD     24h    845   49.7   46.3   12.1    0.0
chronos-24h                1h   ETH-USD     24h    880   53.4   50.1    8.4    0.0
chronos-24h                1h   XAG-USD     24h    839   50.7   47.3   12.7    0.0
chronos-24h                1h   XAU-USD     24h    679   48.3   44.6   29.3    0.0
chronos-3h                 1h   BTC-USD      3h    589   50.4   46.4   38.7    0.0
chronos-3h                 1h   ETH-USD      3h    666   49.2   45.5   30.7    0.0
chronos-3h                 1h   XAG-USD      3h    670   53.3   49.5   30.3    0.0
chronos-3h                 1h   XAU-USD      3h    376   51.9   46.8   60.9    0.0
chronos-4h                 4h   BTC-USD      4h    564   48.2   44.1   22.0    0.0
chronos-4h                 4h   ETH-USD      4h    627   51.5   47.6   13.3    0.0
chronos-4h                 4h   XAG-USD      4h    275   51.3   45.4   16.7    0.0
chronos-4h                 4h   XAU-USD      4h    259   56.4   50.3   32.6    0.0
chronos-72h                1d   BTC-USD     72h    149   51.0   43.1    7.5    0.0
chronos-72h                1d   ETH-USD     72h    154   42.9   35.3    4.3    0.0
chronos-72h                1d   XAG-USD     72h     63   42.9   31.4    6.0    0.0
chronos-72h                1d   XAU-USD     72h     73   28.8   19.7   22.3    0.0
chronos-72h                4h   BTC-USD     72h    667   49.9   46.1    7.7    0.0
chronos-72h                4h   ETH-USD     72h    692   47.5   43.8    4.3    0.0
chronos-72h                4h   XAG-USD     72h    314   45.9   40.4    4.8    0.0
chronos-72h                4h   XAU-USD     72h    355   56.1   50.9    7.6    0.0
kronos-12h                 4h   BTC-USD     12h    646   51.2   47.4   10.7    0.0
kronos-12h                 4h   ETH-USD     12h    678   49.6   45.8    6.2    0.0
kronos-12h                 4h   XAG-USD     12h    298   48.7   43.0    9.7    0.0
kronos-12h                 4h   XAU-USD     12h    316   51.6   46.1   17.7    0.0
kronos-168h                1d   BTC-USD    168h    161   49.7   42.1    0.0    0.0
kronos-168h                1d   ETH-USD    168h    161   46.0   38.4    0.0    0.0
kronos-168h                1d   XAG-USD    168h     66   30.3   20.6    1.5    0.0
kronos-168h                1d   XAU-USD    168h     89   53.9   43.6    5.3    0.0
kronos-1h                  1h   BTC-USD      1h    536   49.1   44.9   44.2    0.0
kronos-1h                  1h   ETH-USD      1h    619   46.8   43.0   35.6    0.0
kronos-1h                  1h   XAG-USD      1h    643   50.4   46.5   33.1    0.0
kronos-1h                  1h   XAU-USD      1h    350   50.3   45.1   63.6    0.0
kronos-24h                 1d   BTC-USD     24h    152   52.6   44.7    5.6    0.0
kronos-24h                 1d   ETH-USD     24h    159   50.9   43.2    1.2    0.0
kronos-24h                 1d   XAG-USD     24h     60   48.3   36.2   10.4    0.0
kronos-24h                 1d   XAU-USD     24h     79   55.7   44.7   16.0    0.0
kronos-24h                 1h   BTC-USD     24h    933   48.8   45.6    2.9    0.0
kronos-24h                 1h   ETH-USD     24h    947   50.7   47.5    1.5    0.0
kronos-24h                 1h   XAG-USD     24h    911   50.9   47.7    5.2    0.0
kronos-24h                 1h   XAU-USD     24h    886   53.3   50.0    7.8    0.0
kronos-3h                  1h   BTC-USD      3h    756   50.3   46.7   21.3    0.0
kronos-3h                  1h   ETH-USD      3h    765   51.9   48.4   20.4    0.0
kronos-3h                  1h   XAG-USD      3h    777   49.5   46.0   19.1    0.0
kronos-3h                  1h   XAU-USD      3h    534   51.5   47.3   44.4    0.0
kronos-4h                  4h   BTC-USD      4h    549   50.8   46.6   24.1    0.0
kronos-4h                  4h   ETH-USD      4h    616   51.1   47.2   14.8    0.0
kronos-4h                  4h   XAG-USD      4h    271   49.4   43.5   17.9    0.0
kronos-4h                  4h   XAU-USD      4h    255   45.9   39.9   33.6    0.0
kronos-72h                 1d   BTC-USD     72h    158   53.2   45.4    1.9    0.0
kronos-72h                 1d   ETH-USD     72h    161   48.4   40.9    0.0    0.0
kronos-72h                 1d   XAG-USD     72h     66   40.9   29.9    1.5    0.0
kronos-72h                 1d   XAU-USD     72h     82   52.4   41.8   12.8    0.0
kronos-72h                 4h   BTC-USD     72h    708   53.4   49.7    2.1    0.0
kronos-72h                 4h   ETH-USD     72h    717   49.8   46.1    0.8    0.0
kronos-72h                 4h   XAG-USD     72h    324   49.4   44.0    1.8    0.0
kronos-72h                 4h   XAU-USD     72h    359   47.4   42.2    6.5    0.0
```

### C. XGBoost (--horizon-match)

```
model                      iv   ticker      hor  votes   acc%  CIlow  hold%  abst%
----------------------------------------------------------------------------------
xgboost-12h                4h   BTC-USD     12h    723   49.0   45.3    0.0    0.0
xgboost-12h                4h   ETH-USD     12h    723   49.9   46.3    0.0    0.0
xgboost-12h                4h   XAG-USD     12h    330   50.9   45.5    0.0    0.0
xgboost-12h                4h   XAU-USD     12h    384   50.3   45.3    0.0    0.0
xgboost-168h               1d   BTC-USD    168h    135   44.4   36.3    0.0   16.1
xgboost-168h               1d   ETH-USD    168h    135   42.2   34.2    0.0   16.1
xgboost-168h               1d   XAG-USD    168h     67   61.2   49.2    0.0    0.0
xgboost-168h               1d   XAU-USD    168h     94   52.1   42.1    0.0    0.0
xgboost-1h                 1h   BTC-USD      1h    961   48.3   45.1    0.0    0.0
xgboost-1h                 1h   ETH-USD      1h    961   50.3   47.1    0.0    0.0
xgboost-1h                 1h   XAG-USD      1h    961   55.2   52.0    0.0    0.0
xgboost-1h                 1h   XAU-USD      1h    961   53.2   50.0    0.0    0.0
xgboost-24h                1d   BTC-USD     24h    141   44.7   36.7    0.0   12.4
xgboost-24h                1d   ETH-USD     24h    141   52.5   44.3    0.0   12.4
xgboost-24h                1d   XAG-USD     24h     67   35.8   25.4    0.0    0.0
xgboost-24h                1d   XAU-USD     24h     94   40.4   31.1    0.0    0.0
xgboost-24h                1h   BTC-USD     24h    961   51.4   48.2    0.0    0.0
xgboost-24h                1h   ETH-USD     24h    961   51.4   48.2    0.0    0.0
xgboost-24h                1h   XAG-USD     24h    961   48.4   45.2    0.0    0.0
xgboost-24h                1h   XAU-USD     24h    961   52.2   49.1    0.0    0.0
xgboost-3h                 1h   BTC-USD      3h    961   51.3   48.1    0.0    0.0
xgboost-3h                 1h   ETH-USD      3h    961   54.3   51.2    0.0    0.0
xgboost-3h                 1h   XAG-USD      3h    961   48.4   45.2    0.0    0.0
xgboost-3h                 1h   XAU-USD      3h    961   47.3   44.2    0.0    0.0
xgboost-4h                 4h   BTC-USD      4h    723   49.8   46.2    0.0    0.0
xgboost-4h                 4h   ETH-USD      4h    723   48.4   44.8    0.0    0.0
xgboost-4h                 4h   XAG-USD      4h    330   45.8   40.5    0.0    0.0
xgboost-4h                 4h   XAU-USD      4h    384   50.3   45.3    0.0    0.0
xgboost-72h                1d   BTC-USD     72h    139   51.1   42.9    0.0   13.7
xgboost-72h                1d   ETH-USD     72h    139   49.6   41.5    0.0   13.7
xgboost-72h                1d   XAG-USD     72h     67   52.2   40.5    0.0    0.0
xgboost-72h                1d   XAU-USD     72h     94   51.1   41.1    0.0    0.0
xgboost-72h                4h   BTC-USD     72h    723   49.8   46.2    0.0    0.0
xgboost-72h                4h   ETH-USD     72h    723   47.6   44.0    0.0    0.0
xgboost-72h                4h   XAG-USD     72h    330   45.8   40.5    0.0    0.0
xgboost-72h                4h   XAU-USD     72h    384   48.7   43.7    0.0    0.0
```
