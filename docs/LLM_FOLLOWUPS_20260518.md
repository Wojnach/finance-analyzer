# LLM Shadow Pipeline — Follow-up Investigations 2026-05-18

Trail of three diagnostic threads opened after the shadow-gate + LoRA
fix merge (`07702358`). Not all of them require code changes; the doc
captures the findings so the next session does not re-investigate.

## 1. BUY-asymmetry in directional accuracy — CLOSED, not a bug

### Observation
Filtering `data/llm_probability_log.jsonl` to directional rows
(`conf>0 AND chosen in {BUY,SELL}`) showed striking asymmetry:

| Signal     | BUY hit-rate | SELL hit-rate | sample |
|------------|--------------|---------------|--------|
| ministral  | 0/48  (0%)   | 13/17 (76%)   |   65   |
| qwen3      | 0/16  (0%)   | 11/11 (100%)  |   27   |

The 0% BUY rate looked like a sign flip in the outcome backfill.

### Verification

`portfolio.llm_outcome_backfill` labels outcomes by `pct_change` sign
with a HOLD band for small moves. Sanity-check the full
`llm_probability_outcomes.jsonl` confusion against the move direction:

```
BUY_with_up      19966
BUY_with_flat     8273
BUY_with_down        0  ← no negative-move row ever labelled BUY
SELL_with_up         0  ← no positive-move row ever labelled SELL
SELL_with_down   19176
SELL_with_flat    6073
HOLD_with_*       only "flat" populated
```

The sign is consistent. Spot-checking 10 ministral BUY rows: all on
2026-05-13 / 05-14, all on BTC-USD or XAG-USD, all with
`pct_change` between -0.6% and -9.8%. Ministral was calling BUY into
the actual crypto + silver drop. **0/48 is real bad timing, not a
sign flip.**

### What to use instead

`data/accuracy_cache.json` uses all-time signal_log data
(`total_buy + total_sell` denominator). Ministral 58% on 6284 samples
is the stable measure. The 30-day shadow log window is too short to
be representative — keep it for promotion gating (sample-size guard
already enforces this) but do not draw conclusions about individual
LLMs from short windows.

### Action

None. Closing the thread.

---

## 2. `custom-trading-lora.gguf` — KEEP, defer wiring

### What it is

`Q:\models\custom-trading-lora.gguf` — 15MB GGUF LoRA adapter, our
own homebrew fine-tune produced by `training/lora/pipeline.py` on
2026-02-13. Trained against `Q:\models\ministral-8b-hf` base, rank-8
PEFT, alpha 16, `general.name: Final`.

Two existing repo touchpoints:

* `training/lora/pipeline.py` — full QLoRA training pipeline:
  generate_data → download_model → train → convert_gguf → verify →
  deploy_shadow. The pipeline has produced one output (this GGUF) and
  has not been re-run since Feb.
* `scripts/lora_backtest.py` — offline harness that downloads 14d of
  1h Binance candles, runs both `cryptotrader-lm-lora` AND
  `custom-trading-lora` against the same labelled outcomes, prints
  per-model accuracy + agreement. Single-threaded llama-cpp-python
  through `Q:\models\.venv-llm\Scripts\python.exe`.

### Live probe (2026-05-18)

Injected a temporary `_MODEL_CONFIGS["ministral8_custom_trading_lora"]`
entry pointing at the GGUF, probed via the production
`ministral_trader._build_prompt`:

```
wall: 8.9s (first call, included swap)
text len: 1359
parsed: ('BUY', 'Bullish EMA crossover with MACD positive, RSI recovering from
        oversold, volume increasing, multi-timeframe agreement, neutral to
        positive sentiment.', 0.75)
```

Real reasoning, parses cleanly. **The LoRA works.**

### Quick 5-prompt head-to-head vs CryptoTrader-LM (2026-05-18)

Ran the same 5 production-shaped prompts through both LoRAs via
`portfolio.llama_server`:

```
Ctx  CryptoTrader-LM   Custom LoRA      agree
 0   BUY/0.80          BUY/0.75         YES
 1   SELL/0.80         HOLD/0.45         no
 2   SELL/0.80         HOLD/0.45         no
 3   HOLD/0.60         HOLD/0.45        YES
 4   BUY/0.85          BUY/0.85         YES

agreement: 3/5 (60%)
```

Interpretation:

* **Custom LoRA is more conservative on the SELL side.** Where
  CryptoTrader-LM calls SELL on the BTC-profit-taking + ETH-oversold
  contexts (1 and 2), Custom LoRA returns HOLD/conf=0.45. With the
  new directional filter (`conf>0 AND chosen in {BUY,SELL}`),
  Custom's HOLDs don't count toward accuracy either way — so the
  conservatism isn't a calibration risk.
* **Both agree on BUY-bullish setups** (ctx 0, 4) with similar high
  confidence (0.75-0.85).
* **No surprises on HOLD-ambiguous** (ctx 3).

Net: Custom LoRA is functional and uncorrelated enough to be useful
as a separate voter. Not redundant with CryptoTrader-LM. 8.9s first
call (cold swap), ~4s subsequent.

### Why not wire it yet

`cryptotrader_lm` just came online on 2026-05-18 with the v2 GGUF
regen. It needs 24-72h of accumulating directional predictions
against outcome backfill before we can judge its actual promotion
fitness — and adding a second crypto-LoRA at the same time would:

* double the cycle-time impact of the ministral8_lora swap slot
* confuse the question "is the LoRA approach working?" by mixing two
  different fine-tunes
* compete for the same GPU swap window (both pin Ministral-8B base +
  a LoRA — the rotation would thrash if both ran every 3 cycles)

### Action

* Document in this file (done).
* Add a registry entry for `custom_trading_lora` once cryptotrader_lm
  has either crossed the promotion bar or been retired. If
  cryptotrader_lm is promoted at ~60-70% accuracy, the custom LoRA
  becomes a redundant voter; if cryptotrader_lm is retired, the slot
  is free.
* Optional: run `scripts/lora_backtest.py --days 14` as an offline
  comparison before wiring. The harness is already in the repo.

### Status

`KEEP — wire pending cryptotrader_lm 72h verdict`

---

## 3. Qwen3 HOLD bias — confirmed, A/B needed before fix

### Observation

Live production data shows qwen3 emitting `HOLD` on >95% of cycles.
Directional row count over the last 30 days: 27 (vs ministral 65).
The model is functionally a non-voter despite registering on the
voter slate at 1.2% activation rate.

### Root cause

`portfolio/qwen3_trader.py:_build_prompt` system message contains TWO
reinforcements pushing toward HOLD:

```
Use HOLD when evidence is mixed, weak, or conflicting. A confident HOLD
is better than a low-confidence BUY/SELL.
Confidence guide: 80+ = strong conviction, 60-79 = moderate, 40-59 =
weak/mixed, <40 = default to HOLD.
```

Sentences (a) "A confident HOLD is better than..." and (b)
"<40 = default to HOLD" together tell the model that HOLD is the
preferred safe answer. The model takes that direction literally.

### Why we don't flip it speculatively

`accuracy_cache.json` for qwen3 on 1d: 60.0% on 3809 samples
(`correct_buy: 426 / total_buy: 1288` = 33.1%, `correct_sell: 1859 /
total_sell: 2521` = 73.7%). The headline accuracy is dominated by
SELL precision. Changing the system prompt could:

* shift the BUY/SELL/HOLD mix without changing precision (good)
* destabilise the SELL precision the model currently delivers (bad —
  qwen3 is one of two LLMs above 47% on the active voter slate)

A speculative flip is the kind of change that the `feedback_weight_calibration_warnings`
memory explicitly warns against. Need offline A/B first.

### A/B plan (proposed, NOT shipped)

1. Save current prompt as `_build_prompt_conservative_v1`.
2. Add `_build_prompt_neutral_v2` that removes ONLY sentence (a) —
   "A confident HOLD is better than a low-confidence BUY/SELL." —
   keeping the confidence guide so low-conf falls to HOLD naturally
   but the explicit preference is gone.
3. Extend `scripts/lora_backtest.py` (or write a sibling) to accept
   `--prompt-variant {v1, v2}` and re-score the same labelled candles
   under each variant.
4. If v2 raises BUY/SELL recall without hurting precision by >2pp,
   ship the prompt flip. Otherwise discard.

### Action this PR

* TODO comment added to `portfolio/qwen3_trader.py:_build_prompt`
  documenting the suspicion + the A/B requirement so future readers
  do not flip the prompt blindly.
* No code change to the prompt itself.

### Status

`SUSPECTED — A/B needed before fix`

---

## Defer list (open questions for next session)

* `finance_llama` 73% abstain rate in production — Plex-VRAM gate may
  be over-eager; parse failures may be cleanable.
* 30pp gap `accuracy_cache.json` (ministral 58%) vs short-window log
  (ministral 20% directional) — sample-population mismatch, not a
  methodology bug, but worth a one-time consolidation pass so the
  dashboard does not show two different "accuracies" for the same
  signal.
* `meta_trader` Qwen2-36L real wiring (Item 3 of the shadow plan).
  Multi-session.
* Brier denominator: current `dashboard/app.py:_compute_llm_leaderboard`
  Brier uses the directional set (same as accuracy denominator). If
  we want Brier over the full distribution including HOLD, add a
  separate field rather than changing the existing one.
