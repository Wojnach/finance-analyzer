# Local LLM Accuracy Plan — 2026-03-09

## Goal

Improve the decision quality of the local-model paths (`Chronos`, `Kronos`, `Ministral`) without blindly swapping models in the live loop. The priority order is:

1. Improve calibration and abstention.
2. Improve observability and benchmark quality.
3. Only then swap or fine-tune models.

This follows the repo codex guidelines: isolated worktree, smallest safe change first, clear rollback, tests before trust.

## External Research

### Chronos / Chronos-Bolt / Chronos-2

- Amazon's Chronos repository documents `Chronos-Bolt` as materially faster and more accurate than the original T5 Chronos family for probabilistic forecasting, which matters because the repo still defaults to `amazon/chronos-t5-small`.
- Amazon also released `Chronos-2`, which adds multivariate forecasting and covariates. That is relevant here because this repo already has regime, sentiment, and cross-asset context that could become covariates instead of being fused only after the forecast.
- Repo implication: do not jump straight from `chronos-t5-small` to `Chronos-2` in production. First benchmark `chronos-bolt-small` as a like-for-like replacement, then test `Chronos-2` only if we are ready to build a covariate-aware evaluation harness.

Primary sources:
- https://github.com/amazon-science/chronos-forecasting
- https://huggingface.co/amazon/chronos-bolt-small

### Kronos

- Kronos is explicitly built for financial OHLCV/K-line data rather than generic time series, which makes it the right family to keep evaluating for market data.
- The paper/repo architecture uses tokenized market structure and sampling-based generation, which means decoding parameters and input resolution are not secondary details; they are part of model quality.
- Repo implication: accuracy improvements should come from walk-forward parameter search (`temperature`, `top_p`, `sample_count`, interval, lookback) and per-asset benchmarking, not from assuming one global setting is best.

Primary sources:
- https://github.com/ffjzz/Kronos
- https://arxiv.org/abs/2508.02738

### Ministral / newer Mistral small models

- The repo is currently on older `Ministral-8B-Instruct-2410`-era prompting, while Mistral has since shipped newer small/open model generations and current docs position `Ministral 8B` as replaced by newer small-model lines.
- That does not mean "swap immediately." It does mean the current local LLM path should be treated as legacy until benchmarked against a newer Mistral small model with the same prompt contract.
- Repo implication: fix the output contract and calibration first, because changing the base model before the measurement layer is trustworthy will just move the noise around.

Primary sources:
- https://docs.mistral.ai/getting-started/models/weights/
- https://mistral.ai/news/mistral-small-3-1

## Repo Findings

### 1. The forecast path is partially calibrated, but only at the composite level

- [`portfolio/signals/forecast.py`](Q:/finance-analyzer/.worktrees/local-llm-accuracy/portfolio/signals/forecast.py) already gates the composite `forecast` vote using per-ticker accuracy, volatility, and regime.
- It does **not** gate `chronos_1h`, `chronos_24h`, `kronos_1h`, and `kronos_24h` independently before majority voting.
- That means a bad sub-signal can still pollute the composite as long as the combined ticker-level history looks acceptable.

### 2. Raw forecast sub-signal accuracy tracking is fragile in practice

- The code can compute raw forecast sub-signal accuracy, but live data currently appears sparse or unbackfilled enough that `compute_forecast_accuracy()` often returns empty results.
- Meanwhile, local data shows forecast health is good enough to run, so the bigger issue is evaluation completeness, not just uptime.
- This makes "Kronos vs Chronos" decisions too anecdotal right now.

### 3. Ministral had a repo control gap

- [`portfolio/ministral_signal.py`](Q:/finance-analyzer/.worktrees/local-llm-accuracy/portfolio/ministral_signal.py) was launching a script outside the repo (`Q:\models\...`), which means repo-side prompt/parser changes would not reliably affect runtime behavior.
- That breaks the version-control loop and makes accuracy debugging harder than it needs to be.

### 4. Ministral voting had no model-specific abstention gate

- `signal_engine.py` weighted all signals later, but `Ministral` still entered the vote set as a raw categorical action.
- For local LLMs, that is too trusting. These models need per-ticker gating the same way forecasts do.

### 5. Current measured behavior does not justify blindly increasing local-model weight

- Recent local data shows:
  - `Ministral` 30-day 1d accuracy is decent for BTC/ETH, but not so dominant that it should vote unconditionally.
  - metals-side local LLM / Chronos stats are actively weak enough that abstention and downweighting are the right immediate move.
  - forecast prediction logs are healthy enough operationally, but confidence is often tiny and gating often falls back to `insufficient_data`.

## Decision Matrix

| Option | Expected value | Risk | Complexity | Decision |
| --- | --- | --- | --- | --- |
| Swap `chronos-t5-small` to `chronos-bolt-small` immediately | Medium | Medium | Low | Defer until offline benchmark |
| Swap `Ministral-8B` to newer Mistral small model immediately | Medium | Medium/High | Medium | Defer until prompt + parser + benchmark layer is stable |
| Fine-tune custom local financial model now | Potentially high | High | High | Defer |
| Add model-specific abstention/calibration | High | Low | Low/Medium | Do now |
| Tighten prompt/output contracts | Medium | Low | Low | Do now |
| Add benchmark/reporting for per-model, per-ticker results | High | Low | Medium | Do next |

## Ordered Plan

### Phase 1 — Safety and measurement hardening

1. Keep all work in an isolated worktree branch.
2. Route `Ministral` subprocess execution through a version-controlled repo script.
3. Make `Ministral` output structured JSON first, with a conservative fallback parser.
4. Add per-signal, per-ticker accuracy gating for `Ministral`.
5. Add per-sub-signal gating for `Chronos`/`Kronos` so weak horizons abstain before the composite vote.

Expected outcome:
- Fewer low-quality local-model votes entering consensus.
- Better debuggability when model quality regresses.

### Phase 2 — Benchmarking and calibration

1. Backfill forecast outcomes consistently enough to populate raw sub-signal accuracy.
2. Add a repeatable local-model report for:
   - per-ticker accuracy
   - samples
   - uptime / failure rate
   - abstention rate
   - calibration gap
3. Benchmark `chronos-t5-small` vs `chronos-bolt-small` on the same historical windows.
4. Run Kronos walk-forward sweeps over:
   - `5m` vs `15m` vs `1h`
   - lookback depth
   - `temperature`
   - `top_p`
   - `sample_count`

Expected outcome:
- Evidence-based model routing instead of static assumptions.

### Phase 3 — Model upgrades

1. If `chronos-bolt-small` beats current baseline on your own data, promote it via config first, not code default.
2. If newer Mistral small models beat current `Ministral` under the exact same prompt/output contract, add a side-by-side shadow path before replacing the active one.
3. If Kronos consistently wins on crypto/metals while Chronos wins on stocks, route by asset class instead of enforcing one winner globally.

Expected outcome:
- Model choice becomes per-domain, not ideological.

### Phase 4 — Fine-tuning only if the benchmark layer is stable

1. Revisit local SFT/LoRA only after:
   - raw outcome collection is healthy
   - prompt contracts are stable
   - abstention/gating is in place
2. Fine-tune only on the repo's real feature schema and real decision horizon.

Expected outcome:
- Fine-tuning effort is aimed at measured bottlenecks rather than guesswork.

## Immediate Implementation in This Branch

This branch implements the safe Phase 1 pieces:

- `Ministral` now runs through the repo-managed script path.
- `Ministral` has accuracy-based abstention by ticker.
- `Ministral` prompt/output handling is stricter.
- `Chronos`/`Kronos` sub-signals are gated individually before composite voting.
- Raw forecast sub-signals are still logged so model benchmarking is preserved.

## Rollout Guidance

1. Run tests first.
2. Run a paper/dry-run report cycle.
3. Inspect:
   - `ministral_gating`
   - `forecast_subsignal_gating`
   - raw vs effective sub-signals
4. Only then consider enabling model swaps in `config.json`.
