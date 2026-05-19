# PLAN — Add Gemma 4 E4B To The Shared GPU LLM Loop (2026-04-16)

## Goal

Add `gemma4` as a fourth local LLM in the live signal loop without replacing
`ministral` or `qwen3`.

Initial rollout is **shadow mode**:

- Gemma runs on GPU.
- Gemma computes a real signal on the same path as the other local LLMs.
- Gemma is tracked as if it voted.
- Gemma does **not** affect live consensus yet.

The implementation must satisfy all of these:

- Run on the local GPU.
- Use the same shared `llama-server` / `llm_batch` pipeline as the other local
  LLMs.
- Participate in the same cache, batching, rotation, gating, and outcome
  tracking flow as the other local LLMs.
- Be measurable afterward so we can decide whether it is actually useful.

## Hard Constraints

- Use `Gemma 4 E4B`, not `26B A4B` or `31B`.
  The larger variants do not fit the 10 GB RTX 3080 budget used by this repo.
- Use the shared `portfolio.llama_server` path with `-ngl 99` and `-c 4096`.
- Do not add an Ollama-only path.
- Do not add a separate long-running Gemma daemon.
- CPU subprocess fallback is acceptable for parity with the existing wrappers,
  but GPU via shared `llama-server` is the primary path.
- Keep the signal additive.
  No replacement of `ministral`, `qwen3`, or `fingpt`.

## Current Repo State

### Shared runtime

- [`portfolio/llama_server.py`](Q:/finance-analyzer/portfolio/llama_server.py)
  currently knows `ministral3`, `qwen3`, `ministral8_lora`, and
  `finance-llama-8b`.
- The shared server is explicitly designed for one local GGUF at a time on the
  10 GB card.

### Batch scheduler

- [`portfolio/llm_batch.py`](Q:/finance-analyzer/portfolio/llm_batch.py)
  currently queues `ministral`, `qwen3`, and `fingpt`.
- Rotation is currently:
  `("ministral", "qwen3", "fingpt")`.
- [`portfolio/main.py`](Q:/finance-analyzer/portfolio/main.py) only snapshots
  queued keys for Ministral and Qwen3 before `flush_llm_batch()`.

### Signal integration

- [`portfolio/signal_engine.py`](Q:/finance-analyzer/portfolio/signal_engine.py)
  has explicit voting blocks for `ministral` and `qwen3`.
- Local-model gating is generic via `_gate_local_model_vote(signal_name, ...)`,
  so Gemma can reuse that path cleanly.
- Fallback per-ticker accuracy override later in the file only knows about
  `("qwen3", "ministral")`.

### Tracking and observability

- [`portfolio/outcome_tracker.py`](Q:/finance-analyzer/portfolio/outcome_tracker.py)
  only derives passthrough votes for `ministral` and `qwen3`.
- [`portfolio/local_llm_report.py`](Q:/finance-analyzer/portfolio/local_llm_report.py)
  only reports `ministral` today.
  That is already a measurement gap for `qwen3`; Gemma will make that gap worse
  unless the report is generalized.

### Operator/config/docs

- [`scripts/download_models.py`](Q:/finance-analyzer/scripts/download_models.py)
  has no Gemma entry.
- [`config.example.json`](Q:/finance-analyzer/config.example.json) has a
  `local_models.ministral` block but no `qwen3` or `gemma4` example block.
- The Gemma E4B GGUF is not present on this machine yet.

## Target Design

### Canonical names

- Signal name: `gemma4`
- Shared-server model key: `gemma4`
- Trader wrapper: `portfolio/gemma4_trader.py`
- Signal wrapper: `portfolio/gemma4_signal.py`
- Model file:
  `Q:\models\gemma-4-e4b-gguf\gemma-4-E4B-it-Q4_K_M.gguf`

### Recommended config contract

Use an explicit mode flag instead of overloading `enabled`:

```json
{
  "local_models": {
    "gemma4": {
      "enabled": true,
      "mode": "shadow",
      "hold_threshold": 0.55,
      "min_samples": 30,
      "accuracy_days": 30
    }
  }
}
```

Why this shape:

- `enabled: false` means do not run Gemma at all.
- `enabled: true, mode: "shadow"` means run inference and track it, but force
  the effective vote to `HOLD`.
- `enabled: true, mode: "vote"` means promote Gemma to a normal live voter.

This is cleaner than using `enabled = "shadow"` because the existing
`local_models.*.enabled` callers are written as booleans.

### Shadow-mode precedent in this repo

There are already two strong precedents:

- [`portfolio/signals/forecast.py`](Q:/finance-analyzer/portfolio/signals/forecast.py)
  supports `kronos_enabled = "shadow"`:
  Kronos runs real inference, stores raw actions like `kronos_1h_raw`, but
  forces the live sub-signal vote to `HOLD`.
- [`portfolio/sentiment.py`](Q:/finance-analyzer/portfolio/sentiment.py)
  runs FinGPT/FinBERT as shadow models:
  they compute in the background and log A/B results without affecting the
  primary sentiment vote.

Gemma should combine those patterns:

- shared GPU background execution like FinGPT
- force-HOLD live behaviour like Kronos
- raw vote preservation for later accuracy measurement

### Required runtime path

Gemma must follow the exact same critical path as the existing local voting
models:

1. `signal_engine.generate_signal()` builds LLM context.
2. `_cached_or_enqueue(...)` returns cached/stale data or queues the request.
3. `llm_batch.flush_llm_batch()` groups queued Gemma requests into one phase.
4. `query_llama_server_batch("gemma4", ...)` runs them through the shared
   GPU server.
5. `main.py` writes flushed results back into shared cache.
6. `signal_engine` gates the raw Gemma vote using per-ticker accuracy.
7. `outcome_tracker` logs `gemma4` votes so accuracy accumulates naturally.
8. `local_llm_report` exposes Gemma accuracy once samples exist.

If any step above is skipped, Gemma is not actually in the same pipeline.

### Shadow-mode mechanics

The critical implementation detail is **where** Gemma gets forced to `HOLD`.

Correct behavior:

1. Gemma runs through the normal local-LLM path and produces a real
   `raw_action`.
2. That raw action is stored in:
   - `extra_info["gemma4_raw_action"]`
   - `extra_info["gemma4_action"]` as the gated action before shadow forcing
   - `votes["gemma4"]` temporarily as the gated action
3. `signal_engine` snapshots `raw_votes = dict(votes)` before later rewrites.
4. Only **after** that snapshot, if Gemma is in shadow mode, force:
   `votes["gemma4"] = "HOLD"`.
5. Weighted consensus and buy/sell counts see Gemma as `HOLD`.
6. [`portfolio/outcome_tracker.py`](Q:/finance-analyzer/portfolio/outcome_tracker.py)
   logs `_raw_votes` preferentially, so Gemma's raw BUY/SELL/HOLD history still
   accumulates in the signal log for future `accuracy_by_signal_ticker("gemma4")`.

Wrong behavior to avoid:

- Forcing Gemma to `HOLD` inside the Gemma signal block before `_raw_votes`
  is captured.
  That would make Gemma look like permanent `HOLD` noise and destroy the point
  of shadow evaluation.

### Shadow-mode effect on live counts

While Gemma is in shadow mode, it should **not** inflate live
`_total_applicable`.

Reason:

- If Gemma is force-held for live consensus, counting it as an applicable live
  signal would distort UI/reporting counts without providing a real vote.

Recommended implementation:

- Extend `_compute_applicable_count(...)` to accept `config`.
- Exclude `gemma4` from the applicable count when
  `local_models.gemma4.mode == "shadow"`.

This keeps live counts stable until Gemma is promoted.

## Implementation Batches

### Batch 1 — Assets and single-model wrapper

Files:

- `portfolio/llama_server.py`
- `portfolio/gemma4_trader.py` (new)
- `portfolio/gemma4_signal.py` (new)
- `scripts/download_models.py`
- `config.example.json`

Work:

- Add a new `_MODEL_CONFIGS["gemma4"]` entry in `llama_server.py`.
- Point it at the Gemma 4 E4B GGUF path on Windows/Linux.
- Keep launch flags aligned with the existing shared-server path:
  `-ngl 99`, `-t 4`, `-c 4096`.
- Create `gemma4_trader.py` with the same repo-local contract as the other
  trader wrappers:
  `_build_prompt(context)`, `_parse_response(text)`, `predict(context)`.
- Create `gemma4_signal.py` for parity and future direct calls, even though the
  steady-state loop will use `llm_batch`.
- Extend `scripts/download_models.py` with `--gemma4`.
- Add a conservative `local_models.gemma4` example config block.
  Recommended example default:
  `enabled: false` until the model file is present and a smoke test passes,
  with `mode: "shadow"` as the first enabled state.

Notes:

- Prompt formatting must use the real Gemma instruct template supported by the
  GGUF/llama.cpp path at implementation time.
  Do not guess token framing from another model family.
- The parser must return the same normalized shape used by the other LLM
  wrappers: `action`, `reasoning`, optional `confidence`, stable `model`.

Tests:

- Add import/path/parser tests modeled on `tests/test_model_upgrades.py`.
- Add direct wrapper tests modeled on the existing `qwen3_signal` /
  `ministral_signal` tests.

### Batch 2 — Shared batch queue and rotation

Files:

- `portfolio/llm_batch.py`
- `portfolio/main.py`
- `portfolio/shared_state.py`
- `tests/test_llm_batch.py`

Work:

- Add `_gemma4_queue` and `enqueue_gemma4(cache_key, context)`.
- Add a Gemma phase to `flush_llm_batch()`.
- Put Gemma before `fingpt` so the voting models still run before the shadow
  sentiment phase.
  Recommended order:
  `ministral -> qwen3 -> gemma4 -> fingpt`.
- Extend `_LLM_ROTATION` to include Gemma:
  `("ministral", "qwen3", "gemma4", "fingpt")`.
- Update `main.py` queued-key snapshot/cleanup so Gemma loading keys are
  released correctly when a Gemma batch returns no result.

Important operational consequence:

- Adding a fourth rotating LLM increases staleness windows.
- Current call sites use `max_stale_factor=5` with a 15 min TTL because the
  3-model rotation gives roughly 45-60 min between fresh votes.
- With 4 models, that becomes roughly 60-80 min.
- The Gemma implementation batch must review stale tolerance and likely raise
  it for all rotating local LLM callers to avoid avoidable cache misses and
  enqueue churn.

Tests:

- Extend `tests/test_llm_batch.py` beyond fingpt-only coverage to verify:
  - Gemma queue dedup
  - Gemma phase ordering
  - queued-key cleanup when Gemma returns nothing
  - rotation behaviour with 4 slots

### Batch 3 — Signal engine, vote plumbing, and accuracy logging

Files:

- `portfolio/tickers.py`
- `portfolio/signal_engine.py`
- `portfolio/outcome_tracker.py`
- `tests/test_model_upgrades.py`
- `tests/test_signal_pipeline.py`
- `tests/test_consensus.py`
- `tests/test_outcome_tracker_core.py`
- `tests/test_per_ticker_accuracy_override.py`

Work:

- Add `gemma4` to `SIGNAL_NAMES`.
- Add `gemma4` to `GPU_SIGNALS`.
- Add `gemma4` to `CORE_SIGNAL_NAMES`.
- Add a Gemma voting block in `signal_engine.py` matching the `qwen3` pattern:
  - build context
  - enqueue via `_cached_or_enqueue`
  - gate via `_gate_local_model_vote("gemma4", ...)`
  - store `gemma4_raw_action`, `gemma4_action`, `gemma4_reasoning`,
    `gemma4_accuracy`, `gemma4_samples`, `gemma4_gating`,
    optional `gemma4_confidence`
- Add a local-model mode helper for Gemma shadow vs vote.
- In shadow mode:
  - let Gemma produce a normal gated action first
  - preserve that action in `_raw_votes`
  - force the effective `votes["gemma4"]` to `HOLD` before live consensus
    counts and weighting
- Extend the per-ticker LLM accuracy fallback override loop to include
  `gemma4`.
- Extend `outcome_tracker._derive_signal_vote()` with a `gemma4` passthrough.

Decision for applicability:

- Gemma should be all-ticker like `qwen3`, not crypto-only.
- Do not add a special-case `if sig == "gemma4" and not is_crypto`.
- While in shadow mode, Gemma should still execute for all supported tickers,
  but it should be excluded from `_total_applicable`.

Pre-existing nuance to handle deliberately:

- `_compute_applicable_count()` still has an old `ministral` crypto-only
  special-case even though the live dispatch block now runs Ministral for all
  tickers.
- Do not accidentally copy that drift into Gemma.
- Either:
  - leave the existing Ministral inconsistency untouched and add Gemma cleanly,
    or
  - fix the Ministral inconsistency in the same batch with explicit test updates.
- Do not mix the two by accident.

Initial weighting/blacklist policy:

- Do not add Gemma-specific horizon boosts or regime boosts on day one.
- Let Gemma start with default weighting until real samples exist.
- Do not add ticker blacklists without evidence.
  The existing per-ticker blacklist system is for measured failures, not for
  speculative preemption.
- While Gemma is in shadow mode, its live weighting is irrelevant because the
  effective vote is forced to `HOLD`. Preserve the raw gated action anyway so
  the eventual promotion decision is based on realistic pre-promotion behavior.

Tests:

- Update hard-coded applicable-count expectations.
  Gemma will increase total applicable signal counts where GPU signals are
  allowed.
- Add passthrough tests for `gemma4_action`.
- Add signal-engine integration tests mirroring the existing Qwen3 tests.

### Batch 4 — Reporting and operator visibility

Files:

- `portfolio/local_llm_report.py`
- `tests/test_local_llm_report.py`
- Optional follow-up:
  - `portfolio/main.py`
  - `portfolio/analyze.py`
  - `tests/test_analyze.py`

Work:

- Generalize `local_llm_report.py` from a single-model Ministral report into a
  local-LLM matrix covering at least:
  - `ministral`
  - `qwen3`
  - `gemma4`
- Pull per-ticker accuracy for each model via the same `accuracy_by_signal_ticker`
  path so comparisons are apples-to-apples.
- Keep config sanitization intact.
- Make Gemma shadow status visible in the report/config view so operators can
  distinguish:
  - disabled
  - enabled shadow
  - enabled voting

Optional but recommended operator visibility:

- Add Gemma summary output to the compact runtime log/status line in `main.py`.
- Add Gemma reasoning to manual analysis/watch prompts if that improves operator
  debugging.

Rationale:

- Without report support, Gemma can vote but cannot be judged.
- The repo already has a blind spot here because `qwen3` is not included in the
  local LLM report.
  Fix that once, not three times.

## Rollout Sequence

1. Land Batch 1 with imports/parser tests only.
2. Download the Gemma GGUF to `Q:\models\gemma-4-e4b-gguf\`.
3. Smoke-test the wrapper on one ticker outside the live loop.
4. Land Batch 2 and verify `flush_llm_batch()` emits a Gemma phase.
5. Land Batch 3 and run targeted signal/outcome tests with
   `local_models.gemma4.mode = "shadow"`.
6. Land Batch 4 so Gemma usefulness can be measured instead of guessed.
7. Let Gemma accumulate shadow samples and compare it against Ministral/Qwen3.
8. Promote by config flip only:
   `local_models.gemma4.mode = "vote"`.
9. Run paper/observation cycles before trusting any weight or blacklist changes.

## Verification Checklist

Implementation is not complete until all of these are true:

- `llama-server` can load Gemma E4B from the configured GGUF path.
- `nvidia-smi` shows Gemma inference consuming GPU memory during the batch phase.
- Logs show a Gemma phase in `flush_llm_batch()`.
- `agent_summary` / signal snapshots contain a `gemma4` vote path.
- In shadow mode, live consensus math still treats Gemma as `HOLD`.
- `outcome_tracker` writes Gemma votes into the signal log.
- `accuracy_by_signal_ticker("gemma4", ...)` starts returning data once
  outcomes backfill.
- `local_llm_report` shows Gemma beside Ministral/Qwen3.

## Risks

- Prompt-template mismatch:
  Gemma may underperform or produce malformed JSON if the wrong chat template or
  stop tokens are used.
- Rotation staleness:
  a 4-model rotation may push local-LLM freshness farther out than the current
  stale allowance expects.
- Shadow-force timing:
  if Gemma is forced to `HOLD` too early, the raw signal never gets logged and
  the shadow test becomes useless.
- Measurement blind spot:
  if report/tracking changes are skipped, Gemma can vote without ever being
  auditable.
- Hard-coded count regressions:
  several tests assume fixed applicable counts that will move by +1.
- False certainty from day-one weights:
  Gemma should not receive special boosts before it has real outcome history.

## Non-Goals

- No replacement of existing local LLMs.
- No Gemma fine-tune/LoRA work in this task.
- No separate Ollama integration path.
- No immediate Gemma-specific weight boosts, blacklist additions, or strategy
  tuning before outcome data exists.

## Success Criteria

Gemma is considered correctly integrated when it is:

- queued by `signal_engine`
- flushed by `llm_batch`
- executed by the shared GPU `llama-server`
- cached and rotation-aware like the other local LLMs
- included in consensus as a first-class signal
- logged into outcome tracking
- visible in local-LLM reporting for later usefulness decisions
