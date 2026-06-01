# Meta-Trader (Qwen2-3B) shadow wiring — PLAN (2026-06-01)

Status: **PLAN ONLY — implementation deferred to a supervised session.**
Written during a Ralph-loop iteration; the loop self-terminated here because
the implementation is NOT safe bounded autonomous work (see §Why deferred).

## Goal
Wire real inference for the `meta_trader` shadow signal (currently a scaffold:
`portfolio/signals/meta_trader.py`, `_FEATURE_AVAILABLE=False`) and enroll it in
`signal_engine._SHADOW_LLM_SIGNALS` so it collects directional-accuracy data
(force-HOLD in consensus, like phi4_mini/finance_llama/cryptotrader_lm).

## Model (verified 2026-06-01)
`Q:/models/custom-meta-trader/` — `Qwen2ForCausalLM`, 36 layers, hidden 2048,
36 layers, vocab 151936, 32K context, bf16, ~6.17 GB safetensors (2 shards).
~3B params. unsloth-fixed. Standard Qwen2 arch → convertible with the existing
`Q:/models/convert_hf_to_gguf.py`. Has `chat_template.jinja` + tokenizer.json.
**No GGUF exists yet.**

## What makes meta_trader different (the hard part)
It is a **meta-model**: its value is consuming the OTHER voters' outputs as
features, not re-reading OHLCV (that would just duplicate qwen3/ministral). Per
the scaffold contract it must:
  1. run AFTER ministral/qwen3/finance_llama/cryptotrader_lm in the cycle,
  2. read their per-model {action, confidence} for the current ticker,
  3. synthesize a verdict.
The "dispatched AFTER" ordering is a REQUIREMENT that is **not enforced** today.

## Implementation steps
1. **Convert to GGUF** (offline). Needs `gguf` pip pkg (absent from both venvs)
   + torch (main .venv has it). `pip install gguf` into `.venv`, then:
   `.venv/Scripts/python.exe Q:/models/convert_hf_to_gguf.py Q:/models/custom-meta-trader --outfile Q:/models/custom-meta-trader/custom-meta-trader-q8_0.gguf --outtype q8_0`
   (q8_0 ~3.2 GB, good quality for a 3B; or f16→llama-quantize Q4_K_M ~2 GB).
   Verify with a one-shot llama-server probe.
2. **llama_server slot** `meta_trader` in `_MODEL_CONFIGS` (path + `extra_args: []`).
3. **Upstream-feature plumbing** (the careful hot-loop change):
   - Enforce meta_trader dispatch order: it must compute after the other LLM
     signals have populated `extra_info[{sig}_action]/{sig}_confidence]`.
     Options: (a) sort the enhanced-signal dispatch so `_SHADOW_LLM_SIGNALS`
     run last with meta_trader strictly last; (b) in the shadow-LLM elif,
     special-case meta_trader to build `context["upstream_llm_votes"]` from
     `extra_info` (the other LLMs' actions already set this cycle).
   - Confirm ministral/qwen3 actually land in `extra_info`/`votes` BEFORE the
     enhanced disabled-branch runs (they're active LLM voters — verify the
     ordering empirically; if they compute in a separate phase, the upstream
     dict may be empty and meta_trader degrades to OHLCV-only = low value).
4. **Wrapper rewrite** `meta_trader.py`: build the meta-prompt from
   `upstream_llm_votes` + a compact OHLCV summary (use `chat_template.jinja`'s
   format), `query_llama_server("meta_trader", ...)`, parse decision+confidence
   (reuse the phi4 parser pattern — strip any think, anchored confidence regex,
   abstain on no-parse). Set `_FEATURE_AVAILABLE=True`. n_predict sized to the
   model's verbosity (probe first; Qwen2-instruct is not a reasoning model so
   likely ~256-512 suffices, unlike phi4's 4096).
5. **Enroll**: add `meta_trader` to `_SHADOW_LLM_SIGNALS`. cycle_modulo=5,
   phase=2 already in registry. CHECK co-fire budget: meta_trader(cyc%5==2) vs
   finance_llama(cyc%3==2)/cryptotrader(cyc%3==0)/phi4(cyc%10==1) — meta_trader
   is the heaviest; ensure it never co-fires with phi4 (both heavy). If it does
   on some cyc, re-phase so at most one HEAVY shadow (phi4|meta_trader) per cycle.
6. **Test**: dispatch tests (order enforced, upstream dict populated, co-fire
   budget incl. meta_trader) + wrapper parse/abstain tests + a real probe.

## Premortem (pre-implementation, abbreviated)
- **P0 cycle blowout**: meta_trader (~3B, 32K ctx) + phi4 (22s) co-firing →
  two heavy calls/cycle. Mitigation: phase so heavy shadows never co-fire;
  assert in the co-fire test.
- **P1 empty upstream features**: if the other LLM votes aren't in `extra_info`
  when meta_trader runs, it silently degrades to a redundant OHLCV reasoner and
  its "meta" accuracy is meaningless. Detection: log the upstream-vote count;
  abstain if < 2 upstream votes present.
- **P1 dispatch-order coupling**: forcing meta_trader last is a global ordering
  change that could perturb other signals' `_set_last_signal` diag / timing.
  Keep the change minimal (sort key, not a rewrite); full regression suite.
- **P2 prod-venv dep**: `gguf` install touches the production env. Pure-python,
  import-only-at-conversion (loop never imports it) → low risk, but do it in a
  supervised session, not autonomously.

## Why deferred (not autonomous-loop work)
Two reasons this needs a human-in-the-loop session, not an unsupervised loop:
1. **Prod-venv dependency install** (`gguf`) — a system change on a live box.
2. **Hot-loop dispatch-ordering change** (step 3) — touches the live signal
   flow ordering, not a contained shadow addition. Reliability #1.
Everything up to here (model verified, GGUF path, design, premortem) is the
safe part and is captured above. Resume from step 1 with supervision.
