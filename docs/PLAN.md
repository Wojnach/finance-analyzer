# PLAN — Active Work (2026-04-16)

## Current Active Plan

Add `Gemma 4 E4B` as the fourth local LLM in the shared GPU loop, starting in
shadow mode so it computes real signals without affecting live consensus.

Detailed plan:
`docs/plans/2026-04-16-gemma4-loop-plan.md`

### Non-Negotiables

- Gemma must run on GPU through the shared `portfolio.llama_server` path.
- Gemma must use the same `_cached_or_enqueue -> llm_batch ->
  query_llama_server_batch -> outcome_tracker` pipeline as the other local
  LLMs.
- Initial rollout is `enabled=true, mode="shadow"`: preserve Gemma raw votes
  for tracking, but force its effective live vote to `HOLD`.
- The additive target model is `Gemma 4 E4B` only.
  The larger Gemma 4 variants do not fit this repo's 10 GB VRAM budget.
- The plan includes queueing, rotation, gating, accuracy tracking, and report
  visibility.
  A wrapper alone is not enough.

### Execution Order

1. Add Gemma assets/wrappers and shared-server model registration.
2. Extend `llm_batch` / `main.py` rotation and queue handling.
3. Wire Gemma into `signal_engine`, `tickers`, and `outcome_tracker`.
4. Generalize the local-LLM report so Gemma usefulness can be measured.

---

## Completed plans (archived)

- `docs/plans/2026-04-16-accuracy-gating-plan.md` — accuracy gating
  reconfiguration shipped in merge `a739a56` (5 batches; Tier-1 1d
  replay delta +3.80pp average, MSTR +5.80pp, XAG +9.55pp).
