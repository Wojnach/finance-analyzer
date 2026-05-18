# Plan — Fix shadow accuracy gate + cryptotrader_lm LoRA (2026-05-18)

## Scope

Two production bugs blocking the LLM shadow→promote pipeline. Both are
data-quality failures masquerading as success.

## Findings (2026-05-18 investigation)

### Bug A — accuracy gate counts abstain rows as predictions

`scripts/review_shadow_signals.py:_collect_stats` joins
`llm_probability_log.jsonl` against `llm_probability_outcomes.jsonl` and
counts every joined row in the denominator. Rows with
`confidence == 0.0` are abstentions emitted by the canonical
`_abstain()` helper in `portfolio/signals/finance_llama.py`,
`cryptotrader_lm.py`, `meta_trader.py` etc. — they should NOT count.

`scripts/review_shadow_signals.py --promote --dry-run` reports:

```
[DRY] would promote cryptotrader_lm: matched=779 accuracy=0.630 → promote
[DRY] would promote finance_llama:   matched=770 accuracy=0.635 → promote
[DRY] would promote meta_trader:     matched=732 accuracy=0.641 → promote
```

Reality (recomputed by filtering `confidence > 0`):

| signal           | matched | real_acc | abstain |
|------------------|--------:|---------:|--------:|
| cryptotrader_lm  |       0 |        – |    992  |
| meta_trader      |       0 |        – |    752  |
| finance_llama    |     123 |   0.439  |    692  |

`cryptotrader_lm` and `meta_trader` have zero real predictions. Outcome
backfill labels ~64% of 1d windows as HOLD; scaffold rows always emit
`chosen: "HOLD"`, so the join shows 64% accuracy on garbage. Without the
fix the next 03:30 cron run would auto-promote three broken voters into
production consensus.

### Bug B — cryptotrader_lm GGUF LoRA produces empty completions

Server probe with the actual production prompt:

```
prompt length: 1191 chars
status: 200
completion_tokens: 2
finish_reason: stop
text: '```'
```

`/v1/chat/completions` on the trivial prompt `Reply: HELLO` also
returns `content: ""`. Base Ministral-8B without the LoRA serves
correctly (verified upstream — `ministral` signal accumulates real
predictions). The breakage is the LoRA file.

GGUF header is well-formed:

```
metadata:
  general.architecture: llama
  general.type: adapter
  adapter.type: lora
  adapter.lora.alpha: 16.0
  general.base_model.0.name: Ministral 8B Instruct 2410

tensors (144 total):
  blk.0.attn_q.weight.lora_a  dims=[4096, 8] type=1
  blk.0.attn_q.weight.lora_b  dims=[8, 4096]
  blk.0.attn_v.weight.lora_a  dims=[4096, 8]
  blk.0.attn_v.weight.lora_b  dims=[8, 1024]   ← matches Ministral GQA kv_heads
```

So format is correct, base/adapter match. Two hypotheses for why
output collapses to EOS in ≤3 tokens:

1. **Homebrew conversion bug** — adapter_config.json declares
   `model_type: "gpt2"` (wrong; should be `mistral`). Some converter
   path may have used the wrong target-module mapping, scrambling the
   weights even though the resulting tensor shapes look OK.
2. **Quantization/precision mismatch** — LoRA was trained on bf16
   base, applied here against Q4_K_M. With `lora_alpha/r = 16/8 = 2.0`
   the scaled contribution lands in the residual stream of a heavily
   quantized base; the model may collapse to EOS.

User hint: "the LoRA might be our attempt to modify the original
cryptotrader" — meaning the GGUF was locally converted from the
PEFT safetensors via `convert_lora_to_gguf.py` against an updated
llama.cpp. Most plausible single root cause.

### Bug C — scaffolds pollute probability log every cycle

`meta_trader` still has `_FEATURE_AVAILABLE=False` and returns
`HOLD/conf=0` on every dispatch. Every dispatch reaches `signal_engine`
which calls `log_vote()` with that row. Result: 752 abstain rows for
meta_trader, 992 for cryptotrader_lm (because every call hits the
empty-completion path), 692 for finance_llama (Plex VRAM gate + parse
failures + ticker abstains).

The `bc2c659e` throttle-skip guard (2026-05-17) handles the
cycle-modulo throttle case but not the abstain-result case.

## What we will fix

### Batch 1 — Accuracy gate filters abstain + HOLD-only rows

`scripts/review_shadow_signals.py:_collect_stats`:

* Skip rows where `confidence <= 0` (canonical abstain signal).
* Skip rows where `chosen == "HOLD"`. HOLD is not a directional
  prediction; counting it against the outcome label inflates accuracy
  for HOLD-biased shadows. This matches the methodology in
  `data/accuracy_cache.json`'s `correct_buy + correct_sell` rule.

Add `tests/test_review_shadow_signals_filter.py` exercising:

* abstain rows excluded from denominator
* HOLD-on-HOLD ties don't count as correct (regression for the bogus
  64% pass)
* directional-only rows still count

### Batch 2 — signal_engine.log_vote skips abstain results

Extend the `bc2c659e` guard in `portfolio/signal_engine.py` so
`log_vote` is also skipped when:

* `confidence <= 0`, OR
* `indicators.get("feature_unavailable") is True`

This stops new pollution at the source. Existing rows stay (immutable
journal) — the Batch 1 filter handles them at read time.

Add `tests/test_signal_engine_log_vote_skip_abstain.py`.

### Batch 3 — cryptotrader_lm LoRA repair attempt

Try in order, stop at first success:

1. Run current `convert_lora_to_gguf.py` from
   `/mnt/q/models/llama.cpp` against
   `Q:\models\cryptotrader-lm\adapter_model.safetensors` to produce a
   fresh `cryptotrader-lm-lora.gguf`. Probe with real prompt;
   `completion_tokens > 5` and non-empty `text` ⇒ success.
2. If still empty: try with `--base Q:\models\ministral-8b-gguf\Ministral-8B-Instruct-2410-Q4_K_M.gguf`
   (newer converters accept GGUF base for shape lookup).
3. If both fail: retire the `cryptotrader_lm` shadow in
   `data/shadow_registry.json` with status=retired, notes documenting
   the LoRA bug + a follow-up TODO to re-enable via PEFT-in-Python
   path. Do NOT keep emitting abstain rows.

### Batch 4 — Registry hygiene

`data/shadow_registry.json` updates:

* `cryptotrader_lm`: result-dependent (kept-shadow with new notes, or
  retired).
* `meta_trader`: notes updated to say scaffold-only, abstains filtered
  by gate fix; do not retire (it's the next planned wiring per Item 3
  of the shadow plan).
* Reset `last_reviewed_ts` on the 3 affected entries so the next cron
  re-evaluates with the fixed gate.

## Files touched

| File | Change |
|---|---|
| `scripts/review_shadow_signals.py` | accuracy filter + docstring update |
| `tests/test_review_shadow_signals_filter.py` | NEW |
| `portfolio/signal_engine.py` | extend log_vote skip guard |
| `tests/test_signal_engine_log_vote_skip_abstain.py` | NEW |
| `Q:\models\cryptotrader-lm\cryptotrader-lm-lora.gguf` | regen attempt (outside repo) |
| `data/shadow_registry.json` | metadata cleanup |
| `docs/SESSION_PROGRESS.md` | session log entry |

## Verification

1. Unit tests added in batches 1 + 2 pass before commit.
2. `python scripts/review_shadow_signals.py --promote --dry-run` after
   Batch 1 + 4 should NOT propose to promote `cryptotrader_lm` or
   `meta_trader`. `finance_llama` likely won't either with the real
   filter (123 directional rows < 200 sample bar).
3. `pytest -n auto` green vs main baseline (modulo the documented
   worktree-symlink ~26 baseline failures).
4. `caveman:cavecrew-reviewer` on the diff — fix all P1/P2.
5. After merge + push + `PF-DataLoop` restart, watch one cycle:
   * scaffold-emitted abstain rows must NOT appear in new
     `llm_probability_log.jsonl` entries for `meta_trader`.
   * cryptotrader_lm: either real predictions appearing (regen worked),
     or zero rows because retired.

## Out of scope

* The 30 percentage-point gap between `accuracy_cache.json`
  (ministral 58% on 1d) and `llm_probability_log` directional-only
  accuracy (ministral 20% on 65 BUY+SELL rows). Two different
  accounting systems with different gating; needs separate
  investigation. Documented in SESSION_PROGRESS as a deferred item.
* The 73% abstain rate for `finance_llama` in production. Plex-VRAM
  guard + parse-failure path may both be over-eager. Defer.
* Item 3 (meta_trader real wiring), Item 5 (Brier UI), Item 8 (qwen3
  prompt revision) from the LLM-shadow plan — those are separate work.

## Premortem

(to be filled by general-purpose Agent before implementation)
