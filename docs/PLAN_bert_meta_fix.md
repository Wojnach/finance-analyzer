# PLAN — bert_sentiment meta-tensor defensive load (2026-05-04)

**Date:** 2026-05-04
**Branch:** `fix/bert-meta-tensor-2026-05-04`
**Scope:** 1 module (`portfolio/bert_sentiment.py`) + 1 test file. Detect &
retry meta-tensor corruption at BERT model load time so silent FinBERT
prediction failures stop polluting the A/B sentiment log.

> Written to `PLAN_bert_meta_fix.md` (not `PLAN.md`) to avoid clobbering
> any in-flight uncommitted edits to the canonical plan file.

---

## Context — observed in production tonight

Starting `2026-05-03 23:38:02` (right after a loop restart following the
fingpt-observability merge), every FinBERT prediction in every cycle has
been failing with:

```
2026-05-04 00:27:36 [WARNING] portfolio.bert_sentiment:
  BERT FinBERT batched predict failed, falling back to per-text loop:
  Tensor on device meta is not on the expected device cpu!
2026-05-04 00:27:36 [WARNING] portfolio.bert_sentiment:
  BERT FinBERT per-text predict failed for '<headline>':
  Tensor on device meta is not on the expected device cpu!
```

~20-30 such warnings per cycle. All headlines hit the per-text fallback
which also fails, producing a zero-confidence neutral placeholder. The
A/B log (`data/sentiment_ab_log.jsonl`) ends up with zeroed FinBERT
shadow rows for the entire post-23:38 period.

CryptoBERT and Trading-Hero-LLM do **not** show this — only FinBERT.

## Root cause

Race condition between Chronos's CUDA load and BERT's CPU load, both
happening on different ticker threads via the main loop's
`ThreadPoolExecutor` (8 workers).

**Triggering commit:** `789cc91c` ("perf(forecast): run Chronos before
Kronos to unblock GPU on cold start") at 21:08 UTC on 2026-05-03.
Pre-commit, Kronos's subprocess held the GPU file-lock at the start of
the forecast phase, so Chronos's in-process load happened *after* the
sentiment phase had finished loading BERTs sequentially. Post-commit,
Chronos loads first, *during* the parallel ticker phase, *concurrently*
with the BERT loads.

Compared traces (extracts):

| time | event | concurrent? |
|---|---|---|
| **20:35:34** (worked) | Trading-Hero-LLM loading | — |
| **20:35:35** | FinBERT loading | sequential |
| **20:35:38** | CryptoBERT loading | sequential |
| **(later)** | Chronos loading | after BERTs |
| **23:38:01** (broken) | Trading-Hero-LLM loading | — |
| **23:38:02** | Chronos-2 loading on CUDA | **overlaps FinBERT** |
| **23:38:02** | FinBERT loading | **overlaps Chronos** |
| **23:38:04** | first FinBERT predict failure | — |

**Why FinBERT specifically:** loaded from a snapshot path
(`Q:\models\finbert\models--ProsusAI--finbert\snapshots\<hash>`)
without `cache_dir` / `local_files_only` kwargs. The snapshot dir
contains both `pytorch_model.bin` *and* `flax_model.msgpack` *and*
`tf_model.h5`, putting `transformers.from_pretrained()` into a code
path that's more sensitive to `accelerate`'s lazy/meta init when CUDA
init is happening on another thread. The other two BERTs use the
standard `cache_dir + hf_name` path which doesn't hit it.

**Standalone reproduction failed.** A bare-script load of FinBERT from
the same snapshot returns 0 meta params and a clean forward pass. The
race needs the loop's specific concurrent-thread timing — which is why
this slipped past unit tests for `bert_sentiment.py`.

**Impact:** FinBERT is shadow-only (primary sentiment is CryptoBERT for
crypto / Trading-Hero-LLM for stocks). Voting is unaffected. The A/B
accuracy comparison between FinBERT shadow and the primary is broken
until this fix lands — every shadow entry shows a zero-confidence
neutral placeholder regardless of the actual headline.

## What this PR does

### Change: defensive meta-tensor detection + retry in `_load_model`

In `portfolio/bert_sentiment.py:_load_model`, immediately after the
`AutoModelForSequenceClassification.from_pretrained(...)` call, before
the `model.train(False)` line:

```python
# 2026-05-04 (fix/bert-meta-tensor): defensive meta-tensor detection.
# Race between Chronos's CUDA load and concurrent BERT loads (commit
# 789cc91c, 2026-05-03) can leave some FinBERT weights on the `meta`
# device when accelerate's lazy init interleaves with CUDA init on
# another thread. Without this guard, predict-time forward passes
# silently fail per-text with "Tensor on device meta is not on the
# expected device cpu!" and the per-text fallback writes a zero-
# confidence neutral placeholder for every headline. Detect at load
# time, retry once with explicit eager-init kwargs, fail loudly if
# still broken so the caller's try/except routes around the bad
# model rather than corrupting the A/B log.
if any(p.is_meta for p in model.parameters()):
    logger.warning(
        "BERT %s loaded with meta tensors (likely accelerate race with "
        "concurrent CUDA load); retrying with eager init",
        name,
    )
    if name == "FinBERT" and snapshot is not None:
        model = AutoModelForSequenceClassification.from_pretrained(
            snapshot,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_name,
            cache_dir=cache_dir,
            local_files_only=config.get("local_files_only", False),
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
        )
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError(
            f"BERT {name} still has meta tensors after retry "
            f"(accelerate version: {_accelerate_version() or 'unknown'})"
        )
```

The retry uses:
- `low_cpu_mem_usage=False` — explicitly disables `accelerate`'s
  init-empty-weights-then-load path, even though it's the default;
  belt-and-suspenders if the default got flipped by a transformers
  upgrade or by env state.
- `torch_dtype=torch.float32` — forces eager allocation in float32
  (FinBERT was trained as float32 anyway, no precision change).

If the retry still produces meta tensors, raise — caller in
`_get_model` sees the exception, doesn't cache the broken model in
`_models`, next predict call will try again from scratch. If the
underlying race is permanent, every predict call retries, which is
expensive but loud. Better than silent corruption.

### Tests

`tests/test_bert_sentiment.py` — new tests in
`TestMetaTensorRecovery`:

1. `test_load_with_meta_tensors_retries_with_eager_init` — patch
   `AutoModelForSequenceClassification.from_pretrained` to return a
   model with one meta param on the first call, a clean model on the
   second. Assert `_load_model` retries and returns the clean one.
2. `test_load_with_persistent_meta_tensors_raises` — patch to return
   meta tensors on both calls. Assert RuntimeError.
3. `test_clean_load_does_not_retry` — happy-path: no meta tensors,
   `from_pretrained` called exactly once.
4. `test_meta_warning_logged_on_retry` — assert the warning text
   includes "meta tensors" and the model name.

Each test stubs `AutoModelForSequenceClassification.from_pretrained`
and `AutoTokenizer.from_pretrained` so no real model files are
touched — runs in <1s.

## What this PR does NOT do

- Does not serialize Chronos vs BERT loads in `main.py`. That's the
  cleaner fix for the underlying race but adds startup latency (the
  whole point of the parallel executor). The defensive detection is
  non-invasive and handles the race without changing scheduling.
- Does not alter the per-text fallback loop. If a clean FinBERT load
  happens but a *single* prediction errors for an unrelated reason
  (tokenizer edge case), the per-text fallback still kicks in — same
  behavior as before this PR.
- Does not change CryptoBERT / Trading-Hero-LLM behavior — they don't
  hit the bug, but the defensive check applies to them too. Free
  insurance for any future regression.

## Risk

- Low. The defensive check runs once per model at load time
  (`_load_model` is called inside `_init_lock` from `_get_model`).
  Cost: one `is_meta` walk over 200 parameters = <1ms.
- Retry path uses kwargs that should be no-ops in the happy case
  (eager init is already the default). Worst case: retry doubles load
  time for the corrupted-load cycle.
- If the retry succeeds, the loop transitions from "all FinBERT
  predictions fail silently" to "FinBERT works correctly", which is
  the desired behavior. No way for the fix to break anything that's
  currently working.
- Loop restart required to pick up the change (per ops-lesson:
  `taskkill /F` not `schtasks /end`, the singleton lock can outlast a
  graceful term).

## Execution

| Batch | Files | Tests |
|---|---|---|
| 1 | `portfolio/bert_sentiment.py` | `tests/test_bert_sentiment.py` |

Single batch.

## Verification after merge + restart

1. `taskkill /F` the loop's main python (the >500 MB one).
2. Wait for bat wrapper to auto-restart.
3. Tail `data/loop_out.txt`:
   ```
   tail -F data/loop_out.txt | grep -aE "BERT.*meta|BERT FinBERT.*predict"
   ```
4. **Expected:** zero "predict failed" lines after the fresh load. If
   the race occurs, expect ONE warning line "BERT FinBERT loaded with
   meta tensors, retrying with eager init" followed by silence — no
   per-prediction failures.
5. After ~10 min, check `data/sentiment_ab_log.jsonl` — recent FinBERT
   shadow rows should have non-zero confidence and real sentiment
   labels.
