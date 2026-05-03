# PLAN — fingpt batch observability fix (2026-05-03)

**Date:** 2026-05-03
**Branch:** `fix/fingpt-batch-observability-2026-05-03`
**Scope:** 1 module + 1 test file. Make the fingpt LLM phase outcome legible from logs alone.

> Note: written to `PLAN_fingpt_observability.md` (not `PLAN.md`) because main has
> in-flight uncommitted edits to `PLAN.md` (midfinance follow-ups, 2026-05-02). Do
> not stomp the user's working state.

---

## Context — how this issue surfaced

While running `/fin-status` tonight I saw this in `data/loop_out.txt`:

```
20:29:45  LLM batch start: rotation_slot=fingpt counter=6 queues M=0 Q=0 F=6
20:29:45  LLM batch: 6 fingpt queries
20:29:53  llama-server finance-llama-8b ready on port 8787
20:29:56  LLM batch: 0 results in 10.4s (M:0 Q:0 F:6)
```

`"0 results"` looked like a silent failure: 6 prompts in, 0 sentiments out. I
even spent ~30 minutes on a wrong-endpoint diagnosis (hit `/v1/chat/completions`
with model=finance-llama-8b, found Qwen3 was loaded with empty `message.content`)
before grepping `data/sentiment_ab_log.jsonl` and finding 4 fully-populated
`fingpt:finance-llama-8b` entries timestamped `2026-05-03T18:29:56.4xxxxxZ` —
exactly the same cycle that logged `"0 results"`.

## Root cause of the misleading log

`portfolio/llm_batch.py:258`:

```python
logger.info("LLM batch: %d results in %.1fs (M:%d Q:%d F:%d)",
            len(results), elapsed, len(m_batch), len(q_batch), len(f_batch))
```

`results` is the dict from `_flush_via_server()` for Phase 1 (Ministral) and
Phase 2 (Qwen3). Phase 3 (`_flush_fingpt_phase`) does **not** add to `results` —
it stashes via `sentiment._stash_fingpt_result` directly into
`sentiment._pending_ab_entries`, and `flush_ab_log()` writes the rows out
post-cycle. So `len(results) == 0` whenever the cycle was fingpt-only, **whether
fingpt succeeded or failed**.

The line is doubly misleading because the trailing `(M:%d Q:%d F:%d)` shows the
**queue sizes** (input), not the result counts (output). Reader naturally pairs
them up as if they're matching, but they're not.

## Secondary problem — silent failure modes in `_flush_fingpt_phase`

The whole phase body is wrapped in:

```python
except Exception:
    logger.warning("LLM batch fingpt phase failed", exc_info=True)
```

If the parser regresses, or `fingpt_infer` import fails, or every server call
returns `None`, the loop logs **one** generic warning and moves on. There is
no per-cycle metric on fingpt success rate. A genuine silent fingpt failure
could last weeks before anyone noticed it (compare: the ~3-week silent Layer 2
auth outage from March-April 2026 referenced in `CLAUDE.md`'s startup check).

## What this PR does

### Change 1 — `_flush_fingpt_phase` returns metrics

Today it returns implicit `None`. After: it returns a dict on every code path
(success, partial, exception):

```python
{
  "queries": int,         # prompts sent to llama-server
  "received": int,        # non-None text completions back
  "parsed": int,          # parsed dicts (non-None) handed to _stash_fingpt_result
  "stashed_groups": int,  # distinct (ab_key, sub_key) tuples stashed
  "exception": str|None,  # exception class name if the bare except fired
}
```

If the function failed before measurement (e.g. `import fingpt_infer` raised),
the dict is `{"queries": 0, "received": 0, "parsed": 0, "stashed_groups": 0,
"exception": "ImportError"}`. The caller can always read the dict — no None
guard needed.

### Change 2 — summary log line includes fingpt stash count

Replace the misleading line with:

```python
logger.info(
    "LLM batch: M=%d/%d Q=%d/%d F=%d/%d in %.1fs",
    m_results, len(m_batch),       # M parsed / queued
    q_results, len(q_batch),       # Q parsed / queued
    fingpt_metrics["parsed"], len(f_batch),  # F parsed / queued
    elapsed,
)
```

Examples in the wild after the fix:

| log line | meaning |
|---|---|
| `M=4/4 Q=4/4 F=6/6 in 83.2s` | warmup, all phases fully successful |
| `M=0/0 Q=0/0 F=6/6 in 10.6s` | fingpt-only cycle, 6 stashed cleanly |
| `M=0/0 Q=0/0 F=0/6 in 10.4s` | fingpt-only cycle, **silent failure** — easy to spot |
| `M=4/4 Q=0/0 F=0/0 in 40.7s` | ministral-only cycle |

### Change 3 — failure-mode warnings inside `_flush_fingpt_phase`

Currently one bare `except` swallows all errors. After:

- Before the bare `except`, **inside** the success path:
  - If `received == 0 and queries > 0`: `WARN "fingpt: server returned None for all %d prompts"` (server connectivity / model swap failed)
  - Else if `parsed < received and parsed < received * 0.5`: `WARN "fingpt: parser returned None for %d/%d completions (>50%%)"` (parser regression — see `project_fingpt_parser_defaulting_neutral` memory)
- The bare `except Exception` stays as a final safety net but logs the **exception class name + repr** (not just `exc_info`) so a one-line scan of loop_out tells you what blew up:
  ```python
  logger.warning("LLM batch fingpt phase failed: %s", repr(e), exc_info=True)
  ```

### Change 4 — tests

New tests in `tests/test_llm_batch.py`:

1. `test_flush_fingpt_phase_returns_metrics_on_success` — happy path; `parsed == queries`, `exception is None`.
2. `test_flush_fingpt_phase_metrics_on_all_none_response` — server returns all `None`; metrics show `received=0, parsed=0`; `caplog` contains `"server returned None for all"`.
3. `test_flush_fingpt_phase_metrics_on_parser_failure` — server returns text but parser returns None for all; `received=N, parsed=0`; warning about parser.
4. `test_flush_fingpt_phase_metrics_on_exception` — `query_llama_server_batch` raises; metrics dict has `exception` set; warning logs the exception repr.
5. `test_flush_llm_batch_log_includes_fingpt_metrics` — drive `flush_llm_batch` end-to-end with a stubbed fingpt phase that returns known metrics, capture the summary log via `caplog`, assert the new format.

## What this PR does NOT do

- Does not change rotation logic. `_LLM_ROTATION` is correct.
- Does not change live trading behavior at all (no signal weights, no thresholds, no order paths touched).
- Does not add a persistent `data/fingpt_health.json` or contract-alert hookup. That's worth doing as a follow-up — if `parsed/queries < 0.5` for K consecutive cycles, the contract dispatcher should fire — but it's a bigger scope.
- Does not address the Qwen3 `/v1/chat/completions` thinking-mode envelope. The project doesn't use that endpoint anywhere (uses `/completion`), so it's a non-issue here.

## Risk

- Low. Cosmetic logging change + observability metrics. Functional path through fingpt phase is unchanged.
- The previous return value of `_flush_fingpt_phase` was implicit `None`; the only caller (`flush_llm_batch`) ignored it. New caller code now reads the dict — but since the function always returns a dict on every path, no None guard is required. (Tests cover this explicitly.)
- No live-config changes. No restart of loops required for the fix to take effect — but next loop restart will pick up the new log line, so verifying in production is one `schtasks /run /tn PF-DataLoop` away.

## Execution order

| Batch | Files | Tests |
|---|---|---|
| 1 | `portfolio/llm_batch.py` (refactor `_flush_fingpt_phase` to return metrics, update `flush_llm_batch` summary log) | `tests/test_llm_batch.py` (5 new) |
| 2 | (only if Codex finds something) | — |

Single batch expected. Final verify: `pytest tests/ -n auto`.

## Verification after merge

After merging + restarting the loop, tail `data/loop_out.txt`:

```bash
tail -F data/loop_out.txt | grep "LLM batch: M="
```

Within 5 minutes you should see at least one of M-only / Q-only / F-only cycles
(rotation cycles every ~3-5 minutes depending on cache hit rate). Each cycle's
`F=k/n` should reflect ground truth — if `F=0/n` shows up, that's a real
silent failure and the `WARN fingpt:` lines above will say which kind.
