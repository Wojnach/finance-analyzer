# Improvement Backlog

Standing work items surfaced during sessions but intentionally deferred.
Each entry: title, reason-for-deferral, scope estimate, and any pointers
to prior triage docs.

---

## ~~TEST-HYGIENE-1 — xdist module-state leak audit~~ RESOLVED

**Discovered:** 2026-04-17. **Resolved:** 2026-04-19 auto-session.

Global autouse fixture in `tests/conftest.py` (`_reset_module_state`)
resets all HIGH-risk module state (agent_invocation, signal_engine,
shared_state) before/after every test. Reset helpers in
`tests/_state_reset.py` also cover MEDIUM/LOW-risk modules (forecast,
logging_config, api_utils, trigger).

Result: 5+ random xdist flakes eliminated per run. Remaining 24
failures are all pre-existing infrastructure dependencies (freqtrade,
Ministral model).

---

## ~~TEST-HYGIENE-2 — `tests/test_llama_server_job_object.py` (untracked)~~ RESOLVED

**Discovered:** 2026-04-17. **Resolved:** 2026-05-01 cleanup session.
**Prior triage:** `docs/plans/2026-04-17-pre-existing-tests.md`.

### Resolution
The aspirational test file `tests/test_llama_server_job_object.py` is
no longer present in the working tree of `main` (was never tracked in
git history and has been cleaned out at some point between 2026-04-17
and 2026-05-01). Stale `__pycache__` artifacts (`.pyc` files) from
prior collection runs are also being cleaned up — pytest does not
collect from `.pyc` files but they are confusing residue.

Verified state on 2026-05-01:
- `git ls-files tests/ | grep llama_server` → only the legitimate
  `tests/test_llama_server.py` (model management + query
  serialization), not the job-object file.
- `pytest tests/ -k 'llama_server' --collect-only` from a fresh
  worktree of `main` collects 23 tests cleanly with 0 errors.
- Production code partially implements the feature anyway:
  `popen_in_job` and `close_job` now exist in
  `portfolio/subprocess_utils.py` (used by the metals subsystem).
  The remaining symbols (`_local_job_handle`, `_sweep_done`,
  `kill_orphaned_llama_server`, `_kill_orphaned_by_name`,
  `atexit.register(stop_all_servers)`) were never landed because
  `llama_server.py`'s lifecycle is solved differently — via PID
  files, file locks, and an external orphan reaper
  (`kill_orphaned_llama` in `subprocess_utils.py`).

### Why deletion was correct
The features the test file enumerated were aspirational. The
production solution chose a different shape (PID file + orphan
reaper). Implementing the test's vision would require ~300+ LOC of
restructuring `llama_server.py` for a feature that was never
prioritized. The lower-risk path (delete the file, accept the actual
production design) was the right call.

---

## Pattern for adding new backlog items

Append a new section with:
- Short ID (`TEST-HYGIENE-N`, `FEATURE-N`, `RISK-N`, etc.)
- Title
- Discovery session / date
- Prior triage doc (if any)
- Scope estimate
- What the problem actually is
- What the acceptance criteria look like
- Why it was deferred this time

---

## LLM-CUSTOM-LORA-RETIRED — `Q:\models\custom-trading-lora.gguf` will not be wired

**Discovered:** 2026-05-18.
**Prior triage:** `docs/LLM_FOLLOWUPS_20260518.md` §2; Feb 22 commit
`53a15df8`; `data/lora_backtest_results.json` (2026-02-12).
**Scope:** XS — no action; documentation only.

### What

Custom-trained LoRA at `Q:\models\custom-trading-lora.gguf` produced by
`training/lora/pipeline.py` in Feb 2026. Considered as a candidate
fourth LLM voter alongside cryptotrader_lm during the 2026-05-18 shadow
enrollment review.

### Why retired

Feb 2026 A/B (`data/lora_backtest_results.json`, 260 prompts) measured
Custom LoRA at BTC 51.5% / ETH 30.8% with BUY recall = 0% on BTC. ETH
agreement vs original 17.7% (model is doing its own thing). Commit
`53a15df8` documented "Custom LoRA already disabled at 20.9%" in a
later evaluation window. The GGUF has been dormant since Feb 22 2026.

User explicit decision 2026-05-18: "we've tried and it's not good."

### Why not just delete

Keep the GGUF on disk for archival — it's our own training output and
the source `training/lora/pipeline.py` references it. Future re-runs
of the pipeline should produce `custom-trading-lora-v2.gguf` rather
than overwrite the historical artefact.

### Re-open condition

A fresh training run via `training/lora/pipeline.py` against post-2026-Q2
data producing a NEW GGUF. Re-evaluating the existing Feb GGUF is not a
re-open trigger — its weights aren't going to get better by sitting on
disk.

---

## LLM-CRYPTOTRADER-72H — Verify cryptotrader_lm v2 LoRA on real outcomes

**Discovered:** 2026-05-18.
**Prior triage:** `docs/LLM_FOLLOWUPS_20260518.md`; merge `07702358`
(shadow-gate-lora-20260518); `scripts/probe_cryptotrader_lm.py`.
**Scope:** S — passive accumulation + one review.

### What

The v1 cryptotrader_lm GGUF (Feb 2026) emitted empty completions on
every production prompt. v2 was regenerated 2026-05-18 from the
original HF safetensors via current `convert_lora_to_gguf.py` against
`Q:\models\ministral-8b-hf`. Live probe shows real BUY/SELL/HOLD
output with mixed decisions and conf 0.6-0.85. Production wiring is
unchanged — same registry entry, same `_LLM_SIGNALS` membership, same
shadow status.

Accumulate ≥72h of directional predictions (`conf>0 AND chosen in
{BUY,SELL}`) joined with outcome backfill. Then run
`scripts/review_shadow_signals.py --promote --dry-run` and inspect
the matched count + accuracy.

### Acceptance criteria

Decision tree by 2026-05-21:
- `n_directional` ≥ 50 AND `accuracy` ≥ 0.60 → promote candidate; manual review of confusion matrix; flip status if confusion looks reasonable.
- `n_directional` ≥ 50 AND `accuracy` < 0.55 → retire (consistent with the Feb measurement on the broken v1, suggesting the base CryptoTrader-LM training was not generalisable).
- `n_directional` < 50 → extend window to 7d; investigate why so few directional emissions (Plex-VRAM gate? non-BTC/ETH dispatch path?).

### Why deferred

Need data, not code, to answer. Re-check in 72h.

---

## LLM-QWEN3-HOLD-AB — Qwen3 prompt A/B for HOLD-bias

**Discovered:** 2026-05-18.
**Prior triage:** `docs/LLM_FOLLOWUPS_20260518.md` §3; TODO comment
inline at `portfolio/qwen3_trader.py:_build_prompt`.
**Scope:** M — needs offline harness + decision.

### What

`portfolio/qwen3_trader.py:_build_prompt` system message contains two
reinforcements pushing toward HOLD: (a) "A confident HOLD is better
than a low-confidence BUY/SELL" and (b) "<40 = default to HOLD" in the
confidence guide. Production data shows qwen3 emitting HOLD on >95%
of cycles. `accuracy_cache.json` reports qwen3 at 60% on 3809 1d
samples — driven primarily by SELL precision (73.7%) since BUY
precision is only 33.1%.

### Acceptance criteria

1. Build offline harness (extend `scripts/lora_backtest.py` or write a
   sibling) that scores qwen3 with two prompt variants:
   - `_build_prompt_conservative_v1` (current)
   - `_build_prompt_neutral_v2` (remove sentence (a) only — keep the
     confidence guide so low-conf still falls to HOLD naturally)
2. Run against 14d+ of labelled candles.
3. If v2 raises BUY/SELL recall without hurting precision by >2pp,
   ship the v2 prompt as a feature-flagged switch defaulting to v1.
4. After 7d of v2-shadow accumulation in production, promote v2 if
   the offline result holds up.

### Why deferred

Speculative behaviour change on a currently-passing voter (60% on
3809 samples). The `feedback_weight_calibration_warnings` memory
explicitly warns against speculative flips of working signals.

---

## LLM-FINANCE-LLAMA-ABSTAIN — 73% production abstain rate

**Discovered:** 2026-05-18.
**Prior triage:** `docs/LLM_FOLLOWUPS_20260518.md` Defer list; merge
`07702358`.
**Scope:** M — diagnostic + small code fix.

### What

`finance_llama` shadow signal emits abstain rows on 73% of cycles.
Production sample (post-`07702358` filter): 692 conf=0 vs 304 real
directional rows out of 996 total. Without the recent gate fix, those
abstain rows polluted the accuracy denominator and trivially passed
promotion at HOLD-bias-against-outcome-backfill matching.

Two likely causes for the high abstain rate:
1. Plex-VRAM guard (`model_load_safe()`) triggering when Plex is
   transcoding on the same host — abstain via `plex_vram_tight`.
2. JSON parse failures in `_parse_response` for the Llama-completion
   prompt format → abstain via `inference_error` or
   `prompt_build_failed`.

### Acceptance criteria

Aggregate the indicator `reason` field from each finance_llama log
row across 7d. Identify dominant cause (Plex-VRAM vs parse failure
vs other). Fix the dominant cause without making the rate worse on
the other.

### Why deferred

Diagnostic work that needs the post-fix gate to have collected enough
clean rows. Re-check after 7d.

---

## LLM-META-TRADER-WIRE — Wire meta_trader Qwen2-36L (Item 3 of shadow plan)

**Discovered:** 2026-05-15 (original shadow-enrollment plan).
**Prior triage:** `/root/.claude/plans/no-we-don-t-these-glowing-ullman.md`
Step 3; `portfolio/signals/meta_trader.py` (scaffold only).
**Scope:** L — multi-session.

### What

`portfolio/signals/meta_trader.py` is currently a scaffold returning
`HOLD/conf=0` on every call. Designed to consume other LLM voter
outputs from the same cycle as features (meta-model role). Model at
`Q:\models\custom-meta-trader\` (Qwen2 36-layer unsloth safetensors,
5.8GB, 32K context).

Requires:
1. New `_MODEL_CONFIGS` entry for Qwen2-36L (GGUF conversion of the
   safetensors first).
2. Dispatch-order coupling: meta_trader must run AFTER
   ministral/qwen3/finance_llama in the signal_engine loop so their
   votes are populated in the prompt context.
3. Cycle-time budget: Qwen2-36L inference is the most expensive in
   our stack. Currently registered with `cycle_modulo=5` so it runs
   every 5 minutes when wired.

### Acceptance criteria

`_FEATURE_AVAILABLE=True` in `portfolio/signals/meta_trader.py` with
real inference path. Shadow accumulates ≥200 directional predictions
within 30d. Cycle time impact stays <30s incremental p95.

### Why deferred

L-effort, no quick win, and the LLM voter slate is healthier with
cryptotrader_lm v2 verification first.

---

## LLM-ACCURACY-SOURCE-UNIFY — Reconcile accuracy_cache.json vs llm_probability_log

**Discovered:** 2026-05-18.
**Prior triage:** `docs/LLM_FOLLOWUPS_20260518.md` Defer list.
**Scope:** M — investigation + one cross-source contract.

### What

Two parallel accuracy accounting systems track LLM signals:

1. `data/accuracy_cache.json` (signal_log.jsonl-derived): ministral
   58.1% on 1d/6284 samples; qwen3 60.0% on 1d/3809 samples.
2. `data/llm_probability_log.jsonl` (per-vote, joined with outcomes):
   ministral directional 20% on 65 rows; qwen3 directional 41% on 27
   rows (recent 30d window).

The 30pp gap is best explained by sample-population mismatch (cache
has long history; log started 2026-04 and dropped HOLD votes from
denominator). But the dashboard, the Layer 2 prompt context, and the
auto-promotion cron should not be reading different accuracy numbers
for the same signal.

### Acceptance criteria

Single source-of-truth contract: pick one accuracy column for each
consumer (dashboard tile, Layer 2 summary, promotion gate). Document
the choice in `docs/SYSTEM_OVERVIEW.md` and `dashboard/app.py`. Add a
daily cross-source consistency assert: if the two pipelines disagree
by >10pp for the same signal+horizon, log a `critical_errors.jsonl`
entry.

### Why deferred

Not blocking trades. The 2026-05-18 gate fix already keeps the
promotion path consistent within itself.

---

## LLM-BRIER-FULL-DIST — Optional Brier-over-full-distribution metric

**Discovered:** 2026-05-18.
**Prior triage:** `docs/LLM_FOLLOWUPS_20260518.md` Defer list; cavecrew
review of merge `07702358`.
**Scope:** XS — single dashboard field.

### What

`dashboard/app.py:_compute_llm_leaderboard` Brier denominator now uses
the directional set (matching accuracy denominator). That's correct
for accuracy parity but loses information about HOLD-confident
calibration. If we want both:

* Add a `brier_full_dist` field computed over the unfiltered
  per-signal row set.
* Keep the existing `brier` as "directional Brier".

### Acceptance criteria

New field on `/api/llm-leaderboard` payload populated. Dashboard
HTML/JS update optional — the field is consumable via curl regardless.

### Why deferred

Low impact. Wait until we have a concrete question that needs it.
