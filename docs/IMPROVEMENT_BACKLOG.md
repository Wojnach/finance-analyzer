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
**Auto-scheduled:** Yes (2026-05-19). Entry
`LLM-CRYPTOTRADER-72H` in `data/pending_pickups.json` due
2026-05-21T08:00 CET. `PF-PendingPickups` Windows task runs
`scripts/process_pending_pickups.py` daily 08:00; on/after the due
date it dispatches `scripts.pickups.llm_cryptotrader_72h`, applies
the decision tree below, writes the verdict to
`docs/SESSION_PROGRESS.md`, sends Telegram, and surfaces on the
dashboard at More → Pickups (`/api/pickups`).

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

---

## ARCH-17 — main.py re-exports 100+ symbols at module level

**Discovered:** 2026-05-21 auto-session.
**Scope:** L — refactor touches every import site in the codebase.

### What

`portfolio/main.py` (1532 lines) imports and re-exports 100+ symbols
from submodules. Downstream code does `from portfolio.main import X`
instead of importing from the owning module directly. This creates a
single-file dependency bottleneck, slows import time, and makes the
module boundary unclear.

### Acceptance criteria

Move consumers to import from the owning module. `main.py` should only
import what it uses for loop orchestration.

### Why deferred

High-risk refactor across dozens of files. No functional impact — purely
structural. Needs a dedicated session with full test coverage verification.

---

## ARCH-18 — metals_loop.py is a 7,880-line monolith

**Discovered:** 2026-05-21 auto-session (previously noted informally).
**Scope:** XL — split into 5-8 focused modules.

### What

`data/metals_loop.py` (7,880 lines) contains market data collection,
signal computation, warrant selection, order execution, position
management, exit optimization, and Telegram reporting in a single file.
Functions are well-separated internally but the file is too large for
effective review, testing, or parallel development.

### Suggested split

* `metals_data.py` — price feeds, orderbook, cross-asset
* `metals_signals.py` — signal computation, voting
* `metals_warrants.py` — warrant selection, grid logic
* `metals_execution.py` — order placement, position tracking
* `metals_reporting.py` — Telegram, logging, journal

### Why deferred

Working code that runs 24/7 in production. A monolith split has high
regression risk and needs careful integration testing. No functional bug.

---

## ARCH-19 — No CI/CD pipeline

**Discovered:** 2026-05-21 auto-session.
**Scope:** M — GitHub Actions workflow + pre-push hook.

### What

The repo has 430 test files and 5,994+ tests but no automated CI.
Tests run locally via `pytest -n auto` (~5.5 min). Pre-commit hooks
exist but there's no pre-push gate or PR check. Regressions are caught
manually or by the auto-improve sessions.

### Acceptance criteria

GitHub Actions workflow that runs `pytest -n auto` on push/PR to main.
Optional: lint pass, type-check pass.

### Why deferred

Needs GitHub repo admin access and decisions about runner environment
(Windows-specific tests, GPU models, external API mocking). Not a
code-level fix.

---

## SIGNAL-1 — Close walk-forward weight loop

**Discovered:** 2026-05-26 after-hours research.
**Prior triage:** `data/daily_research_quant.json` (2026-05-26); arxiv 2602.00080.
**Scope:** M — 2 days estimated.

### What

`train_signal_weights.py` and `signal_weight_optimizer.py` exist and
produce trained weights, but those weights never flow into
`_weighted_consensus`. The MWU path was removed as dead code.
All infrastructure exists — just needs connection.

### Acceptance criteria

Trained weights from `signal_weight_optimizer.py` load at
`_weighted_consensus` call time. Walk-forward validation shows no
accuracy regression on 30d holdout.

### Why deferred

Medium-effort integration touching the critical consensus path.
Needs careful A/B validation to avoid regression.

---

## SIGNAL-2 — Rolling per-horizon IC weighting

**Discovered:** 2026-05-26 after-hours research.
**Prior triage:** `data/daily_research_quant.json`; `memory/quant_research_priorities.md`;
arxiv 2509.01393.
**Scope:** M — 3 days estimated.

### What

Replace accuracy-only signal weighting with EWMA Spearman IC per
signal/ticker/horizon. `ic_computation.py` exists but only supports
single-horizon. Per-ticker accuracy is near coin-flip (49-53%),
but IC captures magnitude prediction that accuracy misses.

### Acceptance criteria

`ic_computation.py` supports per-horizon IC. `_weighted_consensus`
optionally uses IC-based weights when sample count >= 100.

### Why deferred

Needs `ic_computation.py` per-horizon extension first. Research
plan ready in `memory/quant_research_priorities.md`.

---

## SIGNAL-3 — Fix dynamic correlation groups (agreement rate)

**Discovered:** 2026-05-26 (known bug since Apr 2026).
**Prior triage:** `memory/dynamic_corr_bug.md`.
**Scope:** M — 2 days estimated.

### What

Dynamic correlation groups use Pearson on BUY/SELL/HOLD encoded as
numeric. HOLD majority (>80% of votes) dilutes Pearson to ~0 for all
pairs. Fix: use agreement rate (% of cycles where both signals emit
the same directional vote, excluding HOLD-HOLD pairs).

### Acceptance criteria

Correlation groups computed with agreement rate. At least one
non-trivial cluster identified in production data.

### Why deferred

Known bug per `memory/dynamic_corr_bug.md`. Needs careful validation
since correlation groups feed into weight caps.

---

## SIGNAL-4 — Extract gs_ratio_velocity as standalone signal

**Discovered:** 2026-05-26 after-hours research.
**Scope:** S — 1 day estimated.

### What

Gold/silver ratio velocity is already computed inside
`metals_cross_assets.py` but buried in the composite signal. XAG
consensus accuracy is 49.6% vs metals 3h at 67.3% -- extracting
the strongest sub-feature as a standalone shadow signal could
surface the edge more cleanly.

### Acceptance criteria

`portfolio/signals/gs_ratio_velocity.py` deployed as shadow signal.
Accumulates >= 50 directional predictions within 14d.

### Why deferred

Easy but low urgency. SIGNAL-1 (walk-forward weights) has higher
expected impact.

---

## ARCH-20 — Signal schema validation missing

**Discovered:** 2026-05-21 auto-session.
**Scope:** M — schema definition + validation in signal_engine.py.

### What

Signal modules return dicts with inconsistent `sub_signal` key
structures. Some use flat keys (`rsi_14`), others use nested dicts
(`{momentum: {stoch: ...}}`), and some omit `sub_signals` entirely.
`signal_engine.py` tolerates all variants via `.get()` with defaults,
but there's no schema enforcement. A signal module can return malformed
data that silently degrades consensus quality.

### Acceptance criteria

TypedDict or dataclass for signal return values. Validation at the
`_compute_single_signal` boundary. Malformed returns logged and
force-HOLD (same as current exception path, but explicit).

### Why deferred

63 signal modules would need return-type updates. Risk of breaking
working signals during migration. Better done incrementally — validate
new signals immediately, migrate existing ones in batches.

---

## ~~BUG-A — Escalation gate ThreadPoolExecutor leak~~ RESOLVED

**Discovered:** 2026-05-27 auto-session.
**Resolved:** 2026-05-27 auto-session (commit e3657a7f).

`portfolio/escalation_gate.py` created a new `ThreadPoolExecutor(max_workers=1)`
on every `should_escalate()` call. On timeout (>10s) the hung thread lingered.
Fixed: module-level singleton executor. Same fail-open semantics, no thread leak.

---

## ~~BUG-B — Silent exception swallowing (12 sites)~~ RESOLVED

**Discovered:** 2026-05-27 auto-session.
**Resolved:** 2026-05-27 auto-session (commits e3657a7f, b7d82290).

12 `except Exception: pass` blocks across portfolio/ were hiding errors without
logging. Added `logger.debug(..., exc_info=True)` to each while preserving the
fail-safe control flow. Files: trigger.py, btc_etf_flow.py, crypto_precompute.py
(3 sites), loop_contract.py, signal_engine.py, grid_fisher.py,
gold_overnight_bias.py, intraday_seasonality.py.

---

## ~~BUG-C — test_consensus.py accuracy-cache drift~~ RESOLVED

**Discovered:** 2026-05-27 auto-session.
**Resolved:** 2026-05-27 auto-session (commit b06a2b82).

`test_consensus.py` signal count assertions were stale after crypto_evrp
disable (commit af2b5336, 2026-05-26). Updated MSTR 10→9, BTC-USD 15→14.

---

## ARCH-21 — Horizon accuracy collapse (3d/5d/10d → 1d)

**Discovered:** 2026-05-27 auto-session (tagged as TODO in signal_engine.py:4117).
**Scope:** M — needs per-horizon accuracy cache infrastructure.

### What

`signal_engine.py:4119` maps all horizons longer than 12h to "1d" accuracy
stats: `acc_horizon = horizon if horizon in ("3h", "4h", "12h") else "1d"`.
Signals that are good at 1d but bad at 5d (or vice versa) get the same
accuracy weight at all longer horizons.

### Acceptance criteria

Per-horizon accuracy cache entries for 3d, 5d, 10d horizons.
`_weighted_consensus` uses horizon-specific accuracy when available.

### Why deferred

Requires outcome_tracker changes (different backfill windows) and accuracy_cache
schema expansion. The current 1d proxy is conservative — it doesn't create
false positives, just misses horizon-specific edge.

---

## SIGNAL-5 — TrustTrade temporal consistency filter

**Discovered:** 2026-05-27 research session (ArXiv: TrustTrade selective consensus).
**Priority:** P1. **Effort:** 3 days.

Discard signals that flip direction within 2 consecutive checks (known noise
pattern on ETH/BTC Now-TF). Weight by inter-signal agreement + temporal
stability. Would have filtered all 5 ETH BUY→HOLD flips on 2026-05-27.

### Why deferred

Requires signal history tracking infrastructure. The in-memory persistence
filter (raised to 2 cycles for crypto on 2026-05-27) partially addresses
the same symptom. Full TrustTrade implementation needs cross-check temporal
alignment which is a larger architectural change.

---

## SIGNAL-6 — Fractional Kelly + ATR vol-targeting

**Discovered:** 2026-05-27 research session (ArXiv: fractional Kelly).
**Priority:** P2. **Effort:** 3 days.

75% growth of full Kelly at <50% max drawdown. Combine with ATR-based
volatility targeting for regime-adaptive position sizing. Currently using
fixed position sizing.

### Why deferred

Requires backtesting infrastructure for validation. Need walk-forward
testing before live deployment.

---

## SIGNAL-7 — Adaptive ATR trailing stops (regime-aware)

**Discovered:** 2026-05-27 research session.
**Priority:** P3. **Effort:** 2 days.

1.5x ATR in low-vol regimes, 3.0x ATR in high-vol. 45-65% drawdown
reduction vs fixed stops in backtests. System currently uses fixed ATR
multipliers.

### Why deferred

Requires regime classification integration with stop-loss management.
Current fixed stops work acceptably.

---

## ~~FGL-T0 — Alert fatigue: autonomous failure stubs~~ RESOLVED

**Discovered:** 2026-05-29 FGL review. **Resolved:** 2026-05-30 auto-session.

autonomous.py exception before journal write → no stub → loop_contract
fires false CRITICAL. Fixed: failure journal stub + loop_contract
recognizes `autonomous_*` statuses. Stops 22+ false CRITICALs/week.

---

## ~~FGL-P0-1 — Warrant SELL of non-existent position~~ RESOLVED

**Discovered:** 2026-05-29 FGL review. **Resolved:** 2026-05-30 auto-session.

SELL with config_key not in holdings was silently ignored. Transaction
was recorded but holdings unchanged → ledger mismatch. Fixed: refuse
SELL of missing position; clamp over-sell to held units.

---

## ~~FGL-P0-2 — Avanza orderId="?" placeholder~~ RESOLVED

**Discovered:** 2026-05-29 FGL review. **Resolved:** 2026-05-30 auto-session.

Missing orderId on SUCCESS response defaulted to "?", saved as real ID.
Downstream cancel/track operations fail. Fixed: reject and mark error
with Telegram alert.

---

## ~~FGL-P0-3 — Avanza orderbook_id unvalidated~~ RESOLVED

**Discovered:** 2026-05-29 FGL review. **Resolved:** 2026-05-30 auto-session.

_place_order accepted None/""/non-numeric orderbook_id. Fixed: validate
non-empty + numeric before POST.

---

## ~~FGL-P0-4 — Multi-agent claude_gate bypass~~ RESOLVED

**Discovered:** 2026-05-29 FGL review. **Resolved:** 2026-05-30 auto-session.

Specialist spawns bypassed claude_gate's process tree killing and
invocation journaling. Fixed: use _kill_process_tree on timeout; log
specialist invocations to invocations.jsonl.

---

## ~~FGL-TB — Choppiness tie-breaker leak~~ RESOLVED

**Discovered:** 2026-05-29 FGL review. **Resolved:** 2026-05-30 auto-session.

choppiness_regime_gate forced direction@0.35 when votes were HOLD.
Injects signal where sub-signals disagree. Fixed: removed the override.
Added REGIME_GATE_ONLY_SIGNALS engine mechanism for future use.

---

## ~~FGL-TF — loop_health status hardcode~~ RESOLVED

**Discovered:** 2026-05-29 FGL review. **Resolved:** 2026-05-30 auto-session.

write_heartbeat() hardcoded "status":"ok" ignoring ok param. Fixed.

---

## ~~FGL-TA — http_retry fatal-vs-transient~~ RESOLVED

**Discovered:** 2026-05-29 FGL review. **Resolved:** 2026-05-30 auto-session.

401/403/404 retried like 503. Fixed: FATAL_STATUS set, immediate return.

---

## QUANT-1 — ETF flow momentum signal (BTC)

**Discovered:** 2026-05-30 after-hours research. **Status:** Proposed.

Track BTC spot ETF net flows as a new signal module. 5-day rolling sum:
>$500M weekly outflow = SELL, >$500M inflow = BUY. Currently the strongest
single predictor for BTC price direction (6-day outflow streak = $2.54B
correlates with $73K weakness). Data from CoinGlass or similar API.

**Effort:** 2 days. **Files:** `portfolio/signals/etf_flow.py`, `signal_registry.py`.
**Priority:** P2.

---

## QUANT-2 — VIX-Rank continuous volatility scaling

**Discovered:** 2026-05-30 quant research (ArXiv 2508.16598). **Status:** Proposed.

Replace discrete regime multipliers (ranging=0.75, high-vol=0.80) with
continuous `conf *= (1 - vol_rank)` where `vol_rank` is the percentile of
current realized vol over a lookback window. GVZ already fetched for metals
via `metals_cross_assets.py`. For crypto, use realized vol percentile.

**Effort:** 1 day. **Files:** `portfolio/signal_engine.py` lines 3044-3050.
**Priority:** P2. **Risk:** Affects all trade decisions — needs thorough backtesting.

---

## QUANT-3 — Friction-adjusted Kelly sizing for gold warrants

**Discovered:** 2026-05-30 quant research (ArXiv 2511.08571). **Status:** Proposed.

Adapt the friction-adjusted fractional Kelly formula (lambda=0.40, Sharpe 2.88
walk-forward on gold futures) for XAU warrant position sizing. Formula:
`f* = (-3γn^1.5 + sqrt(9γ²n³ + 16σ²(μ-nk))) / (4σ²)`.
3-layer sizing: vol target, regime confidence, friction-aware Kelly.

**Effort:** 2 days. **Files:** `portfolio/risk_management.py`, `portfolio/golddigger/`.
**Priority:** P2.

---

## QUANT-4 — Gold-silver pair trade with Kalman filter

**Discovered:** 2026-05-30 quant research. **Status:** Proposed.

Add cointegration-based dynamic hedge ratio (Kalman filter) to the existing
gold/silver ratio velocity signal in `metals_cross_asset`. ML regime filter
(Gradient Boosting) trained on volatility, macro, and sentiment features
distinguishes stable vs unstable spread conditions. Could coordinate XAU/XAG
warrant entries.

**Effort:** 3 days. **Files:** `portfolio/signals/metals_cross_asset.py`.
**Priority:** P2.

---

## QUANT-5 — Berry Phase Rate geometric regime detector

**Discovered:** 2026-05-30 quant research (ArXiv 2605.17117). **Status:** Proposed.

Maps returns into complex Hilbert space, extracts Berry Phase Rate as regime
indicator. Orthogonal to all 6 existing regime signals (|rho|=0.22). Simple
overlay rule: exit to cash when z-score >2.0 cut max drawdown from -55% to -27%.
Novel but requires eigendecomposition per timestep.

**Effort:** 5 days. **Files:** new signal module.
**Priority:** P3.

---

## QUANT-6 — Walk-forward validation gate for IC weighting

**Discovered:** 2026-05-30 quant research (ArXiv 2601.05716). **Status:** Required.

Cautionary finding: adaptive in-sample signal weights FAIL out-of-sample.
Any IC-based weighting implementation (per `memory/quant_research_priorities.md`)
MUST use strict walk-forward validation (train on N days, test on next M days,
roll forward). Current accuracy_stats.py uses all available data.

**Effort:** 2 days. **Files:** `portfolio/accuracy_stats.py`, `portfolio/signal_engine.py`.
**Priority:** P1 (prerequisite for IC weighting).

---

## INFRA-1 — LLM confidence calibration v2

**Discovered:** 2026-05-30. **Status:** Proposed.

The current calibration map (commit 82bac99a) was reverted because it was
fitted on only 304 Qwen3 / 99 Ministral samples with incomplete outcomes.
Re-implement with: (a) minimum 500 samples per bin, (b) out-of-sample
validation on holdout set, (c) outcome backfill complete before fitting,
(d) per-action resolution (BUY/SELL/HOLD calibrated separately).

**Effort:** 2 days. **Files:** `scripts/fit_llm_confidence_calibration.py`,
`portfolio/llm_confidence_calibration.py`, `data/llm_confidence_calibration.json`.
**Priority:** P2.

---

## SIGNAL-WEIGHT-1 — Exponential-decay signal weighting

**Discovered:** 2026-06-01 after-hours research session.
**Source:** arxiv 1802.07543 (Exponential Weights online learning).

Replace static accuracy-based weighting with exponential-decay:
`weight_i = exp(-eta * cumulative_loss_i)` where loss = `(1 - recent_accuracy)`.
Signals that degrade auto-downweight within 1-2 weeks without manual
DISABLED_SIGNALS entries. Floor at 0.05x to prevent permanent kill.

**Why deferred:** Needs careful backtesting on signal_log.db to validate
that exp-decay doesn't over-react to short transient accuracy dips.
**Effort:** 2 days. **Files:** `portfolio/signal_engine.py` (_weighted_consensus).
**Priority:** P1.

---

## REGIME-SOFT-1 — Soft regime assignments (sigmoid thresholds)

**Discovered:** 2026-06-01 after-hours research (arxiv 2510.03236).

Replace hard regime thresholds with sigmoid smoothing:
`p_trending = sigmoid((ADX - 25) / 5)`. Regime multipliers become continuous
instead of discrete (0.75 + 0.25 * p_trending). Eliminates flip-flopping at
boundaries that causes near-coinflip accuracy in regime signals.

**Why deferred:** Touches multiple signal modules + signal_engine regime logic.
Need coordinated change across 6 regime signals.
**Effort:** 3 days. **Files:** `portfolio/signals/adx_regime_switch.py`,
`bocpd_regime_switch.py`, `choppiness_regime_gate.py`, `drift_regime_gate.py`,
`amihud_illiquidity_regime.py`, `vol_ratio_regime.py`, `signal_engine.py`.
**Priority:** P1.

---

## IC-ROLLING-1 — Rolling Spearman IC recomputation (14d window)

**Discovered:** 2026-06-01 after-hours research.

Increase IC recomputation to daily with 14d rolling Spearman rank IC.
IC is a leading indicator of accuracy decay — catches degradation 1-2 weeks
before accuracy stats. Track IC trend: 7+ consecutive decline days → 0.5x
dampening.

**Why deferred:** Needs ic_computation.py refactoring and careful TTL tuning.
**Effort:** 2 days. **Files:** `portfolio/ic_computation.py`, `signal_engine.py`.
**Priority:** P2.

---

## RISK-ATR-1 — Regime-conditional ATR stop multipliers

**Discovered:** 2026-06-01 (2026 quant consensus).

Replace static ATR multipliers with regime-dependent: trending 2.0x, ranging
1.2x, high-vol 1.5x. For silver warrants: `max(3% of bid, regime_mult * ATR)`.
Add trailing stop activation at +1.5x ATR in profit.

**Effort:** 2 days. **Files:** `portfolio/risk_management.py`.
**Priority:** P2.

---

## SIZING-VOL-1 — Inverse-volatility position sizing

**Discovered:** 2026-06-01 (Concretum Group research).

Target_exposure = (target_vol / realized_vol) * base_position. Use 20d rolling
realized vol. Target portfolio 15% annualized. Naturally sizes down BTC (~60% vol)
and up XAU (~15% vol). Sharpe improved 0.99→1.54 in backtests.

**Effort:** 2 days. **Files:** `portfolio/risk_management.py`, `portfolio_mgr.py`.
**Priority:** P2.
