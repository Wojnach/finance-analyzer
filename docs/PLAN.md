# Plan — LLM Health & Shadow Logging Audit (2026-04-21)

Protocol: `/fgl` → `docs/GUIDELINES.md` (explore → plan → implement → verify → ship).
Previous PLAN (warrant-exit v2) shipped as commit `14568c32`.

## Scope

Investigate how every LLM in the loops is performing, fix the broken ones,
and add probability/calibration journaling so we can promote or retire shadow
models on evidence rather than vibes.

---

## Findings (evidence-based, 30-day window)

### 1. Kronos shadow model is effectively dead — CRITICAL
- Config: `forecast.kronos_enabled = "shadow"` (runs inference + logs, forced HOLD in composite vote).
- `forecast_health.jsonl` last 85 entries show mixed errors:
  CUDA OOM, `timeout_1s`, `empty_results`, `json_extract_failed`.
- **Subprocess success rate: 59.2%** (3250/5491) — far below the 90% bar the
  `local_llm_report` recommendations flag requires to promote.
- **Raw sub-signal distribution: 100% HOLD** (3668/3668) — Kronos has not
  emitted BUY or SELL a single time in the logged window.
- Even after 3668 attempts, only **6 rows passed the `raw` gating pass** —
  i.e. shadow evidence is statistically uninformative.
- `Q:/models/kronos_infer.py` exists (14.8 KB, last modified 2026-03-27).
  Needs a real probe run, not just log inspection.

### 2. Forecast logging stopped 10 days ago — CRITICAL
- Last production prediction per live Tier-1 ticker:
  - BTC-USD: 2026-04-17T22:03 (3d ago — likely a test fixture row)
  - ETH-USD: 2026-04-11T10:42 (**10d ago**)
  - MSTR:    2026-04-10T20:03 (**10d ago**)
  - XAG-USD: 2026-04-11T10:42 (**10d ago**)
  - XAU-USD: 2026-04-11T10:52 (**10d ago**)
- Main loop is alive: `health_state.json` uptime 258,333 s (~3 d), last
  heartbeat 2026-04-21T11:26.
- Chronos subprocess success rate 99.7 % historically, so the model itself
  is not the blocker.
- Hypothesis: silent early-return path in `portfolio/signals/forecast.py`
  (latched circuit breaker, dedup-cache misfire, or the `atomic_append_jsonl`
  call gated behind a condition that's now always false). Needs tracing.

### 3. Claude Fundamental cascade has ~40 % empty rows — HIGH
- `data/claude_fundamental_log.jsonl`: since 2026-04-07 about 30-50 % of
  rows carry `reasoning: ""`, `confidence: 0.0`, empty `sub_signals`.
- Affects all three tiers (Haiku 288 / Sonnet 60 / Opus 38 on 2026-04-08 alone).
- Not a recent regression — persistent for 2 + weeks.
- Unclear whether these are abstention placeholders (intended when the
  cascade decides not to run) or silent failures. Either way the headline
  30-day 1 d accuracy (61.6 %) may be polluted by empty rows counting as
  HOLD matches against sideways moves.
- 10 008 samples at 1 d — this is the biggest signal in the system, so
  the number must be trustworthy.

### 4. Ministral-8B is catastrophic on crypto/metals — MEDIUM
- BTC-USD: 13.4 % (67 samples @ 1 d) — worse than random inversion.
- ETH-USD: 15.9 % (63 samples).
- XAG-USD: 19.8 % (96 samples).
- XAU-USD: 43.2 % (81 samples) — below 47 % gate.
- MSTR: 69.0 % (239 samples) — strong.
- Ticker-gate is supposed to force-HOLD below threshold. Verify that the
  live signal pipeline is actually dropping votes for BTC/ETH/XAG/XAU
  (grep the per-ticker gate path) and lock it in with a test.

### 5. MSTR loop Phase B shadow log never wrote — MEDIUM
- `portfolio/mstr_loop/config.py` defines `SHADOW_LOG = "data/mstr_loop_shadow.jsonl"`.
- File doesn't exist on disk.
- Either the loop is in Phase A (live sizing, no shadow log) or it never
  started. Need to check the scheduled task + phase state.

### 6. No probability / calibration logging for LLM votes — HIGH (infra)
- Current accuracy is argmax-binary: "did BUY turn into a +move?"
- Never records the model's **predicted class probabilities** (e.g. LLM
  returned `{BUY: 0.62, HOLD: 0.28, SELL: 0.10}` — we keep only `"BUY"`).
- Without probabilities we can't compute Brier score, log-loss, or
  calibration curves. We can't detect "confidently wrong" vs "uncertain and
  wrong".
- Biggest structural gap: shadow-mode promotion decisions require
  calibration, not just accuracy.

### 7. FinGPT post-fix distribution still neutral-heavy — MEDIUM
- Parser fix shipped 2026-04-09 (commits `fde9cf8` + `28aa5d0`).
- Since 2026-04-10: 55.4 % neutral / 41.6 % positive / 3.0 % negative.
- Probe-sane but real headlines cluster neutral.
- Agreement with primary: 61.5 % (353/574).
- Not a bug, but the shadow hasn't hit the 60 % / 200-sample promotion
  gate — needs more time or richer comparison.

### 8. FinBERT shadow is a signal-value desert — LOW
- 86.1 % neutral output (1 155 samples since 2026-04-10).
- 87.9 % agreement with primary — agrees by defaulting neutral.
- Costs zero CPU (cached in-process post 2026-04-09). Keep it but
  document that it adds no independent signal.

### 9. Chronos sub-signal has no 3 h / 3 d coverage — INFO (answers user's Q)
- `_run_chronos(prices, horizons=(1, 24))` is hardcoded at
  `portfolio/signals/forecast.py:363`.
- `forecast_accuracy.py:320` only backfills `("1h", 1)` and `("24h", 24)`.
- The `forecast @ 3h/3d` accuracy figures measure the **composite forecast
  signal's verdict vs future return** — not the Chronos sub-signal.
- Not a bug; the model is simply never asked for those horizons. Adding
  them is only worth it if 1 h / 24 h accuracy were above the floor
  (currently 45 % / 52 % effective) — deferred to backlog.

---

## Subagent investigation — RESULTS (2026-04-21 13:45)

Four `Explore` subagents ran in parallel. Findings condensed:

### A1 — Forecast logging stopped 10 d ago ✅ ROOT CAUSE IDENTIFIED

**Not a logging bug. Not a circuit breaker. A config change.**

Commit `70603577` on 2026-04-12 23:46 added `"forecast"` to the
`DISABLED_SIGNALS` set at `portfolio/tickers.py:29-35`. The reason given
in the commit was "36-39% accuracy" (below the force-HOLD floor).

The `signal_engine.py` dispatcher skips any signal in `DISABLED_SIGNALS`
*before invocation* — so `compute_forecast_signal()` is never called,
`_run_chronos` / `_run_kronos` are never called, `atomic_append_jsonl`
is never called. Both `forecast_predictions.jsonl` and `forecast_health.jsonl`
go silent as a side-effect.

The 2026-04-17 rows in the logs are artifacts of test runs / manual invocations,
not production cycles.

**Fix** (Batch 2): one-line change. Either:
- Remove `"forecast"` from `DISABLED_SIGNALS` (accepts the accuracy gate will
  force-HOLD most calls, but at least we restore logging + health tracking).
- Or move it into `REGIME_GATED_SIGNALS` so it only runs at horizons / regimes
  where historical accuracy clears 47%.

Before flipping the switch we need a fresh look at whether 1h/24h Chronos
accuracy has recovered — last measurement was pre-disable.

### A2 — Claude Fundamental empty rows ✅ NOT A BUG (mostly)

**Root cause classification: intentional abstention from Haiku tier.**

The 2,349 confidence-0.0 rows in the last 30 days are all from the Haiku
tier. The Haiku prompt explicitly instructs the model to emit
`{action: "HOLD", confidence: 0.0}` when it has no strong fundamental
view on a ticker. Sonnet and Opus never produce empty rows.

**The 61.6% @ 1d accuracy number is real.** Verified by reconciling:
- 5,718 cascade refresh rows in 30 d (half abstentions).
- 10,008 per-ticker cycle samples — cached refresh values replayed every
  main-loop cycle, HOLD votes dropped from accuracy denominator at
  `portfolio/accuracy_stats.py:150`.
- BUY 2,217 / SELL 633 / HOLD 2,868 in cascade log; these produce
  6164 / 10008 = 61.6 % at 1 d after the cache-replay expansion.

So accuracy is clean — but:
- Haiku's **53 % abstention rate** means Haiku is the tier contributing
  most noise / least signal. Keep it for macro context, don't escalate.
- The log file is noisy (2,349 unused rows per 30 d). Worth suppressing
  the write, not the decision.
- The Haiku tier also has no error-surfacing path (`_refresh_tier`
  swallows empty API responses without a `record_critical_error` call).
  That is a latent bug — if the Claude CLI stops responding cleanly
  we'd see the same empty rows and never know.

**Fix** (Batch 3): suppress empty-row writes at
`portfolio/signals/claude_fundamental.py:642` *and* add explicit
error-surface at `_refresh_tier` for genuine empty-API-response cases.
Accuracy tracker needs no change.

### A3 — Kronos shadow ✅ VERDICT: RETIRE

**Not salvageable.** The 59 % subprocess success rate has two
structurally unfixable root causes:

1. **Custom KronosPredictor API + shared venv** → VRAM contention with
   Chronos / Ministral / Qwen3 that rotate through `gpu_gate`. No async
   queue, blocking subprocess design. Windows has no way to pre-reserve
   VRAM before `subprocess.Popen` forks, so collisions are unavoidable.
2. **100 % HOLD output in shadow mode** (3,668 / 3,668). Even when
   inference succeeds, `_KRONOS_SHADOW = True` forces HOLD at
   `forecast.py:811,820`. Only 6 predictions ever contributed a raw vote.
   The shadow has accumulated zero statistical signal.

Fix attempts already in the code (stdout-scrub `_extract_json_from_stdout`,
circuit breaker) hit diminishing returns. Retire is the right call.

**Fix** (Batch 4): delete `_run_kronos`, `_run_kronos_inner`, `_trip_kronos`,
`_kronos_circuit_open`, module-globals `_kronos_tripped_until` /
`_KRONOS_ENABLED` / `_KRONOS_SHADOW`. Remove the call block in `forecast()`.
Drop `kronos_1h` / `kronos_24h` from sub-signals. Update tests.
Net diff ~ −150 lines + test update. Frees the GPU gate for Chronos /
Ministral rotations.

### A4 — MSTR loop Phase B ✅ WORKING AS DESIGNED

**Not a bug.** The MSTR loop runs in Phase B (shadow mode, `PHASE = "shadow"`
default in `portfolio/mstr_loop/config.py:19`). `mstr_loop_shadow.jsonl`
doesn't exist simply because the loop is **outside its NASDAQ session
window** (15:30 – 22:00 CET) and exits early with
`outside_session_window` at `portfolio/mstr_loop/loop.py:83-87`.

Last cycle logged: 2026-04-18 20:55 UTC. Next session: today 15:30 CET.
The shadow file will appear on the first BUY / SELL / PARTIAL_SELL
decision within the session window. Nothing to fix.

**Action** (Batch 6, demoted): document this behavior in
`docs/SESSION_PROGRESS.md` so future-Claude doesn't repeat the
investigation. No code change.

---

## Implementation batches (revised post-investigation)

Worktree: `Q:/finance-analyzer-llm-health`, branch `fix/llm-health-20260421`
(already created).

After each batch: run touched-file tests, commit with conventional message,
update `docs/SESSION_PROGRESS.md`.

### Batch 2 — Forecast re-enable (TRIVIAL, do first)

**Root cause:** `"forecast"` in `DISABLED_SIGNALS` since 2026-04-12.
**File (1):**
- `portfolio/tickers.py` — remove `"forecast"` from `DISABLED_SIGNALS`
  (or move to `REGIME_GATED_SIGNALS` if current accuracy data still
  shows sub-47 %). Verify against live `accuracy_cache.json` first.

**Plus regression protection:**
- `portfolio/health.py` (or equivalent stale-file checker) — add a
  `forecast_predictions_stale_hours` gauge. Log a `critical_error` if
  no new row in >6 h during market hours. Prevents a future silent
  disable from going unnoticed for 10 days.
- `tests/test_signals_forecast.py` — add a test that asserts forecast
  is NOT in `DISABLED_SIGNALS` unless explicitly intended (scan the set
  and compare to a canonical "expected disabled" allowlist).

**Risk:** re-enabling a signal that was rightly disabled for poor
accuracy. Mitigation: accuracy gate (47 %) still fires; the signal will
just return HOLD most of the time, same as a disabled signal, but now
we see the health + prediction data.

### Batch 4 — Kronos retire (CONTAINED, ~150-line delete)

**Files (~3):**
- `portfolio/signals/forecast.py` — delete `_run_kronos*` wrappers,
  circuit-breaker globals, config-reading init, and the Kronos block
  in `forecast()`. Keep the `kronos_1h` / `kronos_24h` keys in the
  `sub_signals` / `raw_sub_signals` dicts set permanently to `"HOLD"`
  **for one release** so downstream consumers (dashboard, accuracy
  tracker) don't `KeyError`. Next release can drop the keys.
- `portfolio/forecast_accuracy.py` — stop trying to backfill `kronos_*`
  outcomes. Filter them out of the accuracy aggregation.
- `tests/test_signals_forecast.py` — update assertions. Any test that
  mocked `_run_kronos` can be deleted.
- Note in `docs/CHANGELOG.md` explaining the retire. Memory entry so
  future-Claude knows we decided, not forgot.

**Config:** set `forecast.kronos_enabled = false` in `config.example.json`.
Leave `config.json` alone (outside-repo symlink, API keys) — user
overrides.

### Batch 1 — Probability / calibration journaling infra (highest value)

**Files (~5):**
- `portfolio/llm_probability_log.py` (new) — append-only JSONL logger
  writing `{ts, signal, ticker, horizon, probs{BUY,SELL,HOLD}, chosen,
  confidence, tier?}`.
- `portfolio/signal_engine.py` — call logger whenever an LLM-family
  signal produces a vote (ministral, qwen3, sentiment, news_event,
  forecast, claude_fundamental). Keyed by signal name so we can
  selectively skip non-probabilistic signals.
- `portfolio/accuracy_stats.py` — add `brier_score_by_signal()` and
  `log_loss_by_signal()` helpers reading the new log + backfilled
  outcomes.
- `portfolio/local_llm_report.py` — include Brier + log-loss +
  calibration-bucket histogram (predicted bucket vs empirical hit rate)
  in the daily export.
- `tests/test_llm_probability_log.py` (new) — round-trip write/read,
  schema stability, Brier-score arithmetic sanity check.

**Shape of the log row:**
```json
{"ts":"...","signal":"ministral","ticker":"BTC-USD","horizon":"1d",
 "probs":{"BUY":0.12,"HOLD":0.55,"SELL":0.33},"chosen":"HOLD",
 "confidence":0.55,"tier":null}
```
Tier present for `claude_fundamental` (haiku/sonnet/opus), null for
single-model signals.

**Why this matters:** today we know Ministral is 13 % on BTC. We don't
know whether it was *confidently wrong* or *uncertainly wrong*.
Brier + calibration tells us which.

### Batch 3 — Claude Fundamental empty-row suppression + error surface

**Files (~2):**
- `portfolio/signals/claude_fundamental.py`
  - In `_journal_refresh()` (~line 642): skip `atomic_append_jsonl`
    when `action == "HOLD" and confidence == 0.0 and reasoning == ""`.
    Keeps the cache update; just doesn't bloat the log.
  - In `_refresh_tier()` (~line 691): when the Claude CLI returns an
    empty response *after* making the API call, call
    `record_critical_error("claude_empty_response", "claude_fundamental_<tier>", ...)`.
    Distinguishes abstention-by-choice from API-went-dark.
- No accuracy-stats change — A2 confirmed it already filters HOLDs.

### Batch 5 — Ministral per-ticker gate verification

**Files (~2):**
- `tests/test_ministral_gate_live.py` (new) — snapshot test that with
  the current `data/accuracy_cache.json`, Ministral returns HOLD for
  BTC / ETH / XAG at 1 d (below 50 %) and BUY/SELL only for tickers
  above 50 %. Keeps drift from sneaking back in.
- Small doc note in `portfolio/ministral_signal.py` pointing to where
  the per-ticker gate lives, if not already documented.

### Batch 6 — MSTR loop documentation (demoted, no code change)

- Append a short "MSTR Loop behavior" section to
  `docs/SESSION_PROGRESS.md`: Phase B (shadow), session window
  15:30 – 22:00 CET, file writes first row on first in-session decision.
- Optionally extend `portfolio/mstr_loop/__main__.py` to log one
  startup line declaring `PHASE=<value>` so future investigators see
  it immediately.

### Batch 7 — Shadow-signal registry & auto-retire (additive)

**Files (~3):**
- `portfolio/shadow_registry.py` (new) — structure of
  `(signal_name, entered_shadow_ts, promotion_criteria, last_reviewed_ts)`
  backed by `data/shadow_registry.json`.
- `portfolio/local_llm_report.py` — include shadow-registry status
  + days-in-shadow in the daily export.
- `scripts/review_shadow_signals.py` (new) — CLI that flags signals
  in shadow > 30 d without hitting promotion criteria.
- Seed with: FinGPT (promotion gate 60 % @ 200 samples, current 61.5 %
  agreement @ 574 — close; needs accuracy not agreement), FinBERT
  (no promotion path — archival only), CreditSpread (disabled pending
  validation), CryptoMacro (disabled pending validation), Qwen3
  (in shadow rotation — 54.7 % is the watermark to beat).

**Note on Kronos:** after Batch 4, Kronos is retired — not in shadow.
Don't seed it.

---

## Deferred to backlog (`docs/IMPROVEMENT_BACKLOG.md`)

- Chronos 3 h / 3 d horizon expansion (low value while 1 h / 24 h near 50 %).
- Replacing FinGPT with a newer 8 B finance model.
- Qwen3 promotion / retire decision — needs more samples.
- Migrating all three sentiment BERTs into a single batched inference call.

---

## Verification plan

1. After each batch: `.venv/Scripts/python.exe -m pytest tests/<touched> -v`.
2. After all batches: `.venv/Scripts/python.exe -m pytest tests/ -n auto --timeout=60`.
3. Spot-check `data/local_llm_report_latest.json` after a full loop cycle —
   confirm new calibration fields populate.
4. Confirm `data/forecast_predictions.jsonl` gets a fresh row per live
   ticker within one full main-loop cycle post-deploy.
5. Codex adversarial review on the branch per `/fgl`:
   `/codex:adversarial-review --wait --scope branch --effort xhigh`.
6. Merge into main, push via Windows git, restart both loops.

---

## Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| Probability logger adds per-cycle IO | Use `atomic_append_jsonl` (lock-free amortised) |
| Forecast fix reintroduces a write-spam bug | Keep dedup window (`_PREDICTION_DEDUP_TTL = 60 s`) |
| Kronos retire breaks tests that import the sub-signal name | Keep `raw_sub_signals["kronos_*"]` key but always HOLD |
| Excluding empty Claude Fundamental rows changes the 61.6 % headline | Document the pre/post delta in `docs/CHANGELOG.md` — we want the real number |

---

## Execution order (post-investigation revision)

Revised order reflects A1-A4 findings. Trivial / highest-impact first.

1. ✅ Spawn A1/A2/A3/A4 investigation subagents (done 13:45).
2. ✅ Commit the v1 plan on `main` (done: `68793cc9`).
3. ✅ Create worktree `Q:/finance-analyzer-llm-health` on branch
   `fix/llm-health-20260421` (done).
4. Commit this expanded plan on `main` (v2).
5. In the worktree, execute in this order:
   - **Batch 2** (forecast re-enable) — immediate, trivial, restores
     signal + logging. Highest impact per diff.
   - **Batch 4** (Kronos retire) — contained ~150-line delete.
     Mechanical.
   - **Batch 1** (probability / calibration infra) — bigger; needs new
     file + integration points.
   - **Batch 3** (Claude Fundamental empty-row suppression).
   - **Batch 5** (Ministral gate test lock-in).
   - **Batch 7** (shadow registry + auto-retire script).
   - **Batch 6** (MSTR loop docs).
6. After each batch: `pytest tests/<touched>` + commit.
7. After all batches: full `pytest -n auto`.
8. Codex adversarial review:
   `codex review --commit <branch-tip-SHA>` (or `/codex:adversarial-review`
   per `/fgl`).
9. Merge into `main`, push via Windows git, restart `PF-DataLoop` +
   `PF-MetalsLoop` (Batch 2 changes signal invocation — loop restart
   mandatory).
10. Clean up worktree + branch.
11. Send Telegram summary of the LLM-health audit + fixes landed.

## What this plan does NOT fix (explicit non-goals)

- Does not replace FinGPT with a newer model (backlog).
- Does not expand Chronos to 3 h / 3 d horizons (backlog — low value
  while 1 h / 24 h accuracy is near 50 %; answers user's question).
- Does not touch live production `config.json` (outside repo).
- Does not retrain Ministral / Qwen3 (separate project,
  `training/unsloth/`).
- Does not re-enable SHORT-side metals trading (pre-existing
  `TODO(short-reenable)` anchor).
