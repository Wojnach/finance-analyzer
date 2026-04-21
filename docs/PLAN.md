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

## Subagent strategy

The four investigation questions are independent and read-only.
Run them in parallel via `Explore` subagents (thoroughness: medium) so
implementation can start once they return. Each returns a finding + a
concrete fix proposal.

| ID | Question | Files in scope |
|----|----------|----------------|
| **A1** | Why did `forecast_predictions.jsonl` stop writing on 2026-04-11? Circuit breaker latched? Dedup bug? Is `_log_prediction` being called but silently short-circuiting? Trace the full write path. | `portfolio/signals/forecast.py`, `portfolio/forecast_accuracy.py`, `portfolio/file_utils.py` |
| **A2** | When a `claude_fundamental_log.jsonl` row has `reasoning=""`, `confidence=0.0`, is that an intentional abstention or a swallowed error? What path writes it? Is it polluting accuracy? | `portfolio/signals/claude_fundamental.py`, `portfolio/accuracy_stats.py` |
| **A3** | Is `Q:/models/kronos_infer.py` salvageable? Run it once manually with a small candle sample. Classify failure modes in the health log. Decide: retire-from-shadow, fix-subprocess, or keep-as-is. | `Q:/models/kronos_infer.py`, `portfolio/signals/forecast.py:283-360` |
| **A4** | Is `portfolio/mstr_loop/` running? Which phase? Where does Phase A → Phase B transition live? Why is `data/mstr_loop_shadow.jsonl` absent? | `portfolio/mstr_loop/*.py`, scheduled-task config |

---

## Implementation batches

All work on worktree branch `fix/llm-health-20260421`.
After each batch: run touched-file tests, commit with conventional message,
update `docs/SESSION_PROGRESS.md`.

### Batch 1 — Probability / calibration journaling infra (additive, highest value)

**Files (~5):**
- `portfolio/llm_probability_log.py` (new) — append-only JSONL logger writing
  `{ts, signal, ticker, horizon, probs{BUY,SELL,HOLD}, chosen, confidence}`.
- `portfolio/signal_engine.py` — call logger whenever an LLM-family signal
  produces a vote (ministral, qwen3, sentiment, news_event, forecast, claude_fundamental).
- `portfolio/accuracy_stats.py` — add `brier_score_by_signal()` and
  `log_loss_by_signal()` helpers reading the new log.
- `portfolio/local_llm_report.py` — include Brier + log-loss + calibration
  bucket histograms (predicted bucket vs empirical hit rate).
- `tests/test_llm_probability_log.py` (new) — round-trip test.

**Why first:** every subsequent fix needs this to measure its effect.

### Batch 2 — Forecast logging regression fix (based on A1)

**Files (~3-5):** depend on subagent root-cause. Candidates:
- `portfolio/signals/forecast.py` — fix the silent early-return / circuit
  breaker latch / dedup cache bug.
- `portfolio/health.py` — add `forecast_predictions_stale_hours` gauge that
  trips a critical error if >6 h since last write during market hours.
- `tests/test_signals_forecast.py` — regression test for the specific
  short-circuit path that dropped writes.

### Batch 3 — Claude Fundamental empty-row triage (based on A2)

**Files (~2-3):**
- `portfolio/signals/claude_fundamental.py` — either suppress empty-row
  emission (abstention path) or fix the upstream error that produced them.
- `portfolio/accuracy_stats.py` — exclude empty-reasoning rows from
  accuracy computation regardless of cause (never let them inflate stats).
- Add a gauge: `claude_fundamental_empty_row_rate_24h`. Critical-error if >20 %.

### Batch 4 — Kronos decision gate (based on A3)

Two possible paths; subagent picks.
- **Retire path**: remove Kronos sub-signal from `forecast.py` majority
  vote, keep it as a pure logging stub only if `kronos_enabled != false`.
  Update `portfolio/forecast_accuracy.py` to drop Kronos backfill.
  Delete `Q:/models/kronos_infer.py` invocation wiring if fully retired.
- **Fix path**: add a VRAM pre-check + retry-on-OOM wrapper, widen timeout
  budgets, normalise stdout parsing. Only if A3 finds the bugs are small.

### Batch 5 — Ministral per-ticker gate verification (mostly read + tests)

**Files (~2):**
- `tests/test_ministral_gate_live.py` (new) — assert that with the current
  `accuracy_cache.json`, Ministral returns HOLD for BTC/ETH/XAG at 1 d and
  BUY/SELL only for tickers above 50 %.
- Small doc note in `portfolio/ministral_signal.py` explaining the gate
  path if it isn't already there.

No production config change — if the gate is already working the tests
lock behavior in. If a bug is found, fix in place.

### Batch 6 — MSTR loop phase audit + shadow-log guard (based on A4)

**Files (~2):**
- `portfolio/mstr_loop/__main__.py` — add a startup log line declaring
  which phase the loop booted into. Fail loud if `SHADOW_LOG` configured
  but directory not writable.
- `docs/SESSION_PROGRESS.md` — document the MSTR loop's current phase.

### Batch 7 — Shadow-signal registry & auto-retire (additive)

**Files (~3):**
- `portfolio/shadow_registry.py` (new) — tuple of
  `(signal_name, entered_shadow_ts, promotion_criteria, last_reviewed_ts)`
  backed by `data/shadow_registry.json`.
- `portfolio/local_llm_report.py` — include shadow-registry status in the
  daily report.
- `scripts/review_shadow_signals.py` (new) — CLI that flags signals in
  shadow >30 d without hitting promotion criteria.
- Seed registry with current shadows: Kronos, FinGPT, FinBERT, CreditSpread,
  CryptoMacro.

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

## Execution order

1. Spawn A1/A2/A3/A4 investigation subagents in parallel (1 message, 4 Agent calls).
2. Commit this plan on `main`.
3. Create worktree `Q:/finance-analyzer-llm-health` on branch `fix/llm-health-20260421`.
4. In the worktree, start Batch 1 (probability logger) while A1-A4 run.
5. As each subagent finishes, queue the follow-on batch (2/3/4/6).
6. Batch 5 and 7 after Batch 1 (infra) is in.
7. Test, Codex review, merge, push, restart loops.
