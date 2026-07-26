# FGL 2026-07-26 — Orchestrator Independent Pass

Baseline: `main @ 597176b2`. Run separately from the 9 review subagents so it can
cross-critique them rather than echo them. Scope chosen for what a per-subsystem
reviewer structurally _cannot_ see: (a) defects introduced by this session's own
changes, (b) contracts that span subsystem boundaries.

Documentation only — no code changed.

## Verdict

The system's newest capability is also its newest hazard. Two changes shipped in
the last 8 days — the component-registry overlay (per-instrument enablement as
data) and a pytest write-guard in the atomic I/O primitive — each introduce a
silent-divergence path of exactly the class this project keeps getting burned by.
Neither is currently active (the overlay file does not exist yet; production runs
are not under pytest), so both are **latent**: they arm the moment someone uses
the feature as designed. That timing is the point of catching them now.

I also confirm the dashboard's zero-denominator paths are correctly guarded, so
the "total disablement" scenario below degrades to silence rather than a crash.

## Findings

- `portfolio/file_utils.py:396` — **P1** — The pytest write-guard I added on
  2026-07-19 keys on `PYTEST_CURRENT_TEST`, which is **inherited by every child
  process**. Any production writer whose environment inherited that variable —
  a Layer-2 `claude -p` subprocess spawned from a test, or a loop started from a
  shell where the variable leaked — has **all** of its `atomic_append_jsonl`
  calls into the real `data/` directory silently discarded, at `logger.debug`
  only. That is the audit trail: `critical_errors.jsonl`, `signal_log.jsonl`,
  `layer2_journal.jsonl`. A guard added to stop tests from _faking_ journal rows
  can therefore _delete_ real ones.
  **Blast radius — treat as P0 if any production path can inherit the variable:**
  `grid_fisher.py:1813-1830` is the defence for a naked leveraged position (the
  fix for a previous FGL "P0-3"): when a stop re-arm fails it sets
  `stop_needs_rearm`, logs CRITICAL, and files a `grid_fisher_naked_position`
  row — **via `atomic_append_jsonl` into the real `data/critical_errors.jsonl`**.
  So the one alarm that says "leveraged inventory currently has no broker-side
  stop" routes through the function this guard can silently no-op. The stop logic
  itself is sound; the _alarm_ is what gets muted.
  Verified the discriminator exists: a spawned
  child inherits the env var (`True`) but does **not** have pytest in
  `sys.modules` (`False`). Fix: require both — `"PYTEST_CURRENT_TEST" in
os.environ and "pytest" in sys.modules` — or better, gate on an explicit
  `PF_TEST_ISOLATION=1` set by `tests/conftest.py`, and log the drop at WARNING
  so a mistake is visible rather than debug-buried.

- `portfolio/component_registry.py` (overlay path) + 14 legacy readers —
  **P1** — **Registry/legacy divergence.** With `signals.use_registry=true`
  (live since 2026-07-18), the engine's enablement answer comes from the
  registry _including_ the operator overlay `data/control/registry_overrides.json`.
  The overlay cannot change `tickers.DISABLED_SIGNALS`, which these still read
  directly: `accuracy_stats.py`, `reporting.py`, `ic_computation.py`,
  `ticker_accuracy.py`, `shadow_registry.py`, `backtester.py`,
  `llm_probability_log.py`, `instrument_profile.py`, `signal_registry.py`,
  `signals/phi4_mini_reasoning.py`, `dashboard/app.py` (accuracy-history site),
  plus 3 scripts. Demonstrated: an overlay entry
  `{"XAG-USD": {"rsi": {"enabled": false}}}` makes the engine force-HOLD `rsi`
  for XAG while every listed reader still counts `rsi` as an active XAG voter —
  so accuracy is attributed to a signal that never voted, and the dashboard
  reports it enabled. This is measurement contamination plus a dishonest UI,
  and it is triggered by using the feature as intended (`tune_instrument.py
--write`, or the Command Central instrument/LLM toggles). Currently latent:
  the overlay file does not exist. Fix: route the remaining readers through
  `component_registry` (Phase 4.3 was only partially completed — 2 of 3
  `dashboard/app.py` sites were migrated, the accuracy-history one deliberately
  not), or refuse to honour an overlay until its consumers agree.

- `portfolio/component_registry.py` (`_overlay_entry` / `is_globally_disabled`)
  — **P2** — **Type coercion inverts operator intent.** An overlay value of
  `"enabled": "false"` (JSON string, a plausible hand-edit or a stringly-typed
  writer) is truthy, so `not bool("false")` is `False` and the signal ends up
  **enabled** — the exact opposite of the instruction. Verified live: XAG
  `applicable_count` stays 12. On a surface whose whole purpose is disabling
  components on a real-money system, a silent inversion is unacceptable. Fix:
  accept only real booleans; reject/log anything else instead of coercing.

- `portfolio/component_registry.py` (overlay validation) — **P2** — **Typos are
  accepted silently.** An overlay naming a signal that does not exist
  (`"not_a_real_signal"`) or a ticker that does not exist (`"FAKE-USD"`) is
  loaded without complaint and has no effect; `applicable_count` stays 12. An
  operator who mistypes a signal name believes a component was disabled when it
  is still voting. Same for a malformed `"horizons": "1d"` (string not dict),
  silently ignored. Fix: validate keys against `SIGNAL_NAMES` / `ALL_TICKERS` on
  load and log unknown entries loudly; surface them in `/api/control/registry`
  so the dashboard can show "override not applied".

- `portfolio/component_registry.py` + `signal_engine._applicable_count` —
  **P2** — **An overlay can silently disable an instrument entirely.** Disabling
  all 12 XAG-applicable signals drives `applicable_count` to 0 (verified). With
  zero applicable voters, `MIN_VOTERS` can never be satisfied, so the instrument
  is permanently force-HOLD — correct-by-accident (no bad trades) but wholly
  silent: nothing warns that an instrument has gone dark, and the dashboard shows
  a normal HOLD. Fix: treat `applicable_count == 0` for a tracked instrument as a
  contract violation / health-critical state and surface it.

- `portfolio/accuracy_degradation.py:629-661` — **P2** — **The effective-n
  correction covers one overlap source, not two.** Credit where due: a 2026-06-10
  audit already added `_effective_n` (fixed divisor `AUTOCORR_EFFECTIVE_N_DIVISOR
= 20`) to widen the SE gate, because signals emit identical votes in day-long
  blocks. That addresses **vote persistence**. It does not address **outcome-window
  overlap**: rows are 600s snapshots whose outcomes are scored over an h-long
  horizon, so consecutive rows share nearly all of their outcome window. The two
  effects are independent and multiply. Measured overlap factor (`horizon_seconds
/ 600`) versus the fixed K=20: 3h ≈ 18× (K is about right), **1d ≈ 144× (7×
  under-corrected), 3d ≈ 432× (22×), 5d ≈ 720× (36×)** — and SE is understated by
  the square root of that shortfall. A single constant cannot serve every horizon
  because the true factor is horizon-dependent by construction. Consistent with
  the observed 2026-07-14→18 alert storm on long horizons. Note also that the
  module deliberately keeps the **min-samples gates on RAW counts** (its own
  comment says so), so a comparison can proceed on ~10 effective trials once 200
  raw rows exist. Fix: derive the divisor per horizon (`max(K_persistence,
horizon_seconds / snapshot_interval)`) rather than using one constant, and apply
  it to the min-samples gate as well as the SE.
  Corrects my own earlier framing this session, which implied no guard existed:
  one does, it is simply calibrated for the wrong overlap source on long horizons.

## Cross-critique seeds for the synthesis

- Dashboard zero-denominator paths are **correctly guarded** (`system_status.py`
  `total == 0 → continue`, `total <= 0` guard, `confidence = (active/total) if
total else 0.0`). If any subagent reports a division-by-zero there, verify
  before accepting.
- `data/control/registry_overrides.json` does not exist on disk. Any finding that
  asserts current misbehaviour _from_ the overlay is latent, not active — grade
  accordingly.
- Two of today's fixes (`asset_type` labelling, `UNATTENDSLEEP`) are already
  merged; findings restating them are stale, not new.
- The white paper's null result means "signal X is inaccurate" is not a finding.
  Reviewers drifting into model-quality commentary should be redirected to
  correctness and measurement validity.

## Orchestrator finding REJECTED by verification (recorded deliberately)

- `portfolio/signal_engine.py:4692` — **claimed P0, REJECTED** — On seeing the
  utility-boost formula `min(1.0 + u_score, 1.5)` with no lower bound, and live
  data showing four signals with `avg_return` below -1.0 (`econ_calendar` -3.79,
  `intraday_seasonality` -2.28, `gold_overnight_bias` -1.33, `shannon_entropy`
  -1.05), I hypothesised a **negative** accuracy multiplier inverting vote weight
  — which would have been a P0. Reading the enclosing guard disproved it: line
  4691 is `if samples >= 30 and u_score > 0:`, so negative scores never reach the
  formula. The signals-core reviewer's scoping ("5 of 9 qualifying signals") was
  correct and mine ("5 of 29") was misleading, because only the 9 with a positive
  `avg_return` are eligible at all. Recorded because a review that deletes its own
  wrong guesses teaches nothing about how much to trust the rest — and because the
  same discipline is what I applied to the subagents' claims.
