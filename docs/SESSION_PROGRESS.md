# Session Progress — Macro-event regime gating (2026-04-28 evening)

**Session start:** 2026-04-28 ~15:30 CET
**Status:** COMPLETE — Merged + Pushed (a6b47fbd)

## User's framing

After this morning's audit confirmed 19 of 21 flagged signal degradations
were statistically REAL — not detector noise — the user asked: *"sounds
like we need to identify when these events happen and adapt the config
to this timeline and then change back when events subside?"* (`yes` →
ultraplan).

Existing infra had partial macro awareness (econ_calendar signal,
accuracy_degradation FOMC blackout) but signal *weights* didn't adapt —
only the alert layer did.

## What shipped (merge a6b47fbd)

**Detection:** `econ_dates.is_macro_window(now=None, lookback_hours=24,
lookahead_hours=72)` — True if any high-impact event (FOMC/CPI/NFP)
within ±window. Self-contained iteration over ECON_EVENTS for testability.

**Suppression:**
- `claude_fundamental` → force-HOLD (worst case: 30-120min LLM lag +
  >75% BUY bias)
- `sentiment`, `momentum_factors`, `structure` → weight × 0.5

**Wiring (signal_engine.py):**
- Force-HOLD mutates `votes` upstream of buy/sell/core_active counting
- Leader picker / Top-N gate use macro-adjusted accuracy (`_leader_accuracy_key`,
  `_topn_accuracy_key`)
- Downweight multiplier in weight loop after horizon_mults
- 5-min TTL cache + state-transition logging

**Observability:** `agent_summary.macro_window.active`,
`agent_context_t2.macro_window`, INFO log on transitions.

**Tests:** 22 new. 229 affected tests green. 50 pre-existing failures unchanged.

## Codex review (2 rounds, all P1/P2 addressed)

- R1 P1: macro force-HOLD too late → stale post_persistence_voters
- R1 P2: macro_window not propagated to Tier 2 whitelist
- R1 P2: correlation leader picked by raw accuracy → overlay neutralized
- R2 P1: core_active/buy/sell from pre-mutation votes
- R2 P2: Top-N gate ranked by raw accuracy → overlay neutralized
- R3: codex daily quota hit (same as yesterday)

## v1 limitation

Calendar covers US events only (FOMC/CPI/NFP/GDP). Past week's macro
density (ECB, BoE, Mag 7) is NOT in `econ_dates.py` — detector would
have classified Apr 21-28 as a normal week. Documented; expansion in plan roadmap.

## Deferred

- Backtest harness replaying signal_log against the gate
- ECB/BoE/BoJ calendar expansion
- Per-sector gating via existing `EVENT_SECTOR_MAP`

---

# Session Progress — Accuracy degradation root-cause + statistical rigor (2026-04-28 afternoon)

**Session start:** 2026-04-28 ~13:00 CET
**Status:** COMPLETE — Merged + Pushed (47b4d474)

## User's framing

User pushed back on yesterday's framing of the contract-alert spam:
*"the spam wasn't the problem, the problem was what it was reporting — it's
a red flag that something is broken in the system and THAT needs fixing.
Do a deep investigation. Absorb the entire codebase and get a good
overview what is going on."*

Yesterday's fix muted the smoke detector. This session asked: *what's
actually on fire?*

## Investigation (3 parallel Explore agents + direct data dives)

The investigation surfaced four interacting infrastructure failures
that together caused the alert to fire on noise:

1. **Daily snapshot writer silently no-opped for 7 days.**
   `accuracy_snapshot_state.json` claimed today's snapshot was done
   but `accuracy_snapshots.jsonl` had only 4 entries — none after Apr 21.
   Once `last_snapshot_date_utc=today`, `maybe_save_daily_snapshot`
   returned False every cycle. A single state-without-write desync
   silenced the writer for the full day; the prior session's
   operational fixup poisoned today's natural run.

2. **Detector compared against a 7d-stale baseline computed on small N.**
   Apr 21 sentiment_recent=75.3% on N=223 was a one-week spike well
   above lifetime 46% (N=39k). Today's recent: 43.3% on N=187, roughly
   AT lifetime. The "32pp drop" was largely regression to mean.

3. **Cooldown hash drifted every cycle.** The Telegram cooldown
   hashed on rendered message text containing percentages like
   "33.7%→33.2%". Each new sample shifts the percentage, every cycle
   produces a new hash, multi-hash dedup never traps duplicates.

4. **Real signal weakness exists.** 19 of 21 flagged signals are
   STATISTICALLY REAL degradation per the new audit script
   (sentiment, momentum_factors, claude_fundamental, structure across
   BTC/ETH/XAG/MSTR). 2 NOISE.

## What shipped (merge 47b4d474)

5 batches in worktree `fix/accuracy-pipeline-20260428`:

- **Batch 1**: bulletproof snapshot writer. Verify JSONL grew before
  persisting state; silent failures journal to critical_errors with
  30-min dedup cooldown so PF-FixAgentDispatcher engages. 6 tests new.
- **Batch 2**: `scripts/backfill_accuracy_snapshots.py` regenerates
  Apr 22-27 from signal_log replay. Threads horizon-aware temporal
  cutoffs so backfilled outcomes match what was knowable at each
  target date. Forecast block copied from nearest *past* snapshot.
- **Batch 3**: binomial-SE significance gate in `_maybe_alert`. Drop
  must satisfy `drop_pp >= max(15.0, 2*SE)`; sample-size floor 100→200.
  13 tests new.
- **Batch 4**: stable cooldown hash for accuracy_degradation. Hashes
  sorted (scope::key) set from `details["alerts"]`, ignores message
  text. 7 tests new in TestAccuracyDegradationStableHash. Aligned the
  journal-lookup helper to use the same identity hash (Codex P1).
- **Batch 5**: `scripts/audit_accuracy_drops.py` classifies REAL vs
  NOISE per binomial SE. Today's audit: 19 REAL, 2 NOISE.

## Codex review

Ran `codex review --base main` 7 rounds, all P1/P2 addressed:
- R1: chronological-order, forecast template, audit forecast scope, per-ticker lifetime
- R2: per_ticker.recent re-cutoff, audit forecast prefix, time-aware template
- R3 (P1 critical): journal lookup hash mismatch with dispatch hash
- R4: temporal correctness — outcome maturity in backfill
- R5: window alignment, cross-env warning
- R6: forecast accuracy honors --data-dir
- R7: silent-failure journal rate-limit

## Tests

229 tests passing on files I touched (loop_contract*, accuracy_*).
Full suite has 50 pre-existing failures (test_consensus, test_metals,
test_signal_engine_core, etc.) — verified `test_signal_names_count`
fails on main itself before my changes.

## Operational

- Backfilled Apr 22-27 snapshots (one-shot, not committed): sentiment
  trajectory now visible day-by-day: 75 → 70 → 64 → 58 → 43 → 45 → 45 → 43.
  Gradual decline, not catastrophic cliff.
- Audit ran: `docs/accuracy_audit_20260428.md` — 19 REAL drops worth
  follow-up gating decisions (config, not code).
- Killed running python loops, restarted via schtasks.

## What's deferred (config decisions, not code)

The audit identified 19 statistically real per-ticker degradations.
Top candidates for gating/disabling:
- `MSTR::momentum_factors` 60.1% → 32.5% (27.6pp drop)
- `BTC-USD::claude_fundamental` 63.5% → 38.9% (24.7pp)
- `XAG-USD::claude_fundamental` 58.8% → 37.4% (21.3pp)
- `MSTR::sentiment` 61.8% → 41.7% (20.1pp)
- `signal::sentiment` 60.9% → 42.7% (18.2pp aggregate)

These are signal-by-signal config tuning decisions; the loop_contract
infrastructure can't decide which to disable without trading judgment.

---

# Session Progress — Auto-Improve 2026-04-28

**Session start:** 2026-04-28 ~10:00 CET
**Status:** COMPLETE — Merged + Pushed

## What was done

### Phase 1: Exploration (4 parallel agents)
- Core loop & orchestration: agent_invocation, trigger, market_timing, health
- Signal system: signal_engine, signal_registry, accuracy_stats, outcome_tracker, 41 signal modules
- Data, portfolio & risk: data_collector, portfolio_mgr, trade_guards, risk_management, file_utils, shared_state
- Metals, dashboard, bots: metals_loop, dashboard, golddigger, elongir, avanza, telegram

### Phase 2: Plan
6 bugs found (BUG-230 through BUG-235), ~135 ruff lint violations catalogued.
Implementation planned in 3 batches.

### Phase 3: Implementation (3 batches)

**Batch 1 — Security & Safety (3 files):**
- BUG-230: CORS wildcard → localhost whitelist
- BUG-231: heartbeat .write_text() → atomic_write_text()
- BUG-232: NaN fx_rate guard via math.isfinite()
- BUG-235: Dashboard 500 errors sanitized
- 9 new tests (4 CORS, 5 portfolio_value)

**Batch 2 — Ruff auto-fix + unused vars (18 files):**
- 22 auto-fixed F401/I001/UP045 in portfolio/
- 3 auto-fixed F401/I001 in data/
- 9 manual F841 unused variable removals
- BUG-233: CANCEL_HOUR/CANCEL_MIN defined
- BUG-234: dead variable removed

**Batch 3 — Scripts & SIM cleanup (9 files):**
- 43 auto-fixed violations in scripts/
- 12 E722 bare-except → except Exception
- 6 SIM102/SIM103 collapsible-if/needless-bool
- 1 SIM103 in metals_swing_trader

### Phase 4: Documentation
- Updated SYSTEM_OVERVIEW.md (known issues, date)
- Updated CHANGELOG.md (new entry)

## What's next
- Remaining ruff: 69 E402 (intentional lazy imports), 12 SIM115 (atomic I/O patterns), 5 E741
- ARCH-18: metals_loop.py (7667 lines) monolith decomposition (deferred)
- ARCH-19: CI/CD pipeline (deferred)
- ARCH-20: mypy type checking (deferred)

## Blockers
None. All 3 batches implemented and tested clean.

### 2026-04-28 10:54 UTC | fix/sentiment-relevance-and-aggregation
45aaf76e docs(plan): sentiment relevance filter + aggregation fixes
docs/PLAN_sentiment_2026_04_28.md

### 2026-04-28 10:55 UTC | fix/sentiment-relevance-and-aggregation
43f1b1b3 test(sentiment): pin contract for relevance filter + decisive aggregation
tests/test_portfolio.py
tests/test_sentiment_relevance_filter.py

### 2026-04-28 10:59 UTC | fix/sentiment-relevance-and-aggregation
70207639 fix(sentiment): relevance filter + decisive aggregation + Trading-Hero primary
portfolio/news_keywords.py
portfolio/sentiment.py

### 2026-04-28 11:06 UTC | main
4e64bae7 plan: accuracy degradation root-cause + statistical rigor (2026-04-28)
docs/PLAN.md

### 2026-04-28 11:09 UTC | fix/accuracy-pipeline-20260428
ad7d9500 fix(accuracy): bulletproof daily snapshot writer against silent desync
portfolio/accuracy_degradation.py
tests/test_accuracy_degradation.py
tests/test_accuracy_degradation_writer_safety.py

### 2026-04-28 11:11 UTC | fix/accuracy-pipeline-20260428
753febab fix(accuracy): add backfill script for missing daily snapshots
scripts/backfill_accuracy_snapshots.py

### 2026-04-28 11:14 UTC | fix/accuracy-pipeline-20260428
9e13060e fix(accuracy): add binomial-SE significance gate to degradation detector
portfolio/accuracy_degradation.py
tests/test_accuracy_degradation_significance.py

### 2026-04-28 11:16 UTC | fix/accuracy-pipeline-20260428
f8960b42 fix(contract): stable identity hash for accuracy_degradation cooldown
portfolio/loop_contract.py
tests/test_loop_contract_alert_cooldown.py

### 2026-04-28 11:19 UTC | fix/accuracy-pipeline-20260428
d201bbc8 fix(accuracy): add audit script that classifies real degradation vs noise
scripts/audit_accuracy_drops.py

### 2026-04-28 11:20 UTC | fix/sentiment-relevance-and-aggregation
a7a9a24b docs(changelog): sentiment relevance + decisive aggregation entry
docs/CHANGELOG.md

### 2026-04-28 11:30 UTC | fix/accuracy-pipeline-20260428
07364a04 fix(accuracy): address codex review findings on backfill + audit scripts
scripts/audit_accuracy_drops.py
scripts/backfill_accuracy_snapshots.py

### 2026-04-28 11:40 UTC | fix/accuracy-pipeline-20260428
eb593258 fix(accuracy): codex round 2 findings on backfill + audit scripts
scripts/audit_accuracy_drops.py
scripts/backfill_accuracy_snapshots.py

### 2026-04-28 11:52 UTC | fix/accuracy-pipeline-20260428
be2e2743 fix(contract,accuracy): codex round 3 findings — dedup hash + storage
portfolio/loop_contract.py
scripts/audit_accuracy_drops.py
scripts/backfill_accuracy_snapshots.py
tests/test_loop_contract_accuracy_dispatcher.py

### 2026-04-28 12:02 UTC | fix/accuracy-pipeline-20260428
206639e7 fix(accuracy): codex round 4 — temporal correctness in backfill
scripts/backfill_accuracy_snapshots.py

### 2026-04-28 12:12 UTC | fix/accuracy-pipeline-20260428
0ffdeb1d fix(accuracy): codex round 5 — window alignment + cross-env warning
scripts/audit_accuracy_drops.py
scripts/backfill_accuracy_snapshots.py

### 2026-04-28 12:23 UTC | fix/accuracy-pipeline-20260428
2c417ad0 fix(accuracy): codex round 6 P2 — forecast accuracy honors --data-dir
scripts/audit_accuracy_drops.py

### 2026-04-28 12:32 UTC | fix/accuracy-pipeline-20260428
5ba9eae2 fix(accuracy): codex round 7 P3 — rate-limit silent-failure journaling
portfolio/accuracy_degradation.py
tests/test_accuracy_degradation_writer_safety.py

### 2026-04-28 12:39 UTC | main
378897e8 docs(session): accuracy degradation root-cause + statistical rigor session summary
docs/SESSION_PROGRESS.md

### 2026-04-28 13:03 UTC | main
26657e2f docs(accuracy): audit artifact + post-commit log entry
docs/SESSION_PROGRESS.md
docs/accuracy_audit_20260428.md

### 2026-04-28 13:16 UTC | fix/accuracy-pipeline-followups-20260428
1728447d fix(accuracy): C1 atomic-I/O + I2 explicit conditional + I5 audit output path
.gitignore
portfolio/accuracy_stats.py
scripts/audit_accuracy_drops.py

### 2026-04-28 13:17 UTC | main
e439a992 plan: macro-event regime gating (auto-adapt signal weights, 2026-04-28)
docs/PLAN.md

### 2026-04-28 13:21 UTC | feat/dashboard-ops-board
e75ffd60 docs: dashboard ops board design + implementation plan
docs/superpowers/plans/2026-04-28-dashboard-ops-board.md
docs/superpowers/specs/2026-04-28-dashboard-ops-board-design.md

### 2026-04-28 13:23 UTC | fix/macro-window-gating-20260428
47a6e41a fix(signals): macro-event regime overlay (auto-down-weight + force-HOLD)
portfolio/econ_dates.py
portfolio/reporting.py
portfolio/signal_engine.py
tests/test_macro_window_gating.py

### 2026-04-28 13:24 UTC | feat/dashboard-ops-board
9b5217a1 feat(dashboard): add OPS_THRESHOLDS + _status_color helper
dashboard/app.py
tests/test_dashboard.py

### 2026-04-28 13:30 UTC | feat/dashboard-ops-board
0b66d23d fix(dashboard): tidy _status_color imports + boundary tests
dashboard/app.py
tests/test_dashboard.py

### 2026-04-28 13:34 UTC | fix/llm-outcome-dedup-null-horizon
87e2569b fix(accuracy): null-horizon dedup + per-ticker bias detection
portfolio/llm_outcome_backfill.py
portfolio/signals/claude_fundamental.py
tests/test_llm_outcome_backfill.py
tests/test_signals_claude_fundamental.py

### 2026-04-28 13:34 UTC | fix/macro-window-gating-20260428
f75434c8 fix(signals): codex round 1 — voter quorum, Tier 2 propagation, leader pick
portfolio/reporting.py
portfolio/signal_engine.py

### 2026-04-28 13:35 UTC | feat/dashboard-ops-board
c72127ad feat(dashboard): _compute_metals_loop_status helper
dashboard/app.py
tests/test_dashboard.py

### 2026-04-28 13:41 UTC | feat/dashboard-ops-board
b46405b3 fix(dashboard): isinstance guard for non-dict JSONL entries
dashboard/app.py
tests/test_dashboard.py

### 2026-04-28 13:42 UTC | fix/macro-window-gating-20260428
7d92ceaa fix(signals): codex round 2 — macro mutations consistent across pipeline
portfolio/signal_engine.py

### 2026-04-28 13:44 UTC | feat/dashboard-ops-board
8fd4902f feat(dashboard): _compute_llm_health_summary helper
dashboard/app.py
tests/test_dashboard.py
