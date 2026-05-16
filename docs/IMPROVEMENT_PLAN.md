# Improvement Plan — 2026-05-16

## Exploration Summary

6 parallel agents + direct code reading covered the full codebase:
- Signal pipeline (signal_engine.py, accuracy_stats.py, accuracy_degradation.py, loop_contract.py)
- Data collection (data_collector, fear_greed, sentiment, futures_data, onchain, fx_rates)
- Orchestration (main.py, agent_invocation, trigger, portfolio_mgr, risk_management)
- Infrastructure (dashboard/app.py, auth.py, house_blueprint.py, export_static.py)
- Metals subsystem (metals_loop.py, grid_fisher.py)
- Portfolio & risk (trade_guards.py, equity_curve.py, monte_carlo.py)

Previous session (2026-05-15) fixed B1-B6 (ADX cache, flip cooldown, alert_budget thread
safety, reporting thread safety, data_collector error visibility). Those are verified fixed.

---

## 1. Bugs Found

### B7 [P1] blend_accuracy_data directional sample counts use max() instead of sum()
**File:** `portfolio/accuracy_stats.py:963-967`
**Bug:** When blending all-time and recent accuracy, directional sample counts
(`total_buy`, `total_sell`) use `max(at_v, rc_v)`. This discards samples from the
smaller source. Downstream `_weighted_consensus` uses these counts for directional
gating — understated counts can bypass the min-sample floor for directional accuracy.
**Fix:** Sum the counts: `at_v + rc_v`. The directional *accuracy* values (lines 953-962)
already pick the source with more samples, which is correct — but counts must aggregate.
**Risk:** Low. Produces slightly more conservative gating (more samples = harder to gate).

### B8 [P2] Dashboard UnicodeDecodeError not caught (5 locations)
**Files:**
- `dashboard/app.py:905,917` — `/api/mstr_loop` catches `OSError` but not `UnicodeDecodeError`
- `dashboard/auth.py:65` — `_read_config_uncached()` catches `(FileNotFoundError, JSONDecodeError, OSError)` but not `UnicodeDecodeError`
- `dashboard/export_static.py:65` — `_get_dashboard_token()` catches `(OSError, JSONDecodeError)` but not `UnicodeDecodeError`
- `dashboard/house_blueprint.py:109,289,396` — `json.loads(manifest.read_text())` catches `JSONDecodeError` but not `UnicodeDecodeError`
**Bug:** `UnicodeDecodeError` inherits from `ValueError`, not `OSError`. Corrupt UTF-8 in
any of these files causes unhandled 500s. The auth.py case is worst — blocks all endpoints.
**Fix:** Add `UnicodeDecodeError` (or `ValueError`) to each except clause.
**Risk:** Trivial. Pure exception broadening.

### B9 [P2] SYSTEM_OVERVIEW.md signal counts stale
**File:** `docs/SYSTEM_OVERVIEW.md:8,35`
**Bug:** Says "52 signals (33 active, 19 disabled)". Reality: 65 modules, 17 active, 49 disabled.
Also says "52-signal voting" on line 35. These became stale after the 6-signal disable on 2026-05-15.
**Fix:** Update counts to match CLAUDE.md (65 modules, 17 active, 49 disabled).
**Risk:** None. Documentation only.

---

## 2. False Positives Investigated

- **change_pct=0 default in accuracy_stats.py** — SAFE. `_vote_correct(vote, 0)` returns None
  (neutral) since `abs(0) < 0.05`. Belt-and-suspenders with explicit None check.
- **trade_guards.py race condition** — Already fixed. Has `_state_lock` with proper `with` guards.
- **DISABLED_SIGNALS leaking into consensus** — Not confirmed. Signals filtered before
  `_weighted_consensus` at lines 3513-3556.
- **JSONL parsing in dashboard** — All JSONL reads have proper JSONDecodeError handling.
- **Silent failures in _run_post_cycle** — All tasks wrapped in individual try/except with logging.
- **Invariant check crashes in loop_contract** — All checks guarded, division-by-zero prevented.

---

## 3. Implementation Batches

### Batch 1: Signal accuracy fix + docs (3 files)
1. `portfolio/accuracy_stats.py` — B7: fix directional sample count blending
2. `docs/SYSTEM_OVERVIEW.md` — B9: update signal counts
3. Test: `tests/test_accuracy_stats.py` — add test for blend_accuracy_data directional counts

### Batch 2: Dashboard resilience (4 files)
1. `dashboard/app.py` — B8: add UnicodeDecodeError to mstr_loop catches
2. `dashboard/auth.py` — B8: add UnicodeDecodeError to config reader
3. `dashboard/export_static.py` — B8: add UnicodeDecodeError to token reader
4. `dashboard/house_blueprint.py` — B8: add UnicodeDecodeError to 3 manifest reads
5. Test: verify dashboard endpoints handle corrupt UTF-8 gracefully

---

## 4. Skipped (Too Risky for Autonomous)

- **metals_loop.py monolith (ARCH-18)**: 7,882 lines. Highest bug density but too intertwined
  for autonomous refactoring. Would need manual review.
- **main.py re-exports (ARCH-17)**: 100+ re-exports. Breaking change to remove — callers unknown.
- **No CI/CD (ARCH-19)**: Infrastructure change, not code fix.
- **No type checking (ARCH-20)**: Would be a multi-session project.
