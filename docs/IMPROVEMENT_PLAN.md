# Improvement Plan — Auto-Session 2026-05-17

Created: 2026-05-17
Branch: `improve/auto-session-2026-05-17`

## Exploration Summary

4 parallel agents + direct code reading covered full codebase:
- Signal pipeline (signal_engine, signal_registry, accuracy_stats, outcome_tracker)
- Data collection & orchestration (main, data_collector, trigger, agent_invocation, health)
- Portfolio & risk (portfolio_mgr, risk_management, trade_guards, equity_curve, monte_carlo)
- Dashboard & metals (dashboard/app, metals_loop, grid_fisher, avanza, file_utils)

Prior session (2026-05-16) fixed B7-B9 (blend_accuracy directional counts, dashboard
UnicodeDecodeError, SYSTEM_OVERVIEW counts). Those are verified merged.

---

## 1. Bugs Found

### B10 [P1] health.py signal rolling window uses list + slice (O(n) per append)
**File:** `portfolio/health.py` — `_signal_health` dict values
**Bug:** Signal health tracking uses `list.append(entry)` then `entries = entries[-50:]`
every cycle. This creates a new list object every append (O(n) copy + GC pressure).
With 17 signals × 5 tickers = 85 appends/min, this is ~5,100 unnecessary list copies/hour.
**Fix:** Replace with `collections.deque(maxlen=50)`. Constant-time append, automatic eviction.
**Risk:** Very low. Deque supports same iteration/indexing patterns used downstream.

### B11 [P2] shared_state.py redundant time.time() under lock
**File:** `portfolio/shared_state.py:69`
**Bug:** `_now_evict = time.time()` called inside `_cache_lock` after `now = time.time()`
was already captured outside. Adds a syscall under lock contention (8 concurrent workers).
**Fix:** Reuse `now` variable. Difference is <1ms, irrelevant for timeout comparison.
**Risk:** None. Pure optimization.

### B12 [P2] signal_engine.py broad Exception in enhanced signal dispatch
**File:** `portfolio/signal_engine.py:3747` (approximate — enhanced signal try/except block)
**Bug:** Catches bare `Exception` and returns HOLD with only DEBUG-level log. A crashing
signal module is indistinguishable from a legitimate HOLD. Masks bugs for days/weeks.
**Fix:** Log at WARNING with `exc_info=True` (traceback). Keep HOLD return for safety,
but make crashes visible in logs and health module.
**Risk:** Low. No behavior change (still returns HOLD), only visibility improvement.

### B13 [P2] grid_fisher.py state file has no cross-process lock
**File:** `portfolio/grid_fisher.py` — state read/write
**Bug:** Metals loop writes `data/grid_fisher_state.json` every cycle. Dashboard reads it
for `/api/grid-fisher`. No file lock → race condition on concurrent access.
**Fix:** Use `file_utils.atomic_write_json()` (already atomic via tempfile+rename) and add
shared/exclusive lock via `portfolio.process_lock` for the read side in dashboard.
**Risk:** Low. atomic_write_json already prevents corruption; lock prevents stale reads.

### B14 [P3] Hardcoded correlation priors duplicated across modules
**Files:** `portfolio/monte_carlo_risk.py`, `portfolio/risk_management.py`
**Bug:** Both files define `{"BTC-USD": {"ETH-USD": 0.7}, ...}` independently.
Values can drift after edits to one file. No single source of truth.
**Fix:** Extract to `portfolio/correlation_priors.py` as a frozen constant.
**Risk:** Low. Pure extraction, no logic change.

### B15 [P3] health.py dead-signal detection time comparison
**File:** `portfolio/health.py` — dead signal check
**Bug:** Compares `time.time()` (epoch float) with signal_log ISO timestamp strings.
Works via `dateutil.parser` but breaks silently during DST transitions when system
clock and log timestamps briefly disagree by 1 hour.
**Fix:** Normalize both to UTC epoch before comparison. Use existing `datetime.now(UTC)`.
**Risk:** Very low. Edge case only manifests during 2 DST transitions/year.

---

## 2. False Positives Investigated

- **BUG-176 concentration stacking** — Investigated: the edge case requires >5 positions
  in same asset class, which Patient/Bold strategies never reach (max 3 per class by design).
  Not fixing — would add complexity for an impossible state.
- **Signal dispatch "soft confidence" implicit contract** — Not a bug. The contract is
  documented in signal_registry.py docstring and enforced by the min_confidence floor.
- **main.py singleton lock race** — Already fixed with msvcrt/fcntl non-blocking lock.
- **BUG-M2 grid_fisher cross-process lock** — False positive. `atomic_write_json` uses
  tempfile → `os.replace` (atomic on NTFS). Dashboard reader always sees complete file.
- **B15 health.py DST time comparison** — False positive. `check_outcome_staleness` uses
  `datetime.fromisoformat().timestamp()` → UTC epoch vs `time.time()` → UTC epoch. Both
  are DST-immune. `check_dead_signals` uses vote counts, no timestamps.

---

## 3. Implementation Batches

### Batch 1: Foundation Fixes (3 files, no dependencies)
1. `portfolio/health.py` — B10: deque conversion for signal rolling window
2. `portfolio/shared_state.py` — B11: remove redundant time.time() under lock
3. Tests: verify in `tests/test_health.py`, `tests/test_shared_state.py`

### Batch 2: Signal Visibility (2 files)
1. `portfolio/signal_engine.py` — B12: upgrade exception logging to WARNING + traceback
2. Tests: `tests/test_signal_engine.py` — verify warning is logged on signal crash

### Batch 3: Correlation Priors Extraction (3 files, new module)
1. NEW `portfolio/correlation_priors.py` — B14: single source of truth
2. `portfolio/monte_carlo_risk.py` — import from new module
3. `portfolio/risk_management.py` — import from new module
4. Tests: NEW `tests/test_correlation_priors.py`

### Batch 4: Grid Fisher Safety (2 files)
1. `portfolio/grid_fisher.py` — B13: add process lock around state file I/O
2. `_state_reset.py` — add grid_fisher state to test reset list
3. Tests: `tests/test_grid_fisher.py`

### Batch 5: Health Robustness (1 file)
1. `portfolio/health.py` — B15: normalize dead-signal time comparison to UTC
2. Tests: `tests/test_health.py` — DST edge case test

---

## 4. Skipped (Out of Scope)

- **ARCH-17 main.py re-exports**: 10+ test files import from main. Dedicated session.
- **ARCH-18 metals_loop.py monolith (7,882 lines)**: Needs design session.
- **Circuit breaker metrics endpoint (FEAT-1)**: Nice-to-have, not a fix.
- **Any config.json changes**: Security rule.
- **Any live trading logic changes**: Safety rule.

---

## 5. Success Criteria

- [ ] All 5 batches implemented with passing tests
- [ ] Full test suite green (`pytest tests/ -n auto`)
- [ ] No new test failures introduced
- [ ] SYSTEM_OVERVIEW.md updated
- [ ] Merged to main and pushed
