# Improvement Plan — Auto-Session 2026-05-18

Created: 2026-05-18
Branch: `improve/auto-session-2026-05-18`

## Exploration Summary

5 parallel agents covered full codebase (core orchestration, portfolio/risk, metals/Avanza,
infrastructure/dashboard, signal modules). Inherited unimplemented B10-B14 from 2026-05-17
session. New bugs found and verified with exact line references.

---

## 1. Bugs & Improvements Found

### B10 [P2] health.py signal rolling window uses list + del slice (O(n) per cycle)
**File:** `portfolio/health.py:292-296`
**Bug:** `recent.append(success); del recent[:-50]` creates unnecessary copies under 85
appends/min (17 signals × 5 tickers).
**Fix:** Replace with `collections.deque(maxlen=50)`.
**Risk:** Very low. Deque supports same iteration patterns.

### B12 [P1] signal_engine.py broad Exception in enhanced signal dispatch
**File:** `portfolio/signal_engine.py` — enhanced signal try/except block
**Bug:** Catches bare `Exception` and returns HOLD at DEBUG level. Crashing signals are
invisible for days/weeks.
**Fix:** Log at WARNING with `exc_info=True`. Keep HOLD return for safety.
**Risk:** None — behavior unchanged, only visibility improved.

### B14 [P2] Hardcoded correlation priors duplicated across modules
**Files:** `portfolio/monte_carlo_risk.py`, `portfolio/risk_management.py`
**Bug:** Both files define correlation priors independently. Values can drift.
**Fix:** Extract to `portfolio/correlation_priors.py` (already exists, verify imports).
**Risk:** Low. Pure extraction.

### N1 [P1] trade_guards.py ticker_trades grows unbounded
**File:** `portfolio/trade_guards.py:269-273`
**Bug:** `state["ticker_trades"][key] = now_str` accumulates forever. `new_position_timestamps`
has pruning (298-310) but `ticker_trades` does not.
**Fix:** Add pruning for entries older than 90 days in `record_trade()`.
**Risk:** Low. Only removes stale data that's never accessed.

### N3 [P1] risk_management.py FX cache persist silently swallowed
**File:** `portfolio/risk_management.py:160-161`
**Bug:** `logger.debug("fx cache persist failed: %s", e)` — DEBUG level means invisible in
production. Operator can't detect broken disk writes.
**Fix:** Upgrade to `logger.warning()`.
**Risk:** None. Purely a log level change.

### N4 [P2] reporting.py _module_warnings list allows theoretical duplicates
**File:** `portfolio/reporting.py:152, 780-781`
**Bug:** `_module_warnings` is a list; same module name could appear if multiple failure paths
exist per module. Layer 2 sees duplicated warnings.
**Fix:** Use `sorted(set(_module_warnings))` before writing to summary.
**Risk:** None.

### N5 [P1] trade_guards.py naive datetime comparison on every access
**File:** `portfolio/trade_guards.py:142-145` (approximate)
**Bug:** `if last_trade.tzinfo is None: last_trade = last_trade.replace(tzinfo=UTC)` is
checked every time a trade timestamp is accessed. Should parse once at load time.
**Fix:** Add `_normalize_timestamps()` to `_load_state()` that enforces UTC awareness.
**Risk:** Low. Purely defensive code movement.

### N6 [P2] signal_engine.py enhanced signal dispatch exception detail
**File:** `portfolio/signal_engine.py` — same block as B12
**Bug:** When enhanced signal crashes, error message doesn't include signal name in the
log line. Hard to grep for which signal is broken.
**Fix:** Include `signal_name` in the warning log format.
**Risk:** None.

---

## 2. False Positives Investigated

- **B11** shared_state.py redundant time.time(): The second call is in `_loading_timestamps`
  dict write, not the same as outer `now`. Different purpose. Not worth changing.
- **B15** health.py DST comparison: Already uses UTC epoch on both sides. DST-immune.
- **B13** grid_fisher state lock: `atomic_write_json` already uses tempfile+rename (atomic on
  NTFS). Dashboard reads complete files. Not a real race.
- **message_store.py truncation BUG-131**: Working as designed. When no newline found, cuts at
  byte boundary (max-20 chars). Not ideal but not a bug.
- **health.py del recent[:-50]**: Correctly keeps LAST 50 entries. But still O(n) per delete.
- **portfolio_mgr.py update_state return**: Return is inside `with lock:` context. Safe.

---

## 3. Implementation Batches

### Batch 1: Visibility & Logging (3 files, no dependencies)
1. `portfolio/signal_engine.py` — B12+N6: upgrade exception logging to WARNING + include signal name
2. `portfolio/risk_management.py` — N3: FX cache persist `debug` → `warning`
3. `portfolio/reporting.py` — N4: dedup _module_warnings before writing to summary

### Batch 2: Data Integrity (2 files)
1. `portfolio/trade_guards.py` — N1: add ticker_trades 90-day pruning
2. `portfolio/trade_guards.py` — N5: normalize timestamps at load time

### Batch 3: Performance & Refactoring (3 files)
1. `portfolio/health.py` — B10: deque for signal rolling window
2. `portfolio/correlation_priors.py` — B14: verify single source of truth + update imports
3. `portfolio/monte_carlo_risk.py` + `portfolio/risk_management.py` — B14: import from priors

---

## 4. Skipped (Out of Scope)

- **ARCH-18** metals_loop.py monolith: 7882 lines, needs dedicated design session
- **ARCH-17** main.py re-exports: 100+ symbols, breaks 10+ test files
- **Grid fisher partial fill**: Complex state machine change, too risky for auto-session
- **price_targets.py fill_probability_buy**: GBM symmetry assumption, needs quant validation
- **Dashboard blueprint split**: 1600 lines → blueprints, dedicated session
- **Any config.json or live trading logic changes**

---

## 5. Success Criteria

- [ ] All 3 batches implemented with passing tests
- [ ] Full test suite green (`pytest tests/ -n auto`)
- [ ] No new test failures introduced
- [ ] SYSTEM_OVERVIEW.md updated with session findings
- [ ] Merged to main and pushed
