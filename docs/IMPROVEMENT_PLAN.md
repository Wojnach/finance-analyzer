# Improvement Plan — Auto-Session 2026-05-27

**Branch:** `improve/auto-session-2026-05-27`
**Created:** 2026-05-27 10:30 CET
**Status:** IN PROGRESS

---

## 1. Bugs & Problems Found

### BUG-A: Escalation gate ThreadPoolExecutor leak (P1)
**File:** `portfolio/escalation_gate.py:203`
**Problem:** Creates a new `ThreadPoolExecutor(max_workers=1)` per
`should_escalate()` call. On timeout (>10s), `shutdown(wait=False)` is called
but the hung thread lingers. Leaks OS threads over time.
**Fix:** Module-level singleton executor. Reuse across calls.

### BUG-B: Silent exception swallowing — 12 locations (P2)
**Problem:** `except Exception: pass` hides errors without logging.
Locations needing fix (cleanup code excluded):
- `trigger.py:55` — config load
- `signal_engine.py:3681` — shadow registry import
- `btc_etf_flow.py:106` — flow data load
- `crypto_precompute.py:186,196,216` — data parsing (3x)
- `loop_contract.py:941` — contract state cleanup
- `grid_fisher.py:1542` — order state cleanup
- `gold_overnight_bias.py:54` — data load
- `intraday_seasonality.py:89` — data load
- `claude_gate.py:159` — process state
**Fix:** Add `logger.debug()` to each. Keep fail-safe behavior.

### BUG-C: Accuracy-cache-coupled test flakes (P2)
**File:** `tests/test_consensus.py` — 7 tests
**Problem:** Tests read production `data/accuracy_cache.json`. When accuracy
values drift past the 47% gate, assertions about vote counts go stale.
**Fix:** Mock `get_or_compute_accuracy` in affected tests.

---

## 2. Dead Code & Cleanup

### DEAD-1: Per-ticker consensus gate for removed tickers
**File:** `signal_engine.py` ~line 788
**Problem:** Gate entries only apply to AMD, GOOGL, META — tickers removed
from Tier 1 months ago. Evaluates every cycle but never triggers.
**Fix:** Remove stale entries. Keep gate mechanism.

---

## 3. Documentation Updates

### DOC-1: SYSTEM_OVERVIEW.md refresh
Stale numbers: signal count (66→69), active list (crypto_evrp disabled),
line counts, test stats.

### DOC-2: IMPROVEMENT_BACKLOG.md additions
Add new findings: ARCH-21 (dead gate entries), BUG-A (executor leak),
BUG-C (test flakes).

---

## 4. Execution Batches

### Batch 1: Escalation gate + silent-except batch 1 (5 files)
1. `portfolio/escalation_gate.py` — singleton executor
2. `portfolio/trigger.py:55` — add logging
3. `portfolio/signals/btc_etf_flow.py:106` — add logging
4. `portfolio/crypto_precompute.py:186,196,216` — add logging (3 sites)
5. `portfolio/loop_contract.py:941` — add logging

### Batch 2: Silent-except batch 2 + dead code (5 files)
1. `portfolio/signal_engine.py:3681` — add logging + remove dead gate
2. `portfolio/grid_fisher.py:1542` — add logging
3. `portfolio/signals/gold_overnight_bias.py:54` — add logging
4. `portfolio/signals/intraday_seasonality.py:89` — add logging
5. `portfolio/claude_gate.py:159` — add logging

### Batch 3: Test stabilization
1. `tests/test_consensus.py` — mock accuracy for flaky tests

### Batch 4: Documentation
1. `docs/SYSTEM_OVERVIEW.md` — full refresh
2. `docs/IMPROVEMENT_BACKLOG.md` — add new items

---

## Risk Assessment

**Batch 1-2:** Adding logging = additive only. Executor change = same
semantics, fewer resources. Fail-open contract preserved. **Low risk.**

**Batch 3:** Test-only changes. No production code. **Zero risk.**

**Batch 4:** Documentation. **Zero risk.**
