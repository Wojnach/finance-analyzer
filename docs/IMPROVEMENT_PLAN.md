# Improvement Plan — Auto-Session 2026-03-24

Updated: 2026-03-24
Branch: improve/auto-session-2026-03-24

## 1. Bugs & Problems Found

### P1 — Critical (affects trading decisions or system reliability)

#### BUG-115: `structure.py` signal swallows exceptions silently
- **File**: `portfolio/signals/structure.py:217-235`
- **Problem**: All 4 sub-indicator try/except blocks catch bare `Exception` with no logging. No `import logging` or logger in the file. When a sub-indicator crashes, it silently falls back to HOLD with zero visibility.
- **Impact**: Trading decisions based on incomplete signal data. Systematic failures in structure signal invisible for days.
- **Fix**: Add logger + `logger.exception()` in each except block.

#### BUG-119: Layer 2 tier config mismatch
- **File**: `portfolio/agent_invocation.py:42`, `CLAUDE.md`
- **Problem**: `TIER_CONFIG[2]` has `max_turns: 40`, but `CLAUDE.md` documents T2 as "25 turns".
- **Impact**: T2 sessions run longer than documented, or docs mislead Layer 2 agents.
- **Fix**: Align CLAUDE.md with code (40 turns is the actual runtime value).

### P2 — Important (could cause incorrect behavior)

#### BUG-116: Trigger state pruning silently drops tickers
- **File**: `portfolio/trigger.py:52-59`
- **Problem**: When tickers are removed from tracking, their trigger state entries are pruned without logging. Temporary API failures wipe baselines, causing spurious triggers on return.
- **Impact**: Potential spurious Layer 2 invocations after temporary API outages.
- **Fix**: Add `logger.info()` when pruning entries.

#### BUG-117: FX rate hardcoded fallback may be stale
- **File**: `portfolio/fx_rates.py:48`
- **Problem**: Fallback rate is `10.85` SEK/USD. If SEK moves to 9.x or 12.x, portfolio valuations could be off by 10-15%.
- **Impact**: Portfolio P&L display could be significantly wrong in extreme FX movements.
- **Fix**: Add bounds check — warn at ERROR level when using hardcoded fallback, and validate cached rate is within reasonable range (8-15 SEK).

#### BUG-120: `_safe_series()` forward-fills silently
- **File**: `portfolio/signals/volume_flow.py`
- **Problem**: `_safe_series()` replaces inf/-inf with NaN and forward-fills. Forward-filling volume data creates fictitious volume levels.
- **Impact**: Volume Flow signal could produce false BUY/SELL votes based on stale forward-filled volume.
- **Fix**: Add logging when forward-fill is applied to >5% of the series.

### P3 — Minor (code quality, future maintenance)

#### BUG-118: FOMC/econ dates hardcoded for 2026-2027
- **Files**: `portfolio/signals/econ_calendar.py`, `portfolio/signals/calendar_seasonal.py`, `portfolio/signals/macro_regime.py`
- **Problem**: Economic calendar dates are hardcoded through 2027. After that, these signals silently degrade to HOLD.
- **Impact**: After 2027, three signal modules silently stop contributing. No warning emitted.
- **Fix**: Add a log warning when the latest event date is in the past.

#### BUG-121: news_event.py sector mapping hardcoded
- **File**: `portfolio/signals/news_event.py`
- **Problem**: Sector→ticker representative mapping is a hardcoded dict. Adding a new ticker requires code changes.
- **Impact**: Low — current ticker list is stable.
- **Fix**: Deferred — low value.

---

## 2. Architecture Improvements

### ARCH-17: main.py re-export cleanup — DEFERRED
100+ re-exports from submodules. Breaking change risk too high without full caller audit.

### ARCH-18: metals_loop.py split — DEFERRED
1000+ line monolith. Risks destabilizing live metals trading. Requires dedicated session.

### ARCH-19: No CI/CD — DEFERRED
Out of scope for code improvement session. Needs GitHub Actions + Windows runner setup.

### ARCH-20: No type checking — DEFERRED
Adding mypy would require type annotations across 142 modules. Incremental adoption not worth the session time.

---

## 3. Improvements to Implement

### Batch 1: Signal module logging & safety (5 files)
**Priority**: High — directly improves signal reliability visibility.

| # | Change | File | Bug |
|---|--------|------|-----|
| 1 | Add logging to exception handlers | `portfolio/signals/structure.py` | BUG-115 |
| 2 | Add forward-fill count warnings | `portfolio/signals/volume_flow.py` | BUG-120 |
| 3 | Add event staleness warning | `portfolio/signals/econ_calendar.py` | BUG-118 |
| 4 | Add event staleness warning | `portfolio/signals/calendar_seasonal.py` | BUG-118 |
| 5 | Add event staleness warning | `portfolio/signals/macro_regime.py` | BUG-118 |

**Impact**: Signal modules are isolated pure functions. Changes cannot break other signals or the main loop.

### Batch 2: Trigger, FX & documentation fixes (3 files)
**Priority**: High — reduces noise, improves accuracy.

| # | Change | File | Bug |
|---|--------|------|-----|
| 1 | Add logging to trigger state pruning | `portfolio/trigger.py` | BUG-116 |
| 2 | Add FX rate bounds validation | `portfolio/fx_rates.py` | BUG-117 |
| 3 | Fix T2 turns documentation mismatch | `CLAUDE.md` | BUG-119 |

**Impact**: trigger.py affects Layer 2 invocation frequency. FX rate affects portfolio display only.

### Batch 3: Test coverage for under-tested modules (test files)
**Priority**: Medium — improves confidence in infrastructure.

| # | Change | File | Bug |
|---|--------|------|-----|
| 1 | Add tests for `health.py` staleness + silence detection | `tests/test_health_extended.py` | TEST-2 |
| 2 | Add tests for `structure.py` logging behavior | `tests/test_structure_logging.py` | BUG-115 verify |

**Impact**: Test-only changes. No production code affected.

---

## 4. Refactoring TODOs (Deferred)

- **REF-18**: Standardize signal exception handling with a decorator — low value, merge conflict risk.
- **REF-19**: Extract `_MAX_CONFIDENCE = 0.7` to shared constant — already enforced by registry.

---

## 5. Dependency & Ordering

Batches 1 and 2 are independent — can be implemented in parallel.
Batch 3 depends on Batch 1 (tests verify new logging behavior).

```
Batch 1 (signal logging)  ──┐
                             ├──→ Batch 3 (tests) ──→ Final verification
Batch 2 (trigger/fx/docs) ──┘
```

### Risk Summary

| Batch | Files Changed | Production Risk | Test Risk |
|-------|--------------|-----------------|-----------|
| 1 | 5 signal modules | Low — isolated, additive logging only | Low — existing tests unaffected |
| 2 | 3 files (trigger, fx, docs) | Medium — trigger affects L2 frequency | Low — additive changes |
| 3 | 2 test files (new) | None — test-only | None |
