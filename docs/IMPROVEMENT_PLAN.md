# Improvement Plan — Auto-Session 2026-03-31

Updated: 2026-03-31
Branch: improve/auto-session-2026-03-31

## 1. Bugs & Problems Found

### P1 — Critical (affects correctness or test reliability)

#### BUG-157: `analyze.py:434` — Loop variable capture in closure (B023)
- **File**: `portfolio/analyze.py:434`
- **Problem**: Inner function `_vote_str(name)` references `votes` from an enclosing
  loop scope. If `_vote_str` is called after the loop variable changes, it will use
  the wrong `votes` dict. Currently not triggered because the function is used
  immediately, but fragile — any refactoring that defers the call will silently break.
- **Impact**: Latent bug — currently works but violates Python closure semantics safety.
- **Fix**: Bind `votes` as a default parameter: `def _vote_str(name, votes=votes):`

#### BUG-158: `test_signal_improvements.py:402,411` — Undefined `datetime` (F821)
- **File**: `tests/test_signal_improvements.py:402, 411`
- **Problem**: Two test methods reference `datetime` in a lambda but the name is not
  imported or available in the local scope (it's been patched by `@patch`). These tests
  would fail at runtime if the code path is reached.
- **Impact**: Tests may pass coincidentally due to mock setup, but contain undefined
  name references that ruff flags as F821.
- **Fix**: Add `from datetime import datetime` import or bind the real datetime before patch.

### P2 — Important (code quality, resource leaks)

#### BUG-159: `avanza_session.py:255` — `raise` without `from` (B904)
- **File**: `portfolio/avanza_session.py:255`
- **Problem**: `raise RuntimeError(...)` inside an `except` clause doesn't chain
  the original exception. The original JSON decode error is lost.
- **Fix**: `raise RuntimeError(...) from None` (intentional suppression) or `from err`.

#### BUG-160: `avanza_session.py:331-334` — Bare `pass` exception handlers
- **File**: `portfolio/avanza_session.py:331-334`
- **Problem**: Two consecutive `except: pass` handlers with no logging. If shutdown
  cleanup fails, there's zero observability.
- **Fix**: Add `logger.debug()` calls for observability.

#### REF-21: 4 unused imports in portfolio/ (F401)
- **Files**: `avanza/types.py:12` (`Sequence`), `avanza_control.py:11` (`Any`),
  `ministral_signal.py:10` (`subprocess`), `oil_precompute.py:12` (`json`)
- **Fix**: Remove unused imports.

#### REF-22: 3 unused variables in portfolio/ (F841)
- **Files**: `crypto_scheduler.py:119` (`forecast`), `crypto_scheduler.py:302` (`gp`),
  `fin_fish.py:749` (`warrant_price_now`)
- **Fix**: Remove or prefix with `_`.

#### REF-23: 2 f-strings without placeholders (F541)
- **Files**: `meta_learner.py:252, 300`
- **Fix**: Remove extraneous `f` prefix.

### P3 — Minor (lint, style, consistency)

#### REF-24: 7 unsorted imports (I001) in portfolio/
- **Fix**: `ruff check --fix --select I001`

#### REF-25: 11 non-PEP604 Optional annotations (UP045)
- **Fix**: `ruff check --fix --select UP045`

#### REF-26: 3 datetime.timezone.utc → datetime.UTC (UP017)
- **Fix**: `ruff check --fix --select UP017`

#### REF-27: 3 deprecated imports (UP035)
- **Fix**: `ruff check --fix --select UP035`

#### REF-28: 2 redundant open modes (UP015)
- **Fix**: `ruff check --fix --select UP015`

#### REF-29: Unregistered `slow` pytest mark
- **File**: `pyproject.toml`
- **Problem**: 6 tests use `@pytest.mark.slow` but it's not registered, generating warnings.
- **Fix**: Add `"slow: marks tests that take a long time to run"` to `markers` list.

#### REF-30: pyproject.toml description says "29-signal" (should be 30)
- **File**: `pyproject.toml:4`
- **Fix**: Update to "30-signal".

#### REF-31: Test lint cleanup — 78 unused imports, 65 unused variables
- **Files**: Various test files
- **Fix**: `ruff check --fix --select F401,I001` for auto-fixable; manual review for F841.

---

## 2. Architecture Improvements

### ARCH-29: Avanza package migration — wire new package into metals_loop.py
- **Status**: The new `portfolio.avanza` package exists with 10 modules and full test
  coverage (2,351+ tests in `tests/test_avanza_pkg/`), but nothing in the codebase
  actually imports from it yet. The old `avanza_session.py` + `avanza_client.py` +
  `avanza_orders.py` are still the active code paths.
- **Risk**: Too high for autonomous session — touches live trading code, needs manual
  review and staged rollout.
- **Decision**: **DEFERRED** — document in plan, don't implement.

### ARCH-30: `SIM105` — Replace 14 `try/except/pass` with `contextlib.suppress`
- **Files**: Across `streaming.py`, `avanza_orders.py`, `exit_optimizer.py`,
  `equity_curve.py` (5 instances), `accuracy_stats.py`, `daily_digest.py`,
  `avanza_session.py`, `bigbet.py`
- **Impact**: Cleaner code, fewer lines, same behavior. Some handlers need investigation
  first — equity_curve.py has 5 bare `pass` handlers that may need logging.
- **Decision**: Implement for clear cases; add logging for currently-silent handlers.

---

## 3. Implementation Batches

### Batch 1: Ruff auto-fixes (portfolio/) — SAFE, auto-fixable
**Scope**: F401, F541, I001, UP045, UP017, UP035, UP015 in portfolio/
**Files**: ~15 files modified
**Risk**: Zero — all auto-fixable, no behavioral change
**Tests**: Run full suite after

### Batch 2: Manual bug fixes (portfolio/)
**Scope**: BUG-157 (B023), BUG-159 (B904), BUG-160 (silent pass), REF-22 (F841)
**Files**: `analyze.py`, `avanza_session.py`, `crypto_scheduler.py`, `fin_fish.py`
**Risk**: Low — localized fixes
**Tests**: Run full suite after

### Batch 3: SIM105 contextlib.suppress conversions + equity_curve pass audit
**Scope**: ARCH-30
**Files**: ~8 files
**Risk**: Low — behavioral equivalence guaranteed by ruff
**Tests**: Run full suite after

### Batch 4: Test fixes and improvements
**Scope**: BUG-158 (F821 undefined datetime), REF-29 (slow mark), REF-30 (description),
  REF-31 (test lint cleanup — auto-fixable F401/I001 only)
**Files**: `test_signal_improvements.py`, `pyproject.toml`, various test files
**Risk**: Zero — test-only changes
**Tests**: Run full suite after

---

## 4. Deferred Items (from prior sessions)

- **ARCH-17**: main.py re-exports 100+ symbols (breaking change risk)
- **ARCH-18**: metals_loop.py 4465-line monolith (risks live trading)
- **ARCH-19**: No CI/CD pipeline (needs GitHub Actions + Windows runner)
- **ARCH-20**: No type checking/mypy (incremental adoption)
- **ARCH-21**: autonomous.py function decomposition (stable, low ROI)
- **ARCH-22**: agent_invocation.py class extraction (touches every caller)
- **ARCH-29**: Avanza package migration (needs manual staged rollout)
- **BUG-121**: news_event.py sector mapping hardcoded (low value)
- **BUG-132**: orb_predictor.py no caching (low priority)
- **BUG-149**: meta_learner orphaned — predict() never called
- **TEST-1**: gpu_gate.py zero test coverage (requires GPU mocking)
- **TEST-3**: 26 pre-existing test failures (integration, config)
- **FEAT-3**: Integrate meta_learner as signal #31

---

## 5. Dependency & Ordering

```
Batch 1 (ruff auto-fixes) → no dependencies, do first
Batch 2 (manual bug fixes) → independent of Batch 1
Batch 3 (SIM105 conversions) → after Batch 1 (imports may shift)
Batch 4 (test fixes) → after Batch 1-3 (ensures clean test run)

Run full test suite after each batch.
```

### Risk Summary

| Batch | Files Changed | Production Risk | Test Risk |
|-------|--------------|-----------------|-----------|
| 1 | ~15 (modify) | Zero — auto-fix only | Zero |
| 2 | 4 (modify) | Low — localized bug fixes | Low |
| 3 | ~8 (modify) | Low — behavioral equivalence | Low |
| 4 | ~20+ (modify) | Zero — test files + config | Low |
