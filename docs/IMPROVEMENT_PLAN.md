# Improvement Plan — Auto-Session 2026-04-07

Updated: 2026-04-07
Branch: improve/auto-session-2026-04-07
Status: Planning complete, ready for implementation

**Source:** Deep exploration of 198 portfolio modules, 212+ test files, ruff analysis,
and pattern review. Previous sessions fixed BUG-80 through BUG-170 + REF-16 through REF-44.
This session addresses remaining code quality, silent exceptions, and lint compliance.

---

## 1. Bugs & Problems Found

### BUG-171 (P2): ~14 remaining `except Exception: pass` silent swallowers
- **Files**: agent_invocation.py:529, earnings_calendar.py:128, fin_fish.py:1229/1390/1409,
  fish_instrument_finder.py:77, fish_monitor_smart.py:148/243, llama_server.py:154/179,
  avanza/scanner.py:69/279, golddigger/runner.py:359, elongir/runner.py:234, ministral_signal.py:97
- **Problem**: Silent exception swallowing hides bugs and makes debugging impossible.
  Some are in cleanup/teardown code where suppression is appropriate, but several
  are in operational paths where failures should at least be logged.
- **Fix**: Convert cleanup-path `except Exception: pass` to `contextlib.suppress(Exception)`.
  For operational paths, add `logger.debug()` so failures are traceable.
- **Risk**: Zero — only changes error visibility, not behavior.

### BUG-172 (P3): `fin_fish.py:1226` uses `datetime.timezone.utc` instead of `datetime.UTC`
- **File**: `portfolio/fin_fish.py:1226`
- **Problem**: Uses deprecated `_dt.timezone.utc` instead of `_dt.UTC` (Python 3.11+).
- **Fix**: Replace with `_dt.UTC` (UP017).
- **Risk**: Zero.

### BUG-173 (P3): `strategies/orchestrator.py:7` imports from `typing` instead of `collections.abc`
- **File**: `portfolio/strategies/orchestrator.py:7`
- **Problem**: `from typing import Callable` — deprecated, should be `from collections.abc import Callable`.
- **Fix**: UP035 auto-fix.
- **Risk**: Zero.

### BUG-174 (P3): Unused import `Path` in `strategies/golddigger_strategy.py:12`
- **File**: `portfolio/strategies/golddigger_strategy.py:12`
- **Problem**: `pathlib.Path` imported but unused (F401).
- **Fix**: Remove unused import.
- **Risk**: Zero.

### BUG-175 (P3): Unsorted imports in `strategies/golddigger_strategy.py:7`
- **File**: `portfolio/strategies/golddigger_strategy.py:7`
- **Problem**: Import block not sorted per isort (I001).
- **Fix**: Auto-sort.
- **Risk**: Zero.

---

## 2. Refactoring TODOs

### REF-45: 9 collapsible nested `if` statements (SIM102)
- **Files**: accuracy_stats.py:851, autonomous.py:323, crypto_macro_data.py:113,
  daily_digest.py:78, journal.py:249, prophecy.py:251/253, risk_management.py:637,
  warrant_portfolio.py:237
- **Problem**: Nested `if` statements that can be combined with `and`.
- **Fix**: Combine nested `if` into single `if X and Y:` where readability allows.
  Skip if the combined condition becomes unreadable.
- **Risk**: Zero — logic-preserving simplification.

### REF-46: 3 SIM114 (if-with-same-arms) in indicators.py, crypto_macro.py
- **Files**: indicators.py:161, crypto_macro.py:150/154
- **Problem**: Adjacent `if`/`elif` branches with identical bodies.
- **Fix**: Combine conditions with `or` operator.
- **Risk**: Zero.

### REF-47: 2 SIM105 (suppressible-exception) in bot runners
- **Files**: elongir/runner.py:231, golddigger/runner.py:356
- **Problem**: `try/except/pass` that should be `contextlib.suppress(Exception)`.
- **Fix**: Use contextlib.suppress.
- **Risk**: Zero.

### REF-48: 253 ruff violations in test files (84 auto-fixable)
- **Breakdown**: 69 F841 (unused vars), 50 F401 (unused imports), 42 SIM117 (multi-with),
  35 E741 (ambiguous vars), 21 I001 (unsorted), 8 UP017, 4 SIM300, others.
- **Fix**: Auto-fix the 84 safe fixes (F401, I001, UP017, SIM300). Manual review F841.
- **Risk**: Zero for auto-fixes. Low for manual F841 (test assertions may use vars).

---

## 3. Implementation Order & Dependencies

```
Batch 1 (Portfolio code lint)    → No dependencies, do first
  BUG-172, BUG-173, BUG-174, BUG-175  → fin_fish.py, strategies/*.py
  REF-46                                → indicators.py, crypto_macro.py
  REF-47                                → elongir/runner.py, golddigger/runner.py

Batch 2 (Silent exceptions)     → Independent of Batch 1
  BUG-171                              → 14 files with except/pass patterns

Batch 3 (Collapsible ifs)       → Independent
  REF-45                                → 8 files with SIM102

Batch 4 (Test cleanup)          → Independent
  REF-48                                → tests/*.py auto-fix + manual F841

Batch 5 (Remaining ruff auto-fix) → After Batch 1-4
  Run `ruff check --fix` on full codebase for any remaining safe fixes
```

### Risk Summary

| Batch | Files Changed | Production Risk | Test Risk |
|-------|--------------|-----------------|-----------|
| 1 | 5 (fin_fish, strategies/*, indicators, crypto_macro, runners) | Zero | Zero |
| 2 | ~14 (silent exception files) | Zero | Zero |
| 3 | 8 (SIM102 files) | Zero | Zero |
| 4 | ~50+ (test files) | Zero | Low |
| 5 | Variable | Zero | Zero |

---

## 4. Deferred Items (NOT in this session)

### Previously deferred (still valid)
- **ARCH-17**: main.py ~120 re-exports — too many consumers, needs gradual migration
- **ARCH-18**: metals_loop.py 4,553-line monolith — too risky for autonomous session
- **ARCH-19**: CI/CD pipeline — requires infrastructure decisions
- **ARCH-20**: mypy type checking — large effort, separate initiative
- **BUG-132**: orb_predictor.py uncached 5000-candle fetch — performance, not correctness
- **BUG-162**: metals_loop.py high bug density — coupled with ARCH-18
- **E402**: 51 module-import-not-at-top violations — most are intentional lazy imports
- **SIM115**: 5 open-file-with-context-handler — 3 are intentional (subprocess log handles)
- **F841 in tests**: 69 unused variables — many are intentional (assert side effects)
- **E741 in tests**: 35 ambiguous variable names — `l`, `O` etc. are common in test data
