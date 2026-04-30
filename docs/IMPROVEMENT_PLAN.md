# Improvement Plan — Auto-Session 2026-04-30

## 1. Bugs & Problems Found

### P1 (Critical — production impact)

**BUG-236: `crypto_swing_trader.py:675` — TypeError on naive datetime subtraction**
- `_has_fresh_momentum_candidate()` calls `fromisoformat(ts_str)` without timezone handling.
  `_now_utc()` returns aware datetime. If `ts_str` is naive, subtraction raises `TypeError`.
- The `except (ValueError, AttributeError)` does NOT catch `TypeError`.
- Currently inert (DRY_RUN=True), but will crash once momentum state file is populated.
- **Fix:** Add `TypeError` to the except clause, or normalize timestamp with `.replace("Z", "+00:00")`.
- **Impact:** Low (DRY_RUN), but critical path once live.

**BUG-237: `accuracy_stats.py:1` — unused `json` import (F401)**
- Harmless but causes ruff lint failure. Production module loaded every cycle.
- **Fix:** Remove the unused import.

### P2 (Important — correctness/maintainability)

**BUG-238: `crypto_swing_trader.py:454,471` — fragile timezone handling in exit/cooldown**
- `_evaluate_exit` max-hold check and `_cooldown_cleared` parse timestamps without
  `.replace("Z", "+00:00")`. Currently safe because `_now_iso()` produces `+00:00`
  suffix, but if any external source writes a `Z`-terminated or naive timestamp to
  `crypto_swing_state.json`, these paths will throw `TypeError`.
- **Fix:** Normalize all `fromisoformat()` calls consistently.

**BUG-239: `crypto_loop.py:65-100` — Singleton lock TOCTOU race**
- Check file → read PID → check alive → write PID has a window where two processes
  starting simultaneously could both succeed.
- **Fix:** Use `os.O_CREAT | os.O_EXCL` atomic create (same pattern as `gpu_gate.py`).

**BUG-240: `crypto_swing_trader.py` — ruff violations (I001 + UP035)**
- Unsorted imports and deprecated `typing.Callable` (should be `collections.abc.Callable`).
- **Fix:** Ruff auto-fix.

**BUG-241: `crypto_cross_asset.py:20` — unsorted import block (I001)**
- **Fix:** Ruff auto-fix.

**BUG-242: `crypto_loop.py` — 4x try-except-pass should use contextlib.suppress**
- Lines 110, 192, 206, 257.
- **Fix:** Replace with `contextlib.suppress()`.

### P3 (Nice-to-have — code quality)

**REF-52: `crypto_loop.py:332` — extraneous parentheses (UP034)**
- **Fix:** Ruff auto-fix.

**REF-53: Forecast dedup cache unbounded growth**
- `forecast.py` `_prediction_dedup` dict grows without bound. Has eviction age
  (600s) but never actually evicts old entries.
- **Fix:** Add periodic eviction in `compute_forecast_signal()`.

**REF-54: `trade_guards.py` — loss escalation has no decay**
- Once 4 consecutive losses, multiplier stays at 8x indefinitely. No time-based
  decay or ceiling.
- **Fix:** Add time-based decay (e.g., halve escalation every 24h without a trade).

## 2. Architecture Improvements

None proposed this session. The codebase is architecturally solid after 70+ prior
improvement sessions. The crypto subsystem is newly added but follows established
patterns (metals mirror).

## 3. Useful Features

None proposed this session. The recent crypto+MSTR swing subsystem (2026-04-29)
is still in DRY_RUN validation. Adding features before stabilization would be premature.

## 4. Refactoring TODOs

All items in this section are lint/style cleanups with zero functional impact:

- **REF-55:** Auto-fix 162 fixable ruff violations in tests/ (F401 unused imports, I001 unsorted).
- **REF-56:** Auto-fix 5 fixable ruff violations in portfolio/ + data/ (see BUG-237, 240, 241 above).

## 5. Dependency/Ordering — Batch Plan

### Batch 1: Bug fixes in new crypto subsystem (5 files)
Files: `data/crypto_swing_trader.py`, `data/crypto_loop.py`,
       `portfolio/signals/crypto_cross_asset.py`, `portfolio/accuracy_stats.py`
- Fix BUG-236 (TypeError), BUG-237 (unused import), BUG-238 (timezone),
  BUG-239 (singleton lock), BUG-240/241 (lint), BUG-242 (contextlib.suppress)
- **Risk:** Low. All changes are in new crypto modules (DRY_RUN=True) or trivial lint.

### Batch 2: Forecast dedup eviction + trade guard decay (2 files)
Files: `portfolio/signals/forecast.py`, `portfolio/trade_guards.py`
- Fix REF-53 (dedup eviction), REF-54 (loss escalation decay)
- **Risk:** Medium. Forecast signal is production-active. Trade guards affect
  all trading decisions. Changes must be minimal and well-tested.

### Batch 3: Test file lint cleanup (auto-fix, no behavior change)
Files: All test files with auto-fixable ruff violations.
- Fix REF-55 (162 auto-fixable violations).
- **Risk:** Extremely low. Auto-fix only, no semantic changes.

### Batch 4: Documentation update
Files: `docs/SYSTEM_OVERVIEW.md`, `docs/SESSION_PROGRESS.md`
- Update SYSTEM_OVERVIEW with new crypto subsystem modules.
- Write session progress for next session context.
