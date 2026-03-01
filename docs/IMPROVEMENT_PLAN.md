# Improvement Plan — Auto-Session 2026-03-01

## Priority: Critical Bugs > Architecture > Features > Polish

---

## Session #3 (2026-03-01): Signal Correctness Bugs

### BUG-18: momentum.py — Two sub-signals silently dead (UnboundLocalError)
- **File:** `portfolio/signals/momentum.py` lines 46, 127
- **Severity:** CRITICAL (2 of 8 sub-signals permanently broken)
- **Issue:** `rsi = rsi(close)` shadows the imported `rsi` function from `signal_utils`. Python marks `rsi` as a local variable at compile time, so the RHS `rsi(close)` tries to reference the unassigned local → `UnboundLocalError`. Same pattern on line 127 in `_stochasticrsi()`.
- **Impact:** `_rsi_divergence()` and `_stochasticrsi()` crash on every call. The exception is caught by the composite signal's try/except → silently skipped. The momentum composite has been running with only 6 of 8 sub-signals for its entire lifetime.
- **Verified:** `python -c "from portfolio.signals.momentum import _rsi_divergence; ..."` → `UnboundLocalError`
- **Fix:** Rename local variable: `rsi_values = rsi(close)` (line 46) and `rsi_vals = rsi(close, period)` (line 127). Update all subsequent references in each function.

### BUG-19: trend.py — NaN guard using `is np.nan` (never catches NaN)
- **File:** `portfolio/signals/trend.py` line 45
- **Severity:** MEDIUM (broken guard, defaults to correct behavior by accident)
- **Issue:** `sma50.iloc[-1] is np.nan` always returns `False` because pandas NaN values are `float64` objects, not the `np.nan` singleton. The guard never triggers.
- **Impact:** When SMA data is NaN (insufficient data), the function proceeds to compare NaN values. `np.nan > np.nan` is `False`, so no golden/death cross is detected → defaults to HOLD. Functionally correct by accident, but fragile.
- **Verified:** `sma.iloc[-1] is np.nan` → `False`, `pd.isna(sma.iloc[-1])` → `True`
- **Fix:** Change to `pd.isna(sma50.iloc[-1]) or pd.isna(sma200.iloc[-1])`.

### BUG-20: reporting.py — focus_tickers variable scope fragility
- **File:** `portfolio/reporting.py` lines 319-341
- **Severity:** LOW (handled by existing NameError catch, but fragile)
- **Issue:** `focus_tickers` is defined inside a try-block but referenced in a later try-block. If the first try fails before line 323, NameError occurs. Currently caught by a defensive `except NameError`.
- **Fix:** Initialize `focus_tickers = []` before the try-block.

---

## Implementation Batches

### Batch 5: Fix momentum signal crashes (1 file + tests)
**Files:** `portfolio/signals/momentum.py`, `tests/test_momentum_fix.py` (NEW)
**Changes:** BUG-18
**Steps:**
1. Write tests proving `_rsi_divergence()` and `_stochasticrsi()` crash
2. Fix variable shadowing: `rsi` → `rsi_values`/`rsi_vals`
3. Add `logger.warning()` to sub-signal exception handlers
4. Run tests, verify both sub-signals now produce valid outputs
5. Commit

### Batch 6: Fix trend NaN guard + reporting scope (2 files + tests)
**Files:** `portfolio/signals/trend.py`, `portfolio/reporting.py`, `tests/test_trend_nan_fix.py` (NEW)
**Changes:** BUG-19, BUG-20
**Steps:**
1. Write test proving `is np.nan` fails
2. Fix NaN comparison in trend.py
3. Fix focus_tickers scope in reporting.py
4. Run tests, verify
5. Commit

---

## Items NOT Planned (Justified)

1. **structure.py `_highlow_breakout` logic** — Agent flagged as inverted, but verified correct: breakout strategy (near high = BUY, near low = SELL). Docstring matches code.
2. **shared_state.py cache lock race** — Low severity, single-threaded loop. Not worth complexity.
3. **signal_utils.py majority_vote simplification** — Works correctly, cosmetic only.

---

## Previous Sessions (Completed)

### Session #2 (2026-02-28): Reporting & Infrastructure
- Batch 1: test_avanza_session.py rewrite (31 tests) — commit `2646baa`
- Batch 2: Reporting robustness — 13 silent exceptions → warnings — commit `91b06a8`
- Batch 3: Cache stale factor 5→3, trigger pruning — commit `1187f4e`
- Batch 4: Health module per-module failure tracking — commit `c08378e`

### Session #2.5 (2026-03-01): Forecast Accuracy
- Per-ticker accuracy gating, volatility gate, regime discount — commit `f966dfe`
- Parallel test isolation — commit `64435a6`
- Kronos stdout fix + config enable — commits `5af14e0`, `983f8c0`

---

## Verification

1. `python -m pytest tests/ -n auto --tb=short -q` — all tests pass
2. `python -c "from portfolio.signals.momentum import _rsi_divergence; ..."` — no crash
3. Full existing test suite: no regressions
