# Improvement Plan — Auto-Session #5 (2026-03-02)

## Status: COMPLETE

## Priority: Critical Bugs > Architecture > Tests > Features > Polish

Previous sessions fixed BUG-13 through BUG-36, ARCH-1/2/3/4/6/7, REF-5/6/7/10/11, TEST-1/2/3/4.
This session continues from BUG-37 onward.

---

## 1. Bugs & Problems Found

### BUG-37: equity_curve FIFO fee double-counting on multi-partial sells
- **File:** `portfolio/equity_curve.py:400`
- **Severity:** MEDIUM (inflates round-trip fee metrics)
- **Issue:** Proportional buy-fee allocation uses `buy["remaining_shares"]` as denominator:
  ```python
  buy_fee_share = (buy["fee_sek"] * matched / buy["remaining_shares"])
  ```
  On the FIRST partial match, this is correct (e.g., `fee * 50/100 = 0.5*fee`). But after
  `remaining_shares` is decremented on line 416, the SECOND partial match uses the smaller
  denominator: `fee * 50/50 = 1.0*fee`. Total allocated: 1.5x the actual fee.
- **Fix:** Track the original buy quantity separately. Use it as the fee-allocation denominator
  instead of `remaining_shares`.
- **Impact:** `equity_curve.py` only. Fixes profit factor and round-trip P&L accuracy.

### BUG-38: trigger.py _save_state skips pruning when current_tickers is empty set
- **File:** `portfolio/trigger.py:56`
- **Severity:** LOW (edge case: only when all data sources fail simultaneously)
- **Issue:** `if current_tickers:` evaluates to `False` for an empty set. If a cycle runs with
  zero signals (all APIs down), pruning of removed tickers from `triggered_consensus` is skipped.
  State entries for removed tickers persist until a non-empty cycle happens.
- **Fix:** Change to `if current_tickers is not None:` — explicit None check instead of truthiness.
- **Impact:** `trigger.py` only. Prevents stale ticker state from accumulating.

### BUG-39: risk_management ATR stop price can go negative without warning
- **File:** `portfolio/risk_management.py:181`
- **Severity:** LOW (handled but not logged)
- **Issue:** `stop_price = entry_price * (1 - 2 * atr_pct / 100)`. When `atr_pct > 50`,
  stop price becomes negative. Code handles this (sets `distance_to_stop_pct = inf`) but
  doesn't log the extreme ATR condition.
- **Fix:** Add a floor at 1% of entry price and log a warning when ATR exceeds 50%.
- **Impact:** `risk_management.py` only. Improves debugging visibility.

### BUG-40: _prev_sentiment dict never pruned for removed tickers
- **File:** `portfolio/signal_engine.py:36`
- **Severity:** LOW (dict is small, ~19 entries currently)
- **Issue:** `_prev_sentiment` accumulates entries for every ticker that has ever been
  tracked. After the Mar 1 ticker cleanup (27→19), the dict still holds entries for
  removed tickers (MSTR, BABA, GRRR, etc.). These persist in `sentiment_state.json`.
- **Fix:** Prune `_prev_sentiment` against configured tickers on load.
- **Impact:** `signal_engine.py` only. Keeps state files clean.

### BUG-41: _cross_asset_signals only checks BTC→ETH
- **File:** `portfolio/reporting.py:40-57`
- **Severity:** LOW (missing lead-lag detection for metals)
- **Issue:** `followers = {"ETH-USD": "BTC-USD"}` is hardcoded. No XAU→XAG or semiconductor
  lead-lag pairs. These could provide useful cross-asset context for Layer 2.
- **Fix:** Expand followers dict to include `{"ETH-USD": "BTC-USD", "XAG-USD": "XAU-USD"}`.
- **Impact:** `reporting.py` only. Adds metals cross-asset context.

---

## 2. Architecture Improvements

### ARCH-8: Deduplicate test fixtures across test files
- **Files:** `tests/test_consensus.py:33`, `tests/test_portfolio.py:43`, `tests/test_signal_pipeline.py:37`, `tests/test_signal_improvements.py:68`
- **Why:** 4 test files define their own `make_indicators()` fixture instead of using `conftest.py`. `test_portfolio.py:29` also duplicates `make_candles()`, and `test_signal_pipeline.py:59` duplicates `make_ohlcv_df()`. Maintenance burden: changes to conftest fixtures don't propagate.
- **Change:** Remove duplicate fixtures from test files. Import from conftest.py instead. Where defaults differ (e.g., `close=130.0` vs `close=69000.0`), use the conftest fixture with overrides.
- **Impact:** 4 test files. Reduces duplication, improves consistency.

### ARCH-9: Fix test_kronos_backtest_feb27.py hardcoded file paths
- **File:** `tests/test_kronos_backtest_feb27.py`
- **Why:** 6 writes to hardcoded `data/` paths (lines 337, 367, 395, 423, 451, 512). Breaks under `pytest -n auto` (parallel execution) due to file conflicts between test workers.
- **Change:** Use `tmp_path` fixture for all JSON writes.
- **Impact:** 1 test file. Enables safe parallel execution.

---

## 3. Test Coverage Improvements

### TEST-5: equity_curve FIFO round-trip regression tests
- **File:** `tests/test_equity_curve_fifo.py` (new)
- **Why:** BUG-37 (fee double-counting) needs regression coverage. Also test: multiple partial sells from one buy batch, zero-share edge case, round-trip metrics accuracy, empty transaction list.

### TEST-6: trigger.py edge case tests
- **File:** `tests/test_trigger_pruning.py` (new)
- **Why:** BUG-38 (empty-set pruning). Test: empty signals pruning, removed ticker cleanup, sustained count reset after restart.

### TEST-7: risk_management ATR stop edge case tests
- **File:** augment existing `tests/test_risk_management.py`
- **Why:** BUG-39 (negative stop price). Test: extreme ATR (>50%), zero ATR, normal ATR range.

---

## 4. Refactoring

### REF-12: Simplify _cross_asset_signals config
- **File:** `portfolio/reporting.py:40-57`
- **Why:** Hardcoded `followers` dict should be a module-level constant, not buried in a function.
- **Fix:** Extract to `_CROSS_ASSET_PAIRS = {"ETH-USD": "BTC-USD", "XAG-USD": "XAU-USD"}`.
- **Impact:** `reporting.py` only. Cleaner, easier to extend.

---

## 5. Items NOT Planned (Justified)

1. **Dashboard accuracy endpoint caching** — Underlying `accuracy_stats` functions already cache (1h TTL). Adding endpoint-level caching would add complexity for marginal benefit.
2. **Sentiment I/O on every update** — Hysteresis in `_set_prev_sentiment()` prevents excessive writes. Each cycle only writes if sentiment actually changes (not on every call).
3. **REGIME_WEIGHTS missing "breakout"/"capitulation"** — NOT a bug. `detect_regime()` only returns 4 values (high-vol, trending-up, trending-down, ranging). "breakout" and "capitulation" are Layer 2 journal labels, not Layer 1 regime values.
4. **Untested utility modules** (log_rotation, backup, ml_trainer, migrate_signal_log, etc.) — These are scripts/utilities, not core trading logic. Testing them has low ROI.
5. **FX hardcoded fallback rate 10.85** — Only used when API is down AND no cache exists (cold start scenario). Rate drift (10.85 vs ~10.3) causes ~5% error, acceptable for the rare case. Not worth adding config complexity.

---

## 6. Dependency/Ordering — Implementation Batches

### Batch 1: Bug fixes (5 files, BUG-37 through BUG-41) — DONE
**Files:** `portfolio/equity_curve.py`, `portfolio/trigger.py`, `portfolio/risk_management.py`, `portfolio/signal_engine.py`, `portfolio/reporting.py`
**Changes:** BUG-37, BUG-38, BUG-39, BUG-40, BUG-41
**Tests:** TEST-5 (38 FIFO regression), TEST-6 (11 trigger pruning), TEST-7 (7 ATR stops). All 56 pass.
**Commit:** `ba2d9ad` — fix: 5 bugs + 56 regression tests

### Batch 2: Test infrastructure (ARCH-8, ARCH-9) — DONE
**Files:** `tests/test_consensus.py`, `tests/test_portfolio.py`, `tests/test_signal_pipeline.py`, `tests/test_signal_improvements.py`, `tests/test_kronos_backtest_feb27.py`, `pyproject.toml`
**Changes:** ARCH-8 (deduplicate fixtures — 116 lines removed, 45 added), ARCH-9 (6 hardcoded paths → tmp_path)
**Tests:** All affected tests pass. 1 pre-existing failure in test_portfolio.py (trigger cooldown) unchanged.
**Commit:** `8fd7b02` — refactor: deduplicate test fixtures + fix hardcoded paths

### Batch 3: Refactoring (REF-12) — DONE (merged into Batch 1)
**Files:** `portfolio/reporting.py`
**Changes:** REF-12 (cross-asset constant extracted to `_CROSS_ASSET_PAIRS` module-level dict)
**Note:** Implemented alongside BUG-41 since they modify the same code.
