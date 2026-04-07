# Fix: Fishing Fill Probability Underestimation

## Date: 2026-04-07

## Problem

`fin_fish.py:_compute_vol_and_drift` feeds daily-range-derived volatility into
`volatility_from_atr()` which annualizes assuming hourly ATR candles. This
underestimates annual vol by 3.7x (sqrt(14)), making fill probabilities
~10x too low. Result: only 1 fishing level passes the 5% min-fill filter
instead of 5+, making the fishing tool nearly useless.

## Root Cause

`_compute_vol_and_drift` computes `realized_atr_pct = avg_daily_range / 1.5`
(a daily sigma estimate) then calls `volatility_from_atr(realized_atr_pct)`
which does `atr_frac * sqrt(252/14)`. For a daily input, the correct
annualization is `daily_sigma * sqrt(252)`, not `daily_sigma * sqrt(252/14)`.

The `/14` factor is correct for the hourly ATR path (signal["atr_pct"]),
but wrong for the daily range path.

## Fix Plan (2 files changed, 1 test file added)

### Fix 1: Correct vol in `_compute_vol_and_drift` (portfolio/fin_fish.py:326-365)

- Keep the hourly ATR path through `volatility_from_atr()` unchanged
- Add separate daily-range annualization: `daily_sigma * sqrt(252)`
- Take the max of both (existing logic, just with correct math)

### Fix 2: Best-warrant-per-level in report (portfolio/fin_fish.py:1026-1083)

- In `format_report`, after getting `warrant_results`, deduplicate by level
  (keep highest-EV warrant per unique price level). Shows different PRICE
  LEVELS in the table instead of repeating one level across 6 instruments.

### Fix 3: Dynamic leverage computation (portfolio/fin_fish.py:691-822)

- In `evaluate_warrants`, compute live leverage from spot and barrier
  for MINI/TURBO warrants. Daily certs keep config leverage.

### Non-goals

- NOT touching `volatility_from_atr()` — 4 other callers depend on it
- NOT touching `fill_probability_buy()` — GBM math is correct
- NOT adding instruments to catalog — separate task

### Tests

- `tests/test_fin_fish_vol.py`:
  - `_compute_vol_and_drift` returns correct annual vol from daily ranges
  - `compute_fishing_levels_bull` generates 5+ levels passing 5% filter
  - `evaluate_warrants` returns distinct levels
  - Dynamic leverage matches `spot / (spot - barrier)`

### Risk

- Other callers of fin_fish get higher fill probs -> more levels shown.
  Correct behavior. FISHING_ENABLED=False is the safety gate.
- `monte_carlo.py:volatility_from_atr` NOT changed -> zero blast radius.
