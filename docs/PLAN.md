# PLAN — Implement btc_gold_correlation_regime Signal

**Date:** 2026-05-25
**Type:** New signal module (starts disabled/shadow)
**Branch:** `feat/btc-gold-corr-regime`

## Objective

Implement `btc_gold_correlation_regime` — a cross-asset intermarket signal measuring
the 30-day rolling Pearson correlation between BTC and Gold daily returns. Extreme
negative correlation (z < -2.0) historically precedes BTC rallies; high positive
correlation (z > 1.5) precedes mean-reversion. Composite score 8.05/10.

## Signal Spec

- **Formula**: `corr_30d = pearson(btc_returns[-30:], gold_returns[-30:])`.
  `z = (corr_30d - rolling_mean_252d) / rolling_std_252d`.
  BUY BTC when z < -2.0. SELL when z > 1.5.
  For XAU-USD: inverse (BUY when z > 1.5, SELL when z < -2.0).
- **Parameters**: corr_window=30, z_lookback=252, buy_z=-2.0, sell_z=1.5
- **Data**: OHLCV only — BTC via `binance_klines("BTCUSDT")`, XAU via `binance_fapi_klines("XAUUSDT")`
- **Target assets**: BTC-USD, XAU-USD (cross-asset)
- **Category**: intermarket
- **Max confidence**: 0.7 (external counterpart data)
- **Starts**: DISABLED (shadow mode)

## Files to Create/Modify

| # | File | Action |
|---|------|--------|
| 1 | `portfolio/signals/btc_gold_correlation_regime.py` | CREATE ~100 lines |
| 2 | `portfolio/signal_registry.py` | ADD register_enhanced() call |
| 3 | `portfolio/tickers.py` | ADD to SIGNAL_NAMES + DISABLED_SIGNALS |
| 4 | `tests/test_signal_btc_gold_correlation_regime.py` | CREATE ~120 lines |

## Implementation Details

### Signal Module (`portfolio/signals/btc_gold_correlation_regime.py`)

Pattern follows `metals_cross_asset.py` for cross-asset data fetching:
- Internal fetch of counterpart asset data (BTC fetches XAU, XAU fetches BTC)
- Thread-safe cache with TTL (4h, matching metals_cross_asset pattern)
- `_compute_zscore()` helper
- `compute_btc_gold_correlation_regime_signal(df, context)` main function
- Returns `{"action": ..., "confidence": ..., "sub_signals": {...}, "indicators": {...}}`
- HOLD when insufficient data (<252 bars) or fetch failure

### Registration

```python
register_enhanced(
    "btc_gold_correlation_regime",
    "portfolio.signals.btc_gold_correlation_regime",
    "compute_btc_gold_correlation_regime_signal",
    requires_context=True,
    max_confidence=0.7,
)
```

### tickers.py Changes

- Add `"btc_gold_correlation_regime"` to `SIGNAL_NAMES` list
- Add `"btc_gold_correlation_regime"` to `DISABLED_SIGNALS` set with comment

### Tests

- Valid BUY (z < -2.0), SELL (z > 1.5), HOLD (neutral z)
- Insufficient data → HOLD
- Counterpart fetch failure → HOLD
- XAU inverse logic
- Confidence capped at 0.7
- Output schema validation

## Execution Order

1. Commit PLAN.md
2. Spawn premortem agent
3. Create worktree `Q:/fa-btc-gold-wt` on branch `feat/btc-gold-corr-regime`
4. Implement signal module
5. Register + tickers.py
6. Tests
7. Run pytest
8. Adversarial review subagent
9. Fix findings
10. Full test suite
11. Merge to main, push
12. Clean up worktree
13. Update progress tracking files

## Premortem

### F1: Applicable-count tripwire tests break
Adding to SIGNAL_NAMES while NOT in DISABLED_SIGNALS (or later promotion) breaks
hardcoded count assertions in `test_consensus.py` and `test_metals.py`.
**Hook:** Verify signal is in DISABLED_SIGNALS. Update tripwire counts if needed.

### F2: Lock contention on dual counterpart fetches
BTC and XAU processed in parallel — both fetch each other's data. Module-level lock
around fetch = contention. metals_cross_asset uses `_fred_cache_lock` but FRED is
fast; Binance can be slow.
**Hook:** Use `shared_state._cached()` with per-ticker cache keys instead of custom lock.
No lock held across network calls.

### F3: Layer 2 double-counts correlated signals
New signal + metals_cross_asset share gold-price common factor. Not in
`_STATIC_CORRELATION_GROUPS` = phantom independent confirmation.
**Hook:** Add to correlation group with metals_cross_asset, or document decision
to leave unclustered with measured correlation evidence.

### F4: Weekend gold zero-returns bias correlation toward 0
Gold flat on weekends, BTC trades 24/7. Dilutes correlation, can push z < -2.0
spuriously. Phantom BUY accumulates false accuracy on weekend rises.
**Hook:** Count zero-return days in gold series. If >25% in 30d window, emit HOLD.
Add `stale_data_ratio` indicator.

### F5: NaN z-scores from insufficient history / timestamp misalignment
Binance XAU FAPI may have <252 days history. NaN z-score → HOLD (safe) but NaN
in indicators dict → `json.dumps(NaN)` = invalid JSON downstream.
Also: array-position alignment vs timestamp alignment bug.
**Hook:** Guard z-score computation with length check. Align on datetime index
(merge/join), not array position. Replace NaN with 0.0 before returning.

### F6: Signal in SIGNAL_NAMES but not registered in _register_defaults()
Silently never runs. No error. No shadow tracking. Can sit dead for weeks.
**Hook:** Add registration. Verify in test that signal name appears in
`get_enhanced_signals()`.

### F7: 4h cache stale during macro events — ACCEPT
Matches metals_cross_asset TTL precedent. Signal starts disabled; stale-cache
artifacts will degrade shadow accuracy, caught by 47% gate before promotion.
TODO: add to MACRO_WINDOW_DOWNWEIGHT_SIGNALS when promoted.
