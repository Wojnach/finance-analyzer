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
