# Session Progress — Microstructure Research Upgrade 2026-04-07

**Date:** 2026-04-07
**Branch:** `research/microstructure-upgrade`
**Source:** Quantitative signals research paper on 1-3h metals forecasting

## Completed

### Batch 1: Quick Wins (4 changes, all tests passing)
1. **Oil fetch**: Added `get_oil_data()` to metals_cross_assets.py — CL=F via yfinance
2. **GVZ active voter**: High GVZ = BUY gold/SELL silver, low = inverse
3. **OFI z-score**: Rolling OFI distribution per asset, ±1.5σ thresholds
4. **Multi-scale OFI**: Fast/medium/slow windows + flow acceleration metric
5. **VPIN toxicity flag**: VPIN > 0.7 → high_toxicity for risk management

### Batch 2: Feature Preprocessing (4 changes)
1. **feature_normalizer.py**: Rolling z-score normalization, 100-period window
2. **Vol-scaled ROC-20**: z_roc = ROC/sigma_20, ±1.5σ thresholds
3. **Seasonality integration**: mean_reversion + momentum_factors accept context
4. **Signal registry update**: Both signals now requires_context=True

### Batch 3: Accuracy & Correlation (3 changes)
1. **Cost-aware accuracy**: signal_accuracy_cost_adjusted(), 5bps default
2. **Return magnitude**: Already existed as signal_utility() — no changes needed
3. **Dynamic correlation groups**: Signal vote correlation matrix, 30-day rolling

## Test Status
- Batch 1: 43 tests passing (all new + existing)
- Batch 2+3 tests: being written by subagent

## Commits
1. `154885d` — docs: plan
2. `476b080` — feat: batch 1
3. `4f5e0c7` — feat: batch 2
4. `8e92878` — feat: batch 3
5. `ebf7934` — test: batch 1 tests

## Remaining
- Batch 2+3 test commit
- Full test suite run
- Rebase against main (parallel changes may exist)
- Codex adversarial review
- Merge into main + push
