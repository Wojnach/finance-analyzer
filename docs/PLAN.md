# Microstructure Research Upgrade Plan

**Date**: 2026-04-07
**Branch**: research/microstructure-upgrade
**Source**: Executive research summary on quantitative signals for 1-3h gold/silver forecasting

## Motivation

A comprehensive research paper on short-term metals forecasting was analyzed against our
existing 32-signal system. Finding: we have almost all the right raw signals but several
calibration bugs, dead sub-signals, and a missing statistical preprocessing layer between
raw data and signal voting.

## Batch 1: Quick Wins (fix broken/dead signals)

### 1a. Oil Fetch in Cross-Asset Signal
- **File**: `portfolio/metals_cross_assets.py`
- **Problem**: `oil_change_5d` hardcoded to `0.0` — dead sub-signal in 5-voter composite
- **Fix**: Add `get_oil_data()` fetching `CL=F` via yfinance, include in `get_all_cross_asset_data()`
- **File**: `portfolio/signals/metals_cross_asset.py`
- **Fix**: Read oil from cross-asset data instead of macro dict fallback

### 1b. GVZ as Active Voter
- **File**: `portfolio/signals/metals_cross_asset.py`
- **Problem**: GVZ sub-signal always votes HOLD (informational only)
- **Fix**: High GVZ (>1.5 z) = BUY gold / SELL silver (fear/safe-haven), Low GVZ (<-1.0 z) = SELL gold / BUY silver (complacency)

### 1c. OFI Z-Score Normalization
- **Files**: `portfolio/microstructure_state.py`, `portfolio/signals/orderbook_flow.py`
- **Problem**: OFI threshold = 5.0 for all assets. Gold vs BTC have different OFI scales
- **Fix**: Track rolling OFI distribution per ticker, use z-score (+-1.5s) instead of absolute threshold

### 1d. VPIN Volatility Predictor
- **File**: `portfolio/signals/orderbook_flow.py`
- **Problem**: VPIN only confirms trade_flow direction — never initiates, never predicts vol
- **Fix**: Add VPIN > 0.7 as `high_toxicity` flag in indicators for risk management consumption

## Batch 2: Feature Preprocessing Pipeline

### 2a. Feature Normalizer Module
- **New file**: `portfolio/feature_normalizer.py`
- **Purpose**: Rolling z-score normalization for signal inputs, regime-adaptive thresholds
- **Design**: Per-ticker, per-indicator rolling stats (mean, std) over 100-period window

### 2b. Volatility-Scaled Momentum
- **File**: `portfolio/signals/momentum_factors.py`
- **Problem**: ROC-20 uses fixed +-5% thresholds — permanently triggered in high-vol, silent in low-vol
- **Fix**: Normalize by realized vol: `z_mom = ROC_20 / sigma_20`, threshold at +-1.5s

### 2c. Multi-Scale OFI
- **File**: `portfolio/microstructure_state.py`
- **Purpose**: 3-window OFI (5-snap fast, 15-snap medium, all slow) + flow acceleration metric
- **Impact**: Captures flow derivative, not just level

### 2d. Seasonality Integration
- **Files**: `portfolio/seasonality.py` (exists), signal pipeline
- **Problem**: Seasonality module exists but isn't connected to signal inputs
- **Fix**: Apply detrend_return() to metals momentum/mean-reversion signal inputs

## Batch 3: Accuracy & Correlation Upgrades

### 3a. Cost-Aware Accuracy
- **File**: `portfolio/accuracy_stats.py`
- **Problem**: Accuracy counts 0.06% moves as "correct" even when spread is 0.03%
- **Fix**: Add cost_adjusted_accuracy: correct only if |move| > half_spread + slippage

### 3b. Return Magnitude Tracking
- **File**: `portfolio/accuracy_stats.py`
- **Problem**: All correct predictions weighted equally (0.06% same as 3%)
- **Fix**: Track avg_return_magnitude and return_sharpe per signal

### 3c. Dynamic Correlation Grouping
- **File**: `portfolio/signal_engine.py`
- **Problem**: Static correlation groups (5 hardcoded) miss regime-dependent correlations
- **Fix**: Compute rolling correlation from signal_log, cluster >0.7 corr signals dynamically

## What Could Break

1. **Oil yfinance fetch**: Could fail if CL=F ticker changes. Mitigated by existing try/except pattern.
2. **OFI z-score**: During cold start (no history), z-score returns 0.0 = HOLD. Same as current behavior.
3. **GVZ voting**: Changes from 4 to 5 active voters in cross-asset. Could shift consensus in edge cases.
4. **Momentum thresholds**: Vol-scaling changes ROC thresholds. Some signals that were HOLD may become BUY/SELL.
5. **Dynamic correlation groups**: Could produce different groupings than static. Fallback to static if <30 samples.

## Execution Order

1. Write plan (this file) — commit
2. Batch 1 (4 quick wins) — implement in parallel subagents, commit
3. Batch 1 tests — write and run, commit
4. Batch 2 (4 preprocessing) — implement, commit
5. Batch 2 tests — write and run, commit
6. Batch 3 (3 accuracy/correlation) — implement, commit
7. Batch 3 tests — write and run, commit
8. Full test suite — fix any failures
9. Merge into main, push, clean up worktree
