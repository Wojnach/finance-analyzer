# Research Plan — 2026-04-07

## Session Findings

### Signal Audit Results
- **5 always-HOLD signals**: fear_greed, funding, ml, metals_cross_asset, orderbook_flow (zero variance)
- **Correlation gaps**: mean_reversion + rsi (r=0.537) not grouped; macro_regime + trend (r=0.520) in separate groups
- **Directional bias**: calendar (100% BUY), sentiment (97% BUY), claude_fundamental (89% BUY, 81% activity)
- **Regime distribution**: 91.3% ranging, making BUY-biased signals unreliable
- **Accuracy shifts**: fibonacci 42.5% -> 68.2% (improved), fear_greed 58.6% -> 25.9% (collapsed)

### Accuracy (Recent 1d)
Top: fibonacci 68.2%, ministral 68.0%, mean_reversion 65.4%, calendar 62.8%, bb 60.8%
Bottom: fear_greed 25.9%, news_event 29.5%, macro_regime 30.3%, volatility_sig 35.0%

## Implementation Plan

### Batch 1: Correlation group updates + bias penalty (signal_engine.py)
1. Add mean_reversion + rsi correlation group (r=0.537, both RSI-based)
2. Move macro_regime into trend_direction group (r=0.520 with trend)
3. Add directional bias penalty to _weighted_consensus()

### Batch 2: Directional accuracy tracking (accuracy_stats.py)
1. Extend signal_accuracy() to track buy/sell accuracy separately
2. Add buy_accuracy, sell_accuracy fields (additive, backward compatible)

### Batch 3: Tests for new functionality
1. Test correlation group membership
2. Test directional bias penalty computation
3. Test directional accuracy tracking

## Deferred
- Walk-forward signal reweighting (hard)
- Transformer model evaluation (hard)
- Multi-agent debate system (hard)

## Risk Assessment
- All changes additive, no existing behavior removed
- Bias penalty conservative (0.5x for extreme >85% bias only)
- Directional accuracy purely observational
