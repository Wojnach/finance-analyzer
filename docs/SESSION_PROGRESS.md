# Session Progress — After-Hours Research 2026-04-07

**Date:** 2026-04-07
**Branch:** `research/daily-2026-04-07`
**Focus:** Signal audit, correlation group optimization, directional accuracy tracking

## Completed

### Phase 0: Daily Review
- System healthy: 298 cycles, 0 errors, all 21 signal modules passing
- No Layer 2 decisions today (last Apr 3)
- Prices: XAG $73.05, XAU $4,709, BTC $69,592, ETH $2,131
- Both portfolios fully liquidated (0 cash, 0 holdings)

### Phase 3: Signal Audit
- 5 always-HOLD signals: fear_greed, funding, ml, metals_cross_asset, orderbook_flow
- Correlation gaps found: mean_reversion+rsi (r=0.537), macro_regime+trend (r=0.520)
- Extreme directional bias: calendar 100% BUY, sentiment 97% BUY, econ_calendar 100% SELL
- 91.3% ranging regime — mean reversion and calendar signals dominant
- Accuracy regime shift: fibonacci 42.5% -> 68.2%, fear_greed 58.6% -> 25.9%

### Phase 5: Plan
- Wrote implementation plan in docs/RESEARCH_PLAN.md
- 3 batches: correlation groups, bias penalty, directional accuracy

### Implementation Batches
1. **Batch 1** (signal_engine.py): Added rsi_based correlation group, moved macro_regime to trend_direction, added directional bias penalty (0.5x for bias > 85%)
2. **Batch 2** (accuracy_stats.py): Added directional accuracy tracking (buy_accuracy, sell_accuracy) to signal_accuracy()
3. **Batch 3** (tests): 16 new tests added (correlation groups, bias penalty, directional accuracy). All 24 targeted tests pass. Fixed 1 pre-existing test (macro_regime group membership).

### Phase 1 & 2: Market + Quant Research (background agents)
- Market research: Iran-Hormuz deadline CRITICAL, PCE/CPI/FOMC this week, gold ETF outflows record
- Quant research: MWU adaptive weighting, copula-based pruning, HMM regime detection, regime-adaptive signal selection

### Morning Briefing
- Written to data/morning_briefing.json
- Telegram notification sent

## Commits
1. `docs: research plan for 2026-04-07 session`
2. `feat: improve signal correlation groups and add directional bias penalty`
3. `feat: track directional accuracy (buy/sell) separately`
4. `test: add tests for correlation groups, bias penalty, directional accuracy`
5. `test: update macro_regime correlation group test for new placement`

## Key Finding
The system's accuracy gate and crisis mode are working, but the directional bias issue is subtle: signals like `calendar` (100% BUY) inflate BUY consensus in ranging markets. The existing `normalized_weight` bias_penalty mitigates this (0.1x floor), but the new explicit directional bias penalty adds a stronger check for extreme cases. The directional accuracy tracking will reveal whether high-accuracy signals are genuinely predictive or just riding market drift.
