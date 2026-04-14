# After-Hours Research Plan — 2026-04-14

## Research Findings Summary

### Signal Audit KEY FINDINGS
1. **Mega trend cluster**: trend+macro_regime+structure+heikin_ashi+momentum_factors+oscillators+volatility_sig — 8 signals with 85-99.7% mutual agreement
2. **Sentiment in freefall**: 33.8% recent 3h accuracy
3. **Calendar/econ_calendar**: 0% agreement — perfectly opposing, cancel each other
4. **Directional bias**: claude_fundamental 10:1 BUY:SELL, structure 11.5:1, calendar BUY-only
5. **Hyperactive**: volume_flow 87.7%, mean_reversion 76.7%, momentum 72.7%
6. **MSTR at 46.5%** — generic signals on a BTC proxy don't work
7. **ETH at 48.1%** — follows BTC 97% of the time

## Implementation Plan

### Batch 1: Correlation group realignment
**Files:** `portfolio/signal_engine.py`
- Move `momentum_factors` from `macro_external` to `trend_direction` (0.593-0.621 corr with cluster)
- Remove `structure` from `volatility_cluster` (belongs in `trend_direction`: 0.608 with trend)
- Add `oscillators` to `trend_direction` (0.463 with heikin_ashi, 83.4% agreement)
- Tighten `trend_direction` penalty from 0.2 to 0.12 (now 8 members, need strict penalty)
- Keep `structure` also in `fundamental_cluster` (dynamic groups will pick best fit)

### Batch 2: Directional bias penalty
**Files:** `portfolio/signal_engine.py`
- Add directional bias detection in `_weighted_consensus`
- Signals with >90% one-directional activity get 0.5x weight penalty
- Affected: calendar (100% BUY), econ_calendar (100% SELL), news_event (100% SELL)

### Batch 3: Research deliverables commit
- Commit all JSON research outputs

## Deferred
- IC-based weighting — needs rolling IC infrastructure
- HMM regime detection — complex, separate session
- MSTR BTC-proxy — needs shared state refactoring
- ATR position sizing — portfolio_mgr changes
