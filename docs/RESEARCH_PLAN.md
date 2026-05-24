# After-Hours Research Plan — 2026-05-24

## Findings Summary

### Phase 0: Daily Review
- System healthy: 0 errors, 76 cycles, ~14h uptime, all signals green
- Last trigger: XAU-USD BUY→HOLD flip at 19:03 UTC
- Market regime: range-bound across all instruments for 3+ days
- 30 unresolved critical errors (14 contract violations, 10 accuracy degradation, 3 Avanza, 3 misc)

### Phase 1: Market Research
- **Monday May 26 = Memorial Day** — US markets closed
- **Binary event: Iran deal** — 50/50 odds. Deal = risk-on (oil -$10-15, gold dip). Failure = risk-off (Brent $110+, gold $4800+)
- **Thursday May 28**: GDP + PCE deflator + durable goods + claims (big data day)
- BTC ~$77K, Gold $4521, Silver $75.35
- MSTR: Saylor signaled willingness to sell BTC — policy shift

### Phase 2: Quant Research
- IC-based dynamic signal weighting outperforms equal weights (already partially implemented)
- HMM regime detection for signal selection (currently flat penalty)
- Gold-silver ratio at ~57 (below 65-70 mean — silver expensive vs gold near-term)
- MVRV Z-Score at 0.41 — BTC mid-cycle, not overvalued
- COT: hedge funds barely positioned in silver (5,472 contracts) — room for re-entry

### Phase 3: Signal Audit
- **DISABLE momentum_factors**: 30% recent accuracy (783 sam) — catastrophic
- **DISABLE btc_proxy**: 44.6% 1d (139 sam), BUY 31.1% — negative evaluation
- Top signals: qwen3 (59.9%), drift_regime_gate (59.3%, 71.5% recent), bb (54.8%, 64.5% recent)
- macro_external cluster misclusters fear_greed (BUY-only) with news_event (SELL-only)
- 28 disabled signals have 0 outcome samples — not in _SHADOW_SAFE_SIGNALS

## Implementation Plan

### Batch 1: Shadow tracking expansion + formal disables
**Files**: portfolio/signal_engine.py, portfolio/tickers.py

1. Add OHLCV-only disabled signals to `_SHADOW_SAFE_SIGNALS`:
   - ttm_squeeze, tsi_chop_mr, amihud_illiquidity_regime, absorption_ratio_regime
   - connors_rsi2, adx_regime_switch, choppiness_regime_gate
   - autotune_adaptive_cycle, bocpd_regime_switch
   - trend_slope_momentum, sentiment_extremity_gate

2. Formally disable in tickers.py:
   - momentum_factors (30% recent, 51.9% all-time — blended ~37%, below gate anyway)
   - btc_proxy (44.6% 1d, BUY 31.1% — evaluation complete, negative verdict)

### Batch 2: Morning briefing synthesis
Write data/morning_briefing.json combining all phase findings.
Send Telegram summary.

### Batch 3: Merge, verify, ship
- Full test suite
- Merge to main, push
- Restart loops
- Update SESSION_PROGRESS.md
