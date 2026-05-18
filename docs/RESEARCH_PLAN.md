# After-Hours Research Plan — 2026-05-18

## Context

After-hours research session. Risk-off day: Dow -1.1%, Nasdaq -1.5%, 10Y yield
>4.60%, oil $108 WTI (Iran/Hormuz), $657M crypto liquidations. System ranging
all day with 30 Layer 2 invocations, only 2-3 actual trades.

## Bugs & Problems Found

1. **credit_spread_risk unclustered** (signal_engine.py:1737): After futures_flow
   disabled (2026-05-07), credit_spread_risk voted at full 1.0x weight despite
   100% correlation with crypto_macro and econ_calendar.
   FIX: Moved to fundamental_cluster.

2. **crypto_evrp disabled but best recent performer** (tickers.py:94): 80.5%
   1d_recent (77 sam), 92.4% 3d. Anti-correlated with crypto_macro (32.8%).
   FIX: Re-enabled.

3. **btc_proxy unprotected in ranging**: 46.5% all-time, MSTR dropped 8.8% on
   BTC -1.5%. Leverage amplifies noise in sideways markets.
   FIX: Added to ranging regime gate.

## Improvements Implemented

### Batch 1 (committed: 03d1b0cf)
- Enable crypto_evrp signal
- Move credit_spread_risk to fundamental_cluster

### Batch 2
- Gate btc_proxy in ranging regime
- Update RESEARCH_PLAN.md

## Deferred to Improvement Backlog

1. Walk-forward signal reweighting — effort: 2 days, impact: high
2. Signal correlation pruning (mean_reversion cluster) — effort: 2 days
3. HMM regime detection (3-state) — effort: 3 days
4. Adaptive MSTR-BTC beta (rolling OLS) — effort: 1 day
5. LLM multi-agent debate for Layer 2 — effort: 5+ days
