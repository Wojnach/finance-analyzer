# Agent Review: signals-modules (2026-04-20)

## P1 Critical (bias, look-ahead, wrong direction)
1. **futures_flow.py funding rate STILL BUY-biased** — -0.03% BUY vs +0.05% SELL. BUY threshold 40% tighter. Fires BUY ~3x more often.
2. **econ_calendar.py STILL structurally SELL-only** — All 4 sub-signals can only vote SELL or HOLD. Permanent SELL bias.
3. **network_momentum.py correlation_regime has structural BUY bias** — When net_div=HOLD (common), corr_regime stays as raw "BUY" giving the majority a permanent BUY lean.
4. **vix_term_structure.py z=0.0 threshold** — Every non-zero z-score votes. Effectively a coin flip on daily jitter.
5. **calendar_seasonal.py residual BUY bias** — 6 of 8 sub-signals are BUY-only.

## P2 High
1. orderbook_flow.spread_health CONFIRMED dead (always HOLD)
2. crypto_macro.py no ImportError guard (crashes engine for crypto tickers if dependency missing)
3. forecast.py GPU hang potential (7.5 min worst case)
4. claude_fundamental.py unbounded daemon threads
5. network_momentum.py yfinance blocks 30+ seconds (no timeout)
6. hurst_regime.py trend_direction and hurst_regime IDENTICAL (double-counts trend)

## P3 Medium
1. copper_gold_ratio.py cache serves stale data indefinitely on outage
2. shannon_entropy.py uses bespoke confidence logic (inconsistent with system)
3. statistical_jump_regime.py persistence has no reset path to neutral
4. gold_real_yield_paradox.py structurally BUY-biased in current regime
5. cross_asset_tsmom.py bond_momentum not asset-class-aware (wrong for crypto)

## Prior Finding Status
- econ_calendar SELL-only: **STILL PRESENT**
- funding_rate BUY-biased: **STILL PRESENT**
- orderbook_flow spread_health dead: **PARTIALLY FIXED** (documented as intentional)
