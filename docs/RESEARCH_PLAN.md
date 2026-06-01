# After-Hours Research Plan — 2026-06-01

## Status: IMPLEMENTING

## Key Findings

### Systemic BUY Accuracy Collapse
Nearly all signals show collapsed BUY accuracy (15-33%) while SELL accuracy
remains strong (56-87%). Bearish regime signature. Per-ticker consensus at
1d horizon: ETH 49.9%, MSTR 46.2%, XAG 49.4% — all below coin-flip.

### 4 Regime Signals Degraded (Small-Sample Illusion)
adx_regime_switch, bocpd_regime_switch, vol_ratio_regime, choppiness_regime_gate
were re-enabled 2026-05-28 with 58-67% accuracy on 158-586 samples. By
2026-06-01 (410-519 samples), all degraded to 49-52% — coin-flip.

### Market Context
S&P at record 7,599 (AI boom). BTC $72,145 with record $2.97B ETF outflows.
Hormuz crisis keeping oil $92-108 and inflation elevated. FOMC June 16-17
all but priced as hold. NFP Friday June 6 is marquee event.

## Implemented Changes

### Batch 1: Disabled 4 Redundant Regime Signals
- `adx_regime_switch` (49.0% all-time, 492 sam)
- `bocpd_regime_switch` (51.1%, 519 sam)
- `vol_ratio_regime` (48.8%, 2427 sam)
- `choppiness_regime_gate` (51.7%, 410 sam)
Kept: `drift_regime_gate` (58.9%) and `amihud_illiquidity_regime` (58.6%).

### Batch 2: Raised Directional Gate 43% → 44%
Catches marginal BUY noise in bearish regime. Stays below assertion floor (0.45).

### Batch 3: Resolved Critical Errors
Two accuracy_degradation entries resolved with description of fixes applied.

## Deferred to Backlog
1. Exponential-decay signal weighting (high impact, 2 days)
2. Soft regime assignments with sigmoid thresholds (3 days)
3. Rolling Spearman IC recomputation (2 days)
4. Regime-conditional ATR stop multipliers (2 days)
5. Inverse-volatility position sizing (2 days)
6. Thompson sampling signal selection (4 days)
7. Multi-agent debate for Layer 2 (3 days)
8. 3-state NHMM for BTC/ETH (5 days)
