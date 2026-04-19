# After-Hours Research Plan — 2026-04-19

## Findings Summary

### Phase 0: System Review
- System healthy: all 30+ signal modules OK, 158 cycles, 0 errors, ~34h uptime
- 4 Layer 2 invocations today (Sunday), all HOLD — correct for weekend
- Patient ETH: near flat (-0.6% to -1%), razor-thin above 2xATR stop
- Bold BTC: +0.5-0.9%, ETH: -0.3-0.5%
- Module failures from Apr 17: monte_carlo, price_targets, equity_curve (non-critical)

### Phase 1: Market Research — CRITICAL
- **Iran RE-CLOSED Strait of Hormuz** on Apr 19 after US Navy seized Iranian cargo ship
- **Ceasefire expires Monday Apr 21** — most binary event of the week
- Oil rebounded to $90-96 WTI from $83.85 low
- S&P hit record 7,126 BEFORE re-closure — Monday gap down likely
- Tesla + Alphabet earnings Tuesday Apr 22
- FOMC Apr 28-29 (94% hold expected)
- COT: Gold specs net-long 22.8K (not stretched), silver specs slightly trimming

### Phase 3: Signal Audit — CRITICAL FINDINGS
1. **REGIME SHIFT DETECTED**: Trend-following signals surging (trend 40.3%→61.6%, EMA 52.4%→62.9%), mean-reversion collapsing (BB 52.6%→41.7%, mean_rev 52.5%→45.5%)
2. **Crisis mode is HARMFUL**: 3/5 macro signals broken → crisis mode ON → penalizing trend signals by 0.6x — but trend signals are our BEST performers. System is sabotaging itself.
3. **Orphaned correlated signals**: credit_spread_risk (100% agree with macro_regime) and futures_flow (100% agree) getting full 1.0x weight despite redundancy.
4. **Trend_direction mega-cluster**: 9 members at 0.3x penalty = 3.4x effective weight, excessive.
5. **21/42 signals always HOLD for XAG-USD** — dead weight, don't contribute but inflate denominator.

### Ticker Deep Dive (XAG, BTC, ETH)
- Silver: Solar PV demand dropping 19% in 2026 (bearish revision). G/S ratio at 57:1, near mean. Seasonal weakness May-June.
- BTC: Funding rates most negative since 2023 (bottom signal). Six failed breakouts above $76K. MVRV well below cycle peak.
- ETH: ETH/BTC bouncing from 0.028 lows, must reclaim 0.035. 30% supply staked (ATH).

## Implementation Plan

### Batch 1: Fix Crisis Mode Logic (HIGHEST PRIORITY)
**Why:** Crisis mode is actively penalizing our best-performing signals. This is a 40% weight reduction on signals with 61-63% accuracy. Fixing this has immediate positive impact on consensus quality.

**Files:** `portfolio/signal_engine.py`

**Changes:**
1. Add trend-accuracy check before applying crisis penalty: if blended trend signal accuracy > 55%, skip `_CRISIS_TREND_PENALTY`
2. Make the crisis response conditional: only penalize trend signals when they're ALSO underperforming
3. Add logging so we can track when crisis mode activates/skips

### Batch 2: Orphaned Signal Correlation Fix
**Why:** credit_spread_risk and futures_flow are 100% correlated with trend_direction group members but get full 1.0x weight, inflating the trend-direction cluster's effective weight.

**Files:** `portfolio/signal_engine.py`

**Changes:**
1. Add `credit_spread_risk` to `trend_direction` group
2. Add `futures_flow` to `trend_direction` group (or a new `cross_asset_flow` group)
3. The existing 0.3x penalty will automatically apply

### Batch 3: Tighten Mega-Cluster Penalty
**Why:** trend_direction now has 11 members (after Batch 2) at 0.3x = 1.0 + 10*0.3 = 4.0x effective weight. Even at 0.15x = 1.0 + 10*0.15 = 2.5x.

**Files:** `portfolio/signal_engine.py`

**Changes:**
1. Add per-cluster penalty override for trend_direction at 0.15x (existing `_CLUSTER_CORRELATION_PENALTIES` dict)
2. Only tighten trend_direction — other clusters are smaller and 0.3x is appropriate

### Batch 4: Tests
**Files:** `tests/test_signal_engine.py` (or relevant test files)

**Changes:**
1. Add test for crisis mode with high-accuracy trend signals → crisis penalty NOT applied
2. Add test for credit_spread_risk in correlation group → penalty applied
3. Run full suite to verify no regressions

## What to Defer
- Per-ticker signal exclusion (needs more analysis)
- New signal modules from quant research (needs backtesting)
- Prophecy revisions (silver PV demand drop — needs user discussion)

## Execution Order
1. Create worktree: `git worktree add ../research-0419 -b research/daily-2026-04-19`
2. Batch 1: Crisis mode fix → test → commit
3. Batch 2: Correlation groups → test → commit
4. Batch 3: Mega-cluster penalty → test → commit
5. Batch 4: New tests → full suite → commit
6. Merge to main, push, clean up worktree
