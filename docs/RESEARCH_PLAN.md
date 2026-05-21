# After-Hours Research Plan — 2026-05-21

## Context

System: 96% ranging regime, **48.6% overall 1d consensus accuracy** (below coin flip).
Active instruments: XAG-USD ($76.68, -13.1% from avg), BTC-USD ($77.5K, dust position),
XAU-USD (no position), ETH-USD (no position), MSTR (no position).

Macro: Hawkish FOMC minutes (Kevin Warsh as new chair), bogus Iran deal whipsaw,
oil elevated ($99.5 WTI), gold $4,517, silver recovering from $73 low.

Critical errors: 7× Layer 2 silent failures, 2× accuracy degradation, 1× Avanza expired.

## Bugs & Problems Found

1. **credit_spread_risk at 23% blended accuracy** — Blended 0.90×0.20 + 0.10×0.50 = 23%.
   Should be accuracy-gated at runtime, but formally disabling saves compute and prevents
   edge cases where regime/per-ticker overrides might restore it.

2. **Five enabled signals with severely degraded recent accuracy**:
   - credit_spread_risk: 50% all-time → 20% recent (blended 23%)
   - ministral: 58% all-time → 30.8% recent at 1d (BUT 62.6% at 3h)
   - crypto_macro: 54.6% all-time → 34% recent (blended 36.3%)
   - econ_calendar: 57.5% all-time → 40.1% recent (blended 42.1%)
   - momentum_factors: 52.1% all-time → 41.4% recent at 1d (BUT 60.1% at 3h)

3. **Regime-gated signals have inflated "recent" accuracy** — heikin_ashi shows 82.2%
   recent but is regime-gated for ranging (96% of time). Its "recent" accuracy only
   measures the 4% non-ranging periods — survivorship bias, not genuine improvement.

4. **7 Layer 2 contract violations today** — trigger fires, no journal entry. Same
   pattern as yesterday. Agent timeouts or auth failures.

5. **Overall consensus 48.6% at 1d** — system predictions are net-negative. The
   combination method is destroying individual signal edge. Top signals (Qwen3 69.4%
   recent, BB 58.6%, RSI 61.1%) are diluted by many weak/harmful signals.

## Improvements Prioritized

### Batch 1: Disable Harmful Signal (tickers.py)
- **B1.1**: Add `credit_spread_risk` to DISABLED_SIGNALS with documentation
- Rationale: 23% blended accuracy, actively harming consensus

### Batch 2: Signal Backlog Updates (data/)
- **B2.1**: Add 9 new signal candidates from quant research + ticker deep dives to backlog
- **B2.2**: Update seasonality profiles with latest research

### Batch 3: Accuracy Gate Recency Enhancement (accuracy_stats.py, signal_engine.py)
- **B3.1**: Log blended accuracy per signal in agent_summary for visibility
  (currently only all-time shown in cache — makes audit misleading)

## Deferred to IMPROVEMENT_BACKLOG

- IR-based signal weighting (effort: 3d, impact: HIGH) — top quant research recommendation
- BOCPD regime detection (effort: 2d, impact: HIGH) — faster than rule-based
- Structured Bull/Bear debate in Layer 2 prompt (effort: 1d, impact: HIGH)
- LightGBM meta-learner for metals (effort: 3d, impact: MEDIUM-HIGH)
- Composite on-chain BTC meta-model (effort: 3d, impact: HIGH)
- BTC-lead signal for ETH (effort: 1d, impact: MEDIUM-HIGH)
- Funding rate regime transition detector (effort: 1d, impact: MEDIUM)
- Regime gate audit with current data (regime gates set Apr 2, now stale)

## Execution Order

1. Worktree `research/2026-05-21`
2. Batch 1 (disable credit_spread_risk) → test → commit
3. Batch 2 (backlog updates) → commit
4. Batch 3 (blended accuracy logging) → test → commit
5. Morning briefing → Telegram
6. Review → merge → push → clean up
