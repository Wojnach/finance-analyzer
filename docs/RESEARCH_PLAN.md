# After-Hours Research Plan — 2026-04-17

## Findings Summary

### Phase 0: System Review
- System healthy, all 28 signal modules OK, 21 cycles, 0 errors
- Both portfolios profitable: Patient +4.2% ETH, Bold +3.7% BTC / +4.7% ETH
- No trades executed, all HOLD decisions (correct — ranging regime)

### Phase 1: Market Research
- **CRITICAL**: US-Iran ceasefire expires Monday April 21 — binary event
- S&P 500 record 7,126, Nasdaq 13-straight gain
- Oil crashed 11.5% (Hormuz reopening)
- BTC whale accumulation: 270K BTC in 30 days (largest since 2013)
- Gold holding $4,878 despite risk-on = very bullish structural bid
- Silver broke $80, supply deficit deepening (6th year)
- DXY fell to 97.70, dollar weakness tailwind for metals + crypto
- ETH ETF inflows rotating in ($187M/week)
- MSTR: 3 BTC purchases in April, 780K total holdings

### Phase 3: Signal Audit
1. **Directional asymmetry** — Qwen3 BUY=30.3% (gated), SELL=74.3%. Claude_fundamental BUY=65.5%, SELL=33.6% (gated). System already gates at 40%; raising to 45% catches `momentum` SELL (45.0%).
2. **Correlation clusters** — Already well-handled. `trend_direction` has 8 signals at 0.12x penalty. Dynamic clustering from signal_log correlations.
3. **Per-ticker consensus BELOW coin flip** — ETH-USD: 47.7-49.6%. MSTR: 45.9-47.2%. System is net negative on these. Need per-ticker gate.
4. **Zero-sample signals** — 8 signals (cot_positioning, futures_basis, hurst_regime, shannon_entropy, vix_term_structure, gold_real_yield_paradox, cross_asset_tsmom, copper_gold_ratio) have 0 BUY/SELL votes at ALL horizons. Always HOLD = wasting compute.
5. **Regime shift** — Fear & Greed crashed 58.6% -> 25.9% at 1d recent. Trend surged 40.3% -> 61.6% at 3h recent. Mean-reversion and pattern signals outperforming macro signals.

---

## Implementation Plan

### Batch 1: Signal Gating Improvements (signal_engine.py)
**Files**: `portfolio/signal_engine.py`

1. **Raise directional gate threshold from 0.40 -> 0.45**
   - Catches `momentum` SELL at 45.0% (currently ungated)
   - Also catches any other borderline cases in the 40-45% range
   - Risk: may over-gate some signals with marginal directional accuracy. But 45% is still below coin flip — these directions are noise.
   - Verify: assertion at line 383 that relaxed overall gate > directional gate still holds

2. **Add per-ticker consensus accuracy gate**
   - When per-ticker consensus accuracy at this horizon is < 50% with 500+ samples, log a warning and apply confidence penalty (0.5x)
   - Don't force HOLD (too aggressive) — just reduce confidence so Layer 2 sees the signal is untrustworthy
   - Needs per-ticker accuracy data passed into consensus function (already available via `ticker` param)

### Batch 2: Zero-Sample Signal Investigation
**Files**: `portfolio/signals/*.py` (read-only investigation)

- Audit why 8 signals always vote HOLD — are they gated by horizon, asset class, or always missing data?
- If they're always HOLD: add them to a `NEVER_VOTED_SIGNALS` log entry for monitoring
- Don't disable them — they may start voting when conditions change

### Batch 3: Tests
**Files**: `tests/test_signal_engine.py`

- Test directional gate at new 0.45 threshold
- Test per-ticker confidence penalty

### Batch 4: Research Deliverables
**Files**: `data/morning_briefing.json`

- Write morning briefing combining all phase findings
- Send Telegram notification

---

## What NOT to implement (deferred)

- **Regime-adaptive signal selection** — System already has 70/30->90/10 recency blending + crisis mode. Good enough for now.
- **IC-based weighting** — IC module exists (`ic_computation.py`). Integration is medium effort, defer to next session.
- **Walk-forward validation** — Needs backtesting infra, not a quick fix.
- **New signal modules** — Wait for Phase 2 quant research results.

## Execution Order

1. Create worktree: `research/daily-2026-04-17`
2. Batch 1: signal_engine.py changes + run tests
3. Batch 2: zero-sample signal investigation (read-only)
4. Batch 3: new tests
5. Merge to main, push
6. Write morning briefing
