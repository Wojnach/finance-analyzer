# Research Implementation Plan — 2026-05-19 After-Hours Session

## Context

System accuracy at ~50% consensus across all Tier 1 tickers — near coin-flip.
Bond rout (30Y at 5.198%) driving metals down, crypto range-bound.
XAG -5.6% today, XAU -1.7%. Signal system better at SELL than BUY (80.2% vs 41.8% RSI).

## Bugs & Problems Found

1. **crypto_macro + credit_spread_risk ungrouped** despite 100% agreement (60 samples).
   Double-counting at 3h for BTC. (`signal_engine.py` correlation groups)
2. **XAG trigger persistence at 1 cycle** (relaxed 2026-05-11) causes consensus
   flip-flop with only 6-7 effective voters after gating. (`trigger.py`)
3. **BTC 12h phantom BUY** — no 12h-specific regime gate for mean_reversion/RSI.
   False oversold readings in ranging regime. (`signal_engine.py`)
4. **Overtriggering** — 20 invocations today, mostly HOLD. Trigger threshold calibrated
   for 17 voters but only 7-9 effective. (`trigger.py`)

## Improvements Prioritized

### Batch 1: Signal System Fixes (signal_engine.py + trigger.py)
1. Add correlation group: crypto_macro + credit_spread_risk
2. Raise XAG/XAU trigger persistence from 1 → 2 cycles
3. Add 12h regime gate for mean_reversion + RSI on BTC
4. Scale ranging trigger threshold by effective voter ratio

### Batch 2: New Signal — ConnorsRSI(2) for Crypto
- New module: `portfolio/signals/connors_rsi2.py`
- RSI(2) < 10 = BUY, > 90 = SELL for BTC/ETH only
- Register in `portfolio/signal_registry.py`
- Add to DISABLED_SIGNALS initially (shadow mode)
- Tests: `tests/test_signal_connors_rsi2.py`

### Batch 3: New Signal — ADX Dual Regime Meta-Signal
- New module: `portfolio/signals/adx_regime_switch.py`
- ADX <= 25: emit mean-reversion context, ADX > 25: trending context
- Register, disable initially, tests

## Deferred

- Walk-forward signal reweighting (medium risk, 2+ days)
- Absorption Ratio regime detection (already added as module, pending validation)
- MSTR mNAV premium signal (needs BTC treasury data source)
- Multi-LLM disagreement meta-signal (0.5 day, lower urgency)

## Execution Order

1. Create worktree: `research/daily-2026-05-19`
2. Batch 1: signal_engine.py + trigger.py fixes → test → commit
3. Batch 2: ConnorsRSI(2) signal → test → commit
4. Batch 3: ADX regime switch → test → commit
5. Full test suite
6. Merge into main, push
7. Clean up worktree
