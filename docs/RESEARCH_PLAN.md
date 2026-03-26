# After-Hours Research Plan — 2026-03-27

## Bugs & Problems Found

### Signal System Issues
1. **10 signals gated** (blended accuracy < 45%): ema, fear_greed, forecast, heikin_ashi,
   macro_regime, news_event, oscillators, structure, trend, volatility_sig
2. **Signal correlation**: calendar+econ_calendar+forecast+futures_flow have 100% agreement
   rate — they are effectively one voter, not four. Additional cluster: volatility_sig,
   smart_money, fibonacci, oscillators at 97-99% agreement.
3. **Slow regime adaptation**: The 70/30 recent/all-time blend takes too long to react when
   accuracy shifts dramatically (Fear & Greed crashed 31.7% in 7 days).
4. **Regime weights incomplete**: Only rsi, macd, ema, bb, volume have regime multipliers.
   The 19 enhanced signals have no regime adjustment at all.

### Not Bugs (Working as Designed)
- Accuracy gate at 0.45 is correctly gating the 10 worst signals
- Fear & Greed is also regime-gated (only votes in ranging/high-vol)
- Local model per-ticker accuracy gating works for ministral and qwen3
- Dynamic MIN_VOTERS correctly raises bar in uncertain regimes

## Improvements Prioritized (impact × ease)

| # | Title | Impact | Effort | Files |
|---|-------|--------|--------|-------|
| 1 | Adaptive recency blend weight | HIGH | EASY | `signal_engine.py` |
| 2 | Raise accuracy gate 0.45 → 0.47 | MED | TRIVIAL | `signal_engine.py` |
| 3 | Add regime weights for enhanced signals | HIGH | EASY | `signal_engine.py` |
| 4 | Signal correlation grouping | HIGH | MEDIUM | `signal_engine.py`, `accuracy_stats.py` |
| 5 | Per-ticker accuracy for all signals | HIGH | MEDIUM | `signal_engine.py`, `accuracy_stats.py` |

## What to Implement NOW

### Batch 1: Adaptive Recency + Gate Threshold (signal_engine.py only)
1. **Adaptive recency blend**: When `|recent_acc - alltime_acc| > 0.15`, increase recent weight
   from 0.7 to 0.9. This makes the system react faster to regime shifts.
2. **Raise accuracy gate**: `ACCURACY_GATE_THRESHOLD = 0.45` → `0.47`. Catches marginal signals
   like momentum_factors (45.6% blended).

### Batch 2: Regime Weights for Enhanced Signals (signal_engine.py only)
3. Add regime-specific weight multipliers for enhanced signals:
   - `pullback` regime (new): boost fibonacci, mean_reversion, bb; dampen trend, ema
   - Update existing regime weights with enhanced signal entries
   - Add `detect_pullback` helper using: RSI < 40, price below 20-SMA, recent drawdown > 5%

### Batch 3: Signal Correlation Deduplication (signal_engine.py)
4. Group highly correlated signals and cap their combined vote weight.
   - Define correlation groups in a constant
   - In `_weighted_consensus`, when multiple signals from same group vote, only count
     the highest-accuracy one (others get 0.3x weight penalty)

## What to Defer
- Per-ticker accuracy for all signals (needs significant refactoring of accuracy_stats.py)
- New signal modules from quant research (need testing period)
- LLM multi-agent debate system (complex architecture change)

## Execution Order

1. Batch 1: `signal_engine.py` — adaptive blend + gate threshold → test → commit
2. Batch 2: `signal_engine.py` — regime weights → test → commit
3. Batch 3: `signal_engine.py` — correlation groups → test → commit
4. Write tests for new behavior → commit
5. Merge to main
