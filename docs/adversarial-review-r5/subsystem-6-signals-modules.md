# Subsystem 6: Signals Modules — Round 5 Findings

## CRITICAL (P1)

**SM-R5-1** — vix_term_structure.py _ratio_zscore always votes when _Z_THRESHOLD=0.0.
Near-binary oscillator voting SELL/BUY on microscopic noise. Permanent noise contributor.
Fix: Set _Z_THRESHOLD to 0.75 or 1.0.

**SM-R5-2** — vix_term_structure.py _contango_depth creates permanent BUY bias.
BUY threshold at ratio < 0.90 vs backwardation_flag at < 0.85. Normal market (0.85-0.90)
has one sub-signal always voting BUY. With majority_vote(count_hold=False), this BUYs 90%+ of days.
Fix: Align BUY threshold to 0.85 or remove _contango_depth.

## HIGH (P2)

**SM-R5-3** — hurst_regime.py duplicate sub-signal votes inflate confidence.
hurst_regime and trend_direction both set to same computed value. 4 voters but only 2 independent.
Fix: Remove duplicate voter key.

**SM-R5-6** — calendar_seasonal.py day-of-week effect applied to 24/7 crypto and metals.
Monday SELL / Friday BUY has no empirical basis for assets without weekend closure.
Fix: Gate to equity tickers only.

## MEDIUM (P3)

**SM-R5-7** — structure.py period_low guard uses != 0 instead of > 0.
**SM-R5-8** — futures_flow.py capitulation BUY has no magnitude check (fires on noise).
