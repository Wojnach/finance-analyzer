# After-Hours Research Plan — 2026-04-04

## Context

Saturday session. US markets closed (Good Friday was Apr 3). All instruments in pure
ranging regime (99/100 recent snapshots). Both portfolios 100% cash. System health
excellent: 64 cycles, 0 errors, all 21 signals 100% healthy. Module failures
(monte_carlo, price_targets, equity_curve) at 06:27 UTC — startup transient, recovered.

Markets extremely range-bound: BTC $67K, XAG $73, XAU $4673, ETH $2058. No trades
executed today. Layer 2 all HOLD. 20 trigger events in 7h but all pure noise — consensus
oscillating BUY/SELL/HOLD every 10-30 minutes.

Macro: Iran war ongoing (Trump vowed more strikes), oil $98-103 WTI, NFP massive beat
(178K vs 50K expected), crypto extreme fear (index 9), ETH ETF inflows $71M (bullish
divergence vs BTC's $9M).

## Key Finding: Regime-Accuracy Divergence

The most impactful discovery today is a massive divergence between all-time and recent
7-day signal accuracy, confirming a strong regime shift:

### Recent 7d Winners (surging in ranging regime)
| Signal | All-time | Recent 7d | Delta | Samples |
|--------|----------|-----------|-------|---------|
| fibonacci | 42.9% | 68.2% | +25.3% | 110 |
| macd | 43.4% | 58.7% | +15.3% | 138 |
| mean_reversion | 51.7% | 65.4% | +13.7% | 332 |
| ministral | 59.2% | 68.0% | +8.8% | 125 |
| bb | 53.4% | 60.8% | +7.4% | 293 |

### Recent 7d Losers (collapsing in ranging regime)
| Signal | All-time | Recent 7d | Delta | Samples |
|--------|----------|-----------|-------|---------|
| fear_greed | 58.6% | 25.9% | -32.7% | 170 |
| news_event | 51.7% | 29.5% | -22.2% | 112 |
| macro_regime | 47.1% | 30.3% | -16.8% | 145 |
| structure | 52.1% | 36.1% | -16.0% | 147 |
| volatility_sig | 47.1% | 35.0% | -12.1% | 223 |

### Signal Correlations (from 100 recent snapshots)
| Pair | Ticker | Correlation | Issue |
|------|--------|-------------|-------|
| candlestick-fibonacci | BTC | 0.708 | Double-counting — nearly identical votes |
| fibonacci-mean_reversion | XAG | -0.694 | Anti-correlated, cancel out in consensus |
| macd-candlestick | BTC | 0.453 | Moderate redundancy |
| volume_flow-oscillators | BTC | 0.445 | Oscillators dragging (34.3%) |

## Bugs & Problems Found

### BUG-161: oscillators boosted 1.2x in ranging despite 34-39% accuracy (P1)
- **File:** `portfolio/signal_engine.py` line 147
- **Root cause:** `REGIME_WEIGHTS["ranging"]["oscillators"] = 1.2` — this BOOSTS oscillators
  in ranging regime, but per-ticker accuracy is 34.3% (BTC), 35.1% (XAG), 39.5% (XAU),
  34.5% (ETH). The system is amplifying its worst signal.
- **Fix:** Change to 0.3x or remove entry entirely (defaults to 1.0).
  Also add oscillators to `REGIME_GATED_SIGNALS["ranging"]["_default"]`.
- **Impact:** MEDIUM — removes negative alpha from worst signal.

### BUG-162: candlestick-fibonacci not in CORRELATION_GROUPS (P2)
- **File:** `portfolio/signal_engine.py` lines 467-483
- **Root cause:** 0.708 correlation for BTC between candlestick and fibonacci signals.
  Both vote the same way most of the time but count as independent votes.
- **Fix:** Add correlation group: `"pattern_based": frozenset({"candlestick", "fibonacci"})`.
- **Impact:** MEDIUM — reduces double-counting for BTC consensus.

### BUG-163: candlestick not in ranging regime gate (P2)
- **File:** `portfolio/signal_engine.py` line 176-181
- **Root cause:** candlestick recent 7d accuracy is 44.5% (292 samples), below the 45%
  accuracy gate threshold, but not explicitly in the regime gate. The accuracy gate
  should catch it but explicit gating is more robust.
- **Fix:** Add "candlestick" and "oscillators" to
  `REGIME_GATED_SIGNALS["ranging"]["_default"]`.
- **Impact:** LOW-MEDIUM — eliminates two more noise signals from ranging consensus.

## Improvements Prioritized

1. Fix oscillators weight and regime gating (BUG-161) — easy, high certainty
2. Add candlestick-fibonacci correlation group (BUG-162) — easy, medium impact
3. Add candlestick+oscillators to ranging regime gate (BUG-163) — easy, defensive

## What to Implement NOW

### Batch 1: Signal Engine Config Fixes (signal_engine.py only)
1. Change `REGIME_WEIGHTS["ranging"]["oscillators"]` from 1.2 to 0.3
2. Add "oscillators" and "candlestick" to `REGIME_GATED_SIGNALS["ranging"]["_default"]`
3. Add correlation group: `"pattern_based": frozenset({"candlestick", "fibonacci"})`
4. Run existing tests to verify no regressions

### Batch 2: Verify and Ship
1. Run full test suite
2. Commit, merge, push
3. Write morning briefing

## What NOT to Change
- No parameter changes to live config.json
- No changes to DISABLED_SIGNALS list (crypto_macro already disabled)
- No changes to accuracy gate threshold (45% is correct)
- No changes to per-ticker override logic (BUG-158 already fixed)
- No changes to MIN_VOTERS (already at 3)
