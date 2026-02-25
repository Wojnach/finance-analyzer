# Silver ORB (Opening Range Breakout) Trading Strategy

## Overview

This module implements an Opening Range Breakout (ORB) strategy for silver trading
via MINI Long warrants on Avanza. It observes the 9-11 CET price window and predicts
the day's maximum and minimum prices using historical extension statistics.

## Theoretical Foundation

### Opening Range Breakout (ORB)
- **Origin**: Toby Crabel, "Day Trading with Short Term Price Patterns and Opening Range Breakout" (1990)
- **Concept**: The first N minutes of trading establish a "reference range" that statistically contains
  predictive information about the day's full range.
- **Application**: We use 9-11 CET (08:00-10:00 UTC) as the observation window for silver,
  coinciding with the European market open when silver liquidity is high.

### Market Profile / Initial Balance
- **Origin**: Peter Steidlmayer, CBOT (1980s)
- **Concept**: The "Initial Balance" (first 60-90 min) defines the value area. ~80% of days,
  price stays within 2x the Initial Balance range.
- **Our data**: 29% of days, the morning HIGH is the day's high. 26% of days, the morning LOW
  is the day's low. On remaining days, price extends by a statistically predictable amount.

### Mark Fisher's ACD Method
- **Used at**: MBF Clearing (largest floor operation at NYMEX — a commodities exchange)
- **Concept**: A-up/A-down levels from the opening range predict daily extremes
- **Relevance**: Silver is a NYMEX-traded commodity — this method was literally designed for it

## Our Implementation

### Data Source
- Binance FAPI: `XAGUSDT` perpetual futures (15-minute candles)
- ~50 days of history (5 batches × 1000 candles × 15min)

### Morning Window: 9-11 CET (08:00-10:00 UTC)
- **Why this window?**
  - European market open (London/Zurich fix at 10:30 CET)
  - COMEX electronic trading active
  - Overlaps with Asian close, establishing the day's initial price discovery
  - 2-hour window provides 8 fifteen-minute candles — statistically robust

### Prediction Method
For each historical day, we calculate:
- **Upside extension**: How much higher did price go beyond the morning HIGH? (as % of morning high)
- **Downside extension**: How much lower did price go beyond the morning LOW? (as % of morning low)

We use percentile-based predictions:
- **Conservative (25th percentile)**: 75% of days extend at least this much
- **Median (50th percentile)**: 50% of days extend at least this much
- **Aggressive (75th percentile)**: 25% of days extend at least this much

### Filters
1. **Morning direction**: UP mornings (close > open) and DOWN mornings have different extension profiles.
   DOWN mornings show much larger downside extensions (-4.38% avg vs -2.00%).
2. **Morning range size**: Small ranges tend to precede larger day ranges (compression → expansion).
3. **Volume**: Higher morning volume = more reliable range anchor.

### Historical Statistics (35 trading days, Jan-Feb 2026)

| Metric | Value |
|--------|-------|
| Days analyzed | 35 |
| HIGH set in morning | 29% |
| LOW set in morning | 26% |
| Median upside extension | +1.61% |
| Median downside extension | -1.44% |
| Avg extension as multiple of morning range | 1.11x up, 1.51x down |

**Timing of extremes:**
- Day's HIGH clusters at: 09-10 CET (morning) and 19-21 CET (US afternoon)
- Day's LOW clusters at: 09 CET (morning) and 15-16 CET (US open)

## Trading Workflow (Tomorrow)

1. **09:00-11:00 CET**: Observe morning range (automated by silver_monitor.py)
2. **11:00 CET**: ORB predictor calculates targets:
   - BUY target: predicted LOW (median) → place limit buy order on Avanza
   - SELL target: predicted HIGH (median) → place limit sell order
3. **11:00-22:00 CET**: Monitor positions, adjust if needed
4. **22:00 CET**: Postmortem — compare predicted vs actual, log lessons

## Warrant Translation

MINI L SILVER AVA 301:
- Leverage: 4.76x
- Financing level: ~$71.53
- Warrant intrinsic = silver_price - financing_level
- A 1% silver move ≈ 4.76% warrant move

Example (entry $90.55):
| Silver | Warrant % | 150K SEK P&L |
|--------|-----------|-------------|
| $91.20 (+0.72%) | +3.4% | +5,125 SEK |
| $92.00 (+1.60%) | +7.6% | +11,425 SEK |
| $89.00 (-1.71%) | -8.2% | -12,228 SEK |

## Backtesting

Walk-forward validation: for each day D, use only days before D to make predictions.
This prevents overfitting — the model never sees future data.

See `portfolio/orb_backtest.py` for the full backtesting engine.

## Files

| File | Purpose |
|------|---------|
| `portfolio/orb_predictor.py` | Core prediction engine |
| `portfolio/orb_backtest.py` | Walk-forward backtesting |
| `portfolio/orb_postmortem.py` | End-of-day analysis & lessons learned |
| `data/silver_monitor.py` | Real-time integration |
| `data/orb_predictions_today.json` | Today's predictions (written by monitor) |
| `data/orb_postmortem.jsonl` | Historical postmortem log |
| `tests/test_orb_predictor.py` | Test suite |
