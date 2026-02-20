# Enhanced Composite Signal Modules (12-25)

> **Last updated:** 2026-02-20
> **Source:** `portfolio/signals/` directory (14 modules, ~85 sub-indicators total)

## Overview

Each enhanced composite signal module:
- Reads raw OHLCV data (no external dependencies except macro_regime which takes a macro dict)
- Runs 4-8 sub-indicators internally
- Each sub-indicator votes BUY, SELL, or HOLD independently
- The composite vote is determined by majority voting among active (non-HOLD) voters
- Confidence is the fraction of active voters agreeing with the majority direction
- All modules degrade gracefully with insufficient data (return HOLD with 0.0 confidence)
- All modules return a dict with `action`, `confidence`, `sub_signals`, and `indicators`

### Signal Applicability

- All 14 enhanced modules apply to every ticker (crypto, metals, and stocks)
- They run on the "Now" timeframe only (15m candles, 100 bars)
- Longer timeframes (12h through 6mo) use only the 4 core technical signals (RSI, MACD, EMA, BB)

---

## Signal #12: Trend (`signals/trend.py`)

**Function:** `compute_trend_signal(df)`
**Sub-indicators:** 7
**Minimum data:** 30 rows (200 recommended for full SMA coverage)

| # | Sub-indicator | Buy condition | Sell condition | Hold condition |
|---|---------------|---------------|----------------|----------------|
| 1 | Golden/Death Cross | 50-SMA > 200-SMA AND price > 50-SMA | 50-SMA < 200-SMA AND price < 50-SMA | Insufficient data or transitioning |
| 2 | MA Ribbon | SMA 10/20/50/100/200 in perfect bullish order | Perfect bearish order | Not perfectly aligned |
| 3 | Price vs MA200 | Price > 200-SMA by >1% | Price < 200-SMA by >1% | Within 1% band |
| 4 | Supertrend(10,3) | Price above Supertrend band | Price below Supertrend band | — |
| 5 | Parabolic SAR(0.02,0.2) | SAR below price | SAR above price | — |
| 6 | Ichimoku Cloud | Price above cloud, Tenkan > Kijun, lagging above cloud | Price below cloud, Tenkan < Kijun, lagging below cloud | Mixed signals or inside cloud |
| 7 | ADX(14) + DI | ADX > 25 AND +DI > -DI | ADX > 25 AND -DI > +DI | ADX < 25 (no trend) |

**Notes:**
- The MA Ribbon uses 5 SMAs (10, 20, 50, 100, 200) and checks for perfect sequential order
- Ichimoku uses standard parameters: Tenkan=9, Kijun=26, Senkou B=52
- ADX below 25 means the market is not trending — all DI-based sub-signals abstain

---

## Signal #13: Momentum (`signals/momentum.py`)

**Function:** `compute_momentum_signal(df)`
**Sub-indicators:** 8
**Minimum data:** 50 rows

| # | Sub-indicator | Buy condition | Sell condition | Hold condition |
|---|---------------|---------------|----------------|----------------|
| 1 | RSI Divergence(14) | Bullish divergence: price makes lower low but RSI makes higher low | Bearish divergence: price makes higher high but RSI makes lower high | No divergence detected |
| 2 | Stochastic(14,3,3) | %K < 20 AND %K crosses above %D | %K > 80 AND %K crosses below %D | Between 20-80 |
| 3 | StochRSI(14) | StochRSI < 0.2 (oversold) | StochRSI > 0.8 (overbought) | Between 0.2-0.8 |
| 4 | CCI(20) | CCI < -100 (oversold) | CCI > 100 (overbought) | Between -100 and 100 |
| 5 | Williams %R(14) | %R < -80 (oversold) | %R > -20 (overbought) | Between -80 and -20 |
| 6 | ROC(12) | ROC > 0 AND increasing | ROC < 0 AND decreasing | Near zero or mixed |
| 7 | PPO(12,26,9) | PPO > signal line AND PPO > 0 | PPO < signal line AND PPO < 0 | Mixed conditions |
| 8 | Bull/Bear Power(13) | Bull power > 0 AND increasing | Bear power < 0 AND decreasing | Mixed |

**Notes:**
- RSI Divergence looks back over a configurable window (default 14 periods) for higher lows/lower highs
- Stochastic uses standard %K and %D smoothing with period 3
- PPO (Percentage Price Oscillator) is similar to MACD but uses percentages for cross-asset comparison

---

## Signal #14: Volume Flow (`signals/volume_flow.py`)

**Function:** `compute_volume_flow_signal(df)`
**Sub-indicators:** 6
**Minimum data:** 50 rows

| # | Sub-indicator | Buy condition | Sell condition | Hold condition |
|---|---------------|---------------|----------------|----------------|
| 1 | OBV vs 20-SMA | OBV > 20-period SMA of OBV | OBV < 20-period SMA of OBV | Near SMA |
| 2 | VWAP Cross | Price crosses above VWAP | Price crosses below VWAP | Near VWAP |
| 3 | A/D Line vs 20-SMA | A/D Line > its 20-period SMA | A/D Line < its 20-period SMA | Near SMA |
| 4 | CMF(20) | Chaikin Money Flow > 0.05 | CMF < -0.05 | Between -0.05 and 0.05 |
| 5 | MFI(14) | MFI < 20 (oversold) | MFI > 80 (overbought) | Between 20-80 |
| 6 | Volume RSI(14) | Volume RSI > 60 (buying volume dominant) | Volume RSI < 40 (selling volume dominant) | Between 40-60 |

**Notes:**
- OBV (On Balance Volume) is a cumulative volume indicator showing buying/selling pressure
- VWAP resets intraday for stocks; for crypto it uses a rolling window
- CMF (Chaikin Money Flow) measures money flow volume over a period
- MFI (Money Flow Index) is similar to RSI but incorporates volume
- Volume RSI applies the RSI formula to volume data rather than price

---

## Signal #15: Volatility (`signals/volatility.py`)

**Function:** `compute_volatility_signal(df)`
**Sub-indicators:** 6
**Minimum data:** 50 rows

| # | Sub-indicator | Buy condition | Sell condition | Hold condition |
|---|---------------|---------------|----------------|----------------|
| 1 | BB Squeeze | Bollinger Bands inside Keltner Channels (compression) → release upward | Release downward after squeeze | No squeeze detected |
| 2 | BB Breakout | Price closes above upper BB with volume confirmation | Price closes below lower BB with volume confirmation | Price inside bands |
| 3 | ATR Expansion(14) | ATR expanding from low base + price rising | ATR expanding + price falling | ATR stable or contracting |
| 4 | Keltner Channel(20,1.5) | Price above upper Keltner | Price below lower Keltner | Inside channels |
| 5 | Historical Volatility(20) | HV decreasing from high (volatility compression = setup) | HV increasing rapidly (expansion, risk) | Normal volatility |
| 6 | Donchian Channel(20) | Price at 20-period high (breakout) | Price at 20-period low (breakdown) | Inside channel |

**Notes:**
- BB Squeeze is one of the most reliable sub-indicators in the system — it detects compression (low volatility) followed by expansion (breakout)
- The volatility_sig signal had the highest normalized weight (4.29) as of Feb 19, 2026
- ATR Expansion distinguishes between bullish and bearish volatility expansion using price direction
- Keltner Channels use EMA(20) with 1.5x ATR bands

---

## Signal #16: Candlestick (`signals/candlestick.py`)

**Function:** `compute_candlestick_signal(df)`
**Sub-indicators:** 4 (pattern families)
**Minimum data:** 3 rows (20 recommended for context)

| # | Sub-indicator | Buy patterns | Sell patterns | Hold condition |
|---|---------------|--------------|---------------|----------------|
| 1 | Hammer family | Hammer (downtrend), Inverted Hammer (downtrend) | Shooting Star (uptrend), Hanging Man (uptrend) | No pattern or no trend context |
| 2 | Engulfing | Bullish Engulfing (body fully engulfs prior red candle) | Bearish Engulfing (body fully engulfs prior green candle) | No engulfing pattern |
| 3 | Doji | Doji after downtrend (reversal) | Doji after uptrend (reversal) | Doji without trend context |
| 4 | Star patterns | Morning Star (3-candle bullish reversal) | Evening Star (3-candle bearish reversal) | No star pattern |

**Notes:**
- Patterns are context-dependent: a Hammer is only bullish if it appears after a downtrend
- Trend context is determined by the prior 5-10 candles' direction
- Doji detection uses a body-to-range ratio threshold (body < 10% of total range)
- Engulfing patterns require the current candle's body to fully contain the previous candle's body
- Morning/Evening Star is a 3-candle pattern: large candle, small-bodied candle (star), then large reversal candle

---

## Signal #17: Structure (`signals/structure.py`)

**Function:** `compute_structure_signal(df)`
**Sub-indicators:** 4
**Minimum data:** Uses active voters for confidence (not total sub-indicators)

| # | Sub-indicator | Buy condition | Sell condition | Hold condition |
|---|---------------|---------------|----------------|----------------|
| 1 | Period High/Low Breakout | Price breaks above recent period high | Price breaks below recent period low | Inside range |
| 2 | Donchian(55) Breakout | Price above 55-period high | Price below 55-period low | Inside channel |
| 3 | RSI(14) Centerline Cross | RSI crosses above 50 (from below 48) | RSI crosses below 50 (from above 52) | Inside 48-52 deadband |
| 4 | MACD Zero-Line Cross | MACD line crosses above zero | MACD line crosses below zero | Near zero |

**Notes:**
- The RSI centerline cross uses a 48-52 deadband to filter noise around the 50 level
- Donchian(55) is a longer lookback than the 20-period Donchian in the volatility module — it captures more significant breakouts
- Confidence uses active voters as denominator, not total sub-indicators

---

## Signal #18: Fibonacci (`signals/fibonacci.py`)

**Function:** `compute_fibonacci_signal(df)`
**Sub-indicators:** 5
**Minimum data:** 50 rows (100+ recommended for reliable swing detection)

| # | Sub-indicator | Buy condition | Sell condition | Hold condition |
|---|---------------|---------------|----------------|----------------|
| 1 | Fibonacci Retracement | Price near 50% or 61.8% retracement in uptrend | Price near 50% or 61.8% retracement in downtrend | Not near key levels |
| 2 | Golden Pocket | Price in 61.8%-65% zone in uptrend (high-probability reversal) | Price in 61.8%-65% zone in downtrend | Not in golden pocket zone |
| 3 | Fibonacci Extension | — | Price near 127.2% or 161.8% extension (exhaustion) | Not near extension levels |
| 4 | Standard Pivot Points | Price bouncing off S1/S2 (support) | Price rejecting at R1/R2 (resistance) | Between pivot levels |
| 5 | Camarilla Pivots | Price breaking above R3 (breakout buy) | Price breaking below S3 (breakdown sell) | Inside R3/S3 range |

**Notes:**
- Swing highs/lows are detected automatically from the price data to compute Fibonacci levels
- The Golden Pocket (61.8-65% zone) is considered the highest-probability retracement area
- Extensions at 127.2% and 161.8% signal potential trend exhaustion
- Pivot points (Standard and Camarilla) are computed from the prior period's high, low, and close
- Camarilla R3/S3 breakouts are used as trend-following signals

---

## Signal #19: Smart Money (`signals/smart_money.py`)

**Function:** `compute_smart_money_signal(df)`
**Sub-indicators:** 5
**Minimum data:** 50 rows

| # | Sub-indicator | Buy condition | Sell condition | Hold condition |
|---|---------------|---------------|----------------|----------------|
| 1 | Break of Structure (BOS) | Higher high breaks previous swing high (bullish BOS) | Lower low breaks previous swing low (bearish BOS) | No structure break |
| 2 | Change of Character (CHoCH) | First higher high after series of lower highs (bullish reversal) | First lower low after series of higher lows (bearish reversal) | No character change |
| 3 | Fair Value Gap (FVG) | Bullish FVG: gap between candle 1 high and candle 3 low (unfilled gap above) | Bearish FVG: gap between candle 1 low and candle 3 high (unfilled gap below) | No FVG or already filled |
| 4 | Liquidity Sweep | Price sweeps below recent low then reverses (stop hunt, bullish) | Price sweeps above recent high then reverses (stop hunt, bearish) | No sweep detected |
| 5 | Supply/Demand Zones | Price enters demand zone (prior strong up-move origin) | Price enters supply zone (prior strong down-move origin) | Not near S/D zones |

**Notes:**
- These concepts come from ICT (Inner Circle Trader) / Smart Money Concepts methodology
- BOS confirms trend continuation; CHoCH signals potential trend reversal
- FVGs represent imbalances that price tends to "fill" — they act as magnets for price
- Liquidity sweeps detect false breakouts designed to trigger stop losses before reversing
- Supply/Demand zones are identified by large impulsive moves away from consolidation areas

---

## Signal #20: Oscillators (`signals/oscillators.py`)

**Function:** `compute_oscillators_signal(df)`
**Sub-indicators:** 8
**Minimum data:** 50 rows

| # | Sub-indicator | Buy condition | Sell condition | Hold condition |
|---|---------------|---------------|----------------|----------------|
| 1 | Awesome Oscillator | Zero-cross (neg→pos) or bullish saucer | Zero-cross (pos→neg) or bearish saucer | Near zero, no pattern |
| 2 | Aroon(25) | Aroon Up > 70 AND Aroon Down < 30 | Aroon Down > 70 AND Aroon Up < 30 | Both moderate |
| 3 | Vortex(14) | +VI crosses above -VI | -VI crosses above +VI | Close together |
| 4 | Chande Momentum(9) | CMO < -50 (oversold) | CMO > 50 (overbought) | Between -50 and 50 |
| 5 | KST | KST crosses above signal line AND KST > 0 | KST crosses below signal line AND KST < 0 | Mixed |
| 6 | Schaff Trend Cycle(23,50) | STC < 25 AND rising | STC > 75 AND falling | Between 25-75 |
| 7 | TRIX(15) | TRIX crosses above zero or signal line | TRIX crosses below zero or signal line | Near zero |
| 8 | Coppock Curve(14,11,10) | Coppock crosses above zero (long-term buy signal) | Coppock below zero AND falling | Above zero or mixed |

**Notes:**
- KST (Know Sure Thing) is a multi-ROC composite using 4 different ROC periods (10, 15, 20, 30) with weighted smoothing
- Schaff Trend Cycle combines MACD with Stochastic for faster signal generation
- TRIX is a triple-smoothed EMA momentum indicator — very smooth, few false signals
- Coppock Curve was originally designed as a monthly indicator for long-term buy signals after bear markets
- Awesome Oscillator "saucer" pattern: 3-bar pattern where AO is above zero, the middle bar is lower than the other two

---

## Signal #21: Heikin-Ashi (`signals/heikin_ashi.py`)

**Function:** `compute_heikin_ashi_signal(df)`
**Sub-indicators:** 7
**Minimum data:** 50 rows

| # | Sub-indicator | Buy condition | Sell condition | Hold condition |
|---|---------------|---------------|----------------|----------------|
| 1 | HA Trend | 3+ consecutive strong green HA candles (no lower shadow) | 3+ consecutive strong red HA candles (no upper shadow) | Mixed candles |
| 2 | HA Doji | Doji after red streak (potential bullish reversal) | Doji after green streak (potential bearish reversal) | Doji without context |
| 3 | HA Color Change | Red to green transition | Green to red transition | No change |
| 4 | Hull MA Cross(9,21) | Hull MA(9) crosses above Hull MA(21) | Hull MA(9) crosses below Hull MA(21) | Close together |
| 5 | Williams Alligator | Lips > Teeth > Jaw (SMMA 5 > 8 > 13, shifted) — "alligator eating" bullish | Jaw > Teeth > Lips — bearish | Intertwined ("sleeping") |
| 6 | Elder Impulse System | EMA(13) rising AND MACD histogram rising (green bar) | EMA(13) falling AND MACD histogram falling (red bar) | Mixed (blue bar) |
| 7 | TTM Squeeze | BB inside Keltner (squeeze on) → momentum positive on release | Squeeze on → momentum negative on release | No squeeze or squeeze still building |

**Notes:**
- Heikin-Ashi candles are computed from regular OHLC: HA_Close = (O+H+L+C)/4, HA_Open = (prev_HA_Open + prev_HA_Close)/2
- Strong HA candles have no shadow on the trend side (no lower shadow = strong bull, no upper shadow = strong bear)
- Williams Alligator uses SMMA (Smoothed Moving Average) with periods 13/8/5 and offsets 8/5/3
- Elder Impulse combines trend (EMA direction) with momentum (MACD histogram direction)
- TTM Squeeze is the same concept as in volatility.py (BB inside Keltner) but interpreted through momentum direction on release

---

## Signal #22: Mean Reversion (`signals/mean_reversion.py`)

**Function:** `compute_mean_reversion_signal(df)`
**Sub-indicators:** 7
**Minimum data:** Uses active voters for confidence

| # | Sub-indicator | Buy condition | Sell condition | Hold condition |
|---|---------------|---------------|----------------|----------------|
| 1 | RSI(2) | RSI(2) < 10 (extreme oversold) | RSI(2) > 90 (extreme overbought) | Between 10-90 |
| 2 | RSI(3) | RSI(3) < 15 | RSI(3) > 85 | Between 15-85 |
| 3 | IBS (Internal Bar Strength) | IBS < 0.2 (close near low) | IBS > 0.8 (close near high) | Between 0.2-0.8 |
| 4 | Consecutive Down Days | 3+ consecutive down days | 3+ consecutive up days | < 3 consecutive |
| 5 | Gap Fade/Fill | Gap down > 0.5% with 30%+ fill (gap is being closed, bullish) | Gap up > 0.5% with 30%+ fill (gap is being closed, bearish) | No significant gap |
| 6 | BB %B | %B < 0 (price below lower BB) | %B > 1 (price above upper BB) | Between 0 and 1 |
| 7 | IBS + RSI(2) Combined | IBS < 0.2 AND RSI(2) < 10 (double confirmation) | IBS > 0.8 AND RSI(2) > 90 | No double confirmation |

**Notes:**
- Mean reversion signals are contrarian by nature — they buy oversold conditions and sell overbought
- RSI(2) and RSI(3) are ultra-short-period RSI variants designed for mean-reversion trading
- IBS = (Close - Low) / (High - Low) — measures where the close falls within the bar's range
- The IBS + RSI(2) combined sub-indicator requires both conditions simultaneously for higher conviction
- These signals work best in ranging markets and can be trapped in trending markets
- Gap Fade looks for gaps that are being filled (>30% of the gap closed), suggesting mean reversion

---

## Signal #23: Calendar / Seasonal (`signals/calendar_seasonal.py`)

**Function:** `compute_calendar_signal(df)`
**Sub-indicators:** 8
**Minimum data:** 2 rows (for Turnaround Tuesday check)
**Max confidence:** Capped at 0.6 (calendar signals are inherently weak)

| # | Sub-indicator | Buy condition | Sell condition | Hold condition |
|---|---------------|---------------|----------------|----------------|
| 1 | Day-of-Week Effect | Friday (historically bullish) | Monday (historically bearish) | Tue/Wed/Thu |
| 2 | Turnaround Tuesday | Tuesday AND prior bar was red (Monday reversal) | — | Not Tuesday, or Monday was green |
| 3 | Month-End Effect | Last 3 calendar days of month | — | Not month-end |
| 4 | Sell in May | Nov, Dec, Jan, Apr (strong months) | May through October (weak half) | Feb, Mar (transitional) |
| 5 | January Effect | January (small caps rally) | December (tax-loss selling) | Feb-Nov |
| 6 | Pre-Holiday Effect | Day before US market holiday | — | Normal trading day |
| 7 | FOMC Drift | 1-2 days before FOMC announcement | — | FOMC day, day after FOMC, or far from FOMC |
| 8 | Santa Claus Rally | Dec 24-31 or Jan 1-3 | — | Rest of year |

**Known issues:**
- **100% BUY activation bias:** Almost every invocation triggers at least one BUY sub-signal
  (Day-of-Week on Fridays, Month-End near end of month, Sell in May during Nov-Apr, etc.).
  Very few sub-indicators ever vote SELL (only Monday and May-Oct). This creates a permanent
  BUY bias that is not predictive.
- **Normalized weight: 0.07** — the lowest of all 25 signals due to the bias correction in
  weighted consensus. Effectively near-zero influence on the composite vote.
- FOMC dates are imported from `portfolio/fomc_dates.py` (hardcoded for 2026)
- Pre-holiday dates are approximate — does not handle observed-date shifts

---

## Signal #24: Macro Regime (`signals/macro_regime.py`)

**Function:** `compute_macro_regime_signal(df, macro=None)`
**Sub-indicators:** 6
**Minimum data:** 200 rows for full SMA coverage; macro dict for external data
**External dependency:** Requires `macro` dict with DXY, treasury, and Fed calendar data

| # | Sub-indicator | Buy condition | Sell condition | Hold condition |
|---|---------------|---------------|----------------|----------------|
| 1 | 200-SMA Regime Filter | Price > 200-SMA by >1% | Price < 200-SMA by >1% | Within 1% transition band, or <200 rows |
| 2 | DXY vs Risk Assets | DXY 5d change < -0.5% (weak dollar, good for risk) | DXY 5d change > +0.5% (strong dollar, bad for risk) | DXY change between -0.5% and +0.5%, or no data |
| 3 | Yield Curve (2s10s) | Spread > 0.5% (normal, healthy) | Spread < 0% (inverted, recession risk) | Between 0% and 0.5% (watch zone), or no data |
| 4 | 10Y Yield Momentum | 10Y < 3.5% (easy money) | 10Y > 5.0% (tight conditions) | Between 3.5% and 5.0%, or no data |
| 5 | FOMC Proximity | — | — | Always HOLD (within 3 days = caution, >3 days = no edge) |
| 6 | Golden/Death Cross | 50-SMA > 200-SMA AND price > 50-SMA | 50-SMA < 200-SMA AND price < 50-SMA | Transitioning or <200 rows |

**Known issues:**
- **0% activation rate:** This signal effectively never votes BUY or SELL in practice.
  The FOMC Proximity sub-signal always votes HOLD (by design — it was defanged to remove a
  permanent BUY bias). The DXY, yield curve, and yield momentum sub-signals require external
  macro data that may be stale or missing, causing them to vote HOLD. The SMA-based
  sub-signals require 200 rows of data which is only available on the Now timeframe.
- **Normalized weight: 0.00** — dead signal due to 0% activation
- When `macro` parameter is None or missing keys, all macro-dependent sub-signals vote HOLD
- The 200-SMA and Golden/Death Cross sub-signals overlap somewhat (both use 200-SMA)

---

## Signal #25: Momentum Factors (`signals/momentum_factors.py`)

**Function:** `compute_momentum_factors_signal(df)`
**Sub-indicators:** 7
**Minimum data:** Uses active voters for confidence

| # | Sub-indicator | Buy condition | Sell condition | Hold condition |
|---|---------------|---------------|----------------|----------------|
| 1 | Time-Series Momentum (12-1) | 12-month return minus 1-month return > 0 (medium-term trend up) | TSM < 0 (medium-term trend down) | Near zero, or insufficient data |
| 2 | ROC-20 | 20-period Rate of Change > 5% | ROC-20 < -5% | Between -5% and 5% |
| 3 | 52-Week High Proximity | Price within 5% of 52-week high (momentum continuation) | — | More than 5% from high |
| 4 | 52-Week Low Reversal | Price within 5% of 52-week low (contrarian bounce) | — | More than 5% from low |
| 5 | Consecutive Bars | 4+ consecutive up bars | 4+ consecutive down bars | < 4 consecutive |
| 6 | Price Acceleration | ROC is increasing (momentum accelerating upward) | ROC is decreasing (momentum accelerating downward) | Flat acceleration |
| 7 | Volume-Weighted Momentum | Volume-weighted price change positive over lookback | Volume-weighted price change negative | Near zero |

**Notes:**
- Time-Series Momentum (TSM 12-1) is a well-documented factor in academic finance: the past 12-month return (excluding the most recent month) predicts future returns
- 52-Week High Proximity is a momentum continuation signal (stocks near highs tend to keep going)
- 52-Week Low Reversal is a contrarian/mean-reversion signal (stocks near lows tend to bounce)
- The combination of momentum and mean-reversion sub-indicators within one module means they can partially cancel each other, which is by design — the module captures whichever force is stronger
- Volume-Weighted Momentum gives more weight to price moves that occurred on high volume

---

## Summary Table

| # | Signal | Sub-indicators | Min data | Special requirements | Known issues |
|---|--------|---------------|----------|---------------------|-------------|
| 12 | Trend | 7 | 30 (200 rec.) | — | — |
| 13 | Momentum | 8 | 50 | — | — |
| 14 | Volume Flow | 6 | 50 | — | — |
| 15 | Volatility | 6 | 50 | — | Highest weight (4.29) |
| 16 | Candlestick | 4 | 3 (20 rec.) | — | — |
| 17 | Structure | 4 | — | Active voter confidence | — |
| 18 | Fibonacci | 5 | 50 (100+ rec.) | — | — |
| 19 | Smart Money | 5 | 50 | — | — |
| 20 | Oscillators | 8 | 50 | — | — |
| 21 | Heikin-Ashi | 7 | 50 | — | — |
| 22 | Mean Reversion | 7 | — | Active voter confidence | Contrarian, risky in trends |
| 23 | Calendar | 8 | 2 | FOMC dates from fomc_dates.py | 100% BUY bias, weight 0.07 |
| 24 | Macro Regime | 6 | 200 | Requires macro dict | 0% activation, dead signal |
| 25 | Momentum Factors | 7 | — | Active voter confidence | — |

**Total: 14 modules, ~85 sub-indicators**
