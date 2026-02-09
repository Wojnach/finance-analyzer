# Phase 2: Pattern Detection + Strategy Enhancement — Research

## Decision: What Is Phase 2?

**Phase 2 = Comprehensive pattern detection + strategy hardening.**

Avanza stock monitoring is a separate small project (standalone script, ~200 LOC).
FreqAI/ML is Phase 3 (depends on having 2 years of data + validated patterns as features).
Dashboard is Phase 4+ (premature until trading is profitable live).

## Goals

1. Detect ALL major chart/candlestick patterns programmatically
2. Use pattern signals as entry/exit filters alongside existing TA indicators
3. Fix weak exit logic (only 2 exit_signal trades in last backtest, both losing)
4. Add multi-timeframe analysis (1h/4h trend context for 5m trades)
5. Add short positions (futures already enabled)
6. Expand pair whitelist

---

## Pattern Categories & Libraries

### 1. Candlestick Patterns — TA-Lib (already available)

TA-Lib has 61 built-in candlestick pattern functions. Already installed in the Podman container.

Freqtrade has an official example strategy:
https://github.com/freqtrade/freqtrade-strategies/blob/main/user_data/strategies/PatternRecognition.py

Usage:

```python
import talib
prs = talib.get_function_groups()['Pattern Recognition']  # all 61
for pr in prs:
    dataframe[pr] = getattr(talib, pr)(dataframe)
```

Returns: 100 (bullish), -100 (bearish), 0 (neutral) per candle.

Key patterns to prioritize:

- CDLHAMMER, CDLINVERTEDHAMMER (reversal)
- CDLENGULFING (strong reversal)
- CDLMORNINGSTAR, CDLEVENINGSTAR (3-candle reversal)
- CDL3WHITESOLDIERS, CDL3BLACKCROWS (continuation)
- CDLDOJI, CDLDRAGONFLYDOJI (indecision)
- CDLSHOOTINGSTAR, CDLHANGINGMAN (bearish reversal)

### 2. Classic Chart Patterns — TradingPatternScanner or PatternPy

**TradingPatternScanner** (recommended — pip installable, 4 detection algorithms):

- URL: https://github.com/white07S/TradingPatternScanner
- Install: `pip install tradingpattern`
- Stars: 270
- Patterns: Head & Shoulders (+ inverse), Double Top/Bottom, Triangles (asc/desc/sym), Wedges, Channels, Support/Resistance, Higher-High/Lower-Low
- 4 algorithms: Window (78.5%), Savitsky-Golay (78.5%), Kalman (73.5%), Wavelet (84.5%)
- Pure pandas/numpy — no heavy deps

**PatternPy** (alternative — more patterns, no PyPI):

- URL: https://github.com/keithorange/PatternPy
- Stars: 429, last update June 2023 (stale)
- Must clone repo, not on PyPI
- Same pattern set as TradingPatternScanner

### 3. Elliott Wave — python-taew

- URL: https://github.com/DrEdwardPCB/python-taew
- Install: `pip install taew`
- Stars: 22, last update May 2021
- Detects 5-wave impulse patterns (1-2-3-4-5)
- Uses Fibonacci retracement analysis
- No data smoothing required

Alternative: https://github.com/alessioricco/ElliottWaves (31 stars, pandas input)

### 4. Harmonic Patterns — HarmonicPatterns (archived)

- URL: https://github.com/djoffrey/HarmonicPatterns
- Stars: 131, ARCHIVED June 2024
- Patterns: ABCD, Gartley, Bat, AltBat, Butterfly, Crab, DeepCrab, Shark, Cypher
- Requires TA-Lib
- Since it's archived, may need to fork and maintain

### 5. ML-Based Pattern Detection (Future/Phase 3)

- YOLOv8 models exist for chart pattern detection on images (not suitable for Freqtrade)
- foduucom/stockmarket-pattern-detection-yolov8: 93.2% mAP on 6 pattern types
- Better approach: use rule-based detection now, ML later via FreqAI

---

## Complete Pattern Catalog (Target Coverage)

### Reversal Patterns (must-have)

1. Head and Shoulders / Inverse H&S
2. Double Top / Double Bottom
3. Triple Top / Triple Bottom
4. Rounding Top / Rounding Bottom (saucer)
5. Diamond Top / Diamond Bottom
6. Island Reversal (bullish/bearish)
7. V-Pattern / Parabolic Capitulation
8. Bump and Run Reversal

### Continuation Patterns (must-have)

9. Ascending / Descending Triangle
10. Bull Flag / Bear Flag
11. Bull Pennant / Bear Pennant
12. Cup and Handle / Inverse Cup and Handle
13. Rectangle (bullish/bearish)
14. Ascending / Descending Channel
15. Measured Move (up/down)

### Bilateral Patterns

16. Symmetrical Triangle
17. Rising Wedge / Falling Wedge
18. Broadening Formation / Megaphone

### Wave Patterns

19. Elliott Wave (5-wave impulse, ABC correction)
20. Wolfe Wave (bullish/bearish)
21. Three Drives

### Harmonic Patterns

22. Gartley
23. Butterfly
24. Bat / AltBat
25. Crab / DeepCrab
26. Shark
27. Cypher
28. ABCD

### Candlestick Patterns (61 from TA-Lib)

All 61 TA-Lib CDL\* functions — see TA-Lib docs for full list.

### Trap Patterns

29. Bull Trap / Bear Trap
30. Shakeout Pattern
31. Dead Cat Bounce

---

## Strategy Improvements (Phase 2 Scope)

### A. Multi-Timeframe Analysis

Use Freqtrade's `informative_pairs()` to access 1h/4h data from 5m strategy:

```python
def informative_pairs(self):
    pairs = self.dp.current_whitelist()
    return [(pair, '1h') for pair in pairs] + [(pair, '4h') for pair in pairs]
```

Then merge with `merge_informative_pair()` or `@informative` decorator.

Use case: Only enter 5m longs when 1h trend is up (EMA fast > slow, ADX > 25).

### B. New Indicators

| Indicator             | Purpose                                             | Integration        |
| --------------------- | --------------------------------------------------- | ------------------ |
| ADX(14)               | Trend strength filter — avoid choppy markets (< 20) | `talib.ADX()`      |
| Bollinger Bands(20,2) | Volatility + mean reversion entries                 | `talib.BBANDS()`   |
| ATR(14)               | Dynamic stop-loss (2x ATR trailing)                 | `talib.ATR()`      |
| Stochastic RSI        | More sensitive oversold/overbought than RSI         | `talib.STOCHRSI()` |

### C. Exit Logic Overhaul

Current problem: exit signals lose money (-1.17% avg). Most profitable strategies don't rely on exit signals — they use ROI + custom_exit + trailing stop combo.

Implement `custom_exit()`:

- Take profit at 20%+ when RSI overbought
- Take profit at 5-15% when EMA crosses down
- Unclog losing trades held > 2 hours
- Time-based exit for small (1-3%) profits sitting > 1 hour

Implement `custom_stoploss()`:

- ATR-based dynamic stop (2x ATR below entry)
- Tighten as profit increases (10% profit → 1% stop, 5% → 2%, 2% → 3%)
- Time-based tightening (start loose, tighten after 1-2 hours)

### D. Short Positions

Add `can_short = True` to strategy. Mirror long logic for enter_short/exit_short.
Considerations:

- Lower position size for shorts (crypto trends up long-term)
- Tighter stops
- Only short when 1h/4h trend is down + ADX > 25

### E. Pair Expansion

Add high-liquidity futures pairs one at a time (backtest each):

- SOL/USDT:USDT (high volatility, good for momentum)
- XRP/USDT:USDT
- BNB/USDT:USDT

Or use VolumePairList + filters for dynamic selection.

---

## LLM-Powered Trading — Research Summary

### What Exists

- **TradingAgents** (25k stars) — Multi-agent LLM framework (GPT/Claude/Gemini agents as analysts/traders/risk managers). Production-quality architecture but experimental results.
- **AI-Trader** (4-9k stars) — Multi-model competition arena, live at ai4trade.ai
- **FinGPT** (18.5k stars) — Finance-specific LLM, best for sentiment analysis, MIT license

### Reality Check

- No production LLM+Freqtrade integration exists
- LLM API costs prohibitive for high-frequency (5m) trading
- LLMs better suited for: sentiment analysis, news interpretation, longer timeframes
- FreqAI (traditional ML) is more practical for pattern-based signals

### Recommendation

- Phase 2: Rule-based pattern detection (proven, fast, free)
- Phase 3: FreqAI with CatBoost/LightGBM (traditional ML, runs on Deck)
- Phase 4+: Consider LLM integration for sentiment/news (FinGPT or TradingAgents)

---

## Implementation Plan

### Step 1: Candlestick Patterns (TA-Lib)

- Add all 61 CDL\* functions to populate_indicators()
- Create pattern scoring system (weighted bullish/bearish signals)
- Integrate as additional confidence factor in entry/exit

### Step 2: Exit Logic Overhaul

- Implement custom_exit() with profit-based + time-based rules
- Implement custom_stoploss() with ATR-based dynamic stops
- Backtest to verify improvement over current exits

### Step 3: Multi-Timeframe + New Indicators

- Add informative_pairs() for 1h timeframe
- Add ADX, Bollinger Bands, ATR to populate_indicators()
- Use 1h trend as filter for 5m entries

### Step 4: Classic Chart Patterns

- Install TradingPatternScanner in Podman container
- Integrate pattern detection into strategy
- Weight patterns by type (reversal vs continuation)

### Step 5: Short Positions

- Add can_short = True
- Mirror entry/exit logic
- Backtest extensively

### Step 6: Elliott Wave + Harmonic (stretch)

- Install python-taew for Elliott Wave
- Fork HarmonicPatterns for harmonic detection
- These are complex — implement after basics are solid

### Step 7: More Data + Re-Hyperopt

- Download 2 years of data
- Re-run hyperopt with all new indicators/patterns
- Validate on out-of-sample data

---

## Podman Container Changes

The container needs additional Python packages. Update ft.sh or create a custom Dockerfile:

```bash
# Check what's already available
podman run --rm freqtradeorg/freqtrade:stable pip list | grep -E "(catboost|lightgbm|tradingpattern|taew)"

# If missing, either:
# A) Install at runtime in ft.sh (slower but simpler)
# B) Build custom image (faster startup)
```

Packages to add:

- `tradingpattern` (TradingPatternScanner)
- `taew` (Elliott Wave) — optional, Step 6

TA-Lib is already in the Freqtrade image.

---

## References

- Strike Money chart patterns: https://www.strike.money/technical-analysis/chart-patterns
- ChartGuys cheat sheet: https://www.chartguys.com/chart-pattern-cheat-sheet
- Groww candlestick patterns: https://groww.in/blog/candlestick-patterns
- Freqtrade PatternRecognition strategy: https://github.com/freqtrade/freqtrade-strategies/blob/main/user_data/strategies/PatternRecognition.py
- TradingPatternScanner: https://github.com/white07S/TradingPatternScanner
- PatternPy: https://github.com/keithorange/PatternPy
- python-taew: https://github.com/DrEdwardPCB/python-taew
- HarmonicPatterns: https://github.com/djoffrey/HarmonicPatterns
- TradingAgents: https://github.com/TauricResearch/TradingAgents
- FinGPT: https://github.com/AI4Finance-Foundation/FinGPT
- FreqAI docs: https://www.freqtrade.io/en/stable/freqai/
- SRCC Investments PDF: https://www.srcc.edu/sites/default/files/B.Com%28Hons%29_IIIyearVIsem_FundamentalsofInvestments_Week2_DrKanuJain.pdf
