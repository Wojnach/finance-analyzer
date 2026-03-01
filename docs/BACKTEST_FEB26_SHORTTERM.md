# Backtest: February 26, 2026 -- SHORT-TERM Accuracy

**Date:** 2026-02-26
**Entries analyzed:** 26
**Time range:** 2026-02-26T12:51:25 -> 2026-02-26T18:08:47 UTC
**Horizons:** 15min, 30min, 1h, 3h
**Data source:** Binance 1-minute klines (BTC/ETH/XAU/XAG), existing backfill (stocks)

> **Purpose:** Evaluate whether signals can predict price direction in the
> next 15 minutes to 3 hours -- the timeframe that matters for active trading.

## Accuracy Summary (All Horizons)

| Horizon | Consensus Accuracy | Correct/Total | 70% Met? |
|---------|-------------------|---------------|----------|
| 15min | **58.6%** | 17/29 | NO |
| 30min | **51.7%** | 15/29 | NO |
| 1h | **48.3%** | 14/29 | NO |
| 3h | **54.4%** | 186/342 | NO |

---

## 15min Horizon

### Overall Consensus: **58.6%** (17/29)

### Per-Signal Accuracy (raw vote count)

| Signal | Correct | Total | Accuracy | BUY | SELL |
|--------|---------|-------|----------|-----|------|
| volume | 13 | 16 | **81.2%** | 1 | 15 |
| momentum_factors | 2 | 3 | **66.7%** | 0 | 3 |
| bb | 10 | 16 | **62.5%** | 16 | 0 |
| mean_reversion | 7 | 12 | **58.3%** | 12 | 0 |
| candlestick | 4 | 7 | **57.1%** | 4 | 3 |
| ministral | 19 | 35 | **54.3%** | 2 | 33 |
| trend | 10 | 19 | **52.6%** | 0 | 19 |
| ema | 6 | 12 | **50.0%** | 0 | 12 |
| fear_greed | 24 | 52 | **46.2%** | 52 | 0 |
| rsi | 16 | 36 | **44.4%** | 36 | 0 |
| heikin_ashi | 9 | 21 | **42.9%** | 0 | 21 |
| volume_flow | 13 | 31 | **41.9%** | 4 | 27 |
| news_event | 1 | 3 | **33.3%** | 0 | 3 |
| macd | 1 | 5 | **20.0%** | 1 | 4 |
| structure | 1 | 5 | **20.0%** | 0 | 5 |
| volatility_sig | 0 | 7 | **0.0%** | 0 | 7 |
| oscillators | 0 | 2 | **0.0%** | 0 | 2 |

### Per-Signal Accuracy (UNIQUE predictions -- deduped by ticker+direction)

> Each (signal, ticker, direction) combo counted only ONCE, not repeated across 26 entries

| Signal | Correct | Unique Predictions | Accuracy |
|--------|---------|-------------------|----------|
| volume | 3 | 3 | **100.0%** |
| trend | 2 | 2 | **100.0%** |
| news_event | 1 | 1 | **100.0%** |
| fear_greed | 1 | 2 | **50.0%** |
| rsi | 1 | 2 | **50.0%** |
| bb | 1 | 2 | **50.0%** |
| mean_reversion | 1 | 2 | **50.0%** |
| candlestick | 2 | 4 | **50.0%** |
| structure | 1 | 2 | **50.0%** |
| ema | 1 | 2 | **50.0%** |
| momentum_factors | 1 | 2 | **50.0%** |
| volume_flow | 1 | 3 | **33.3%** |
| ministral | 1 | 3 | **33.3%** |
| macd | 0 | 2 | **0.0%** |
| heikin_ashi | 0 | 2 | **0.0%** |
| volatility_sig | 0 | 2 | **0.0%** |
| oscillators | 0 | 1 | **0.0%** |

### Per-Ticker Consensus Accuracy

| Ticker | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| ETH-USD | 10 | 15 | **66.7%** |
| BTC-USD | 7 | 14 | **50.0%** |

---

## 30min Horizon

### Overall Consensus: **51.7%** (15/29)

### Per-Signal Accuracy (raw vote count)

| Signal | Correct | Total | Accuracy | BUY | SELL |
|--------|---------|-------|----------|-----|------|
| momentum_factors | 3 | 3 | **100.0%** | 0 | 3 |
| candlestick | 5 | 7 | **71.4%** | 4 | 3 |
| volume | 10 | 16 | **62.5%** | 1 | 15 |
| heikin_ashi | 13 | 21 | **61.9%** | 0 | 21 |
| ministral | 21 | 35 | **60.0%** | 2 | 33 |
| macd | 3 | 5 | **60.0%** | 1 | 4 |
| ema | 7 | 12 | **58.3%** | 0 | 12 |
| volume_flow | 17 | 31 | **54.8%** | 4 | 27 |
| trend | 10 | 19 | **52.6%** | 0 | 19 |
| bb | 8 | 16 | **50.0%** | 16 | 0 |
| mean_reversion | 6 | 12 | **50.0%** | 12 | 0 |
| oscillators | 1 | 2 | **50.0%** | 0 | 2 |
| rsi | 15 | 36 | **41.7%** | 36 | 0 |
| fear_greed | 21 | 52 | **40.4%** | 52 | 0 |
| structure | 2 | 5 | **40.0%** | 0 | 5 |
| news_event | 1 | 3 | **33.3%** | 0 | 3 |
| volatility_sig | 2 | 7 | **28.6%** | 0 | 7 |

### Per-Signal Accuracy (UNIQUE predictions -- deduped by ticker+direction)

> Each (signal, ticker, direction) combo counted only ONCE, not repeated across 26 entries

| Signal | Correct | Unique Predictions | Accuracy |
|--------|---------|-------------------|----------|
| volume_flow | 3 | 3 | **100.0%** |
| ema | 2 | 2 | **100.0%** |
| momentum_factors | 2 | 2 | **100.0%** |
| news_event | 1 | 1 | **100.0%** |
| candlestick | 3 | 4 | **75.0%** |
| volume | 2 | 3 | **66.7%** |
| rsi | 1 | 2 | **50.0%** |
| bb | 1 | 2 | **50.0%** |
| mean_reversion | 1 | 2 | **50.0%** |
| structure | 1 | 2 | **50.0%** |
| trend | 1 | 2 | **50.0%** |
| ministral | 1 | 3 | **33.3%** |
| fear_greed | 0 | 2 | **0.0%** |
| macd | 0 | 2 | **0.0%** |
| heikin_ashi | 0 | 2 | **0.0%** |
| volatility_sig | 0 | 2 | **0.0%** |
| oscillators | 0 | 1 | **0.0%** |

### Per-Ticker Consensus Accuracy

| Ticker | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| ETH-USD | 9 | 15 | **60.0%** |
| BTC-USD | 6 | 14 | **42.9%** |

---

## 1h Horizon

### Overall Consensus: **48.3%** (14/29)

### Per-Signal Accuracy (raw vote count)

| Signal | Correct | Total | Accuracy | BUY | SELL |
|--------|---------|-------|----------|-----|------|
| momentum_factors | 3 | 3 | **100.0%** | 0 | 3 |
| macd | 4 | 5 | **80.0%** | 1 | 4 |
| volume | 12 | 16 | **75.0%** | 1 | 15 |
| ministral | 25 | 35 | **71.4%** | 2 | 33 |
| heikin_ashi | 13 | 21 | **61.9%** | 0 | 21 |
| ema | 7 | 12 | **58.3%** | 0 | 12 |
| trend | 11 | 19 | **57.9%** | 0 | 19 |
| candlestick | 4 | 7 | **57.1%** | 4 | 3 |
| volume_flow | 16 | 31 | **51.6%** | 4 | 27 |
| oscillators | 1 | 2 | **50.0%** | 0 | 2 |
| bb | 7 | 16 | **43.8%** | 16 | 0 |
| mean_reversion | 5 | 12 | **41.7%** | 12 | 0 |
| structure | 2 | 5 | **40.0%** | 0 | 5 |
| rsi | 13 | 36 | **36.1%** | 36 | 0 |
| news_event | 1 | 3 | **33.3%** | 0 | 3 |
| fear_greed | 15 | 52 | **28.8%** | 52 | 0 |
| volatility_sig | 2 | 7 | **28.6%** | 0 | 7 |

### Per-Signal Accuracy (UNIQUE predictions -- deduped by ticker+direction)

> Each (signal, ticker, direction) combo counted only ONCE, not repeated across 26 entries

| Signal | Correct | Unique Predictions | Accuracy |
|--------|---------|-------------------|----------|
| trend | 2 | 2 | **100.0%** |
| ema | 2 | 2 | **100.0%** |
| momentum_factors | 2 | 2 | **100.0%** |
| news_event | 1 | 1 | **100.0%** |
| volume_flow | 2 | 3 | **66.7%** |
| volume | 2 | 3 | **66.7%** |
| macd | 1 | 2 | **50.0%** |
| heikin_ashi | 1 | 2 | **50.0%** |
| candlestick | 2 | 4 | **50.0%** |
| structure | 1 | 2 | **50.0%** |
| ministral | 1 | 3 | **33.3%** |
| fear_greed | 0 | 2 | **0.0%** |
| rsi | 0 | 2 | **0.0%** |
| bb | 0 | 2 | **0.0%** |
| mean_reversion | 0 | 2 | **0.0%** |
| volatility_sig | 0 | 2 | **0.0%** |
| oscillators | 0 | 1 | **0.0%** |

### Per-Ticker Consensus Accuracy

| Ticker | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| ETH-USD | 8 | 15 | **53.3%** |
| BTC-USD | 6 | 14 | **42.9%** |

---

## 3h Horizon

### Overall Consensus: **54.4%** (186/342)

### Per-Signal Accuracy (raw vote count)

| Signal | Correct | Total | Accuracy | BUY | SELL |
|--------|---------|-------|----------|-----|------|
| mean_reversion | 82 | 119 | **68.9%** | 87 | 32 |
| rsi | 263 | 385 | **68.3%** | 318 | 67 |
| smart_money | 19 | 29 | **65.5%** | 21 | 8 |
| macro_regime | 20 | 31 | **64.5%** | 31 | 0 |
| bb | 110 | 173 | **63.6%** | 121 | 52 |
| news_event | 32 | 54 | **59.3%** | 1 | 53 |
| sentiment | 261 | 442 | **59.0%** | 434 | 8 |
| ministral | 19 | 35 | **54.3%** | 2 | 33 |
| candlestick | 46 | 102 | **45.1%** | 50 | 52 |
| macd | 28 | 63 | **44.4%** | 25 | 38 |
| fear_greed | 22 | 52 | **42.3%** | 52 | 0 |
| fibonacci | 22 | 54 | **40.7%** | 4 | 50 |
| volatility_sig | 53 | 137 | **38.7%** | 39 | 98 |
| volume_flow | 203 | 539 | **37.7%** | 204 | 335 |
| heikin_ashi | 101 | 268 | **37.7%** | 108 | 160 |
| ema | 107 | 290 | **36.9%** | 106 | 184 |
| structure | 29 | 80 | **36.2%** | 39 | 41 |
| volume | 80 | 224 | **35.7%** | 70 | 154 |
| trend | 106 | 301 | **35.2%** | 99 | 202 |
| momentum_factors | 18 | 53 | **34.0%** | 11 | 42 |
| oscillators | 0 | 12 | **0.0%** | 3 | 9 |

### Per-Signal Accuracy (UNIQUE predictions -- deduped by ticker+direction)

> Each (signal, ticker, direction) combo counted only ONCE, not repeated across 26 entries

| Signal | Correct | Unique Predictions | Accuracy |
|--------|---------|-------------------|----------|
| ministral | 2 | 3 | **66.7%** |
| macro_regime | 5 | 8 | **62.5%** |
| candlestick | 24 | 40 | **60.0%** |
| smart_money | 5 | 9 | **55.6%** |
| mean_reversion | 17 | 31 | **54.8%** |
| volatility_sig | 14 | 27 | **51.9%** |
| news_event | 5 | 11 | **45.5%** |
| macd | 14 | 31 | **45.2%** |
| volume | 19 | 43 | **44.2%** |
| heikin_ashi | 19 | 43 | **44.2%** |
| bb | 14 | 32 | **43.8%** |
| rsi | 14 | 35 | **40.0%** |
| ema | 9 | 24 | **37.5%** |
| volume_flow | 15 | 42 | **35.7%** |
| structure | 8 | 23 | **34.8%** |
| trend | 9 | 26 | **34.6%** |
| sentiment | 7 | 21 | **33.3%** |
| fibonacci | 4 | 16 | **25.0%** |
| momentum_factors | 3 | 15 | **20.0%** |
| fear_greed | 0 | 2 | **0.0%** |
| oscillators | 0 | 7 | **0.0%** |

### Per-Ticker Consensus Accuracy

| Ticker | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| XAU-USD | 2 | 2 | **100.0%** |
| META | 1 | 1 | **100.0%** |
| XAG-USD | 2 | 2 | **100.0%** |
| GRRR | 12 | 15 | **80.0%** |
| AAPL | 4 | 5 | **80.0%** |
| NVDA | 12 | 15 | **80.0%** |
| SOUN | 11 | 14 | **78.6%** |
| QQQ | 5 | 7 | **71.4%** |
| VERI | 14 | 20 | **70.0%** |
| GOOGL | 13 | 19 | **68.4%** |
| TSM | 12 | 18 | **66.7%** |
| ETH-USD | 10 | 15 | **66.7%** |
| TTWO | 6 | 9 | **66.7%** |
| AMZN | 11 | 18 | **61.1%** |
| VRT | 8 | 14 | **57.1%** |
| TEM | 13 | 25 | **52.0%** |
| BTC-USD | 7 | 14 | **50.0%** |
| PLTR | 1 | 2 | **50.0%** |
| UPST | 5 | 11 | **45.5%** |
| AMD | 7 | 16 | **43.8%** |
| LMT | 6 | 15 | **40.0%** |
| IONQ | 6 | 15 | **40.0%** |
| MU | 6 | 16 | **37.5%** |
| SMCI | 4 | 14 | **28.6%** |
| AVGO | 3 | 11 | **27.3%** |
| MSTR | 2 | 10 | **20.0%** |
| BABA | 3 | 19 | **15.8%** |

---

## Price Movement Timeline (BTC & ETH)

> Shows actual % price change at each horizon from signal entry time

| Time | BTC Price | BTC Consensus | 15m% | 30m% | 1h% | 3h% | ETH Price | ETH Consensus | 15m% | 30m% | 1h% | 3h% |
|------|-----------|---------------|------|------|-----|-----|-----------|---------------|------|------|-----|-----|
| 12:51:25 | $68,177 | HOLD | +0.03 | -0.24 | -0.17 | -1.18 | $2,069.25 | HOLD | -0.02 | -0.29 | -0.01 | -1.71 |
| 13:03:44 | $68,174 | HOLD | -0.23 | -0.05 | -0.52 | -1.22 | $2,067.54 | HOLD | -0.19 | 0.00 | -0.23 | -1.83 |
| 13:23:24 | $67,981 | HOLD | +0.07 | 0.00 | +0.21 | -1.03 | $2,062.71 | HOLD | +0.28 | +0.22 | +0.34 | -1.56 |
| 13:42:17 | $68,058 | HOLD | -0.23 | -0.05 | -0.20 | -1.39 | $2,068.12 | HOLD | -0.14 | +0.01 | -0.24 | -2.51 |
| 14:02:11 | $67,793 | HOLD | +0.11 | +0.17 | -0.27 | -1.45 | $2,061.52 | HOLD | +0.11 | +0.12 | -0.60 | -3.65 |
| 14:07:19 | $67,718 | BUY | +0.39 | +0.41 | -0.29 | -1.15 | $2,060.33 | HOLD | +0.26 | +0.24 | -0.54 | -3.21 |
| 14:21:44 | $67,908 | BUY | +0.15 | -0.85 | -1.47 | -1.52 | $2,064.69 | HOLD | 0.00 | -1.08 | -2.18 | -3.54 |
| 14:26:13 | $68,107 | BUY | -0.26 | -0.56 | -1.38 | -1.73 | $2,069.98 | SELL | -0.32 | -0.81 | -1.89 | -3.73 |
| 14:41:12 | $67,929 | HOLD | -0.30 | -0.50 | -0.86 | -1.63 | $2,063.31 | HOLD | -0.48 | -0.58 | -1.73 | -3.62 |
| 14:45:29 | $67,421 | BUY | +0.32 | -0.18 | +0.05 | -0.95 | $2,046.75 | HOLD | +0.09 | -0.54 | -0.66 | -2.85 |
| 15:00:37 | $67,636 | BUY | -0.50 | -0.82 | -0.30 | -1.23 | $2,048.63 | BUY | -0.63 | -1.21 | -0.54 | -2.90 |
| 15:06:34 | $67,582 | BUY | -1.00 | -0.74 | -0.67 | -1.29 | $2,050.28 | BUY | -1.49 | -1.44 | -1.27 | -3.28 |
| 15:19:57 | $66,965 | BUY | -0.22 | +0.71 | +0.36 | +0.06 | $2,023.70 | SELL | -0.49 | +0.59 | +0.15 | -1.51 |
| 15:25:51 | $67,012 | BUY | +0.37 | +0.45 | +0.60 | -0.07 | $2,023.19 | BUY | +0.09 | +0.42 | +0.52 | -1.76 |
| 15:39:00 | $67,292 | BUY | +0.07 | -0.25 | -0.31 | -0.07 | $2,025.32 | BUY | +0.40 | -0.04 | -0.52 | -1.05 |
| 15:57:54 | $67,405 | HOLD | -0.43 | +0.10 | -1.17 | -0.11 | $2,033.31 | BUY | -0.39 | +0.12 | -2.46 | -1.38 |
| 16:04:41 | $67,329 | HOLD | -0.18 | -0.19 | -0.62 | +0.63 | $2,029.04 | HOLD | -0.11 | -0.38 | -1.85 | -0.18 |
| 16:18:53 | $67,225 | BUY | -0.06 | -0.43 | -0.48 | +0.46 | $2,028.24 | BUY | -0.20 | -1.59 | -1.73 | -0.20 |
| 16:32:28 | $67,280 | BUY | -0.81 | -0.70 | -0.45 | +0.53 | $2,029.45 | SELL | -1.95 | -2.13 | -1.66 | -0.12 |
| 16:38:32 | $67,087 | HOLD | -0.64 | -0.26 | -0.35 | +0.73 | $2,014.69 | SELL | -1.48 | -1.05 | -1.29 | +0.52 |
| 16:53:42 | $66,660 | HOLD | +0.38 | +0.18 | +0.28 | +1.32 | $1,984.85 | BUY | +0.43 | +0.12 | +0.28 | +2.12 |
| 16:59:41 | $66,644 | BUY | +0.37 | +0.44 | +0.28 | +1.42 | $1,983.80 | BUY | +0.29 | +0.43 | +0.23 | +2.34 |
| 17:13:58 | $66,959 | SELL | +0.03 | -0.29 | -0.39 | +1.13 | $1,991.22 | BUY | +0.06 | -0.16 | -0.56 | +2.21 |
| 17:29:18 | $66,936 | SELL | -0.28 | -0.16 | -0.11 | +0.93 | $1,992.35 | SELL | -0.22 | -0.20 | -0.28 | +1.79 |
| 17:47:38 | $66,772 | HOLD | +0.06 | +0.35 | +0.74 | +0.98 | $1,989.21 | BUY | -0.02 | +0.15 | +0.80 | +1.96 |
| 18:08:47 | $66,588 | HOLD | +0.43 | +1.06 | +1.48 | +0.99 | $1,977.00 | HOLD | +0.44 | +1.40 | +2.25 | +2.38 |

---

## Key Findings

- **Best horizon:** 15min at 58.6%
- **Worst horizon:** 1h at 48.3%

**Top 5 signals at 15min** (>=3 votes):
  - `volume`: 81.2% (13/16)
  - `momentum_factors`: 66.7% (2/3)
  - `bb`: 62.5% (10/16)
  - `mean_reversion`: 58.3% (7/12)
  - `candlestick`: 57.1% (4/7)
**Bottom 5 signals at 15min:**
  - `volume_flow`: 41.9% (13/31)
  - `news_event`: 33.3% (1/3)
  - `macd`: 20.0% (1/5)
  - `structure`: 20.0% (1/5)
  - `volatility_sig`: 0.0% (0/7)

**Top 5 signals at 1h** (>=3 votes):
  - `momentum_factors`: 100.0% (3/3)
  - `macd`: 80.0% (4/5)
  - `volume`: 75.0% (12/16)
  - `ministral`: 71.4% (25/35)
  - `heikin_ashi`: 61.9% (13/21)
**Bottom 5 signals at 1h:**
  - `structure`: 40.0% (2/5)
  - `rsi`: 36.1% (13/36)
  - `news_event`: 33.3% (1/3)
  - `fear_greed`: 28.8% (15/52)
  - `volatility_sig`: 28.6% (2/7)

### Direction Consistency

> Did the price move in the same direction at all horizons?

**BTC-USD:**
  12:51:25: 15min=UP(+0.03%), 30min=DOWN(-0.24%), 1h=DOWN(-0.17%), 3h=DOWN(-1.18%)
  13:03:44: 15min=DOWN(-0.23%), 30min=DOWN(-0.05%), 1h=DOWN(-0.52%), 3h=DOWN(-1.22%)
  13:23:24: 15min=UP(+0.07%), 30min=UP(+0.00%), 1h=UP(+0.21%), 3h=DOWN(-1.03%)
  13:42:17: 15min=DOWN(-0.23%), 30min=DOWN(-0.05%), 1h=DOWN(-0.20%), 3h=DOWN(-1.39%)
  14:02:11: 15min=UP(+0.11%), 30min=UP(+0.17%), 1h=DOWN(-0.27%), 3h=DOWN(-1.45%)

**ETH-USD:**
  12:51:25: 15min=DOWN(-0.02%), 30min=DOWN(-0.29%), 1h=DOWN(-0.01%), 3h=DOWN(-1.71%)
  13:03:44: 15min=DOWN(-0.19%), 30min=DOWN(-0.00%), 1h=DOWN(-0.23%), 3h=DOWN(-1.83%)
  13:23:24: 15min=UP(+0.28%), 30min=UP(+0.22%), 1h=UP(+0.34%), 3h=DOWN(-1.56%)
  13:42:17: 15min=DOWN(-0.14%), 30min=UP(+0.01%), 1h=DOWN(-0.24%), 3h=DOWN(-2.51%)
  14:02:11: 15min=UP(+0.11%), 30min=UP(+0.12%), 1h=DOWN(-0.60%), 3h=DOWN(-3.65%)

### Conclusion

**No horizon meets the 70% accuracy threshold for short-term prediction.**

This means the current signal system is not reliably predicting
price direction in the next 15 minutes to 3 hours on Feb 26.
The signals were designed for daily/multi-day timeframes, which
explains why short-term accuracy is lower.