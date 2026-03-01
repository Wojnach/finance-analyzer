# Backtest: February 26, 2026

**Date:** 2026-02-26
**Entries analyzed:** 26
**Time range:** 2026-02-26T12:51:25 -> 2026-02-26T18:08:47 UTC
**Tickers:** 30
**Horizons evaluated:** 1d, 3d

**Trigger types:** AAPL consensus BUY, AAPL sentiment positive->negative, AMD consensus SELL, AMD moved 2.2% down, AMZN consensus BUY, AMZN consensus SELL, AMZN sentiment positive->negative, AVGO consensus SELL, AVGO moved 2.6% up, AVGO moved 2.7% down

---

## 1d Horizon

### Overall Consensus Accuracy: **53.5%** (183/342) — 70% threshold: **NO**

### Per-Signal Accuracy

| Signal | Correct | Total | Accuracy | BUY votes | SELL votes |
|--------|---------|-------|----------|-----------|------------|
| ministral | 33 | 35 | **94.3%** | 2 | 33 |
| mean_reversion | 72 | 119 | **60.5%** | 87 | 32 |
| fibonacci | 31 | 54 | **57.4%** | 4 | 50 |
| macd | 36 | 63 | **57.1%** | 25 | 38 |
| bb | 98 | 173 | **56.6%** | 121 | 52 |
| rsi | 202 | 385 | **52.5%** | 318 | 67 |
| volume | 108 | 224 | **48.2%** | 70 | 154 |
| volume_flow | 247 | 539 | **45.8%** | 204 | 335 |
| volatility_sig | 59 | 137 | **43.1%** | 39 | 98 |
| news_event | 22 | 54 | **40.7%** | 1 | 53 |
| heikin_ashi | 104 | 268 | **38.8%** | 108 | 160 |
| ema | 111 | 290 | **38.3%** | 106 | 184 |
| candlestick | 39 | 102 | **38.2%** | 50 | 52 |
| momentum_factors | 19 | 53 | **35.8%** | 11 | 42 |
| sentiment | 154 | 442 | **34.8%** | 434 | 8 |
| trend | 99 | 301 | **32.9%** | 99 | 202 |
| macro_regime | 10 | 31 | **32.3%** | 31 | 0 |
| structure | 24 | 80 | **30.0%** | 39 | 41 |
| oscillators | 3 | 12 | **25.0%** | 3 | 9 |
| smart_money | 4 | 29 | **13.8%** | 21 | 8 |
| fear_greed | 0 | 52 | **0.0%** | 52 | 0 |

### Per-Ticker Consensus Accuracy

| Ticker | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| XAU-USD | 2 | 2 | **100.0%** |
| META | 1 | 1 | **100.0%** |
| XAG-USD | 2 | 2 | **100.0%** |
| LMT | 14 | 15 | **93.3%** |
| AMD | 14 | 16 | **87.5%** |
| NVDA | 12 | 15 | **80.0%** |
| GOOGL | 14 | 19 | **73.7%** |
| AMZN | 13 | 18 | **72.2%** |
| QQQ | 5 | 7 | **71.4%** |
| GRRR | 10 | 15 | **66.7%** |
| TSM | 12 | 18 | **66.7%** |
| IONQ | 10 | 15 | **66.7%** |
| SOUN | 9 | 14 | **64.3%** |
| VRT | 9 | 14 | **64.3%** |
| UPST | 7 | 11 | **63.6%** |
| TTWO | 5 | 9 | **55.6%** |
| BABA | 10 | 19 | **52.6%** |
| PLTR | 1 | 2 | **50.0%** |
| ETH-USD | 5 | 15 | **33.3%** |
| MSTR | 3 | 10 | **30.0%** |
| VERI | 6 | 20 | **30.0%** |
| SMCI | 4 | 14 | **28.6%** |
| AVGO | 3 | 11 | **27.3%** |
| MU | 4 | 16 | **25.0%** |
| TEM | 6 | 25 | **24.0%** |
| BTC-USD | 2 | 14 | **14.3%** |
| AAPL | 0 | 5 | **0.0%** |

### Best Calls (highest confidence, correct)

- **UPST** SELL @ $29.99 -> -8.09% (10B/3S)
- **NVDA** SELL @ $188.25 -> -4.18% (4B/8S)
- **IONQ** SELL @ $41.26 -> -9.54% (9B/3S)
- **NVDA** SELL @ $185.94 -> -2.98% (4B/8S)
- **NVDA** SELL @ $186.00 -> -3.01% (4B/8S)

### Worst Calls (wrong, highest vote count)

- **AVGO** SELL @ $315.25 -> +0.95% (4B/8S)
- **AVGO** SELL @ $310.24 -> +2.58% (4B/8S)
- **NVDA** BUY @ $188.50 -> -4.30% (4B/7S)
- **IONQ** BUY @ $40.78 -> -8.47% (8B/3S)
- **NVDA** BUY @ $188.88 -> -4.49% (4B/7S)

### Failure Analysis

**159 incorrect predictions out of 342 total (46.5% error rate)**

### Failures by Ticker

- **TEM**: 19 wrong calls
  - 2026-02-26T12:51 predicted BUY, actual DOWN -0.80% (signals: 2B/2S)
  - 2026-02-26T13:03 predicted BUY, actual DOWN -0.81% (signals: 2B/2S)
  - 2026-02-26T13:23 predicted BUY, actual DOWN -0.81% (signals: 2B/2S)
  - 2026-02-26T13:42 predicted BUY, actual DOWN -0.81% (signals: 2B/2S)
  - 2026-02-26T14:02 predicted BUY, actual DOWN -0.76% (signals: 2B/2S)
  - 2026-02-26T14:07 predicted BUY, actual DOWN -0.76% (signals: 2B/2S)
  - 2026-02-26T14:21 predicted BUY, actual DOWN -0.76% (signals: 2B/2S)
  - 2026-02-26T14:26 predicted BUY, actual DOWN -0.76% (signals: 2B/2S)
  - 2026-02-26T15:39 predicted SELL, actual UP +2.50% (signals: 4B/5S)
  - 2026-02-26T15:57 predicted SELL, actual UP +1.52% (signals: 2B/4S)
  - 2026-02-26T16:04 predicted SELL, actual UP +1.62% (signals: 1B/4S)
  - 2026-02-26T16:18 predicted SELL, actual UP +0.42% (signals: 2B/2S)
  - 2026-02-26T16:38 predicted SELL, actual UP +1.19% (signals: 1B/3S)
  - 2026-02-26T16:53 predicted SELL, actual UP +2.06% (signals: 1B/2S)
  - 2026-02-26T16:59 predicted SELL, actual UP +2.09% (signals: 1B/2S)
  - 2026-02-26T17:13 predicted SELL, actual UP +2.05% (signals: 1B/2S)
  - 2026-02-26T17:29 predicted SELL, actual UP +2.25% (signals: 1B/3S)
  - 2026-02-26T17:47 predicted SELL, actual UP +2.23% (signals: 1B/2S)
  - 2026-02-26T18:08 predicted SELL, actual UP +2.25% (signals: 1B/3S)
- **VERI**: 14 wrong calls
  - 2026-02-26T12:51 predicted BUY, actual DOWN -4.55% (signals: 3B/1S)
  - 2026-02-26T13:03 predicted BUY, actual DOWN -4.55% (signals: 3B/1S)
  - 2026-02-26T13:23 predicted BUY, actual DOWN -4.55% (signals: 3B/1S)
  - 2026-02-26T13:42 predicted BUY, actual DOWN -4.55% (signals: 3B/1S)
  - 2026-02-26T14:02 predicted BUY, actual DOWN -4.72% (signals: 3B/1S)
  - 2026-02-26T14:07 predicted BUY, actual DOWN -4.72% (signals: 3B/1S)
  - 2026-02-26T14:21 predicted BUY, actual DOWN -4.72% (signals: 3B/1S)
  - 2026-02-26T14:26 predicted BUY, actual DOWN -4.72% (signals: 3B/1S)
  - 2026-02-26T14:41 predicted BUY, actual DOWN -4.56% (signals: 3B/3S)
  - 2026-02-26T14:45 predicted BUY, actual DOWN -4.40% (signals: 3B/4S)
  - 2026-02-26T15:19 predicted BUY, actual DOWN -4.71% (signals: 4B/2S)
  - 2026-02-26T15:25 predicted BUY, actual DOWN -4.23% (signals: 4B/2S)
  - 2026-02-26T16:38 predicted BUY, actual DOWN -5.35% (signals: 4B/2S)
  - 2026-02-26T17:47 predicted BUY, actual DOWN -2.59% (signals: 4B/3S)
- **BTC-USD**: 12 wrong calls
  - 2026-02-26T14:07 predicted BUY, actual DOWN -2.48% (signals: 4B/3S)
  - 2026-02-26T14:21 predicted BUY, actual DOWN -2.41% (signals: 4B/3S)
  - 2026-02-26T14:26 predicted BUY, actual DOWN -2.70% (signals: 2B/3S)
  - 2026-02-26T14:45 predicted BUY, actual DOWN -2.70% (signals: 2B/3S)
  - 2026-02-26T15:00 predicted BUY, actual DOWN -2.99% (signals: 4B/4S)
  - 2026-02-26T15:06 predicted BUY, actual DOWN -3.15% (signals: 3B/4S)
  - 2026-02-26T15:19 predicted BUY, actual DOWN -3.08% (signals: 4B/3S)
  - 2026-02-26T15:25 predicted BUY, actual DOWN -2.29% (signals: 4B/4S)
  - 2026-02-26T15:39 predicted BUY, actual DOWN -2.60% (signals: 5B/4S)
  - 2026-02-26T16:18 predicted BUY, actual DOWN -2.44% (signals: 2B/4S)
  - 2026-02-26T16:32 predicted BUY, actual DOWN -2.39% (signals: 2B/3S)
  - 2026-02-26T16:59 predicted BUY, actual DOWN -1.57% (signals: 3B/5S)
- **MU**: 12 wrong calls
  - 2026-02-26T14:45 predicted BUY, actual DOWN -2.80% (signals: 3B/2S)
  - 2026-02-26T15:00 predicted BUY, actual DOWN -1.08% (signals: 4B/3S)
  - 2026-02-26T15:06 predicted BUY, actual DOWN -1.24% (signals: 3B/2S)
  - 2026-02-26T15:25 predicted SELL, actual UP +2.07% (signals: 4B/6S)
  - 2026-02-26T15:39 predicted SELL, actual UP +2.31% (signals: 4B/5S)
  - 2026-02-26T15:57 predicted SELL, actual UP +0.77% (signals: 2B/4S)
  - 2026-02-26T16:04 predicted SELL, actual UP +1.03% (signals: 2B/3S)
  - 2026-02-26T16:18 predicted SELL, actual UP +0.92% (signals: 2B/3S)
  - 2026-02-26T16:53 predicted BUY, actual DOWN -0.49% (signals: 1B/4S)
  - 2026-02-26T16:59 predicted BUY, actual DOWN -0.37% (signals: 2B/4S)
  - 2026-02-26T17:29 predicted BUY, actual DOWN -0.26% (signals: 2B/4S)
  - 2026-02-26T17:47 predicted BUY, actual DOWN -0.00% (signals: 2B/4S)
- **SMCI**: 10 wrong calls
  - 2026-02-26T12:51 predicted BUY, actual DOWN -4.11% (signals: 3B/2S)
  - 2026-02-26T13:03 predicted BUY, actual DOWN -4.05% (signals: 3B/2S)
  - 2026-02-26T13:23 predicted BUY, actual DOWN -4.05% (signals: 3B/2S)
  - 2026-02-26T13:42 predicted BUY, actual DOWN -4.05% (signals: 3B/2S)
  - 2026-02-26T14:02 predicted BUY, actual DOWN -3.96% (signals: 3B/2S)
  - 2026-02-26T14:07 predicted BUY, actual DOWN -3.96% (signals: 3B/2S)
  - 2026-02-26T14:21 predicted BUY, actual DOWN -3.96% (signals: 3B/2S)
  - 2026-02-26T14:26 predicted BUY, actual DOWN -3.96% (signals: 3B/2S)
  - 2026-02-26T16:04 predicted SELL, actual UP +2.59% (signals: 1B/2S)
  - 2026-02-26T16:18 predicted SELL, actual UP +2.21% (signals: 3B/1S)
- **ETH-USD**: 10 wrong calls
  - 2026-02-26T15:00 predicted BUY, actual DOWN -5.73% (signals: 4B/3S)
  - 2026-02-26T15:06 predicted BUY, actual DOWN -5.91% (signals: 4B/2S)
  - 2026-02-26T15:25 predicted BUY, actual DOWN -4.75% (signals: 4B/6S)
  - 2026-02-26T15:39 predicted BUY, actual DOWN -5.00% (signals: 4B/6S)
  - 2026-02-26T15:57 predicted BUY, actual DOWN -5.17% (signals: 2B/4S)
  - 2026-02-26T16:18 predicted BUY, actual DOWN -4.78% (signals: 3B/4S)
  - 2026-02-26T16:53 predicted BUY, actual DOWN -4.07% (signals: 4B/4S)
  - 2026-02-26T16:59 predicted BUY, actual DOWN -2.69% (signals: 4B/7S)
  - 2026-02-26T17:13 predicted BUY, actual DOWN -3.56% (signals: 3B/5S)
  - 2026-02-26T17:47 predicted BUY, actual DOWN -3.84% (signals: 2B/5S)
- **BABA**: 9 wrong calls
  - 2026-02-26T13:42 predicted BUY, actual DOWN -2.43% (signals: 3B/6S)
  - 2026-02-26T14:02 predicted BUY, actual DOWN -2.40% (signals: 3B/6S)
  - 2026-02-26T14:07 predicted BUY, actual DOWN -2.40% (signals: 3B/6S)
  - 2026-02-26T14:21 predicted BUY, actual DOWN -2.40% (signals: 3B/6S)
  - 2026-02-26T14:26 predicted BUY, actual DOWN -2.40% (signals: 3B/6S)
  - 2026-02-26T15:00 predicted BUY, actual DOWN -2.46% (signals: 2B/5S)
  - 2026-02-26T15:06 predicted BUY, actual DOWN -2.06% (signals: 2B/7S)
  - 2026-02-26T15:19 predicted BUY, actual DOWN -1.91% (signals: 2B/8S)
  - 2026-02-26T16:18 predicted BUY, actual DOWN -1.62% (signals: 1B/4S)
- **AVGO**: 8 wrong calls
  - 2026-02-26T15:00 predicted SELL, actual UP +0.25% (signals: 4B/6S)
  - 2026-02-26T15:19 predicted SELL, actual UP +0.95% (signals: 4B/8S)
  - 2026-02-26T15:39 predicted SELL, actual UP +2.58% (signals: 4B/8S)
  - 2026-02-26T16:04 predicted SELL, actual UP +2.50% (signals: 2B/5S)
  - 2026-02-26T16:18 predicted SELL, actual UP +2.36% (signals: 2B/5S)
  - 2026-02-26T16:32 predicted SELL, actual UP +1.32% (signals: 2B/3S)
  - 2026-02-26T16:38 predicted SELL, actual UP +1.04% (signals: 2B/4S)
  - 2026-02-26T16:53 predicted SELL, actual UP +1.45% (signals: 2B/3S)
- **MSTR**: 7 wrong calls
  - 2026-02-26T12:51 predicted BUY, actual DOWN -5.01% (signals: 4B/0S)
  - 2026-02-26T13:03 predicted BUY, actual DOWN -5.08% (signals: 4B/0S)
  - 2026-02-26T13:23 predicted BUY, actual DOWN -5.08% (signals: 4B/0S)
  - 2026-02-26T15:00 predicted BUY, actual DOWN -3.18% (signals: 1B/4S)
  - 2026-02-26T15:25 predicted BUY, actual DOWN -1.97% (signals: 3B/2S)
  - 2026-02-26T17:13 predicted BUY, actual DOWN -0.58% (signals: 2B/3S)
  - 2026-02-26T18:08 predicted SELL, actual UP +0.33% (signals: 2B/3S)
- **TSM**: 6 wrong calls
  - 2026-02-26T15:00 predicted BUY, actual DOWN -0.03% (signals: 4B/4S)
  - 2026-02-26T15:19 predicted SELL, actual UP +0.73% (signals: 4B/5S)
  - 2026-02-26T15:25 predicted SELL, actual UP +1.31% (signals: 4B/6S)
  - 2026-02-26T15:57 predicted SELL, actual UP +0.37% (signals: 2B/3S)
  - 2026-02-26T16:38 predicted SELL, actual UP +0.08% (signals: 2B/3S)
  - 2026-02-26T17:29 predicted SELL, actual UP +0.43% (signals: 2B/3S)
- **GOOGL**: 5 wrong calls
  - 2026-02-26T14:02 predicted BUY, actual DOWN -1.93% (signals: 6B/3S)
  - 2026-02-26T14:07 predicted BUY, actual DOWN -1.93% (signals: 6B/3S)
  - 2026-02-26T14:21 predicted BUY, actual DOWN -1.93% (signals: 6B/3S)
  - 2026-02-26T14:26 predicted BUY, actual DOWN -1.93% (signals: 6B/3S)
  - 2026-02-26T16:04 predicted SELL, actual UP +1.12% (signals: 1B/2S)
- **SOUN**: 5 wrong calls
  - 2026-02-26T14:07 predicted SELL, actual UP +1.56% (signals: 3B/2S)
  - 2026-02-26T15:19 predicted BUY, actual DOWN -1.73% (signals: 5B/2S)
  - 2026-02-26T15:25 predicted BUY, actual DOWN -0.17% (signals: 4B/0S)
  - 2026-02-26T15:39 predicted BUY, actual DOWN -0.11% (signals: 3B/0S)
  - 2026-02-26T16:38 predicted BUY, actual DOWN -0.57% (signals: 3B/0S)
- **AMZN**: 5 wrong calls
  - 2026-02-26T14:41 predicted BUY, actual DOWN -0.72% (signals: 4B/1S)
  - 2026-02-26T14:45 predicted BUY, actual DOWN -0.74% (signals: 4B/1S)
  - 2026-02-26T15:00 predicted BUY, actual DOWN -0.16% (signals: 3B/4S)
  - 2026-02-26T15:19 predicted BUY, actual DOWN -0.27% (signals: 3B/2S)
  - 2026-02-26T15:57 predicted SELL, actual UP +0.26% (signals: 3B/3S)
- **AAPL**: 5 wrong calls
  - 2026-02-26T14:41 predicted BUY, actual DOWN -2.28% (signals: 4B/3S)
  - 2026-02-26T15:25 predicted BUY, actual DOWN -0.96% (signals: 4B/1S)
  - 2026-02-26T15:39 predicted BUY, actual DOWN -1.12% (signals: 4B/1S)
  - 2026-02-26T15:57 predicted BUY, actual DOWN -0.91% (signals: 4B/1S)
  - 2026-02-26T16:04 predicted BUY, actual DOWN -0.68% (signals: 4B/2S)
- **VRT**: 5 wrong calls
  - 2026-02-26T14:41 predicted BUY, actual DOWN -2.95% (signals: 3B/3S)
  - 2026-02-26T14:45 predicted BUY, actual DOWN -1.26% (signals: 4B/4S)
  - 2026-02-26T15:57 predicted SELL, actual UP +0.81% (signals: 4B/3S)
  - 2026-02-26T17:29 predicted SELL, actual UP +0.98% (signals: 2B/2S)
  - 2026-02-26T17:47 predicted SELL, actual UP +0.73% (signals: 3B/1S)
- **IONQ**: 5 wrong calls
  - 2026-02-26T15:06 predicted BUY, actual DOWN -8.47% (signals: 8B/3S)
  - 2026-02-26T15:25 predicted BUY, actual DOWN -6.34% (signals: 5B/2S)
  - 2026-02-26T16:38 predicted BUY, actual DOWN -6.31% (signals: 5B/1S)
  - 2026-02-26T17:29 predicted BUY, actual DOWN -5.07% (signals: 4B/2S)
  - 2026-02-26T18:08 predicted BUY, actual DOWN -4.72% (signals: 4B/1S)
- **GRRR**: 5 wrong calls
  - 2026-02-26T15:19 predicted SELL, actual UP +2.36% (signals: 3B/2S)
  - 2026-02-26T15:25 predicted SELL, actual UP +2.90% (signals: 2B/2S)
  - 2026-02-26T15:39 predicted SELL, actual UP +2.18% (signals: 2B/2S)
  - 2026-02-26T16:04 predicted SELL, actual UP +0.30% (signals: 3B/2S)
  - 2026-02-26T16:18 predicted SELL, actual UP +0.13% (signals: 4B/2S)
- **TTWO**: 4 wrong calls
  - 2026-02-26T14:41 predicted SELL, actual UP +1.93% (signals: 4B/2S)
  - 2026-02-26T14:45 predicted SELL, actual UP +0.82% (signals: 6B/4S)
  - 2026-02-26T15:57 predicted SELL, actual UP +0.47% (signals: 5B/1S)
  - 2026-02-26T16:18 predicted BUY, actual DOWN -0.07% (signals: 4B/1S)
- **UPST**: 4 wrong calls
  - 2026-02-26T16:38 predicted BUY, actual DOWN -6.65% (signals: 5B/1S)
  - 2026-02-26T16:53 predicted BUY, actual DOWN -5.44% (signals: 3B/0S)
  - 2026-02-26T16:59 predicted BUY, actual DOWN -4.98% (signals: 2B/1S)
  - 2026-02-26T17:13 predicted BUY, actual DOWN -6.44% (signals: 2B/1S)
- **NVDA**: 3 wrong calls
  - 2026-02-26T15:00 predicted BUY, actual DOWN -4.30% (signals: 4B/7S)
  - 2026-02-26T15:19 predicted BUY, actual DOWN -4.49% (signals: 4B/7S)
  - 2026-02-26T15:57 predicted BUY, actual DOWN -3.01% (signals: 2B/7S)
- **QQQ**: 2 wrong calls
  - 2026-02-26T15:00 predicted BUY, actual DOWN -0.73% (signals: 4B/3S)
  - 2026-02-26T15:06 predicted BUY, actual DOWN -0.58% (signals: 4B/3S)
- **AMD**: 2 wrong calls
  - 2026-02-26T15:25 predicted BUY, actual DOWN -1.22% (signals: 4B/6S)
  - 2026-02-26T15:39 predicted BUY, actual DOWN -1.05% (signals: 4B/6S)
- **PLTR**: 1 wrong calls
  - 2026-02-26T15:19 predicted BUY, actual DOWN -0.29% (signals: 4B/2S)
- **LMT**: 1 wrong calls
  - 2026-02-26T16:04 predicted SELL, actual UP +1.67% (signals: 2B/3S)

### Signals Most Often Wrong

- `ema`: contributed to 78 wrong consensus calls
- `sentiment`: contributed to 73 wrong consensus calls
- `volume_flow`: contributed to 70 wrong consensus calls
- `rsi`: contributed to 58 wrong consensus calls
- `bb`: contributed to 44 wrong consensus calls
- `volume`: contributed to 34 wrong consensus calls
- `trend`: contributed to 34 wrong consensus calls
- `mean_reversion`: contributed to 27 wrong consensus calls
- `fear_greed`: contributed to 22 wrong consensus calls
- `heikin_ashi`: contributed to 20 wrong consensus calls

---

## 3d Horizon

### Overall Consensus Accuracy: **57.7%** (127/220) — 70% threshold: **NO**

### Per-Signal Accuracy

| Signal | Correct | Total | Accuracy | BUY votes | SELL votes |
|--------|---------|-------|----------|-----------|------------|
| ministral | 33 | 35 | **94.3%** | 2 | 33 |
| mean_reversion | 49 | 82 | **59.8%** | 64 | 18 |
| macd | 26 | 45 | **57.8%** | 20 | 25 |
| bb | 63 | 115 | **54.8%** | 80 | 35 |
| rsi | 136 | 275 | **49.5%** | 241 | 34 |
| volume | 78 | 162 | **48.1%** | 59 | 103 |
| volume_flow | 169 | 370 | **45.7%** | 161 | 209 |
| volatility_sig | 39 | 87 | **44.8%** | 25 | 62 |
| candlestick | 27 | 61 | **44.3%** | 21 | 40 |
| macro_regime | 7 | 16 | **43.8%** | 16 | 0 |
| fibonacci | 14 | 33 | **42.4%** | 3 | 30 |
| ema | 72 | 172 | **41.9%** | 42 | 130 |
| heikin_ashi | 73 | 190 | **38.4%** | 74 | 116 |
| sentiment | 109 | 294 | **37.1%** | 286 | 8 |
| news_event | 19 | 53 | **35.8%** | 0 | 53 |
| momentum_factors | 12 | 35 | **34.3%** | 1 | 34 |
| trend | 68 | 204 | **33.3%** | 49 | 155 |
| structure | 15 | 52 | **28.8%** | 25 | 27 |
| smart_money | 3 | 11 | **27.3%** | 5 | 6 |
| oscillators | 3 | 12 | **25.0%** | 3 | 9 |
| fear_greed | 0 | 52 | **0.0%** | 52 | 0 |

### Per-Ticker Consensus Accuracy

| Ticker | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| XAU-USD | 2 | 2 | **100.0%** |
| META | 1 | 1 | **100.0%** |
| XAG-USD | 2 | 2 | **100.0%** |
| LMT | 14 | 15 | **93.3%** |
| AMD | 14 | 16 | **87.5%** |
| AMZN | 15 | 18 | **83.3%** |
| NVDA | 12 | 15 | **80.0%** |
| GOOGL | 14 | 19 | **73.7%** |
| TSM | 13 | 18 | **72.2%** |
| SOUN | 10 | 14 | **71.4%** |
| VRT | 8 | 14 | **57.1%** |
| PLTR | 1 | 2 | **50.0%** |
| ETH-USD | 5 | 15 | **33.3%** |
| TTWO | 3 | 9 | **33.3%** |
| SMCI | 4 | 14 | **28.6%** |
| AVGO | 3 | 11 | **27.3%** |
| MU | 4 | 16 | **25.0%** |
| BTC-USD | 2 | 14 | **14.3%** |
| AAPL | 0 | 5 | **0.0%** |

### Best Calls (highest confidence, correct)

- **NVDA** SELL @ $188.25 -> -5.88% (4B/8S)
- **NVDA** SELL @ $185.94 -> -4.71% (4B/8S)
- **NVDA** SELL @ $186.00 -> -4.73% (4B/8S)
- **LMT** BUY @ $639.13 -> +2.96% (4B/7S)
- **VRT** BUY @ $247.37 -> +3.04% (4B/7S)

### Worst Calls (wrong, highest vote count)

- **AVGO** SELL @ $315.25 -> +1.36% (4B/8S)
- **AVGO** SELL @ $310.24 -> +3.00% (4B/8S)
- **NVDA** BUY @ $188.50 -> -6.00% (4B/7S)
- **NVDA** BUY @ $188.88 -> -6.19% (4B/7S)
- **ETH-USD** BUY @ $1,983.61 -> -0.53% (4B/7S)

### Failure Analysis

**93 incorrect predictions out of 220 total (42.3% error rate)**

### Failures by Ticker

- **BTC-USD**: 12 wrong calls
  - 2026-02-26T14:07 predicted BUY, actual DOWN -1.37% (signals: 4B/3S)
  - 2026-02-26T14:21 predicted BUY, actual DOWN -1.31% (signals: 4B/3S)
  - 2026-02-26T14:26 predicted BUY, actual DOWN -1.60% (signals: 2B/3S)
  - 2026-02-26T14:45 predicted BUY, actual DOWN -1.60% (signals: 2B/3S)
  - 2026-02-26T15:00 predicted BUY, actual DOWN -2.01% (signals: 4B/4S)
  - 2026-02-26T15:06 predicted BUY, actual DOWN -2.17% (signals: 3B/4S)
  - 2026-02-26T15:19 predicted BUY, actual DOWN -2.10% (signals: 4B/3S)
  - 2026-02-26T15:25 predicted BUY, actual DOWN -1.31% (signals: 4B/4S)
  - 2026-02-26T15:39 predicted BUY, actual DOWN -1.62% (signals: 5B/4S)
  - 2026-02-26T16:18 predicted BUY, actual DOWN -1.60% (signals: 2B/4S)
  - 2026-02-26T16:32 predicted BUY, actual DOWN -1.55% (signals: 2B/3S)
  - 2026-02-26T16:59 predicted BUY, actual DOWN -0.73% (signals: 3B/5S)
- **MU**: 12 wrong calls
  - 2026-02-26T14:45 predicted BUY, actual DOWN -2.79% (signals: 3B/2S)
  - 2026-02-26T15:00 predicted BUY, actual DOWN -1.21% (signals: 4B/3S)
  - 2026-02-26T15:06 predicted BUY, actual DOWN -1.37% (signals: 3B/2S)
  - 2026-02-26T15:25 predicted SELL, actual UP +1.94% (signals: 4B/6S)
  - 2026-02-26T15:39 predicted SELL, actual UP +2.18% (signals: 4B/5S)
  - 2026-02-26T15:57 predicted SELL, actual UP +0.64% (signals: 2B/4S)
  - 2026-02-26T16:04 predicted SELL, actual UP +1.01% (signals: 2B/3S)
  - 2026-02-26T16:18 predicted SELL, actual UP +0.91% (signals: 2B/3S)
  - 2026-02-26T16:53 predicted BUY, actual DOWN -0.51% (signals: 1B/4S)
  - 2026-02-26T16:59 predicted BUY, actual DOWN -0.39% (signals: 2B/4S)
  - 2026-02-26T17:29 predicted BUY, actual DOWN -0.26% (signals: 2B/4S)
  - 2026-02-26T17:47 predicted BUY, actual DOWN -0.00% (signals: 2B/4S)
- **SMCI**: 10 wrong calls
  - 2026-02-26T12:51 predicted BUY, actual DOWN -4.20% (signals: 3B/2S)
  - 2026-02-26T13:03 predicted BUY, actual DOWN -4.20% (signals: 3B/2S)
  - 2026-02-26T13:23 predicted BUY, actual DOWN -4.20% (signals: 3B/2S)
  - 2026-02-26T13:42 predicted BUY, actual DOWN -4.20% (signals: 3B/2S)
  - 2026-02-26T14:02 predicted BUY, actual DOWN -4.20% (signals: 3B/2S)
  - 2026-02-26T14:07 predicted BUY, actual DOWN -4.20% (signals: 3B/2S)
  - 2026-02-26T14:21 predicted BUY, actual DOWN -4.20% (signals: 3B/2S)
  - 2026-02-26T14:26 predicted BUY, actual DOWN -4.20% (signals: 3B/2S)
  - 2026-02-26T16:04 predicted SELL, actual UP +2.24% (signals: 1B/2S)
  - 2026-02-26T16:18 predicted SELL, actual UP +1.86% (signals: 3B/1S)
- **ETH-USD**: 10 wrong calls
  - 2026-02-26T15:00 predicted BUY, actual DOWN -3.44% (signals: 4B/3S)
  - 2026-02-26T15:06 predicted BUY, actual DOWN -3.62% (signals: 4B/2S)
  - 2026-02-26T15:25 predicted BUY, actual DOWN -2.43% (signals: 4B/6S)
  - 2026-02-26T15:39 predicted BUY, actual DOWN -2.69% (signals: 4B/6S)
  - 2026-02-26T15:57 predicted BUY, actual DOWN -2.86% (signals: 2B/4S)
  - 2026-02-26T16:18 predicted BUY, actual DOWN -2.66% (signals: 3B/4S)
  - 2026-02-26T16:53 predicted BUY, actual DOWN -1.94% (signals: 4B/4S)
  - 2026-02-26T16:59 predicted BUY, actual DOWN -0.53% (signals: 4B/7S)
  - 2026-02-26T17:13 predicted BUY, actual DOWN -0.65% (signals: 3B/5S)
  - 2026-02-26T17:47 predicted BUY, actual DOWN -0.94% (signals: 2B/5S)
- **AVGO**: 8 wrong calls
  - 2026-02-26T15:00 predicted SELL, actual UP +0.66% (signals: 4B/6S)
  - 2026-02-26T15:19 predicted SELL, actual UP +1.36% (signals: 4B/8S)
  - 2026-02-26T15:39 predicted SELL, actual UP +3.00% (signals: 4B/8S)
  - 2026-02-26T16:04 predicted SELL, actual UP +2.95% (signals: 2B/5S)
  - 2026-02-26T16:18 predicted SELL, actual UP +2.82% (signals: 2B/5S)
  - 2026-02-26T16:32 predicted SELL, actual UP +1.76% (signals: 2B/3S)
  - 2026-02-26T16:38 predicted SELL, actual UP +1.48% (signals: 2B/4S)
  - 2026-02-26T16:53 predicted SELL, actual UP +1.89% (signals: 2B/3S)
- **TTWO**: 6 wrong calls
  - 2026-02-26T14:41 predicted SELL, actual UP +1.49% (signals: 4B/2S)
  - 2026-02-26T14:45 predicted SELL, actual UP +0.38% (signals: 6B/4S)
  - 2026-02-26T15:19 predicted BUY, actual DOWN -0.14% (signals: 3B/1S)
  - 2026-02-26T15:57 predicted SELL, actual UP +0.04% (signals: 5B/1S)
  - 2026-02-26T16:04 predicted BUY, actual DOWN -0.32% (signals: 4B/0S)
  - 2026-02-26T16:18 predicted BUY, actual DOWN -0.42% (signals: 4B/1S)
- **VRT**: 6 wrong calls
  - 2026-02-26T14:41 predicted BUY, actual DOWN -1.18% (signals: 3B/3S)
  - 2026-02-26T15:57 predicted SELL, actual UP +2.54% (signals: 4B/3S)
  - 2026-02-26T16:18 predicted SELL, actual UP +1.22% (signals: 3B/2S)
  - 2026-02-26T16:32 predicted SELL, actual UP +0.10% (signals: 3B/1S)
  - 2026-02-26T17:29 predicted SELL, actual UP +0.98% (signals: 2B/2S)
  - 2026-02-26T17:47 predicted SELL, actual UP +0.73% (signals: 3B/1S)
- **GOOGL**: 5 wrong calls
  - 2026-02-26T14:02 predicted BUY, actual DOWN -0.71% (signals: 6B/3S)
  - 2026-02-26T14:07 predicted BUY, actual DOWN -0.71% (signals: 6B/3S)
  - 2026-02-26T14:21 predicted BUY, actual DOWN -0.71% (signals: 6B/3S)
  - 2026-02-26T14:26 predicted BUY, actual DOWN -0.71% (signals: 6B/3S)
  - 2026-02-26T16:04 predicted SELL, actual UP +2.25% (signals: 1B/2S)
- **AAPL**: 5 wrong calls
  - 2026-02-26T14:41 predicted BUY, actual DOWN -3.97% (signals: 4B/3S)
  - 2026-02-26T15:25 predicted BUY, actual DOWN -2.71% (signals: 4B/1S)
  - 2026-02-26T15:39 predicted BUY, actual DOWN -2.86% (signals: 4B/1S)
  - 2026-02-26T15:57 predicted BUY, actual DOWN -2.66% (signals: 4B/1S)
  - 2026-02-26T16:04 predicted BUY, actual DOWN -2.49% (signals: 4B/2S)
- **TSM**: 5 wrong calls
  - 2026-02-26T15:00 predicted BUY, actual DOWN -0.32% (signals: 4B/4S)
  - 2026-02-26T15:19 predicted SELL, actual UP +0.44% (signals: 4B/5S)
  - 2026-02-26T15:25 predicted SELL, actual UP +1.01% (signals: 4B/6S)
  - 2026-02-26T15:57 predicted SELL, actual UP +0.07% (signals: 2B/3S)
  - 2026-02-26T17:29 predicted SELL, actual UP +0.43% (signals: 2B/3S)
- **SOUN**: 4 wrong calls
  - 2026-02-26T15:19 predicted BUY, actual DOWN -3.96% (signals: 5B/2S)
  - 2026-02-26T15:25 predicted BUY, actual DOWN -2.44% (signals: 4B/0S)
  - 2026-02-26T15:39 predicted BUY, actual DOWN -2.38% (signals: 3B/0S)
  - 2026-02-26T16:38 predicted BUY, actual DOWN -2.93% (signals: 3B/0S)
- **AMZN**: 3 wrong calls
  - 2026-02-26T14:41 predicted BUY, actual DOWN -0.23% (signals: 4B/1S)
  - 2026-02-26T14:45 predicted BUY, actual DOWN -0.25% (signals: 4B/1S)
  - 2026-02-26T15:57 predicted SELL, actual UP +0.71% (signals: 3B/3S)
- **NVDA**: 3 wrong calls
  - 2026-02-26T15:00 predicted BUY, actual DOWN -6.00% (signals: 4B/7S)
  - 2026-02-26T15:19 predicted BUY, actual DOWN -6.19% (signals: 4B/7S)
  - 2026-02-26T15:57 predicted BUY, actual DOWN -4.73% (signals: 2B/7S)
- **AMD**: 2 wrong calls
  - 2026-02-26T15:25 predicted BUY, actual DOWN -1.27% (signals: 4B/6S)
  - 2026-02-26T15:39 predicted BUY, actual DOWN -1.11% (signals: 4B/6S)
- **PLTR**: 1 wrong calls
  - 2026-02-26T15:00 predicted SELL, actual UP +0.63% (signals: 5B/2S)
- **LMT**: 1 wrong calls
  - 2026-02-26T16:04 predicted SELL, actual UP +2.11% (signals: 2B/3S)

### Signals Most Often Wrong

- `rsi`: contributed to 43 wrong consensus calls
- `ema`: contributed to 40 wrong consensus calls
- `volume_flow`: contributed to 39 wrong consensus calls
- `sentiment`: contributed to 37 wrong consensus calls
- `bb`: contributed to 30 wrong consensus calls
- `volume`: contributed to 22 wrong consensus calls
- `fear_greed`: contributed to 22 wrong consensus calls
- `mean_reversion`: contributed to 22 wrong consensus calls
- `trend`: contributed to 20 wrong consensus calls
- `heikin_ashi`: contributed to 12 wrong consensus calls

---

## Chronological Prediction Log (1d horizon)

| Time (UTC) | Ticker | Call | Votes | Price | 1d Outcome | Correct? |
|------------|--------|------|-------|-------|------------|----------|
| 12:51 | MSTR | BUY | 4B/0S | $136.12 | -5.01% | **NO** |
| 12:51 | GOOGL | SELL | 3B/2S | $312.96 | -1.56% | YES |
| 12:51 | GRRR | BUY | 3B/3S | $11.50 | +2.26% | YES |
| 12:51 | SOUN | BUY | 2B/3S | $8.52 | +2.58% | YES |
| 12:51 | SMCI | BUY | 3B/2S | $33.81 | -4.11% | **NO** |
| 12:51 | TSM | SELL | 6B/3S | $391.98 | -4.24% | YES |
| 12:51 | TEM | BUY | 2B/2S | $53.73 | -0.80% | **NO** |
| 12:51 | VERI | BUY | 3B/1S | $2.96 | -4.55% | **NO** |
| 12:51 | LMT | BUY | 2B/4S | $647.64 | +1.22% | YES |
| 13:03 | MSTR | BUY | 4B/0S | $136.12 | -5.08% | **NO** |
| 13:03 | GOOGL | SELL | 3B/2S | $312.96 | -1.59% | YES |
| 13:03 | GRRR | BUY | 3B/3S | $11.50 | +2.13% | YES |
| 13:03 | SOUN | BUY | 2B/3S | $8.52 | +2.58% | YES |
| 13:03 | SMCI | BUY | 3B/2S | $33.81 | -4.05% | **NO** |
| 13:03 | TSM | SELL | 6B/3S | $391.98 | -4.28% | YES |
| 13:03 | TEM | BUY | 2B/2S | $53.73 | -0.81% | **NO** |
| 13:03 | VERI | BUY | 3B/1S | $2.96 | -4.55% | **NO** |
| 13:03 | LMT | BUY | 2B/4S | $647.64 | +1.24% | YES |
| 13:23 | MSTR | BUY | 4B/0S | $136.12 | -5.08% | **NO** |
| 13:23 | GOOGL | SELL | 3B/2S | $312.96 | -1.59% | YES |
| 13:23 | AMZN | SELL | 5B/1S | $210.67 | -0.86% | YES |
| 13:23 | GRRR | BUY | 3B/3S | $11.50 | +2.13% | YES |
| 13:23 | SOUN | BUY | 2B/3S | $8.52 | +2.58% | YES |
| 13:23 | SMCI | BUY | 3B/2S | $33.81 | -4.05% | **NO** |
| 13:23 | TSM | SELL | 6B/3S | $391.98 | -4.28% | YES |
| 13:23 | TEM | BUY | 2B/2S | $53.73 | -0.81% | **NO** |
| 13:23 | VERI | BUY | 3B/1S | $2.96 | -4.55% | **NO** |
| 13:23 | LMT | BUY | 2B/4S | $647.64 | +1.24% | YES |
| 13:42 | BABA | BUY | 3B/6S | $148.07 | -2.43% | **NO** |
| 13:42 | GOOGL | SELL | 3B/2S | $312.96 | -1.59% | YES |
| 13:42 | AMZN | SELL | 5B/1S | $210.67 | -0.86% | YES |
| 13:42 | GRRR | BUY | 3B/3S | $11.50 | +2.13% | YES |
| 13:42 | SOUN | BUY | 2B/3S | $8.52 | +2.58% | YES |
| 13:42 | SMCI | BUY | 3B/2S | $33.81 | -4.05% | **NO** |
| 13:42 | TSM | SELL | 6B/3S | $391.98 | -4.28% | YES |
| 13:42 | TEM | BUY | 2B/2S | $53.73 | -0.81% | **NO** |
| 13:42 | VERI | BUY | 3B/1S | $2.96 | -4.55% | **NO** |
| 13:42 | LMT | BUY | 2B/4S | $647.64 | +1.24% | YES |
| 14:02 | BABA | BUY | 3B/6S | $148.07 | -2.40% | **NO** |
| 14:02 | GOOGL | BUY | 6B/3S | $313.99 | -1.93% | **NO** |
| 14:02 | AMZN | SELL | 5B/1S | $210.67 | -0.81% | YES |
| 14:02 | GRRR | BUY | 3B/3S | $11.50 | +2.17% | YES |
| 14:02 | SOUN | BUY | 2B/3S | $8.52 | +3.11% | YES |
| 14:02 | SMCI | BUY | 3B/2S | $33.81 | -3.96% | **NO** |
| 14:02 | TSM | SELL | 6B/3S | $391.98 | -4.26% | YES |
| 14:02 | TEM | BUY | 2B/2S | $53.73 | -0.76% | **NO** |
| 14:02 | VERI | BUY | 3B/1S | $2.96 | -4.72% | **NO** |
| 14:02 | LMT | BUY | 2B/4S | $647.64 | +1.24% | YES |
| 14:07 | BTC-USD | BUY | 4B/3S | $67,774.24 | -2.48% | **NO** |
| 14:07 | BABA | BUY | 3B/6S | $148.07 | -2.40% | **NO** |
| 14:07 | GOOGL | BUY | 6B/3S | $313.99 | -1.93% | **NO** |
| 14:07 | AMZN | SELL | 5B/1S | $210.67 | -0.81% | YES |
| 14:07 | GRRR | BUY | 3B/3S | $11.50 | +2.17% | YES |
| 14:07 | SOUN | SELL | 3B/2S | $8.65 | +1.56% | **NO** |
| 14:07 | SMCI | BUY | 3B/2S | $33.81 | -3.96% | **NO** |
| 14:07 | TSM | SELL | 6B/3S | $391.98 | -4.26% | YES |
| 14:07 | TEM | BUY | 2B/2S | $53.73 | -0.76% | **NO** |
| 14:07 | VERI | BUY | 3B/1S | $2.96 | -4.72% | **NO** |
| 14:07 | LMT | BUY | 2B/4S | $647.64 | +1.24% | YES |
| 14:21 | BTC-USD | BUY | 4B/3S | $67,730.33 | -2.41% | **NO** |
| 14:21 | XAU-USD | BUY | 1B/4S | $5,164.16 | +1.50% | YES |
| 14:21 | BABA | BUY | 3B/6S | $148.07 | -2.40% | **NO** |
| 14:21 | GOOGL | BUY | 6B/3S | $313.99 | -1.93% | **NO** |
| 14:21 | AMZN | SELL | 5B/1S | $210.67 | -0.81% | YES |
| 14:21 | GRRR | BUY | 3B/3S | $11.50 | +2.17% | YES |
| 14:21 | SMCI | BUY | 3B/2S | $33.81 | -3.96% | **NO** |
| 14:21 | TSM | SELL | 6B/3S | $391.98 | -4.26% | YES |
| 14:21 | TEM | BUY | 2B/2S | $53.73 | -0.76% | **NO** |
| 14:21 | VERI | BUY | 3B/1S | $2.96 | -4.72% | **NO** |
| 14:21 | LMT | BUY | 2B/4S | $647.64 | +1.24% | YES |
| 14:26 | BTC-USD | BUY | 2B/3S | $67,926.59 | -2.70% | **NO** |
| 14:26 | ETH-USD | SELL | 2B/3S | $2,065.77 | -5.67% | YES |
| 14:26 | BABA | BUY | 3B/6S | $148.07 | -2.40% | **NO** |
| 14:26 | GOOGL | BUY | 6B/3S | $313.99 | -1.93% | **NO** |
| 14:26 | AMZN | SELL | 5B/1S | $210.67 | -0.81% | YES |
| 14:26 | GRRR | BUY | 3B/3S | $11.50 | +2.17% | YES |
| 14:26 | SMCI | BUY | 3B/2S | $33.81 | -3.96% | **NO** |
| 14:26 | TEM | BUY | 2B/2S | $53.73 | -0.76% | **NO** |
| 14:26 | VERI | BUY | 3B/1S | $2.96 | -4.72% | **NO** |
| 14:26 | LMT | BUY | 2B/4S | $647.64 | +1.24% | YES |
| 14:41 | AMZN | BUY | 4B/1S | $210.48 | -0.72% | **NO** |
| 14:41 | AAPL | BUY | 4B/3S | $275.11 | -2.28% | **NO** |
| 14:41 | GRRR | BUY | 3B/3S | $11.50 | +2.17% | YES |
| 14:41 | TTWO | SELL | 4B/2S | $208.37 | +1.93% | **NO** |
| 14:41 | TEM | BUY | 4B/7S | $52.50 | +1.56% | YES |
| 14:41 | VERI | BUY | 3B/3S | $2.96 | -4.56% | **NO** |
| 14:41 | VRT | BUY | 3B/3S | $257.94 | -2.95% | **NO** |
| 14:41 | LMT | BUY | 4B/4S | $644.72 | +1.70% | YES |
| 14:45 | BTC-USD | BUY | 2B/3S | $67,927.14 | -2.70% | **NO** |
| 14:45 | AMZN | BUY | 4B/1S | $210.52 | -0.74% | **NO** |
| 14:45 | GRRR | BUY | 4B/4S | $11.48 | +2.35% | YES |
| 14:45 | MU | BUY | 3B/2S | $424.20 | -2.80% | **NO** |
| 14:45 | TTWO | SELL | 6B/4S | $210.68 | +0.82% | **NO** |
| 14:45 | TEM | BUY | 4B/7S | $52.49 | +1.58% | YES |
| 14:45 | UPST | SELL | 4B/2S | $28.97 | -4.83% | YES |
| 14:45 | VERI | BUY | 3B/4S | $2.96 | -4.40% | **NO** |
| 14:45 | VRT | BUY | 4B/4S | $253.51 | -1.26% | **NO** |
| 14:45 | LMT | BUY | 4B/7S | $639.13 | +2.59% | YES |
| 15:00 | BTC-USD | BUY | 4B/4S | $67,490.35 | -2.99% | **NO** |
| 15:00 | ETH-USD | BUY | 4B/3S | $2,044.39 | -5.73% | **NO** |
| 15:00 | XAU-USD | BUY | 2B/4S | $5,169.89 | +1.18% | YES |
| 15:00 | MSTR | BUY | 1B/4S | $133.66 | -3.18% | **NO** |
| 15:00 | PLTR | SELL | 5B/2S | $136.33 | -0.15% | YES |
| 15:00 | NVDA | BUY | 4B/7S | $188.50 | -4.30% | **NO** |
| 15:00 | AMD | SELL | 5B/5S | $205.43 | -2.49% | YES |
| 15:00 | BABA | BUY | 2B/5S | $148.16 | -2.46% | **NO** |
| 15:00 | GOOGL | BUY | 4B/4S | $307.51 | +0.16% | YES |
| 15:00 | AMZN | BUY | 3B/4S | $209.38 | -0.16% | **NO** |
| 15:00 | AVGO | SELL | 4B/6S | $317.47 | +0.25% | **NO** |
| 15:00 | IONQ | SELL | 7B/3S | $40.62 | -8.11% | YES |
| 15:00 | META | SELL | 6B/1S | $658.87 | -2.08% | YES |
| 15:00 | MU | BUY | 4B/3S | $417.41 | -1.08% | **NO** |
| 15:00 | TSM | BUY | 4B/4S | $375.77 | -0.03% | **NO** |
| 15:00 | TEM | SELL | 3B/3S | $53.52 | -0.30% | YES |
| 15:00 | UPST | SELL | 7B/4S | $29.34 | -6.03% | YES |
| 15:00 | VERI | SELL | 3B/1S | $2.95 | -4.07% | YES |
| 15:00 | VRT | BUY | 4B/7S | $247.37 | +1.30% | YES |
| 15:00 | QQQ | BUY | 4B/3S | $612.10 | -0.73% | **NO** |
| 15:00 | LMT | BUY | 4B/5S | $640.77 | +2.33% | YES |
| 15:06 | BTC-USD | BUY | 3B/4S | $67,604.01 | -3.15% | **NO** |
| 15:06 | ETH-USD | BUY | 4B/2S | $2,048.24 | -5.91% | **NO** |
| 15:06 | NVDA | SELL | 4B/8S | $188.25 | -4.18% | YES |
| 15:06 | AMD | SELL | 2B/2S | $206.32 | -2.91% | YES |
| 15:06 | BABA | BUY | 2B/7S | $147.54 | -2.06% | **NO** |
| 15:06 | GOOGL | BUY | 5B/5S | $305.60 | +0.79% | YES |
| 15:06 | AMZN | BUY | 4B/2S | $208.89 | +0.08% | YES |
| 15:06 | AVGO | BUY | 3B/6S | $316.98 | +0.40% | YES |
| 15:06 | IONQ | BUY | 8B/3S | $40.78 | -8.47% | **NO** |
| 15:06 | MU | BUY | 3B/2S | $418.08 | -1.24% | **NO** |
| 15:06 | SMCI | BUY | 3B/2S | $32.21 | +0.98% | YES |
| 15:06 | TSM | SELL | 4B/6S | $377.70 | -0.53% | YES |
| 15:06 | TEM | BUY | 2B/3S | $53.04 | +0.60% | YES |
| 15:06 | UPST | SELL | 4B/3S | $29.25 | -5.73% | YES |
| 15:06 | VRT | BUY | 4B/6S | $249.75 | +0.33% | YES |
| 15:06 | QQQ | BUY | 4B/3S | $611.21 | -0.58% | **NO** |
| 15:19 | BTC-USD | BUY | 4B/3S | $67,556.44 | -3.08% | **NO** |
| 15:19 | ETH-USD | SELL | 5B/2S | $2,048.94 | -5.94% | YES |
| 15:19 | PLTR | BUY | 4B/2S | $136.53 | -0.29% | **NO** |
| 15:19 | NVDA | BUY | 4B/7S | $188.88 | -4.49% | **NO** |
| 15:19 | AMD | SELL | 2B/2S | $206.84 | -3.15% | YES |
| 15:19 | BABA | BUY | 2B/8S | $147.32 | -1.91% | **NO** |
| 15:19 | GOOGL | BUY | 5B/4S | $306.56 | +0.48% | YES |
| 15:19 | AMZN | BUY | 3B/2S | $209.63 | -0.27% | **NO** |
| 15:19 | AVGO | SELL | 4B/8S | $315.25 | +0.95% | **NO** |
| 15:19 | GRRR | SELL | 3B/2S | $11.44 | +2.36% | **NO** |
| 15:19 | IONQ | SELL | 9B/3S | $41.26 | -9.54% | YES |
| 15:19 | MU | SELL | 4B/4S | $413.06 | -0.04% | YES |
| 15:19 | SOUN | BUY | 5B/2S | $8.96 | -1.73% | **NO** |
| 15:19 | SMCI | BUY | 3B/2S | $31.84 | +2.17% | YES |
| 15:19 | TSM | SELL | 4B/5S | $372.94 | +0.73% | **NO** |
| 15:19 | TTWO | BUY | 3B/1S | $211.77 | +0.30% | YES |
| 15:19 | TEM | BUY | 2B/5S | $52.47 | +1.70% | YES |
| 15:19 | VERI | BUY | 4B/2S | $2.97 | -4.71% | **NO** |
| 15:19 | VRT | BUY | 4B/7S | $243.19 | +3.04% | YES |
| 15:19 | QQQ | BUY | 4B/5S | $607.27 | +0.06% | YES |
| 15:19 | LMT | BUY | 2B/4S | $641.47 | +2.22% | YES |
| 15:25 | BTC-USD | BUY | 4B/4S | $67,012.73 | -2.29% | **NO** |
| 15:25 | ETH-USD | BUY | 4B/6S | $2,023.19 | -4.75% | **NO** |
| 15:25 | XAG-USD | BUY | 3B/5S | $86.21 | +8.10% | YES |
| 15:25 | MSTR | BUY | 3B/2S | $132.01 | -1.97% | **NO** |
| 15:25 | NVDA | SELL | 4B/8S | $185.94 | -2.98% | YES |
| 15:25 | AMD | BUY | 4B/6S | $202.78 | -1.22% | **NO** |
| 15:25 | BABA | SELL | 3B/8S | $145.71 | -0.82% | YES |
| 15:25 | GOOGL | BUY | 4B/5S | $303.33 | +1.55% | YES |
| 15:25 | AMZN | BUY | 5B/5S | $207.73 | +0.64% | YES |
| 15:25 | AAPL | BUY | 4B/1S | $271.53 | -0.96% | **NO** |
| 15:25 | AVGO | BUY | 4B/7S | $310.05 | +2.64% | YES |
| 15:25 | GRRR | SELL | 2B/2S | $11.38 | +2.90% | **NO** |
| 15:25 | IONQ | BUY | 5B/2S | $39.85 | -6.34% | **NO** |
| 15:25 | MU | SELL | 4B/6S | $404.52 | +2.07% | **NO** |
| 15:25 | SOUN | BUY | 4B/0S | $8.81 | -0.17% | **NO** |
| 15:25 | SMCI | BUY | 3B/3S | $31.43 | +3.50% | YES |
| 15:25 | TSM | SELL | 4B/6S | $370.84 | +1.31% | **NO** |
| 15:25 | TTWO | BUY | 5B/0S | $211.00 | +0.66% | YES |
| 15:25 | TEM | BUY | 3B/6S | $52.21 | +2.20% | YES |
| 15:25 | VERI | BUY | 4B/2S | $2.96 | -4.23% | **NO** |
| 15:25 | VRT | BUY | 4B/7S | $242.53 | +3.32% | YES |
| 15:25 | QQQ | BUY | 4B/6S | $605.31 | +0.39% | YES |
| 15:25 | LMT | BUY | 2B/4S | $640.10 | +2.44% | YES |
| 15:39 | BTC-USD | BUY | 5B/4S | $67,225.98 | -2.60% | **NO** |
| 15:39 | ETH-USD | BUY | 4B/6S | $2,028.62 | -5.00% | **NO** |
| 15:39 | XAG-USD | BUY | 3B/5S | $86.02 | +8.34% | YES |
| 15:39 | NVDA | SELL | 4B/8S | $186.00 | -3.01% | YES |
| 15:39 | AMD | BUY | 4B/6S | $202.45 | -1.05% | **NO** |
| 15:39 | BABA | SELL | 3B/8S | $145.71 | -0.82% | YES |
| 15:39 | GOOGL | BUY | 4B/6S | $302.73 | +1.75% | YES |
| 15:39 | AMZN | BUY | 4B/6S | $207.22 | +0.89% | YES |
| 15:39 | AAPL | BUY | 4B/1S | $271.97 | -1.12% | **NO** |
| 15:39 | AVGO | SELL | 4B/8S | $310.24 | +2.58% | **NO** |
| 15:39 | GRRR | SELL | 2B/2S | $11.46 | +2.18% | **NO** |
| 15:39 | IONQ | SELL | 4B/2S | $40.01 | -6.71% | YES |
| 15:39 | MU | SELL | 4B/5S | $403.58 | +2.31% | **NO** |
| 15:39 | SOUN | BUY | 3B/0S | $8.81 | -0.11% | **NO** |
| 15:39 | SMCI | BUY | 3B/5S | $31.33 | +3.82% | YES |
| 15:39 | TSM | BUY | 4B/6S | $370.67 | +1.35% | YES |
| 15:39 | TTWO | BUY | 3B/1S | $210.00 | +1.14% | YES |
| 15:39 | TEM | SELL | 4B/5S | $52.06 | +2.50% | **NO** |
| 15:39 | UPST | SELL | 6B/0S | $29.21 | -5.61% | YES |
| 15:39 | VRT | BUY | 2B/5S | $245.01 | +2.27% | YES |
| 15:39 | QQQ | BUY | 4B/5S | $605.76 | +0.31% | YES |
| 15:39 | LMT | BUY | 2B/3S | $641.22 | +2.26% | YES |
| 15:57 | ETH-USD | BUY | 2B/4S | $2,032.16 | -5.17% | **NO** |
| 15:57 | NVDA | BUY | 2B/7S | $185.99 | -3.01% | **NO** |
| 15:57 | AMD | SELL | 2B/3S | $203.70 | -1.66% | YES |
| 15:57 | BABA | SELL | 1B/5S | $147.18 | -1.82% | YES |
| 15:57 | GOOGL | BUY | 2B/3S | $303.93 | +1.35% | YES |
| 15:57 | AMZN | SELL | 3B/3S | $208.51 | +0.26% | **NO** |
| 15:57 | AAPL | BUY | 4B/1S | $271.41 | -0.91% | **NO** |
| 15:57 | AVGO | BUY | 2B/4S | $311.31 | +2.23% | YES |
| 15:57 | IONQ | SELL | 6B/2S | $40.18 | -7.11% | YES |
| 15:57 | MU | SELL | 2B/4S | $409.76 | +0.77% | **NO** |
| 15:57 | SOUN | SELL | 6B/2S | $9.06 | -2.87% | YES |
| 15:57 | TSM | SELL | 2B/3S | $374.31 | +0.37% | **NO** |
| 15:57 | TTWO | SELL | 5B/1S | $211.41 | +0.47% | **NO** |
| 15:57 | TEM | SELL | 2B/4S | $52.56 | +1.52% | **NO** |
| 15:57 | VERI | SELL | 5B/0S | $3.00 | -5.67% | YES |
| 15:57 | VRT | SELL | 4B/3S | $248.56 | +0.81% | **NO** |
| 15:57 | QQQ | BUY | 2B/3S | $607.26 | +0.07% | YES |
| 16:04 | NVDA | SELL | 3B/3S | $186.96 | -3.40% | YES |
| 16:04 | AMD | SELL | 2B/3S | $203.04 | -1.25% | YES |
| 16:04 | BABA | SELL | 0B/3S | $147.34 | -1.90% | YES |
| 16:04 | GOOGL | SELL | 1B/2S | $304.90 | +1.12% | **NO** |
| 16:04 | AMZN | BUY | 3B/2S | $208.34 | +0.37% | YES |
| 16:04 | AAPL | BUY | 4B/2S | $270.92 | -0.68% | **NO** |
| 16:04 | AVGO | SELL | 2B/5S | $310.39 | +2.50% | **NO** |
| 16:04 | GRRR | SELL | 3B/2S | $11.72 | +0.30% | **NO** |
| 16:04 | IONQ | SELL | 7B/1S | $40.23 | -6.82% | YES |
| 16:04 | MU | SELL | 2B/3S | $408.23 | +1.03% | **NO** |
| 16:04 | SOUN | SELL | 4B/0S | $8.98 | -1.89% | YES |
| 16:04 | SMCI | SELL | 1B/2S | $31.68 | +2.59% | **NO** |
| 16:04 | TSM | BUY | 2B/4S | $373.85 | +0.46% | YES |
| 16:04 | TTWO | BUY | 4B/0S | $212.16 | +0.03% | YES |
| 16:04 | TEM | SELL | 1B/4S | $52.48 | +1.62% | **NO** |
| 16:04 | UPST | SELL | 5B/1S | $29.53 | -6.65% | YES |
| 16:04 | VERI | SELL | 6B/1S | $2.99 | -5.35% | YES |
| 16:04 | VRT | BUY | 2B/2S | $250.54 | +0.01% | YES |
| 16:04 | LMT | SELL | 2B/3S | $644.47 | +1.67% | **NO** |
| 16:18 | BTC-USD | BUY | 2B/4S | $67,239.20 | -2.44% | **NO** |
| 16:18 | ETH-USD | BUY | 3B/4S | $2,027.03 | -4.78% | **NO** |
| 16:18 | NVDA | SELL | 3B/4S | $187.19 | -3.52% | YES |
| 16:18 | AMD | SELL | 2B/3S | $203.04 | -1.25% | YES |
| 16:18 | BABA | BUY | 1B/4S | $146.92 | -1.62% | **NO** |
| 16:18 | GOOGL | BUY | 2B/2S | $305.41 | +0.95% | YES |
| 16:18 | AMZN | BUY | 3B/2S | $208.48 | +0.30% | YES |
| 16:18 | AVGO | SELL | 2B/5S | $310.80 | +2.36% | **NO** |
| 16:18 | GRRR | SELL | 4B/2S | $11.74 | +0.13% | **NO** |
| 16:18 | IONQ | SELL | 7B/1S | $40.12 | -6.57% | YES |
| 16:18 | MU | SELL | 2B/3S | $408.67 | +0.92% | **NO** |
| 16:18 | SOUN | SELL | 5B/1S | $9.01 | -2.22% | YES |
| 16:18 | SMCI | SELL | 3B/1S | $31.80 | +2.21% | **NO** |
| 16:18 | TSM | BUY | 2B/4S | $372.80 | +0.75% | YES |
| 16:18 | TTWO | BUY | 4B/1S | $212.37 | -0.07% | **NO** |
| 16:18 | TEM | SELL | 2B/2S | $53.11 | +0.42% | **NO** |
| 16:18 | UPST | SELL | 10B/3S | $29.99 | -8.09% | YES |
| 16:18 | VERI | SELL | 6B/1S | $2.99 | -5.35% | YES |
| 16:18 | VRT | SELL | 3B/2S | $251.82 | -0.49% | YES |
| 16:32 | BTC-USD | BUY | 2B/3S | $67,204.88 | -2.39% | **NO** |
| 16:32 | ETH-USD | SELL | 3B/3S | $2,027.26 | -4.79% | YES |
| 16:32 | NVDA | SELL | 2B/3S | $187.34 | -3.60% | YES |
| 16:32 | AMD | SELL | 2B/3S | $203.20 | -1.32% | YES |
| 16:32 | BABA | SELL | 0B/3S | $147.07 | -1.72% | YES |
| 16:32 | GOOGL | BUY | 2B/2S | $306.57 | +0.57% | YES |
| 16:32 | AVGO | SELL | 2B/3S | $314.01 | +1.32% | **NO** |
| 16:32 | IONQ | SELL | 6B/1S | $40.45 | -7.32% | YES |
| 16:32 | MU | SELL | 1B/3S | $414.38 | -0.47% | YES |
| 16:32 | SOUN | SELL | 5B/0S | $8.84 | -0.34% | YES |
| 16:32 | TSM | SELL | 2B/3S | $375.79 | -0.06% | YES |
| 16:32 | UPST | SELL | 7B/1S | $29.66 | -7.06% | YES |
| 16:32 | VERI | SELL | 5B/1S | $2.99 | -5.35% | YES |
| 16:32 | VRT | SELL | 3B/1S | $254.64 | -1.60% | YES |
| 16:38 | ETH-USD | SELL | 2B/3S | $2,025.00 | -4.68% | YES |
| 16:38 | NVDA | SELL | 2B/3S | $187.83 | -3.85% | YES |
| 16:38 | AMD | SELL | 2B/4S | $203.74 | -1.59% | YES |
| 16:38 | BABA | SELL | 0B/5S | $146.85 | -1.57% | YES |
| 16:38 | AVGO | SELL | 2B/4S | $314.89 | +1.04% | **NO** |
| 16:38 | IONQ | BUY | 5B/1S | $40.01 | -6.31% | **NO** |
| 16:38 | MU | SELL | 1B/4S | $415.99 | -0.85% | YES |
| 16:38 | SOUN | BUY | 3B/0S | $8.86 | -0.57% | **NO** |
| 16:38 | TSM | SELL | 2B/3S | $375.29 | +0.08% | **NO** |
| 16:38 | TEM | SELL | 1B/3S | $52.71 | +1.19% | **NO** |
| 16:38 | UPST | BUY | 5B/1S | $29.53 | -6.65% | **NO** |
| 16:38 | VERI | BUY | 4B/2S | $2.99 | -5.35% | **NO** |
| 16:38 | VRT | SELL | 3B/1S | $255.12 | -1.78% | YES |
| 16:53 | ETH-USD | BUY | 4B/4S | $2,012.01 | -4.07% | **NO** |
| 16:53 | NVDA | SELL | 2B/4S | $187.47 | -3.67% | YES |
| 16:53 | AMD | SELL | 2B/4S | $203.17 | -1.31% | YES |
| 16:53 | BABA | SELL | 1B/5S | $146.74 | -1.50% | YES |
| 16:53 | GOOGL | BUY | 2B/2S | $306.24 | +0.68% | YES |
| 16:53 | AVGO | SELL | 2B/3S | $313.61 | +1.45% | **NO** |
| 16:53 | IONQ | SELL | 5B/1S | $40.09 | -6.51% | YES |
| 16:53 | MU | BUY | 1B/4S | $414.48 | -0.49% | **NO** |
| 16:53 | TEM | SELL | 1B/2S | $52.26 | +2.06% | **NO** |
| 16:53 | UPST | BUY | 3B/0S | $29.15 | -5.44% | **NO** |
| 16:59 | BTC-USD | BUY | 3B/5S | $66,649.47 | -1.57% | **NO** |
| 16:59 | ETH-USD | BUY | 4B/7S | $1,983.61 | -2.69% | **NO** |
| 16:59 | MSTR | SELL | 3B/2S | $130.56 | -0.70% | YES |
| 16:59 | NVDA | SELL | 2B/4S | $186.48 | -3.16% | YES |
| 16:59 | AMD | SELL | 2B/3S | $202.37 | -0.92% | YES |
| 16:59 | BABA | SELL | 1B/5S | $146.34 | -1.23% | YES |
| 16:59 | GOOGL | BUY | 1B/3S | $305.39 | +0.96% | YES |
| 16:59 | IONQ | SELL | 4B/1S | $40.13 | -6.59% | YES |
| 16:59 | MU | BUY | 2B/4S | $413.99 | -0.37% | **NO** |
| 16:59 | TEM | SELL | 1B/2S | $52.24 | +2.09% | **NO** |
| 16:59 | UPST | BUY | 2B/1S | $29.01 | -4.98% | **NO** |
| 17:13 | BTC-USD | SELL | 2B/7S | $66,692.98 | -2.07% | YES |
| 17:13 | ETH-USD | BUY | 3B/5S | $1,988.84 | -3.56% | **NO** |
| 17:13 | MSTR | BUY | 2B/3S | $130.25 | -0.58% | **NO** |
| 17:13 | NVDA | SELL | 1B/4S | $186.38 | -4.93% | YES |
| 17:13 | AMD | SELL | 2B/3S | $202.63 | -1.19% | YES |
| 17:13 | BABA | SELL | 0B/3S | $146.55 | -1.66% | YES |
| 17:13 | MU | SELL | 2B/4S | $416.12 | -0.90% | YES |
| 17:13 | TEM | SELL | 1B/2S | $52.18 | +2.05% | **NO** |
| 17:13 | UPST | BUY | 2B/1S | $29.11 | -6.44% | **NO** |
| 17:29 | BTC-USD | SELL | 2B/6S | $66,890.89 | -2.36% | YES |
| 17:29 | ETH-USD | SELL | 2B/6S | $1,988.79 | -3.56% | YES |
| 17:29 | MSTR | SELL | 2B/2S | $130.00 | -0.38% | YES |
| 17:29 | NVDA | SELL | 1B/4S | $187.06 | -5.28% | YES |
| 17:29 | AMD | SELL | 2B/3S | $202.63 | -1.19% | YES |
| 17:29 | AMZN | BUY | 1B/4S | $206.74 | +1.58% | YES |
| 17:29 | IONQ | BUY | 4B/2S | $40.42 | -5.07% | **NO** |
| 17:29 | MU | BUY | 2B/4S | $413.44 | -0.26% | **NO** |
| 17:29 | TSM | SELL | 2B/3S | $372.97 | +0.43% | **NO** |
| 17:29 | TEM | SELL | 1B/3S | $52.08 | +2.25% | **NO** |
| 17:29 | VERI | SELL | 3B/2S | $2.91 | -3.09% | YES |
| 17:29 | VRT | SELL | 2B/2S | $252.41 | +0.98% | **NO** |
| 17:47 | ETH-USD | BUY | 2B/5S | $1,994.56 | -3.84% | **NO** |
| 17:47 | MSTR | SELL | 1B/2S | $130.70 | -0.92% | YES |
| 17:47 | AMD | SELL | 2B/3S | $203.05 | -1.40% | YES |
| 17:47 | BABA | SELL | 0B/3S | $146.79 | -1.83% | YES |
| 17:47 | IONQ | SELL | 5B/1S | $40.67 | -5.66% | YES |
| 17:47 | MU | BUY | 2B/4S | $412.39 | -0.00% | **NO** |
| 17:47 | TTWO | BUY | 3B/0S | $211.31 | +0.08% | YES |
| 17:47 | TEM | SELL | 1B/2S | $52.09 | +2.23% | **NO** |
| 17:47 | VERI | BUY | 4B/3S | $2.90 | -2.59% | **NO** |
| 17:47 | VRT | SELL | 3B/1S | $253.04 | +0.73% | **NO** |
| 18:08 | MSTR | SELL | 2B/3S | $129.07 | +0.33% | **NO** |
| 18:08 | NVDA | SELL | 2B/3S | $186.42 | -4.95% | YES |
| 18:08 | AMD | SELL | 2B/3S | $202.20 | -0.98% | YES |
| 18:08 | AMZN | BUY | 2B/4S | $206.10 | +1.89% | YES |
| 18:08 | IONQ | BUY | 4B/1S | $40.27 | -4.72% | **NO** |
| 18:08 | TEM | SELL | 1B/3S | $52.08 | +2.25% | **NO** |
| 18:08 | QQQ | BUY | 3B/3S | $605.05 | +0.37% | YES |

---

## Summary & Recommendations

- **1d consensus accuracy:** 53.5% (183/342)
- **3d consensus accuracy:** 57.7% (127/220)

### Below 70% threshold — improvement needed

**Signals performing below 50% (coin-flip):**

- `fear_greed`: 0.0% (0/52)
- `smart_money`: 13.8% (4/29)
- `oscillators`: 25.0% (3/12)
- `structure`: 30.0% (24/80)
- `macro_regime`: 32.3% (10/31)
- `trend`: 32.9% (99/301)
- `sentiment`: 34.8% (154/442)
- `momentum_factors`: 35.8% (19/53)
- `candlestick`: 38.2% (39/102)
- `ema`: 38.3% (111/290)
- `heikin_ashi`: 38.8% (104/268)
- `news_event`: 40.7% (22/54)
- `volatility_sig`: 43.1% (59/137)
- `volume_flow`: 45.8% (247/539)
- `volume`: 48.2% (108/224)

**Recommended actions:**

1. Verify signal inversion is active for sub-50% signals
2. Check if these signals should be disabled entirely
3. Investigate if a specific market condition (ranging/trending) caused systematic failures