# Phase 1: Foundation Design

## Overview

Set up Freqtrade with Binance integration, basic TA strategy, and Telegram alerts.
Run in paper trading mode with full test coverage.

## Architecture

```
┌─────────────────────────────────────────┐
│            Freqtrade Core               │
│  ┌─────────────────────────────────┐   │
│  │     TABaseStrategy              │   │
│  │  - RSI (overbought/oversold)    │   │
│  │  - MACD (crossover)             │   │
│  │  - EMA crossovers (9/21, 50/200)│   │
│  │  - Volume spike detection       │   │
│  └─────────────────────────────────┘   │
└───────────────┬─────────────────────────┘
                │
    ┌───────────┴───────────┐
    ▼                       ▼
┌─────────┐           ┌──────────┐
│ Binance │           │ Telegram │
│  API    │           │   Bot    │
└─────────┘           └──────────┘
```

## Components

### 1. Freqtrade Configuration

- Binance exchange connection
- Trading pairs: BTC/USDT, ETH/USDT
- Dry-run mode enabled
- 5m and 1h timeframes

### 2. TABaseStrategy

Custom strategy implementing:

- **RSI**: Buy when < 30, sell when > 70
- **MACD**: Buy on bullish crossover, sell on bearish
- **EMA**: 9/21 fast cross, 50/200 slow cross
- **Volume**: Spike detection (> 2x average)

### 3. Telegram Integration

- Trade entry/exit notifications
- Daily summary reports
- Error alerts

## Test Strategy

### Unit Tests

- Individual indicator calculations
- Signal logic (buy/sell triggers)
- Risk management (stop-loss, take-profit)

### Integration Tests

- Strategy loads in Freqtrade
- Dry-run executes trades
- Telegram messages delivered

### Backtest Tests

- No look-ahead bias
- Metrics within expected bounds
- Comparison vs buy-and-hold

## Success Criteria

| Metric                 | Target    |
| ---------------------- | --------- |
| Unit tests             | 100% pass |
| Backtest profit factor | > 1.0     |
| Backtest max drawdown  | < 30%     |
| Dry run trades (48h)   | > 0       |
| Telegram delivery      | 100%      |

## Timeline

- Day 1: Environment setup, dependency installation
- Day 2: Freqtrade config, Binance connection, data download
- Day 3: TABaseStrategy implementation with TDD
- Day 4: Telegram integration
- Day 5: Backtesting and validation
- Day 6-7: Dry run monitoring

## Data Requirements

- BTC/USDT: 2 years historical (5m, 1h, 4h, 1d)
- ETH/USDT: 2 years historical (5m, 1h, 4h, 1d)
