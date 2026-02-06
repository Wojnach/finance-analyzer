# Finance Analyzer — TODO

## Phase 1: Foundation (In Progress)

### Done

- [x] Freqtrade running in Podman (`docker.io/freqtradeorg/freqtrade:stable`, v2026.1)
- [x] TABaseStrategy — RSI(14), MACD(12/26/9), EMA(9/21), Volume SMA(20), confidence scoring, all params hyperoptable
- [x] Podman wrapper scripts (`scripts/ft*.sh`)
- [x] Config template with strategy, order_types, unfilledtimeout
- [x] Unit tests (22/26 pass, 4 pre-existing in reference code)
- [x] Integration tests (6/6 pass inside container)
- [x] 30 days BTC/ETH 5m data downloaded
- [x] Backtest runs (50 trades, 12% drawdown, outperformed buy-and-hold by 21%)

### Next Up

- [ ] **Binance API keys** — read-only keys sufficient for dry run. Create `config.json` from `config.example.json`
- [ ] **Hyperopt** — tune 5 params (rsi_oversold, rsi_overbought, min_confidence, volume_spike_mult, base_confidence). Target: profit factor > 1.0
- [ ] **Telegram bot** — create via @BotFather, add token + chat_id to config.json
- [ ] **Dry run** — paper trade 48h, verify trades fire (`./scripts/ft-dry-run.sh`)
- [ ] **More data** — 2 years across 5m, 1h, 4h, 1d timeframes
- [ ] **Fix unit tests** — 4 failures in indicators.py (pandas shift+invert bug, fix known)

### Success Criteria

| Metric                 | Target    | Current        |
| ---------------------- | --------- | -------------- |
| Unit tests             | 100% pass | 85%            |
| Backtest profit factor | > 1.0     | 0.27           |
| Backtest max drawdown  | < 30%     | 12.12%         |
| Dry run trades (48h)   | > 0       | Not started    |
| Telegram delivery      | 100%      | Not configured |

### Last Backtest (2026-02-06)

- 50 trades, 24% win rate, -10.69% (market was -31.96%)
- ROI exits working well (+1.24% avg), exit signals need tuning (-0.73% avg)

## Future Phases

- Avanza stock monitoring (alerts only)
- Sentiment analysis
- Dashboard (FastAPI + frontend)
- FreqAI ML models
