# Finance Analyzer — TODO

## Phase 1: Foundation (In Progress)

### Done

- [x] Freqtrade running in Podman (`docker.io/freqtradeorg/freqtrade:stable`, v2026.1)
- [x] TABaseStrategy — RSI(14), MACD(12/26/9), EMA(9/21), Volume SMA(20), confidence scoring, all params hyperoptable
- [x] Podman wrapper scripts (`scripts/ft*.sh` — backtest, dry-run, hyperopt, download, test)
- [x] Config template with labeled fake keys, strategy, order_types, unfilledtimeout
- [x] Unit tests (26/26 pass) — pandas shift+invert bug fixed with `fill_value=False`
- [x] Integration tests (6/6 pass inside container via `ft-test.sh`)
- [x] 30 days BTC/ETH 5m data downloaded (+ 1h mark/funding rate)
- [x] Multi-timeframe download script (`./scripts/ft-download-data.sh 730 5m 1h 4h 1d`)
- [x] Hyperopt script (`./scripts/ft-hyperopt.sh`) — SharpeHyperOptLossDaily, buy+sell spaces
- [x] Hyperopt run (500 epochs, SharpeHyperOptLoss, all spaces) — params saved to `ta_base_strategy.json`, auto-loaded by Freqtrade
- [x] Podman DNS fix (`--network=host` in ft.sh)

### Hyperopt Results (2026-02-09)

Best params from 500 epochs (SharpeHyperOptLoss, buy/sell/roi/stoploss/trailing):

- rsi_oversold=38, rsi_overbought=67, min_confidence=0.638
- base_confidence=0.291, volume_spike_mult=1.728
- stoploss=-0.107, trailing_stop_positive=0.024 (offset 0.071)
- ROI: {0: 11.9%, 25: 7.8%, 36: 1.6%, 105: 0%}

### Last Backtest (2026-02-09, post-hyperopt)

- 13 trades, 46.2% win rate (6W/1D/6L), +0.58% ($5.76 on $1000)
- Market was -31.96% — strategy outperformed by 32.5%
- Max drawdown: 0.49%, profit factor: 1.77, Sharpe: 1.57, Sortino: 1.97
- ROI exits (11 trades, +0.4% avg) working well; exit_signal (2 trades, -1.17% avg) needs work
- Previous (pre-hyperopt): 50 trades, -10.69%, profit factor 0.27, 12.12% drawdown

### Next Up

- [ ] **Binance API keys** — create local `config.json` from template, insert read-only keys
- [ ] **Telegram bot** — create via @BotFather, add token + chat_id to config.json
- [ ] **Dry run** — paper trade 48h, verify trades fire (`./scripts/ft-dry-run.sh`)
- [ ] **More data** — download 2 years across 5m/1h/4h/1d (`./scripts/ft-download-data.sh 730 5m 1h 4h 1d`)
- [ ] **Re-run hyperopt** on larger dataset (current params tuned on only 30 days)

### Success Criteria

| Metric                 | Target    | Current        |
| ---------------------- | --------- | -------------- |
| Unit tests             | 100% pass | 100% (26/26)   |
| Backtest profit factor | > 1.0     | 1.77 ✓         |
| Backtest max drawdown  | < 30%     | 0.49% ✓        |
| Dry run trades (48h)   | > 0       | Not started    |
| Telegram delivery      | 100%      | Not configured |

## Future Phases

- Avanza stock monitoring (alerts only)
- Sentiment analysis
- Dashboard (FastAPI + frontend)
- FreqAI ML models
