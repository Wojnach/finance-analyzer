# Finance Analyzer — TODO

> **Updated 2026-02-10.** See `docs/plans/` for research docs, `SESSION_PLAN.md` for parallel work.

## Phase 1: Foundation + Data Validation — COMPLETE

- [x] Freqtrade in Podman (`freqtradeorg/freqtrade:stable`, v2026.1, `--network=host`)
- [x] TABaseStrategy — RSI(14), MACD(12/26/9), EMA(9/21), Volume SMA(20), ATR(14), confidence scoring, all hyperoptable
- [x] Podman wrapper scripts (`scripts/ft*.sh` — backtest, dry-run, hyperopt, download, test, walkforward)
- [x] Config template with labeled fake keys
- [x] Unit tests 26/26 pass, integration tests 9/9 pass
- [x] Binance API keys — read-only, in `config.json` (gitignored), API name "Steamdeck2"
- [x] Telegram bot — @RaanmanFinanceBot, chat ID 8455715091, verified working
- [x] Download 2yr data (2024-02-10 to present, 5m/1h/4h/1d, ~14MB feather)
- [x] Hyperopt on 2yr data — confirmed 30-day results were overfit
- [x] Exit overhaul — ATR trailing stoploss via `custom_stoploss()`, simplified single-signal exit, stale trade timeout via `custom_exit()`
- [x] Dry run started 2026-02-09 — container running

### Latest Backtest (2026-02-10, 2yr data, post exit overhaul)

- 174 trades over 730 days, 32.8% win rate (57W/117L)
- -0.30% total profit (-$3.03 on $1000), market was +16.89%
- Profit factor < 1.0, max drawdown 3.16%
- Exit reasons: trailing_stop_loss (ATR), exit_signal, stop_loss (floor)
- **Strategy is not yet profitable — Phase 2 improvements needed**

### Success Criteria

| Metric                     | Target                   | Current           |
| -------------------------- | ------------------------ | ----------------- |
| Historical data            | 2 years                  | 2 years ✓         |
| Backtest trades            | > 100                    | 174 ✓             |
| Walk-forward profit factor | > 1.0 across all windows | Not yet run       |
| Backtest max drawdown      | < 20%                    | 3.16% ✓           |
| Dry run trades (48h)       | > 0                      | Running since 2/9 |
| Telegram delivery          | 100%                     | Verified ✓        |

---

## Phase 2: Strategy Hardening — IN PROGRESS

> Two parallel sessions working. See `SESSION_PLAN.md` for division of labor.

### ~~Step 1 — Exit Logic Overhaul~~ DONE (2026-02-10)

- [x] `custom_stoploss()` — ATR-based dynamic trailing stop (atr_sl_mult=3.641)
- [x] `custom_exit()` — simplified single-signal exit + stale trade timeout (max_trade_candles=457)
- [x] Static trailing stop removed, replaced by ATR-based custom_stoploss
- [x] Hyperoptable params: atr_sl_mult, max_trade_candles, rsi_overbought

### Step 2 — Multi-Timeframe + Trend Filter (Session A)

- [ ] Add `informative_pairs()` for 1h timeframe
- [ ] Merge 1h EMA(50) trend into 5m dataframe
- [ ] Add **ADX(14)** on 1h — only enter when ADX > 25 (trending market)
- [ ] Gate entries: require 1h EMA uptrend + ADX > 25

### Step 3 — Candlestick Patterns (Session A)

- [ ] Add top 15 TA-Lib patterns as weighted confidence factor
- [ ] Hyperoptable pattern weight parameter
- [ ] Patterns: CDLHAMMER, CDLENGULFING, CDLMORNINGSTAR, CDLEVENINGSTAR, CDL3WHITESOLDIERS, CDL3BLACKCROWS, CDLSHOOTINGSTAR, CDLHANGINGMAN, CDLDOJI, CDLHARAMI, CDLPIERCING, CDLDARKCLOUDCOVER, CDLINVERTEDHAMMER, CDLMARUBOZU, CDLKICKING

### Step 4 — Risk Management (Session A)

- [ ] ATR-based position sizing via `custom_stake_amount()`
- [ ] Max daily loss tracking — stop entering after X% daily drawdown
- [ ] Drawdown kill switch — block new entries + Telegram alert at threshold

### Step 5 — Infrastructure + Validation (Session B)

- [x] Walk-forward validation script (`scripts/ft-walkforward.py`) — 9 rolling windows, 6mo train / 2mo test
- [ ] Systemd user service for dry-run persistence across reboots
- [ ] Health check script with optional Telegram alerts
- [ ] Run walk-forward validation after Session A merges

### Step 6 — Re-Hyperopt + Final Validation

- [ ] Re-hyperopt with all new indicators on 2yr data
- [ ] Walk-forward validate — must show avg profit factor > 1.0
- [ ] Compare before/after: must beat Phase 1 results

### Deferred

- ~~Elliott Wave~~ — dead library (22 stars, last update 2021)
- ~~Harmonic Patterns~~ — archived repo
- ~~Short positions~~ — get longs profitable first
- ~~All 61 candlestick patterns~~ — reduced to top 15

---

## Phase 3: ML Integration (FreqAI)

> Only start after Phase 2 is walk-forward validated and profitable in dry run.

- [ ] FreqAI with CatBoost on Steam Deck CPU
- [ ] Use Phase 2 indicators as features
- [ ] A/B test: must beat Phase 2 or don't ship
- [ ] Sentiment only if reliable crypto data source found

Full plan: `docs/plans/2026-02-09-phase2-research.md`

---

## Phase 4: Claude as Trading Copilot

> Only start after 2+ weeks profitable dry run.

- [ ] Freqtrade-MCP + CCXT MCP setup
- [ ] Weekly Haiku review (~$2/month) — evaluate after 1 month
- [ ] Use cases: regime analysis, news alerts, anomaly detection

Full plan: `docs/plans/2026-02-09-llm-trading-research.md`

---

## Phase 5: Expansion (after 1+ month profitable)

- [ ] Short positions
- [ ] Pair expansion (SOL, XRP — one at a time)
- [ ] Classic chart patterns (if needed)
- [ ] Daily automated reports

---

## Watch List

| Project       | Stars | What                                      | Revisit When                        |
| ------------- | ----- | ----------------------------------------- | ----------------------------------- |
| TradingAgents | 29.6k | Multi-agent LLM trading (simulation only) | When it supports live exchanges     |
| FinGPT        | 18.5k | Fine-tune financial LLM for $300          | When we need custom sentiment model |
| FinRL         | 13.9k | RL trading framework                      | When we outgrow Freqtrade           |
| MAHORAGA      | 480   | Deployed Claude trading bot (Alpaca)      | When we want US stock trading       |
| Freqtrade-MCP | —     | Claude ↔ Freqtrade bridge                 | Phase 4                             |
| CCXT MCP      | —     | Claude reads 100+ exchanges               | Phase 4                             |
