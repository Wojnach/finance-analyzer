# Finance Analyzer — TODO

> **Revised 2026-02-09.** See `docs/plans/` for full research docs.

## Phase 1: Foundation + Data Validation

### Done

- [x] Freqtrade in Podman (`freqtradeorg/freqtrade:stable`, v2026.1, `--network=host`)
- [x] TABaseStrategy — RSI(14), MACD(12/26/9), EMA(9/21), Volume SMA(20), confidence scoring, all hyperoptable
- [x] Podman wrapper scripts (`scripts/ft*.sh` — backtest, dry-run, hyperopt, download, test)
- [x] Config template with labeled fake keys
- [x] Unit tests 26/26 pass, integration tests 6/6 pass
- [x] Hyperopt 500 epochs → profit factor 1.77, 0.49% drawdown (on 30 days — **not validated**)

### Current Backtest (2026-02-09, 30 days only)

- 13 trades, 46.2% win rate, +0.58% — **too few trades to be statistically significant**
- Profit factor 1.77 on 13 trades could easily be luck
- Current params almost certainly overfit to this 30-day window

### Step 1 — Get Online (manual, no code)

- [ ] **Binance API keys** — create `config.json` from template, read-only keys
- [ ] **Telegram bot** — @BotFather, add token + chat_id to config.json

### Step 2 — Get Real Data

- [ ] **Download 2 years** of data: `./scripts/ft-download-data.sh 730 5m 1h 4h 1d`
- [ ] Verify data integrity (no gaps, correct timeframes)

### Step 3 — Validate the Strategy (CRITICAL)

- [ ] **Re-run hyperopt** on 2yr data (current 30-day params are likely overfit)
- [ ] **Walk-forward validation** — split into 6-month train / 2-month test windows, roll forward
- [ ] Target: **100+ trades** in backtest for statistical significance
- [ ] Test across market regimes: bull run, bear crash, sideways chop (2yr data has all three)
- [ ] **Model fee/slippage** — verify backtest accounts for funding rates and realistic slippage

### Step 4 — Dry Run

- [ ] Paper trade 48h+ with validated params (`./scripts/ft-dry-run.sh`)
- [ ] Verify trades fire and match backtest behavior
- [ ] Monitor via Telegram

### Success Criteria (Phase 1 Complete When ALL Met)

| Metric                     | Target                   | Current          |
| -------------------------- | ------------------------ | ---------------- |
| Historical data            | 2 years                  | 30 days          |
| Backtest trades            | > 100                    | 13               |
| Walk-forward profit factor | > 1.0 across all windows | Not tested       |
| Backtest max drawdown      | < 20%                    | 0.49% (overfit?) |
| Dry run trades (48h)       | > 0                      | Not started      |
| Telegram delivery          | 100%                     | Not configured   |

---

## Phase 2: Strategy Hardening

> Only start after Phase 1 validation passes on 2yr data.

### Step 1 — Exit Logic Overhaul (highest impact)

The #1 problem: exit signals lose money (-1.17% avg). Most profitable strategies don't use exit signals — they use ROI + custom_exit + trailing stop.

- [ ] Implement `custom_exit()` — profit-based (take profit tiers) + time-based (unclog stale trades)
- [ ] Implement `custom_stoploss()` — ATR-based dynamic stops instead of fixed %
- [ ] Backtest exits in isolation: compare custom_exit vs current exit_signal vs ROI-only
- [ ] Target: zero losing exit_signal trades

### Step 2 — Multi-Timeframe + Trend Filter

Proven technique, standard in profitable Freqtrade strategies.

- [ ] Add `informative_pairs()` for 1h timeframe
- [ ] Add **ADX(14)** — only trade when ADX > 25 (avoid choppy/ranging markets)
- [ ] Add **ATR(14)** — used by custom_stoploss in Step 1
- [ ] Use 1h EMA trend as filter: only enter 5m longs when 1h trend is up
- [ ] Backtest with vs without trend filter

### Step 3 — Candlestick Patterns (selective, not all 61)

Most of the 61 TA-Lib patterns are noise. Use the proven high-signal ones only.

- [ ] Add top ~15 patterns: CDLHAMMER, CDLENGULFING, CDLMORNINGSTAR, CDLEVENINGSTAR, CDL3WHITESOLDIERS, CDL3BLACKCROWS, CDLSHOOTINGSTAR, CDLHANGINGMAN, CDLDOJI, CDLHARAMI, CDLPIERCING, CDLDARKCLOUDCOVER, CDLINVERTEDHAMMER, CDLMARUBOZU, CDLKICKING
- [ ] Add as weighted confidence factor (not hard signal — confirmation only)
- [ ] Hyperopt the pattern weights
- [ ] If >15 patterns don't improve backtest, cut back further

### Step 4 — Re-Hyperopt + Walk-Forward

- [ ] Re-run hyperopt with all new indicators (exits, ADX, ATR, 1h filter, candlestick patterns)
- [ ] Walk-forward validate again on the full 2yr dataset
- [ ] Compare before/after: must beat Phase 1 validated results

### Step 5 — Risk Management (MISSING from original plan)

- [ ] **Max daily loss limit** — stop trading for the day if drawdown exceeds X%
- [ ] **Position sizing** — stake_amount based on ATR/volatility, not fixed "unlimited"
- [ ] **Max open trades** review — currently 3, is this right for BTC+ETH on 5m?
- [ ] **Drawdown kill switch** — if total drawdown hits 15%, pause bot and alert via Telegram

### Deferred from Original Phase 2

These items are NOT proven to add value and have weak library support:

- ~~Elliott Wave~~ — python-taew has 22 stars, last update 2021, dead library. Elliott Wave is subjective.
- ~~Harmonic Patterns~~ — HarmonicPatterns is archived. Niche technique, controversial effectiveness.
- ~~Short positions~~ — premature. Get longs profitable first. Revisit after Phase 2 validation.
- ~~Classic chart patterns (TradingPatternScanner)~~ — 270 stars, unverified quality. Revisit if candlestick patterns + exits don't provide enough edge.
- ~~All 61 candlestick patterns~~ — reduced to top 15. Add more only if backtest demands it.

---

## Phase 3: ML Integration (FreqAI)

> Only start after Phase 2 strategy is walk-forward validated and profitable in dry run.

Full research: `docs/plans/2026-02-09-phase2-research.md` (FreqAI section)

### Step 1 — FreqAI with CatBoost

- [ ] Verify ML libs in container: `podman run --rm freqtradeorg/freqtrade:stable pip list | grep -E "(catboost|lightgbm|scikit)"`
- [ ] Add FreqAI config (CatBoostRegressor, train_period_days=90, backtest_period_days=30)
- [ ] Implement `feature_engineering_expand_all()` — use Phase 2 indicators as features
- [ ] Implement `set_freqai_targets()` — predict mean return over next 20 candles
- [ ] Use `do_predict` + `&-target` as additional entry filter (not replacement)
- [ ] **A/B backtest**: FreqAI-enhanced vs Phase 2 strategy. Must beat Phase 2 or don't ship it.

### Step 2 — Sentiment (if data source exists)

- [ ] **FIRST: Find a reliable crypto news/sentiment data source** — CryptoPanic API? LunarCrush? RSS?
- [ ] Evaluate: does FinBERT (trained on corporate text) even work on crypto tweets? Test before integrating.
- [ ] Alternative: skip FinBERT, use raw sentiment scores from LunarCrush/CryptoPanic (no model needed)
- [ ] If sentiment data source exists and adds alpha: integrate as FreqAI feature
- [ ] If no reliable source: skip sentiment entirely, ML on price/indicator features is enough

### Key Constraints

- CatBoost/LightGBM run on Steam Deck CPU (2-10min training per pair)
- Don't add network dependencies (no remote model serving for live trading)
- FreqAI retrains automatically — set reasonable frequency (every 7-30 days)

---

## Phase 4: Claude as Trading Copilot

> Only start after strategy is profitable in live dry run for 2+ weeks.

Full research: `docs/plans/2026-02-09-llm-trading-research.md`

### Concrete Use Case (must justify $10-28/month)

Claude adds value ONLY if it does things the automated strategy can't:

1. **Market regime analysis** — "BTC is entering distribution phase, reduce position sizes"
2. **News-driven risk alerts** — "Major exchange hack detected, exit all positions"
3. **Strategy review** — analyze losing streaks, suggest parameter adjustments
4. **Anomaly detection** — "bot made 5 trades in 10 minutes, something looks wrong"

If these don't happen often enough to justify cost, this phase gets skipped.

### Step 1 — MCP Server Setup

- [ ] Install **Freqtrade-MCP** (https://github.com/kukapay/freqtrade-mcp)
- [ ] Install **CCXT MCP** (https://github.com/Nayshins/mcp-server-ccxt)
- [ ] Test: can Claude read bot status, open trades, profit, balance?

### Step 2 — Weekly Review (cheapest entry point)

- [ ] Claude Haiku analyzes weekly performance (~$2/month)
- [ ] Summarizes: win rate, drawdown, regime, notable trades
- [ ] Suggests: parameter tweaks, pair additions/removals
- [ ] Human acts on recommendations manually
- [ ] **Evaluate after 1 month:** is this worth the cost? If not, cancel.

### Cost Estimates (Claude + prompt caching)

| Frequency       | Model  | Monthly |
| --------------- | ------ | ------- |
| Weekly review   | Haiku  | ~$2     |
| Daily review    | Haiku  | ~$10    |
| Hourly analysis | Sonnet | ~$28    |

Start at weekly. Only increase if proven valuable.

---

## Phase 5: Expansion (Only After Proven Profitability)

> Prerequisites: strategy profitable in live trading for 1+ month.

### 5a — Short Positions

- [ ] Add `can_short = True`, mirror long logic with inverted signals
- [ ] Tighter stops, smaller position size than longs
- [ ] Only short when 1h/4h trend is down + ADX > 25
- [ ] Backtest extensively before enabling in dry run

### 5b — Pair Expansion

- [ ] Add SOL/USDT:USDT (backtest first)
- [ ] Add XRP/USDT:USDT (backtest first)
- [ ] One pair at a time, validate each independently
- [ ] Consider VolumePairList for dynamic selection

### 5c — Classic Chart Patterns (if needed)

- [ ] Evaluate TradingPatternScanner quality on real data
- [ ] If it works: add H&S, triangles, double tops/bottoms as entry filters
- [ ] If it doesn't: build custom pattern detection for the top 3-5 patterns

### 5d — Bot Monitoring + Health

- [ ] Health check script (is container running? is bot trading? last trade time?)
- [ ] Telegram alert if bot stops or crashes
- [ ] Daily automated report: PnL, open trades, drawdown

---

## Watch List (Not Planned, Just Tracking)

Research docs: `docs/plans/2026-02-09-llm-trading-research.md`

| Project       | Stars | What                                      | Revisit When                        |
| ------------- | ----- | ----------------------------------------- | ----------------------------------- |
| TradingAgents | 29.6k | Multi-agent LLM trading (simulation only) | When it supports live exchanges     |
| FinGPT        | 18.5k | Fine-tune financial LLM for $300          | When we need custom sentiment model |
| FinRL         | 13.9k | RL trading framework                      | When we outgrow Freqtrade           |
| MAHORAGA      | 480   | Deployed Claude trading bot (Alpaca)      | When we want US stock trading       |
| Freqtrade-MCP | —     | Claude ↔ Freqtrade bridge                 | Phase 4                             |
| CCXT MCP      | —     | Claude reads 100+ exchanges               | Phase 4                             |

---

## Separate Projects (Not in This Repo)

- **Avanza stock alerts** — standalone polling script (~200 LOC), `avanza-api` on PyPI, unofficial API
- **Dashboard** — premature until live trading is profitable. FreqUI (built into Freqtrade) is sufficient for now.
