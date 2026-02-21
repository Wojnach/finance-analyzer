# Finance Analyzer — TODO

> **Updated 2026-02-21.**

## Phase 1: Foundation + Data Validation — COMPLETE

- [x] Freqtrade in Podman (`freqtradeorg/freqtrade:stable`, v2026.1, `--network=host`)
- [x] TABaseStrategy — RSI(14), MACD(12/26/9), EMA(9/21), Volume SMA(20), ATR(14), confidence scoring, all hyperoptable
- [x] Podman wrapper scripts (`scripts/ft*.sh` — backtest, dry-run, hyperopt, download, test, walkforward, health)
- [x] Config template with labeled fake keys
- [x] Unit tests 26/26 pass, integration tests 16/16 pass
- [x] Binance API keys — read-only, in `config.json` (gitignored), API name "Steamdeck2"
- [x] Telegram bot — @RaanmanFinanceBot, chat ID 8455715091, verified working
- [x] Download 2yr data (2024-02-10 to present, 5m/1h/4h/1d, ~14MB feather)
- [x] Hyperopt on 2yr data — confirmed 30-day results were overfit
- [x] Exit overhaul — ATR trailing stoploss, simplified single-signal exit, stale trade timeout
- [x] Dry run started 2026-02-09

---

## Phase 2: Strategy Hardening — BLOCKED

### ~~Step 1 — Exit Logic Overhaul~~ DONE

- [x] `custom_stoploss()` — ATR-based dynamic trailing stop
- [x] `custom_exit()` — simplified single-signal exit + stale trade timeout
- [x] Hyperoptable params: atr_sl_mult, max_trade_candles, rsi_overbought

### ~~Step 2 — Multi-Timeframe + Trend Filter~~ DONE

- [x] `informative_pairs()` for 1h timeframe
- [x] 1h EMA(50) trend merged into 5m dataframe
- [x] ADX(14) on 1h — gate entries when ADX > 25 (trending market)
- [x] Entry gate: require 1h EMA uptrend + ADX > 25

### ~~Step 3 — Candlestick Patterns~~ DONE

- [x] Top 15 TA-Lib patterns as weighted confidence factor
- [x] Hyperoptable pattern weight parameter

### ~~Step 4 — Risk Management~~ DONE

- [x] ATR-based position sizing via `custom_stake_amount()`
- [x] Max daily loss tracking — stop entering after X% daily drawdown
- [x] Drawdown kill switch — block new entries + Telegram alert at threshold

### ~~Step 5 — Infrastructure + Validation~~ DONE

- [x] Walk-forward validation script (`scripts/ft-walkforward.py`)
- [x] Walk-forward bash wrapper (`scripts/ft-walkforward.sh`)
- [x] Health check script with optional Telegram alerts (`scripts/ft-health.sh`)
- [x] Systemd user service for dry-run (`~/.config/systemd/user/ft-dry-run.service`)
- [x] herc2 setup — Windows-native scripts, all tests pass, dry-run working

### Step 6 — Re-Hyperopt + Final Validation — BLOCKED

- [x] Re-hyperopt 500 epochs on herc2 (14 workers, 11 minutes)
- [ ] **FIX:** Optimizer converges to "don't trade" (min_confidence=2.494). Sharpe loss rewards low volatility → minimizes trading. Need to either improve signal quality or use a different loss function with minimum trade count.
- [ ] Walk-forward validate — must show avg profit factor > 1.0
- [ ] Compare before/after: must beat Phase 1 results

### Blocker: Strategy Signal Quality

The core entry signals (RSI + MACD + EMA + volume + candlestick patterns + 1h trend filter) are not individually predictive enough. When hyperoptimized on 2yr data, the optimizer finds "don't trade at all" as the optimal strategy. Options:

1. **Change loss function** — use profit-based loss with minimum trade count constraint
2. **Improve signals** — add more predictive indicators (Bollinger Bands, OBV, Ichimoku, etc.)
3. **Move to ML** — skip to Phase 3, let FreqAI/LightGBM find signal combinations
4. **Simplify** — strip back to fewer, stronger signals instead of many weak ones

---

## Phase 3: ML Integration (FreqAI) — PREP DONE

> Waiting on Phase 2 resolution. All infrastructure is ready.

- [x] FreqAI available in Podman (`stable_freqai` image, 1.73GB)
- [x] LightGBM 4.6.0 + XGBoost 3.1.3 in `stable_freqai` image
- [x] PyTorch available in `stable_freqaitorch` image (8.87GB)
- [x] FinBERT downloaded and tested (`~/models/finbert/`, 836MB, CPU, sentiment analysis)
- [x] Trading-Hero-LLM downloaded and tested (`~/models/trading-hero-llm/`, 419MB, CPU, trading signals)
- [x] Models venv at `~/models/.venv` (transformers 5.1.0, torch 2.10.0+cpu)
- [ ] FreqAI strategy with LightGBM (use Phase 2 indicators as features)
- [ ] A/B test: must beat Phase 2 or don't ship
- [ ] Integrate FinBERT sentiment as FreqAI feature (if crypto news source found)

**Note:** CatBoost is NOT in any Freqtrade image. Use LightGBM (default FreqAI backend).

---

## Phase 4: Claude as Trading Copilot — COMPLETE

> Claude Code is operational as Layer 2 decision-maker. Two-layer architecture live since 2026-02-11.

- [x] Freqtrade-MCP installed (`~/mcp-servers/freqtrade-mcp/`)
- [x] CCXT MCP installed (`~/mcp-servers/mcp-server-ccxt/`)
- [x] CoinGecko MCP configured (remote HTTP, no API key)
- [x] `.mcp.json` in project root (gitignored), `.mcp.example.json` as template
- [x] Two-layer architecture: Python fast loop (Layer 1) + Claude Code agent (Layer 2)
- [x] 25 signal models (11 core + 14 enhanced composite, ~85 sub-indicators)
- [x] Dual portfolio strategy: Patient (conservative) + Bold (aggressive breakout)
- [x] Trigger-based invocation (consensus, sustained flips, price moves, F&G, sentiment, cooldowns)
- [x] Telegram notifications with signal grid, timeframe heatmap, reasoning
- [x] Signal accuracy tracking with outcome backfilling (1d/3d/5d/10d horizons)
- [x] Weighted consensus (accuracy + regime + activation frequency)
- [x] 31 Tier 1 tickers (2 crypto, 2 metals, 27 US equities)
- [x] Tier 2 (Avanza Nordic stocks) + Tier 3 (warrants with underlying signals)
- [x] Flask dashboard (port 5055) with signal heatmap, equity curve, accuracy views
- [x] Layer 2 journal system (theses, reflections, watchlist, price tracking)
- [ ] LunarCrush MCP (needs API key from lunarcrush.com) — deprioritized

Full plan: `docs/plans/2026-02-09-llm-trading-research.md`

---

## Phase 5: Expansion (after 1+ month profitable)

- [ ] Short positions
- [ ] Pair expansion (SOL, XRP — one at a time)
- [ ] Classic chart patterns (if needed)
- [ ] Daily automated reports

---

## Phase 6: System Hardening — IN PROGRESS

> Improve reliability, signal quality, and operational resilience.

### Signal Quality

- [x] **Fix custom_lora bias:** Fully disabled — model no longer invoked, removed from votes/core. Saves ~60s GPU per crypto check. (Feb 20)
- [x] **Fix calendar signal bias:** Added quorum (min 2 active votes in winning direction). Re-enabled as voting signal. (Feb 20)
- [x] **Fix macro_regime dead signal:** Adaptive SMA (50-200), FOMC ≤2d=SELL, DXY ±0.3%, 10Y change direction ±1.5%, golden/death cross 20/50 fallback. Now activates. (Feb 20)
- [ ] **Improve consensus accuracy:** Currently 43.7% — worse than a coin flip. Three broken signals now fixed; monitor if accuracy improves over next week
- [ ] **Extend ML/Ministral to stocks:** Currently crypto-only (signals 7, 8, 9). Stocks have only 21 of 24 signals

### Infrastructure

- [x] Add log rotation utility (prevent unbounded growth of JSONL files)
- [x] **Integrate HTTP retry:** `http_retry.py` existed but was never wired in. Now integrated into all API calls (Binance, Alpaca, FX, macro). (Feb 21)
- [x] **Add health monitoring:** New `portfolio/health.py` — heartbeat tracking, error counts, staleness detection. (Feb 21)
- [ ] Migrate signal_log.jsonl to SQLite (JSONL files grow unbounded, slow to query for accuracy)
- [ ] Add /api/health endpoint to dashboard (use health.py data)

### Code Quality

- [x] **Add digest tests:** `test_digest.py` was a standalone script with 0 tests. Rewritten as proper pytest module (13 tests). (Feb 21)
- [x] **Add health tests:** New `test_health.py` for health monitoring module. (Feb 21)
- [x] **Update README.md:** Was outdated (Freqtrade references). Now reflects two-layer architecture. (Feb 21)
- [x] **Write system design doc:** New `docs/system-design.md` — comprehensive engineering reference. (Feb 21)
- [ ] **Modularize main.py:** Currently 2,124 lines. Extract signal_engine.py, data_collector.py
- [ ] Add test_http_retry.py for retry integration
- [ ] Add dashboard tests (currently 0)
- [ ] Add performance/stress tests for 7TF x 31 ticker scale

### Dashboard Improvements

- [ ] Real-time WebSocket updates (currently requires manual refresh or polling)
- [ ] Signal accuracy charts (per-signal hit rate over time)
- [ ] Trade annotation on equity curve (mark BUY/SELL points)
- [ ] Mobile-responsive layout improvements

### Operational

- [ ] Telegram alert on Layer 1 crash (currently silent — only discoverable via loop_out.txt)
- [ ] Automated Layer 2 health monitoring (detect "silent agent" — no invocation for >2 hours during market hours)
- [ ] Structured error logging (replace print() with proper logging framework)

---

## Watch List

| Project       | Stars | What                                      | Revisit When                        |
| ------------- | ----- | ----------------------------------------- | ----------------------------------- |
| TradingAgents | 29.6k | Multi-agent LLM trading (simulation only) | When it supports live exchanges     |
| FinGPT        | 18.5k | Fine-tune financial LLM for $300          | When we need custom sentiment model |
| FinRL         | 13.9k | RL trading framework                      | When we outgrow Freqtrade           |
| MAHORAGA      | 480   | Deployed Claude trading bot (Alpaca)      | When we want US stock trading       |
