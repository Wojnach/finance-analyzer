# Portfolio Intelligence System — Architecture (Source of Truth)

> **Last updated:** 2026-02-23
> **Status:** LIVE on herc2 (simulated 500K SEK dual portfolio)
> **Canonical doc:** This file is THE source of truth for the system architecture.
> Other agents, sessions, and memory files should reference this document.

## Core Principle

**The Python fast loop collects data. Claude Code makes all decisions.**

The fast loop (Layer 1) runs every 60 seconds, fetching prices, computing indicators, and
running all 27 signal models (11 core + 16 enhanced composite). It detects when something
meaningful changes. When a trigger fires, it invokes Claude Code (Layer 2) with the full
context. Claude Code analyzes the data, decides whether to trade, and sends Telegram
messages. The fast loop NEVER trades or sends Telegram on its own.

## Two-Layer Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 1: PYTHON FAST LOOP (every 60s) — FREE, runs 24/7        │
│                                                                   │
│  1. Fetch Binance/Alpaca prices + candles (7 timeframes)         │
│  2. Compute indicators: RSI, MACD, EMA, BB, ATR                 │
│  3. Fetch Fear & Greed Index (cached 5min)                       │
│  4. Run CryptoBERT / Trading-Hero-LLM sentiment (cached 15min)  │
│  5. Fetch Reddit social sentiment (cached 15min)                 │
│  6. Run Ministral-8B + CryptoTrader-LM LoRA (cached 15min)      │
│  7. Run Ministral-8B + Custom LoRA (cached 15min)                │
│  8. Run ML classifier (HistGradientBoosting, cached 15min)       │
│  9. Fetch Binance funding rate (cached 15min)                    │
│ 10. Check volume confirmation (>1.5x avg)                        │
│ 11. Run 16 enhanced composite signal modules (from raw OHLCV)   │
│ 12. Fetch macro context: DXY, treasury yields, Fed calendar      │
│ 13. Tally votes → BUY/SELL/HOLD per symbol (27 signals)         │
│ 14. Compute weighted consensus (accuracy + regime + activation)  │
│ 15. Save everything to data/agent_summary.json                   │
│                                                                   │
│  CHANGE DETECTION (trigger.py):                                   │
│  • Signal consensus: ticker reaches BUY or SELL consensus        │
│  • Signal flip sustained for 3 consecutive checks (~3 min)       │
│  • Price moved >2% since last trigger                            │
│  • Fear & Greed crossed threshold (20 or 80)                     │
│  • Sentiment reversal (positive↔negative)                        │
│  • Cooldown expired (10min market hours, 2hr off-hours)          │
│  • Post-trade: cooldown resets after BUY/SELL for reassessment   │
│                                                                   │
│  If trigger fires → invoke Claude Code (Layer 2) via subprocess  │
│  If no trigger → log locally, keep looping                       │
│                                                                   │
│  NEVER trades. NEVER sends Telegram. Data collection only.       │
└──────────────┬───────────────────────────────────────────────────┘
               │ (only when something meaningful changes)
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 2: CLAUDE CODE AGENT (invoked on triggers)                │
│                                                                   │
│  Reads:                                                           │
│  • data/layer2_context.md (memory from previous invocations)     │
│  • data/agent_summary.json (all signals, timeframes, indicators) │
│  • data/portfolio_state.json (Patient: cash, holdings, txns)     │
│  • data/portfolio_state_bold.json (Bold: cash, holdings, txns)   │
│  • Trigger reasons (why was this invocation triggered?)          │
│                                                                   │
│  Analyzes:                                                        │
│  • All 27 signals across all timeframes                          │
│  • Macro context (DXY, yields, FOMC proximity)                   │
│  • Portfolio risk (concentration, drawdown, cash reserves)       │
│  • Recent trade history (avoid whipsaw, respect patterns)        │
│  • Market regime (trending vs ranging, volatility)               │
│  • Signal accuracy data (per-signal hit rates)                   │
│  • Cross-asset lead/lag relationships                            │
│                                                                   │
│  Decides (independently for Patient AND Bold):                    │
│  • TRADE: Edit portfolio_state JSON (update cash, holdings, log) │
│  • NOTIFY: Send Telegram with reasoning                          │
│  • HOLD: Send Telegram with analysis (most common)               │
│                                                                   │
│  Claude Code is the SOLE decision-maker for trades + Telegram    │
└──────────────────────────────────────────────────────────────────┘
```

## Dual Strategy Mode

Two independent simulated portfolios, both starting at 500K SEK:

| Strategy    | File                        | Personality                                               | BUY size    | SELL size        | Max positions |
| ----------- | --------------------------- | --------------------------------------------------------- | ----------- | ---------------- | ------------- |
| **Patient** | `portfolio_state.json`      | "The Regime Reader" — conservative, multi-TF confirmation | 15% of cash | 50% of position  | 5             |
| **Bold**    | `portfolio_state_bold.json` | "The Breakout Trend Rider" — aggressive, breakout entries | 30% of cash | 100% of position | 3             |

See `docs/trading-bot-personalities.md` for full personality definitions.

## 27 Signals (11 Core + 16 Enhanced Composite)

### Core Signals (1-11)

| #   | Signal          | Source                                | Buy condition                   | Sell condition                   | Cache TTL |
| --- | --------------- | ------------------------------------- | ------------------------------- | -------------------------------- | --------- |
| 1   | RSI(14)         | Binance/Alpaca 15m klines             | < adaptive threshold (oversold) | > adaptive threshold (overbought)| 0         |
| 2   | MACD(12,26,9)   | Binance/Alpaca 15m klines             | Histogram crossover (neg→pos)   | Histogram crossover (pos→neg)    | 0         |
| 3   | EMA(9,21)       | Binance/Alpaca 15m klines             | Fast > slow + gap >0.5%         | Fast < slow + gap >0.5%         | 0         |
| 4   | BB(20,2)        | Binance/Alpaca 15m klines             | Price below lower band          | Price above upper band           | 0         |
| 5   | Fear & Greed    | Crypto→Alternative.me, Stocks→VIX     | ≤ 20 (extreme fear, contrarian) | ≥ 80 (extreme greed, contrarian) | 5min      |
| 6   | Sentiment       | Crypto→CryptoBERT, Stocks→TH-LLM      | Positive (confidence > 0.4)     | Negative (confidence > 0.4)      | 15min     |
| 7   | CryptoTrader-LM | Ministral-8B + CryptoTrader LoRA      | LLM outputs BUY                 | LLM outputs SELL                 | 15min     |
| 8   | ML Classifier   | HistGradientBoosting on 1h data       | Model predicts BUY              | Model predicts SELL              | 15min     |
| 9   | Funding Rate    | Binance perpetual futures             | < -0.01% (contrarian buy)       | > 0.03% (contrarian sell)        | 15min     |
| 10  | Volume Confirm. | Binance/Alpaca 15m klines             | Spike + price up                | Spike + price down               | 5min      |
| 11  | Custom LoRA     | Ministral-8B + custom fine-tuned LoRA | LLM outputs BUY                 | LLM outputs SELL                 | 15min     |

### Enhanced Composite Signals (12-27)

Each composite module runs 4-8 sub-indicators internally on raw OHLCV data and produces
one BUY/SELL/HOLD vote via majority voting. All modules live in `portfolio/signals/`.
See `docs/enhanced-signals.md` for full documentation of each module.

| #   | Signal            | Module                   | Sub-indicators | Description                                      |
| --- | ----------------- | ------------------------ | -------------- | ------------------------------------------------ |
| 12  | Trend             | `signals/trend.py`       | 7              | Golden/Death Cross, MA Ribbon, Supertrend, PSAR, Ichimoku, ADX |
| 13  | Momentum          | `signals/momentum.py`    | 8              | RSI Divergence, Stochastic, StochRSI, CCI, Williams %R, ROC, PPO, Bull/Bear Power |
| 14  | Volume Flow       | `signals/volume_flow.py` | 6              | OBV, VWAP, A/D Line, CMF, MFI, Volume RSI       |
| 15  | Volatility        | `signals/volatility.py`  | 6              | BB Squeeze, BB Breakout, ATR Expansion, Keltner, Historical Vol, Donchian |
| 16  | Candlestick       | `signals/candlestick.py` | 4              | Hammer/Shooting Star, Engulfing, Doji, Morning/Evening Star |
| 17  | Structure         | `signals/structure.py`   | 4              | High/Low Breakout, Donchian 55, RSI Centerline, MACD Zero-Line |
| 18  | Fibonacci         | `signals/fibonacci.py`   | 5              | Retracement, Golden Pocket, Extensions, Standard Pivots, Camarilla |
| 19  | Smart Money       | `signals/smart_money.py` | 5              | BOS, CHoCH, Fair Value Gaps, Liquidity Sweeps, Supply/Demand |
| 20  | Oscillators       | `signals/oscillators.py` | 8              | Awesome Osc, Aroon, Vortex, Chande, KST, Schaff, TRIX, Coppock |
| 21  | Heikin-Ashi       | `signals/heikin_ashi.py` | 7              | HA Trend/Doji/Color, Hull MA, Alligator, Elder Impulse, TTM Squeeze |
| 22  | Mean Reversion    | `signals/mean_reversion.py` | 7           | RSI(2), RSI(3), IBS, Consecutive Days, Gap Fade, BB %B, IBS+RSI Combined |
| 23  | Calendar          | `signals/calendar_seasonal.py` | 8       | Day-of-Week, Turnaround Tuesday, Month-End, Sell in May, January Effect, Pre-Holiday, FOMC Drift, Santa Rally |
| 24  | Macro Regime      | `signals/macro_regime.py`| 6              | 200-SMA Filter, DXY vs Risk, Yield Curve, 10Y Momentum, FOMC Proximity, Golden/Death Cross |
| 25  | Momentum Factors  | `signals/momentum_factors.py` | 7        | Time-Series Momentum, ROC-20, 52-Week High/Low, Consecutive Bars, Acceleration, Vol-Weighted |
| 26  | News Event        | `signals/news_event.py`  | 5              | Headline velocity, keyword severity, sentiment shift, source credibility, sector impact |
| 27  | Econ Calendar     | `signals/econ_calendar.py`| 5             | Event proximity risk-off, event type classification, pre-event binary, sector exposure, FOMC/CPI/NFP |

**Total sub-indicators across all 16 enhanced modules: ~95**

**Non-voting context** (macro section in agent_summary.json):

- DXY — Dollar Index trend and 5d change
- Treasury Yields — 2Y, 10Y, 30Y + 2s10s spread
- Fed Calendar — Next FOMC date and days until

**Vote threshold:** MIN_VOTERS=3 for all asset classes (crypto, stocks, and metals). Crypto has 27 applicable signals (11 core + 16 enhanced), stocks and metals have 23 (7 core + 16 enhanced). CryptoTrader-LM, Custom LoRA, ML Classifier, and Funding Rate are crypto-only. Confidence is calculated using active voters (BUY + SELL) as the denominator, not total applicable signals. A majority (>=50% of active voters) is needed for BUY/SELL consensus.

**Weighted consensus:** In addition to raw vote counting, Layer 1 computes a weighted
consensus using per-signal accuracy data, market regime adjustments, and activation
frequency normalization. Rare, balanced, accurate signals get more weight; noisy, biased
signals get less.

**Sentiment hysteresis:** When sentiment flips direction (e.g., positive→negative), the confidence threshold increases from 0.40 to 0.55. This prevents rapid oscillation from models returning ~50% confidence, which previously caused 46 false triggers in 97 minutes for MSTR.

## 7 Timeframes (crypto/metals instruments)

| Horizon | Candle interval | Candles fetched | Cache TTL       | Signal set                     |
| ------- | --------------- | --------------- | --------------- | ------------------------------ |
| Now     | 15m             | 100 (~25h)      | 0 (every cycle) | All 27 signals (11+16 enhanced)|
| 12h     | 1h              | 100 (~4d)       | 5min            | 4 technical only               |
| 2d      | 4h              | 100 (~17d)      | 15min           | 4 technical only               |
| 7d      | 1d              | 100 (~100d)     | 1hr             | 4 technical only               |
| 1mo     | 3d              | 100 (~300d)     | 4hr             | 4 technical only               |
| 3mo     | 1w              | 100 (~2yr)      | 12hr            | 4 technical only               |
| 6mo     | 1M              | 48 (~4yr)       | 24hr            | 4 technical only               |

## 7 Timeframes (stock instruments — Alpaca IEX feed)

| Horizon | Candle interval | Candles fetched | Cache TTL       | Signal set                     |
| ------- | --------------- | --------------- | --------------- | ------------------------------ |
| Now     | 15m             | 100 (~25h)      | 0 (every cycle) | All 23 signals (7+16 enhanced) |
| 12h     | 1h              | 100 (~4d)       | 5min            | 4 technical only               |
| 2d      | 1h              | 48 (~2d)        | 15min           | 4 technical only               |
| 7d      | 1d              | 30 (~30d)       | 1hr             | 4 technical only               |
| 1mo     | 1d              | 100 (~100d)     | 1hr             | 4 technical only               |
| 3mo     | 1w              | 100 (~2yr)      | 12hr            | 4 technical only               |
| 6mo     | 1M              | 48 (~4yr)       | 24hr            | 4 technical only               |

## Instruments (31 Tier 1 tickers)

### Crypto (2) — 24/7, Binance spot

| Ticker  | Name     | Data source       | Signals |
| ------- | -------- | ----------------- | ------- |
| BTC-USD | Bitcoin  | Binance (BTCUSDT) | 27 (11 core + 16 enhanced) |
| ETH-USD | Ethereum | Binance (ETHUSDT) | 27 (11 core + 16 enhanced) |

### Metals (2) — 24/7, Binance futures

| Ticker  | Name   | Data source        | Signals |
| ------- | ------ | ------------------ | ------- |
| XAU-USD | Gold   | Binance FAPI (XAUUSDT) | 23 (7 core + 16 enhanced) |
| XAG-USD | Silver | Binance FAPI (XAGUSDT) | 23 (7 core + 16 enhanced) |

### US Equities (27) — Market hours, Alpaca IEX

| Ticker | Name            | Exchange | Signals |
| ------ | --------------- | -------- | ------- |
| MSTR   | MicroStrategy   | NASDAQ   | 23      |
| PLTR   | Palantir        | NASDAQ   | 23      |
| NVDA   | NVIDIA          | NASDAQ   | 23      |
| AMD    | AMD             | NASDAQ   | 23      |
| BABA   | Alibaba         | NYSE     | 23      |
| GOOGL  | Alphabet        | NASDAQ   | 23      |
| AMZN   | Amazon          | NASDAQ   | 23      |
| AAPL   | Apple           | NASDAQ   | 23      |
| AVGO   | Broadcom        | NASDAQ   | 23      |
| GRRR   | Gorilla Tech    | NASDAQ   | 23      |
| IONQ   | IonQ            | NYSE     | 23      |
| MRVL   | Marvell         | NASDAQ   | 23      |
| META   | Meta            | NASDAQ   | 23      |
| MU     | Micron          | NASDAQ   | 23      |
| PONY   | Pony AI         | NASDAQ   | 23      |
| RXRX   | Recursion       | NASDAQ   | 23      |
| SOUN   | SoundHound      | NASDAQ   | 23      |
| SMCI   | Super Micro     | NASDAQ   | 23      |
| TSM    | TSMC            | NYSE     | 23      |
| TTWO   | Take-Two        | NASDAQ   | 23      |
| TEM    | Tempus AI       | NASDAQ   | 23      |
| UPST   | Upstart         | NASDAQ   | 23      |
| VERI   | Veritone        | NASDAQ   | 23      |
| VRT    | Vertiv          | NYSE     | 23      |
| QQQ    | QQQ ETF         | NASDAQ   | 23      |
| LMT    | Lockheed Martin | NYSE     | 23      |

### Tier 2: Avanza Price-Only (Nordic stocks, no signals)

| Name             | Config key | Notes                           |
| ---------------- | ---------- | ------------------------------- |
| SAAB B           | SAAB-B     | Price + P&L only via Avanza API |
| SEB C            | SEB-C      | Price + P&L only via Avanza API |
| K33              | K33        | Price + P&L only via Avanza API |
| H100 Group       | H100       | Price + P&L only via Avanza API |
| B Treasury Cap B | BTCAP-B    | Price + P&L only via Avanza API |

### Tier 3: Warrants (Avanza price + underlying's signals)

| Name                   | Config key   | Underlying |
| ---------------------- | ------------ | ---------- |
| BULL NASDAQ X3 AVA 1   | BULL-NDX3X   | QQQ        |
| CoinShares XBT Tracker | XBT-TRACKER  | BTC-USD    |
| CoinShares ETH Tracker | ETH-TRACKER  | ETH-USD    |
| MINI L SILVER AVA 140  | MINI-SILVER  | XAG-USD    |
| MINI L TSMC AVA 19     | MINI-TSMC    | TSM        |

## Trading Rules (enforced by Claude Code, Layer 2)

Rules are **guiding principles, not mechanical constraints.** Layer 2 may deviate with reasoning.

- **Bold:** BUY 30% of cash, SELL 100% of position (full exit), prefer max 3 positions
- **Patient:** BUY 15% of cash, SELL 50% of position (partial exit), prefer max 5 positions
- Minimum trade size: 500 SEK
- Trading fees: 0.05% crypto, 0.10% stocks (deducted per trade, tracked in `total_fees_sek`)
- No mechanical cooldowns or state-change gating — Layer 2 decides freely
- Claude Code exercises judgment — may override raw signal consensus

## File Layout on herc2

```
Q:\finance-analyzer\
├── portfolio\
│   ├── main.py              # Layer 1: Fast loop — data collection + change detection
│   ├── trigger.py           # Change detection: consensus, sustained flips, price moves, F&G, sentiment
│   ├── fear_greed.py        # Signal #5: Fear & Greed (crypto→Alternative.me, stocks→VIX)
│   ├── sentiment.py         # Signal #6: Sentiment (crypto→CryptoBERT, stocks→TH-LLM)
│   ├── social_sentiment.py  # Reddit social sentiment (merged into signal #6)
│   ├── ministral_signal.py  # Signals #7/#11: Ministral-8B + CryptoTrader-LM / Custom LoRA
│   ├── ml_signal.py         # Signal #8: ML classifier inference (HistGradientBoosting)
│   ├── ml_trainer.py        # ML model training (weekly retrain on 1h Binance data)
│   ├── data_refresh.py      # Refresh training data from Binance API
│   ├── funding_rate.py      # Signal #9: Binance perpetual futures funding rate
│   ├── macro_context.py     # Non-voting: DXY, treasury yields, Fed calendar + volume signal
│   ├── outcome_tracker.py   # Backfills price outcomes at 1d/3d/5d/10d horizons
│   ├── accuracy_stats.py    # Per-signal hit rate reporting + activation rate analysis
│   ├── stats.py             # Invocation & Telegram message statistics
│   ├── journal.py           # Layer 2 journal/context management
│   ├── avanza_tracker.py    # Tier 2/3 instrument price tracking via Avanza
│   ├── avanza_client.py     # Avanza API client
│   ├── bigbet.py            # Big Bet detection (extreme setup alerts)
│   ├── iskbets.py           # ISKBETS monitoring system
│   ├── telegram_poller.py   # Telegram command listener
│   ├── fomc_dates.py        # FOMC calendar constants
│   ├── econ_dates.py        # Full economic calendar (FOMC, CPI, NFP) 2026-2027
│   ├── news_keywords.py     # Keyword severity/sector mapping for news signal
│   ├── analyze.py           # Standalone analysis tool (--analyze, --watch)
│   ├── signal_utils.py      # Shared helpers: sma, ema, rsi, true_range, majority_vote
│   ├── signal_registry.py   # Dynamic signal registration and discovery
│   ├── signal_db.py         # SQLite WAL-mode signal log (dual-write with JSONL)
│   ├── file_utils.py        # load_json, load_jsonl, atomic file I/O helpers
│   ├── http_retry.py        # HTTP retry with exponential backoff + jitter
│   ├── api_utils.py         # Config loading, API URL centralization
│   ├── config_validator.py  # Config.json validation at startup
│   ├── health.py            # Heartbeat, error tracking, silence detection
│   ├── shared_state.py      # Thread-safe cache, global state management
│   ├── logging_config.py    # Structured logging with rotating file handler
│   ├── kelly_sizing.py      # Kelly Criterion position sizing
│   ├── regime_alerts.py     # Market regime change alerting
│   ├── risk_management.py   # Portfolio risk metrics and alerts
│   ├── weekly_digest.py     # Weekly performance digest
│   ├── forecast_signal.py   # Prophet + Chronos GPU forecast signal
│   ├── signals\             # Enhanced composite signal modules (16 modules)
│   │   ├── __init__.py
│   │   ├── trend.py         # #12: 7 sub-indicators (Golden Cross, MA Ribbon, Supertrend, etc.)
│   │   ├── momentum.py      # #13: 8 sub-indicators (RSI Divergence, Stochastic, CCI, etc.)
│   │   ├── volume_flow.py   # #14: 6 sub-indicators (OBV, VWAP, A/D Line, CMF, MFI, etc.)
│   │   ├── volatility.py    # #15: 6 sub-indicators (BB Squeeze, ATR Expansion, Keltner, etc.)
│   │   ├── candlestick.py   # #16: 4 sub-indicators (Hammer, Engulfing, Doji, Star)
│   │   ├── structure.py     # #17: 4 sub-indicators (High/Low, Donchian 55, RSI Center, MACD Zero)
│   │   ├── fibonacci.py     # #18: 5 sub-indicators (Retracement, Golden Pocket, Extensions, Pivots)
│   │   ├── smart_money.py   # #19: 5 sub-indicators (BOS, CHoCH, FVG, Liquidity, S/D Zones)
│   │   ├── oscillators.py   # #20: 8 sub-indicators (AO, Aroon, Vortex, Chande, KST, etc.)
│   │   ├── heikin_ashi.py   # #21: 7 sub-indicators (HA Trend/Doji/Color, Hull, Alligator, etc.)
│   │   ├── mean_reversion.py# #22: 7 sub-indicators (RSI(2), RSI(3), IBS, Gap Fade, BB %B, etc.)
│   │   ├── calendar_seasonal.py # #23: 8 sub-indicators (Day-of-Week, FOMC Drift, etc.)
│   │   ├── macro_regime.py  # #24: 6 sub-indicators (200-SMA, DXY, Yield Curve, etc.)
│   │   ├── momentum_factors.py  # #25: 7 sub-indicators (TS Momentum, ROC-20, etc.)
│   │   ├── news_event.py       # #26: 5 sub-indicators (headline velocity, keyword severity, etc.)
│   │   └── econ_calendar.py    # #27: 5 sub-indicators (event proximity, type, sector exposure, etc.)
│   └── __init__.py
├── data\
│   ├── portfolio_state.json      # Patient strategy (cash, holdings, transactions)
│   ├── portfolio_state_bold.json # Bold strategy (cash, holdings, transactions)
│   ├── agent_summary.json        # Latest signals snapshot (written by Layer 1)
│   ├── trigger_state.json        # Trigger state (sustained counts, prices, timestamps)
│   ├── signal_log.jsonl          # Signal history for accuracy tracking
│   ├── layer2_journal.jsonl      # Layer 2 decision journal (theses, reflections)
│   ├── layer2_context.md         # Layer 2 memory (written by journal.py before invocation)
│   ├── telegram_messages.jsonl   # All sent Telegram messages
│   ├── invocations.jsonl         # Layer 2 invocation log (when/why triggered)
│   ├── agent_summary_compact.json # Compact signals snapshot for tiered invocation
│   ├── agent_context_t1.json     # Tier 1 (Quick Check) context
│   ├── agent_context_t2.json     # Tier 2 (Signal Analysis) context
│   ├── agent.log                 # Layer 2 stdout/stderr log
│   └── loop_out.txt              # Layer 1 stdout (for debugging silent failures)
├── training\lora\
│   ├── train_lora.py        # LoRA fine-tuning on labeled 1h candles
│   ├── generate_data.py     # Training dataset generation
│   ├── pipeline.py          # End-to-end training pipeline
│   └── steps.py             # Training steps
├── dashboard\
│   ├── app.py               # Flask API server (port 5055)
│   └── static\index.html    # Web frontend
├── config.json              # API keys: Telegram, Binance, Alpaca (gitignored)
├── CLAUDE.md                # Layer 2 instructions: trading rules, personalities, execution formulas
├── docs\
│   ├── architecture-plan.md # THIS FILE
│   ├── enhanced-signals.md  # Detailed documentation of all 16 enhanced signal modules
│   ├── operational-runbook.md # Operations guide: restart, debug, monitoring
│   ├── dashboard-api.md     # Dashboard API endpoint documentation
│   └── trading-bot-personalities.md  # Bold/Patient personality definitions (v2)
├── scripts\win\
│   ├── pf-loop.bat          # Starts Layer 1 fast loop (auto-restarts on crash)
│   └── pf-agent.bat         # Invokes Claude Code (Layer 2)
└── .venv\                   # Python 3.12 venv
```

## Models on herc2

```
Q:\models\
├── ministral-8b-gguf\       # bartowski/Ministral-8B-Instruct-2410 Q4_K_M (4.9GB)
├── cryptotrader-lm\         # agarkovv/CryptoTrader-LM LoRA adapter (7.7MB GGUF)
├── custom-lora\             # Custom fine-tuned LoRA on labeled 1h candles
├── cryptobert\              # ElKulako/cryptobert (~500MB) — crypto sentiment
├── trading-hero-llm\        # fuchenru/Trading-Hero-LLM (419MB) — stock sentiment
├── .venv-llm\               # Separate venv for llama-cpp-python (CUDA)
├── ministral_trader.py      # Ministral-8B inference script
├── cryptobert_infer.py      # CryptoBERT inference script
└── trading_hero_infer.py    # Trading-Hero-LLM inference script
```

## Scheduled Tasks (Windows Task Scheduler)

| Task name       | Schedule              | Purpose                                           |
| --------------- | --------------------- | ------------------------------------------------- |
| PF-DataLoop     | On logon (continuous) | Layer 1: 60s data collection loop                 |
| PF-OutcomeCheck | Daily 06:00 UTC       | Backfill signal outcomes at 1d/3d/5d/10d horizons |
| PF-MLRetrain    | Weekly Sunday 04:00   | Retrain ML classifier on fresh 1h Binance data    |
| PF-Dashboard    | On logon (continuous) | Flask dashboard at localhost                      |

## Forward Tracking

Every trigger invocation is logged to `data/signal_log.jsonl` with all 27 signal votes and
current prices per ticker. The daily `PF-OutcomeCheck` task backfills what actually happened
at 1d/3d/5d/10d horizons. Use `--accuracy` to see per-signal hit rates.

Signal accuracy data is used for:
- **Weighted consensus:** Accurate signals get more weight in the composite vote
- **Layer 2 calibration:** Claude Code sees per-signal hit rates to judge trustworthiness
- **Bias detection:** Signals that always vote one direction (e.g., calendar=100% BUY) are penalized

## Token Budget (Claude Code usage)

| Scenario            | Invocations/day | Tokens/call | Tokens/day |
| ------------------- | --------------- | ----------- | ---------- |
| Smart triggers only | 20-50           | ~6K         | 120-300K   |

## Git & Sync

- **Repo:** `git@github.com:Wojnach/finance-analyzer.git`
- **herc2:** `Q:\finance-analyzer\` (working copy, runs live)
- **Steam Deck:** `~/projects/finance-analyzer/` (development copy)
