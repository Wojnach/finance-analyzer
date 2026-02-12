# Portfolio Intelligence System — Architecture (Source of Truth)

> **Last updated:** 2026-02-12
> **Status:** LIVE on herc2 (simulated 500K SEK dual portfolio)
> **Canonical doc:** This file is THE source of truth for the system architecture.
> Other agents, sessions, and memory files should reference this document.

## Core Principle

**The Python fast loop collects data. Claude Code makes all decisions.**

The fast loop (Layer 1) runs every 60 seconds, fetching prices, computing indicators, and
running all 11 signal models. It detects when something meaningful changes. When a trigger
fires, it invokes Claude Code (Layer 2) with the full context. Claude Code analyzes the
data, decides whether to trade, and sends Telegram messages. The fast loop NEVER trades
or sends Telegram on its own.

## Two-Layer Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 1: PYTHON FAST LOOP (every 60s) — FREE, runs 24/7        │
│                                                                   │
│  1. Fetch Binance/yfinance prices + candles (7 timeframes)       │
│  2. Compute indicators: RSI, MACD, EMA, BB                      │
│  3. Fetch Fear & Greed Index (cached 5min)                       │
│  4. Run CryptoBERT / Trading-Hero-LLM sentiment (cached 15min)  │
│  5. Fetch Reddit social sentiment (cached 15min)                 │
│  6. Run Ministral-8B + CryptoTrader-LM LoRA (cached 15min)      │
│  7. Run Ministral-8B + Custom LoRA (cached 15min)                │
│  8. Run ML classifier (HistGradientBoosting, cached 15min)       │
│  9. Fetch Binance funding rate (cached 15min)                    │
│ 10. Check volume confirmation (>1.5x avg)                        │
│ 11. Fetch macro context: DXY, treasury yields, Fed calendar      │
│ 12. Tally votes → BUY/SELL/HOLD per symbol                      │
│ 13. Save everything to data/agent_summary.json                   │
│                                                                   │
│  CHANGE DETECTION (trigger.py):                                   │
│  • Signal flip sustained for 3 consecutive checks (~3 min)       │
│  • Price moved >2% since last trigger                            │
│  • Fear & Greed crossed threshold (20 or 80)                     │
│  • Sentiment reversal (positive↔negative)                        │
│  • Cooldown expired (2h max silence, market hours only)          │
│                                                                   │
│  If trigger fires → invoke Claude Code (Layer 2) via subprocess  │
│  If no trigger → log locally, keep looping                       │
│                                                                   │
│  ⚠ NEVER trades. NEVER sends Telegram. Data collection only.    │
└──────────────┬───────────────────────────────────────────────────┘
               │ (only when something meaningful changes)
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 2: CLAUDE CODE AGENT (invoked on triggers)                │
│                                                                   │
│  Reads:                                                           │
│  • data/agent_summary.json (all signals, timeframes, indicators) │
│  • data/portfolio_state.json (Patient: cash, holdings, txns)     │
│  • data/portfolio_state_bold.json (Bold: cash, holdings, txns)   │
│  • Trigger reasons (why was this invocation triggered?)          │
│                                                                   │
│  Analyzes:                                                        │
│  • All 11 signals across all timeframes                          │
│  • Macro context (DXY, yields, FOMC proximity)                   │
│  • Portfolio risk (concentration, drawdown, cash reserves)       │
│  • Recent trade history (avoid whipsaw, respect patterns)        │
│  • Market regime (trending vs ranging, volatility)               │
│                                                                   │
│  Decides (independently for Patient AND Bold):                    │
│  • TRADE: Edit portfolio_state JSON (update cash, holdings, log) │
│  • NOTIFY: Send Telegram with reasoning                          │
│  • HOLD: Send Telegram with analysis (most common)               │
│                                                                   │
│  ⚠ Claude Code is the SOLE decision-maker for trades + Telegram │
└──────────────────────────────────────────────────────────────────┘
```

## Dual Strategy Mode

Two independent simulated portfolios, both starting at 500K SEK:

| Strategy    | File                        | Personality                                               | BUY size    | SELL size        | Max positions |
| ----------- | --------------------------- | --------------------------------------------------------- | ----------- | ---------------- | ------------- |
| **Patient** | `portfolio_state.json`      | "The Regime Reader" — conservative, multi-TF confirmation | 15% of cash | 50% of position  | 5             |
| **Bold**    | `portfolio_state_bold.json` | "The Breakout Trend Rider" — aggressive, breakout entries | 30% of cash | 100% of position | 3             |

See `docs/trading-bot-personalities.md` for full personality definitions.

## 11 Signals

| #   | Signal          | Source                                | Buy condition                   | Sell condition                   | Cache TTL |
| --- | --------------- | ------------------------------------- | ------------------------------- | -------------------------------- | --------- |
| 1   | RSI(14)         | Binance 15m klines                    | < 30 (oversold)                 | > 70 (overbought)                | 0         |
| 2   | MACD(12,26,9)   | Binance 15m klines                    | Histogram crossover (neg→pos)   | Histogram crossover (pos→neg)    | 0         |
| 3   | EMA(9,21)       | Binance 15m klines                    | Fast > slow (uptrend)           | Fast < slow (downtrend)          | 0         |
| 4   | BB(20,2)        | Binance 15m klines                    | Price below lower band          | Price above upper band           | 0         |
| 5   | Fear & Greed    | Crypto→Alternative.me, Stocks→VIX     | ≤ 20 (extreme fear, contrarian) | ≥ 80 (extreme greed, contrarian) | 5min      |
| 6   | Sentiment       | Crypto→CryptoBERT, Stocks→TH-LLM      | Positive (confidence > 0.4)     | Negative (confidence > 0.4)      | 15min     |
| 7   | CryptoTrader-LM | Ministral-8B + CryptoTrader LoRA      | LLM outputs BUY                 | LLM outputs SELL                 | 15min     |
| 8   | ML Classifier   | HistGradientBoosting on 1h data       | Model predicts BUY              | Model predicts SELL              | 15min     |
| 9   | Funding Rate    | Binance perpetual futures             | < -0.01% (contrarian buy)       | > 0.03% (contrarian sell)        | 15min     |
| 10  | Volume Confirm. | Binance 15m klines                    | Spike + price up                | Spike + price down               | 0         |
| 11  | Custom LoRA     | Ministral-8B + custom fine-tuned LoRA | LLM outputs BUY                 | LLM outputs SELL                 | 15min     |

**Non-voting context** (macro section in agent_summary.json):

- DXY — Dollar Index trend and 5d change
- Treasury Yields — 2Y, 10Y, 30Y + 2s10s spread
- Fed Calendar — Next FOMC date and days until

**Vote threshold:** MIN_VOTERS=3 must cast a vote. Confidence is calculated against total applicable signals (11 crypto, 7 stocks), not just active voters. Majority (≥50% of applicable) needed for BUY/SELL consensus. Signals that don't meet their threshold abstain but still count in the denominator. ML Classifier and Funding Rate are crypto-only (BTC, ETH).

## 7 Timeframes (crypto instruments)

| Horizon | Candle interval | Candles fetched | Cache TTL       | Signal set       |
| ------- | --------------- | --------------- | --------------- | ---------------- |
| Now     | 15m             | 100 (~25h)      | 0 (every cycle) | All 11 signals   |
| 12h     | 1h              | 100 (~4d)       | 5min            | 4 technical only |
| 2d      | 4h              | 100 (~17d)      | 15min           | 4 technical only |
| 7d      | 1d              | 100 (~100d)     | 1hr             | 4 technical only |
| 1mo     | 3d              | 100 (~300d)     | 4hr             | 4 technical only |
| 3mo     | 1w              | 100 (~2yr)      | 12hr            | 4 technical only |
| 6mo     | 1M              | 48 (~4yr)       | 24hr            | 4 technical only |

## 5 Timeframes (stock instruments)

| Horizon | Candle interval | Cache TTL |
| ------- | --------------- | --------- |
| Now     | 1d              | 0         |
| 7d      | 1d              | 1hr       |
| 1mo     | 1d              | 1hr       |
| 3mo     | 1w              | 12hr      |
| 6mo     | 1M              | 24hr      |

## Instruments

| Name          | Ticker  | Market      | Data source       |
| ------------- | ------- | ----------- | ----------------- |
| Bitcoin       | BTC-USD | Crypto 24/7 | Binance (BTCUSDT) |
| Ethereum      | ETH-USD | Crypto 24/7 | Binance (ETHUSDT) |
| MicroStrategy | MSTR    | NASDAQ      | yfinance          |
| Palantir      | PLTR    | NASDAQ      | yfinance          |
| NVIDIA        | NVDA    | NASDAQ      | yfinance          |

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
│   ├── collect.py           # Aggregates signals into agent_summary.json
│   ├── trigger.py           # Change detection: sustained signal flips, price moves, F&G, sentiment
│   ├── fear_greed.py        # Signal #5: Fear & Greed (crypto→Alternative.me, stocks→VIX)
│   ├── sentiment.py         # Signal #6: Sentiment (crypto→CryptoBERT, stocks→TH-LLM)
│   ├── social_sentiment.py  # Reddit social sentiment (merged into signal #6)
│   ├── ministral_signal.py  # Signals #7/#11: Ministral-8B + CryptoTrader-LM / Custom LoRA
│   ├── ml_signal.py         # Signal #8: ML classifier inference (HistGradientBoosting)
│   ├── ml_trainer.py        # ML model training (weekly retrain on 1h Binance data)
│   ├── funding_rate.py      # Signal #9: Binance perpetual futures funding rate
│   ├── macro_context.py     # Non-voting: DXY, treasury yields, Fed calendar
│   ├── outcome_tracker.py   # Backfills price outcomes at 1d/3d/5d/10d horizons
│   ├── accuracy_stats.py    # Per-signal hit rate reporting
│   ├── stats.py             # Invocation & Telegram message statistics
│   └── __init__.py
├── data\
│   ├── portfolio_state.json      # Patient strategy (cash, holdings, transactions)
│   ├── portfolio_state_bold.json # Bold strategy (cash, holdings, transactions)
│   ├── agent_summary.json        # Latest signals snapshot (written by Layer 1)
│   ├── trigger_state.json        # Trigger state (sustained counts, prices, timestamps)
│   ├── signal_log.jsonl          # Signal history for accuracy tracking
│   ├── telegram_messages.jsonl   # All sent Telegram messages
│   ├── invocations.jsonl         # Layer 2 invocation log (when/why triggered)
│   └── shadow_lora_config.json   # Custom LoRA A/B test configuration
├── training\lora\
│   ├── train_lora.py        # LoRA fine-tuning on labeled 1h candles
│   ├── generate_data.py     # Training dataset generation
│   ├── pipeline.py          # End-to-end training pipeline
│   └── steps.py             # Training steps
├── dashboard\
│   ├── app.py               # Flask API server
│   └── static\index.html    # Web frontend
├── config.json              # API keys: Telegram, Binance (gitignored)
├── CLAUDE.md                # Layer 2 instructions: trading rules, personalities, execution formulas
├── docs\
│   ├── architecture-plan.md # THIS FILE
│   └── trading-bot-personalities.md  # Bold/Patient personality definitions (v2)
├── scripts\win\
│   ├── pf-loop.bat          # Starts Layer 1 fast loop
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

Every trigger invocation is logged to `data/signal_log.jsonl` with all 11 signal votes and
current prices per ticker. The daily `PF-OutcomeCheck` task backfills what actually happened
at 1d/3d/5d/10d horizons. Use `--accuracy` to see per-signal hit rates.

## Token Budget (Claude Code usage)

| Scenario            | Invocations/day | Tokens/call | Tokens/day |
| ------------------- | --------------- | ----------- | ---------- |
| Smart triggers only | 20-50           | ~6K         | 120-300K   |

## Git & Sync

- **Repo:** `git@github.com:Wojnach/finance-analyzer.git`
- **herc2:** `Q:\finance-analyzer\` (working copy, runs live)
- **Steam Deck:** `~/projects/finance-analyzer/` (development copy)
