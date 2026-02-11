# Portfolio Intelligence System — Architecture (Source of Truth)

> **Last updated:** 2026-02-11
> **Status:** LIVE on herc2 (simulated 500K SEK portfolio)
> **Canonical doc:** This file is THE source of truth for the system architecture.
> Other agents, sessions, and memory files should reference this document.

## Core Principle

**The Python fast loop collects data. Claude Code makes all decisions.**

The fast loop (Layer 1) runs every 60 seconds, fetching prices, computing indicators, and
running all 7 signal models. It detects when something meaningful changes. When a trigger
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
│  5. Run Ministral-8B + CryptoTrader-LM LoRA (cached 15min)      │
│  6. Tally votes → BUY/SELL/HOLD per symbol                      │
│  7. Save everything to data/agent_summary.json                   │
│                                                                   │
│  CHANGE DETECTION (trigger.py):                                   │
│  • Signal flip (HOLD→BUY, BUY→SELL, etc.)                       │
│  • Price moved >2% since last trigger                            │
│  • Fear & Greed crossed threshold (20 or 80)                     │
│  • Sentiment reversal (positive↔negative)                        │
│  • Cooldown expired (2h max silence)                             │
│                                                                   │
│  If trigger fires → invoke Claude Code (Layer 2)                 │
│  If no trigger → log locally, keep looping                       │
│                                                                   │
│  ⚠ NEVER trades. NEVER sends Telegram. Data collection only.    │
└──────────────┬───────────────────────────────────────────────────┘
               │ (only when something meaningful changes)
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 2: CLAUDE CODE AGENT (invoked ~20-50x/day)               │
│                                                                   │
│  Reads:                                                           │
│  • data/agent_summary.json (all signals, timeframes, indicators) │
│  • data/portfolio_state.json (cash, holdings, transaction history)│
│  • Trigger reasons (why was this invocation triggered?)          │
│                                                                   │
│  Analyzes:                                                        │
│  • All 7 signals across all timeframes                           │
│  • Portfolio risk (concentration, drawdown, cash reserves)       │
│  • Recent trade history (avoid whipsaw, respect patterns)        │
│  • Market regime (trending vs ranging, volatility)               │
│                                                                   │
│  Decides:                                                         │
│  • TRADE: Edit portfolio_state.json (update cash, holdings, log) │
│  • NOTIFY: Send Telegram with reasoning                          │
│  • HOLD: Do nothing (most common — discipline over action)       │
│                                                                   │
│  ⚠ Claude Code is the SOLE decision-maker for trades + Telegram │
└──────────────────────────────────────────────────────────────────┘
```

## 7 Signals

| #   | Signal          | Source                            | Buy condition                   | Sell condition                   | Cache TTL |
| --- | --------------- | --------------------------------- | ------------------------------- | -------------------------------- | --------- |
| 1   | RSI(14)         | Binance 15m klines                | < 30 (oversold)                 | > 70 (overbought)                | 0         |
| 2   | MACD(12,26,9)   | Binance 15m klines                | Histogram crossover (neg→pos)   | Histogram crossover (pos→neg)    | 0         |
| 3   | EMA(9,21)       | Binance 15m klines                | Fast > slow (uptrend)           | Fast < slow (downtrend)          | 0         |
| 4   | BB(20,2)        | Binance 15m klines                | Price below lower band          | Price above upper band           | 0         |
| 5   | Fear & Greed    | Crypto→Alternative.me, Stocks→VIX | ≤ 20 (extreme fear, contrarian) | ≥ 80 (extreme greed, contrarian) | 5min      |
| 6   | Sentiment       | Crypto→CryptoBERT, Stocks→TH-LLM  | Positive (confidence > 0.4)     | Negative (confidence > 0.4)      | 15min     |
| 7   | CryptoTrader-LM | Ministral-8B + CryptoTrader LoRA  | LLM outputs BUY                 | LLM outputs SELL                 | 15min     |

**Vote threshold:** MIN_VOTERS=3 must cast a vote. Majority wins. Signals that don't meet their threshold (e.g., RSI between 30-70) abstain.

## 7 Timeframes (crypto instruments)

| Horizon | Candle interval | Candles fetched | Cache TTL       | Signal set       |
| ------- | --------------- | --------------- | --------------- | ---------------- |
| Now     | 15m             | 100 (~25h)      | 0 (every cycle) | All 7 signals    |
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

## Trading Rules (enforced by Claude Code, Layer 2)

- BUY: Allocate 20% of cash per trade
- SELL: Liquidate 50% of position per trade
- Minimum trade size: 500 SEK
- Per-symbol cooldown: 1 hour between trades on same ticker
- State-change gating: Only trade on signal transitions (HOLD→BUY, not BUY→BUY)
- Claude Code exercises judgment — may override raw signal consensus

## File Layout on herc2

```
Q:\finance-analyzer\
├── portfolio\
│   ├── main.py              # Layer 1: Fast loop — data collection + change detection
│   ├── collect.py           # Aggregates signals into agent_summary.json
│   ├── trigger.py           # Change detection: signal flips, price moves, F&G, sentiment
│   ├── fear_greed.py        # Fear & Greed (crypto→Alternative.me, stocks→VIX)
│   ├── sentiment.py         # Sentiment (crypto→CryptoBERT, stocks→TH-LLM)
│   ├── ministral_signal.py  # Wrapper for Ministral-8B + CryptoTrader-LM LoRA
│   └── __init__.py
├── data\
│   ├── portfolio_state.json # Live portfolio (cash, holdings, transactions) — edited by Claude Code
│   ├── agent_summary.json   # Latest signals snapshot — written by fast loop, read by Claude Code
│   └── trigger_state.json   # Trigger state (last signals, prices, timestamps)
├── config.json              # Telegram token, chat_id, Binance keys, newsapi_key (gitignored)
├── CLAUDE.md                # Instructions for Claude Code when invoked as trading agent
├── docs\
│   ├── architecture-plan.md # THIS FILE — canonical architecture reference
│   └── plans\               # Research docs and future plans
├── scripts\win\
│   ├── pf-loop.bat          # Starts Layer 1 fast loop
│   └── pf-agent.bat         # Collects data + invokes Claude Code (Layer 2)
└── .venv\                   # Python 3.12 venv
```

## Models on herc2

```
Q:\models\
├── ministral-8b-gguf\       # bartowski/Ministral-8B-Instruct-2410 Q4_K_M (4.9GB)
├── cryptotrader-lm\         # agarkovv/CryptoTrader-LM LoRA adapter (7.7MB GGUF)
├── cryptobert\              # ElKulako/cryptobert (~500MB) — crypto sentiment
├── trading-hero-llm\        # fuchenru/Trading-Hero-LLM (419MB) — stock sentiment
├── .venv-llm\               # Separate venv for llama-cpp-python (CUDA)
├── ministral_trader.py      # Ministral-8B inference script
├── cryptobert_infer.py      # CryptoBERT inference script
└── trading_hero_infer.py    # Trading-Hero-LLM inference script
```

## Scheduled Tasks (Windows Task Scheduler)

| Task name       | Script         | Schedule              | Purpose                               |
| --------------- | -------------- | --------------------- | ------------------------------------- |
| PF-DataLoop     | `pf-loop.bat`  | On logon (continuous) | Layer 1: 60s data collection loop     |
| Portfolio-Agent | `pf-agent.bat` | Every 15 min          | Layer 2: Claude Code agent invocation |

## What Needs to Change (Implementation TODO)

### Bug fix: Remove auto-trading from main.py

Currently `main.py` calls `execute_trade()` inside the loop — this is wrong. The fast loop
should ONLY collect data and detect triggers. Trading must happen exclusively through Claude Code.

**Changes needed in `main.py`:**

1. Remove `execute_trade()` calls from `run()`
2. Remove direct `send_telegram()` calls from `run()`
3. When trigger fires: write `agent_summary.json` with trigger reasons
4. Add state-change gating: track previous signal per symbol, only trigger on transitions
5. Add per-symbol cooldown tracking in trigger_state.json

### Smart trigger → Claude Code invocation

When `trigger.py` detects a meaningful change, the fast loop should either:

- **Option A:** Directly invoke `claude -p` from within the Python loop (subprocess)
- **Option B:** Write a trigger flag file, let `pf-agent.bat` (scheduled every 15min) pick it up

Option A is preferred for lower latency. Option B is simpler and already partially implemented.

### Planned: ML signal (scikit-learn HistGradientBoosting)

Signal #8: A trained classifier using 30+ features from 2yr of 1h candles.
Walk-forward validated. New files: `portfolio/ml_trainer.py`, `portfolio/ml_signal.py`.

### Planned: Custom LoRA fine-tuning

Train custom LoRA adapter on BTC/ETH trading decisions using hindsight labeling.
Replace current CryptoTrader-LM LoRA. Plan doc in `docs/plans/`.

## Token Budget (Claude Code usage)

| Scenario             | Invocations/day | Tokens/call | Tokens/day |
| -------------------- | --------------- | ----------- | ---------- |
| Fixed 15min interval | 96              | ~6K         | ~576K      |
| Smart triggers only  | 20-50           | ~6K         | 120-300K   |

## Git & Sync

- **Repo:** `git@github.com:Wojnach/finance-analyzer.git`
- **herc2:** `Q:\finance-analyzer\` (working copy, runs live)
- **Steam Deck:** `~/projects/finance-analyzer/` (development copy)
- Both copies at commit `327646b` as of 2026-02-11
