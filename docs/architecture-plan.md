# Portfolio Intelligence — Architecture

## Overview

Two-layer autonomous trading system. A Python fast loop runs 24/7 collecting
data and detecting meaningful changes. When triggered, Claude Code is invoked
as the decision-maker — it reads all signals, reasons about portfolio state,
and either acts (trade + Telegram) or passes.

## Current State (2026-02-11)

**What exists and works:**

- `main.py --loop 60` — collects data, computes 7 signals, **auto-trades** when 3+ agree
- `trigger.py` — detects signal flips, 2% price moves, F&G thresholds, sentiment reversals
- When triggered: sends Telegram report, but does NOT invoke Claude Code
- `pf-agent.bat` — exists on herc2 with Claude prompt, but is disconnected from the loop
- `pf` CLI — deployed, works from anywhere via SSH

**What needs to change:**

- main.py must STOP auto-trading
- When trigger fires → invoke Claude Code instead of just sending Telegram
- Claude Code reads all data, makes the final BUY/SELL/HOLD decision
- Only Claude Code executes trades and sends Telegram messages

---

## Target Architecture

```
┌──────────────────────────────────────────────────────┐
│  LAYER 1: PYTHON FAST LOOP (every 60s)               │
│  Scheduled task: PF-DataLoop on herc2                 │
│  Script: main.py --loop 60                            │
│                                                       │
│  COLLECTS (free, no LLM):                             │
│  • Binance prices + candles (BTC, ETH)               │
│  • Yahoo Finance (MSTR, PLTR)                        │
│  • RSI, MACD, EMA, BB (just math)                    │
│  • Fear & Greed (Alternative.me / VIX)               │
│  • CryptoBERT / Trading-Hero-LLM sentiment           │
│  • CryptoTrader-LM via Ministral-8B LoRA (GPU)       │
│  • All 7 timeframes per crypto, 5 per stock          │
│  • Writes: agent_summary.json, trigger_state.json     │
│                                                       │
│  DETECTS CHANGES (trigger.py):                        │
│  • Signal flip (BUY↔SELL↔HOLD)                       │
│  • Price moved >2% since last trigger                │
│  • Fear & Greed crossed 20 or 80                     │
│  • Sentiment reversal (positive↔negative)             │
│  • 2h cooldown expired (periodic check-in)           │
│                                                       │
│  If triggered → invoke Claude Code (Layer 2)          │
│  If not → log locally, keep looping                   │
└──────────────┬───────────────────────────────────────┘
               │ (~20-50 triggers/day, not 1,440)
               ▼
┌──────────────────────────────────────────────────────┐
│  LAYER 2: CLAUDE CODE (decision-maker)                │
│  Invoked by: main.py via `claude -p` when triggered   │
│                                                       │
│  READS:                                               │
│  • agent_summary.json (all signals, all timeframes)  │
│  • portfolio_state.json (cash, holdings, trades)     │
│  • trigger_state.json (what triggered and why)       │
│                                                       │
│  DECIDES:                                             │
│  • Analyzes all 7 signals + timeframes               │
│  • Considers portfolio risk, position sizes, P&L     │
│  • Weighs trigger reasons (why was I called?)        │
│  • Makes BUY / SELL / HOLD decision with reasoning   │
│                                                       │
│  ACTS (only if decision is BUY or SELL):              │
│  • Edits portfolio_state.json (cash, holdings, txn)  │
│  • Sends Telegram with reasoning                     │
│  • If HOLD: sends brief Telegram only if noteworthy  │
│                                                       │
│  DOES NOT:                                            │
│  • Fetch any market data (Layer 1 already did)       │
│  • Make API calls to Binance/Yahoo/etc               │
│  • Run any models (signals already computed)          │
└──────────────────────────────────────────────────────┘
```

## 7 Signals (Current)

| #   | Signal          | Source               | Buy           | Sell          |
| --- | --------------- | -------------------- | ------------- | ------------- |
| 1   | RSI(14)         | Technical (candles)  | < 30          | > 70          |
| 2   | MACD(12,26,9)   | Technical (candles)  | Histogram > 0 | Histogram < 0 |
| 3   | EMA(9,21)       | Technical (candles)  | EMA9 > EMA21  | EMA9 < EMA21  |
| 4   | Bollinger Bands | Technical (candles)  | Price < lower | Price > upper |
| 5   | Fear & Greed    | Alternative.me / VIX | ≤ 20          | ≥ 80          |
| 6   | News Sentiment  | CryptoBERT / T-H-LLM | Positive >0.4 | Negative >0.4 |
| 7   | CryptoTrader-LM | Ministral-8B + LoRA  | BUY output    | SELL output   |

Layer 1 computes all 7 and writes the consensus to agent_summary.json.
Layer 2 (Claude Code) reads the consensus AND individual signals, then
makes the final decision. Claude is not bound by the consensus — it can
override if it has good reason (e.g., extreme F&G contradicts technicals).

## Timeframes

| Horizon | Crypto interval | Stock interval |
| ------- | --------------- | -------------- |
| Now     | 15m             | 1d             |
| 12h     | 1h              | —              |
| 2d      | 4h              | —              |
| 7d      | 1d              | 1d             |
| 1mo     | 3d              | 1d             |
| 3mo     | 1w              | 1w             |
| 6mo     | 1M              | 1M             |

## Trigger Conditions

Claude Code gets invoked when ANY of these fire:

1. **Signal flip** — any ticker's action changed (HOLD→BUY, BUY→SELL, etc.)
2. **Price move >2%** — since last trigger
3. **Fear & Greed threshold** — crossed 20 (extreme fear) or 80 (extreme greed)
4. **Sentiment reversal** — flipped positive↔negative
5. **2h cooldown** — no trigger for 2 hours, periodic check-in

## Trading Rules

- Simulated 500,000 SEK portfolio
- Instruments: BTC-USD, ETH-USD, MSTR, PLTR
- Position sizing: max 20% of cash per buy, sell 50% of position per sell
- Min trade: 500 SEK
- Min voters to act: 3 signals agreeing
- Manual trades via `pf buy/sell` bypass Claude (direct to portfolio_state.json)

## Key Files on herc2 (Q:\finance-analyzer)

| File                            | Purpose                                                |
| ------------------------------- | ------------------------------------------------------ |
| `portfolio/main.py`             | Core loop: data collection, signals, trigger check     |
| `portfolio/trigger.py`          | Change detection (signal flips, price, F&G, sentiment) |
| `portfolio/collect.py`          | Writes agent_summary.json                              |
| `portfolio/fear_greed.py`       | Fear & Greed (crypto→Alternative.me, stocks→VIX)       |
| `portfolio/sentiment.py`        | News sentiment (CryptoBERT, Trading-Hero-LLM)          |
| `portfolio/ministral_signal.py` | CryptoTrader-LM signal via Ministral-8B LoRA           |
| `data/portfolio_state.json`     | Portfolio state (cash, holdings, transactions)         |
| `data/agent_summary.json`       | Latest signals, timeframes, F&G, prices                |
| `data/trigger_state.json`       | Last trigger state for change detection                |
| `config.json`                   | Telegram bot token + chat_id                           |
| `scripts/pf.py`                 | CLI tool for mobile SSH                                |
| `scripts/win/pf.bat`            | Windows batch wrapper for pf                           |
| `scripts/win/pf-loop.bat`       | Runs main.py --loop 60                                 |
| `scripts/win/pf-agent.bat`      | Claude Code agent invocation (to be wired in)          |

## Implementation TODO

1. **Stop auto-trading in main.py** — remove execute_trade() calls from the loop
2. **On trigger: invoke Claude Code** — `claude -p` with structured prompt + data paths
3. **Claude Code prompt** — read agent_summary.json + portfolio_state.json, decide, act
4. **Claude sends Telegram** — with reasoning, not just signal dump
5. **Test end-to-end** — trigger fires → Claude invoked → decision made → trade + Telegram

## iPhone SSH Pipeline (DONE)

`pf` CLI deployed on herc2, in PATH. Commands:

- `pf status` / `pf signals` / `pf trades` / `pf log` — read-only
- `pf report` — force Telegram report
- `pf buy <ticker> [pct]` / `pf sell <ticker> [pct]` — manual trades
- `pf pause` / `pf resume` — toggle PF-DataLoop scheduled task

## Endgame: Automated Exchange Trading

Once the two-layer system proves profitable vs buy-and-hold:

1. Add Binance API trading module (authenticated, config.json keys)
2. Replace portfolio_state.json edits with real API calls
3. Keep simulated portfolio running in parallel for comparison
4. Add safety rails: max position size, daily loss limit, kill switch
