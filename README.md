# Portfolio Intelligence — Trading Agent

Two-layer automated trading intelligence system. Layer 1 (Python) collects market data and computes 25 signals every 60 seconds. Layer 2 (Claude Code) makes trading decisions when meaningful changes are detected.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 1: PYTHON FAST LOOP (every 60s)                          │
│                                                                   │
│  Fetch prices → Compute 25 signals → Detect triggers             │
│  Binance (crypto) · Alpaca (stocks) · Avanza (Nordic)            │
│                                                                   │
│  NEVER trades. NEVER sends Telegram. Data collection only.       │
└──────────────┬───────────────────────────────────────────────────┘
               │ (only when something meaningful changes)
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 2: CLAUDE CODE AGENT (invoked on triggers)                │
│                                                                   │
│  Reads all signals + portfolio state → Analyzes → Decides        │
│  Manages two portfolios: Patient (conservative) & Bold (aggro)   │
│  Sends Telegram notifications. Sole authority on trades.         │
└──────────────────────────────────────────────────────────────────┘
```

## Instruments (31 Tier 1 + 5 Tier 2 + 5 Tier 3)

- **Crypto**: BTC-USD, ETH-USD (Binance, 25 signals, 24/7)
- **Metals**: XAU-USD, XAG-USD (Binance FAPI, 21 signals)
- **US Stocks**: 27 tickers via Alpaca IEX (NVDA, AMD, AAPL, GOOGL, META, TSM, etc.)
- **Nordic**: SAAB-B, SEB-C, K33, H100, BTCAP-B (Avanza, price-only)
- **Warrants**: BULL-NDX3X, XBT-TRACKER, etc. (Avanza price + underlying's signals)

## 25 Signals

| # | Signal | Type | Notes |
|---|--------|------|-------|
| 1-4 | RSI, MACD, EMA, BB | Core TA | Classic indicators |
| 5 | Fear & Greed | Core | alternative.me (crypto), VIX (stocks) |
| 6 | Sentiment | Core | CryptoBERT / Trading-Hero-LLM |
| 7 | ML Classifier | Core | HistGradientBoosting, crypto only |
| 8 | Funding Rate | Core | Binance perps, crypto only |
| 9 | Volume | Core | Spike >1.5x + direction |
| 10 | Ministral-8B | Core | LLM reasoning, crypto only |
| 11 | Custom LoRA | Core | **DISABLED** (20.9% accuracy) |
| 12-25 | Enhanced Composite | 14 modules | Trend, Momentum, Smart Money, etc. |

## Dual Portfolio Strategy

| Strategy | BUY | SELL | Max Positions | Style |
|----------|-----|------|---------------|-------|
| **Patient** | 15% of cash | 50% of position | 5 | Conservative, multi-TF alignment |
| **Bold** | 30% of cash | 100% of position | 3 | Aggressive breakout, conviction sizing |

Both start at 500K SEK (simulated).

## Setup

### Prerequisites

- Python 3.10+ with venv
- Windows 11 (Task Scheduler for automation)
- Binance API keys (read-only)
- Alpaca API keys (IEX feed)
- Telegram bot token
- Avanza credentials (optional, Nordic stocks)
- GPU recommended for Ministral-8B inference

### Installation

```bash
git clone https://github.com/Wojnach/finance-analyzer.git
cd finance-analyzer

python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt

cp config.example.json config.json
# Edit config.json with your API keys
```

### Running

```bash
# Start Layer 1 loop (continuous)
.venv\Scripts\python.exe -u portfolio\main.py --loop

# Or via batch script (auto-restart on crash)
scripts\win\pf-loop.bat

# One-shot signal report
.venv\Scripts\python.exe -u portfolio\main.py --report

# Backfill signal accuracy
.venv\Scripts\python.exe -u portfolio\main.py --check-outcomes

# Print accuracy stats
.venv\Scripts\python.exe -u portfolio\main.py --accuracy

# Run dashboard (Flask, port 5055)
.venv\Scripts\python.exe -u dashboard\app.py

# Run tests
.venv\Scripts\pytest.exe tests/ -v
```

## Project Structure

```
finance-analyzer/
├── CLAUDE.md                  # Layer 2 trading rules (source of truth)
├── portfolio/                 # Layer 1 core code
│   ├── main.py                # Orchestrator (data + signals + triggers)
│   ├── trigger.py             # Change detection (7 trigger types)
│   ├── signals/               # 14 enhanced composite signal modules
│   ├── accuracy_stats.py      # Signal accuracy tracking + weighting
│   ├── health.py              # Health monitoring
│   ├── http_retry.py          # HTTP retry with exponential backoff
│   ├── weekly_digest.py       # Periodic performance summaries
│   └── journal.py             # Layer 2 memory/context management
├── dashboard/                 # Flask web UI (port 5055)
├── scripts/win/               # Windows batch scripts
│   ├── pf-loop.bat            # Layer 1 launcher (auto-restart)
│   └── pf-agent.bat           # Layer 2 invocation (claude -p)
├── tests/                     # pytest test suite (~720 tests)
├── docs/                      # Architecture + design documents
│   ├── architecture-plan.md   # Canonical architecture reference
│   ├── system-design.md       # Engineering design document
│   └── operational-runbook.md # Ops procedures
├── data/                      # Runtime state (gitignored)
└── training/                  # LoRA fine-tuning pipeline
```

## Documentation

- **[Architecture Plan](docs/architecture-plan.md)** — Canonical system architecture
- **[System Design](docs/system-design.md)** — Engineering reference
- **[CLAUDE.md](CLAUDE.md)** — Layer 2 trading rules and instructions
- **[Operational Runbook](docs/operational-runbook.md)** — Ops procedures
- **[TODO](TODO.md)** — Active roadmap

## License

Private — All rights reserved
