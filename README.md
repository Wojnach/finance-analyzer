# Portfolio Intelligence — Trading Agent

> Facts last reconciled to code 2026-06-11. For the authoritative,
> continuously-maintained system description see `CLAUDE.md` and
> `docs/SYSTEM_OVERVIEW.md`.

Two-layer automated trading intelligence system. Layer 1 (Python) collects market data and computes 15 active signals every 600 seconds. Layer 2 (Claude Code) makes trading decisions when meaningful changes are detected.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 1: PYTHON LOOP (every 600s)                              │
│                                                                   │
│  Fetch prices → Compute 15 active signals → Detect triggers      │
│  Binance (crypto) · Alpaca (stocks) · Avanza (Nordic)            │
│                                                                   │
│  NEVER trades. Sends error alerts + 4h/morning digests only.     │
└──────────────┬───────────────────────────────────────────────────┘
               │ (only when something meaningful changes)
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 2: CLAUDE CODE AGENT (invoked on triggers)                │
│                                                                   │
│  Reads all signals + portfolio state → Analyzes → Decides        │
│  Manages two portfolios: Patient (conservative) & Bold (aggro)   │
│  Sole authority on TRADE notifications. Layer 1 sends ops alerts. │
└──────────────────────────────────────────────────────────────────┘
```

## Instruments (5 Tier 1 + 3 Tier 2 + 3 Tier 3)

- **Crypto**: BTC-USD, ETH-USD (Binance spot, 24/7)
- **Metals**: XAU-USD, XAG-USD (Binance FAPI, 24/7)
- **US Stocks**: MSTR (Alpaca)
- **Nordic**: SAAB-B, SEB-C, INVE-B (Avanza, price-only)
- **Warrants**: XBT-TRACKER (→BTC), ETH-TRACKER (→ETH), MINI-SILVER (→XAG 5x) (Avanza price + underlying's signals)

(All US stocks except MSTR removed Mar 15 / Apr 09 2026; MINI-TSMC retired with TSM.)

## Signals (15 active of 89 tracked; reconciled 2026-06-11)

89 signal names are tracked, 76 are force-HOLD via `DISABLED_SIGNALS`, leaving
15 active globally (plus 2 per-ticker overrides). The active set as of
2026-06-11: RSI, BB, Fear & Greed, Ministral-8B, Qwen3-8B, Momentum, Mean
Reversion, News Event, Econ Calendar, Crypto Macro, COT Positioning, On-Chain
BTC, Statistical Jump Regime, Drift Regime Gate, Amihud Illiquidity Regime.

The active roster changes as accuracy gates promote/disable signals — see the
**Signal System** section of `CLAUDE.md` for the live list and derive it with:

```bash
python -c "from portfolio.tickers import SIGNAL_NAMES, DISABLED_SIGNALS; print([s for s in SIGNAL_NAMES if s not in DISABLED_SIGNALS])"
```

Forecast (Chronos) was fully disabled 2026-05-12 and Kronos retired 2026-04-21.

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
│   ├── signals/               # 79 enhanced signal modules (70 registered)
│   ├── accuracy_stats.py      # Signal accuracy tracking + weighting
│   ├── health.py              # Health monitoring
│   ├── http_retry.py          # HTTP retry with exponential backoff
│   ├── weekly_digest.py       # Periodic performance summaries
│   └── journal.py             # Layer 2 memory/context management
├── dashboard/                 # Flask web UI (port 5055)
├── scripts/win/               # Windows batch scripts
│   ├── pf-loop.bat            # Layer 1 launcher (auto-restart)
│   └── pf-agent.bat           # Layer 2 invocation (claude -p)
├── tests/                     # pytest test suite (~11,100 tests, ~446 files; 2026-06-11)
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
