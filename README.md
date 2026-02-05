# Finance Analyzer

Real-time financial technical analysis system with ML-powered pattern detection and automated trading.

## Features

- **Real-time analysis** of crypto (Binance) and stocks (Avanza)
- **Classic TA** pattern recognition (head & shoulders, double tops, triangles, etc.)
- **ML/AI** anomaly detection and price prediction via FreqAI
- **Sentiment analysis** from news and social media
- **Automated trading** for crypto (paper trading first, then live)
- **Alerts** via Telegram and desktop notifications
- **Web dashboard** for monitoring and control
- **Feedback loop** for continuous improvement

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              FreqUI + Custom Dashboard                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 Freqtrade + FreqAI                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ Classic TA  │ │  FreqAI ML  │ │ Sentiment Plugin    │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
└──────────┬──────────────────────────────────┬──────────────┘
           │                                  │
┌──────────▼──────────┐          ┌───────────▼─────────────┐
│ Binance (Crypto)    │          │ Avanza (Stocks)         │
│ Auto-trade          │          │ Alerts only             │
└─────────────────────┘          └─────────────────────────┘
```

## Assets Monitored

- **Crypto**: BTC, ETH (via Binance)
- **Stocks**: MSTR, PLTR (via Avanza - alerts only, manual execution)

## Setup

### Prerequisites

- Python 3.10+
- Docker (optional, for isolated deployment)
- Binance API keys
- Telegram bot token
- Avanza account (for stock alerts)

### Installation

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/finance-analyzer.git
cd finance-analyzer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy config template
cp config.example.json user_data/config.json
# Edit config.json with your API keys
```

### Running

```bash
# Paper trading (dry-run)
freqtrade trade --config user_data/config.json --strategy TABaseStrategy --dry-run

# Backtesting
freqtrade backtesting --config user_data/config.json --strategy TABaseStrategy

# Run tests
pytest tests/
```

## Project Structure

```
finance-analyzer/
├── user_data/
│   ├── strategies/          # Trading strategies
│   ├── freqaimodels/        # ML models
│   └── data/                # Historical data (gitignored)
├── avanza_monitor/          # Stock monitoring service
├── sentiment/               # Sentiment analysis
├── dashboard/               # Web UI
├── tests/                   # Test suite
│   ├── unit/
│   ├── integration/
│   └── backtest/
└── docs/plans/              # Design documents
```

## Development Phases

1. **Foundation** - Freqtrade + Binance + basic TA + Telegram
2. **Pattern Detection** - Chart pattern recognition
3. **ML Integration** - FreqAI price prediction + anomaly detection
4. **Sentiment** - News/social media analysis
5. **Avanza** - Stock monitoring (alerts only)
6. **Dashboard** - Web interface
7. **Feedback Loop** - Self-improvement system

## License

Private - All rights reserved
