# Portfolio Intelligence -- System Design

A practical engineering reference for the finance-analyzer trading system.
Last updated: 2026-02-22 (audit v3: + circuit breaker, config validator, Telegram consolidation).

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Layer 1: Python Fast Loop](#layer-1-python-fast-loop)
3. [Layer 2: Claude Code Agent](#layer-2-claude-code-agent)
4. [Signal Pipeline](#signal-pipeline)
5. [Trigger System](#trigger-system)
6. [Consensus Engine](#consensus-engine)
7. [Dual Portfolio System](#dual-portfolio-system)
8. [Data Sources](#data-sources)
9. [File Layout](#file-layout)
10. [Data Files](#data-files)
11. [Infrastructure](#infrastructure)
12. [Operational Notes](#operational-notes)
13. [Known Issues and TODOs](#known-issues-and-todos)

---

## Architecture Overview

```
+------------------------------------------------------------------+
|                     Windows 11 Pro Host                          |
|                                                                  |
|  Task Scheduler                                                  |
|  +---------------------------+   +----------------------------+  |
|  | PF-DataLoop (at logon)    |   | PF-Dashboard (at logon)    |  |
|  | pf-loop.bat               |   | dashboard/app.py :5000     |  |
|  +---------------------------+   +----------------------------+  |
|              |                                                   |
|              v                                                   |
|  +---------------------------+                                   |
|  | Layer 1: main.py --loop   |                                   |
|  | Every 60s (market open)   |                                   |
|  | Every 300s (market closed) |                                   |
|  | Every 600s (weekends)     |                                   |
|  |                           |                                   |
|  | 1. Fetch prices (31 tkrs) |                                   |
|  | 2. Compute 25 signals x7  |                                   |
|  | 3. Check triggers         |-----> NO TRIGGER: sleep & retry  |
|  | 4. Write agent_summary    |                                   |
|  | 5. Invoke Layer 2 --------+---+                               |
|  +---------------------------+   |                               |
|                                  v                               |
|  +------------------------------------------+                   |
|  | Layer 2: claude -p (Claude Code agent)    |                   |
|  | pf-agent.bat -> Claude Opus 4             |                   |
|  |                                           |                   |
|  | 1. Read layer2_context.md (memory)        |                   |
|  | 2. Read agent_summary_compact.json        |                   |
|  | 3. Read both portfolio states             |                   |
|  | 4. Analyze signals + macro + regime       |                   |
|  | 5. Decide: Patient HOLD/BUY/SELL          |                   |
|  | 6. Decide: Bold HOLD/BUY/SELL             |                   |
|  | 7. Edit portfolio_state*.json (if trade)  |                   |
|  | 8. Append layer2_journal.jsonl            |                   |
|  | 9. Send Telegram message                  |                   |
|  +------------------------------------------+                   |
|              |                                                   |
|  +---------------------------+   +----------------------------+  |
|  | PF-OutcomeCheck (daily)   |   | PF-MLRetrain (weekly)      |  |
|  | --check-outcomes          |   | --retrain                  |  |
|  +---------------------------+   +----------------------------+  |
+------------------------------------------------------------------+
```

**Key invariant:** Layer 1 NEVER trades or sends Telegram messages.
Layer 2 is the sole authority on both.

---

## Layer 1: Python Fast Loop

**Entry point:** `portfolio/main.py --loop` (thin orchestrator, ~435 lines; logic in 12 extracted modules)

### Loop Lifecycle

```
loop()
  |
  +-> run(force_report=True)           # initial report on startup
  |
  +-> while True:
        get_market_state()              # open / closed / weekend
        time.sleep(interval)            # 60s / 300s / 600s
        run()
        _maybe_send_digest()            # 4-hour digest to Telegram
```

### run() -- Single Cycle

```
run(force_report, active_symbols)
  |
  +-> fetch_usd_sek()                  # FX rate for SEK conversion
  |
  +-> for each ticker in active_symbols:
  |     collect_timeframes(source)      # 7 timeframes of OHLCV
  |     compute_indicators(df)          # RSI, MACD, EMA, BB, etc.
  |     generate_signal(ind, ticker)    # 25 signal votes + consensus
  |
  +-> check_triggers(signals, prices, fear_greeds, sentiments)
  |     |
  |     +-> if triggered:
  |           write_agent_summary()     # full JSON (~64K tokens)
  |           _write_compact_summary()  # stripped version (~18K tokens)
  |           log_signal_snapshot()     # forward tracking for accuracy
  |           invoke_agent(reasons)     # launch Layer 2 subprocess
```

### Market State Scheduling

| State      | Condition                         | Interval | Symbols        |
|------------|-----------------------------------|----------|----------------|
| `open`     | Weekday, 07:00-20/21:00 UTC       | 60s      | All 31 tickers |
| `closed`   | Weekday night                      | 300s     | All 31 tickers |
| `weekend`  | Saturday/Sunday                    | 600s     | All 31 tickers |

NYSE close is DST-aware: 20:00 UTC (EDT, Mar-Nov) or 21:00 UTC (EST, Nov-Mar).
Layer 2 agent window: 06:00 UTC to 1 hour after NYSE close (weekdays only).

### Multi-Timeframe Analysis

Seven timeframes per ticker, each with its own candle resolution and cache TTL:

| Label | Binance Interval | Candles | Cache TTL | Coverage  |
|-------|-----------------|---------|-----------|-----------|
| Now   | 15m             | 100     | 0s        | ~25 hours |
| 12h   | 1h              | 100     | 5 min     | ~4 days   |
| 2d    | 4h              | 100     | 15 min    | ~17 days  |
| 7d    | 1d              | 100     | 1 hour    | ~100 days |
| 1mo   | 3d              | 100     | 4 hours   | ~300 days |
| 3mo   | 1w              | 100     | 12 hours  | ~2 years  |
| 6mo   | 1M              | 48      | 24 hours  | ~4 years  |

Stocks use Alpaca-specific intervals that map to similar coverage.

### Rate Limiting

| Provider  | Limit Target | Actual Limit     |
|-----------|-------------|------------------|
| Alpaca    | 150/min     | 200 req/min      |
| Binance   | 600/min     | 1200 weight/min  |
| yfinance  | 30/min      | No official limit |

Implemented via a token-bucket `_RateLimiter` class with per-provider instances.

### Tool Caching

Expensive tools are cached in-memory (`_tool_cache` dict) with configurable TTLs:

| Tool           | TTL    |
|----------------|--------|
| Fear & Greed   | 5 min  |
| Sentiment      | 15 min |
| Ministral LLM  | 15 min |
| ML Classifier  | 15 min |
| Funding Rate   | 15 min |
| Volume         | 5 min  |

On error, the cache is extended by 60 seconds (retry cooldown) to avoid
hammering a failing API.

---

## Layer 2: Claude Code Agent

**Entry point:** `scripts/win/pf-agent.bat` which runs `claude -p` with the
CLAUDE.md project instructions.

### Invocation Flow

```
Layer 1: invoke_agent(reasons)
  |
  +-> journal.write_context()         # build layer2_context.md from recent journal
  +-> Popen(["cmd", "/c", "pf-agent.bat"])
  |     env: CLAUDECODE=, CLAUDE_CODE_ENTRYPOINT= (cleared)
  |     stdout/stderr -> data/agent.log
  +-> Track PID + start time for timeout
```

### Agent Session

The Claude Code agent receives CLAUDE.md automatically (project root) and is
prompted to:

1. Read `data/layer2_context.md` (memory from previous invocations)
2. Read `data/agent_summary_compact.json` (signals, ~18K tokens)
3. Read `data/portfolio_state.json` and `data/portfolio_state_bold.json`
4. Analyze, decide, execute trades (edit portfolio JSON)
5. Append to `data/layer2_journal.jsonl`
6. Send Telegram message via API

**Max turns:** 25
**Allowed tools:** Edit, Read, Bash, Write
**Timeout:** 600 seconds (10 min). Killed via `taskkill /F /T` if exceeded.

### Memory System

```
layer2_journal.jsonl          # append-only decision log (structured JSON)
        |
        v
journal.py: write_context()   # reads recent entries, builds markdown summary
        |
        v
layer2_context.md             # human-readable memory for Layer 2
                              # includes: theses, prices, regime, watchlist
```

The context file is regenerated before every Layer 2 invocation so the agent
always has its most recent reasoning available.

---

## Signal Pipeline

### Overview

```
Raw OHLCV data
      |
      v
compute_indicators(df)        # RSI(14), MACD(12,26,9), EMA(9,21), BB(20,2),
      |                       # ATR, volume stats, rolling percentiles
      v
generate_signal(ind, ticker)  # 25 votes -> consensus
      |
      +-> Core signals (1-11): inline in main.py
      +-> Enhanced composites (12-25): portfolio/signals/*.py
      +-> Weighted consensus: _weighted_consensus()
      +-> Confluence score
      |
      v
{ action, confidence, weighted_action, weighted_confidence, votes, ... }
```

### 11 Core Signals

| # | Signal            | Logic                                             | Asset Class |
|---|-------------------|----------------------------------------------------|-------------|
| 1 | RSI(14)           | <30 BUY, >70 SELL (adaptive thresholds)             | All         |
| 2 | MACD(12,26,9)     | Histogram crossover only                            | All         |
| 3 | EMA(9,21)         | Fast>slow BUY, fast<slow SELL (0.5% deadband)       | All         |
| 4 | BB(20,2)          | Below lower BUY, above upper SELL                   | All         |
| 5 | Fear & Greed      | <=20 contrarian BUY, >=80 contrarian SELL            | All         |
| 6 | Sentiment         | CryptoBERT (crypto) / Trading-Hero-LLM (stocks)    | All         |
| 7 | Ministral-8B      | CryptoTrader-LM LoRA, full LLM reasoning           | Crypto      |
| 8 | ML Classifier     | HistGradientBoosting on 1h candles                  | Crypto      |
| 9 | Funding Rate      | Binance perps, contrarian (>0.03% SELL, <-0.01% BUY)| Crypto      |
| 10| Volume Confirm    | >1.5x 20-period avg confirms 3-candle direction     | All         |
| 11| Custom LoRA       | **DISABLED** (model removed from pipeline)           | --          |

### 14 Enhanced Composite Signals

Each module lives in `portfolio/signals/` and runs 4-8 sub-indicators internally,
returning a single BUY/SELL/HOLD vote via majority voting.

| #  | Module            | File                     | Sub-Indicators (abbreviated)                                      |
|----|-------------------|--------------------------|------------------------------------------------------------------|
| 12 | Trend             | `trend.py`               | Golden/Death Cross, MA Ribbon, Price vs MA200, Supertrend, SAR, Ichimoku, ADX |
| 13 | Momentum          | `momentum.py`            | RSI Divergence, Stochastic, StochRSI, CCI, Williams %R, ROC, PPO, Bull/Bear Power |
| 14 | Volume Flow       | `volume_flow.py`         | OBV, VWAP, A/D Line, CMF, MFI, Volume RSI                       |
| 15 | Volatility        | `volatility.py`          | BB Squeeze/Breakout, ATR Expansion, Keltner, Historical Vol, Donchian |
| 16 | Candlestick       | `candlestick.py`         | Hammer/Shooting Star, Engulfing, Doji, Morning/Evening Star      |
| 17 | Structure         | `structure.py`           | High/Low Breakout, Donchian 55, RSI Centerline, MACD Zero-Line  |
| 18 | Fibonacci         | `fibonacci.py`           | Retracement, Golden Pocket, Extensions, Pivot Points, Camarilla  |
| 19 | Smart Money       | `smart_money.py`         | BOS, CHoCH, Fair Value Gaps, Liquidity Sweeps, Supply/Demand     |
| 20 | Oscillators       | `oscillators.py`         | Awesome Oscillator, Aroon, Vortex, Chande, KST, Schaff, TRIX, Coppock |
| 21 | Heikin-Ashi       | `heikin_ashi.py`         | HA Trend/Doji/Color, Hull MA, Williams Alligator, Elder Impulse, TTM |
| 22 | Mean Reversion    | `mean_reversion.py`      | RSI(2), RSI(3), IBS, Consecutive Days, Gap Fade, BB %B           |
| 23 | Calendar          | `calendar_seasonal.py`   | Day-of-Week, Turnaround Tuesday, Month-End, Sell in May, FOMC Drift |
| 24 | Macro Regime      | `macro_regime.py`        | SMA filter (adaptive 50-200), DXY, Yield Curve, FOMC proximity   |
| 25 | Momentum Factors  | `momentum_factors.py`    | Time-Series Momentum, ROC-20, 52-Week High/Low, Acceleration     |

### Signal Applicability by Asset Class

| Asset Class     | Core Signals | Enhanced | Total |
|-----------------|-------------|----------|-------|
| Crypto (BTC/ETH)| 10 (custom LoRA disabled) | 14 | 24 |
| Stocks (27 tickers) | 7 (no Ministral, ML, Funding, custom LoRA) | 14 | 21 |
| Metals (XAU/XAG) | 7 (same as stocks) | 14 | 21 |

---

## Trigger System

**File:** `portfolio/trigger.py` (202 lines)

Layer 2 is expensive (Claude API call, ~10 min). The trigger system gates
invocations so Layer 2 is only called when something meaningful changes.

### Trigger Types

```
check_triggers(signals, prices_usd, fear_greeds, sentiments)
      |
      +-> 0. Post-trade reset: if Layer 2 traded since last trigger,
      |      reset cooldown for immediate reassessment
      |
      +-> 1. Signal consensus: ticker newly reaches BUY or SELL
      |      (compares current vs previously-triggered consensus)
      |
      +-> 2. Signal flip sustained: action held for 3 consecutive
      |      cycles (~3 min) to filter BUY<->HOLD noise
      |
      +-> 3. Price move: >2% change since last trigger
      |
      +-> 4. Fear & Greed: crossed threshold 20 or 80
      |
      +-> 5. Sentiment reversal: positive <-> negative
      |
      +-> 6. Cooldown expired:
      |      1 min (market hours) -- max silence
      |      120 min (nights/weekends) -- crypto-only check-ins
      |
      +-> return (triggered: bool, reasons: list[str])
```

### State Persistence

Trigger state is saved atomically to `data/trigger_state.json` after every check.
Tracks:
- `last_trigger_time` -- epoch timestamp of last Layer 2 invocation
- `last` -- snapshot of signals, prices, F&G, sentiments at last trigger
- `sustained_counts` -- per-ticker consecutive cycle counts for flip detection
- `last_checked_tx_count` -- transaction counts for post-trade detection

---

## Consensus Engine

### Vote Aggregation

```python
# In generate_signal(), after collecting all 25 votes:

core_signals = {rsi, macd, ema, bb, fear_greed, sentiment, ministral, ml, funding, volume}
enhanced_signals = {trend, momentum, volume_flow, ..., momentum_factors}

# Count active voters (BUY or SELL, not HOLD)
active_voters = count(vote != "HOLD" for vote in all_votes)
core_active = count(vote != "HOLD" for vote in core_signals)

# Gates (both must pass)
if core_active == 0:     -> HOLD  # no core signal voted
if active_voters < MIN:  -> HOLD  # MIN_VOTERS = 3 crypto, 3 stocks

# Raw consensus
buy_conf = buy_count / active_voters
sell_conf = sell_count / active_voters
action = BUY if buy_conf > sell_conf else SELL if sell_conf > buy_conf else HOLD
```

### Weighted Consensus

Beyond raw vote counting, the system computes an accuracy-weighted consensus:

```python
def _weighted_consensus(votes, accuracy_data, regime, activation_rates):
    # weight = accuracy_weight * regime_multiplier * normalized_weight
    #
    # accuracy_weight:   from signal_accuracy_1d (historical hit rate)
    # regime_multiplier: trending -> trust EMA/MACD more; ranging -> trust RSI/BB more
    # normalized_weight: rarity_bonus * bias_penalty (from activation frequency)
    #                    rare, balanced signals weighted up; noisy/biased down
```

### Signal Accuracy Tracking

**File:** `portfolio/accuracy_stats.py` (563 lines)

Every trigger invocation logs a snapshot to `data/signal_log.jsonl` with all 25
signal votes and the current price. The outcome tracker (`outcome_tracker.py`,
360 lines) backfills what actually happened at 1d/3d/5d/10d horizons.

```
signal_log.jsonl entry:
  { ts, ticker, price, votes: {rsi: "BUY", macd: "HOLD", ...}, consensus: "BUY" }
      |
      | (daily --check-outcomes job)
      v
  { ..., outcomes: { "1d": { price: X, pct: +2.1%, correct: true }, ... } }
      |
      v
accuracy_stats.py: compute per-signal hit rates
      -> accuracy_cache.json (1-hour TTL)
      -> embedded in agent_summary.json as signal_accuracy_1d
```

Current accuracy (as of Feb 2026, sample sizes vary):

| Signal     | 1d Accuracy | Samples |
|------------|------------|---------|
| Funding    | 88%        | 25      |
| ML         | 66%        | 595     |
| MACD       | 66%        | 142     |
| F&G        | 60%        | 1016    |
| Ministral  | 58%        | 892     |
| Volume     | 48%        | 890     |
| BB         | 43%        | 162     |
| RSI        | 41%        | 101     |
| EMA        | 40%        | 2020    |
| Sentiment  | 31%        | 407     |
| LoRA       | 21%        | 522     |
| Consensus  | 44%        | 503     |

---

## Dual Portfolio System

Two independent simulated portfolios, each starting at 500,000 SEK:

### Patient Strategy -- "The Regime Reader"

| Parameter        | Value                                    |
|------------------|------------------------------------------|
| File             | `data/portfolio_state.json`               |
| BUY allocation   | 15% of cash per trade                    |
| SELL allocation  | 50% of position (partial exit)           |
| Max positions    | 5 concurrent                             |
| Averaging down   | Allowed once per ticker, if thesis intact |
| Hold time        | Days to weeks                            |
| Style            | Conservative, multi-TF confirmation      |
| FOMC behavior    | Avoid new positions within 4 hours       |

### Bold Strategy -- "The Breakout Trend Rider"

| Parameter        | Value                                    |
|------------------|------------------------------------------|
| File             | `data/portfolio_state_bold.json`          |
| BUY allocation   | 30% of cash per trade                    |
| SELL allocation  | 100% of position (full exit)             |
| Max positions    | 3 concurrent                             |
| Averaging down   | Strongly avoid                           |
| Hold time        | Days to weeks                            |
| Style            | Aggressive trend follower, breakout entry |
| FOMC behavior    | Don't trade event; watch for post-event breakouts |

### Fee Model

| Asset Class | Fee Rate | Round-Trip Cost |
|-------------|----------|-----------------|
| Crypto      | 0.05%    | ~0.10%          |
| Stocks      | 0.10%    | ~0.20%          |

Fees are deducted from the allocation (BUY) or proceeds (SELL) and accumulated
in `total_fees_sek` within the portfolio state.

### Trade Execution Math

```
BUY:
  alloc = cash * 0.30 (bold) or 0.15 (patient)
  fee = alloc * fee_rate
  shares = (alloc - fee) / price_sek
  cash -= alloc

SELL:
  shares_sold = position * 1.00 (bold) or 0.50 (patient)
  proceeds = shares_sold * price_sek
  fee = proceeds * fee_rate
  cash += (proceeds - fee)
```

### Portfolio State Schema

```json
{
  "initial_value_sek": 500000,
  "cash_sek": 464535.0,
  "total_fees_sek": 285.0,
  "holdings": {
    "BTC-USD": {
      "shares": 0.1322,
      "avg_cost_usd": 68209.91
    }
  },
  "transactions": [
    {
      "timestamp": "2026-02-13T14:22:00Z",
      "ticker": "BTC-USD",
      "action": "BUY",
      "shares": 0.1322,
      "price_usd": 68209.91,
      "price_sek": 605432.0,
      "total_sek": 80000.0,
      "fee_sek": 40.0,
      "confidence": 0.7,
      "fx_rate": 8.88,
      "reason": "4/9 consensus + 5 consecutive BUY checks"
    }
  ]
}
```

---

## Data Sources

### Price Data

```
                    +------------------+
                    |   Binance Spot   |  BTC-USD, ETH-USD
                    |  api.binance.com |  OHLCV klines
                    +------------------+
                             |
+------------------+         |         +------------------+
|  Binance FAPI    |         |         |   Alpaca IEX     |  27 US stocks
| fapi.binance.com |  XAU, XAG        |  api.alpaca.com  |  OHLCV bars
+------------------+         |         +------------------+
                             |                   |
                    +--------+---------+---------+
                    |                            |
                    v                            v
              compute_indicators()         yfinance (fallback)
```

### External Signals

| Signal Source       | Provider              | Method                    | TTL    |
|--------------------|-----------------------|---------------------------|--------|
| Fear & Greed       | alternative.me (crypto), VIX-derived (stocks) | REST API | 5 min  |
| Sentiment (crypto) | CryptoBERT model      | Local inference + NewsAPI + Reddit | 15 min |
| Sentiment (stocks) | Trading-Hero-LLM      | Local inference + NewsAPI + Reddit | 15 min |
| Ministral-8B       | Local Ollama/vLLM     | CryptoTrader-LM LoRA, LLM inference | 15 min |
| ML Classifier      | Local model           | HistGradientBoosting (sklearn), 1h candles | 15 min |
| Funding Rate       | Binance futures API   | REST (perpetual swap funding) | 15 min |
| Social Sentiment   | Reddit via PRAW       | Subreddit scraping, merged into sentiment | 15 min |

### Macro Context (non-voting, for Layer 2 reasoning)

| Data Point     | Source      | Fields                                        |
|----------------|-------------|-----------------------------------------------|
| DXY            | yfinance    | Current, 5d change, SMA20, trend              |
| Treasury Yields| yfinance    | 2Y, 10Y, 30Y yields + 2s10s spread            |
| FOMC Calendar  | Static list | Next date, days until                         |
| FX Rate        | ECB / forex | USD/SEK for portfolio SEK conversion          |

---

## File Layout

### Source Code

```
Q:/finance-analyzer/
|
+-- portfolio/                    # Layer 1 core
|   +-- main.py                   # Thin orchestrator: loop, run, CLI entry (~435 lines)
|   +-- shared_state.py           # Mutable globals, caching, rate limiters
|   +-- market_timing.py          # DST-aware market hours, agent window
|   +-- fx_rates.py               # USD/SEK exchange rate with caching
|   +-- indicators.py             # compute_indicators, detect_regime, technical_signal
|   +-- data_collector.py         # Binance/Alpaca/yfinance kline fetchers, multi-TF collector
|   +-- signal_engine.py          # 25-signal voting system, generate_signal (~570 lines)
|   +-- portfolio_mgr.py          # Portfolio state load/save/value, atomic writes
|   +-- reporting.py              # agent_summary.json builder, compact summary
|   +-- telegram_notifications.py # Telegram send/escape/alert
|   +-- digest.py                 # 4-hour digest builder
|   +-- agent_invocation.py       # Layer 2 Claude Code subprocess invocation
|   +-- logging_config.py         # Structured logging with RotatingFileHandler
|   +-- signal_registry.py       # Plugin-style signal registration system (NEW v2)
|   +-- file_utils.py            # Shared atomic_write_json utility (NEW v2)
|   +-- circuit_breaker.py       # Circuit breaker for data source APIs (NEW v3)
|   +-- config_validator.py      # Startup config.json validation (NEW v3)
|   +-- signal_db.py              # SQLite storage for signal snapshots
|   +-- migrate_signal_log.py     # One-time JSONL → SQLite migration script
|   +-- trigger.py                # Trigger detection (202 lines)
|   +-- journal.py                # Layer 2 memory/context builder (407 lines)
|   +-- accuracy_stats.py         # Signal accuracy computation (563 lines)
|   +-- outcome_tracker.py        # Outcome backfilling for signal_log (360 lines)
|   +-- tickers.py                # Single source of truth for all ticker lists
|   +-- http_retry.py             # HTTP retry with exponential backoff (integrated into all API calls)
|   +-- health.py                 # Health monitoring: heartbeat, error tracking, agent silence
|   +-- fear_greed.py             # Fear & Greed fetcher
|   +-- sentiment.py              # CryptoBERT / Trading-Hero-LLM sentiment
|   +-- social_sentiment.py       # Reddit scraper for social posts
|   +-- ministral_signal.py       # Ministral-8B inference wrapper
|   +-- ministral_trader.py       # CryptoTrader-LM LoRA trading logic
|   +-- ml_signal.py              # ML classifier inference
|   +-- ml_trainer.py             # HistGradientBoosting training pipeline
|   +-- funding_rate.py           # Binance perpetual funding rate
|   +-- macro_context.py          # DXY, treasury yields, yield curve
|   +-- fomc_dates.py             # FOMC calendar (static list)
|   +-- api_utils.py              # Config loading, shared API utilities
|   +-- avanza_client.py          # Avanza API client (Tier 2/3 Nordic stocks)
|   +-- avanza_tracker.py         # Avanza position tracking
|   +-- avanza_watch.py           # Avanza watchlist monitoring
|   +-- backup.py                 # Data backup utilities
|   +-- config_validator.py       # Config schema validation
|   +-- data_refresh.py           # Manual data refresh tools
|   +-- equity_curve.py           # Equity curve computation
|   +-- iskbets.py                # ISKBETS game integration
|   +-- kelly_sizing.py           # Kelly criterion position sizing
|   +-- log_rotation.py           # Log file rotation
|   +-- portfolio_validator.py    # Portfolio state validation
|   +-- regime_alerts.py          # Regime change alerting
|   +-- risk_management.py        # Risk checks and limits
|   +-- signal_history.py         # Historical signal data
|   +-- stats.py                  # Portfolio statistics
|   +-- telegram_poller.py        # Incoming Telegram command handler
|   +-- weekly_digest.py          # Weekly performance digest
|   +-- signals/                  # Enhanced composite signal modules
|       +-- trend.py
|       +-- momentum.py
|       +-- volume_flow.py
|       +-- volatility.py
|       +-- candlestick.py
|       +-- structure.py
|       +-- fibonacci.py
|       +-- smart_money.py
|       +-- oscillators.py
|       +-- heikin_ashi.py
|       +-- mean_reversion.py
|       +-- calendar_seasonal.py
|       +-- macro_regime.py
|       +-- momentum_factors.py
|
+-- dashboard/                    # Web dashboard
|   +-- app.py                    # Flask API + static serving (387 lines)
|   +-- static/                   # Frontend assets
|
+-- scripts/win/                  # Windows automation
|   +-- pf-loop.bat               # Loop launcher with auto-restart
|   +-- pf-agent.bat              # Layer 2 invocation (claude -p)
|   +-- pf.bat                    # Manual signal report
|
+-- docs/                         # Documentation
|   +-- architecture-plan.md      # Canonical architecture reference
|   +-- system-design.md          # This document
|   +-- operational-runbook.md    # Ops procedures
|   +-- enhanced-signals.md       # Enhanced signal module specs
|   +-- dashboard-api.md          # Dashboard API reference
|
+-- CLAUDE.md                     # Layer 2 agent instructions (auto-loaded)
+-- config.json                   # API keys, Telegram token, settings
```

---

## Data Files

All runtime data lives in `data/`:

### Core Data (read/written every cycle)

| File                          | Format  | Size    | Purpose                                    |
|-------------------------------|---------|---------|--------------------------------------------|
| `agent_summary.json`          | JSON    | ~64K    | Full signal output for all tickers         |
| `agent_summary_compact.json`  | JSON    | ~18K    | Stripped version for Layer 2 (no enhanced_signals detail) |
| `trigger_state.json`          | JSON    | ~5K     | Trigger counters, cooldowns, last-seen state |
| `portfolio_state.json`        | JSON    | ~2K     | Patient strategy state                     |
| `portfolio_state_bold.json`   | JSON    | ~2K     | Bold strategy state                        |

### Append-Only Logs

| File                          | Format  | Growth   | Purpose                                   |
|-------------------------------|---------|----------|-------------------------------------------|
| `signal_log.jsonl`            | JSONL   | Unbounded | Forward tracking: every signal snapshot + outcomes |
| `layer2_journal.jsonl`        | JSONL   | ~1 entry/trigger | Layer 2 decision history (structured)  |
| `telegram_messages.jsonl`     | JSONL   | ~1 entry/trigger | Archive of all sent Telegram messages  |
| `invocations.jsonl`           | JSONL   | ~1 entry/trigger | Layer 2 invocation metadata            |
| `analysis_log.jsonl`          | JSONL   | Per analysis | Detailed analysis traces                |

### Derived / Cache

| File                          | Format  | TTL      | Purpose                                   |
|-------------------------------|---------|----------|-------------------------------------------|
| `layer2_context.md`           | Markdown| Rebuilt per trigger | Layer 2 memory (recent journal entries) |
| `accuracy_cache.json`         | JSON    | 1 hour   | Cached signal accuracy stats              |
| `activation_cache.json`       | JSON    | Per cycle | Signal activation frequency for weighting |

### Operational

| File              | Purpose                                           |
|-------------------|---------------------------------------------------|
| `loop_out.txt`    | Stdout from the main loop (for debugging)         |
| `loop_err.txt`    | Stderr from the main loop                         |
| `agent.log`       | Stdout/stderr from Layer 2 invocations            |
| `heartbeat.txt`   | Last-known-alive timestamp                        |

---

## Infrastructure

### Host Environment

```
OS:       Windows 11 Pro 10.0.26200
Python:   .venv/Scripts/python.exe (venv)
Shell:    Git Bash (C:\Program Files\Git\usr\bin\bash.exe)
          CLAUDE_CODE_GIT_BASH_PATH must point here, NOT to WSL bash
```

### Task Scheduler Jobs

| Task Name        | Schedule     | Command              | Status   |
|------------------|-------------|----------------------|----------|
| PF-DataLoop      | At logon    | `pf-loop.bat`        | ENABLED  |
| PF-Dashboard     | At logon    | `dashboard/app.py`   | ENABLED  |
| PF-OutcomeCheck  | Daily 18:00 | `main.py --check-outcomes` | ENABLED |
| PF-MLRetrain     | Weekly      | `main.py --retrain`  | ENABLED  |
| PF-Loop          | --          | (duplicate)          | DISABLED |
| Portfolio-Agent  | --          | (bypassed triggers)  | DISABLED |

### Auto-Restart Mechanism

```batch
@echo off
cd /d Q:\finance-analyzer

:restart
set CLAUDECODE=
set CLAUDE_CODE_ENTRYPOINT=
echo [%date% %time%] Starting loop...
.venv\Scripts\python.exe -u portfolio\main.py --loop
echo [%date% %time%] Loop exited (code %ERRORLEVEL%). Restarting in 30s...
timeout /t 30 /nobreak >nul
goto restart
```

If `main.py` crashes for any reason, the batch file waits 30 seconds and
restarts. The `CLAUDECODE` env var is cleared to prevent "nested session" errors
when Claude Code is invoked as a subprocess.

### Layer 2 Timeout

```python
AGENT_TIMEOUT = 600  # seconds (10 min)

# In invoke_agent():
if agent_proc.poll() is None and elapsed > AGENT_TIMEOUT:
    # Windows: taskkill /F /T /PID <pid>
    # Unix: agent_proc.kill()
```

### Crash Alerting

```python
def _crash_alert(error_msg):
    # Sends truncated traceback to Telegram (max 3000 chars)
    # Called from loop() except handler
```

Loop errors print to stdout only (captured in `loop_out.txt`). Proactive health
monitoring via `health.py` detects agent silence (`check_agent_silence()`) and
sends Telegram alerts when the agent hasn't been invoked within expected windows
(2h market hours, 4h off-hours).

---

## Operational Notes

### CLI Commands

```bash
# Force a signal report (single Layer 1 cycle)
.venv/Scripts/python.exe -u portfolio/main.py --report

# Backfill price outcomes for signal accuracy
.venv/Scripts/python.exe -u portfolio/main.py --check-outcomes

# Print signal accuracy report
.venv/Scripts/python.exe -u portfolio/main.py --accuracy

# Get current Fear & Greed
.venv/Scripts/python.exe -u portfolio/fear_greed.py

# Get current sentiment
.venv/Scripts/python.exe -u portfolio/sentiment.py

# Retrain ML model
.venv/Scripts/python.exe -u portfolio/main.py --retrain
```

### Proven Noise Patterns

These patterns have been observed and confirmed as noise -- do not trust them:

1. **BTC 12h BUY phantom:** Appeared 20+ times, always fades. Died, resurrected,
   died, resurrected. Ignore.
2. **ETH Now TF:** Oscillates SELL->BUY within single cycles. Pure noise.
3. **BTC/ETH Now TF swaps:** Swap directions each check-in. No persistence.
4. **Ministral/ML flips:** SELL->HOLD->SELL in consecutive checks. Treat as
   noise unless sustained 3+ consecutive check-ins.
5. **Single-check improvements** (MACD, RSI, volume, BB): Need 3+ consecutive
   checks to be meaningful.
6. **MSTR/NVDA after-hours volatility:** Not actionable; wait for market session.

### Digest System

A 4-hour periodic digest is sent via Telegram with a summary of current state
across all tickers, even when no triggers have fired. Built by
`_build_digest_message()` and gated by `_maybe_send_digest()` with a 4-hour
interval check.

---

## Known Issues and TODOs

### Architecture Debt

| Issue | Impact | Effort |
|-------|--------|--------|
| ~~`main.py` is 2,124 lines monolithic~~ | **RESOLVED** — extracted 12 modules, main.py is ~435 lines | Feb 21 |
| ~~`http_retry.py` exists but NOT integrated into any API calls~~ | **RESOLVED** — integrated into all API calls (Feb 21 audit) | Done |
| `signal_log.jsonl` grows unbounded | Disk usage, slow accuracy scans | Add rotation (log_rotation.py exists but may not cover this) |

### New Modules (Feb 21-22 Audit)

| Module | Purpose |
|--------|---------|
| `portfolio/signal_registry.py` | Plugin-style signal registration; signals register at import, signal_engine discovers from registry instead of hardcoded lists |
| `portfolio/file_utils.py` | Shared `atomic_write_json()` extracted from 6 modules (portfolio_mgr, trigger, bigbet, iskbets, health, accuracy_stats) |
| `portfolio/circuit_breaker.py` | Circuit breaker for data source APIs (Binance spot/fapi, Alpaca). CLOSED→OPEN after 5 failures, OPEN→HALF_OPEN after 60s recovery timeout |
| `portfolio/config_validator.py` | Startup validation of config.json. Checks required keys (telegram, alpaca), warns on missing optional keys. Called once in `main.loop()` |

### Bug Fixes (Feb 21 Audit v2)

| Bug | Fix |
|-----|-----|
| `indicators.py` regime cache used `id()` as key (memory address reuse → stale cache) | Changed to hashable tuple of indicator values |
| `health.py` hardcoded market hours (7-21 UTC, not DST-aware) | Uses `market_timing.get_market_state()` now |
| `iskbets.py` circular import via `from portfolio.main import fetch_usd_sek` | Changed to `from portfolio.fx_rates import fetch_usd_sek` |
| `data_collector.py` duplicated Alpaca headers (local `_get_alpaca_headers()`) | Replaced with `api_utils.get_alpaca_headers()` import |
| `bigbet.py` + `iskbets.py` used `print()` instead of structured logger | Converted to `logging.getLogger()` calls |
| `tickers.py` had 4 redundant mappings (TICKER_SOURCE_MAP, YF_MAP, BINANCE_MAP) | Derived from SYMBOLS dict |
| `iskbets.py` had unused imports (`requests`, `numpy`) | Removed |

### Testing Gaps

| Area | Current State |
|------|---------------|
| `test_digest.py` | Has tests (digest formatting, interval logic) |
| Enhanced signal modules | Untested individually |
| Trigger system | 40+ tests in `test_trigger_edge_cases.py` |

### Signal Quality

| Issue | Detail |
|-------|--------|
| Consensus accuracy: 44% | Barely above coin flip; weighted consensus improving |
| EMA accuracy: 40% | High sample count (2020) confirms poor predictive value |
| Sentiment accuracy: 31% | Worse than random -- consider inverting or disabling |
| Custom LoRA: DISABLED | Model was 21% accurate, removed from pipeline |
| MIN_VOTERS=3 for stocks | Stocks reach consensus easily with only 21 signals; Layer 2 judgment is critical filter |

### Operational Risks

| Risk | Mitigation | Gap |
|------|------------|-----|
| Loop dies silently | Auto-restart in bat file | No alerting on prolonged silence |
| `CLAUDECODE` env var causes nested session error | Cleared in bat + invoke_agent | Caused 34h outage on Feb 18-19 |
| Agent hangs past 600s | taskkill timeout | No monitoring for repeated hangs |
| Windows venv double-process | Normal (launcher stub) | Can confuse process monitoring |
| yfinance rate limiting | 30/min token bucket | No backoff on 429s (http_retry not integrated) |
