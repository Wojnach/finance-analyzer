# Architecture Plan: Drop Ministral, Add Smart Trigger System

## Problem

Running Claude Code every 60 seconds = 1,440 calls/day = rate-limited fast.
Ministral-8B adds noise, not signal. Most cycles return "HOLD — nothing changed."

## Solution

Local loop handles data collection + change detection (free, 24/7).
Claude Code gets invoked only when something meaningful changes (~20-50x/day).

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  LOCAL LOOP (every 60s) — FREE, runs forever         │
│                                                       │
│  • Fetch Binance prices + candles                    │
│  • Compute RSI, MACD, EMA, BB (just math)            │
│  • Fetch Fear & Greed (cached 5 min)                 │
│  • Run CryptoBERT sentiment (local model)            │
│  • Store everything in agent_summary.json            │
│                                                       │
│  CHANGE DETECTION:                                    │
│  • Did any signal flip? (BUY→SELL, HOLD→BUY, etc)   │
│  • Price moved >2% since last Claude call?           │
│  • Fear & Greed crossed a threshold? (20/80)         │
│  • New high-confidence sentiment shift?              │
│                                                       │
│  If YES to any → trigger Claude Code                 │
│  If NO → log locally, keep looping                   │
└──────────────┬───────────────────────────────────────┘
               │ (only when something meaningful changes)
               ▼
┌──────────────────────────────────────────────────────┐
│  CLAUDE CODE (maybe 20-50x/day instead of 1,440)     │
│                                                       │
│  • Reads full context (all timeframes, signals)      │
│  • Considers portfolio state + risk                  │
│  • Makes BUY/SELL/HOLD decision with reasoning       │
│  • Executes trade (edit portfolio_state.json)        │
│  • Sends Telegram with explanation                   │
└──────────────────────────────────────────────────────┘
```

## Token Budget

| | Per call | Per hour | Per day |
|---|---|---|---|
| Dumb loop (every 60s) | ~6K tokens | ~360K | **~8.6M** |
| Smart triggers (~30/day) | ~6K tokens | ~7.5K | **~180K** |

**~48x reduction in Claude usage.**

## What Changes

### Remove
- `portfolio/ministral_signal.py` — wrapper, no longer needed
- `models/ministral_trader.py` — Ministral-8B inference script
- Ministral vote from `generate_signal()` in `main.py`
- Ministral reasoning from Telegram report

### Add
- `portfolio/trigger.py` — change detection module
  - Tracks last signal state per symbol
  - Detects signal flips (BUY↔SELL↔HOLD)
  - Detects price moves >2% since last Claude call
  - Detects Fear & Greed threshold crossings (20/80)
  - Detects sentiment shifts (positive↔negative)
  - Returns `should_trigger: bool` + `reasons: list[str]`

### Modify
- `portfolio/main.py`
  - Remove Ministral from signal chain (7 signals → 6)
  - Add trigger check after computing signals
  - If triggered: invoke `claude -p` with context
  - If not triggered: log and continue
- `portfolio/collect.py`
  - Add trigger reasons to `agent_summary.json`
- `scripts/win/pf-loop.bat` (and Linux equivalent)
  - Runs `main.py --loop 60` as before
  - Claude invocation happens inside the loop when triggered

## Signal Chain (After)

6 signals, no LLM in the loop:

| Signal | Source | Buy | Sell |
|--------|--------|-----|------|
| RSI(14) | Technical | < 30 | > 70 |
| MACD(12,26,9) | Technical | Histogram > 0 | Histogram < 0 |
| EMA(9,21) | Technical | 9 > 21 | 9 < 21 |
| Bollinger Bands | Technical | Price < lower | Price > upper |
| Fear & Greed | alternative.me | ≤ 20 | ≥ 80 |
| News Sentiment | CryptoBERT | Positive (conf > 0.4) | Negative (conf > 0.4) |

Decision: BUY/SELL when 4+ of 6 agree (67%+). HOLD otherwise.

## Trigger Conditions

Claude Code gets called when ANY of these fire:

1. **Signal flip** — Any symbol's action changed (e.g., HOLD → BUY)
2. **Price move** — >2% change since last Claude call
3. **Fear & Greed shift** — Crossed 20 (extreme fear) or 80 (extreme greed)
4. **Sentiment reversal** — Overall sentiment flipped (positive ↔ negative)
5. **Cooldown expired** — No Claude call in >2 hours (periodic check-in)

## Endgame: Automated Exchange Trading

Once the signal + trigger system is proven:

1. Add Binance API trading module (authenticated, uses `config.json` keys)
2. Replace `portfolio_state.json` edits with real API calls
3. Keep simulated portfolio running in parallel for comparison
4. Add safety rails: max position size, daily loss limit, kill switch
