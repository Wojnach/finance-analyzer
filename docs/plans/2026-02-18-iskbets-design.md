# ISKBETS Design — 2026-02-18

## Summary

Intraday "quick gamble" mode. User allocates a fixed SEK amount and targets one or more
instruments. System actively hunts for a good entry, alerts via Telegram, mirrors the real
Avanza trade in a shadow portfolio, and monitors for exit conditions.

## Architecture (Option C — Hybrid Layer 1 / Layer 2)

Layer 1 (`iskbets.py`) handles all mechanics — entry evaluation, exit checks, state management,
hard guardrails. Layer 2 (Claude) is invoked only for entry judgment when Layer 1 sees
threshold conditions met. Hard exits (stop-loss, time exit) are mechanical and never
depend on Claude being available.

```
Main loop (60s)
├── Normal signal run (existing)
├── bigbet.check_bigbet() (existing)
└── iskbets.check_iskbets() (NEW, if config enabled)
        │
        ├── [No active position] → scan all target tickers
        │       ├── Threshold not met → log, do nothing
        │       └── Threshold met on ticker(s) → invoke Layer 2 for entry judgment
        │               └── Layer 2 sends Telegram alert
        │
        └── [Active position] → check exit conditions every cycle
                ├── Hard stop hit → send exit alert immediately (Layer 1 only)
                ├── Stage 1 target hit → move stop to breakeven, notify
                ├── Trailing stop hit → send exit alert (Layer 1 only)
                ├── Time exit (15:50 ET) → send exit alert (Layer 1 only)
                └── Signal reversal (≥3 sell votes, confirmed 2 cycles) → send exit alert
        │
        └── [Backup candidates] → continue scanning, alert if threshold met
                └── User can switch by replying "cancel" + "bought TICKER PRICE AMOUNT"

Telegram poller (background thread, polls getUpdates every 5s)
└── "bought MSTR 129.50 100000" → set active_position in iskbets_state.json
└── "sold"                       → close position, log P&L, keep mode active
└── "cancel"                     → disable ISKBETS mode entirely
└── "status"                     → send current position summary
```

## Configuration — `data/iskbets_config.json`

```json
{
  "enabled": true,
  "amount_sek": 100000,
  "tickers": ["MSTR", "PLTR"],
  "expiry": "2026-02-18T23:59:00Z"
}
```

- `tickers`: any subset of [BTC-USD, ETH-USD, MSTR, PLTR, NVDA]
- `expiry`: auto-disable after this timestamp (prevents stale mode running overnight)
- Enabled via dashboard toggle or direct file edit

## State — `data/iskbets_state.json`

```json
{
  "active_position": {
    "ticker": "MSTR",
    "entry_price_usd": 129.5,
    "amount_sek": 100000,
    "shares": 71.2,
    "entry_time": "2026-02-18T14:23:00Z",
    "atr_15m": 1.8,
    "stop_loss": 125.9,
    "stage1_target": 132.2,
    "stop_at_breakeven": false,
    "highest_close": 129.5
  },
  "backup_candidates": ["PLTR"],
  "trade_history": []
}
```

## Entry Threshold

Both gates must pass:

1. **Big Bet conditions**: ≥2 of 6 conditions met (RSI extreme, BB extremes on 2+ TFs,
   F&G extreme, 24h price drop/spike ≥5%, volume capitulation, MACD turning)
2. **Main signal grid**: ≥3 buy votes on the ticker

No entry alerts after **14:30 ET**. No entry during FOMC or major scheduled events.

## Exit Strategy (research-backed)

Priority order (highest wins):

1. **Hard stop** (Layer 1, immediate): `entry_price − (2.0 × ATR_15m)` — fires unconditionally
2. **Time exit** (Layer 1, immediate): 15:50 ET hard close alert — no exceptions
3. **Stage 1 target** (Layer 1): `entry_price + (1.5 × ATR_15m)` → move stop to breakeven
4. **Trailing stop** (Layer 1): `highest_close − (1.0 × ATR_15m)`, recalculated each 60s cycle
5. **Signal reversal** (Layer 1): ≥3 sell votes confirmed over 2 consecutive cycles (~2 min)

ATR is computed from 15-minute candles at entry time and stays fixed for stop calculation.
Trailing stop recalculates dynamically as price moves in our favor.

Research basis: Zarattini et al. (2024), Maróy (2025) — ladder exits with trailing ATR stop
produce Sharpe ratios over 3.0 on high-beta US equities vs fixed percentage targets.

## Telegram Flow

```
System  → "🟡 ISKBETS: Entry alert — MSTR $129.50
           4B/0S signals, RSI 24 + BB below lower (2 TFs)
           Stop: $125.90 | Stage 1: $132.20 | No entry after 14:30 ET"

You buy on Avanza, reply:
User    → "bought MSTR 129.50 100000"

System  → "✅ Position tracked.
           Stop: $125.90 (-2.8%) | Stage 1: $132.20 (+2.1%)
           Trailing thereafter | Hard close: 15:50 ET"

... price reaches $132.20 ...
System  → "🟢 Stage 1 hit — MSTR $132.20 (+2.1%)
           Stop moved to breakeven $129.50. Trailing from here."

... trailing stop hit at $131.40 ...
System  → "🔴 EXIT — MSTR $131.40 (+1.5%)
           Trailing stop triggered. Consider selling."

User sells, replies:
User    → "sold"

System  → "📊 ISKBETS closed. MSTR +1,500 SEK (+1.5%) in 2h 10min"
```

## Dashboard Panel

- Enable/disable toggle (writes `iskbets_config.json`)
- Ticker multi-select (any of the 5 tracked instruments)
- SEK amount input
- Expiry time (default: today's market close)
- Active state: pulsing indicator, live P&L %, stop level, stage 1 target, time remaining
- Backup candidates panel: other tickers still being scanned
- Trade history table: past ISKBETS trades with outcome

## Files to Create

| File                           | Description                                                         |
| ------------------------------ | ------------------------------------------------------------------- |
| `portfolio/iskbets.py`         | Layer 1 — entry eval, exit checks, state r/w, alert formatting      |
| `portfolio/telegram_poller.py` | Background thread, getUpdates every 5s, command parser              |
| `dashboard/static/iskbets.js`  | Dashboard panel JS                                                  |
| `tests/test_iskbets.py`        | Unit tests — entry threshold, exit logic, command parsing, P&L math |

## Files to Modify

| File                          | Change                                                   |
| ----------------------------- | -------------------------------------------------------- |
| `portfolio/main.py`           | Call `check_iskbets()` in main loop, start poller thread |
| `dashboard/app.py`            | Add `/api/iskbets` GET + POST endpoints                  |
| `dashboard/static/index.html` | Add ISKBETS panel                                        |
| `config.json`                 | Add `iskbets` section (min_conditions, etc.)             |
| `data/.gitignore`             | Add `iskbets_config.json`, `iskbets_state.json`          |

## Research References

- Zarattini, Aziz, Barbon (2024) — VWAP trailing stop, hard EOD close — SSRN 4824172
- Maróy (2025) — Ladder exits with ATR trailing — SSRN 5095349
- Dai et al. — Trailing stops reduce risk without sacrificing return — SSRN 3338243
