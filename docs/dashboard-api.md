# Dashboard API Documentation

> **Last updated:** 2026-02-20
> **Source:** `dashboard/app.py`
> **Port:** 5055
> **Base URL:** `http://localhost:5055`

## Authentication

All API endpoints (except `GET /`) require authentication when `dashboard_token` is set in
`config.json`. If no token is configured, access is unauthenticated (backwards compatible).

### Authentication methods

1. **Query parameter:** `?token=YOUR_TOKEN`
   ```
   GET /api/signals?token=abc123
   ```

2. **Authorization header:** `Authorization: Bearer YOUR_TOKEN`
   ```
   GET /api/signals
   Authorization: Bearer abc123
   ```

### Error response (401)

```json
{
  "error": "Unauthorized",
  "message": "Invalid or missing token"
}
```

---

## Endpoints

### `GET /`

Serves the static dashboard frontend.

- **Auth required:** No
- **Response:** HTML page from `dashboard/static/index.html`

---

### `GET /api/summary`

Combined endpoint for auto-refresh. Returns signals, both portfolios, and recent Telegram
messages in a single request. This is the primary endpoint used by the dashboard frontend.

- **Auth required:** Yes
- **Response:**

```json
{
  "signals": { ... },
  "portfolio": { ... },
  "portfolio_bold": { ... },
  "telegrams": [ ... ]
}
```

| Field | Type | Source file | Description |
|-------|------|------------|-------------|
| `signals` | object or null | `data/agent_summary.json` | Latest signal snapshot (all 25 signals, all timeframes, all tickers) |
| `portfolio` | object or null | `data/portfolio_state.json` | Patient strategy state (cash, holdings, transactions) |
| `portfolio_bold` | object or null | `data/portfolio_state_bold.json` | Bold strategy state (cash, holdings, transactions) |
| `telegrams` | array | `data/telegram_messages.jsonl` | Last 50 Telegram messages (most recent last) |

---

### `GET /api/signals`

Returns the latest signal snapshot from Layer 1.

- **Auth required:** Yes
- **Source:** `data/agent_summary.json`
- **Response:** The full agent_summary object containing:
  - Per-ticker signal votes (all 25 signals)
  - Timeframe heatmaps (7 timeframes)
  - Indicator values (RSI, MACD, EMA, BB, ATR, etc.)
  - Enhanced signal details (sub-signals and indicators)
  - Macro context (DXY, treasury yields, Fed calendar)
  - Fear & Greed index
  - Sentiment scores
  - Signal accuracy data
- **Error (404):**
  ```json
  {"error": "no data"}
  ```

---

### `GET /api/portfolio`

Returns the Patient strategy portfolio state.

- **Auth required:** Yes
- **Source:** `data/portfolio_state.json`
- **Response:**

```json
{
  "cash_sek": 500000.0,
  "initial_value_sek": 500000,
  "holdings": {},
  "transactions": [],
  "total_fees_sek": 0.0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `cash_sek` | float | Current cash balance in SEK |
| `initial_value_sek` | int | Starting portfolio value (500000) |
| `holdings` | object | Map of ticker to `{shares, avg_cost_usd}` |
| `transactions` | array | Chronological list of all trades |
| `total_fees_sek` | float | Cumulative trading fees paid |

- **Error (404):**
  ```json
  {"error": "no data"}
  ```

---

### `GET /api/portfolio-bold`

Returns the Bold strategy portfolio state.

- **Auth required:** Yes
- **Source:** `data/portfolio_state_bold.json`
- **Response:** Same structure as `/api/portfolio`
- **Error (404):**
  ```json
  {"error": "no data"}
  ```

---

### `GET /api/invocations`

Returns recent Layer 2 invocation events (when and why Claude Code was triggered).

- **Auth required:** Yes
- **Source:** `data/invocations.jsonl`
- **Response:** Array of up to 50 most recent invocation records (most recent last)

```json
[
  {
    "ts": "2026-02-20T14:30:00+00:00",
    "trigger": "consensus:BTC-USD:BUY",
    "tickers_changed": ["BTC-USD"],
    "duration_s": 45.2
  }
]
```

---

### `GET /api/telegrams`

Returns recent Telegram messages sent by Layer 2.

- **Auth required:** Yes
- **Source:** `data/telegram_messages.jsonl`
- **Response:** Array of up to 50 most recent messages (most recent last)

```json
[
  {
    "ts": "2026-02-20T14:31:00+00:00",
    "text": "*HOLD*\n\n`BTC  $67,800  HOLD  1B/2S/8H`\n..."
  }
]
```

---

### `GET /api/signal-log`

Returns recent signal log entries (raw signal votes per cycle for accuracy tracking).

- **Auth required:** Yes
- **Source:** `data/signal_log.jsonl`
- **Response:** Array of up to 50 most recent log entries (most recent last)

Each entry contains per-ticker signal votes and prices at the time of logging.

---

### `GET /api/accuracy`

Computes and returns signal accuracy statistics across multiple time horizons.

- **Auth required:** Yes
- **Source:** Computed from `data/signal_log.jsonl` using `portfolio.accuracy_stats`
- **Response:**

```json
{
  "1d": {
    "signals": {
      "rsi": {"correct": 42, "total": 101, "accuracy": 0.416},
      "macd": {"correct": 93, "total": 142, "accuracy": 0.655},
      ...
    },
    "consensus": {
      "correct": 220, "total": 503, "accuracy": 0.437
    },
    "per_ticker": {
      "BTC-USD": {"correct": 45, "total": 100, "accuracy": 0.45},
      ...
    }
  },
  "3d": { ... },
  "5d": { ... },
  "10d": { ... }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `signals` | object | Per-signal accuracy: correct predictions, total samples, hit rate |
| `consensus` | object | Overall consensus accuracy |
| `per_ticker` | object | Accuracy broken down by ticker |

Horizons are only included if they have data (`total > 0`).

- **Error (500):**
  ```json
  {"error": "error message"}
  ```

---

### `GET /api/iskbets`

Returns ISK bets configuration and state.

- **Auth required:** Yes
- **Source:** `data/iskbets_config.json` and `data/iskbets_state.json`
- **Response:**

```json
{
  "config": { ... },
  "state": { ... }
}
```

Both fields may be `null` if the respective files do not exist.

---

### `GET /api/lora-status`

Returns LoRA training state and progress.

- **Auth required:** Yes
- **Source:** `training/lora/state.json` and `training/lora/training_progress.json`
- **Response:**

```json
{
  "state": { ... },
  "training_progress": { ... }
}
```

Both fields may be `null` if the respective files do not exist.

---

### `POST /api/validate-portfolio`

Validates a portfolio JSON for mathematical integrity.

- **Auth required:** Yes
- **Content-Type:** `application/json`
- **Request body:** A portfolio state object (same structure as `/api/portfolio` response)

**Checks performed:**
1. `cash_sek` is non-negative
2. All share counts in holdings are non-negative
3. Cash reconciliation: `initial_value - sum(BUY allocs) + sum(SELL proceeds) = cash_sek` (1 SEK tolerance)
4. Holdings reconciliation: `total_bought - total_sold = current_shares` per ticker (0.0001 tolerance)

- **Response (valid):**
```json
{
  "valid": true,
  "errors": [],
  "computed_cash": 500000.0,
  "ticker_balances": {}
}
```

- **Response (invalid):**
```json
{
  "valid": false,
  "errors": [
    "Cash mismatch: computed 499000.00 vs recorded 500000.00 (diff 1000.00 SEK)",
    "Holdings mismatch for BTC-USD: bought 0.500000 - sold 0.000000 = expected 0.500000, actual 0.000000"
  ],
  "computed_cash": 499000.0,
  "ticker_balances": {
    "BTC-USD": 0.5
  }
}
```

- **Error (400):**
  ```json
  {"valid": false, "errors": ["No JSON body provided"]}
  ```

---

### `GET /api/equity-curve`

Returns portfolio value history for charting.

- **Auth required:** Yes
- **Source:** `data/portfolio_value_history.jsonl`
- **Response:** Array of up to 5000 most recent value snapshots

```json
[
  {
    "ts": "2026-02-20T14:30:00+00:00",
    "patient_value_sek": 500000.0,
    "bold_value_sek": 464535.0
  }
]
```

Returns an empty array if the file does not exist.

---

### `GET /api/signal-heatmap`

Returns the full 25-signal x all-tickers grid for visualization.

- **Auth required:** Yes
- **Source:** Computed from `data/agent_summary.json`
- **Response:**

```json
{
  "tickers": ["BTC-USD", "ETH-USD", "MSTR", ...],
  "signals": ["rsi", "macd", "ema", "bb", ..., "momentum_factors"],
  "core_signals": ["rsi", "macd", "ema", "bb", "fear_greed", "sentiment", "ministral", "ml", "funding", "volume", "custom_lora"],
  "enhanced_signals": ["trend", "momentum", "volume_flow", "volatility_sig", "candlestick", "structure", "fibonacci", "smart_money", "oscillators", "heikin_ashi", "mean_reversion", "calendar", "macro_regime", "momentum_factors"],
  "heatmap": {
    "BTC-USD": {
      "rsi": "HOLD",
      "macd": "SELL",
      "ema": "BUY",
      ...
    },
    ...
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `tickers` | array | List of all ticker symbols |
| `signals` | array | All 25 signal names (core + enhanced) |
| `core_signals` | array | 11 core signal names |
| `enhanced_signals` | array | 14 enhanced composite signal names |
| `heatmap` | object | Map of ticker to signal-name to vote (BUY/SELL/HOLD) |

- **Error (404):**
  ```json
  {"error": "no data"}
  ```

---

### `GET /api/triggers`

Returns recent trigger/invocation events.

- **Auth required:** Yes
- **Source:** `data/invocations.jsonl`
- **Response:** Array of up to 50 most recent trigger events (same as `/api/invocations`)

---

## Data Sources

All API endpoints read from files in the `data/` directory. These files are written by
Layer 1 (the Python fast loop). The dashboard is read-only and never modifies data files
(except for the `/api/validate-portfolio` endpoint which only validates, does not write).

| File | Written by | Update frequency |
|------|-----------|-----------------|
| `agent_summary.json` | Layer 1 main loop | Every 60 seconds |
| `portfolio_state.json` | Layer 2 (Claude Code) | On trades only |
| `portfolio_state_bold.json` | Layer 2 (Claude Code) | On trades only |
| `signal_log.jsonl` | Layer 1 main loop | Every 60 seconds |
| `invocations.jsonl` | Layer 1 (on trigger) | On trigger events |
| `telegram_messages.jsonl` | Layer 2 (Claude Code) | On every invocation |
| `portfolio_value_history.jsonl` | Layer 1 main loop | Periodically |
| `iskbets_config.json` | Manual / setup | Rarely |
| `iskbets_state.json` | Layer 1 | Periodically |
| `training/lora/state.json` | LoRA training pipeline | During training |
| `training/lora/training_progress.json` | LoRA training pipeline | During training |
