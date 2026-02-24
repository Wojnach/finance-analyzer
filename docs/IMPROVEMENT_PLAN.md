# Improvement Plan — Telegram Message Routing & Dashboard Integration

**Session:** 2026-02-24 (telegram routing)
**Branch:** `improve/auto-session-2026-02-24-telegram`

## Goal

Disable most Telegram sending while preserving message generation. Route messages by category:
- **Always send to Telegram:** ISKBETS, BIG BET, simulated trades (Patient/Bold BUY/SELL), 4-hourly digest
- **Save only (no Telegram):** Analysis/HOLD messages, Layer 2 invocation notifications, regime alerts, FX warnings, errors

All messages saved to `data/telegram_messages.jsonl` with category metadata for dashboard viewing.

## Architecture

### Message Categories

| Category     | Source                    | Send to Telegram | Description                          |
|-------------|--------------------------|-----------------|--------------------------------------|
| `trade`     | Layer 2 agent (CLAUDE.md)| YES             | Simulated BUY/SELL executions        |
| `iskbets`   | iskbets.py               | YES             | Intraday entry/exit alerts           |
| `bigbet`    | bigbet.py                | YES             | Mean-reversion BIG BET alerts        |
| `digest`    | digest.py                | YES             | 4-hourly activity report             |
| `analysis`  | Layer 2 agent (CLAUDE.md)| NO              | HOLD analysis, market commentary     |
| `invocation`| agent_invocation.py      | NO              | "Layer 2 T2 invoked" notifications   |
| `regime`    | regime_alerts.py         | NO              | Regime shift alerts                  |
| `fx_alert`  | fx_rates.py              | NO              | FX rate staleness warnings           |
| `error`     | main.py                  | NO              | Loop crash notifications             |

### JSONL Format (enhanced)

```json
{"ts": "ISO-8601", "text": "message", "category": "trade", "sent": true}
```

## Batches

### Batch 1: Core routing infrastructure

**Files:** `portfolio/message_store.py` (NEW), `portfolio/telegram_notifications.py`

1. Create `portfolio/message_store.py`:
   - `SEND_CATEGORIES = {"trade", "iskbets", "bigbet", "digest"}`
   - `log_message(text, category, sent=False)` — append to JSONL with metadata
   - `send_or_store(msg, config, category)` — routes: if category in SEND_CATEGORIES, send + log; else log only
   - `_do_send_telegram(msg, config)` — actual API call extracted from send_telegram

2. Modify `portfolio/telegram_notifications.py`:
   - Keep existing functions for backwards compatibility

### Batch 2: Update Layer 1 senders

**Files:** `bigbet.py`, `iskbets.py`, `agent_invocation.py`, `regime_alerts.py`, `fx_rates.py`, `main.py`

### Batch 3: CLAUDE.md — Update Layer 2 agent instructions

**File:** `CLAUDE.md`

### Batch 4: Dashboard — API endpoint + Messages tab

**Files:** `dashboard/app.py`, `dashboard/static/index.html`

### Batch 5: Enhanced 4-hourly digest

**File:** `portfolio/digest.py`
