# Avanza API Reference

## Authentication

Avanza uses cookie-based browser sessions. Authentication requires BankID on the user's phone.

### Login Flow
1. Run `scripts/avanza_login.py` — opens Chromium, user authenticates via BankID
2. Saves Playwright storage state to `data/avanza_storage_state.json` (cookies + localStorage)
3. Saves session tokens to `data/avanza_session.json` (security_token, authentication_session, customer_id)
4. Storage state is reusable in headless mode until session expires (~24h)

### Session Usage (Playwright)
```python
from playwright.sync_api import sync_playwright
with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)
    ctx = browser.new_context(storage_state="data/avanza_storage_state.json")
    page = ctx.new_page()
    page.goto("https://www.avanza.se/min-ekonomi/oversikt.html", wait_until="domcontentloaded")
    page.wait_for_timeout(2000)
```

### CSRF Token
Required for write operations (placing/cancelling orders). Extract from cookies:
```python
def _get_csrf(page):
    for c in page.context.cookies():
        if c["name"] == "AZACSRF":
            return c["value"]
    return None
```

## Endpoints

### GET `/_api/position-data/positions`
Returns all positions across all accounts. **This is the canonical positions endpoint.**

**Response structure (as of Mar 2026):**
```json
{
  "withOrderbook": [
    {
      "account": {
        "id": "1625505",
        "type": "INVESTERINGSSPARKONTO",
        "name": "1625505"
      },
      "instrument": {
        "id": "857913",
        "type": "CERTIFICATE",         // CERTIFICATE, WARRANT, STOCK, FUND, etc.
        "name": "BULL GULD X8 N",
        "orderbook": {
          "id": "856394",              // This is the orderbook_id for trading
          "name": "BULL GULD X8 N",
          "type": "CERTIFICATE",
          "tradeStatus": "BUYABLE_AND_SELLABLE",
          "quote": {
            "buy": {"value": 869.5, "unit": "SEK"},    // bid
            "sell": {"value": 874.1, "unit": "SEK"},   // ask
            "latest": {"value": 869.5, "unit": "SEK"}, // last trade
            "highest": {"value": 989.0, "unit": "SEK"},
            "lowest": {"value": 871.0, "unit": "SEK"},
            "change": {"value": 74.8, "unit": "SEK"},
            "changePercent": {"value": 9.41, "unit": "percentage"},
            "updated": "2026-03-02T16:20:03.125"
          }
        },
        "currency": "SEK",
        "isin": "SE0011171388"
      },
      "volume": {"value": 4},                          // number of units held
      "value": {"value": 3478.0, "unit": "SEK"},       // current market value
      "averageAcquiredPrice": {"value": 907.5, "unit": "SEK"},  // entry price
      "acquiredValue": {"value": 3630.0, "unit": "SEK"}         // cost basis
    }
  ],
  "withoutOrderbook": [],
  "cashPositions": [
    {
      "account": {"id": "1625505"},
      "totalBalance": {"value": 18363.68, "unit": "SEK"}
    }
  ]
}
```

**Key field paths:**
- Instrument name: `item.instrument.name`
- Orderbook ID (for trading): `item.instrument.orderbook.id`
- Instrument type: `item.instrument.type`
- Volume (units held): `item.volume.value`
- Entry price: `item.averageAcquiredPrice.value`
- Current bid: `item.instrument.orderbook.quote.buy.value`
- Current ask: `item.instrument.orderbook.quote.sell.value`
- Account ID: `item.account.id`

**IMPORTANT:** All numeric values are wrapped in `{"value": N, "unit": "...", "unitType": "..."}` objects. Always access `.value` to get the number.

### POST `/_api/trading-critical/rest/order/new`
Place a new order (buy or sell).

**Headers required:**
- `Content-Type: application/json`
- `X-SecurityToken: <CSRF token from AZACSRF cookie>`

**Payload:**
```json
{
  "accountId": "1625505",
  "orderbookId": "856394",
  "side": "SELL",               // "BUY" or "SELL"
  "condition": "NORMAL",
  "price": 900.0,               // limit price
  "validUntil": "2026-03-02",   // YYYY-MM-DD
  "volume": 4                   // number of units
}
```

**Response:**
```json
{
  "orderRequestStatus": "SUCCESS",
  "orderId": "123456789"
}
```

**Common errors:**
- `"short.sell.not.allowed"` — trying to sell units you don't have (e.g., broker already executed a stop-loss)
- Session expired — re-authenticate via BankID

### GET `/_api/trading-critical/rest/order/{accountId}/{orderId}`
Check order status.

**Response includes:**
- `state`: "FILLED", "EXECUTED", "DONE" = order completed
- Other states indicate pending/cancelled

### DELETE `/_api/trading-critical/rest/order/{accountId}/{orderId}`
Cancel a pending order.

**Headers required:**
- `Content-Type: application/json`
- `X-SecurityToken: <CSRF token>`

## Account IDs

| Account | Type | Usage |
|---------|------|-------|
| 1625505 | ISK (Investeringssparkonto) | Metals trading (gold, silver warrants) |
| 2674244 | Tjänstepension | Pension (stocks: Exxon, Vale, Vertiv) |

## Instrument Types

| Type | Example |
|------|---------|
| CERTIFICATE | BULL GULD X8 N (leveraged gold certificate) |
| WARRANT | MINI L SILVER AVA 301 (mini future on silver) |
| STOCK | NextEra Energy, Exxon Mobil, etc. |

## Files

| File | Purpose |
|------|---------|
| `scripts/avanza_login.py` | Interactive BankID login, saves session |
| `data/avanza_storage_state.json` | Playwright browser state (cookies) |
| `data/avanza_session.json` | Session tokens |
| `portfolio/avanza_session.py` | Session management, `get_positions()`, `api_get()` |
| `portfolio/avanza_client.py` | Trading functions: `place_buy_order()`, `place_sell_order()`, `delete_order()` |
| `portfolio/avanza_orders.py` | Human-in-the-loop order confirmation via Telegram |
| `portfolio/avanza_tracker.py` | Price tracking for Avanza instruments |

## Known Issues

- **Endpoint 404s:** Many undocumented endpoints return 404. The positions endpoint `/_api/position-data/positions` is confirmed working (Mar 2026). Do NOT try variations like `/_api/account-data/positions` or `/_api/trading/positions` — they don't exist.
- **Short sell error:** If Avanza's own stop-loss triggers before our SELL order, the API returns `"short.sell.not.allowed"` because the broker already sold the units.
- **Session expiry:** Storage state expires after ~24h. The metals loop should detect redirect to `/logga-in` and alert via Telegram.
- **Value wrapping:** As of Mar 2026, all numeric fields use structured objects `{"value": N, "unit": "...", "unitType": "..."}`. This changed from simpler flat numbers. Always access `.value`.
