# Avanza Communication Pipeline Overhaul — Design Spec

**Date:** 2026-03-30
**Status:** Draft
**Scope:** Replace 4 overlapping Avanza modules with a unified package, add TOTP auth, WebSocket streaming, and 10+ new API endpoints.

---

## Problem Statement

The current Avanza integration has **4 overlapping modules** (`avanza_session.py`, `avanza_client.py`, `avanza_control.py`, `metals_avanza_helpers.py`) with **2 different transport layers** (Playwright `context.request` and `page.evaluate` JS fetch). This causes:

1. **Unnecessary complexity** — Playwright launches a full Chromium browser (~150MB RAM) for simple HTTP requests
2. **Manual auth dependency** — BankID session requires human phone interaction every ~24h
3. **No real-time data** — polling at 10-60s intervals when sub-second WebSocket exists
4. **Missing capabilities** — no order depth, no tick-size awareness, no fill verification, broken instrument search, no order modification
5. **Code duplication** — same operations implemented differently in each module
6. **Playwright lock contention** — single `_pw_lock` serializes all API calls (BUG-129)

## Architecture Decision

**Replace Playwright-based HTTP calls with `requests.Session` + TOTP authentication.**

Rationale:
- TOTP auth is fully programmatic — no manual BankID needed
- `requests.Session` provides HTTP connection pooling with ~2ms overhead vs ~800ms Playwright startup
- TOTP auth returns `pushSubscriptionId` required for WebSocket streaming
- Thread-safe with proper session management (vs single Playwright lock)
- ~10x lower memory footprint (no Chromium process)
- `scripts/avanza_login.py` stays as BankID backup for initial TOTP setup

## Module Structure

```
portfolio/avanza/
    __init__.py          # Public API — all imports come from here
    auth.py              # TOTP authentication + session lifecycle
    client.py            # HTTP client (requests.Session + connection pooling)
    streaming.py         # WebSocket streaming (CometD/Bayeux protocol)
    trading.py           # Order placement, modification, cancellation, stop-losses
    market_data.py       # Quotes, order depth, OHLC, instrument info
    account.py           # Positions, buying power, deals, transactions
    search.py            # Instrument search, warrant/cert discovery
    tick_rules.py        # Tick size logic, price rounding
    types.py             # Dataclasses for API responses
```

### Backward Compatibility

The old modules (`avanza_session.py`, `avanza_client.py`, `avanza_control.py`, `metals_avanza_helpers.py`) will be kept temporarily as thin wrappers that delegate to the new package. This allows incremental migration without a big-bang change.

```python
# portfolio/avanza_session.py (after migration)
"""Legacy compatibility — delegates to portfolio.avanza package."""
from portfolio.avanza import (
    api_get, api_post, api_delete,
    get_positions, get_buying_power, get_quote,
    place_buy_order, place_sell_order, cancel_order,
    # ... etc
)
```

Callers that import `_get_playwright_context` or `close_playwright` (6 files) need direct migration since those concepts don't exist in the new architecture.

---

## Phase 1: TOTP Authentication (`auth.py`, `client.py`)

### Auth Flow

```
1. POST /_api/authentication/sessions/usercredentials
   Body: {username, password, maxInactiveMinutes: 1440}
   Response: {twoFactorLogin: {method: "TOTP", transactionId: "..."}}

2. POST /_api/authentication/sessions/totp
   Body: {method: "TOTP", totpCode: "<6-digit>"}
   Response: {
     authenticationSession: "...",
     pushSubscriptionId: "...",   ← needed for WebSocket
     customerId: "...",
     registrationComplete: true
   }
   Headers: X-SecurityToken: "..." ← CSRF token for all mutations
```

### Session Lifecycle

- **Auto-renewal:** Re-authenticate before session expires (configurable, default 1440 min = 24h)
- **Thread-safe:** `threading.Lock` around auth state, not around every API call
- **Health check:** Lightweight `GET /_api/position-data/positions` every 30 min
- **Fallback:** If TOTP fails, log warning + send Telegram alert (don't try BankID automatically — that needs human)

### `client.py` — HTTP Client

```python
class AvanzaClient:
    """Thread-safe HTTP client with connection pooling and TOTP auth."""

    def __init__(self, config: dict):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 ...",
            "Accept": "application/json",
        })
        self._auth = AvanzaAuth(config, self._session)
        self._csrf_token: str = ""

    def get(self, path: str) -> dict: ...
    def post(self, path: str, payload: dict) -> dict: ...
    def delete(self, path: str) -> dict: ...
```

Key design choices:
- Single `requests.Session` instance — reuses TCP connections (connection pooling)
- CSRF token stored and auto-refreshed on auth
- 401 response triggers automatic re-auth (one retry)
- 429 response triggers exponential backoff (unlikely but defensive)
- All responses parsed as JSON with proper error raising

---

## Phase 2: Unified API Surface

### `trading.py` — Order Management

```python
def place_order(side: str, ob_id: str, price: float, volume: int,
                condition: str = "NORMAL", valid_until: str | None = None) -> OrderResult:
    """Place a limit order. Condition: NORMAL, FILL_OR_KILL, FILL_AND_KILL."""

def modify_order(order_id: str, price: float | None = None,
                 volume: int | None = None) -> OrderResult:
    """Modify an existing order in-place (no cancel+replace needed)."""

def cancel_order(order_id: str) -> bool:
    """Cancel an open order. Returns True on success."""

def get_orders() -> list[Order]:
    """Get all open orders."""

def get_deals(since: str | None = None) -> list[Deal]:
    """Get recent fills with buyer/seller codes."""

# Stop-losses
def place_stop_loss(ob_id: str, trigger_price: float, sell_price: float,
                    volume: int, valid_days: int = 8,
                    trigger_type: str = "LESS_OR_EQUAL",
                    value_type: str = "MONETARY") -> StopLossResult:
    """Place hardware stop-loss. trigger_type: LESS_OR_EQUAL, FOLLOW_DOWNWARDS, etc."""

def place_trailing_stop(ob_id: str, trail_percent: float, volume: int,
                        valid_days: int = 8) -> StopLossResult:
    """Hardware trailing stop — Avanza manages the trail, not us."""

def get_stop_losses() -> list[StopLoss]:
    """List all active stop-loss orders."""

def delete_stop_loss(stop_id: str) -> bool:
    """Delete a stop-loss. Idempotent (404 = already gone)."""
```

### `market_data.py` — Prices, Depth, Charts

```python
def get_quote(ob_id: str) -> Quote:
    """Fast quote: bid, ask, last, spread, change, volume."""

def get_market_data(ob_id: str) -> MarketData:
    """Full market data: quote + order depth (5 levels) + recent trades."""

def get_ohlc(ob_id: str, period: str = "one_month",
             resolution: str | None = None) -> list[OHLC]:
    """OHLC candle data. Periods: one_week, one_month, three_months, etc.
    Resolutions: MINUTE, 5MIN, 10MIN, 30MIN, HOUR, DAY, WEEK."""

def get_instrument_info(ob_id: str, instrument_type: str = "certificate") -> InstrumentInfo:
    """Full instrument data including underlying, leverage, barrier."""

def get_orderbook_info(ob_id: str) -> OrderbookInfo:
    """Tick sizes, trading status, valid-until range, collateral value."""

def get_news(ob_id: str) -> list[NewsArticle]:
    """Recent news articles for an instrument."""
```

### `account.py` — Account State

```python
def get_positions() -> list[Position]:
    """All positions with value, profit, acquired value."""

def get_buying_power(account_id: str | None = None) -> AccountCash:
    """Buying power, total value, own capital."""

def get_transactions(from_date: str, to_date: str,
                     types: list[str] | None = None) -> list[Transaction]:
    """Transaction history. Types: BUY, SELL, DIVIDEND, DEPOSIT, WITHDRAW."""
```

### `search.py` — Instrument Discovery

```python
def search(query: str, limit: int = 10) -> list[SearchHit]:
    """Filtered search — returns orderbook IDs, names, types, prices."""

def find_warrants(underlying_id: str, direction: str = "LONG",
                  sort_by: str = "LEVERAGE") -> list[Warrant]:
    """Find warrants by underlying instrument."""

def find_certificates(underlying_id: str, direction: str = "LONG") -> list[Certificate]:
    """Find certificates by underlying instrument."""
```

### `tick_rules.py` — Correct Pricing

```python
def get_tick_size(ob_id: str) -> list[TickEntry]:
    """Get tick size rules for an instrument. Cached per session."""

def round_to_tick(price: float, ob_id: str, direction: str = "down") -> float:
    """Round a price to the nearest valid tick increment.
    direction='down' for buy orders, 'up' for sell orders."""
```

---

## Phase 3: New Endpoints Integration

### Endpoints to Add (all tested live, confirmed working)

| Endpoint | Method | Module | Latency |
|----------|--------|--------|---------|
| `/_api/trading-critical/rest/marketdata/{id}` | GET | market_data | 15ms |
| `/_api/search/filtered-search` | POST | search | ~50ms |
| `/_api/trading-critical/rest/orderbook/{id}` | GET | tick_rules | ~20ms |
| `/_api/price-chart/stock/{id}?timePeriod=X` | GET | market_data | ~30ms |
| `/_api/trading-critical/rest/order/modify` | POST | trading | untested |
| `/_api/trading/rest/deals` | GET | trading | 23ms |
| `/_api/trading/rest/orders` | GET | trading | 20ms |
| `/_api/market-guide/news/{id}` | GET | market_data | ~40ms |
| `/_api/market-guide/warrant/search?...` | GET | search | untested |
| `/_api/market-guide/certificate/list?...` | GET | search | untested |
| `/_api/transactions/list?...` | GET | account | untested |

### Endpoints NOT Adding (tested, don't work or low value)

| Endpoint | Reason |
|----------|--------|
| `/_api/account-performance/overview/chart/...` | Returns 500 |
| `/_api/orderbook/{id}` | Socket hang up |
| `/_api/trading/stoploss/list/{accountId}` | Returns 500 |
| `/_api/isk/transactions` | Socket hang up |
| Watchlist endpoints | Low value for trading |

---

## Phase 4: WebSocket Streaming (`streaming.py`)

### Protocol: CometD/Bayeux over WebSocket

```
URL: wss://www.avanza.se/_push/cometd

Handshake:
  → {channel: "/meta/handshake", ext: {subscriptionId: pushSubscriptionId},
     version: "1.0", supportedConnectionTypes: ["websocket"]}
  ← {clientId: "abc123", successful: true}

Connect:
  → {channel: "/meta/connect", clientId: "abc123", connectionType: "websocket"}

Subscribe:
  → {channel: "/meta/subscribe", subscription: "/quotes/856394", clientId: "abc123"}
  ← Price updates pushed automatically
```

### Channels to Implement

| Channel | Priority | Use Case |
|---------|----------|----------|
| `quotes/{obId}` | P0 | Replace 10s polling with sub-second price updates |
| `orderdepths/{obId}` | P0 | Live order book for better entry/exit |
| `orders/_{accountIds}` | P1 | Real-time order status (fill confirmation) |
| `deals/_{accountIds}` | P1 | Instant fill notification |
| `trades/{obId}` | P2 | Tape reading (see individual trades) |
| `positions/_{obId}_{accountIds}` | P2 | Position change alerts |

### Design

```python
class AvanzaStream:
    """WebSocket streaming client using CometD/Bayeux protocol."""

    def __init__(self, push_subscription_id: str):
        self._ws: websocket.WebSocket = None
        self._client_id: str = ""
        self._callbacks: dict[str, list[Callable]] = {}
        self._reconnect_delay: float = 1.0

    def connect(self) -> None: ...
    def subscribe_quotes(self, ob_ids: list[str], callback: Callable) -> None: ...
    def subscribe_order_depth(self, ob_ids: list[str], callback: Callable) -> None: ...
    def subscribe_orders(self, account_ids: list[str], callback: Callable) -> None: ...
    def subscribe_deals(self, account_ids: list[str], callback: Callable) -> None: ...
    def close(self) -> None: ...

    # Internal
    def _run_loop(self) -> None:
        """Background thread: read messages, dispatch to callbacks."""
    def _reconnect(self) -> None:
        """Auto-reconnect with exponential backoff."""
    def _heartbeat(self) -> None:
        """Send /meta/connect at regular intervals to keep alive."""
```

### Integration with Metals Loop

```python
# Before (polling):
while True:
    price = fetch_price(page, ob_id, api_type)  # 15-50ms, every 10s
    sleep(10)

# After (streaming):
stream.subscribe_quotes([ob_id], on_price_update)  # Sub-second push
```

The metals loop can subscribe to all warrant orderbook IDs at startup and receive price updates pushed to callbacks. The 10-second polling loop for silver fast-tick becomes unnecessary.

---

## Phase 5: Advanced Order Features

### Hardware Trailing Stops

Currently we implement trailing stops in software (metals_loop.py). Avanza supports hardware trailing stops:

```python
place_stop_loss(
    trigger_type="FOLLOW_DOWNWARDS",  # Avanza trails automatically
    value_type="PERCENTAGE",
    trigger_value=5.0,  # 5% trailing distance
    ...
)
```

This is superior because:
- Works even if our process crashes
- No polling needed to update stop levels
- Avanza's system is faster to react

### Fill-or-Kill / Fill-and-Kill

```python
place_order(side="BUY", condition="FILL_OR_KILL", ...)  # All or nothing
place_order(side="BUY", condition="FILL_AND_KILL", ...)  # Fill what you can
```

Useful for:
- FOK: When we want a specific size and won't accept partial fills
- FAK: When we want immediate execution of whatever liquidity exists

### Order Modification

```python
modify_order(order_id="12345", price=25.50)  # Change price without cancel+replace
```

Eliminates the current pattern of cancel → wait → replace → hope nobody filled the gap.

---

## Types (`types.py`)

```python
@dataclass
class Quote:
    bid: float | None
    ask: float | None
    last: float | None
    spread: float | None
    change_percent: float | None
    high: float | None
    low: float | None
    volume: int
    vwap: float | None
    updated: str

@dataclass
class OrderDepthLevel:
    price: float
    volume: int

@dataclass
class MarketData:
    quote: Quote
    bid_levels: list[OrderDepthLevel]
    ask_levels: list[OrderDepthLevel]
    recent_trades: list[Trade]
    market_maker_expected: bool

@dataclass
class OrderResult:
    success: bool
    order_id: str
    status: str  # SUCCESS, ERROR, etc.
    message: str

@dataclass
class StopLossResult:
    success: bool
    stop_id: str
    status: str

@dataclass
class Position:
    name: str
    orderbook_id: str
    instrument_type: str
    volume: int
    value: float
    acquired_value: float
    profit: float
    profit_percent: float
    last_price: float
    change_percent: float
    account_id: str
    currency: str

@dataclass
class SearchHit:
    orderbook_id: str
    name: str
    instrument_type: str
    tradeable: bool
    price_last: str
    change_percent: str
    spread: str

@dataclass
class TickEntry:
    min_price: float
    max_price: float
    tick_size: float

@dataclass
class OHLC:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: int
```

---

## Migration Plan

### Files Requiring Direct Migration (use `_get_playwright_context`)

These 6 files directly access Playwright internals and need rewriting:
1. `portfolio/fin_snipe_manager.py` — uses `_get_playwright_context`, `close_playwright`
2. `scripts/fin_fish_monitor.py` — same + `metals_avanza_helpers` functions
3. `data/place_stoploss_once.py` — uses `_get_playwright_context`
4. `data/gold_sell_final.py` — uses `_get_playwright_context` + helpers
5. `data/gold_sell_retry.py` — same
6. `data/gold_sell_debug.py` — same

### Files Using Legacy Wrappers (seamless migration)

~30 files import from `avanza_session`, `avanza_client`, `avanza_control`, or `metals_avanza_helpers`. The legacy wrapper modules will re-export from the new package, so these callers work without changes initially. They can be migrated incrementally.

### metals_loop.py — Special Case

The metals loop currently manages its own Playwright browser instance. Migration path:
1. Replace `metals_avanza_helpers` calls with `portfolio.avanza` calls
2. Remove browser/page management entirely
3. Add WebSocket streaming for price updates
4. Keep the 60s main loop for signal computation, but prices arrive via WebSocket callbacks

---

## Testing Strategy

1. **Unit tests** for each new module (auth, client, trading, market_data, account, search, tick_rules)
2. **Integration test** with mocked HTTP responses (not live API)
3. **Live smoke test** script that verifies TOTP auth + key endpoints work
4. **WebSocket test** with mock CometD server
5. **Legacy wrapper tests** — ensure old test_avanza_*.py tests still pass via wrappers

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| TOTP auth fails | BankID session stays as backup; Telegram alert on failure |
| Avanza changes API | Error monitoring + Telegram alerts; legacy modules remain until validated |
| WebSocket disconnects | Auto-reconnect with exponential backoff; fall back to HTTP polling |
| Session expiry during trade | Auto-renewal 30 min before expiry; retry on 401 |
| Breaking existing callers | Legacy wrapper modules provide backward compat |
| Rate limiting (unknown limits) | Conservative polling, exponential backoff on 429 |

---

## Success Criteria

1. No Chromium process needed for runtime API calls
2. Sessions auto-renew without human intervention
3. Sub-second price updates via WebSocket
4. Order depth visible before every trade
5. Prices rounded to valid tick sizes
6. Fill verification after every order
7. Instrument search returns correct results
8. All existing tests pass via legacy wrappers
9. Metals loop works without Playwright
