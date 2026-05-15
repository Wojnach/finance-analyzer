# Adversarial Review — 5 avanza-api (second-reviewer / codex-substitute)

> Codex CLI quota was exhausted at start of session. This review is produced by a
> second Claude subagent with isolated context as a substitute second opinion.

## P0 — money-losing or data-corrupting (must fix)

### `portfolio/avanza/trading.py:80-102` — unified `place_order` bypasses ALL safety guards

The "new unified" trading path has neither the **account whitelist** nor the
**MAX_ORDER_TOTAL_SEK** ceiling that protect the legacy session path. Compare:

```python
# portfolio/avanza/trading.py (unified)
client = AvanzaClient.get_instance()
acct = account_id or client.account_id          # NO whitelist check
# ... no MAX_ORDER_TOTAL_SEK check ...
raw: dict[str, Any] = client.avanza.place_order(acct, ob_id, OrderType(side), ...)
```

vs. the session path which has both (`avanza_session.py:591-606`):

```python
if effective_account_id not in ALLOWED_ACCOUNT_IDS:
    raise ValueError(f"Refusing to trade on non-whitelisted account {effective_account_id!r}")
MAX_ORDER_TOTAL_SEK = 50_000.0
if order_total > MAX_ORDER_TOTAL_SEK:
    raise ValueError(...)
```

A caller importing from `portfolio.avanza` (the canonical package per `__init__.py:29`)
that resolves `client.account_id` from `config["avanza"]["account_id"]` (line 65 of
`avanza/client.py`) gets whatever the config says — including the pension account
`2674244` if the operator ever fat-fingers that into config. The unified path is
also missing `_place_order`'s `volume<1` / `price<=0` guards for `modify_order`
entirely. In production this means any path migrating to `portfolio.avanza.place_order`
silently loses the H7 account guard and the BUG-211 runaway-order ceiling.

### `portfolio/avanza/account.py:64-94` — `get_buying_power` returns silent zero on miss

```python
for account in accounts:
    if str(account.get("accountId", account.get("id", ""))) == acct:
        return AccountCash.from_api(account)
# Account not found — return zeroes
logger.warning("get_buying_power account_id=%s not found in overview", acct)
return AccountCash(buying_power=0.0, total_value=0.0, own_capital=0.0)
```

This is *exactly* the silent-failure pattern the C7 fix in `avanza_session.py:385-539`
was written to *eliminate* — see that docstring at line 405-412:

> "Previously this function … silently returned fake numbers derived from the
> first category's totalValue. That made callers like `fish_straddle` and
> `fish_monitor_live` size positions off wrong cash balances. … On any failure
> path we return ``None`` so callers can distinguish 'API call failed' from
> 'balance is legitimately zero'"

The unified-package version regressed back to returning a zero-filled `AccountCash`.
A caller doing `bp = get_buying_power(); if bp.buying_power < trade_size: skip` will
*always* skip; a caller doing `size = bp.buying_power * 0.5` will compute `0`; a caller
checking `if bp.buying_power == 0: emergency_close()` will trigger panic-flatten on
every API blip. This is the C7 outage pattern reintroduced.

### `portfolio/avanza_session.py:720-811` — `place_stop_loss` has no MAX_ORDER guard, no whitelist on trailing path

`place_buy_order` / `place_sell_order` raise on `order_total > 50_000 SEK`
(line 602-606). `place_stop_loss` has neither cap. A malformed/hallucinated
`volume * sell_price` of e.g. 250,000 SEK will be accepted as a stop-loss leg,
**and on trigger will fire a 250,000 SEK sell** with the same fill semantics as
the regular order path. The 1000-SEK warn-only check at line 770-774 is
fee-oriented, not safety-oriented:

```python
if value_type == "MONETARY" and sell_price > 0:
    leg_total = round(volume * sell_price, 2)
    if leg_total < 1000.0:
        logger.warning(...)  # warn only
```

`place_trailing_stop` (line 814-846) passes `trigger_price=trail_percent`
with **no validation** that `0 < trail_percent < 100`. A caller bug passing
`trail_percent=0.0` would create a degenerate trail that triggers immediately;
`trail_percent=-5` (sign error from an LLM-generated rule) gets quietly accepted.
The whitelist guard at line 750-751 fires on `place_stop_loss` but only when
called through `place_trailing_stop` since the latter delegates — so the
whitelist *is* there, but the trail-magnitude isn't.

## P1 — high-confidence bugs (should fix)

### `portfolio/avanza_session.py:88-95` — expired session falls through on unparseable `expires_at`

```python
try:
    exp = datetime.fromisoformat(expires_at)
    now = datetime.now(UTC)
    if exp <= now:
        raise AvanzaSessionError(f"Session expired at {expires_at}. ...")
except ValueError:
    logger.warning("Cannot parse expires_at %r — cannot verify expiry, proceeding with caution", expires_at)
```

A corrupted or schema-drifted `expires_at` ("Fri 14 May 2026 12:00" vs ISO) drops to
"proceeding with caution" and the call returns successfully. Combined with the
**dead** `EXPIRY_BUFFER_MINUTES = 30` constant at line 32 (never referenced
anywhere in the codebase), there is effectively no safeguard. The March 2026
3-week silent-auth outage pattern was: process kept running, all writes failed
silently. Same risk here — once a session file gets a schema-drifted expiry
string, the loop will call `api_post` and get 401s in `_op`, which closes the
browser and re-opens it from the *same* now-expired storage state, infinite
retry loop. The `is_session_expiring_soon` helper uses a hardcoded 60-minute
default (line 123) ignoring `EXPIRY_BUFFER_MINUTES` entirely.

### `portfolio/avanza_session.py:251-261` — `api_get` returns 5xx errors as exceptions but loses no-retry semantics on 404

```python
if resp.status == 401:
    close_playwright()
    raise AvanzaSessionError(...)
if not resp.ok:
    raise RuntimeError(f"Avanza API error {resp.status}: {resp.text()[:500]}")
return resp.json()
```

Note `get_buying_power` catches the resulting `Exception` at line 428 and returns
`None`, but `get_stop_losses` catches only `RuntimeError` (line 862) and re-raises
on `AvanzaSessionError`. Inconsistency: a 401 during `get_stop_losses` (which is on
the safety-critical "is the stop still alive before I sell" path) propagates out,
which is correct — but `get_buying_power` swallows the *same 401*, returning `None`
silently, and the caller has no way to tell auth-expired from balance-fetch-failed.
Both cases should fail closed with explicit error info.

### `portfolio/avanza/scanner.py:78` — `_search` ignores `itype_str` filter on BankID path

```python
def _search(itype_str, query, limit):
    return api_post("/_api/search/filtered-search", {"query": query, "limit": limit})
```

The TOTP path on line 60 passes `InstrumentType(itype_str)` correctly, but the
BankID fallback drops the type filter entirely, then `scan_instruments` post-filters
by direction string (line 187). For a `query="OLJA"` `instrument_type="warrant"` scan,
the BankID path returns stocks, ETFs, certificates, *and* warrants in the hit list,
then the parallel detail fetch hits the wrong market-guide endpoint for stocks
(forced to `api_type="warrant"` if the name contains "MINI" per line 213-217),
which causes silent detail-fetch failures and excludes legitimate results. This
isn't money-losing but the scanner ranks instruments the user trades against;
silent miscategorization can promote a stock-shaped result over a thinner
warrant-shaped one.

### `portfolio/avanza/tick_rules.py:124-126` — fallback to last tick entry hides API shape drift

```python
# Fallback: if price exceeds all ranges, use the last entry
if entries:
    return entries[-1].tick_size
return None
```

If Avanza's tick ladder gets truncated by an API change (e.g. high-price band
removed in a response shape drift), every price above the highest known band
silently uses whatever the last-listed band's tick is. Orders rounded with the
wrong tick get rejected with "ogiltigt pris"-style errors. The `_find_tick_for_price`
helper does explicitly handle `max_price == 0` as "unbounded" at line 122; the
unconditional fallback at 124 is redundant with that guarded path and only fires
on shape drift. Should raise `ValueError` so the caller bails rather than rounding
to a stale tick size.

### `portfolio/avanza_session.py:653-664` — `get_open_orders` silent-empty on RuntimeError

```python
try:
    data = api_get(f"/_api/trading/rest/order/account/{aid}")
    if isinstance(data, list):
        return data
    return data.get("orders", data.get("openOrders", []))
except RuntimeError:
    # Endpoint may vary — fallback to deal endpoint
    try:
        data = api_get("/_api/trading/rest/deals-and-orders")
        ...
    except RuntimeError:
        logger.warning("Could not fetch open orders")
        return []
```

Returns empty list on double-fetch failure. Callers (`fin_snipe`, grid fisher,
metals loop) use "no open orders" as a precondition to placing new orders ("nothing
in flight, safe to add"). API outage → empty list → loop happily places duplicate
buy-ladders. Should propagate or return a sentinel that callers MUST handle —
same `get_stop_losses_strict` / `get_stop_losses` two-API pattern this very file
already established at line 849-885 should be replicated here.

### `portfolio/avanza/streaming.py:79-85` — channel string built from raw account IDs

```python
def on_orders(self, account_ids: list[str], callback: Callable[[dict], None]) -> None:
    channel = "/orders/_" + ",".join(account_ids)
    self._callbacks.setdefault(channel, []).append(callback)
```

`account_ids` flows directly into the CometD channel name. While `account_ids`
is internally sourced today, this is a textbook "trust boundary entering a URL/path"
pattern flagged by the prompt at item 9. If `account_ids` ever derives from a
dashboard form field or config import, a malformed entry like `"1625505/quotes/0"`
silently subscribes to a different channel. Also no whitelist check that the
subscribed account is in `ALLOWED_ACCOUNT_IDS` — the streaming surface bypasses
the trade-side guards entirely (it's read-only, but the data flows into decisions).

### `portfolio/avanza_orders.py:351-403` — `_execute_confirmed_order` doesn't re-check expiry before placing

```python
def _execute_confirmed_order(order: dict, config: dict) -> None:
    """Execute a confirmed order on Avanza and notify via Telegram."""
    action = order["action"]
    try:
        if action == "BUY":
            result = place_buy_order(
                orderbook_id=order["orderbook_id"],
                price=order["price"],
                volume=order["volume"],
            )
```

A user can `CONFIRM <token>` *after* the 5-minute expiry if the `check_pending_orders`
cycle hasn't fired since expiry. Walkthrough: order created at T+0 with expiry T+5min;
user takes 6 min to type CONFIRM; if the loop's previous `check_pending_orders` call
landed at T+4:50 (not yet expired), the next call at T+5:50 sees `confirmed_tokens`
*and* `now > expires`. The expiry branch at line 192-196 `continue`s before
matching, so token-confirm wins races against expiry only by ordering. But the
flow is: line 192 sets status=expired and continues, so the confirmed branch is
NOT reached this cycle. *However* — if the Telegram poll returns the CONFIRM on the
same cycle the order would otherwise expire, the user's stated intent ("YES execute")
is silently dropped to "expired" without re-prompting. Minor UX bug; not a money
leak. Note: `place_buy_order` uses today's date for `valid_until` (the
default at `avanza_session.py:614`) — a confirmation arriving after market close
will place a day-order that immediately expires. No use-by check against market hours.

### `portfolio/avanza_session.py:1064-1070` — `cancel_all_stop_losses_for` busy-waits without rate-limit

```python
while True:
    try:
        poll_stops = get_stop_losses_strict()
    ...
    still = _filter_for_ob(poll_stops)
    remaining = [s.get("id", "") for s in still if s.get("id")]
    if not remaining:
        break
    if (time.monotonic() - started) >= max_wait:
        break
    time.sleep(poll_interval)
```

`poll_interval=0.5s`, `max_wait=3.0s` → up to 6 `get_stop_losses_strict` calls in
3 seconds = 6 Playwright `ctx.request.get(...)` calls under the global `_pw_lock`.
Every concurrent thread that needs ANY Avanza HTTP work (price fetch, position
check, even a different orderbook's order placement) blocks on this lock for up
to 3 seconds. In a multi-loop deployment (metals_loop + golddigger + grid_fisher
+ main) this is a regular pause. Not a bug per se, but the RLock at line 54 is
held across the whole `_with_browser_recovery` block — combine this with the
8-worker pool fanout described in the same comment and you get pathological
serialization on cancel storms.

## P2 — concerns / smells (worth addressing)

### `portfolio/avanza_session.py:32` — dead `EXPIRY_BUFFER_MINUTES`

```python
# Minimum remaining session life before we consider it expired (minutes)
EXPIRY_BUFFER_MINUTES = 30
```

Never referenced. `is_session_expiring_soon` (line 123) takes a parameter
`threshold_minutes: float = 60.0` instead. Either wire the buffer into the
expiry check at line 89 (treat session as expired at `now > exp - buffer`) or
delete the constant. Documentation drift = future maintainer assumes the
buffer is enforced.

### `portfolio/avanza_session.py:1031-1032` — `import copy` inside a hot function

```python
import copy as _copy
snapshot = [_copy.deepcopy(sl) for sl in initial]
```

Function-local import on every `cancel_all_stop_losses_for` call. Cheap, but
this function runs in the safety-critical cancel-before-sell hot path and the
locked section just above (`with _pw_lock`) was specifically designed to be
fast. Move to module-level.

### `portfolio/avanza/streaming.py:144-150` — WebSocket reconnect doesn't re-handshake new clientId on stale

```python
def _connect(self) -> None:
    self._ws = websocket.create_connection(WS_URL, timeout=_HEARTBEAT_INTERVAL + 10)
```

On a transient disconnect the `_run_loop` re-handshakes (line 122-127) so
`self._client_id` is replaced — good. But `_read_loop` keeps using `self._client_id`
in `heartbeat_msg` (line 201) without verifying it's still valid. If Avanza
expires the clientId mid-session (their CometD docs allow it after ~5 min of
silence), every heartbeat is rejected and only the *next* full reconnect cycle
recovers. The reconnect backoff at line 137 (`_MIN_BACKOFF * 2 → 60s max`)
means a missed-heartbeat → forced disconnect can yield minute-scale gaps in
streaming quotes.

### `portfolio/avanza/types.py:67-73` — `Quote.from_api` `spread` computed from possibly-None bid/ask

```python
bid = _val(raw.get("buy"), _val(raw.get("bid"), 0.0))
ask = _val(raw.get("sell"), _val(raw.get("ask"), 0.0))
last = _val(raw.get("last"), _val(raw.get("latest"), 0.0))
spread = _val(raw.get("spread"))
if spread is None:
    spread = round(ask - bid, 6) if (ask and bid) else 0.0
```

If `bid=0` (truthy-falsy gotcha — `0.0` is falsy), `spread` is set to `0.0`
silently. A caller using `if quote.spread > MAX_SPREAD: skip` will not skip,
will execute against a quote with no bid, will get filled at whatever sits
on the book. Affects every trading decision that consults a `Quote` from the
unified package. Bid=0 means "no bid" — should propagate as `None` or raise.

### `portfolio/avanza_session.py:710` — silent-success on partial position data

```python
"profit": val - acq if val and acq else 0,
"profit_percent": ((val - acq) / acq * 100) if acq else 0,
```

Truthy guard on `val` and `acq` (each `0.0` is falsy). A legitimate `acq=0`
(brand-new free position, e.g. via corporate action) silently reports profit=0
even when value is non-zero. Same `0.0`-is-falsy trap. Position reporting
isn't trade-critical but feeds the daily P&L digest.

### `portfolio/avanza_session.py:633-645` — `cancel_order` does NOT validate `account_id` against ALLOWED_ACCOUNT_IDS

```python
def cancel_order(order_id: str, account_id: str | None = None) -> dict:
    payload = {
        "accountId": str(account_id or DEFAULT_ACCOUNT_ID),
        "orderId": str(order_id),
    }
```

`place_buy_order` / `place_sell_order` / `place_stop_loss` all guard
`effective_account_id not in ALLOWED_ACCOUNT_IDS`. `cancel_order` does not.
Cancelling pension-account orders by mistake isn't directly money-losing, but
the "whitelist applies to every mutation" principle is silently broken; in the
worst case a caller mis-routing IDs gets a 200 OK on a foreign account's order
cancel without any guard firing.

## Did NOT find

1. **Silent failures**: Found several (P0: get_buying_power, P1: get_open_orders, get_stop_losses inconsistency). One pattern that *was* correctly handled: `cancel_all_stop_losses_for` fails closed on read errors per its docstring (line 950-953).
2. **Race conditions**: Looked. RLock + `avanza_order_lock` filelock + `_with_browser_recovery` form a reasonable layered defense; the explicit comment at line 50-54 documents the prior race. Could not find a residual race.
3. **Money-losing bugs**: Found P0 (unified package missing MAX_ORDER + whitelist; missing stop-loss size cap; trail_percent unvalidated). Stop-loss endpoint `/_api/trading/stoploss/new` is correctly used (`avanza_session.py:801`).
4. **State corruption**: `_save_pending` uses `atomic_write_json` (line 79). Offset file uses `atomic_write_json` (line 343). No raw `json.loads(open(...).read())` paths in the diff. Looks clean.
5. **Logic errors that pass tests**: Suspect the `_HEX_TOKEN_RE` / `_CONFIRM_PREFIX_RE` design is over-engineered against tests that mock the very thing they test, but couldn't confirm without reading the tests (out of scope). Did NOT find a smoking gun.
6. **Resource leaks**: `close_playwright` properly tears down context/browser/instance; `ResilientPage._close_quietly` closes context+browser. Streaming `stop()` closes WebSocket. Did not check for leaks under exception-during-launch (a partial `_open` could leave `_browser` non-None and `_ctx` None; teardown handles that).
7. **Time/timezone bugs**: `datetime.now(UTC)` used consistently in `avanza_session.py` / `avanza_orders.py` / `avanza_account_check.py`. `date.today()` is naive but used for ISO date strings (Avanza expects local date), which is the correct semantic. The `expires_at` parse uses `datetime.fromisoformat` which preserves tzinfo if present.
8. **API misuse**: Stop-loss endpoint correct. POST verb for cancel-order (per API change comment line 636) is documented. Did NOT find Binance-`10m` style mistakes in this subsystem.
9. **Trust boundary violations**: `orderbookId` flows into URL paths in `api_get(f"/_api/market-guide/{instrument_type}/{orderbook_id}")` (line 1241) and `/_api/trading/stoploss/{acct}/{stop_id}` (line 916). Cast to `str(...)` but no escaping. Avanza orderbook IDs are numeric in practice; a malformed alphanumeric ID gets passed through. Not exploitable in this codebase but flagged at P2-ish severity in `streaming.py`.
10. **Incorrect assumptions about partial state**: `get_buying_power` handles three response shapes and four ID field names defensively. `Position.from_api` uses `.get()` chains throughout. Decent.
