# Avanza API Subsystem — Adversarial Review (2026-05-23)

Scope: portfolio/avanza/{__init__,account,auth,client,market_data,scanner,search,
streaming,tick_rules,trading,types}.py and portfolio/avanza_{session,orders,client,
account_check,control,order_lock,resilient_page,tracker}.py.

Cross-referenced: data/metals_avanza_helpers.py (alternate page-based order path).

Confidence convention: P0 = will lose money / corrupt state on the next bad day,
P1 = will lose money under a specific (plausible) race or shape drift, P2 =
fragility / silent degradation, P3 = nits.

------------------------------------------------------------------------------

## P0 — Critical

### P0-1 Unified `portfolio/avanza/trading.py::place_order` has no account whitelist, no max-order-size cap, and no cross-process order lock

**File:** `portfolio/avanza/trading.py:38-102`

The BankID-session order path enforces three explicit guards (see
`avanza_session.py:589-606`):

* `effective_account_id not in ALLOWED_ACCOUNT_IDS` → ValueError
* `order_total > MAX_ORDER_TOTAL_SEK (50_000)` → ValueError (BUG-211)
* `with avanza_order_lock(op="place_order/...")` → cross-process lock

The newer unified package path (`portfolio/avanza/trading.place_order`) has
NONE of these. It only enforces volume>=1, price>0, and a 1000 SEK minimum.
The function will happily route to any `account_id` the caller passes,
including the pension account `2674244`, and will happily submit a 500K SEK
order if the LLM hallucinates a volume.

The CLAUDE.md / `memory/feedback_isk_only.md` invariant ("ISK account 1625505
ONLY. NEVER touch pension account 2674244") is enforced ONLY in the
session/legacy paths. `AvanzaClient.account_id` defaults to "1625505" but is
overrideable via config and via every public `*_order` `account_id=` kwarg.
The `__init__.py` re-exports `place_order`, `place_stop_loss`,
`place_trailing_stop`, `modify_order`, `cancel_order` — any future caller
importing from `portfolio.avanza` bypasses every guard.

**Fix:** Replicate `ALLOWED_ACCOUNT_IDS`, `MAX_ORDER_TOTAL_SEK`, and
`avanza_order_lock` wrapping in `portfolio/avanza/trading.py:place_order`,
`modify_order`, `cancel_order`, `place_stop_loss`, `place_trailing_stop`,
`delete_stop_loss`. Centralize the whitelist (a single
`portfolio/avanza/_guards.py` module imported by both paths) so the next code
fork can't drop a guard silently.

### P0-2 `place_trailing_stop` packs the trail percentage into `trigger_price` AND sends `sell_price=0` — accepted by the non-trailing guard only by accident

**File:** `portfolio/avanza/trading.py:291-319` and `portfolio/avanza_session.py:814-846`

Both `place_trailing_stop` wrappers forward `trail_percent` as
`trigger_price` and `sell_price=0`, then call `place_stop_loss(..., 
trigger_type="FOLLOW_DOWNWARDS", value_type="PERCENTAGE")`. The
session-path's `BUG-223` guard reads:

```
if trigger_type not in _TRAILING_TYPES and value_type == "MONETARY" and sell_price <= 0:
    raise ValueError(...)
```

If a future refactor flips the default `value_type` of the trailing wrapper
from `PERCENTAGE` to `MONETARY` (an easy LLM/copy-paste mistake — the
parameter is named `trail_percent` but the unit lives in `value_type`), the
guard fires only for non-trailing. A trailing call with `MONETARY` and
`sell_price=0` would be a market-sell at the worst available price — exactly
the Mar 3 incident the stoploss API was supposed to prevent. The current
code is correct but the API design encourages this exact mistake.

Worse, the unified `portfolio/avanza/trading.py` path has NO equivalent
BUG-223 guard at all (line 213-288). A bad `place_stop_loss` call with
`MONETARY`+`sell_price=0`+non-trailing trigger goes straight to the broker.

**Fix:** Add the BUG-223 guard to `portfolio/avanza/trading.place_stop_loss`.
Reject `sell_price <= 0` unless `trigger_type in {FOLLOW_DOWNWARDS,
FOLLOW_UPWARDS}` AND `value_type == "PERCENTAGE"` (both conditions, not
just trigger_type — that's the actual semantic).

### P0-3 `portfolio/avanza/account.py::get_buying_power` returns silent zeroes on account-not-found instead of `None`

**File:** `portfolio/avanza/account.py:87-94`

```python
for account in accounts:
    if str(account.get("accountId", account.get("id", ""))) == acct:
        return AccountCash.from_api(account)
logger.warning("get_buying_power account_id=%s not found in overview", acct)
return AccountCash(buying_power=0.0, total_value=0.0, own_capital=0.0)
```

This is the bug that `avanza_session.get_buying_power` (lines 385-539) was
explicitly fixed for — see its docstring's "dangerous silent failure"
paragraph and 2026-04-09 Bug C7. A caller doing
`if cash.buying_power < order_total:` will see "no money, skip the order"
which is the safe direction; a caller doing
`size = cash.buying_power * 0.95` proceeds with size=0 (harmless); but a
caller doing `available = cash.total_value - locked` will see total_value=0
and conclude "I have no positions, free to allocate elsewhere" — silently
wrong on the new flat shape that exposes only `id` (not `accountId`).

Also, this function tries only `accountId` then `id`, but
`avanza_session.get_buying_power` tries `("accountId", "id", "accountNumber",
"number")`. Same Avanza response, two ID-field strategies in two files.

**Fix:** Return `None` on miss (matching the session path), and try all four
known ID fields (matching the session path). Better yet, share the lookup
helper.

### P0-4 `portfolio/avanza/account.py::get_positions` returns ALL positions when `account_id=None`, with no whitelist filter

**File:** `portfolio/avanza/account.py:27-61`

```python
def get_positions(account_id: str | None = None) -> list[Position]:
    ...
    if account_id is not None:
        positions = [p for p in positions if p.account_id == str(account_id)]
    return positions
```

Calling `get_positions()` with the default returns every position across
every account the BankID session can see — explicitly including the pension
account `2674244`. Compare `portfolio/avanza_client.get_positions` which
hard-filters to `ALLOWED_ACCOUNT_IDS = {"1625505"}` before returning. Any
new dashboard/report code path that calls `from portfolio.avanza import
get_positions` will leak pension holdings into the trading view, breaking
the `memory/feedback_isk_only.md` invariant.

The same applies to `Position.from_api` which captures `account_id` but
does NOT enforce a whitelist anywhere in the typed pipeline.

**Fix:** Default `get_positions()` to filter to a module-level
`ALLOWED_ACCOUNT_IDS` set; require an explicit
`get_positions(account_id="all")` (or boolean kwarg) to bypass. Same for
`get_transactions`.

### P0-5 BankID `verify_session` returns False on a 5xx and the auto-recovery loop tears down the *valid* session

**File:** `portfolio/avanza_session.py:180-196`

```python
def verify_session() -> bool:
    try:
        with _pw_lock:
            ctx = _get_playwright_context()
            resp = ctx.request.get(f"{API_BASE}/_api/position-data/positions")
            return resp.ok
    except Exception as e:
        logger.warning("Session verification failed: %s", e)
        close_playwright()
        return False
```

`resp.ok` is False for 4xx AND 5xx. A 502/503/504 from Avanza calls
`close_playwright()` on the next exception path and forces a full browser
relaunch on next API call, even though the cookies are still valid. Combined
with `_try_session_auth` in `avanza_client.py` that caches `_session_client
= True`, a transient gateway error during startup will silently flip the
client to the TOTP fallback path for the rest of the process — TOTP doesn't
share the same `account_id`, `ALLOWED_ACCOUNT_IDS` enforcement (see P0-1).

**Fix:** `verify_session` should only return False on auth failures (401/403)
and on actual exceptions. A 5xx should be retried at most once, then propagate
"unknown" — not "definitely dead." Don't `close_playwright()` on 5xx.

------------------------------------------------------------------------------

## P1 — Important

### P1-1 `_get_csrf(ctx)` linear scan + stale cookie risk on relaunch

**File:** `portfolio/avanza_session.py:271-291`

After `_with_browser_recovery` relaunches the browser the freshly-loaded
`storage_state.json` may not contain a current `AZACSRF` cookie until the
first page navigation triggers the cookie refresh — Avanza rotates CSRF on
session resume. The retry path inside `_with_browser_recovery` for POST/DELETE
reads CSRF from the *just-relaunched* ctx, so if the cookie isn't seeded the
function raises `AvanzaSessionError("No AZACSRF cookie found")` and the
caller sees a hard error mid-order. The order *might* still be in flight
broker-side because the recovery happens AFTER the request started.

**Fix:** After relaunch, perform a lightweight authenticated GET (e.g.
`/_api/position-data/positions`) to force cookie rotation before the next
mutating call, then re-read CSRF.

### P1-2 `cancel_all_stop_losses_for` filters by `orderbook.id` only — ignores `instrumentId` shape

**File:** `portfolio/avanza_session.py:984-997`

```python
ob = (sl.get("orderbook") or {}).get("id")
if str(ob) != target_ob:
    continue
```

Newer Avanza stoploss responses expose `orderBookId` directly on the SL
dict (no `orderbook` wrapper) — see the schema drift handling in
`portfolio/avanza/types.py:StopLoss.from_api` (lines 333-345) which tries
`orderbook.id`, `orderBookId`, and `orderbookId`. If Avanza ships the flat
shape, `_filter_for_ob` matches nothing, returns the orderbook as "no
stops, success" — and a follow-up SELL crashes with
`short.sell.not.allowed` because the encumbered stops are still live.

**Fix:** Try all three shapes in `_filter_for_ob`, mirroring
`StopLoss.from_api`'s precedence.

### P1-3 `rearm_stop_losses_from_snapshot` re-uses ORIGINAL `valid_until` so re-armed stops can have 0 valid_days

**File:** `portfolio/avanza_session.py:1171-1181`

```python
parsed = datetime.strptime(valid_until, "%Y-%m-%d").date()
delta = (parsed - today_iso).days
if delta > 0:
    valid_days = delta
```

If the snapshot was taken at e.g. 23:50 on an 8-day SL's last valid day,
`delta` is 0 or negative, the `if delta > 0:` branch doesn't update, and the
fallback `valid_days = 8` is used — fine. But if the snapshot's
`validUntil` parses to today (`delta == 0`), the explicit `if > 0` keeps
the 8-day default, so the actual broker-side validity becomes 8 days from
today, NOT the original expiry. Subtle but consistent — re-arming an SL
that was supposed to die at end-of-day extends it. For a cancel-then-sell
rollback that's likely benign, but for any caller that snapshots SLs to
preserve their natural expiry it's a footgun. Also, `strptime` errors are
caught silently, falling back to 8 days — again, extends validity beyond
the original.

**Fix:** Either preserve the literal `validUntil` string and pass it through
the new place call (Avanza's API accepts an explicit date), or document
loudly that re-armed stops always run 8 days regardless of the snapshot.

### P1-4 `avanza_orders.py::_check_telegram_confirm` ignores `from.is_bot`

**File:** `portfolio/avanza_orders.py:300-311`

```python
if allowed_user is not None:
    sender = msg.get("from") or {}
    sender_id = sender.get("id")
    if sender_id is None or str(sender_id) != allowed_user:
        ... drop ...
```

A user can be impersonated by a Telegram bot in a group chat if the bot has
the right user_id (extremely rare but possible if `allowed_user` ever equals
a bot's id — bot ids are stable). More importantly, `chat.id == chat_id`
+ `from.id == allowed_user` does NOT protect against a Telegram channel
where the configured chat is a *channel* (Telegram channels report
`from.id` of the channel admin, not the original poster). For a
single-user DM chat the check is sound, but the design implies group/channel
support.

**Fix:** Require `from.is_bot is False` if `allowed_user` is set; document
that `allowed_user_id` is only safe for direct chats.

### P1-5 `place_order` confirms a token-less LEGACY order via bare CONFIRM forever

**File:** `portfolio/avanza_orders.py:200-217`

The migration design (P1-10) accepts bare `CONFIRM` ONLY for orders without
`confirm_token`. The intent is "legacy in-flight orders." But there's no
sweep that retires the bare-CONFIRM path after the legacy queue drains —
any future code path that creates an order without a token (a test mock
leaked into prod, a hand-edit of `avanza_pending_orders.json` for debugging)
will silently fall into the legacy path and be confirmable by bare CONFIRM.

The legacy path also doesn't enforce sender auth — wait, it does (the
`allowed_user` check is before token parsing). OK that's safer than I
thought. Still — the bare-CONFIRM legacy path should be feature-flagged
off after a known cutover date so nobody silently regresses.

**Fix:** Add a `_LEGACY_CONFIRM_CUTOFF` constant; reject bare CONFIRM if
the matching order's `timestamp` is later than the cutoff.

### P1-6 `avanza_session._place_order`'s `MAX_ORDER_TOTAL_SEK = 50_000` is per-function, not per-account

**File:** `portfolio/avanza_session.py:599-606`

50K SEK is the per-call cap. Nothing prevents the LLM from firing five
50K orders in a row (separate Layer 2 invocations, separate
`avanza_order_lock` cycles) and burning through 250K SEK in seconds. The
cross-process lock is for racing on `buying_power` snapshot, not for
rate-limiting total daily spend.

**Fix:** Add a daily-spend tracker (rolling 24h sum across all orders
placed, persisted to `data/avanza_daily_spend.json`), refuse new orders
when the day-total exceeds a configured cap (e.g. 100K SEK / day). Same
file should also count cancelled-and-replaced orders (so flip-flopping
doesn't reset the budget).

### P1-7 Streaming reconnect loses subscription state when `clientId` changes

**File:** `portfolio/avanza/streaming.py:118-127`

```python
while not self._stop_event.is_set():
    try:
        self._connect()
        self._do_handshake()
        for channel in self._callbacks:
            self._subscribe_channel(channel)
        ...
```

On reconnect, callbacks are re-subscribed but any IN-FLIGHT messages between
disconnect and reconnect are LOST. For a quote stream that's fine; for the
`/orders/` and `/deals/` channels it means a partial-fill notification that
landed during reconnect goes missing. The system has no compensating
mechanism — `get_orders()` polling is not invoked after a reconnect to
reconcile state.

**Fix:** After every successful reconnect to the `/orders/` or `/deals/`
channel, do a one-shot `get_orders()` / `get_deals()` REST poll to
reconcile in-flight state. Document the loss-window explicitly.

### P1-8 Tick rules: `_decimal_places` mishandles scientific notation, integer ticks

**File:** `portfolio/avanza/tick_rules.py:131-135`

```python
s = f"{value:.10f}".rstrip("0")
if "." in s:
    return len(s.split(".")[1])
return 0
```

For `tick_size=0.0001`: `f"{0.0001:.10f}"` = `"0.0001000000"`, rstrip → 
`"0.0001"`, returns 4. Correct.

For `tick_size=1.0` (whole-krona ticks for high-priced instruments):
`"1.0000000000"`, rstrip → `"1."`, length of `s.split(".")[1]` = 0,
returns 0. Then `multiplier=1`, `tick_int=1`, fine.

But for `tick_size=0.00001` (5 decimal places): the rstrip path keeps
trailing zeros stripped to `"0.00001"`, returns 5. multiplier = 100_000.
`price_int = 0.12345 * 100000 = 12344.999999...` due to FP, `floor(12344.999/1) 
= 12344`. That's the floor of 12345, off by one tick. The "integer
arithmetic" comment is misleading — the multiplier is float-multiplied,
not exact.

**Fix:** Use `decimal.Decimal` for the multiply step, or compute via
`round(price / tick) * tick` with proper `quantize`. The current code
will silently round prices down by one tick on certain edge values,
producing rejected limit orders below the visible bid.

### P1-9 `get_open_orders` fallback path silently mis-types

**File:** `portfolio/avanza_session.py:648-664`

```python
try:
    data = api_get(f"/_api/trading/rest/order/account/{aid}")
    if isinstance(data, list):
        return data
    return data.get("orders", data.get("openOrders", []))
except RuntimeError:
    try:
        data = api_get("/_api/trading/rest/deals-and-orders")
        orders = data.get("orders", [])
        ...
```

The fallback `deals-and-orders` endpoint returns BOTH open orders AND
recent deals. Treating `data.get("orders")` as "open orders" works only
if the API returns just open ones — which historically has included
filled/cancelled orders mixed in. A caller using `get_open_orders` to
decide whether to place a duplicate could double-submit on a recently
filled order.

**Fix:** Filter the fallback by `status in {"PENDING", "OPEN",
"AT_MARKET", ...}` before returning.

------------------------------------------------------------------------------

## P2 — Fragility / silent degradation

### P2-1 `Playwright sync_api` from threads — `_pw_lock` doesn't fix everything

**File:** `portfolio/avanza_session.py:54-153, 199-232`

`sync_playwright` is documented as not thread-safe even with external
locking when used outside its starting thread. The `_pw_lock = RLock()`
serializes calls, but `sync_playwright().start()` was called from
whichever thread won the first `_get_playwright_context()` race. If a
subsequent call from a different thread holds the lock and calls
`ctx.request.get(...)`, Playwright may still raise
`greenlet.error: cannot switch to a different thread` in some sync_api
versions. The comment claims this is fixed by RLock; reality depends on
Playwright internals. The MetalsLoop GPU pattern uses a dedicated
executor for exactly this reason (see `avanza_account_check.py:154-177`).

**Fix:** Run all sync_playwright calls through a single dedicated
worker thread (like `avanza_account_check._api_get_categorized_accounts`
already does). Or migrate to `async_playwright`.

### P2-2 `_with_browser_recovery` retries POST after relaunch — order may already have landed

**File:** `portfolio/avanza_session.py:212-232`

```python
try:
    return op(ctx)
except Exception as exc:
    if not is_browser_dead_error(exc):
        raise
    ...
    return op(ctx)  # retries the POST
```

For POST `/_api/trading-critical/rest/order/new`, retrying after a
browser-dead error is NOT idempotent. If the browser died after sending
the request but before reading the response, Avanza may have accepted the
order — the retry submits a SECOND identical order. The `confirm_token`
in `avanza_orders.py` does not propagate to the payload (Avanza has no
idempotency key in the public API).

**Fix:** For mutating endpoints (`order/new`, `order/delete`,
`stoploss/new`, `stoploss/{acct}/{id}` DELETE), do NOT auto-retry on
browser death. Surface the failure to the caller and let `get_open_orders`
reconcile post-hoc before retry.

### P2-3 `avanza_session.cancel_order` uses POST `/order/delete` — not idempotent for double-fire

**File:** `portfolio/avanza_session.py:633-645`

A POST that doesn't return until processed has no idempotency on transient
network errors. With the `avanza_order_lock` wrapper, a concurrent
intra-process retry is prevented, but an HTTP retry inside `fetch_with_retry`
(used by the underlying Playwright fetch — actually no, `api_post` doesn't
use it). Confirmed: no auto-retry, so this is OK. Downgrade: also worth
noting that the response shape isn't checked for "already cancelled" —
calling cancel twice intentionally would log a SECOND warning even though
the first cancel succeeded.

**Fix:** Treat `orderRequestStatus == "ORDER_NOT_FOUND"` as success.

### P2-4 `avanza/auth.py` singleton has no expiry detection

**File:** `portfolio/avanza/auth.py:74-114`

The TOTP-authenticated `Avanza` instance is cached forever (until
`reset()`). The underlying `avanza-api` library refreshes its session via
the requests `Session` but there's no detection of "session expired —
re-auth needed" here. If TOTP authentication drops mid-day, every call
fails until the process restarts — no automated reset, no Telegram alert.
Compare `avanza_session.py` which at least detects 401/403 and calls
`close_playwright()` on the BankID path.

**Fix:** Wrap `client.avanza.*` calls in a 401-detector that calls
`AvanzaAuth.reset()` and re-auths once.

### P2-5 `avanza_order_lock` lock-file is not cleaned up — Windows lingering filelock

**File:** `portfolio/avanza_order_lock.py:80-100`

`filelock` on Windows leaves `data/avanza_order.lock` on disk after release
(the file IS released but the inode persists). This is normally fine, but
combined with the `RLock` semantics: a Python process that crashes mid-lock
on Windows can leave the lock held until the OS releases the file handle,
which can outlive the process for seconds-minutes during heavy I/O. A 2s
timeout means the next loop tick after a crash sees `OrderLockBusyError`
for the entire window. The fail-fast behavior is correct — the loop
retries — but the diagnostic ("which loop hit the busy lock") prints
nothing because the holder is dead.

**Fix:** Write the holder PID + op + timestamp into the lock file's
adjacent `.meta.json` on acquire; on `OrderLockBusyError`, read the meta
and log "lock probably held by dead PID 1234 (op=...) — investigate."
Also consider a stale-lock breaker (if meta is older than 10s, force-break).

### P2-6 `scanner.py::fetch_detail` swallows MarketData errors silently — sets `mm=False, bid_vol=0`

**File:** `portfolio/avanza/scanner.py:265-279`

```python
with suppress(Exception):
    md = marketdata_fn(ob_id)
    ...
```

A scan that can't reach marketdata returns ranked results with `mm=False,
bid_vol=0, ask_vol=0` for every instrument. Callers that filter by "must
have market maker" silently get an empty result; callers that rank by
`bid_volume` get garbage rankings. The `with suppress(Exception)` should
at least set a `marketdata_ok` flag the caller can see.

**Fix:** Return `marketdata_ok: bool` in `ScannedInstrument` so callers
can distinguish "no MM" from "couldn't check."

### P2-7 `account_check.verify_default_account` cache survives shape drift

**File:** `portfolio/avanza_account_check.py:215-219, 318-320`

Once verified, the result is cached for the process lifetime. If Avanza
ships a shape change mid-process that re-classifies the account into a
new category (or worse, drops it from the response entirely), the
verifier never re-checks. The `use_cache=False` opt-out is not invoked
anywhere I can see by the loops.

**Fix:** Cache with a TTL (e.g. 1 hour). Re-verify hourly.

### P2-8 Pending orders file has no per-process exclusivity

**File:** `portfolio/avanza_orders.py:68-79`

`_load_pending` + `_save_pending` use `load_json` / `atomic_write_json`,
but there's no file lock around the read-modify-write cycle in
`request_order` and `check_pending_orders`. Two processes (main loop +
metals loop + golddigger) all calling `request_order` simultaneously
would last-writer-wins, dropping a pending order. The atomic write is
atomic at the OS level, but the read-modify-write isn't atomic at the
business level.

**Fix:** Acquire `avanza_order_lock` (or a dedicated `pending_orders.lock`)
around the load+append+save sequence in `request_order` and the for-loop
in `check_pending_orders`.

------------------------------------------------------------------------------

## P3 — Nits

### P3-1 `avanza_orders.py::_CONFIRM_TOKEN_HEX_CHARS = 6` is documented as 24 bits but `secrets.token_hex(3)` returns 6 hex chars = 24 bits — matches, but the integer-division `// 2` makes the relationship opaque.

`secrets.token_hex(_CONFIRM_TOKEN_HEX_CHARS // 2)` → `secrets.token_hex(3)`
→ 6 hex chars. If anyone changes the constant to 7 (odd), `// 2` floors to
3 and you still get 6 chars. **Fix:** Use `secrets.token_hex(3)` directly
and rename the constant to `_CONFIRM_TOKEN_BYTES = 3`.

### P3-2 `Quote.from_api` defaults to 0.0 for missing `bid`/`ask`, masking "no quote yet" vs "quote is 0"

**File:** `portfolio/avanza/types.py:65-88`

A freshly-launched instrument with no trades returns `bid=0.0, ask=0.0` —
identical to a stale snapshot. Callers computing `spread_pct = (ask-bid)/bid`
divide by zero and may catch+default to "no spread, accept." Suggest using
`Optional[float]` and returning None on missing.

### P3-3 `avanza/types.py::OrderResult` swallows `message` lists into a "; "-joined string

**File:** `portfolio/avanza/types.py:184-197`

```python
if isinstance(message, list):
    message = "; ".join(str(m) for m in message)
```

Lossy. Avanza returns structured error objects; flattening them to a
string loses error codes. Surface the raw list under `OrderResult.messages`.

### P3-4 Two functions named `cancel_order` across modules

`portfolio/avanza/trading.py:cancel_order` and
`portfolio/avanza_session.py:cancel_order` and
`portfolio/avanza_client.py:delete_order` all do the same thing through
different code paths. The naming inconsistency (`cancel` vs `delete`)
creates copy-paste hazards.

### P3-5 `place_stop_loss` defaults `trigger_type="LESS_OR_EQUAL"` everywhere — buy-side stops not possible

Cancellation of bear-side warrant fills via a stop on a SHORT position
would require `MORE_OR_EQUAL`. Nothing wrong with the default but every
caller has to know to override. Document loudly in
`memory/grudges.md` if not already there.

------------------------------------------------------------------------------

## Five-line summary

1. **Critical**: the new `portfolio/avanza/trading.py` package lacks the account whitelist, max-order-size cap, and cross-process lock that the legacy `avanza_session.py` enforces — re-exported via `__init__.py`, one stray import can bypass every guard the ISK-only invariant depends on.
2. **Critical**: `portfolio/avanza/account.py::get_buying_power` silently returns zeroes on account-not-found (regression of 2026-04-09 Bug C7), and `get_positions(None)` leaks the pension account.
3. **Important**: stop-loss cancel verification only matches the old `orderbook.id` shape, ignoring `orderBookId`/`orderbookId` — Avanza shape drift would let a follow-up sell hit `short.sell.not.allowed`. Browser-dead retry in `_with_browser_recovery` blindly re-submits POST to order/stoploss endpoints — likely duplicate orders on real network blips.
4. **Important**: per-order `MAX_ORDER_TOTAL_SEK=50_000` is a per-call cap, not a daily budget; tick rounding's "integer arithmetic" is float-multiplication and drops by one tick on some 5-decimal instruments; streaming reconnect loses in-flight `/orders/` and `/deals/` messages with no reconciliation poll.
5. **Hardening direction**: centralize whitelist + size cap + lock in a single `portfolio/avanza/_guards.py` imported by all four order paths (session/client/control/trading); add daily-spend budget; reconcile stop-loss listing fields against `StopLoss.from_api`'s precedence; never auto-retry mutating POSTs after browser death.
