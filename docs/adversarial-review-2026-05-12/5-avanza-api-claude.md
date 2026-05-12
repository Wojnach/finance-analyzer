# Claude adversarial review: avanza-api (2026-05-12)

Scope: `portfolio/avanza_session.py`, `portfolio/avanza_client.py`,
`portfolio/avanza_orders.py`, `portfolio/avanza_tracker.py`,
`portfolio/avanza_control.py`, `portfolio/avanza_account_check.py`,
`portfolio/avanza_order_lock.py`, `portfolio/avanza_resilient_page.py`,
and `portfolio/avanza/` (auth/client/account/market_data/search/scanner/streaming/trading/tick_rules/types).

## Summary

The Avanza subsystem has *three parallel* trading code paths — the
BankID-session path (`avanza_session.py`), the legacy TOTP path
(`avanza_client.py`), and the unified package (`portfolio/avanza/`) —
and each has its own account-whitelist, its own lock-acquisition
discipline, and its own stop-loss endpoint usage. Most of the
flagged-and-fixed historical incidents (Mar 3 stop-loss-via-regular-API,
A-AV-2 pension leak, BUG-129 Playwright concurrency, BUG-128 offset
corruption, P0-4 TOTP-vs-page race) have been patched in *the path
that experienced them*, but the patches did not propagate to the other
two paths, leaving multiple shadows of the same bug live.

Concrete blockers found:

* `portfolio/avanza/trading.py` places stop-losses with no account
  whitelist guard and no `avanza_order_lock` — the unified-package
  path is wide open while the legacy paths are guarded.
* `portfolio/avanza/account.py:get_buying_power` is **unfiltered** by
  the whitelist and trusts whatever account id the caller passes — a
  Layer-2 call with `account_id=None` defaults to whatever the
  AvanzaClient singleton was constructed with, which can be poisoned
  through `config["avanza"]["account_id"]`.
* `portfolio/avanza/trading.py:cancel_order` and `cancel_stop_loss`
  are **not lock-guarded** — concurrent cancels can race against
  `place_stop_loss` and produce the over-encumbered / naked-position
  failure mode the lock exists to prevent.
* `avanza_session.cancel_order` routes through
  `/_api/trading-critical/rest/order/delete` (correct), but its
  result-shape check on line 622 (`result.get("orderRequestStatus")`)
  trusts a successful 200 response — `api_post` does NOT reject the
  call when the JSON body indicates a business-level failure, so
  callers who only look at `orderRequestStatus` get clean SUCCESS
  signals for failed cancels (the Avanza API sometimes returns 200
  with `"orderRequestStatus": "ERROR"`).

Several P1/P2 issues center on the read-side: `get_open_orders` and
`get_stop_losses` (loose) swallow `RuntimeError` and return `[]`,
which from a safety-critical caller's standpoint is indistinguishable
from "no orders exist" — the exact pattern that produced the
`short.sell.not.allowed` fault `get_stop_losses_strict` was added to
prevent. The "strict" variant exists but most read sites still call
the lossy one.

Finally: the storage-state file (`avanza_storage_state.json`) is
referenced by both `avanza_session.py:28` and `avanza_resilient_page.py`
but is **never written atomically** by any code in scope — it's only
written by `scripts/avanza_login.py` (out of scope) via Playwright's
`context.storage_state(path=...)`. A mid-write crash here corrupts the
auth file and the next loop iteration fails closed, but with no
recovery path other than human BankID re-login.

## P0 — Blockers

### P0-1 Unified-package `place_stop_loss` has NO account-whitelist guard
`portfolio/avanza/trading.py:213-288`

`avanza_session.place_stop_loss:749-751` enforces:
```
if acct not in ALLOWED_ACCOUNT_IDS:
    raise ValueError("Refusing to place stop-loss on non-whitelisted account ...")
```
`avanza_client._place_order:344-358` resolves through `get_account_id()`
which enforces the whitelist via `ALLOWED_ACCOUNT_IDS` at
`avanza_client.py:32, 270-279`. But the unified-package path at
`portfolio/avanza/trading.py:240`:
```
acct = account_id or client.account_id
```
**takes any caller-provided `account_id` at face value**, including
the pension account `2674244`. `AvanzaClient` builds `account_id` from
`config["avanza"]["account_id"]` (`portfolio/avanza/client.py:65`)
with no whitelist check, and `place_stop_loss(account_id="2674244",...)`
will route a stop directly at the pension account. Same hole in
`place_order` (`trading.py:81`), `modify_order` (`trading.py:129`),
`cancel_order` (`trading.py:160`), `delete_stop_loss` (`trading.py:352`),
and `place_trailing_stop` (which calls `place_stop_loss`).

Fix: hardcode `ALLOWED_ACCOUNT_IDS` at the top of
`portfolio/avanza/trading.py` (or import from a shared constants
module that all three paths use) and gate every order/SL/cancel.

### P0-2 Unified-package trading has NO `avanza_order_lock` integration
`portfolio/avanza/trading.py:38-365`

Every order/SL/cancel function in `portfolio/avanza/trading.py` calls
`client.avanza.place_order(...)` / `place_stop_loss_order(...)` /
`delete_order(...)` / `delete_stop_loss_order(...)` **directly with
no `avanza_order_lock` wrapper**. Contrast with the legacy paths,
which all wrap these:
* `avanza_session.py:620, 644, 800, 915` (BankID path)
* `avanza_client.py:358, 396` (legacy TOTP path)
* `avanza_control.py:174, 222` (page path)
* `data/metals_avanza_helpers.place_order` (page path, per
  `avanza_client.py:330-343` docstring)

If any caller in the codebase migrates from the legacy paths to the
unified package and the unified-package code runs alongside the
metals loop, the cross-process lock that has been carefully kept in
place since 2026-04-13 is bypassed. The "two callers observing the
same buying_power" race the lock exists to prevent comes right back.

Fix: every function in `portfolio/avanza/trading.py` that mutates
state (place_order, modify_order, cancel_order, place_stop_loss,
place_trailing_stop, delete_stop_loss) must wrap the underlying call
in `with avanza_order_lock(op=...)`.

### P0-3 `avanza_session.cancel_order` masks broker-side ERROR in 200 responses
`portfolio/avanza_session.py:633-645`

The cancel flow returns whatever `api_post` returns. `api_post`
(`avanza_session.py:294-344`) treats anything 200-299 as success and
returns the parsed body. The Avanza trading-critical delete endpoint
is known to return HTTP 200 with body
`{"orderRequestStatus": "ERROR", "message": "..."}` for business
rejections (stale order id, account mismatch, market closed). The
caller path in `avanza_orders._execute_confirmed_order:365-389` reads
`result.get("orderRequestStatus", "UNKNOWN")` — but the **cancel**
caller path (`portfolio/avanza_control.py:delete_order_no_page:384`)
treats `orderRequestStatus != SUCCESS` as `ok=False` correctly. The
issue is in the *callers that don't check*: e.g. `cancel_order` in
`avanza_session.py` returns the bare dict, and consumers in
`grid_fisher` / `metals_loop` only sometimes look at the field.

This becomes a P0 in combination with the "cancel-before-sell" safety
flow (`feedback_just_do_it.md`, `avanza_session.cancel_all_stop_losses_for`)
because a silently-failed cancel that *looks* successful causes the
follow-up sell to hit `short.sell.not.allowed` — but actually worse
than that: the verify-poll in `cancel_all_stop_losses_for:1074-1085`
filters `cancelled = cancelled - remaining` which depends on the
poll seeing the still-alive stop. If the broker rejected the cancel
with HTTP 200 + body ERROR, the stop is still alive and the verify
loop catches it — but only after the `max_wait` (3s) elapses, during
which the dependent sell may have already fired in another thread.

Fix: `cancel_order` should inspect `orderRequestStatus` and convert
non-SUCCESS into `{"status": "FAILED", ...}` shape, matching what
`cancel_stop_loss` already does for HTTP errors.

### P0-4 `_get_csrf` returns empty-string handling is missing
`portfolio/avanza_session.py:271-291` / `avanza_control.py:167-217`

In `avanza_control.delete_order_live:167-169` and `delete_stop_loss:213-217`:
```python
csrf = get_csrf(page)
if not csrf:
    return False, {"error": "no CSRF token"}
```
But the session-level `_get_csrf` at `avanza_session.py:271-291`
**raises `AvanzaSessionError`** if the cookie is missing. That
exception then escapes through `api_post`/`api_delete` and lands at
the caller as a generic exception — but `cancel_stop_loss:917-919`
catches `Exception` and converts it to `{"status": "FAILED",
"http_status": 0, "stop_id": stop_id, "error": str(exc)}`. The
`AvanzaSessionError` doesn't get the "Run avanza_login" recovery
hint surfaced to Telegram; it gets folded into a generic FAILED with
the error string. The agent loop logs the failure but doesn't escalate.

This is P0 because the failure mode is silent: a missing CSRF means
the session has actually rotated and the next BankID re-auth is
required, but the loop will retry every cycle with the same dead
session, accumulating 401/403s without surfacing the auth-fix path.

Fix: `cancel_stop_loss` and `cancel_all_stop_losses_for` should
re-raise `AvanzaSessionError` (or set a distinguishable status like
`"AUTH_FAILED"`) so the dispatcher can trigger a re-auth Telegram
notification rather than burying it as a generic cancel failure.

## P1 — High

### P1-1 `get_open_orders` and `get_stop_losses` swallow RuntimeError to `[]`
`portfolio/avanza_session.py:648-664, 849-865`

```python
def get_open_orders(...):
    try:
        data = api_get(f"/_api/trading/rest/order/account/{aid}")
        ...
    except RuntimeError:
        # Endpoint may vary — fallback to deal endpoint
        try:
            data = api_get("/_api/trading/rest/deals-and-orders")
            ...
        except RuntimeError:
            logger.warning("Could not fetch open orders")
            return []

def get_stop_losses() -> list[dict]:
    try:
        data = api_get("/_api/trading/stoploss")
        return data if isinstance(data, list) else []
    except RuntimeError:
        logger.warning("Could not fetch stop-losses")
        return []
```

A `RuntimeError` here means "Avanza API returned 4xx/5xx" (see
`api_get:264-265`), not "the account has no orders." Returning `[]`
is indistinguishable from the empty case at every call site. The
strict variant `get_stop_losses_strict` exists for safety-critical
paths but the loose variant is still used wherever "no orders" is
treated equivalent to "could not read" — including the dashboard,
the trade reconciler, and several risk gates. This is the same class
of bug that motivated `get_stop_losses_strict` in the first place;
the loose version should be eliminated, not paralleled.

### P1-2 Unified-package `get_buying_power` returns silent zeros
`portfolio/avanza/account.py:64-94`

```python
# Account not found — return zeroes
logger.warning("get_buying_power account_id=%s not found in overview", acct)
return AccountCash(buying_power=0.0, total_value=0.0, own_capital=0.0)
```

The legacy `avanza_session.get_buying_power:531-539` deliberately
returns **None** on account-not-found so callers can distinguish
"API failure" from "balance is legitimately zero" — and the docstring
explicitly calls out this lesson learned ("previously they silently
got buying_power=0, which was a dangerous silent failure"). The
unified package has *reintroduced* the exact same dangerous silent
failure: a caller using `AccountCash` from the new path has no way
to tell whether the response is real-zero or read-error-zero, and
will size 0 SEK orders (or worse, infer "no holdings, safe to buy").

### P1-3 Account-id check in `avanza_client.get_account_id` uses substring `"ISK"`
`portfolio/avanza/account.py` and `portfolio/avanza_client.py:262-279`

The whitelist short-circuit is correct, but the **discovery** logic
still falls back to scanning `account.accountType for 'ISK' in atype.upper()`
(`avanza_client.py:264-267`). Today only one ISK account is allowed,
so the whitelist filter catches anything else, but if the operator
ever needs to add a sub-account or migrates accounts, the substring
match is fragile — Avanza's account types include `"AKTIE_FOND_KONTO_ISK"`,
`"ISK"`, `"ISKR"` (rente), etc. A category Avanza renames slightly
(say `"PRIVAT_ISK"`) silently drops out of the discovery loop and
the function raises `RuntimeError("No whitelisted ISK account found")`
even though the account is reachable.

Better: drop the type-substring filter entirely and just iterate every
account, checking the whitelist directly. The whitelist already
constrains to `{"1625505"}`.

### P1-4 Pending-order file is *not* multi-process safe under bare CONFIRM
`portfolio/avanza_orders.py:68-79, 151-215`

The bare-CONFIRM legacy compatibility path is documented as "only
matches legacy orders without a token", but `check_pending_orders`
runs in the main loop and the file `data/avanza_pending_orders.json`
is also read+written by `request_order`. There is no advisory lock
around the read-modify-write of `_load_pending()` / `_save_pending()`:
* `request_order` reads pending, appends, writes.
* `check_pending_orders` reads pending, mutates statuses, writes.

If a Layer-2 invocation lands `request_order` *while* the main loop
is in the middle of `check_pending_orders`, the appended order is
silently overwritten when the loop saves. The file is `atomic_write_json`
on the write side (so it's never half-written), but the *logical*
critical section is unprotected. Telegram-confirmed orders for a
just-requested instrument can be lost.

Fix: add a `filelock.FileLock` around `_load_pending → _save_pending`
on both call sites (analogous to `avanza_order_lock`, but a different
lock since order placement and pending-confirmation are separate
operations).

### P1-5 `resilient_page` relaunch counter is unbounded
`portfolio/avanza_resilient_page.py:142-151`

`_relaunch` increments `_relaunch_count` and re-opens the browser
on every TargetClosedError. There is no upper bound, no backoff, no
escalation to "the storage state file is also dead, stop trying."
If the storage state has expired (BankID 24h validity exceeded)
every `new_context(storage_state=...)` succeeds but every
subsequent request immediately 401s, which the recovery wrapper
*does not catch* — `is_browser_dead_error` does not match 401.
Meanwhile `evaluate` retries once on the *first* dead-browser
exception, but on the second one it propagates, so the loop sees
a TargetClosedError every cycle. Add: max relaunches per minute
+ alert to Telegram on threshold, exponential backoff after N
failures.

### P1-6 Stop-loss endpoint correctness — unified path uses library wrapper
`portfolio/avanza/trading.py:272-278`

```python
raw: dict[str, Any] = client.avanza.place_stop_loss_order(
    "0",  # parent_stop_loss_id — "0" for new stop-loss
    acct, ob_id, trigger, order_event,
)
```

The unified-package path delegates to the avanza-api library's
`place_stop_loss_order`. The Mar 3 incident was that the *regular*
order API was used for stops, causing instant-fill at bad prices.
The library's `place_stop_loss_order` is *probably* the right
endpoint, but there is no verification of the underlying HTTP path
in this codebase. If the library is ever upgraded and changes the
endpoint behind the wrapper, this code silently switches without
the Mar 3 guard tripping. Compare against the BankID path where the
exact URL `/_api/trading/stoploss/new` is hardcoded
(`avanza_session.py:801`) — that's the contractual guard. The
unified-package path is opaque.

Fix: add a unit test that mocks `client.avanza._session.post` and
verifies the URL contains `stoploss/new` (or whatever Avanza's
stable endpoint is) — guards against library regressions.

### P1-7 `cancel_all_stop_losses_for` poll loop has unbounded inner-fetch slow-down
`portfolio/avanza_session.py:1052-1070`

```python
while True:
    try:
        poll_stops = get_stop_losses_strict()
    except Exception as exc:
        ...
        break
    still = _filter_for_ob(poll_stops)
    ...
    if (time.monotonic() - started) >= max_wait:
        break
    time.sleep(poll_interval)
```

Each `get_stop_losses_strict()` call goes through `api_get` which
holds `_pw_lock` for the entire duration. With default `poll_interval=0.5`
and `max_wait=3.0`, that's up to 6 lock-holds during cancel
verification, blocking every other API call in the process for the
entire verify window. The metals loop's 10s fast-tick can be
starved during a 3s SL verification, missing a tick. Tolerable
under normal operation, dangerous on a panic exit. Consider:
release-and-reacquire the lock between poll iterations (the strict
fetch already serializes itself), or use a shorter `max_wait` with
exponential backoff.

### P1-8 `place_trailing_stop` bypasses the non-trailing sell_price>0 guard
`portfolio/avanza_session.py:758-762, 814-846`

```python
_TRAILING_TYPES = {"FOLLOW_DOWNWARDS", "FOLLOW_UPWARDS"}
if trigger_type not in _TRAILING_TYPES and value_type == "MONETARY" and sell_price <= 0:
    raise ValueError(...)
```

`place_trailing_stop` correctly passes `trigger_type="FOLLOW_DOWNWARDS"`
+ `value_type="PERCENTAGE"` to skip the guard. But the unified path
`portfolio/avanza/trading.py:291-319` does *not* have this guard at
all — a caller who passes `trigger_type="LESS_OR_EQUAL"` with
`sell_price=0` will succeed at the library layer and place a
MARKET-order stop, which is exactly the failure mode the guard was
added to prevent in the BankID path. Apply the same guard to the
unified-package `place_stop_loss`.

### P1-9 BankID-session storage state is never reloaded after re-auth
`portfolio/avanza_session.py:134-153, 156-177`

`_get_playwright_context` loads `STORAGE_STATE_FILE` once at
context-creation. `close_playwright` tears the context down. But if
the operator runs `scripts/avanza_login.py` *while the main loop is
running* (the documented re-auth flow), the loop's context still
holds the *old* cookies — the new `avanza_storage_state.json` is
ignored. The first 401 on the old cookies will trigger
`close_playwright`, after which the next `_get_playwright_context`
will pick up the new state. But the operator has to **manually wait
for a 401 to happen** for the loop to refresh, and 401 doesn't
happen for ~24h of session life. There is no proactive "the storage
state file has been modified since I loaded it" detection.

Fix: stat the storage state file on every `_get_playwright_context`
acquisition; if mtime is newer than the cached load time, tear
down the context and reload.

## P2 — Medium

### P2-1 Two `ALLOWED_ACCOUNT_IDS` definitions, two single-sources-of-truth
`portfolio/avanza_session.py:43` and `portfolio/avanza_client.py:32`

Both files define `ALLOWED_ACCOUNT_IDS = {"1625505"}`. A future
change to add a sub-account must be made in *both* places. Worse,
the unified package (`portfolio/avanza/*`) defines no whitelist at
all (P0-1). Consolidate to one constant in a shared module
(e.g. `portfolio/avanza_accounts.py`) and import everywhere.

### P2-2 `cancel_order` (session) uses POST not DELETE — undocumented why
`portfolio/avanza_session.py:633-645`, `avanza_control.py:159-165`

Docstring at `avanza_session.py:636` says "Uses POST (not DELETE
verb) — Avanza API change 2026-03-24" — but `cancel_stop_loss` at
`avanza_session.py:888-931` *does* use DELETE. Two cancel verbs for
two cancel types is normal Avanza practice, but a future maintainer
reading just one of these functions cannot tell whether the verb
choice is API-driven or accidental. Add a comment to
`cancel_stop_loss` matching the explicit one on `cancel_order`.

### P2-3 `verify_default_account` `fetch_failed` downgrade is correct but logs only WARNING
`portfolio/avanza_account_check.py:226-249`

The Codex-P2 fix correctly downgrades a transient fetch failure to
a warning instead of raising. But the WARNING-level log line lands
in `agent.log` and may never be surfaced to Telegram, while a
`disallowed_category` or `account_not_found` *does* send a Telegram
alert (lines 274-277, 297-300). A repeated transient fetch failure
(say, 10 cycles in a row → ~10 min of "verification skipped")
silently degrades the safety guarantee with no escalation. Add a
counter: after N consecutive `fetch_failed` results, escalate to
Telegram.

### P2-4 `is_browser_dead_error` substring match is too broad
`portfolio/avanza_resilient_page.py:46-67`

```python
for marker in (
    "Target page, context or browser has been closed",
    "Target closed",
    "Browser has been closed",
    "has been closed",  # <-- This catches anything with "has been closed"
):
```

`"has been closed"` matches messages like "Order book 12345 has been
closed" (delisted instrument), "Account session has been closed"
(needs re-auth — *should* be detected separately), and any user-facing
error string Avanza chooses to surface. Tighten to the three specific
markers above and drop the generic substring.

### P2-5 `place_stop_loss` doesn't check existing-stop overlap
`portfolio/avanza_session.py:720-811`

The metals rules document
(`.claude/rules/metals-avanza.md`: "Sell + stop-loss volume must NOT
exceed position size") is enforced in
`cancel_all_stop_losses_for` + downstream sell flow, but
`place_stop_loss` itself does no pre-check of existing stops on the
same orderbook. A buggy caller that calls `place_stop_loss(vol=100)`
twice in a row creates two stops totaling 200 — the second exceeds
position size and Avanza will reject the next *sell* with
`short.sell.not.allowed`. Defense-in-depth: pre-fetch existing stops
for the same `orderbook_id` and warn (or refuse) if
`sum(existing.volume) + new_volume > position.volume`.

### P2-6 `avanza_tracker.fetch_avanza_prices` swallows per-instrument failures
`portfolio/avanza_tracker.py:58-74`

```python
try:
    info = get_price(ob_id)
    results[key] = {...}
except Exception as e:
    logger.warning("Price fetch failed for %s: %s", key, e)
```

The dashboard surfaces "missing instruments" silently — there's no
way for the loop to know the difference between "MINI-SILVER not
configured" and "MINI-SILVER price API is down for the last hour."
A signal-engine that depends on warrant price for a sell decision
gets no warrant price, falls back to underlying, and the trade
sizing is off. Consider returning a `(price, error)` tuple or a
status dict per instrument.

### P2-7 `streaming.py` heartbeat loop has no auth refresh
`portfolio/avanza/streaming.py:91-251`

The CometD WebSocket reconnects on disconnect with exponential
backoff, but it never re-fetches `push_subscription_id` from the
auth singleton — that ID is captured at `__init__` and reused
forever. If the operator runs `AvanzaAuth.reset()` for re-auth
(`avanza/auth.py:117-121`), the streaming client keeps using the old
subscription id and the next handshake will fail until the next
backoff cycle clears. After max backoff (60s) the stream is
effectively dead while the rest of the system recovered. Pull
`push_subscription_id` lazily from the auth singleton on every
handshake, not at init.

### P2-8 `get_quote` (session) uses `/stock/` path for all instrument types
`portfolio/avanza_session.py:667-673`

```python
def get_quote(orderbook_id: str) -> dict:
    return api_get(f"/_api/market-guide/stock/{orderbook_id}/quote")
```

If called for a certificate, warrant, or ETF, this returns either an
empty quote or a 404 (caught and converted to `RuntimeError` in
`api_get`). Callers expect `bid/ask/last/changePercent` and may
treat the empty response as "instrument has no quote" rather than
"wrong endpoint." The `get_instrument_price` function at
`avanza_session.py:1227-1248` correctly tries multiple endpoints —
`get_quote` should do the same or be marked stock-only.

## P3 — Low

### P3-1 `avanza_client._account_id` cache is process-global with no invalidation
`portfolio/avanza_client.py:239-279`

Once `get_account_id()` succeeds, `_account_id` is cached forever.
If the operator changes the whitelist at runtime (impossible today,
but the cache assumption locks it in) or if the Avanza account id
ever changes (e.g. migration), the cached value is stale until the
process restarts. `reset_client` and `reset_session` exist but don't
clear `_account_id`. Add a `reset_account_id()` helper.

### P3-2 `_create_avanza_client` exception message leaks credentials in dev
`portfolio/avanza/auth.py:36-42`

```python
raise AuthError(f"Avanza authentication failed: {exc}") from exc
```

If `avanza-api` ever included credential fragments in its exception
message (some libraries do, especially TOTP libraries surfacing the
TOTP secret in retry contexts), they'd land in `agent.log` at ERROR
level. Sanitize the exception string before formatting, or use
`raise AuthError("Avanza authentication failed") from exc` to keep
the chained traceback without inlining the message.

### P3-3 `place_buy_order`/`place_sell_order` `valid_until` default is today
`portfolio/avanza_session.py:542-572`

Day orders that expire at end-of-day are correct for active trading,
but for the metals fishing flow that places limits below current
price expecting fills overnight (per
`memory/fishing_system.md`), an explicit multi-day expiry is
required. The callers do pass `valid_until` when they want it, but
there's no compile-time guard against forgetting — and a forgotten
`valid_until` silently expires the order at midnight CET. Consider:
log INFO when `valid_until` defaults to today *and* `price` is more
than 1% below current quote (i.e. a fishing-shaped order with a
day-expiry mismatch).

### P3-4 `cancel_stop_loss` doesn't pre-fetch to validate `stop_id`
`portfolio/avanza_session.py:888-931`

Calling `cancel_stop_loss("not-a-real-id")` returns 404 → treated as
SUCCESS (line 922). That's correct for idempotency, but if the caller
*meant* to cancel a specific stop and passed the wrong id (typo,
stale local state), the function returns SUCCESS without ever having
touched the actual stop. Caller assumes the stop is gone, places a
sell, hits `short.sell.not.allowed`. Optional: pre-fetch
`get_stop_losses_strict()`; if `stop_id` is not in the list, log a
WARNING about a no-op cancel and *still* return SUCCESS to preserve
idempotency, but at least the operator sees the mismatch.

### P3-5 `avanza_orders.confirm_token` is 24 bits of entropy
`portfolio/avanza_orders.py:42-65`

24 bits ≈ 16M tokens, plenty for an in-flight pop of ~5 orders. But
the format is `[0-9a-f]{6}` — same alphabet as a Telegram message
ID, an Avanza order id suffix, or any hex-shaped accident. The
regex `_HEX_TOKEN_RE` requires the token to be *exactly* hex-shaped,
which is correct, but a user pasting a 6-hex Avanza order id into
the chat after a successful execution ("`CONFIRM 5a3b2e — done!`")
would silently match if there's a pending order with that token.
Probability low (1 / 16M per paste), but the cost of a stray buy is
the order_total. Add: a non-hex prefix character (`t-5a3b2e` or
similar) so accidental hex strings can't match.

### P3-6 `place_stop_loss` doesn't validate `valid_days` upper bound
`portfolio/avanza_session.py:720-811`

Avanza caps stop-loss validity at ~90 days. The function takes
`valid_days: int = 8` with no upper-bound check. A caller passing
`valid_days=365` will either succeed (broker silently clamps) or
fail with an opaque error. Add: `if valid_days <= 0 or valid_days > 90: raise ValueError`.

## Tests missing

The following safety-critical paths have no direct test coverage that
exercises the failure mode (judged by adversarial reading; not a
full test inventory):

1. **P0-1**: `portfolio/avanza/trading.py:place_stop_loss(account_id="2674244", ...)`
   does *not* raise — the unified-package path leaks orders to the
   pension account. Test: pass a non-whitelisted account id, assert
   `ValueError`.
2. **P0-2**: `portfolio/avanza/trading.place_order` does not acquire
   `avanza_order_lock`. Test: mock the lock, call the function, assert
   lock was acquired.
3. **P0-3**: `avanza_session.cancel_order` HTTP 200 + body
   `{"orderRequestStatus": "ERROR"}` is treated as success. Test:
   mock `api_post` to return that body shape, assert the wrapper
   surfaces a failed status.
4. **P0-4**: `cancel_stop_loss` swallowing `AvanzaSessionError` into
   generic FAILED. Test: mock `_get_csrf` to raise
   `AvanzaSessionError`, assert the result's `status` is a
   distinguishable auth-failure value.
5. **P1-1**: `get_open_orders` and `get_stop_losses` returning `[]` on
   read failure indistinguishably from real-empty. Test: mock
   `api_get` to raise RuntimeError, then mock it to return `[]`,
   assert callers can tell the difference (currently they cannot).
6. **P1-2**: Unified-package `get_buying_power` returning zeros on
   not-found. Test: feed an overview with no matching account id,
   assert function returns None / raises (not zero).
7. **P1-4**: Concurrent `request_order` + `check_pending_orders`
   write race. Test: spawn two threads racing read-modify-write of
   `avanza_pending_orders.json`, assert no order is lost.
8. **P1-5**: ResilientPage relaunch loop bounded. Test: simulate 100
   consecutive TargetClosedErrors, assert backoff kicks in /
   Telegram alert fires.
9. **P1-9**: Storage-state file mtime detection. Test: write a new
   storage state file mid-session, call `_get_playwright_context`,
   assert it picks up the new file.
10. **P1-7**: `cancel_all_stop_losses_for` holding `_pw_lock` across
    poll loop. Test: launch a thread calling
    `cancel_all_stop_losses_for` and a second thread calling
    `get_quote`; assert quote thread is NOT starved beyond
    `poll_interval`.
11. **P2-3**: `verify_default_account` repeated `fetch_failed`
    escalation. Test: mock `_api_get_categorized_accounts` to raise
    10 times in a row; assert Telegram alert is sent.
12. **P2-4**: `is_browser_dead_error` over-matching. Test: feed an
    error message "Order book has been closed" and assert it does
    NOT trigger browser relaunch.
13. **P2-5**: `place_stop_loss` overlap detection. Test: place a stop
    for vol=100 when an existing stop for vol=100 exists on the same
    orderbook; assert pre-check warns or refuses.
14. **P2-7**: Streaming auth-refresh after `AvanzaAuth.reset()`.
    Test: start stream, reset auth singleton, disconnect-and-reconnect
    underlying ws, assert handshake uses the new push_subscription_id.
15. **P3-5**: Bare hex stray-match in CONFIRM. Test: send
    `CONFIRM 5a3b2e` while an unrelated order with that token is
    pending; verify acceptable (the bug is intentional behavior) —
    but document expected behavior. Then test the proposed `t-`
    prefix as the future fix.
16. **End-to-end whitelist contract**: every public order/SL/cancel
    function across all three paths
    (`avanza_session`, `avanza_client`, `portfolio.avanza.trading`)
    rejects account id `"2674244"` (pension). Currently only two of
    the three paths enforce this. The test must be parameterized
    across all three to make P0-1 a regression rather than a one-off.
