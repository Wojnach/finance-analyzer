# Avanza-API Adversarial Review

Scope: `portfolio/avanza/` (modern typed package) and `portfolio/avanza_*.py`
(legacy single-module flat layout). Real money, BankID + Playwright +
HTTP. Memory note: only ISK account `1625505` is tradable; pension account
`2674244` must never be touched.

## P0 findings

**`portfolio/avanza/trading.py:81` & `:160` & `:240` & `:352` — Typed-package trading bypasses the account whitelist.**
`portfolio/avanza_session._place_order` and `portfolio.avanza_session.place_stop_loss`
gate on `ALLOWED_ACCOUNT_IDS = {"1625505"}` with H7/H8/BUG-211 guards
(`avanza_session.py:589-606`, `:750-751`). The new typed package
(`portfolio.avanza.trading.place_order`, `modify_order`, `cancel_order`,
`place_stop_loss`, `place_trailing_stop`, `delete_stop_loss`) has no such
check — every function reads `acct = account_id or client.account_id` and
forwards verbatim to `client.avanza.place_order(...)`. So a caller
(LLM-generated, refactor, or future plug-in) doing

```python
from portfolio.avanza import place_order
place_order("BUY", ob_id, price, vol, account_id="2674244")
```

routes a live order to the pension account, which the explicit user
memory says must never happen. The whitelist enforcement that
`avanza_session.py` ships with is the entire point of the H7/A-AV-2
work; mirroring the public surface without mirroring the guard is
exactly the regression those guards were written to prevent. Same hole
exists for `modify_order`, `cancel_order`, `place_stop_loss`,
`place_trailing_stop`, `delete_stop_loss`.

Fix: import `ALLOWED_ACCOUNT_IDS` (or define a package-local equivalent
sourced from a single config), and raise `ValueError` at the top of
every mutating function the moment `acct not in ALLOWED_ACCOUNT_IDS`.
Add the same `MAX_ORDER_TOTAL_SEK = 50_000` / `>= 1000 SEK` paired
checks (BUG-211 + H8) so the two code paths cannot diverge in the
opposite direction either.

**`portfolio/avanza/trading.py:80-92` — Typed-package trading bypasses the cross-process order lock.**
`portfolio/avanza_session._place_order` wraps `api_post` with
`avanza_order_lock(...)` (line 620) explicitly because metals_loop /
golddigger / fin_snipe / grid_fisher all share one buying-power view
and overlapping POSTs overdraw the ISK. `portfolio/avanza_client._place_order`
matches that with `avanza_order_lock(op=f"place_order_totp/...")` (line
358). The new `portfolio.avanza.trading.place_order` calls
`client.avanza.place_order(...)` directly with no lock. A caller in
that new package racing against a metals_loop POST will reproduce the
exact failure mode the lock was built to stop. Same for `modify_order`,
`cancel_order`, `place_stop_loss`, `place_trailing_stop`,
`delete_stop_loss`.

Fix: wrap every mutating call with `with avanza_order_lock(op=...)`;
re-use the existing `OrderLockBusyError` propagation pattern so callers
can retry next cycle without blocking the loop.

**`portfolio/avanza/client.py:65` — `account_id` is config-driven, no whitelist.**
`AvanzaClient.get_instance(config)` accepts `account_id` from
`config["avanza"]["account_id"]` and falls back to the
`DEFAULT_ACCOUNT_ID = "1625505"` constant. Any config drift (a stale
key, a copy-paste error, a future "multi-account" feature that
overwrites this) silently changes the trading target — and the package
treats that field as authoritative everywhere (`trading.py:81`,
`account.py:75`, etc.). The legacy `avanza_session.DEFAULT_ACCOUNT_ID`
is a module constant, which makes drift far less likely.

Fix: after reading `account_id` from config, verify it's in
`ALLOWED_ACCOUNT_IDS` (define this in `portfolio.avanza.client` or
import from `avanza_session`). Refuse to construct the singleton if
not — fail closed.

**`portfolio/avanza_session.py:212-232` — Browser-recovery retry can double-submit orders.**
`_with_browser_recovery` retries `op(ctx)` once on `TargetClosedError`
or "Target closed". Every mutation (`api_post` for order/new,
stoploss/new, order/delete, stoploss DELETE) flows through it. If the
POST actually reaches Avanza, the broker processes it, and the browser
dies *while reading the response*, the retry re-issues the same POST.
Avanza order/new is not idempotent — two orders get placed. The
`avanza_order_lock` only serializes *concurrent* callers; it does not
deduplicate retries from the same caller.

Fix: classify ops as read vs. mutate. For mutations, do not retry on
browser death — close, raise, let the caller treat it as "unknown
state" and inspect `get_open_orders()` / `get_stop_losses()` before
deciding what to do. (Or implement an Avanza-side idempotency key, but
the API doesn't appear to expose one.) Right now this is a real
double-order risk that the lock cannot prevent.

## P1 findings

**`portfolio/avanza/account.py:27-61` — `get_positions()` returns positions from EVERY account by default.**
No filter applied unless caller passes `account_id`. The legacy
`portfolio.avanza_client.get_positions` (line 162) explicitly filters
on `ALLOWED_ACCOUNT_IDS` so pension positions do not leak; this new
typed `get_positions` does not. Any consumer (dashboard tile, risk
sizing, signal report) calling the package version sees the pension
account's holdings as if they were tradeable. User memory:
`feedback_isk_only.md` says "Only show ISK account 1625505. Ignore
pension account 2674244" — this contract is violated.

Fix: default the filter to `ALLOWED_ACCOUNT_IDS` (set), not "none". A
caller that wants cross-account access must opt in explicitly.

**`portfolio/avanza_session.py:633-645` — `cancel_order` accepts any `account_id` with no whitelist.**
`_place_order` validates (line 591), `place_stop_loss` validates (line
750), but `cancel_order` blindly forwards whatever the caller passes.
This is asymmetric — a buggy caller can cancel orders on the pension
account. Less catastrophic than placing them, but the same memory rule
applies and the same one-line guard fixes it.

Fix: identical `if acct not in ALLOWED_ACCOUNT_IDS: raise ValueError(...)` at
the top of `cancel_order` and `cancel_stop_loss`.

**`portfolio/avanza/scanner.py:206-301` — Detail fetcher passes through `orderbookId` with no whitelist or tradability re-check before returning to caller.**
`scan_instruments` returns search-derived `orderbook_id` values that
the LLM/agent then feeds straight into `place_order(ob_id=...)`. The
only filtering is `tradeable` from the search hit and a `bid/ask not
None` check. There is no allowlist of expected instrument families,
no underlying-asset cross-check, and no defense against an Avanza
search-result drift that returns an unrelated instrument. Combined
with the P0 trading-package whitelist gap, a malicious or
malformed search response could route money to an unintended
instrument. The legacy `data/metals_warrant_catalog.json` workflow
guards against this; the scanner does not.

Fix: when the scanner is used in any path feeding `place_order`, the
caller must validate `orderbook_id` against a known catalog
(`data/metals_warrant_catalog.json`, `config.avanza.instruments`).
Document this requirement on the scanner's docstring and add an
optional `allowed_ob_ids: set[str] | None = None` parameter.

**`portfolio/avanza_session.py:782-796` — `triggerOnMarketMakerQuote: True` is hardcoded on every stop-loss.**
For market-maker-quoted certificates this is correct, but for
non-MM-quoted instruments it can either be ignored or alter behavior
unexpectedly. Hardcoding a sensitive trigger flag with no per-instrument
opt-out makes the future "stop on real prints, not MM quote" change
require a code edit instead of a parameter override.

Fix: thread `trigger_on_market_maker_quote: bool = True` through the
signature so callers (and especially `golddigger`, `metals_loop`) can
override per instrument type.

**`portfolio/avanza_orders.py:142-145` — Bot-token-compromise vector documented but not closed.**
The `_check_telegram_confirm` docstring (line 237-240) admits that an
attacker with the bot token can forge updates with the right
`chat_id` and execute orders. The added `allowed_user_id` sender
allow-list (AV-P1-3) helps when configured; without it the
chat-only filter is the only guard and is bypassable. The CONFIRM
flow places ORDERS WITH REAL MONEY, so this should be required, not
optional.

Fix: make `telegram.allowed_user_id` mandatory in the
`_check_telegram_confirm` path — if missing, refuse to process any
CONFIRM and log a critical error. Currently a caller can silently lose
sender auth by deleting the config key.

**`portfolio/avanza/tick_rules.py:124-127` — `round_to_tick` never raises "no tick rule"; silently uses last entry.**
Docstring says "Raises ValueError: If no tick rule matches *price*."
but `_find_tick_for_price` falls back to `entries[-1].tick_size` when
no range matches. So an order at a price 10x higher than the last
defined band silently rounds to the wrong tick. Worse, no caller in
the trading paths actually uses `round_to_tick` — `place_order` /
`place_stop_loss` in either codebase accepts arbitrary float prices.
Avanza rejects non-tick prices, so this manifests as "order failed"
with confusing messages, but a near-tick price can still be silently
rounded server-side to something the caller did not intend.

Fix: (1) make the fallback explicit — log a warning and require an
override flag; (2) call `round_to_tick(price, ob_id, direction)` from
every `place_order` / `place_stop_loss` before submission, and surface
the rounded price back to the caller for logging.

**`portfolio/avanza/streaming.py:144-149` — WebSocket connect has no
auth/subscription verification; silent feed loss possible.**
`websocket.create_connection(WS_URL, timeout=...)` with no headers
means the push session relies entirely on `subscriptionId` in the
handshake `ext`. If `push_subscription_id` is empty (e.g.
`AvanzaAuth.__init__` got an empty `getattr(client, "_push_subscription_id", "")`),
the handshake will still report `successful: true` for the meta channel
but subscriptions silently produce zero data. The read loop runs
forever, callbacks never fire, and dependent grid logic acts on the
last cached price.

Fix: refuse to start the stream if `push_subscription_id` is empty;
after subscribe, wait for the first data message (with timeout) and
emit a critical alert if none arrives within, say, 60s of expected
quotes during market hours.

**`portfolio/avanza/scanner.py:78` — BankID fallback path calls `api_post` for search but the BankID `api_post` re-launches Playwright per call from a thread pool.**
The scanner uses `ThreadPoolExecutor(max_workers=workers)` (line 310)
only when `thread_safe == True` (TOTP). With BankID it goes sequential
(line 318). Good — but the BankID code path still calls
`api_post("/_api/search/filtered-search", ...)` for search. That
endpoint is read-only and posting to it abuses the order lock pattern
nowhere; however, the broader concern is that *every* search hit then
calls `instrument_fn(api_type, ob_id)` and `marketdata_fn(ob_id)`
sequentially through Playwright. For `max_search=30` instruments that
is ~60 sequential Playwright round-trips holding the global RLock,
which will block all other Avanza-bound code (price ticks, position
reads) for tens of seconds.

Fix: throttle / chunk the scanner under BankID, or refuse to scan with
BankID and require TOTP auth for any scan call.

## P2 findings

**`portfolio/avanza_session.py:667-673` — `get_quote` hardcodes `/stock/` endpoint.**
Function returns a `dict` but only works for stocks; certs/warrants
404. The same module's `get_instrument_price` (line 1227) already
implements the multi-type fallback. Either remove `get_quote` or have
it delegate to `get_instrument_price`. As-is, "get_quote" is a sharp
edge for callers who reasonably assume it works for any orderbook ID.

**`portfolio/avanza_session.py:849-864` — `get_stop_losses` swallows read errors and returns `[]`.**
Docstring acknowledges the danger and recommends
`get_stop_losses_strict` for safety paths, but the lenient default
remains. Any new caller defaulting to `get_stop_losses()` and using it
in a cancel-before-sell decision reintroduces the
`short.sell.not.allowed` failure mode. The lenient variant should be
renamed `get_stop_losses_lenient` so the default name carries the
fail-closed semantics, forcing an opt-in for the lossy version.

**`portfolio/avanza/scanner.py:259-262` — Direction inference is order-sensitive and lossy.**
Loop `for d in ("BULL", "BEAR", "MINI L", "MINI S")` — a name
containing both "BULL" and "BEAR" (unlikely but possible) takes
whichever appears first in the tuple. Also `"MINI L"` matches
"MINI LONG" *and* "MINI L25" (a leverage prefix). Should use
word-boundary regex or structured fields from the instrument
metadata, not name-substring matching.

**`portfolio/avanza/account.py:88-94` — `get_buying_power` returns zero-value `AccountCash` on "account not found".**
This is the dangerous silent-failure pattern that
`portfolio.avanza_session.get_buying_power` was rewritten (BUG C7) to
fix — it now returns `None` on miss so callers cannot mistake "API
failed" for "balance is zero". The new typed package returns
`AccountCash(0.0, 0.0, 0.0)` which sizing logic happily passes
through. Mirror the C7 fix: return `None` (or raise) on miss.

**`portfolio/avanza/streaming.py:79-85` — `on_orders` / `on_deals` accept arbitrary `account_ids`.**
No filter against `ALLOWED_ACCOUNT_IDS`. A caller subscribing
`on_orders(["2674244"])` gets pension orders into a callback. Read-only
side effect but violates the same "ignore pension" contract.

**`portfolio/avanza_session.py:131` — `is_session_expiring_soon` returns `True` on read failure.**
Treats "cannot read session file" the same as "session is expiring".
The downstream consumer (`avanza_tracker.check_session_expiry`)
constructs an "Avanza session not found" warning when
`session_remaining_minutes()` returns `None`, but the
`is_session_expiring_soon`-only caller path produces a misleading
"about to expire" alert.

**`portfolio/avanza/auth.py:104-107` — `getattr(client, "_push_subscription_id", "")` silently defaults to empty string.**
If the underlying `avanza` library renames its private attribute
(e.g. `_push_subscription_id` → `_push_sub_id`), the singleton
constructs with empty values and `AvanzaStream` will fail silently as
described in P1 streaming finding. Add a runtime assertion that all
four `getattr` results are non-empty, and fail loudly on construction.

**`portfolio/avanza_session.py:1119-1224` — `rearm_stop_losses_from_snapshot` accepts arbitrary `account` from snapshot dict, bypassing the whitelist.**
`place_stop_loss` enforces the whitelist (line 750), but the snapshot
dict was originally produced from `get_stop_losses()` reading every
account. So a future drift that smuggles a non-whitelisted stop into
the snapshot would correctly be rejected by `place_stop_loss`, but the
error path silently appends the original_id to `failed` and continues
rather than alerting the operator. Should log at critical so a
violation surfaces in `data/critical_errors.jsonl`.

## P3 findings

**`portfolio/avanza_session.py:43` — `ALLOWED_ACCOUNT_IDS = {DEFAULT_ACCOUNT_ID}` is a derived set with no envelope.**
A future "add a sub-account" change requires touching one line; that's
fine. Cosmetic: the inline derivation conflates "default routing
target" with "allow-list" so adding a second permitted account also
requires writing `ALLOWED_ACCOUNT_IDS = {DEFAULT_ACCOUNT_ID, "..."}` —
slightly counter-intuitive when the comment above suggests they're
separate concepts.

**`portfolio/avanza_session.py:1032` — `import copy as _copy` inside function.**
Trivially move to module top. The deepcopy is critical for snapshot
isolation — make it visible.

**`portfolio/avanza_orders.py:144` — Bot-token URL passed to `fetch_with_retry`.**
`http_retry._redact_url` masks `/bot<token>/` from log lines so this
is OK in practice — call it out in a comment so a future refactor
doesn't introduce a logger that bypasses redaction.

**`portfolio/avanza_session.py:587` — `if price <= 0` rejects price=0; OK for limit orders but `place_stop_loss(sell_price=0, ...)` for trailing stops takes a different path (line 837 of `place_trailing_stop`).**
Slightly fragile dual-purpose argument. Worth a single
`sell_price_or_trail_distance` typed wrapper.

## Cross-cutting observations

1. **Two parallel Avanza implementations exist** and have drifted on
   safety guards. `portfolio/avanza_session.py` (legacy, BankID +
   Playwright) and `portfolio/avanza_client.py` (legacy, TOTP) both
   enforce the ISK whitelist. `portfolio/avanza/*` (the new typed
   package) does not. Until the new package gets the same guards or
   is removed from the live path, every "use the typed package" hint
   in agent context is a foot-gun. Recommend either (a) make the new
   package a thin facade over the legacy guards, or (b) port the H7 /
   H8 / BUG-211 / A-AV-2 / P0-4 work line-for-line into the new
   package and add tests that diff the two surfaces.

2. **No idempotency on order placement**. `_with_browser_recovery`,
   `fetch_with_retry`, and the TOTP `client.place_order` all can
   re-submit a mutation on transient failure. The only thing
   preventing double-orders today is timing luck — Avanza orders/new
   doesn't expose an idempotency key, and the codebase doesn't
   construct one client-side (e.g. caller-supplied UUID checked
   against `get_open_orders()` before retry). This is the single
   highest-risk gap.

3. **Order lock is local to mutating paths in `avanza_session.py` /
   `avanza_client.py` / `data/metals_avanza_helpers.py` / `avanza_control.py`** but
   `portfolio/avanza/trading.py` and `portfolio/avanza_client.py:place_order`
   under TOTP both call the underlying library directly. The lock is
   only correct as a *contract* if every order-placing entry point
   uses it. The new typed package breaks the contract silently.

4. **No global rate limiter**. Avanza will rate-limit aggressive
   callers, and the scanner under BankID could trivially hammer.
   Recommend a process-wide token-bucket on `api_get`/`api_post` (say
   10 req/s) with an alert at the dashboard.

5. **`get_csrf(ctx)` reads cookies even after a 403 close**. After a
   403 forces `close_playwright()`, the next `api_post` call relaunches
   and `_get_csrf` reads a fresh cookie — that part is correct. But
   the actual user-facing semantics on 403 is "re-run avanza_login.py"
   which is the documented contract. No bug, but the 403 path
   recovers automatically on next call, which can mask repeated CSRF
   rotation problems unless someone watches the log.

6. **Stop-loss success detection is loose** in
   `portfolio/avanza/types.py:211`: `StopLossResult.from_api` treats
   `"ACTIVE"`, `"OK"`, `"SUCCESS"` all as success. Avanza's actual
   response on stop-loss creation is `{"status": "SUCCESS",
   "stoplossOrderId": "..."}`. Accepting unknown synonyms widens the
   surface unnecessarily and a future Avanza addition of a new status
   value matching one of these strings would silently look like
   success.

7. **Confirm-token entropy is small (24 bits / 16M space)** —
   `avanza_orders.py:48`. With ≤5 in-flight orders this is fine, but
   if a future feature batches dozens of confirmations the birthday
   bound starts mattering. 8 hex (32 bits) is the cheap fix; the user
   can still type it on a phone.

## Files reviewed

- `portfolio/avanza/__init__.py`
- `portfolio/avanza/account.py`
- `portfolio/avanza/auth.py`
- `portfolio/avanza/client.py`
- `portfolio/avanza/market_data.py`
- `portfolio/avanza/scanner.py`
- `portfolio/avanza/search.py`
- `portfolio/avanza/streaming.py`
- `portfolio/avanza/tick_rules.py`
- `portfolio/avanza/trading.py`
- `portfolio/avanza/types.py`
- `portfolio/avanza_account_check.py`
- `portfolio/avanza_client.py`
- `portfolio/avanza_control.py`
- `portfolio/avanza_orders.py`
- `portfolio/avanza_order_lock.py`
- `portfolio/avanza_resilient_page.py`
- `portfolio/avanza_session.py`
- `portfolio/avanza_tracker.py`
- `portfolio/http_retry.py` (cross-ref for retry semantics)
- `.gitignore` (verified `data/avanza_session.json` and
  `data/avanza_storage_state.json` are excluded)
