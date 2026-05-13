# Claude adversarial review: avanza-api

Scope: `portfolio/avanza_session.py`, `portfolio/avanza_client.py`,
`portfolio/avanza_orders.py`, `portfolio/avanza_tracker.py`,
`portfolio/avanza_control.py`, `portfolio/avanza_account_check.py`,
`portfolio/avanza_order_lock.py`, `portfolio/avanza_resilient_page.py`,
`portfolio/avanza/*.py`, plus the page-mode helpers in
`data/metals_avanza_helpers.py` that `avanza_control.py` re-exports.

Project at `Q:\finance-analyzer`, HEAD `59ad394e`. Read-only.

## Summary

Three trading stacks live side-by-side: (a) **BankID/Playwright** in
`avanza_session.py` with a singleton headless Chromium and an `RLock`,
(b) **page-mode** in `data/metals_avanza_helpers.py` driven by a caller-owned
Playwright `Page`, and (c) **TOTP / avanza-api library** in `portfolio/avanza/*`
and `avanza_client.py`. `avanza_control.py` re-exports all three.

The biggest issues are not in any single endpoint call but in the gaps where
the three stacks meet:

- **The `avanza_order_lock` is broken inside `avanza_session.py` for in-process
  callers.** Stacks (a) holds an RLock (`_pw_lock`) across the entire HTTP
  request *and* takes the cross-process file lock inside it. Two threads in
  the same loop cannot run concurrently because they both want `_pw_lock` —
  but the file lock is reentrant only across processes, not threads. The
  serialization is incidental and depends on the `_pw_lock`, not the order
  lock. A future refactor that removes the wrapping (or uses async
  Playwright) would silently drop concurrency protection.
- **`avanza_client.ALLOWED_ACCOUNT_IDS` and `avanza_session.ALLOWED_ACCOUNT_IDS`
  are independent literals.** Both currently hold `{"1625505"}`, but a future
  rename in `avanza_session.DEFAULT_ACCOUNT_ID` will silently desynchronize
  one file from the other, and the loud whitelist enforcement in (a) becomes
  a silent allowlist mismatch in (b).
- **`get_buying_power` returns `None` for "could not read"** but several
  *consumers* in the rest of the codebase still go through callers that
  short-circuit `None` to `0`, which is the silent-failure mode the function
  was specifically rewritten to eliminate. The function itself is sound; the
  callers undo the protection (out of scope for this review but flagged).
- **`avanza_orders.check_pending_orders` executes orders without rechecking
  the price**, the user can confirm a 5-minute-stale price and pay through it.
- **`AvanzaClient` / `avanza.account.get_buying_power` does NOT account-filter
  to `ALLOWED_ACCOUNT_IDS`** — it returns zeroes when the configured account
  is missing but happily returns balances for any other account if the
  caller asks. Compared with the loud whitelist in
  `avanza_session.place_*_order`, this is a silent step backwards.
- **`avanza_session.get_open_orders` swallows the second failure path
  silently** (returns `[]`), which `avanza_session.cancel_stop_loss` does NOT
  match — the contract is inconsistent between sister helpers.
- **The unified `portfolio/avanza` package has no order lock anywhere.**
  Trading via `from portfolio.avanza import place_order` bypasses every
  cross-process lock the rest of the codebase relies on. The `place_order`
  in `avanza/trading.py` calls `client.avanza.place_order(...)` directly.
  This is the package the project's TOOLS.md (line "`portfolio/avanza/`
  unified package (163 tests)") presents as a clean API surface.
- **Stop-loss DELETE response handling is inconsistent across the three
  stacks.** Session path treats 404 as success; page path
  (`metals_avanza_helpers.delete_stop_loss`) treats only 2xx as success and
  will fail-on-already-gone. Library path
  (`portfolio/avanza/trading.delete_stop_loss`) string-greps the exception
  text for "404". Three different idempotency semantics for one operation.

Plus a long tail of: bare `Exception` swallowing, token-bracket bugs in
the CONFIRM regex, account ID falling back to default in places that should
fail closed, missing tick-size rounding before placement (Avanza will accept
prices that violate the tick table and reject them with a confusing error),
and a Playwright singleton lifecycle that depends on `_pw_lock` reentrance
which is correct today but fragile.

---

## P0 — Blockers

### P0-1 — `place_order` in `portfolio/avanza/trading.py` bypasses the cross-process order lock

**Files:** `portfolio/avanza/trading.py:38-102`, `portfolio/avanza_order_lock.py:18-33`.

The unified-package `place_order`, `modify_order`, `cancel_order`,
`place_stop_loss`, `place_trailing_stop`, and `delete_stop_loss` all call
`client.avanza.*` directly with no `avanza_order_lock` context. Every other
stack does take the lock:

- `data/metals_avanza_helpers.place_order` (line 298) — locks
- `data/metals_avanza_helpers.place_stop_loss` (line 383) — locks
- `data/metals_avanza_helpers.delete_order` (line 426) — locks
- `data/metals_avanza_helpers.delete_stop_loss` (line 477) — locks
- `portfolio/avanza_session._place_order` (line 620) — locks
- `portfolio/avanza_session.cancel_order` (line 644) — locks
- `portfolio/avanza_session.place_stop_loss` (line 800) — locks
- `portfolio/avanza_session.cancel_stop_loss` (line 915) — locks
- `portfolio/avanza_client._place_order` (line 358) — locks (the TOTP path)
- `portfolio/avanza_client.delete_order` (line 396) — locks

But the unified package — which `portfolio/avanza/__init__.py` exports as
the canonical surface, and which the metals-avanza rules card explicitly
flags ("TOTP auth available in portfolio/avanza/ unified package (163
tests)") — has none. If anything in the codebase imports
`from portfolio.avanza import place_order`, it races every other stack.

This is the exact same class of bug the lock was originally built to
prevent: two callers observing the same `buying_power`, both submitting,
both filling, the ISK overdraws (per the docstring scenario in
`avanza_order_lock.py:6-13`).

**Why P0:** Avanza touches real money. The order lock exists precisely
because the production loops have already lost money to this race. Shipping
a public `place_order` that doesn't honor it is a regression waiting to be
deployed.

**Fix:** Wrap each mutating call in `with avanza_order_lock(op=...)`. Match
the `op` label format of the other stacks ("place_order/SIDE/OB_ID") for
diagnostic continuity. Also add `account_id` whitelist check for parity with
`avanza_session._place_order:591-592`.

### P0-2 — `portfolio/avanza` unified package does not enforce account whitelist on trades

**Files:** `portfolio/avanza/trading.py:80-92`, `portfolio/avanza/client.py:34-72`,
`portfolio/avanza_session.py:43, 590-592`.

`avanza_session._place_order` raises `ValueError` if the resolved account is
not in `ALLOWED_ACCOUNT_IDS = {"1625505"}` (line 591). `avanza_client.get_account_id`
similarly enforces a whitelist (line 270).

The unified package has none. `AvanzaClient` reads `account_id` from
`config["avanza"]["account_id"]` with a `DEFAULT_ACCOUNT_ID = "1625505"`
literal (line 65). If config is malformed, partially loaded, or someone
adds a sibling account_id field for testing, every trade goes to whatever
the config says — including the pension account `2674244`.

`place_order(account_id=...)` (line 45 of `trading.py`) also accepts an
explicit override that is *never* checked against any whitelist before
being passed to `client.avanza.place_order(acct, ...)`. Pension account is
one mistaken Layer 2 hint away.

**Why P0:** Single literal change to config or a single hallucinated
`account_id` kwarg from Layer 2 routes orders to the wrong account, with
real-money consequences. The whole point of the dual whitelist in
`avanza_session.py`/`avanza_client.py` (per the explicit comment
"A-AV-2 (2026-04-11): Hardcoded account whitelist") was to make this
impossible. The unified package undoes it.

**Fix:** Add `ALLOWED_ACCOUNT_IDS` to `portfolio/avanza/client.py` and
gate `place_order`, `modify_order`, `place_stop_loss`,
`place_trailing_stop`, `delete_stop_loss`, `cancel_order` on it (and the
`account_id` override too).

### P0-3 — `avanza/account.get_buying_power` returns silent zeroes on account-not-found

**File:** `portfolio/avanza/account.py:64-94`.

Returns `AccountCash(buying_power=0.0, total_value=0.0, own_capital=0.0)` and
only logs a `warning` when the configured account ID is not found in the
overview. The `avanza_session.get_buying_power` rewrite explicitly forbids
this behaviour (line 405-412 docstring: "we now... return ``None`` so callers
can distinguish 'API call failed' from 'balance is legitimately zero' —
callers must now explicitly handle the ``None`` case (previously they
silently got ``buying_power=0``, which was a dangerous silent failure)").

The same regression now exists in the unified package — zero buying power
silently means "skip the buy" downstream, which means a real signal can be
dropped because the API shape changed (which is exactly what happened on
2026-04-09 mid-day per the avanza_session.py comment).

**Why P0:** Replays the exact 2026-04-09 silent-failure incident the session
rewrite was triggered by, but in the supposedly newer / cleaner package.

**Fix:** Raise an exception (or return `None`) on account-not-found rather
than returning zeroes. Mirror the multi-shape walk in `avanza_session.get_buying_power`
(legacy → flat → new-categorized) since `AvanzaClient.get_overview_raw`
goes through the avanza-api library, which may or may not normalize for you
across Avanza shape drifts.

### P0-4 — `avanza_session._with_browser_recovery` retries a non-idempotent POST after browser death

**File:** `portfolio/avanza_session.py:212-232`, called by `api_post` (line 344).

When the Playwright browser dies mid-flight, `_with_browser_recovery`
catches the exception, tears down, relaunches, and retries the **entire**
`op(ctx)`. For `api_get` that's safe — GET is idempotent. For `api_post`
this is `/_api/trading-critical/rest/order/new` — placing the order again.

The "browser dead" classifier (`is_browser_dead_error`) fires on
`TargetClosedError` and on substring "has been closed". If the browser
dies *after* the order POST left the wire but *before* the response came
back, the retry submits a duplicate order. The duplicate has the same
payload (same orderbook_id, same volume, same price) and Avanza has no
idempotency key — both orders are valid and both can fill.

The same applies to `place_stop_loss` (POST to `/_api/trading/stoploss/new`)
and to `cancel_order` (POST to `/_api/trading-critical/rest/order/delete` —
also non-idempotent, double cancel is harmless but consider the case where
the original delete partial-acked: now you've fired two cancels at slightly
different times).

**Why P0:** Duplicate live orders on real money. The whole `_with_browser_recovery`
retry layer was added because metals loop went 4 days making zero trades —
but the fix introduces a worse failure mode for mutating calls.

**Fix:** Pass an `idempotent: bool` flag to `_with_browser_recovery` (or
split into `_with_browser_recovery_get` and `_with_browser_recovery_post`).
For POSTs: do not retry; close the browser and propagate so the caller
sees a structured failure and can decide on retry (most callers should
NOT — at minimum they should query open orders first to verify the POST
didn't land).

### P0-5 — `cancel_all_stop_losses_for` polls forever on `poll_interval=0`

**File:** `portfolio/avanza_session.py:934-1116`.

Default `poll_interval=0.5` is fine, but the function signature accepts
arbitrary floats with no validation. A caller passing `poll_interval=0`
(or anything below the float epsilon) enters a busy loop until
`max_wait=3.0s` rolls over — every iteration making a fresh
`get_stop_losses_strict()` call to Avanza. Three seconds of unthrottled
hammering at the stop-loss endpoint is enough to trip rate limits and
auto-revoke the session — especially during DST market open when many
loops are calling simultaneously.

Same issue if a caller passes `poll_interval=-1` — that gets passed to
`time.sleep` which on Linux interprets negative as 0 (busy loop), and on
Windows raises `ValueError` mid-flight, propagating out as an unhandled
exception that leaves the position in the cancel-but-not-yet-rearmed
window. From the docstring this is exactly the unsafe state the function
exists to make recoverable.

**Why P0:** A bad caller can DOS the session right before a critical sell.
The function is documented as the safety-critical clearance helper
("``status="FAILED"`` rather than silently treating 'could not read' as
'no stops exist'") — it must be robust against bad parameter values.

**Fix:** Clamp `poll_interval` to `max(0.1, poll_interval)`. Validate
`max_wait > 0`. Consider an upper bound on call rate as well — even a
correctly-passed `poll_interval=0.5` will issue 7 reads in 3 seconds if
the broker never clears, which is still a lot of API surface.

### P0-6 — Stop-loss `validUntil` on the trigger vs `validDays` on the order event can desync

**Files:** `portfolio/avanza_session.py:777-796`, `data/metals_avanza_helpers.py:358-379`.

The payload has two fields:

```python
"stopLossTrigger": {
    "validUntil": valid_until,        # ISO date, today+valid_days
    ...
},
"stopLossOrderEvent": {
    "validDays": valid_days,           # int
    ...
}
```

The trigger expires on a calendar date; the order-event has a day-count.
These are computed in Python from the same `valid_days` integer, but they
disagree by up to a day if the broker's clock is in a different timezone
to the Python process — `date.today()` is local-tz, but Avanza's session
backend resolves `validUntil` in Stockholm time. From a US-based CI runner
(or a misconfigured server), the trigger is valid one calendar day
past midnight CET but the order event still has `validDays = 8` counted
from receipt — the stop becomes a window stop rather than an aligned stop.

Additionally, on Avanza's API the trigger `validUntil` is the
*authoritative* expiry. The `validDays` on `stopLossOrderEvent` is only
the lifetime of the sell order placed *if* the trigger fires. If your
trigger expires before the sell triggers, the sell never fires. If the
trigger fires on day 8 but `validDays` was already at 7, the sell is a
0-day order — likely IOC and may not fill.

**Why P0:** Stop-losses are the last line of defence. Misalignment that
makes them silently inactive is real money.

**Fix:** Use UTC consistently or compute `valid_until` from the Avanza
server clock. Independently, document the semantic difference between
`stopLossTrigger.validUntil` and `stopLossOrderEvent.validDays` so callers
don't think they're redundant.

### P0-7 — `place_order` and `place_stop_loss` accept zero / negative price after the `<= 0` check is masked

**File:** `portfolio/avanza_session.py:586-587, 759-762`.

Line 586: `if price <= 0: raise ValueError(...)` — good.

Line 759: `if trigger_type not in _TRAILING_TYPES and value_type == "MONETARY"
and sell_price <= 0: raise ValueError(...)` — but the check on
**`trigger_price`** is missing. A non-trailing MONETARY stop can be placed
with `trigger_price=0` (or negative), which Avanza will probably reject
but might silently accept depending on shape — and a `trigger_price=0`
stop will fire immediately on the first ≤0 quote (i.e., never, because
prices are positive), so it sits forever as a phantom encumbrance on the
position volume. Subsequent sells then fail with `short.sell.not.allowed`,
which is exactly the failure mode `cancel_all_stop_losses_for` exists
to prevent — but cancelling a never-firing stop with a 0 trigger price
is fine, so the recovery works. The bug is that you've burnt an order
slot and a few hundred milliseconds for nothing.

For trailing stops the math is worse: `trigger_price` becomes
`trail_percent`. A 0 trail-percent in `place_trailing_stop` (line 838-846)
flows through to a stop that triggers on any downtick. That's a market
sell on the next bid below latest, immediately. Combined with a leveraged
warrant on a wide spread, this is a P1 disaster waiting on a typo.

**Why P0:** Easy typo, immediate financial impact on warrants.

**Fix:** Add `if trigger_price <= 0: raise ValueError(...)` unconditionally,
plus separate validation for trailing-percentage (e.g. `0 < trail_percent <= 50`).

---

## P1 — High

### P1-1 — Two independent `ALLOWED_ACCOUNT_IDS` literals can desync

**Files:** `portfolio/avanza_session.py:43`, `portfolio/avanza_client.py:32`.

```python
# avanza_session.py
DEFAULT_ACCOUNT_ID = "1625505"
ALLOWED_ACCOUNT_IDS = {DEFAULT_ACCOUNT_ID}

# avanza_client.py
ALLOWED_ACCOUNT_IDS: set[str] = {"1625505"}
```

Plus `portfolio/avanza/client.py:19`: `DEFAULT_ACCOUNT_ID = "1625505"` —
another literal. Plus `portfolio/avanza_account_check.py` re-imports
`avanza_session.DEFAULT_ACCOUNT_ID` (line 211) — that one's safe. So we
have **three** sources of truth and only one of the writes-vs-reads is
linked.

**Why P1:** Future contributor updates one literal, ships, the other
trade-path silently accepts the old account or silently rejects a new
legitimate account.

**Fix:** Single source of truth — make `avanza_client` and
`portfolio/avanza/client` import `ALLOWED_ACCOUNT_IDS` and
`DEFAULT_ACCOUNT_ID` from `avanza_session`. Or pull both to a new
`portfolio/avanza_constants.py`.

### P1-2 — `_pw_lock` reentrance is the only thing serializing in-process trades through `avanza_session`

**File:** `portfolio/avanza_session.py:54, 219, 286, 800`.

The `RLock` was changed from `Lock` to `RLock` (the comment at line 49-54
explains why) so the API helpers could nest into `_get_playwright_context()`.
That's correct. But the consequence is:

- Thread A enters `place_stop_loss` → grabs `_pw_lock` (recursive).
- Inside, `api_post` grabs `_pw_lock` (recursive +1).
- Inside the cross-process file lock, `_get_csrf(ctx)` does not re-grab
  (uses the passed ctx) — good.

Thread B enters `place_stop_loss` at the same time → waits on `_pw_lock`.
Once A releases, B proceeds. So serialization across threads in the same
Python process works, but **only because** Playwright's sync API requires
serial access and `_pw_lock` is held the entire HTTP round-trip.

If somebody ever drops `_pw_lock` wrapping in favour of an async Playwright
client or a connection pool, all in-process callers will race past each
other. The cross-process `avanza_order_lock` only fires on file-lock
contention, which doesn't trip among threads in the same PID.

**Why P1:** Hidden coupling. A "natural" refactor (use Playwright Async to
unblock the loop while a 300ms request is in flight) would silently disable
in-process ordering serialization. The order lock looks like it's the
serializer but it isn't, for in-process threads.

**Fix:** Inside `_place_order`, `place_stop_loss`, `cancel_order`,
`cancel_stop_loss`: take a **thread-local mutex** independent of
`_pw_lock`, plus the cross-process `avanza_order_lock`. Document explicitly
that the order lock alone is insufficient for in-process serialization.

### P1-3 — `cancel_order` calls POST to `/_api/trading-critical/rest/order/delete` without a route-existence check

**Files:** `portfolio/avanza_session.py:633-645`, `portfolio/avanza_control.py:159-210`.

Comment says: "IMPORTANT: Uses POST (not DELETE verb) — Avanza API change
2026-03-24." That's been the API for 6+ weeks; OK. But if Avanza changes
again, you get an HTTP 404 returned as `result = {"raw": ...}` per the
api_post fallback (line 339-342). Caller does `result.get("orderRequestStatus", "UNKNOWN")` (line 622 wait — wrong reference; let me re-cite).

`cancel_order` (line 644-645) just returns `api_post(...)`. The caller is
expected to check `result["orderRequestStatus"] == "SUCCESS"` — which most
callers do **NOT**. In `portfolio/avanza_control.delete_order_no_page`
(line 384-386): `result = _cancel_order(order_id, account_id); ok =
result.get("orderRequestStatus") == "SUCCESS"`. OK that one does. But if
Avanza returns 404 (route renamed), `_cancel_order` returns
`{"raw": "<html>404</html>"}`, `.get("orderRequestStatus")` is `None`,
`ok = False` — silent failure. The caller logs nothing meaningful and the
order is still open on Avanza.

**Why P1:** Silent failure on next Avanza API change. No alerting.

**Fix:** Have `cancel_order` inspect HTTP status. If non-2xx, raise or
return a structured `{"orderRequestStatus": "FAILED", "http_status": 404,
"error": "endpoint may have changed"}`. Add a heartbeat / health check that
periodically validates that `cancel_order` against a no-op test order
returns the expected schema.

### P1-4 — `cancel_stop_loss` rollback removes ids in-place, breaking the snapshot contract

**File:** `portfolio/avanza_session.py:1083-1116`.

The "verified-cleared" filter:

```python
remaining_set = set(remaining)
cancelled = [c for c in cancelled if c not in remaining_set]
```

is correct per the comment. But on `poll_read_failed=True` it does
`cancelled = []` (line 1109) — which is also conservative. However the
**snapshot** is *not* filtered: `snapshot` still contains every original
stop, including ones whose DELETE call actually succeeded. If the caller
calls `rearm_stop_losses_from_snapshot(result["snapshot"])`, it places
duplicates of every still-live stop.

The contract docstring (line 968-975) says the rollback "uses cancelled-but-
verified" but in practice the snapshot is the full initial set. The caller
must intersect `snapshot` with `cancelled` themselves — which no caller
does (per grep). The comment is misleading: the "verified-cleared" filter
applies to `cancelled` (which a caller might use for accounting), not to
the rollback `snapshot`.

**Why P1:** Rollback creates duplicate stops on partial-failure paths.

**Fix:** Either filter `snapshot` to verified-cleared too, or rename the
field to `original_snapshot` and document that the caller MUST cross-
reference with `cancelled` before re-arming.

### P1-5 — `place_order` `MAX_ORDER_TOTAL_SEK` is hardcoded, comment says configurable

**File:** `portfolio/avanza_session.py:599-606`.

```python
MAX_ORDER_TOTAL_SEK = 50_000.0
if order_total > MAX_ORDER_TOTAL_SEK:
    raise ValueError(
        f"Order total {order_total:.2f} SEK exceeds maximum {MAX_ORDER_TOTAL_SEK:.0f} SEK"
    )
```

Comment says "adjust via config if needed". There's no config plumbing. A
50K cap is too small for a real ISK with 200K+ buying power doing a
multi-leg ladder. Either lift it, or hook it to `config.json`. Right now
the project has a 200K+ ISK and a 50K hard cap that nobody can hit
without code changes.

**Why P1:** Silently turns ladder strategies into single-leg strategies
on a high-conviction signal day. The user has explicitly said they're OK
with 10-20% knockout risk (memory: `feedback_risk_tolerance.md`); a 25%
hard cap on a 200K account is too cautious *and* unconfigurable.

**Fix:** Read from `config["avanza"]["max_order_sek"]` with the current
value as default. Document in CLAUDE.md.

### P1-6 — `place_stop_loss` does not honour `parentStopLossId` for modifications

**File:** `portfolio/avanza_session.py:778`.

Always sends `"parentStopLossId": "0"`. If a caller already has a stop on
this orderbook and calls `place_stop_loss` again, they create a *second*
stop instead of replacing the first. The encumbered volume sums and the
position becomes un-sellable until one of the two is cancelled.

The metals memory rule says: "Cancel existing stops BEFORE placing a sell
(prevents overfill). Use rollback if cancel fails." So today's callers
do cancel first. But `place_stop_loss`'s signature doesn't *prevent*
double-armed stops, and the rollback helper `rearm_stop_losses_from_snapshot`
(line 1188) places a fresh stop with `parentStopLossId: "0"` for each
snapshot entry — which is correct because the originals were cancelled.

If anything ever calls `place_stop_loss` to "update" an existing stop
without cancelling, you get two stops. The signature doesn't communicate
this and there's no guard. Compare against the avanza-api library's
`place_stop_loss_order` which has a `parent_stop_loss_id` parameter
(`portfolio/avanza/trading.py:273` passes `"0"` too — same hardcode).

**Why P1:** Easy way to double-encumber the position on a typo. There's
no test for this case.

**Fix:** Expose `parent_stop_loss_id` parameter. Default `"0"` is fine
but make it explicit, and add a defensive check: before placing,
`get_stop_losses_strict()` for the orderbook; if any exist for the same
account+orderbook, raise unless `parent_stop_loss_id` was passed.

### P1-7 — `_check_telegram_confirm` regex still matches "confirm." (period) as bare-CONFIRM

**File:** `portfolio/avanza_orders.py:59`.

```python
_CONFIRM_PREFIX_RE = re.compile(r"^confirm(?:\s+|$)")
```

Matches "confirm" + (whitespace OR end-of-string). It does *not* match
"confirm." or "confirm!" — the period stays in `text` but the prefix
match fails because period is not whitespace and "confirm." is not the
end of the string.

But: `text = (msg.get("text") or "").strip().lower()`. After strip+lower,
"CONFIRM." is "confirm." — `_CONFIRM_PREFIX_RE.match("confirm.")` is
`None`. So the user typing "confirm." is silently dropped.

More subtle: "confirmed" matches as "confirm" + "ed" → wait no, the regex
requires `\s+` or end-of-string after "confirm". "confirmed" has "e" after
"confirm" which is neither — `_CONFIRM_PREFIX_RE.match("confirmed")` is
`None`. Good.

But: `"confirm abc"` matches as expected, the "abc" is checked against
`_HEX_TOKEN_RE = re.compile(r"^[0-9a-f]+$")`. The full string `"abc"` IS
valid hex — so a user typing "CONFIRM abc" tries to confirm a token
"abc" against the order set. Since real tokens are 6 hex chars, this
mismatch is harmless. But it's not robust: `_HEX_TOKEN_RE` doesn't
require exactly 6 chars, so a 3-char prefix-typo of the right token
might accidentally pass to `found_tokens.add(candidate)` and then
get checked against orders — which only match on full token, so no
false confirmation. OK, that's fine.

The real bug: if `text == "confirm  "` (trailing space, gets stripped
before the prefix match anyway). And `"confirm " == "confirm "` — strip
removes trailing space, leaving "confirm", which matches the
end-of-string branch, `rest == ""`, → adds `""` to found_tokens →
matches a legacy bare CONFIRM. That's the documented backwards-compat
path. OK.

The harmless-but-annoying case: user pastes "CONFIRM abc123\nthanks" →
strip+lower → "confirm abc123\nthanks" → prefix matches at "\s+", rest
is "abc123\nthanks", `.split()[0]` is "abc123" — correct. But if the
text starts with whitespace, "  CONFIRM abc123" → strip → "confirm abc123"
→ matches. OK.

What about messages that start with non-letter then "CONFIRM"? E.g. "/CONFIRM
abc123" — strip+lower → "/confirm abc123". The prefix re is anchored at
`^`, doesn't match. Dropped silently.

So the bug is mild: tokens with trailing punctuation typed by user are
dropped. Worth fixing.

**Fix:** Broaden the regex slightly (`r"^confirm[\s.,!:]*"` with
whitespace-or-punct delimiter). Or document the exact format the user
must send and emit it in the Telegram message verbatim.

### P1-8 — `_execute_confirmed_order` does not re-validate the price before placing the order

**File:** `portfolio/avanza_orders.py:349-402`.

When a user replies CONFIRM (within 5 minutes), the order is placed at the
original `price` captured in `request_order`. No bid/ask check, no
stale-quote check. Over 5 minutes a warrant can move 10%+ — the user
sees "BUY at 8.50 SEK" in Telegram, confirms after 4 minutes of being
distracted, the warrant has dropped to 7.80 → buy lands and immediately
underwater.

Worse on the sell side: limit at 8.50 when ask is 9.20 → sits unfilled
until the next price check (or until the day order expires). The
Telegram notification says "AVANZA SELL EXECUTED" but it's actually
pending limit. The actual fill happens later (or never) and the user
has no idea.

**Why P1:** Real money mismatch between user intent and execution.

**Fix:** On confirm, re-fetch current quote, compare to original,
reject if drift > N% (configurable). If close, re-confirm with the new
price. Alternatively, change `request_order` to ask for a *fraction
above/below the bid/ask* rather than an absolute price, so the
execution recomputes from live quotes.

### P1-9 — `get_open_orders` falls back to the deal endpoint and account-filters on a possibly-string ID

**File:** `portfolio/avanza_session.py:648-664`.

```python
try:
    data = api_get(f"/_api/trading/rest/order/account/{aid}")
    ...
except RuntimeError:
    try:
        data = api_get("/_api/trading/rest/deals-and-orders")
        orders = data.get("orders", [])
        return [o for o in orders if str(o.get("accountId", "")) == aid]
    except RuntimeError:
        logger.warning("Could not fetch open orders")
        return []
```

Three problems:

1. The first `api_get` only raises `RuntimeError` on non-2xx — but
   `api_post` (and by symmetry, future shape changes in `api_get`) can
   also raise `AvanzaSessionError` on 401/403. That escapes the
   `except RuntimeError` and propagates. Either path silently returns
   `[]` or it raises — caller can't predict.
2. The fallback filters on `str(o.get("accountId", "")) == aid`. If the
   deals endpoint returns `accountId` wrapped in a `{value: ...}` dict
   (other Avanza endpoints do this — see `_v()` in `get_buying_power`),
   the comparison silently mismatches and the function returns `[]`,
   making callers think no orders exist when they do.
3. Returning `[]` on the deal-endpoint failure conflates "API down" with
   "no orders" — same class of bug as the pre-fix `get_buying_power`.
   Compare to the `get_stop_losses` / `get_stop_losses_strict` split
   (lines 849-885) which the same module has explicitly fixed.

**Why P1:** Order state misread leads to placing on top of existing
orders, or not cancelling stale ones. The "I thought I had no open
orders" failure mode is real and was the seed for the
`get_stop_losses_strict` split.

**Fix:** Same pattern as stop-losses: split into `get_open_orders` (legacy,
returns `[]` on failure) and `get_open_orders_strict` (raises). Move
safety-critical callers to strict.

### P1-10 — `get_buying_power` `total_value` and `own_capital` can be `None` and silently float through

**File:** `portfolio/avanza_session.py:478-492`.

`_get_balance` returns `None` if neither the primary nor any alternate
field exists. The result dict is:

```python
{
    "buying_power": _get_balance(acc, "buyingPower", (...)),
    "total_value": _get_balance(acc, "totalValue", (...)),
    "own_capital": _get_balance(acc, "ownCapital", (...)),
}
```

Any of these can be `None`. The function docstring says "Dict with
``buying_power``, ``total_value``, ``own_capital`` (all SEK) on success" —
but the contract doesn't promise non-null values. A caller doing
`info = get_buying_power(); cash = info["buying_power"] * 0.8` raises
`TypeError`. A caller doing `if info["buying_power"] > min:` raises
`TypeError` on `None > int`.

Worse, the test for None-shape happens on the *outer* return value, not on
the inner fields:

```python
info = get_buying_power()
if info is None:  # <-- handles outer None
    skip_order
cash = info["buying_power"]  # <-- could STILL be None
```

**Why P1:** Half the protection of the 2026-04-09 rewrite is undone if any
field is missing in the new shape. Better than the old behaviour (which
returned `0`), but the field-level silent-None is the equivalent
silent-zero in a downstream cast.

**Fix:** If *any* of the three balance fields is `None`, fail the whole
lookup (return outer `None` with a logged diagnostic). The caller's
`if info is None` then catches it.

### P1-11 — `delete_stop_loss` in `portfolio/avanza/trading.py` greps exception strings for "404"

**File:** `portfolio/avanza/trading.py:342-364`.

```python
except Exception as exc:
    exc_str = str(exc).lower()
    if "404" in exc_str or "not found" in exc_str:
        logger.info("delete_stop_loss stop_id=%s -> already gone (404)", stop_id)
        return True
    logger.error("delete_stop_loss stop_id=%s -> FAILED: %s", stop_id, exc)
    return False
```

String-matching on exception text is fragile. The avanza-api library can
raise `HTTPError`, `ConnectionError`, or a custom `AvanzaError` depending
on internals — and any of those that happen to contain "404" in their
message (e.g., a different stop-loss-not-found, or a 404 *on a different
endpoint* during a multi-step internal flow) get treated as success.

More urgent: a network error message containing "Network unreachable, see
RFC 1404" or any random "not found" substring in a stack trace gets
treated as "stop already gone" → caller thinks the stop is gone, places
the dependent sell, broker rejects because the stop is in fact still
there, position becomes naked.

Compare to `avanza_session.cancel_stop_loss` (line 920-922) which inspects
the actual `http_status` integer.

**Why P1:** Fragile error classification on a safety-critical path.

**Fix:** Have the underlying client surface HTTP status (the avanza-api
library does carry the response), check `http_status == 404`. Or wrap
the library call with explicit `requests` and inspect status directly.

### P1-12 — `delete_stop_loss` in `metals_avanza_helpers.py` does NOT treat 404 as success

**File:** `data/metals_avanza_helpers.py:457-498`.

```python
http_status = result.get("status", 0)
success = 200 <= http_status < 300
return success, {"http_status": http_status}
```

A 404 returns `success=False`. Compare:

- `avanza_session.cancel_stop_loss` (line 922): `ok = (200 <= http_status < 300) or http_status == 404` — 404 IS success
- `avanza_control.delete_stop_loss` (line 241): `success = (200 <= http_status < 300) or http_status == 404` — 404 IS success
- `metals_avanza_helpers.delete_stop_loss` (line 492): `success = 200 <= http_status < 300` — 404 is FAILURE

So depending on which path is called, the *same broker state* ("stop
already gone") returns different success values. Metals loop calls
`metals_avanza_helpers.delete_stop_loss` (per the imports in
`avanza_control.py:213`); the rollback / verification path then treats
the stop as still-armed and may attempt to cancel again or to refuse a
dependent sell.

**Why P1:** Inconsistent idempotency semantics across three implementations
of the same operation, on a real-money safety-critical primitive.

**Fix:** Treat 404 as success in all three implementations.

### P1-13 — `place_order` and `place_stop_loss` do not round price to tick

**Files:** `portfolio/avanza_session.py:608, 777`, `portfolio/avanza/trading.py:84, 272`.

`portfolio/avanza/tick_rules.py` exists specifically to round prices to
Avanza's tick grid. None of the order placement functions call it. Avanza
will reject orders with non-tick prices with a generic error that the
caller's "SUCCESS"-status check turns into a silent failure (the result
contains `orderRequestStatus != "SUCCESS"` with a message saying "Invalid
price").

For warrants with non-standard tick rules (Mini Futures with 0.001 SEK
ticks at low price tiers), this is easy to hit: Layer 2 hallucinates a
price like 8.5025 SEK because the analysis was in 4-decimal precision
and the warrant is at 8.50/8.51 with tick 0.001 below 10 SEK.

**Why P1:** Orders silently fail; the failure is logged as "Order BUY
failed" but not flagged as a *recoverable* tick-rounding issue — the
caller may retry with the same price.

**Fix:** Inside `_place_order` and `place_stop_loss`, call
`tick_rules.round_to_tick(price, ob_id, direction="down" if side == "BUY"
else "up")` before sending. Surface the rounding-applied delta in the
return value so the caller's accounting matches what the broker accepted.

### P1-14 — `_check_telegram_confirm` saves offset before processing succeeded

**File:** `portfolio/avanza_orders.py:288-346`.

The offset save happens *after* the loop that adds tokens to `found_tokens`
but *before* the caller in `check_pending_orders` actually executes any
order. If `_execute_confirmed_order` raises (and the outer try doesn't
catch it — `_save_pending` is in a finally? — let me re-read):

```python
for order in pending:
    ...
    if confirmed_by_token or confirmed_legacy:
        order["status"] = "confirmed"
        acted_on.append(order)
        ...
        _execute_confirmed_order(order, config)
    elif now > expires:
        ...

_save_pending(pending)
return acted_on
```

`_execute_confirmed_order` catches its own exceptions (line 392-402),
so a normal failure doesn't bubble out. But if it raises something
unexpected (KeyboardInterrupt during a `send_telegram` retry, OS error),
`_save_pending` never runs — the order is still marked `pending_confirmation`
on disk. But the Telegram offset has already moved past the CONFIRM message.
Next cycle: the CONFIRM is gone from getUpdates; the order sits as pending
with 0 seconds left, expires next cycle. Result: user's CONFIRM is consumed,
order is never placed, user gets "expired" notification 5 minutes after
confirming. Bad UX, but no money lost.

The reverse failure is worse: if `_save_pending` runs but `_execute_confirmed_order`
crashes after the order *is* placed at the broker, then on next cycle the
order is marked `confirmed` (status persisted) but `avanza_order_id` is
unset and no Telegram confirmation went out. The user thinks nothing
happened; the position is now live on the broker. The reconciliation
between Avanza state and local state has no automatic loop here.

**Why P1:** Lost user-facing confirmation, possible silent broker fill
that the local state machine doesn't reflect.

**Fix:** Persist the offset *after* the execution loop. Or use a
two-phase commit: write the new offset to a `_next` file, execute, on
success move `_next` over the canonical offset file. On crash recovery,
detect `_next` and re-fetch updates from the older offset to recover the
CONFIRM event.

### P1-15 — `verify_session` returns `False` on any exception, including transient network errors

**File:** `portfolio/avanza_session.py:180-196`.

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

`avanza_client._try_session_auth` calls `verify_session()` and on `False`
falls back to TOTP. A 5xx from Avanza on a healthy session, or a
30-second timeout, returns `False` — the entire process now falls back
to TOTP and rebuilds the singleton, including a full re-handshake. Under
sustained Avanza brownouts (Wednesday morning shape-change windows), this
thrashes between BankID and TOTP every minute.

Worse: `close_playwright()` in the catch tears down the singleton even
on transient errors. Next call to `_get_playwright_context()` relaunches
Chromium (cost ~1.5 seconds, ~80MB RSS). A sustained brownout = launching
60 Chromium instances per minute → RAM pressure → OS swap → loops fall
behind their cadence → Telegram alerts about loop falling behind →
operator wakes up.

**Why P1:** Transient errors look identical to genuine session expiry.
Tearing down on the wrong signal causes thrashing.

**Fix:** Classify the exception: only tear down on `AvanzaSessionError`
(401/403). On network/timeout errors, return `False` but **don't**
`close_playwright()`. Maintain a counter — close after N consecutive
failures, not immediately.

---

## P2 — Medium

### P2-1 — `place_buy_order` / `place_sell_order` allow `valid_until` to be a string the broker may not accept

**File:** `portfolio/avanza_session.py:542-572`, `614`.

```python
"validUntil": valid_until or date.today().isoformat(),
```

Caller passes `valid_until="abc"` and it goes straight to the broker.
The broker rejects, but with a generic message ("Invalid date format")
that the existing logger captures but the test suite doesn't.

`avanza/trading.py:82, 130` validates by calling
`date.fromisoformat(valid_until)`, which raises on garbage. So the unified
package is stricter than the session path. Inconsistent.

**Fix:** Validate format in `avanza_session.place_buy_order` /
`place_sell_order` before placement.

### P2-2 — Logged hash-suffix of CONFIRM token leaks first 4 chars of 6

**File:** `portfolio/avanza_orders.py:139-143`.

```python
logger.info(
    "Order requested: %s %dx %s @ %.2f SEK (id=%s, confirm_token=%s…)",
    ...
    confirm_token[:4] + "****" if confirm_token else "N/A",
)
```

Logs the first 4 of 6 hex chars. Combined with the "confirm out-of-band"
remark in the comment, this is 16 bits of remaining entropy = 65K guesses
to brute-force the token if an attacker can read agent.log AND has a
shell on the same machine that can talk to the bot. Yes, that attacker
already pwned the box, so the additional risk is small — but the comment
explicitly says "an operator reading agent.log can read it if they need
to confirm out-of-band." That's reading the FULL token, not the redacted
one. There's a mismatch between what the comment claims and what the code
logs.

**Fix:** Either log the full token (then the comment is accurate but the
log is now a confirmation key — keep agent.log mode 0600), or log only
2-char prefix.

### P2-3 — `_pw_browser.close()` failure leaves `_pw_instance` running

**File:** `portfolio/avanza_session.py:156-177`.

If `_pw_browser.close()` raises but the exception is caught (line 168-170),
`_pw_browser` is set to `None` but the underlying Chromium process may
still be alive. `_pw_instance.stop()` on the next iteration will then try
to stop a Playwright instance whose Chromium is dead — that *should* be
fine, but the underlying Chromium PID may linger until the OS GC sweep.
Under sustained churn (P1-15), the box accumulates orphan chromium.exe
processes.

**Fix:** Use `playwright._impl._sync_base.run_async` to await each close
with a timeout. Log the actual failure. Or call `os.kill(self._pw_browser.pid,
SIGKILL)` as a fallback. Either way, surface the leak via the
`relaunch_count` observability hook.

### P2-4 — `ALLOWED_ACCOUNT_IDS` enforcement is missing on `cancel_stop_loss`

**File:** `portfolio/avanza_session.py:888-931`.

`place_stop_loss` checks `acct not in ALLOWED_ACCOUNT_IDS` (line 750). But
`cancel_stop_loss` accepts any `account_id` (defaults to `DEFAULT_ACCOUNT_ID`)
and issues `DELETE /_api/trading/stoploss/{acct}/{stop_id}`. Cancelling
on the wrong account just returns 404 (not your stop), so no real-money
harm, but it's an information leak: a caller probing stop-loss IDs against
arbitrary account IDs gets a free "does this stop exist on that account"
oracle.

**Fix:** Add the same `if acct not in ALLOWED_ACCOUNT_IDS: raise ValueError(...)`
guard.

### P2-5 — `get_buying_power` does not log the Avanza account ID it is matching against

The diagnostic message on no-match (line 532-538) logs `account_id=%s` but
not the masked configured ID. Good. But the matched account `ids_seen` may
include the pension account `2674244` (it does, in practice — they all
appear in the categorizedAccounts list). Logging `ids_seen` exposes the
pension account number in agent.log. Account numbers are not secrets in
the same sense as passwords, but Swedish IDs in production logs is a
data-protection concern (GDPR).

**Fix:** Hash or mask the IDs in the log message.

### P2-6 — `_check_telegram_confirm` does not validate `chat_id` format

**File:** `portfolio/avanza_orders.py:247`.

```python
chat_id = str(config.get("telegram", {}).get("chat_id", ""))
```

If config has `chat_id: -100123456789` (a supergroup ID — negative int),
str() handles it fine. But if it's `chat_id: null` or `chat_id: false`,
str() converts to `"None"` or `"False"` and matches against `str(msg.get("chat",
{}).get("id"))` only if Telegram returns a chat with literally that string
ID — which it doesn't. So the function returns no CONFIRMs ever, silently.

A new operator typoing chat_id config sees no order executions and no
error.

**Fix:** Validate `chat_id` is a string of digits (possibly with leading
minus). Log if invalid.

### P2-7 — `is_session_expiring_soon` returns `True` when no session exists

**File:** `portfolio/avanza_session.py:123-131`.

Returns `True` if remaining is `None` (no session) — semantically "needs
refresh". But the docstring suggests it's checking for "expiring soon".
A caller doing `if not is_session_expiring_soon(): place_order()` will
correctly skip the order when there's no session, but the inverse —
`if is_session_expiring_soon(): refresh_session()` — triggers a
re-handshake immediately even though there's nothing to refresh, the
session just needs to be *created*.

This is a UX issue. Fix: split into `needs_login()` (no session) vs
`expiring_soon()` (existing session, < N min remaining).

### P2-8 — `_get_csrf` raises `AvanzaSessionError` from inside `_with_browser_recovery`, classified as non-browser-dead, propagates

**File:** `portfolio/avanza_session.py:271-291`, `212-232`.

If the cookie jar is rotated mid-flight and `AZACSRF` is missing,
`_get_csrf` raises `AvanzaSessionError("No AZACSRF cookie found — session
may be invalid")`. This is raised inside `_op(ctx)` of `api_post`, the
exception is caught by `_with_browser_recovery`, `is_browser_dead_error`
returns `False`, the exception re-raises.

So the api_post call raises `AvanzaSessionError`. The caller's `with
avanza_order_lock(...)` block exits cleanly (lock released). But:

- `_place_order` (line 620-622): `result = api_post(...)` is followed by
  `status = result.get("orderRequestStatus", "UNKNOWN")` — if api_post
  raised, this line never runs, the exception propagates. OK.
- `_execute_confirmed_order` (line 349-402) catches the `Exception`,
  marks the order `"error"`, notifies. OK.
- `cancel_stop_loss` (line 916): `result = api_delete(...)` is inside a
  `try/except Exception`. OK.

So error propagation is consistent. The complaint is more subtle: a
**transient** CSRF rotation is recoverable (re-fetch the token); the
current code propagates the exception immediately. If you've just gone
through a 5xx → relaunch cycle in `_with_browser_recovery`, the new
context's cookie jar takes a few hundred ms to populate (the initial
`goto` is `domcontentloaded`, not full load). The CSRF cookie may be
missing during this window and the *retry* of api_post also raises
`AvanzaSessionError`, even though a wait-and-retry would have succeeded.

**Fix:** Inside `api_post`, on `AvanzaSessionError` from `_get_csrf`,
do a one-shot `wait_for_url` or sleep a few hundred ms and retry. Or
ensure `_open()` blocks until the cookie jar is populated.

### P2-9 — `get_open_orders` route for the legacy fallback uses `data.get("orders", [])` but the legacy fallback returns the full deal endpoint

**File:** `portfolio/avanza_session.py:658-661`.

`/_api/trading/rest/deals-and-orders` returns both deals and orders.
Filtering `data.get("orders", [])` is correct for the order branch, but
the field name could change to `"openOrders"` (as the primary endpoint
in line 655 already attempts). The fallback doesn't try both.

**Fix:** `orders = data.get("orders", data.get("openOrders", []))`.

### P2-10 — `place_trailing_stop` validates `trail_percent` only via downstream `place_stop_loss`

**File:** `portfolio/avanza_session.py:814-846`.

```python
return place_stop_loss(
    orderbook_id=orderbook_id,
    trigger_price=trail_percent,   # <-- passed as MONETARY-style float
    sell_price=0,
    ...
    trigger_type="FOLLOW_DOWNWARDS",
    value_type="PERCENTAGE",
)
```

`place_stop_loss` validates `sell_price <= 0` only when not trailing
(line 759). For trailing, it skips. So `place_trailing_stop` can be
called with `trail_percent = 0` (no protection) or `trail_percent = 100`
(stop fires immediately on first downtick of 100%, never). Neither is
useful but neither is caught.

**Fix:** In `place_trailing_stop`, validate `0.1 <= trail_percent <= 50`
explicitly. The lower bound prevents instant-fire; the upper bound
sanity-checks against typo.

### P2-11 — `_check_telegram_confirm` `allowed_user` coerce can match Python's `None` str

**File:** `portfolio/avanza_orders.py:255-256, 304`.

```python
raw_allowed_user = config.get("telegram", {}).get("allowed_user_id")
allowed_user = str(raw_allowed_user) if raw_allowed_user is not None else None
```

OK that's correct — `None` config → `None` allowed_user → no check.
But what if config has `allowed_user_id: ""` (empty string)? Then
`raw_allowed_user is not None` is True, `allowed_user = ""`. Sender ID
check at line 304: `str(sender_id) != allowed_user` — almost no telegram
user ID is `""`, so every message is dropped. User configures empty
allowed_user_id thinking it disables the check; instead it locks them out.

**Fix:** Treat empty string the same as None: `allowed_user = str(raw)
if raw else None`.

### P2-12 — `_decimal_places` for tick rounding uses string formatting and may miscount precision

**File:** `portfolio/avanza/tick_rules.py:130-135`.

```python
def _decimal_places(value: float) -> int:
    s = f"{value:.10f}".rstrip("0")
    if "." in s:
        return len(s.split(".")[1])
    return 0
```

`f"{0.0001:.10f}"` → `"0.0001000000"`, rstrip("0") → `"0.0001"`, decimals
= 4. OK.

`f"{0.005:.10f}"` → `"0.0050000000"`, rstrip → `"0.005"`, decimals = 3. OK.

But: `f"{0.1:.10f}"` → `"0.1000000000"`, rstrip → `"0.1"`, decimals = 1. OK.

`f"{0.0050001:.10f}"` (a tick value with float precision noise) →
`"0.0050001000"`, rstrip → `"0.0050001"`, decimals = 7. So a tick value
that suffered float representation noise upstream gets a precision
multiplier of 10^7 = 10 million, and the subsequent `round(tick * 10**7)`
hides the noise — but `price_int = price * 10**7` is also at 10M scale,
so small float errors in `price` get amplified. In practice ticks are
0.0001, 0.001, 0.005, 0.01 etc. — all clean — so this is fine today, but
if Avanza ever returns a noisy tick from the API, the rounding silently
becomes wrong.

**Fix:** Use `decimal.Decimal` for the rounding arithmetic. Tick sizes
are inherently decimal, not float.

### P2-13 — `AvanzaStream` does not verify CometD subscriptions succeeded

**File:** `portfolio/avanza/streaming.py:182-190`.

```python
def _subscribe_channel(self, channel: str) -> None:
    sub_msg = [{...}]
    self._ws.send(json.dumps(sub_msg))
    logger.debug("Subscribed to %s", channel)
```

Sends the subscribe message, doesn't wait for the `/meta/subscribe`
response confirming success. If the server rejects (channel doesn't
exist, no permission, rate limit), the client thinks it's subscribed
but no data arrives. Callbacks silently never fire.

The read loop (line 192-229) does dispatch incoming messages, including
`/meta/subscribe` responses, but `_dispatch_message` line 236-237 drops
all `/meta/*` channels: `if channel.startswith("/meta/"): return`.

So subscribe failures are completely invisible.

**Fix:** Track outstanding subscriptions in a dict; in `_dispatch_message`
inspect `/meta/subscribe` responses, log non-`successful` ones, and on
repeated failure re-handshake.

### P2-14 — `AvanzaStream._read_loop` ignores `WebSocketException` subclasses other than the two it names

**File:** `portfolio/avanza/streaming.py:208-214`.

Only catches `WebSocketTimeoutException` and `WebSocketConnectionClosedException`.
Other websocket-client exceptions (e.g., `WebSocketBadStatusException` on
401 — which IS a session-died signal) propagate up to `_run_loop`'s
`except Exception` and trigger reconnect with backoff. Fine, but the
401-specific case should re-handshake instead of just reconnecting with
stale credentials — a 401 means the auth session is dead and reconnecting
won't help until a re-auth has happened. The backoff grows to 60s and stays
there.

**Fix:** Classify 401 separately; on persistent 401, stop and surface to
the caller (via a `failed` event or callback) so the caller can re-init
the auth singleton.

### P2-15 — `_account_id` cache in `avanza_client.py` is a process-wide singleton with no reset hook for account whitelist changes

**File:** `portfolio/avanza_client.py:239-279`.

```python
_account_id: str | None = None

def get_account_id() -> str:
    global _account_id
    if _account_id is not None:
        return _account_id
    ...
```

`reset_client()` and `reset_session()` exist, but neither resets
`_account_id`. If the operator updates `ALLOWED_ACCOUNT_IDS` at runtime
(test or a dynamic config reload), the cached `_account_id` still points
to the pre-change account.

**Fix:** Add `reset_account_id()` or fold it into one of the existing
resets.

### P2-16 — `find_instrument` in `avanza_client.py` returns whatever the library returns; no `tradeable` filter

**File:** `portfolio/avanza_client.py:123-134`.

Caller may pick the first hit which is delisted, restricted, or a
warrant the user doesn't have the option agreement for. Result: Layer 2
generates a trade hint against the wrong instrument.

**Fix:** Filter out `tradeable == False` results. Or document that the
caller is responsible.

---

## P3 — Low

### P3-1 — `avanza_session.py:42` comment misleadingly says ALLOWED_ACCOUNT_IDS is "derived" from DEFAULT_ACCOUNT_ID

The code derives it correctly (`ALLOWED_ACCOUNT_IDS = {DEFAULT_ACCOUNT_ID}`).
But `avanza_client.py:32` is an independent literal. The comment in
`avanza_session.py` reads as if the derivation is system-wide.

### P3-2 — `avanza_session.STORAGE_STATE_FILE` is read by Playwright with no permission check

`data/avanza_storage_state.json` contains the session cookies. If the file
permissions are wrong (world-readable on Unix; default ACL on Windows),
this is a credential leak. No check anywhere.

### P3-3 — `avanza_resilient_page._open` waits a fixed 2000ms after `goto`

Hardcoded value with no override. On a slow VM it may not be enough; on
a fast machine it wastes 2 seconds every relaunch.

### P3-4 — `place_order` accepts `volume = 1` but Avanza warrants are usually traded in lots

Avanza accepts 1-share orders, but for some warrant series there's a
minimum lot size — Avanza returns a rejection. Not surfaced as a
distinct error class.

### P3-5 — `get_positions` returns `last_price` zero when quote is missing

Quote-missing case (`latest = {}`) gives `last_price = 0`. Caller doing
P&L calculation divides by `last_price` and gets a `ZeroDivisionError`
or worse, a silently-wrong %. No `None`-pass-through option.

### P3-6 — `avanza_tracker.fetch_avanza_prices` catches broad `Exception` and logs warning per instrument

Eats all errors including programming errors. Hard to spot a real bug
(`AttributeError`, etc.) in a noisy log.

### P3-7 — `place_stop_loss_no_page` returns dict result with no `stoplossOrderId` on failure

`avanza_control.place_stop_loss_no_page` (line 356-364) returns
`(ok, result)` where `result` is whatever `place_stop_loss` returned.
On failure, `result` has no consistent shape — the test surface for
`stop_id` extraction must handle missing key.

### P3-8 — `delete_stop_loss_no_page` checks `result.get("errorCode")` but Avanza never returns this key on success

Line 401: `if isinstance(result, dict) and result.get("errorCode"):` —
this assumes failure responses have an `errorCode` key. Most success
responses are `{}` (empty), but some Avanza endpoints return
`{"errorCode": null}` on success which evaluates falsy — fine. But if
Avanza ever changes to a string error code like `"errorMessage"` instead,
the failure detection silently breaks.

### P3-9 — `AvanzaStream.on_orders` / `on_deals` concatenate account IDs with comma but the CometD path may need different delimiter

```python
channel = "/orders/_" + ",".join(account_ids)
```

Hardcoded format. Has no test for multi-account subscription. If Avanza
changes the format, all order/deal callbacks silently stop firing.

### P3-10 — `place_stop_loss` payload sets `triggerOnMarketMakerQuote: True` unconditionally

Triggering on MM quote is faster but can fire on a stale MM quote during
illiquid windows (pre-open, after-hours commodity warrant trading).
Not configurable. Hardcoded `True` matches Avanza UI default but should
be a kwarg for low-liquidity instruments.

### P3-11 — `avanza_orders.PENDING_FILE` not locked

Two processes calling `request_order` simultaneously can lose one of the
two orders to the last-writer-wins behaviour in `atomic_write_json`.
Unlikely in practice (only Layer 2 calls request_order, single-process),
but no formal protection.

### P3-12 — `is_browser_dead_error` substring match on "has been closed" is broad

Catches strings like "websocket connection has been closed" which is the
streaming module's own message, unrelated to the trading browser. If a
streaming exception leaks into the avanza_session call path, the trading
browser is needlessly torn down.

### P3-13 — `_pw_lock` reentrance permits nested code to acquire the file lock multiple times under one Playwright lock holder

If a single Python thread enters `place_order`, takes `_pw_lock`,
calls `avanza_order_lock` (file-lock), then deep inside the cross-process
lock acquires another `avanza_order_lock` (e.g., a nested helper),
filelock.FileLock re-acquire on the same process is a no-op — but the
test surface for the nesting case is missing.

### P3-14 — `OrderResult.from_api` returns `status` as `str(status)` but accepts both `orderRequestStatus` (top-level) and `status` (nested in some responses)

The mixing of `orderRequestStatus` vs `status` is consistent across all
the parsers in `types.py` but inconsistent with Avanza's docs. A response
with both fields chooses `orderRequestStatus` (line 186) — fine — but a
response with only `status` (some endpoints?) would parse correctly.
Worth pinning a test against the actual API shape.

### P3-15 — `_save_pending` writes the whole list every cycle even if nothing changed

Disk write per cycle, small but adds I/O. With `EXPIRY_MINUTES = 5` and
60s cycle, that's ~12 unchanged writes for one pending order.

### P3-16 — `avanza_account_check.DISALLOWED_CATEGORY_FRAGMENTS = ()` makes the verifier a logger only

Per the comment that's intentional now, but the function name
`verify_default_account` implies enforcement. Rename to `inspect_default_account`
or document the no-op semantics more loudly.

### P3-17 — `_walk_accounts` in `avanza_account_check.py` uses string label heuristic that's fragile

Falls back through `cat.get("name")`, `cat.get("type")`, `cat.get("category")`.
If Avanza adds a fourth field (e.g., `displayName`) as the only label,
the category is `""` and the disqualification check returns False (line 113).
Coverage works today but is single-API-change away from breaking.

### P3-18 — `place_trailing_stop` documents trail_percent as e.g. 5.0 but doesn't enforce it

```python
trail_percent: float
```

A caller passes `trail_percent=0.05` thinking it's a fraction; the order
goes through with 0.05% trail (instant fire). No range check.

### P3-19 — `find_instrument` (line 132 of avanza_client) returns the raw library result without type adapter

Other functions return parsed structures (positions, orders). Search
returns library-native objects. Inconsistent caller experience.

### P3-20 — `streaming.py` daemon thread doesn't catch exceptions from `_dispatch_message` at the loop level

Line 242-251 wraps each callback, but if `json.loads(raw)` succeeds but
returns a non-dict that doesn't unpack correctly in `_dispatch_message`,
the thread crashes silently. The reconnect happens but the user-supplied
callback (which they registered with `on_quote`) never fires again
because callbacks list is preserved across reconnect — but only if the
reconnect actually happens. If the entire daemon thread died, no
reconnect.

---

## Tests missing

These are missing tests, not the same as the issues above (which may
have partial tests).

- **Order lock + browser-recovery interaction**: there's no test for
  what happens when `_with_browser_recovery` retries a `place_order` —
  duplicate order on the broker is the P0-4 risk and no test exercises
  the simulated death-mid-POST scenario.
- **`cancel_all_stop_losses_for` with `poll_interval=0`**: P0-5 isn't
  covered. Need a test that asserts the poll rate stays sane.
- **`place_stop_loss` with `trigger_price <= 0`**: P0-7 — no test
  catches the negative/zero trigger.
- **`place_trailing_stop` with `trail_percent <= 0` or `> 100`**: P2-10
  / P3-18 — no test catches these.
- **`get_buying_power` with one field missing**: P1-10 — no test where
  `buyingPower` exists but `totalValue` is missing. The multi-shape
  fixture covers shape drift but not field drift.
- **`cancel_order` route 404 fallback**: P1-3 — no test simulates the
  Avanza-changed-the-route case.
- **`portfolio/avanza/trading.delete_stop_loss` with non-"404"
  exception containing "404" in body**: P1-11.
- **`metals_avanza_helpers.delete_stop_loss` returning False on 404**:
  P1-12 — no test compares the three implementations head-to-head.
- **`_check_telegram_confirm` with `allowed_user_id=""`**: P2-11.
- **`_check_telegram_confirm` with chat_id misconfigured to None/false**:
  P2-6.
- **`AvanzaStream` subscribe failure detection**: P2-13.
- **`AvanzaStream` 401 mid-stream**: P2-14.
- **`verify_session` with transient 5xx**: P1-15 — no test that
  asserts the singleton is NOT torn down on transient failures.
- **`_get_csrf` cookie-jar empty window after relaunch**: P2-8.
- **`request_order` + 5-min stale-price execution**: P1-8 — no test
  asserting drift detection on confirm.
- **`AvanzaClient` / unified-package account whitelist enforcement**:
  P0-2 — no test catches a pension-account override.
- **`avanza/account.get_buying_power` returning zeroes for missing account**:
  P0-3 — no test asserts the fail-loud contract.
- **Tick rounding before placement**: P1-13 — no test verifies that
  a sub-tick price would have been rejected pre-flight.

---

## Cross-cutting observations

1. **Three trading stacks, three idempotency models for stop-loss DELETE.**
   This is the highest-leverage cleanup — consolidate to one
   implementation. The session path is the most correct (uses HTTP status,
   treats 404 as success, uses `avanza_order_lock`).
2. **The `avanza_order_lock` is a cross-process file lock; in-process
   thread serialization rides on `_pw_lock`.** Either document this
   clearly in `avanza_order_lock.py` and `avanza_session.py`, or add a
   thread-local mutex so the order-lock module truly owns ordering.
3. **`avanza_session.py` is 1248 lines and growing.** It's the canonical
   trading surface but has comments referencing both `_with_browser_recovery`
   (added 2026-04-13), the `RLock` upgrade (2026-04-11), the buying-power
   shape drift handling (2026-04-09), and the cancel-rollback re-arm flow
   (added recently). A split would help: `avanza_session_auth.py`,
   `avanza_session_orders.py`, `avanza_session_stops.py`,
   `avanza_session_quotes.py`.
4. **`portfolio/avanza/*` was added as a "cleaner" replacement but
   lacks the safety primitives that grew organically in
   `avanza_session.py`** — order lock, account whitelist, multi-shape
   buying-power, browser recovery, fail-loud semantics. Either fold
   them in, or document the unified package as "thin wrapper, not
   trading-grade".
5. **Three independent `DEFAULT_ACCOUNT_ID = "1625505"` literals exist**
   (`avanza_session.py:35`, `avanza_client.py:32`, `avanza/client.py:19`).
   The Book of Grudges entry "Only show ISK account 1625505" suggests
   the author has burned themselves once on this — codify it.
6. **None of the trading helpers tick-round before placement** even
   though `tick_rules.py` exists for this exact purpose. Layer 2 in
   particular hallucinates higher-precision prices than the broker
   accepts.
7. **The CONFIRM token UX has a comment saying "an operator reading
   agent.log can read it if they need to confirm out-of-band" but the
   code logs only the first 4 of 6 chars.** Pick one — either log the
   full token (with a documented "agent.log is sensitive" warning) or
   remove the comment.
8. **Account category check `verify_default_account` is now a logger.**
   `DISALLOWED_CATEGORY_FRAGMENTS = ()` means it never disallows
   anything. Per the docstring this is intentional after 2026-05-11
   discussion, but the function name still implies enforcement. Rename
   or restore a meaningful whitelist of trading-class category
   strings.
9. **No process-level abuse / rate-limit headers respected.** Avanza
   doesn't publish a public limit, but their reverse-proxy 429s land
   here as a generic `RuntimeError("Avanza API error 429: ...")` with
   no backoff. `http_retry.py` exists and honours Telegram's
   `retry_after` (line 43-49) — Avanza calls don't go through it.
   Either route ctx.request through `fetch_with_retry` or add an
   explicit 429-aware backoff in `api_get`/`api_post`/`api_delete`.
10. **Storage state file** `data/avanza_storage_state.json` is the
    crown jewel — full session cookies. There's no integrity check
    (HMAC against `os.environ["AVANZA_STATE_KEY"]`), no permission
    enforcement on the file, no warning if its mtime is older than
    24h (BankID session ~24h validity). A future cleanup that
    quarantines this file behind explicit permissions checks would
    be defence in depth.
