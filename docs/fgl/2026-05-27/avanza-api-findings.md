# Avanza API Subsystem — Adversarial Review

Reviewer: opus 4.7 (1M)
Date: 2026-05-27
Branch: main (uncommitted scope is data/* + scripts/write_morning_briefing.py, none in avanza subsystem)

## Summary

The avanza subsystem has matured significantly — `avanza_session.py` is now
the canonical "safe" path with H7/H8/BUG-211 guards (whitelist, min/max
order size), `ResilientPage` and `_with_browser_recovery` give the
long-running Playwright contexts auto-relaunch, and `avanza_order_lock`
serializes cross-process placement. Stop-loss vs order endpoint
discipline is enforced. The Mar-3 grudge of using order/new for
stop-losses is unreachable in current code paths.

However, three classes of problems remain that put real money at risk:

1. **Defense-in-depth is uneven across the three order paths.** Of the
   three places that POST to `order/new`, only one (`avanza_session.
   _place_order`) carries the full guard stack (whitelist, MIN 1000 SEK,
   MAX 50K SEK, price/volume validation). The page-based path
   (`metals_avanza_helpers.place_order`, used by `metals_loop` and
   `metals_swing_trader`) has NO whitelist, NO max-order cap, NO
   price/volume validation; the TOTP path (`avanza_client._place_order`,
   used by `avanza_orders.py` Telegram-confirm flow) has price/volume
   but NO max-order cap. The same gap applies to `cancel_order` /
   `cancel_stop_loss` and to the page-based `place_stop_loss`.

2. **A stale stop-loss DELETE that silently no-ops is still in tree.**
   `data/place_stoploss_once.py:115` calls
   `api_delete(f"/_api/trading/stoploss/{stop_id}")` without an account
   id. The correct path is `/_api/trading/stoploss/{accountId}/{stopId}`.
   Avanza returns 404 on the malformed path, and `api_delete` treats 404
   as success (line 377), so the script reports "OK" while the old
   stop-loss is still alive. Two issues prior reviews flagged this
   (2026-04-29, 2026-05-02 follow-up plan); never fixed.

3. **Order POSTs are not idempotent.** `_with_browser_recovery` will
   retry the entire op on a Playwright TargetClosedError. If the browser
   dies AFTER the POST hit Avanza but BEFORE the response is read in the
   Python side, the retry posts a fresh `order/new` with no client-side
   nonce — Avanza accepts both, resulting in a double-placed order.
   `avanza_orders.py` per-order tokens guard the Telegram-confirm flow
   but the broker-side idempotency is missing for every path.

Also called out: 5-minute confirmation window allows SELL price drift,
TOTP `accountId` discovery is one-time only with no rotation guard, and
`get_buying_power` returning `None` silently could starve order sizing
in any caller that treats `None` as zero (the rule says callers MUST
handle `None`, but it's worth a runtime grep — out of scope here).

---

### [P0] Page-based `place_order` has no ALLOWED_ACCOUNT_IDS guard
**File:** `data/metals_avanza_helpers.py:253-292`
**Issue:** `place_order(page, account_id, ob_id, side, price, volume)`
accepts any `account_id` from the caller and stuffs it directly into
the order payload (line 285). No whitelist check. The corresponding
guards in `portfolio/avanza_session.py:_place_order` (lines 590-592)
and the wrapper `portfolio/avanza_control.place_order` (lines 130-134)
also forward whatever account_id is given. The metals loop currently
uses `DEFAULT_ACCOUNT_ID` everywhere, but any code path that
constructs an `account_id` from an external source (a JSON file, a
Telegram message body, an LLM-generated argument) reaches the broker
unchecked.

This was previously documented in `docs/PLAN_avanza_followups_20260502.md`
(item 1) and `docs/adversarial-review-r5/subsystem-5-avanza-api.md`
(AV-R5-1). Not implemented.

**Impact:** Wrong account leak (pension 2674244 is on the same login).
Memory `feedback_isk_only.md` makes this a sacred boundary.
**Fix:** Add to `metals_avanza_helpers.place_order` and `place_stop_loss`:
```python
from portfolio.avanza_session import ALLOWED_ACCOUNT_IDS
if str(account_id) not in ALLOWED_ACCOUNT_IDS:
    return False, {"error": f"non-whitelisted account {account_id}"}
```
Same for `avanza_control.place_order` and `place_stop_loss` page-based
wrappers (lines 130-156) so the page-based path is gated even when
called directly.
**Confidence:** High — the wrapper signature explicitly accepts an
unvalidated account_id and the JS payload uses it verbatim.

---

### [P0] `place_stoploss_once.py` issues malformed DELETE that silently no-ops
**File:** `data/place_stoploss_once.py:115`
**Issue:**
```python
def delete_stoploss(stop_id: str) -> dict:
    if DRY_RUN:
        return {"dry_run": True}
    return api_delete(f"/_api/trading/stoploss/{stop_id}")
```
The Avanza endpoint is `/_api/trading/stoploss/{accountId}/{stopId}`
(confirmed at `portfolio/avanza_session.py:916` and
`portfolio/avanza_control.py:227,397`). Sending it without `accountId`
hits a non-existent path, Avanza returns 404, and
`portfolio/avanza_session.api_delete:377` returns
`{"ok": 200 <= http_status < 300 or http_status == 404}` → 404 = ok.

The `main()` flow then prints "OK" for every "deleted" stop and
proceeds to place a NEW cascading stop-loss on top of the still-alive
old one (line 175-176, "Delete old single stop-losses" → "Placing 6
cascading stops"). Result: 2× the volume encumbered, hitting Avanza's
`short.sell.not.allowed` rejection on any subsequent SELL on that
position.

Worse: if the old stop triggers in the same session, both the cascade
and the original SL fire, oversells the position, leaves an
unintended short (which Avanza will then close at a worse price).

This was reported in `docs/ADVERSARIAL_REVIEW_2026-04-29.md` line 91
and `docs/PLAN_avanza_followups_20260502.md` item 2. Not fixed.

**Impact:** Stop-loss silently fails to cancel; cascading-stop
operator workflow over-encumbers position and risks oversell.
**Fix:**
```python
def delete_stoploss(stop_id: str, account_id: str = "1625505") -> dict:
    if DRY_RUN: return {"dry_run": True}
    return api_delete(f"/_api/trading/stoploss/{account_id}/{stop_id}")
```
Also rip out the bare `DELETE_IDS = [...]` constants at line 19 and
require the operator to pair each id with its account, OR call
`cancel_stop_loss(stop_id, ACCOUNT_ID)` from `avanza_session.py` which
already does this correctly.
**Confidence:** Very high — endpoint shape is documented, and 404 →
success is the literal code at `avanza_session.py:377`.

---

### [P0] TOTP path `_place_order` has no MAX_ORDER_TOTAL cap
**File:** `portfolio/avanza_client.py:327-368`
**Issue:** `avanza_session._place_order` carries BUG-211's
`MAX_ORDER_TOTAL_SEK = 50_000.0` ceiling (line 602-606). The TOTP-based
`avanza_client._place_order` enforces only `volume >= 1, price > 0`
and skips the cap. `avanza_orders.py:_execute_confirmed_order` (line
355-365) goes through `avanza_control.place_buy_order` →
`avanza_client.place_buy_order` → `_place_order` (TOTP), so a Telegram-
confirmed order with a malformed volume (LLM hallucination, unit error,
accidental shares→units confusion) lands directly at Avanza with no
ceiling.

The MIN-1000-SEK check (H8) is also missing on the TOTP path.

**Impact:** Full-account exposure from a single malformed order on the
Telegram-confirmation path. CLAUDE.md memory
`feedback_risk_tolerance.md` says user accepts 10-20% knockout risk —
this guard catches the unintended 100% case.
**Fix:** Lift the BUG-211 + H8 checks out of `avanza_session._place_order`
into a shared `_validate_order(orderbook_id, price, volume, account_id,
side)` helper and call it from all three `_place_order` paths
(`avanza_session.py`, `avanza_client.py`, `metals_avanza_helpers.py`).
The whitelist check should live in the same helper for symmetry.
**Confidence:** High — direct read of `avanza_client._place_order`
(only volume/price checks before lock acquire).

---

### [P1] `_with_browser_recovery` retry can double-place orders
**File:** `portfolio/avanza_session.py:212-232,294-344`
**Issue:** `api_post("/_api/trading-critical/rest/order/new", payload)`
is wrapped in `_with_browser_recovery`, which retries the op once on
any `is_browser_dead_error(exc)`. The retry runs a fresh
`ctx.request.post(url, data=body_data, ...)` — same body, same
endpoint, no client-side idempotency token. If the browser died AFTER
the request hit the Avanza edge (network success, response not yet
read) and BEFORE `resp.text()` returned, the retry submits a duplicate.

Avanza's `order/new` accepts no `clientOrderId` field in the current
payload (`{accountId, orderbookId, side, condition, price, validUntil,
volume}` — see lines 608-616), so the broker cannot dedupe.

Same logic applies to `cancel_order` (line 645), `place_stop_loss`
(line 801), and `cancel_stop_loss` (line 916). For cancels, double-
issuing is idempotent on the broker side. For `place_stop_loss`, a
duplicate creates two stops on the same position — same over-
encumbered failure mode as the place_stoploss_once.py bug.

**Impact:** Real risk during memory pressure, OS sleep wakeup, or
high-concurrency periods when the browser is more likely to die mid-
flight. Doubles the per-leg exposure and can blow MAX_ORDER guards
because each placement passes validation individually.
**Fix:** Two options.
  (a) Conservative: do NOT retry `api_post` for mutating endpoints.
      Add a `mutating=True` flag to `_with_browser_recovery` and skip
      the retry; let the caller decide whether to re-attempt after
      reconciling the open-orders view (`get_open_orders`).
  (b) Belt-and-suspenders: before the retry, GET `/_api/trading/rest/
      orders` and check whether an order with the same
      `(orderbookId, side, price, volume)` placed in the last few
      seconds already exists. If yes, treat the retry as success
      with the existing orderId.
Pick (a) for `order/new` and `stoploss/new`; (b) is overkill given the
existing avanza_order_lock + lock-busy-error semantics.
**Confidence:** High on the failure mode (read of
`_with_browser_recovery` + `api_post` shows no idempotency); medium
on real-world frequency (depends on browser death timing).

---

### [P1] `cancel_order` and `cancel_stop_loss` have no whitelist guard
**File:** `portfolio/avanza_session.py:633-645,888-916`
**Issue:** Both functions accept `account_id: str | None = None` and
fall back to `DEFAULT_ACCOUNT_ID` without checking
`ALLOWED_ACCOUNT_IDS`. Same gap exists in `avanza_control.
delete_order_live` (line 159-210) and `delete_stop_loss` (line 213-256)
— both forward `account_id` straight to the JS payload.

This was flagged as AV-R5-3 in
`docs/adversarial-review-5/SYNTHESIS.md` and not implemented.

**Impact:** Lower than `place_order` leaks since a cancel on the wrong
account just gets a "not found" — but it does leak that the account
exists and can be probed. Defense-in-depth violation.
**Fix:** Same whitelist check used in `_place_order` (line 591-592)
applied to `cancel_order`, `cancel_stop_loss`, `delete_order_live`,
`delete_stop_loss`.
**Confidence:** High.

---

### [P1] Page-based `place_order` skips price/volume validation
**File:** `data/metals_avanza_helpers.py:253-327`
**Issue:** `avanza_session._place_order` rejects `volume < 1` and
`price <= 0` (lines 584-587) before doing anything else. The page-
based `place_order` falls straight through to `JSON.stringify(payload)`
with whatever floats/ints the caller provided. A `price=0` payload
would let Avanza interpret it as a market order (or reject for some
instruments) — either way, unintended behavior.

`metals_loop.py` and `metals_swing_trader.py` are the callers; both
compute prices from live quotes, but if the quote fetch returned 0 or
None (degraded quote, weekend after-hours data, bug in upstream
arithmetic), the order is silently placed at 0.

Also: `if 0 < total < 1000.0` (line 276) silently passes orders with
`total == 0` (e.g. volume = 0 OR price = 0) without warning OR refusing
— the warning trigger requires positive total.

**Impact:** Unintended order semantics (zero-price). Combined with the
no-whitelist issue above, a buggy upstream computation reaches Avanza
unfiltered.
**Fix:** Mirror the avanza_session._place_order validation at the top
of `metals_avanza_helpers.place_order` and `place_stop_loss`:
```python
if volume < 1 or price <= 0:
    return False, {"error": f"invalid price={price} volume={volume}"}
```
**Confidence:** High.

---

### [P1] Pre-flight bid/ask price-sanity check is missing
**File:** `portfolio/avanza_session.py:575-630` (and all sibling
place_order paths)
**Issue:** No path verifies the limit `price` is within a sane band
around the current bid/ask before placing. A BUY at 10× current ask
(LLM unit error, decimal-point typo, stale price from a forecast) is
accepted by Avanza and either fills instantly at the ask (giving away
money) or sits as a non-fillable order. A SELL at 0.1× current bid
fills instantly at bid (giving away money).

The Mar-3 grudge was the inverse case (regular order API as stop-loss
instantly filled at bad price) — same outcome from a different mistake.

The metals stop-loss path has `feedback_mini_stoploss.md` knowledge
("never place SL near MINI barrier") but no automated check that
trigger > barrier × 1.03 or sell_price > bid × 0.97, etc.

**Impact:** Single bad limit price = give away the difference between
limit and market. With a 5x warrant, a 10% mispricing = 50% portfolio
hit.
**Fix:** Before any `_place_order` POST, fetch
`get_quote(orderbook_id)` (already implemented at line 667-673) and
check `0.85 * bid <= price <= 1.15 * ask` for both sides. Raise/return
on violation. Tighter band (e.g. 5%) is even better but tests will
need updating.

For stop-losses, add the barrier-distance check from the
`feedback_mini_stoploss` memory — refuse `trigger_price` within 3% of
`keyIndicators.barrierLevel` (already fetched in `fetch_price`).
**Confidence:** Medium — this is a missing safety net rather than a
present bug, but the system has documented prior incidents (Mar 3) of
exactly this class.

---

### [P1] 5-minute Telegram CONFIRM window allows price drift
**File:** `portfolio/avanza_orders.py:41,82-146,350-415`
**Issue:** `EXPIRY_MINUTES = 5`. `request_order` captures the price at
intent time. `_execute_confirmed_order` re-uses that price 0-5 minutes
later. For volatile warrants (5x silver, BTC during news), 5 minutes
is enough for a 2-5% move.

Limit semantics partially mitigate: BUY limit above market fills at
market (no overpay), SELL limit below market fills at market (no
under-sell). But:
  - BUY limit above market = fills at current ask, NOT the intended
    "wait for pullback" — the user thought they were buying at 100,
    they bought at the current 103.
  - SELL limit below current market = sells at the user's stale 100,
    not the current 105 — giveaway of 5%.

No price re-check at confirmation time. No "stale price → ask the
user to re-confirm with current quote" guard.

**Impact:** Silent loss-on-execution proportional to delay × volatility.
Worst case = user CONFIRM'd just before going AFK for 5 minutes; price
moved 5%; they took the haircut.
**Fix:**
  (a) Drop EXPIRY to 60 seconds for volatile asset classes (crypto
      warrants, MINI products). Keep 5 min for equities.
  (b) At execution time, re-fetch `get_quote()` and refuse to execute
      if price drifted > `STALE_BAND_PCT` (e.g. 1%) from order intent;
      send Telegram asking to re-confirm with fresh quote.
**Confidence:** Medium-high — semantically correct critique, real-
world frequency depends on user response speed.

---

### [P1] Avanza-account-id discovery cached without rotation guard
**File:** `portfolio/avanza_client.py:239-279`
**Issue:** `_account_id` module-global is set once after the first
`get_account_id()` call and never invalidated. If Avanza re-orders the
overview response (server side) between login session and the cached
read, the cache holds the previous response's answer.

The whitelist check at line 270 catches the case where the candidate
is NOT in `ALLOWED_ACCOUNT_IDS`, so a non-whitelisted account never
gets cached. But: the cache is set once at process start, and any
later restart (new BankID auth, config change) won't refresh until the
process restarts.

This is mostly fine because the whitelist = single account
`{"1625505"}`. Risk is bounded.

**Impact:** Low — whitelist guarantees correctness for the single-
account configuration. Becomes a real bug if the whitelist ever
grows to >1 account (the cache locks to whichever ISK appears first
in the response).
**Fix:** Either (a) document "cache is single-shot and is correct for
1-account whitelists only; widen → invalidate cache too" or (b) drop
the cache and call `get_account_id()` per-request (one extra
overview API call per order — cheap).
**Confidence:** Medium — current single-account scope makes this a
latent rather than active bug.

---

### [P2] `get_buying_power(None)` failure path could starve order sizing
**File:** `portfolio/avanza_session.py:385-539`
**Issue:** The function returns `None` on any failure (HTTP error,
account not found, shape drift). The docstring (line 408-412) is
explicit: "callers must now explicitly handle the `None` case
(previously they silently got `buying_power=0`, which was a
dangerous silent failure)".

Without grepping every caller, the question is whether a downstream
caller still does e.g. `cash = buying_power or 0` (treats None as
zero) — at which point sizing computes zero units, blocks the trade,
no order placed. Less dangerous than the inverse but still a loop
silently doing nothing.

The fishing/grid systems (`fin_snipe.py`, `grid_fisher.py`) consume
buying_power.

**Impact:** Silently skipped trades during transient API outages.
Failure visible only via the logger.warning line; no Telegram alert.
**Fix:** Either (a) audit all callers (search
`get_buying_power\|fetch_account_cash` in `portfolio/` + `data/`) to
verify they explicitly handle None, or (b) add a Telegram critical
alert on None to surface the outage.
**Confidence:** Medium — known risk, not directly observed in this
review.

---

### [P2] `cancel_all_stop_losses_for` rollback set may include never-armed stops
**File:** `portfolio/avanza_session.py:1029-1116`
**Issue:** CODEX-7's fix at line 1074-1084 correctly removes
DELETE-accepted ids that show up alive in the verification poll. But
the `snapshot` returned (line 1031-1032) is still the full pre-cancel
list, so `rearm_stop_losses_from_snapshot` will try to re-place the
stops the cancel actually rejected. The broker will reject the
duplicates (or accept them, creating two stops on the same volume) —
same over-encumbered failure mode as the place_stoploss_once bug.

The filter at line 1083-1084 only protects `cancelled`. Callers using
`snapshot` directly for rollback don't get the same filter.

**Impact:** Rollback path could over-place stops on the position.
Bounded by the per-place 1000-SEK / 50K-SEK guards, but adds noise
and triggers `short.sell.not.allowed` rejection cycles.
**Fix:** Filter `snapshot` against `remaining` too — don't include
stops that are still alive at the broker in the rollback list:
```python
remaining_set = set(remaining)
snapshot_filtered = [sl for sl in snapshot if sl.get("id") not in remaining_set]
return {..., "snapshot": snapshot_filtered, ...}
```
**Confidence:** Medium — depends on what `rearm_stop_losses_from_snapshot`
does with the input; reading line 1156-1217 confirms it iterates the
full snapshot and calls `place_stop_loss` for each.

---

### [P2] gold_sell_* one-shot scripts use ACCOUNT_ID literal, bypass guards
**File:** `data/gold_sell_debug.py:16`, `data/gold_sell_final.py:17`,
`data/gold_sell_retry.py:19`
**Issue:** All three hardcode `ACCOUNT_ID = "1625505"` and use it
directly with `api_post(...)` for orders (`gold_sell_debug.py:109`,
`gold_sell_retry.py:92`). The path is `_place_order` indirectly so
the whitelist + size guards do apply for the helper-call paths
(`place_sell_order` → `_place_order`). But the `api_post`-direct paths
at `gold_sell_debug.py:109` and `gold_sell_retry.py:92` bypass all of
`_place_order`'s validation entirely — raw payload, no min/max, no
whitelist (well, whitelist holds since the literal is the whitelisted
id).

These are one-shot operator scripts not in the production hot path,
but they're tempting templates for the next "I need to quickly..."
script.

**Impact:** Low for current scripts (literal is whitelisted). Risk is
template propagation — operator copy-pastes for a different
instrument and adjusts the volume, doesn't realize there's no max
guard.
**Fix:** Replace the `api_post(...)` direct calls with
`place_sell_order(...)` (which already exists in the imports). Delete
the literal `ACCOUNT_ID = "1625505"` and import
`DEFAULT_ACCOUNT_ID` instead.
**Confidence:** High — direct read of the three scripts.

---

### [P2] Layer-2 throwaway scripts hardcode and replicate config-load patterns
**File:** `data/layer2_action.py:78-79`, `data/layer2_exec.py:77-78`,
`data/layer2_invoke.py:74-75`
**Issue:** Each opens `config.json` directly with `json.load(open(...))`
instead of using `portfolio.file_utils.load_json`. Per CLAUDE.md rule
4, "Never raw `json.loads(open(...).read())`". These are operator-run
one-shot scripts (no Avanza calls — they only send Telegram), so the
atomic-IO guarantee doesn't matter for correctness here, but they
violate the documented project rule.

Also: same pattern repeated three times, classic copy-paste smell.

**Impact:** Style violation. No live-API risk since these don't touch
the broker.
**Fix:** Use `load_json(BASE/"config.json")` from `file_utils`. Or just
delete these three scripts if they're not part of the production
trigger flow.
**Confidence:** High on the code smell; low on the operational risk.

---

### [P2] `place_stoploss_once.py` safety_checks miss the leveraged-barrier rule
**File:** `data/place_stoploss_once.py:92-108`
**Issue:** Checks are: trigger < bid, trigger >= 3% below bid, sell <
trigger, volume > 0, sell > 0. Memory `feedback_mini_stoploss.md` says
NEVER place SL near MINI warrant barrier. The script processes silver
warrant 2334960 (MINI L SILVER) with trigger=2.30 and a stated
current_bid=5.09 (line 36-37). The script doesn't fetch the barrier
level, so it can't verify trigger is far enough above barrier.

For the supplied operator-known values these specific calls are
safe, but the function as a primitive doesn't enforce the rule for
future operator use.

**Impact:** Operator could re-use the script for a different warrant
with a closer barrier without re-checking. Knockout risk.
**Fix:** Add a barrier-distance check using
`get_instrument_price(ob_id)` which returns the barrier (see
`avanza_session.py:1227-1248`):
```python
if barrier and trigger < barrier * 1.05:
    errors.append(f"trigger {trigger} within 5% of barrier {barrier}")
```
**Confidence:** Medium — the script is operator-curated, but the
guard belongs in code, not in the operator's head.

---

### [P3] `_get_csrf` could log a leak path on multi-context unused branch
**File:** `portfolio/avanza_session.py:271-291`
**Issue:** The function has two branches: one with `ctx` parameter,
one without. The without branch re-enters `_pw_lock` and calls
`_get_playwright_context()`. If logging is ever turned on at DEBUG
level in `_get_playwright_context` it could include the CSRF cookie
value (currently doesn't, but the surface exists). The error message
("No AZACSRF cookie found") doesn't leak the token but does confirm
the auth structure for a hypothetical reader of the log.

**Impact:** Negligible.
**Fix:** None required; mention only for completeness.
**Confidence:** N/A — this is a smell, not a bug.

---

### [P3] `session_remaining_minutes` silently returns None on parse error
**File:** `portfolio/avanza_session.py:106-120`
**Issue:** `datetime.fromisoformat(expires_at)` can raise ValueError;
the bare `except Exception` swallows it and returns None. Caller
`is_session_expiring_soon` then returns True, which is the safe
default ("session expired"). But the warning log line doesn't
distinguish "no expires_at field" from "malformed expires_at" — both
are silently treated as expired.

**Impact:** Low — fail-safe direction. Logged.
**Fix:** None required. Comment for symmetry only.
**Confidence:** N/A.

---

## What's working well

- `avanza_session.py` is now the canonical safe path: H7/H8/BUG-211
  guards, ALLOWED_ACCOUNT_IDS check, atomic CSRF + POST under
  RLock, browser-dead auto-recovery, fail-closed stop-loss read
  (`get_stop_losses_strict`).
- `cancel_all_stop_losses_for` is properly fail-closed on read errors
  and correctly distinguishes verified-cancelled vs DELETE-accepted.
- `avanza_order_lock` cross-process file lock prevents the
  multi-loop buying-power race. 2-second fail-fast prevents stuck-peer
  deadlock.
- `ResilientPage` correctly classifies browser death and tears down
  cleanly before relaunch.
- `verify_default_account` is fail-closed on category mismatch but
  fail-open on transient fetch errors — exactly the right tradeoff
  (per the codex P2 fix comment, line 226-249).
- `avanza_orders.py` per-order `confirm_token` design eliminates the
  three replay races documented in the module docstring.

## Files reviewed

- `Q:\finance-analyzer\portfolio\avanza_session.py` (1248 lines)
- `Q:\finance-analyzer\portfolio\avanza_client.py` (398 lines)
- `Q:\finance-analyzer\portfolio\avanza_control.py` (440 lines)
- `Q:\finance-analyzer\portfolio\avanza_orders.py` (430 lines)
- `Q:\finance-analyzer\portfolio\avanza_account_check.py` (327 lines)
- `Q:\finance-analyzer\portfolio\avanza_order_lock.py` (101 lines)
- `Q:\finance-analyzer\portfolio\avanza_tracker.py` (133 lines)
- `Q:\finance-analyzer\portfolio\avanza_resilient_page.py` (232 lines)
- `Q:\finance-analyzer\data\metals_avanza_helpers.py` (517 lines)
- `Q:\finance-analyzer\data\gold_sell_debug.py` (113 lines)
- `Q:\finance-analyzer\data\gold_sell_final.py` (73 lines)
- `Q:\finance-analyzer\data\gold_sell_retry.py` (102 lines)
- `Q:\finance-analyzer\data\place_stoploss_once.py` (207 lines)
- `Q:\finance-analyzer\data\layer2_action.py` (87 lines)
- `Q:\finance-analyzer\data\layer2_exec.py` (86 lines)
- `Q:\finance-analyzer\data\layer2_invoke.py` (84 lines)
- `Q:\finance-analyzer\portfolio\api_utils.py` (61 lines)
- `Q:\finance-analyzer\portfolio\http_retry.py` (100 lines)

Not reviewed (out of scope): `portfolio\avanza\*` (new TOTP-only
package, has its own 163 tests per repo memory).
