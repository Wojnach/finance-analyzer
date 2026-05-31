# Avanza-API Review

Adversarial whole-file review of the Avanza broker-integration subsystem
(authentication, position reads, order/stop-loss placement & cancel) from the
isolated worktree `Q:/fa-rev-0531`. Live real-money paths.

Scope reviewed:
- `portfolio/avanza_session.py` (page-free BankID REST path — main loop + grid fisher)
- `data/metals_avanza_helpers.py` (Playwright-page path — metals loop + golddigger)
- `portfolio/avanza_client.py` (TOTP path)
- `portfolio/avanza_control.py` (facade re-exporting all three)
- `portfolio/avanza_account_check.py` (startup account verification — source of the seeded error)
- `portfolio/avanza_order_lock.py` (cross-process order lock)
- `portfolio/avanza_resilient_page.py`, `portfolio/avanza_tracker.py`
- `portfolio/avanza/*` package (auth/client/account/trading/streaming/...) — NOT wired into any live loop; used only by `scripts/avanza_smoke_test.py` + tests
- `portfolio/grid_fisher.py`, `data/metals_loop.py`, `portfolio/golddigger/runner.py` (live callers, for tracing)

## Root cause of the seeded daily `avanza_account_mismatch` (06:01)

The daily critical error is NOT an account-routing bug — it is a **session-expiry
bug surfacing through the account verifier**.

Chain:
1. The BankID session is written by `scripts/avanza_login.py` with a hard
   `expires_at = now + 24h` (`max_inactive` default 1440 min, login.py:241-243).
   It is a wall-clock TTL set at login, NOT refreshed by use. There is **no
   auto-reauth** — refresh requires a manual BankID tap on the phone.
2. Each morning a loop restarts (sleep/wake cycle: WakeUp 05:45 → AutoImprove
   06:00 → metals loop init). `metals_loop.py:7042` calls
   `verify_default_account()` during GridFisher init.
3. `verify_default_account` → `_api_get_categorized_accounts()` → `api_get(...)`
   → `_get_playwright_context()` → `load_session()`. Because the session is
   past its 24h TTL, `load_session()` raises
   `AvanzaSessionError("Session expired at 2026-05-30T13:50. Run: python scripts/avanza_login.py")`.
4. `verify_default_account` catches it at avanza_account_check.py:226 and
   **downgrades to `fetch_failed`** (Codex P2 fix, intentional), writing the
   critical-errors entry whose `reason` field literally embeds the
   `AvanzaSessionError` message — exactly the seeded text
   `fetch_failed: Session expired at ... Run: python scripts/avanza_login.py`.

So the misleadingly-named `avanza_account_mismatch` critical fires every day the
session has lapsed past 24h. The system "can't self-recover" because re-auth is
manual by design (BankID). The fix is operational (a daily re-login reminder /
scheduled BankID prompt before market open) plus the code fixes below.

**Does any code path place an order into an expired session?** Not in the live
loops, and this is the one piece of good luck holding the system together — but
it is fragile and rests entirely on the reactive 401 path, not on any proactive
check. See P1-1 and P1-2.

---

## P1 — High

### P1-1 — Order placement never proactively checks session expiry; relies entirely on reactive 401, and the cached Playwright context bypasses the only expiry check
`portfolio/avanza_session.py:134-153` (`_get_playwright_context`),
`:575-635` (`_place_order`), `:725-816` (`place_stop_loss`), `:638-650` (`cancel_order`)

`load_session()` — the ONLY code that inspects `expires_at` — is called inside
`_get_playwright_context()` **only on the cold path** (line 143, guarded by
`if _pw_context is not None: return` at 139-140). In a long-running loop the
context is created once and cached for days, so every subsequent `api_get` /
`api_post` / `api_delete` skips `load_session()` entirely. `_place_order`,
`place_stop_loss`, and `cancel_order` contain **no** `verify_session()` /
`is_session_expiring_soon()` / `load_session()` call before POSTing.

Consequently the *only* thing that prevents an order POST into a dead session is
the broker returning HTTP 401, which `api_post` converts to `AvanzaSessionError`
+ `close_playwright()` (lines 324-329). Causal chain if Avanza ever returns a
non-401 on an expired/half-valid session (e.g. 200 with an auth-redirect HTML
body, or a 302 to login that Playwright follows to a 200): `api_post` would
parse the body, fail `json.loads`, and — because `resp.ok` is true — return
`{"raw": "<html>...login...</html>"}` (line 342). `_place_order` then reads
`result.get("orderRequestStatus")`, gets `None != "SUCCESS"`, logs a warning and
returns. That particular shape is benign for *placement*, but the same
`{"raw": ...}` swallow means a genuinely-rejected order is indistinguishable
from an auth bounce. There is no positive confirmation that the session was live
at POST time.

→ Add a cheap proactive guard at the top of `_place_order` / `place_stop_loss` /
`cancel_order`: if `is_session_expiring_soon(threshold_minutes=0)` (i.e. already
expired by local clock) raise `AvanzaSessionError` *before* the POST. Cheap,
local, no extra round-trip. Additionally, in `_get_playwright_context`,
re-validate `load_session()` (local expiry only) even on the cached path — it is
a sub-millisecond file read and closes the "context launched yesterday, now
expired" window.

### P1-2 — Expired-session order path is silently swallowed in grid_fisher, producing zero trades with only a debug-level journal note
`portfolio/grid_fisher.py:1033-1076` (`_safe_session_call`), `:1443-1453`

`_safe_session_call` catches **every** exception (line 1072, bare `except
Exception`) and returns `default` (None), logging only a journal
`session_call_error`. When the session is expired, the first `place_buy_order`
raises `AvanzaSessionError`, which is swallowed → `place_buy_failed` with
`error="session_call returned None"`. This is the *correct* fail-safe outcome
(no order goes out), but it is logged as a routine placement failure, not as a
session-down condition. Combined with the `verify_default_account` downgrade
(P1 root cause), the system can sit all day making zero trades while the only
signal is a `fetch_failed` critical that is named `avanza_account_mismatch` —
operator sees "account mismatch", not "session expired, re-login needed". This
is the same class of silent-outage failure the CLAUDE.md startup check exists to
prevent (cf. the 3-week Layer 2 auth outage).

→ In `_safe_session_call`, special-case `AvanzaSessionError` (and
`is_browser_dead_error`): re-raise or set a `self._session_dead` flag that the
metals loop surfaces as a distinct, accurately-named critical
(`avanza_session_expired`) with the re-login instruction, instead of folding it
into generic per-instrument placement failures.

### P1-3 — `data/metals_avanza_helpers.place_order` / `place_stop_loss` accept an arbitrary `account_id` with NO whitelist validation; golddigger feeds a config-driven account into them
`data/metals_avanza_helpers.py:253-292` (`place_order`),
`:330-379` (`place_stop_loss`); caller
`portfolio/golddigger/runner.py:180,203,371` passing `cfg.avanza_account_id`;
config default `portfolio/golddigger/config.py:55,154` (`avanza_account_id: str = ""`).

The account-isolation invariant (only ISK `1625505`, never pension `2674244`) is
enforced in `avanza_session._place_order`/`place_stop_loss`
(avanza_session.py:596-597, 755-756) and in `avanza_client` (avanza_client.py:32,
`get_account_id` 270). But the **Playwright-page path used by the live metals
loop and GoldDigger has no such guard** — it stuffs whatever `account_id` it is
handed straight into the order payload (`"accountId": account_id`,
metals_avanza_helpers.py:285, 362). GoldDigger sources that id from
`config.avanza.account_id`/`gd.account_id`, defaulting to `""`. Causal chain: a
config edit (or a copy-paste of the pension account, or a future multi-account
config) routes live BULL/BEAR warrant orders to the wrong account with nothing
in code to stop it. Today it happens to work only because
`metals_loop.ACCOUNT_ID = "1625505"` is hard-coded (metals_loop.py:735) — a
single constant is the entire account-isolation guarantee for the metals path.

→ Add the `ALLOWED_ACCOUNT_IDS` guard to `metals_avanza_helpers.place_order` and
`place_stop_loss` (raise/return `(False, {"error": "non-whitelisted account"})`
if `str(account_id)` not in the whitelist). Import the canonical set from
`avanza_session.ALLOWED_ACCOUNT_IDS` so there is one source of truth. Also
reject empty/`None` account_id explicitly.

---

## P2 — Medium

### P2-1 — `portfolio/avanza` package `place_order`/`place_stop_loss` bypass BOTH the order lock and the account whitelist; re-exported from package root
`portfolio/avanza/trading.py:38-102` (`place_order`), `:213-288` (`place_stop_loss`);
`portfolio/avanza/client.py:65` (`account_id = config.avanza.account_id` no whitelist);
`portfolio/avanza/__init__.py:29-32` (`__all__` exports `place_order`).

`portfolio.avanza.trading.place_order` has the size guards but **no
`avanza_order_lock`** (so it races every locked path on `buying_power`) and **no
account whitelist** (it trades `client.account_id`, which `client.py:65` reads
from config with only a `DEFAULT_ACCOUNT_ID` fallback, never validated against
the whitelist). It is exported at package root, so a casual
`from portfolio.avanza import place_order` gets the unguarded version. Currently
this package is wired into nothing live (only `scripts/avanza_smoke_test.py` and
`tests/test_avanza_pkg/`), which is the only reason this isn't P1 — but it is a
loaded footgun that prior reviews (`docs/fgl/claude-avanza-api.md` P3,
`docs/adversarial-review-2026-05-13/5-avanza-api-claude.md`) already flagged and
which remains unaddressed.

→ Either (a) wrap these in `avanza_order_lock` + add the whitelist guard to
match the legacy paths, or (b) do not export `place_order`/`place_stop_loss`
from `__init__` and rename to `_place_order_unsafe`, forcing callers through a
vetted facade.

### P2-2 — `place_stoploss_once.py` deletes via the 2-segment stop-loss path (missing accountId) — likely a no-op cancel
`data/place_stoploss_once.py:111-115` calls
`api_delete(f"/_api/trading/stoploss/{stop_id}")`.

Every other path uses the 3-segment form `/_api/trading/stoploss/{accountId}/{stopId}`
(avanza_session.cancel_stop_loss:921, avanza_control.delete_stop_loss:227,
metals_avanza_helpers.delete_stop_loss:481). The 2-segment form omits the
account and will likely 404. Because `api_delete` treats 404 as `ok` (line 377),
a delete that never actually removed the stop reports success. This is a manual
one-off script (`DRY_RUN` flag, not in the loop), hence P2 — but it can mislead
an operator into believing an old stop was cleared before placing a cascade,
re-creating an over-encumbered position (the exact failure
`cancel_all_stop_losses_for` exists to prevent).

→ Use `/_api/trading/stoploss/{account_id}/{stop_id}`, or import and call
`avanza_session.cancel_stop_loss`.

### P2-3 — `avanza_session.cancel_order` returns the raw response unchecked; success determination is left to each caller
`portfolio/avanza_session.py:638-650`

`cancel_order` POSTs to `/order/delete` and returns `api_post(...)` verbatim with
no `orderRequestStatus` check and no logging of failure. Callers must remember to
check (metals_loop.py:5469 does; grid_fisher line 1296 routes through
`_safe_session_call` and only checks `is None`, not `orderRequestStatus`). A
cancel that returns `{"raw": ...}` (non-JSON body) or
`orderRequestStatus != SUCCESS` is silently treated by the grid fisher as
"cancel attempted" — a cancel that didn't actually cancel. Contrast with
`cancel_stop_loss`, which returns a structured `{status, http_status}`.

→ Make `cancel_order` return a structured `{status, http_status, order_id}` like
`cancel_stop_loss`, log on non-SUCCESS, so callers cannot accidentally treat a
failed cancel as done.

### P2-4 — `avanza_client.get_positions` TOTP fallback can return an empty list on auth/shape failure, indistinguishable from "flat account"
`portfolio/avanza_client.py:162-200`

The docstring says "Returns empty list if no positions **or on error**." If
session auth fails and the TOTP `client.get_overview()` returns an unexpected
shape (or an account with no `positions` key), the function returns `[]` — a
caller cannot tell "genuinely flat" from "read failed". This violates the
project rule "always fetch LIVE positions before any action; never assume a
position still exists." The page-path `fetch_positions` (metals_avanza_helpers.py:61)
and `avanza_session.get_positions` correctly distinguish None from `{}`/`[]`;
this TOTP path does not.

→ Return `None` (or raise) on read failure; reserve `[]` for a confirmed-empty
account, matching the page path's contract.

---

## P3 — Low / defense-in-depth

### P3-1 — `get_open_orders` fallback only catches `RuntimeError`, but a 401 raises `AvanzaSessionError`
`portfolio/avanza_session.py:653-669`

The primary `api_get` can raise `AvanzaSessionError` (on 401), which is NOT a
`RuntimeError`, so the `deals-and-orders` fallback at line 663 is skipped and the
error propagates. Likely intentional (expired session shouldn't be papered over)
but the `except RuntimeError` is narrower than the comment implies ("Endpoint may
vary"). Document that 401 deliberately propagates, or widen if the fallback was
meant to cover auth blips too.

### P3-2 — `_api_get_categorized_accounts` 30s worker-thread timeout while holding nothing, but the inner `api_get` holds `_pw_lock` (RLock) for the whole Playwright round-trip
`portfolio/avanza_account_check.py:154-177` + `avanza_session.py:219-232`

The verifier offloads onto a 1-worker pool to avoid the asyncio-loop clash. The
inner `api_get` grabs `_pw_lock` (a process-wide RLock) for the entire request.
If that worker thread blocks on a hung browser, the 30s `.result(timeout=30)`
returns to the caller but the worker keeps holding `_pw_lock`, stalling every
other Avanza call in the process until the request finally errors out. Low
likelihood (browser-dead recovery usually fires first) but worth a bounded
Playwright request timeout.

### P3-3 — Confirm-token logged at INFO (truncated) — acceptable, noted for completeness
`portfolio/avanza_orders.py:139-145`

Token is truncated to 4 chars + `****` before logging and is per-order /
5-min-expiry / single-use, so leak surface is minimal and the suppression is
documented. No action needed; flagged only so the next reviewer doesn't re-raise
it. No other secret/credential logging found (CSRF tokens, BankID cookies,
passwords are never logged).

---

## Invariants — verification summary

- **Stop-loss uses `/_api/trading/stoploss/new`, never the regular order API:**
  VERIFIED in all live paths (avanza_session.py:806, metals_avanza_helpers.py:386,
  avanza/trading.py via library `place_stop_loss_order`). No regular-order-API
  stop construction found. Pass.
- **Account isolation (only 1625505):** Enforced in `avanza_session` and
  `avanza_client`; **NOT enforced** in the live page path or the `avanza`
  package (P1-3, P2-1). The metals path is safe today only via a hard-coded
  constant.
- **Fetch live positions before acting:** Honored in grid_fisher (None vs `{}`
  handling, grid_fisher.py:1664-1669) and page path; violated in
  `avanza_client.get_positions` TOTP fallback (P2-4).
- **Order idempotency / lock:** `avanza_order_lock` correctly acquires and
  releases on all paths including exceptions (try/finally, order_lock.py:84-100);
  uses `filelock.FileLock` (atomic). The lock guards the POST in every legacy
  path. Gaps: the `avanza` package `place_order` is unlocked (P2-1). No deadlock
  on the release path. The lock does NOT protect against re-submitting an order
  that already filled on a retry (it only serializes concurrent POSTs) — but the
  human-CONFIRM token flow (avanza_orders.py) and the order-id tracking mitigate
  this. Pass with the P2-1 exception.
