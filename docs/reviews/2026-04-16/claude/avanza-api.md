# Adversarial Review — avanza-api (Claude-independent)

## Critical Findings

### P0-1: Session expiry mid-request creates phantom fills
**File:** `portfolio/avanza_session.py:207–227` (`_with_browser_recovery`), caller `_place_order` at line 606-607
When an `api_post()` hits 401/403 mid-flight, code calls `close_playwright()` and raises. Caller already holds the cross-process lock, does NOT retry. If POST succeeded server-side but response was lost, order is orphaned. Timeline: t=0.00 POST sent → t=0.50 401 arrives → t=0.51 order actually filled on Avanza → t=0.52 exception raised → caller returns "error" → phantom fill sitting unowned.
**Fix:** on 401, call `get_open_orders()` before propagating error; include `request_id` UUID in payload for Avanza-side deduplication if supported.

### P0-2: Stop-loss placement has no idempotency token
**File:** `portfolio/avanza_session.py:706–774` (`place_stop_loss`), `data/metals_avanza_helpers.py:310–373`
POST to `/_api/trading/stoploss/new` with no nonce/request ID. Network timeout → broker places stop → retry places duplicate. With 5x leverage and -8% stops, duplicate stop triggers twice → forced-buy to cover excess short.
**Fix:** add `idempotencyKey` field (check Avanza docs for native support); use 409 handler to detect duplicates.

### P0-3: Order lock scope too narrow — released before network I/O
**File:** `portfolio/avanza_session.py:603-616`
Lock scope ends at line 607 but actual POST happens inside `api_post()` (line 311). Process A releases lock after validation → Process B races in and reads buying_power → A then submits → both orders land. Comment claims lock prevents concurrent submission but it only serializes validation.
**Fix:** extend lock to wrap `api_post()`; move validation inside `_with_browser_recovery`.

## P1

### 4. 5xx treated as fatal — no retry for transient
**File:** `avanza_session.py:251-263, 309-339`
Any non-2xx raises `RuntimeError`. 503/504 (load balancer timeout) identical to 400 (user error). Callers can't distinguish transient from permanent. `_with_browser_recovery` only retries `TargetClosedError`.
**Fix:** catch 503/504 in `_with_browser_recovery`, sleep 5s and retry once.

### 5. Account ID as single magic constant — silent drift risk
**File:** `avanza_session.py:585-587` (good), `avanza_orders.py:213-224` (missing)
`place_buy_order()` in avanza_session.py validates `ALLOWED_ACCOUNT_IDS`. Legacy `_execute_confirmed_order()` calls without account_id → defaults to `DEFAULT_ACCOUNT_ID`. If Avanza re-assigns, wrong account silently.
**Fix:** centralize account config; runtime validation on session startup that `DEFAULT_ACCOUNT_ID` exists AND is ISK type.

### 6. Storage state JSON replay may carry stale CSRF
**File:** `avanza_session.py:142-148`, `avanza_resilient_page.py:142-150`
On relaunch after crash/sleep, reload `storage_state.json`. CSRF tokens are session-bound; 3-day-old token loads but all subsequent POSTs return 403 until manual re-auth.
**Fix:** on relaunch, immediately `verify_session()`; if fails, call `close_playwright()` again and prompt BankID.

### 7. No atexit for Playwright — zombie chrome on crash
**File:** `avanza_session.py:151-173`, `avanza_resilient_page.py:130-140`
`close_playwright()` only called in explicit error paths. Unhandled exception in main loop → browser process survives → memory leak + port contention.
**Fix:** `atexit.register(close_playwright)` at module load; context manager for the global instance.

## P2

### 8. Error response body truncated to 500 chars without JSON parsing
**File:** `avanza_session.py:260, 336`
`resp.text()[:500]` cuts multi-line JSON mid-structure; operators can't debug.
**Fix:** parse as JSON, log `message` + `errorCode` fields cleanly.

### 9. `place_order()` in metals_avanza_helpers ignores HTTP status
**File:** `data/metals_avanza_helpers.py:288-302`
Only checks `body["orderRequestStatus"] == "SUCCESS"`. HTTP 202 (accepted, pending) with no orderRequestStatus → function returns failure even though order is queued.
**Fix:** validate `200 <= http_status < 300` before parsing body; log mismatch between HTTP status and body status.

### 10. Price precision — no tick-size validation
**File:** `avanza_session.py:600-601`, all order payloads
Float price serialized as-is. Avanza tick rules (0.01 SEK stocks, 1 SEK warrants) violated → silent rejection. `portfolio/avanza/tick_rules.py` exists but isn't called.
**Fix:** look up tick size before submission (cached per instrument); round to nearest tick; fail pre-submission if tick unknown.

### 11. `get_stop_losses()` swallows errors → empty list
**File:** `avanza_session.py:812-827`
Returns `[]` on ANY exception. Caller `if not stop_losses:` can't distinguish "no stops" from "API down". Counterpart `get_stop_losses_strict()` raises — risk of using wrong one in pre-sell check → naked sell.
**Fix:** rename to `get_stop_losses_or_none()` returning None on failure; deprecate silent-fail path.

### 12. Telegram confirm offset file not locked
**File:** `avanza_orders.py:99-143`
`_check_telegram_confirm()` reads/advances offset without file lock. Two processes could see same CONFIRM, both execute → double order. Currently mitigated by single-process design, fragile.
**Fix:** wrap read-advance-save in `avanza_order_lock`.

## P3

### 13. `cancel_stop_loss()` treats 404 as success without re-verify
**File:** `avanza_session.py:851-894`
404 on DELETE = stop already gone (idempotent). But Avanza backend can be race-inconsistent; counter-API might still show stop. Caller assumes cleared. `cancel_all_stop_losses_for()` at line 1010-1033 correctly re-queries; single-stop path doesn't.
**Fix:** `cancel_stop_loss()` post-condition check via `get_stop_losses_strict()`.

### 14. Headless Chromium may be blocked by Avanza fraud detection
**File:** `avanza_session.py:144-147`
Banks often reject headless. Error message "No session file found" misleads operators to re-run login instead of checking headless flag.
**Fix:** on verify failure log "may indicate Avanza blocked headless; try non-headless login".

## Looked OK

- **Stop-loss endpoint isolation** — all paths correctly use `/_api/trading/stoploss/new` and `/_api/trading/stoploss/{id}`. Mar 3 2026 regression not repeated.
- **Account whitelist** — enforced at submission in avanza_session.py and avanza_client.py.
- **CSRF extraction** — `_get_csrf(ctx)` accepts passed context to avoid RLock re-entry.
- **Session expiry buffer** — `EXPIRY_BUFFER_MINUTES = 30` guards edge attempts.
- **Browser relaunch** — `ResilientPage` + `_with_browser_recovery()` one-shot recovery on TargetClosedError (no infinite retries).
- **Stop clearance verification** — `cancel_all_stop_losses_for()` properly polls and re-queries (line 1010-1033).
- **Double-lock prevention** — RLock at line 49 defends against re-entry.

## Reviewer confidence
0.78 (moderate-high). Critical gaps are idempotency (P0-1, P0-2) and lock scope (P0-3). Stop-loss endpoint correctly isolated. Missing: automated tests for concurrent order submission + session expiry scenarios.
