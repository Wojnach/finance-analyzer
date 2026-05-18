# Avanza-API Review — subagent result (pr-review-toolkit:code-reviewer)

Totals: 8 P1 (🔴), 10 P2 (🟡), 13 P3 (🔵). STRONGEST findings overall.

## P1 / 🔴

1. **avanza_session.py:191, 257, 316, 364** — `ctx.request.get/post/delete` has NO timeout. Hangs forever under `_pw_lock`. Single stuck call freezes all Avanza calls + main loop + metals + grid. Exact silent-outage pattern from March 2026.
2. **avanza_session.py:258-265, 324-335** — Session expiry detection ONLY on HTTP 401. Avanza returns HTML login page with HTTP 200 on expired cookies. Caller sees 200 + non-JSON body → silent garbage. Exact grudge pattern from CLAUDE.md preamble.
3. **avanza_session.py (entire)** — `AvanzaSessionError` has zero Telegram notification anywhere. `check_session_expiry()` is unreachable (no scheduled caller). Operator never alerted to dead sessions. Repeat of March 2026 outage class.
4. **avanza_orders.py:53** — `_HEX_TOKEN_RE = ^[0-9a-f]+$` no length bound. 24-bit tokens (6 chars). Combined with optional `allowed_user_id` gate → brute-force feasible across many orders. Fix: `^[0-9a-f]{6}$`.
5. **avanza_orders.py:274-278 + http_retry.py:50,54,63,67** — Telegram bot token in URL path; `fetch_with_retry` logs full URL on retry/warning/error → bot token in plaintext in `agent.log`. Same incident class as Mar 15 API key leak.
6. **http_retry.py:36-37** — `fetch_with_retry` returns 4xx as success (`if resp.status_code not in RETRYABLE_STATUS: return resp`). Caller does `r.json()` → silent error body parse.
7. **avanza_session.py:1086-1109** — `cancel_all_stop_losses_for` returns FAILED status with populated `snapshot`; naive rearm path could attempt stops on volume already encumbered. Fix: clear snapshot on FAILED.
8. **avanza_session.py:188-196** — `verify_session()` returns True on HTML-login-page 200; `_session_client` cached True permanently → entire process believes session alive.

## P2 / 🟡

- avanza_orders.py:115-130, 350-365 — Order price captured at request time, executed up to 5min later without staleness check; warrants drift 5-10%.
- avanza_session.py:271-291, 314-322 — CSRF rotation; `api_delete` doesn't call `resp.text()` so `Set-Cookie` may not commit; subsequent `_get_csrf` stale; 403 → forced full re-login (avoidable).
- avanza_session.py:54 — `_pw_lock` is RLock; single hung Playwright call holds RLock indefinitely (compounds P1-1).
- avanza_session.py:768-775 — `place_stop_loss` warns but accepts sub-1000 SEK leg vs documented project rule "≥1000 SEK per leg"; contradiction.
- avanza_session.py:123-131 + avanza_tracker.py:121-131 — `is_session_expiring_soon` doesn't distinguish "missing session" (P1) from "expires soon" (P3).
- avanza_client.py:65-79 — `_session_client = True` cached forever; long-running metals loop runs days with stale cache.
- avanza_session.py:608-621, avanza_orders.py:350-365 — No idempotency key on order POST; `_with_browser_recovery` retry can duplicate orders.
- avanza_session.py:648-664 — `get_open_orders` fallback filters by `accountId` field; field-name drift returns [] → caller thinks no orders → places duplicates.
- avanza_session.py:28, 149-152 — `avanza_storage_state.json` saved with default ACL (world-readable on Windows). BankID session lift-and-replay risk.
- avanza_session.py:193-196 — `verify_session()` swallows all exceptions to False; can't distinguish network blip from auth dead.

## P3 / 🔵

- place_buy_order valid_until=today() at 21:55 → expires same minute.
- MAX_ORDER_TOTAL_SEK=50_000 magic constant, no config wiring.
- cancel_order vs cancel_stop_loss inconsistent error shape (raises vs dict).
- confirm_token partially logged (first 4 of 6 chars = 16 bits leaked).
- _CONFIRM_PREFIX_RE not fullmatch — accepts trailing text (documented but fragile).
- request_order accepts any orderbook_id format (no regex).
- _load_pending: corrupt pending file silently drops all pending orders, no critical_errors entry.
- cancel_order POST vs library `delete_order` DELETE — TOTP path returns 404 (vestigial).
- get_stop_losses (non-strict) returns [] on RuntimeError; safety paths must use _strict.
- place_trailing_stop overloads `trigger_price` semantics by `trigger_type`.
- _pw_instance / _pw_browser globals — single account only.
- avanza_telegram_offset.txt written as JSON to .txt extension.
- _check_telegram_confirm doesn't log Telegram error description.

## Confirmed clean
- Stop-loss endpoint consistently `/_api/trading/stoploss/new` (Mar 3 grudge resolved).
- Order lock acquired on every mutating call.
- Account ID whitelist enforced at 3 layers.
- Cookie-based storage (no plaintext API keys).
