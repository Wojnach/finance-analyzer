# Agent Review: avanza-api (2026-04-20)

## P1 Critical
1. **No account whitelist on page-based order path** — `avanza_control.place_order(page, account_id)` forwards arbitrary account_id to `metals_avanza_helpers.place_order()` with zero validation. Also `portfolio/avanza/trading.py:80-81` has no whitelist.
2. **No maximum order size limit** — Min 1000 SEK enforced, no max. A single bad call can commit entire account (~200K+ SEK).
3. **Browser recovery duplicates non-idempotent orders** — `_with_browser_recovery` and `ResilientPage` retry POSTs without dedup verification after browser death.
4. **TOTP singleton has no session expiry detection** — `AvanzaAuth` is a singleton with no expiry tracking. Silent degradation on session expiry.

## P2 High
1. Cross-process lock is advisory only (new processes can bypass)
2. `cancel_all_stop_losses_for` polling timeout very short (3s, Avanza needs 3-5s)
3. CSRF token replay window after session re-auth
4. Streaming websocket has no auth refresh

## P3 Medium
1. Telegram confirmation is single-match, no order ID binding
2. avanza_orders.py uses TOTP path (singleton expiry issue compounds)
3. Tick rules cache never invalidates
4. Bot-level `place_order` re-exports don't add validation layer

## Prior Finding Status
- Missing account whitelist: **PARTIALLY FIXED** (session path fixed, page-based and unified package NOT)
- Browser recovery duplicates: **NOT FIXED**
- TOTP singleton expiry: **NOT FIXED**
