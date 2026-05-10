# Adversarial Review: avanza-api subsystem (2026-05-08)

[P0] portfolio/avanza_orders.py:348
**Missing atomicity guard on order execution + file write.**
Problem: `_execute_confirmed_order()` mutates `order["status"]` but `_save_pending()`
runs ~50ms later. A crash between execute and save leaves the order "confirmed" on
disk; replay on restart triggers a duplicate fill.
Fix: Persist status BEFORE the place_buy_order() RPC, or wrap execute+save in a single
critical section/lock.

[P0] portfolio/avanza_session.py:84
**Expiry boundary off-by-one.**
Problem: `if exp <= now` rejects sessions that expire at the current second. A session
expiring at 13:00:00 fails at 13:00:00 sharp.
Fix: Use `exp < now`.

[P1] portfolio/avanza_orders.py:186
**Unguarded `datetime.fromisoformat()` on `order["expires"]`.**
Problem: A truncated/old-format pending-orders file crashes the entire
`check_pending_orders()` loop; orders stay in `pending_confirmation` forever.
Fix: Try/except, log warning, mark order expired, continue iterating.

[P1] portfolio/avanza_orders.py:257-265
**Telegram offset deserialization accepts dict OR plain int with no schema check.**
Problem: BUG-128 migration tries `load_json` then `read_text` fallback, but doesn't
validate the dict shape. `int("not_a_number")` will crash; or fallback path is skipped
because `load_json` returned a non-None invalid dict.
Fix: Validate offset value is castable to int; default to 0 with warning otherwise.

[P1] portfolio/avanza_session.py:82-86
**Session expiry not re-checked on every API call.**
Problem: 24h session validated once at startup; `_with_browser_recovery()` retries
HTTP failures but never re-checks session expiry. A session expiring mid-day produces
silent 401s until process restart.
Fix: Call `is_session_expired()` inside `_with_browser_recovery` before invoking
`_op(ctx)`; trigger reauth if so.

[P1] portfolio/avanza_orders.py:268-269
**Telegram offset omitted when `offset == 0`.**
Problem: `if offset:` is falsy at 0; getUpdates returns ALL updates from the start of
bot history (potentially thousands), inflating offset persistence and consuming 30+s.
Fix: Always include `offset` param; ensure file initialised with 0.

[P1] portfolio/avanza_orders.py:202-207
**Confirm-token discarded BEFORE Telegram send acknowledgement.**
Problem: If `send_telegram()` raises, the order is marked "executed" but the token
removed; the original CONFIRM message re-poll on next cycle finds no token, silently
drops the confirmation feedback. User sees no execution receipt.
Fix: Discard token only AFTER Telegram delivery succeeds; or persist orphaned-token
warning so operator knows the receipt was lost.

[P1] portfolio/avanza_session.py:280-286
**`_get_csrf()` has two lock-acquisition paths producing stale-cookie windows.**
Problem: `_get_csrf(ctx=None)` re-acquires `_pw_lock` to fetch fresh context; if the
browser is relaunched between csrf-read and the POST, CSRF is from the old context but
the POST is on the new context — Avanza rejects with 403.
Fix: Only call `_get_csrf(ctx)` with the currently-held context; never call without
ctx inside `_with_browser_recovery`.

[P1] portfolio/avanza_orders.py:115-142
**Confirm token logged at INFO level.**
Problem: Plaintext order-confirmation nonce reaches centralised log aggregation
(CloudWatch/Datadog/syslog/backups). Anyone with log read access can confirm orders
without Telegram access.
Fix: Never log raw token; log truncated SHA256 hash for correlation.

[P2] portfolio/avanza_session.py:82
**`session_remaining_minutes()` does not handle naive `expires_at`.**
Problem: `fromisoformat` on naive ISO string parses fine, but
`exp <= now (UTC-aware)` raises TypeError swallowed by outer except. The user gets
"Cannot parse" instead of "timezone mismatch", masking the root cause.
Fix: If parsed datetime is naive, attach UTC explicitly with a warning log.

[P2] portfolio/avanza_client.py
**`place_buy_order()` TOTP-fallback path does not re-validate `ALLOWED_ACCOUNT_IDS`.**
Problem: `avanza_session.place_buy_order` validates account whitelist; the alternate
`avanza_client.place_buy_order` (TOTP fallback) scans for "any ISK account". If the
pension account 2674244 is ever renamed to include "ISK", trades could route there.
Fix: Add hardcoded whitelist check in `avanza_client.get_account()` too.

[P2] portfolio/avanza_orders.py:126
**`total_sek` stored as `round(volume * price, 2)` float.**
Problem: Mixing float SEK in JSON loses precision on large orders; spreadsheet copy
introduces rounding drift on reconciliation.
Fix: Store as integer cents (`int(round(volume * price * 100))`) and convert at the
display boundary.

[P2] portfolio/avanza_resilient_page.py:189
**Context proxy `__getattr__` does not explicitly forward `request`.**
Problem: Falls through generic `__getattr__` to `_ctx.request`, which works by accident
today. Future Playwright internals churn could silently break this footgun.
Fix: Explicit `__getattr__` branch returning `self._rp._ctx.request`.

[P3] portfolio/avanza_session.py:32
**Dead constant `EXPIRY_BUFFER_MINUTES = 30`.**
Problem: Defined but never referenced; intended buffer never applied to expiry check.
Fix: Wire into `is_session_expiring_soon()` default, or delete.

## Summary

2 P0 + 7 P1 + 4 P2 + 1 P3. Themes: order-state durability gaps that risk duplicate
fills, BankID expiry not detected mid-flight, secrets in logs (token), missing schema
validation on persisted state, account-whitelist gap on TOTP fallback path.
