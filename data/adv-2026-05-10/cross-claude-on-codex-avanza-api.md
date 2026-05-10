# Claude critique of codex findings — avanza-api

## Verdicts

- [P0] Missing atomicity guard on order execution + file write — avanza_orders.py:348
  Verdict: CONFIRMED
  Reason: `_execute_confirmed_order()` modifies `order["status"]` in-memory on line 370, but `_save_pending()` is called after the RPC on line 214 of the parent function. A crash between status mutation and disk write leaves "executed" orders unflushed; restart replays and re-executes.

- [P0] Expiry boundary off-by-one — avanza_session.py:84
  Verdict: CONFIRMED
  Reason: `if exp <= now` on line 84 rejects sessions with expiry time equal to current time. Should be `<` to allow sessions that expire at the boundary instant.

- [P1] Unguarded `datetime.fromisoformat()` on `order["expires"]` — avanza_orders.py:186
  Verdict: CONFIRMED
  Reason: Line 186 calls `datetime.fromisoformat(order["expires"])` with no try/except. Malformed ISO string crashes the loop; orders stuck in `pending_confirmation` until process restart.

- [P1] Telegram offset deserialization accepts dict OR plain int with no schema check — avanza_orders.py:257-265
  Verdict: PARTIAL
  Reason: Lines 260-265 attempt to load offset. If `load_json` returns an invalid dict (e.g., `{"offset": "not_a_number"}`), line 262 crashes on `int()` with no fallback. However, the code DOES validate the dict has an "offset" key and falls back to legacy plain-text on file-not-found. Missing validation is only on non-dict dicts that parse but contain garbage.

- [P1] Session expiry not re-checked on every API call — avanza_session.py:82-86
  Verdict: CONFIRMED
  Reason: Session expiry is validated in `load_session()` once on startup. `_with_browser_recovery()` (lines 207-227) retries HTTP failures but never calls `is_session_expired()`. A session expiring mid-day produces silent 401s because `ctx.request.get()` on line 252 sees the 401 but only closes Playwright; line 254 raises AvanzaSessionError without re-checking the expiry timestamp.

- [P1] Telegram offset omitted when `offset == 0` — avanza_orders.py:268-269
  Verdict: CONFIRMED
  Reason: Line 268 uses `if offset:` which is falsy at 0. When offset is legitimately 0, the param is omitted; getUpdates defaults to returning ALL updates from the bot start, inflating response size and consuming time.

- [P1] Confirm-token discarded BEFORE Telegram send acknowledgement — avanza_orders.py:202-207
  Verdict: FALSE-POSITIVE
  Reason: Lines 203-207 discard the token from the in-memory `confirmed_tokens` set (not from the order). The order is marked "confirmed" on line 199, then `_execute_confirmed_order()` is called on line 208, which sends Telegram on line 389. If send fails, the exception is caught in the outer try/except block (lines 351-401). The order is already persisted with status "confirmed" before the Telegram send, so no loss of state occurs.

- [P1] `_get_csrf()` has two lock-acquisition paths producing stale-cookie windows — avanza_session.py:280-286
  Verdict: CONFIRMED
  Reason: Lines 266-287 define `_get_csrf(ctx=None)`. If called without ctx (line 281-286 path), it re-acquires `_pw_lock` and fetches a fresh context. If `_with_browser_recovery` relaunches the browser between a caller's csrf-read (before the lock) and the POST (with old csrf), the token mismatches the new context cookies.

- [P1] Confirm token logged at INFO level — avanza_orders.py:115-142
  Verdict: CONFIRMED
  Reason: Lines 139-142 log the plaintext `confirm_token` at INFO level. This reaches centralized logging and backups without redaction.

- [P2] `session_remaining_minutes()` does not handle naive `expires_at` — avanza_session.py:82
  Verdict: CONFIRMED
  Reason: Lines 110-112 parse `expires_at` with `fromisoformat()` which may return a naive datetime if the ISO string has no timezone. Line 111 compares `exp - now` where `now` is UTC-aware (line 111), raising TypeError. The exception is caught by the outer except (line 113), masking the root cause.

- [P2] `place_buy_order()` TOTP-fallback path does not re-validate `ALLOWED_ACCOUNT_IDS` — avanza_client.py
  Verdict: PARTIAL
  Reason: The TOTP fallback (`avanza_client.place_buy_order`) calls `get_account_id()` on line 228/351, which DOES validate `ALLOWED_ACCOUNT_IDS` (lines 269-276). However, the codex finding is about a hypothetical future where the pension account is renamed to include "ISK". The current code IS hardcoded, but the concern is valid if Avanza response structure changes.

- [P2] `total_sek` stored as `round(volume * price, 2)` float — avanza_orders.py:126
  Verdict: CONFIRMED
  Reason: Line 126 stores `round(volume * price, 2)` as a float in JSON. Large orders lose precision; copy/paste to spreadsheet introduces rounding drift on reconciliation.

- [P2] Context proxy `__getattr__` does not explicitly forward `request` — avanza_resilient_page.py:189
  Verdict: FALSE-POSITIVE
  Reason: Lines 228-231 define `_ResilientContextProxy.__getattr__`, which falls through to `getattr(self._rp._ctx, name)`. The `request` attribute is NOT explicitly handled but WILL correctly forward to `_ctx.request` via the generic `__getattr__`. This is not a footgun; explicit forwarding would be redundant.

- [P3] Dead constant `EXPIRY_BUFFER_MINUTES = 30` — avanza_session.py:32
  Verdict: CONFIRMED
  Reason: Line 32 defines `EXPIRY_BUFFER_MINUTES = 30` but it is never referenced in the file. The buffer was intended but never wired into `is_session_expiring_soon()` (line 118), which uses a hardcoded 60-minute default instead.

## New findings (mine)

- [P0] Order status mutation not atomic with disk write — avanza_orders.py:348-389
  Lines 370, 380, 392 mutate `order["status"]` in-memory, but `_save_pending(orders)` is not called until line 214 of the parent `check_pending_orders()` function. This introduces a window where the process could crash after mutating status but before flushing. On restart, `_load_pending()` reloads from disk with the stale status, and the order may be re-executed if the confirmation retry logic re-enters. Atomicity should wrap both the status mutation AND the Telegram send in a critical section before returning to the parent's batch save.

- [P1] `offset == 0` initialization never written — avanza_orders.py:257-341
  On first run, `avanza_telegram_offset.txt` does not exist. Line 260 calls `load_json(offset_file)` which returns None/empty. Lines 261-265 handle this gracefully, initializing `offset = 0`. However, the offset is only persisted on line 341 AFTER processing updates. If the loop exits or crashes before the first getUpdates succeeds, offset is never written, and on restart the same window repeats. This is low-risk but means the initialization is implicit rather than explicit.

- [P1] ValueError swallowed on fromisoformat in session_remaining_minutes — avanza_session.py:110
  Line 113 catches all Exception with a generic warning "Failed to compute session minutes remaining". If fromisoformat() fails due to naive timezone or invalid format, the caller sees no indication of the root cause. A timestamp with timezone info is critical for safety — naive timestamps should fail loudly, not silently log and return None.

## Summary

- Confirmed: 10 | Partial: 2 | False-positive: 2 | New: 3
