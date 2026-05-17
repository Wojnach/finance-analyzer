# Avanza API Subsystem Audit — 2026-05-17

## Critical Findings

None.

## Risk Findings

portfolio/avanza_orders.py:391-402: 🟡 risk: send_telegram failure swallows order status. Line 391 calls send_telegram inside try block (353–391). If it fails with any exception, the outer except at 393 catches it, marks order["status"] = "error" (L394), and logs it as an order execution error — hiding the fact that the order was ACTUALLY placed successfully (status="SUCCESS" at L372 was already set). Caller cannot distinguish "order failed to place" from "order placed but notification failed". Split: catch Telegram failures separately from order placement failures (place order, update status, THEN send notification in separate try/except that doesn't overwrite status).

portfolio/avanza_orders.py:402: 🟡 risk: exception variable shadowing. Nested `except Exception as e:` (L402) overwrites outer exception `e` (L393). If the inner send_telegram fails AND we need to log the original order-placement exception for diagnostics, it's lost. Use `except Exception as notify_err:` at L402.

portfolio/avanza_session.py:192: 🟡 risk: verify_session() returns resp.ok without checking resp.status. If Playwright context exists but Avanza returned HTTP 401 (auth expired), resp.ok is False and we call close_playwright() — correct behavior. BUT if resp.ok is True but status is 401, we return True and don't close. Better: check `resp.status` explicitly and only trust ok=True if status is 2xx (200–299). As-is, a 401 with malformed response could return ok=True and leave a stale auth context.

portfolio/avanza_session.py:193-196: 🟡 risk: broad exception catch swallows auth failures. Line 193 catches ALL exceptions (including AvanzaSessionError from _get_playwright_context at L143 which calls load_session at L143). If load_session detects an expired session (L89–93), that AvanzaSessionError is caught, logged as "Session verification failed", and we return False without distinguishing "network error" from "session expired". For session expiry detection during loop runtime, the operator needs to know which failure mode occurred. Re-raise AvanzaSessionError, or return a structured result with reason.

portfolio/avanza_session.py:160-177: 🟡 risk: close_playwright exception swallows close failures. Lines 160–177 catch and log close failures per resource (context, browser, _pw_instance) but silently continue if one throws. If _ctx.close() fails with a resource-hold error (OS file lock, orphaned process), we leak it and continue closing _browser. On subsequent relaunch, the old browser port might still be bound, causing launch failure. Better: log with exc_info=True or re-raise on critical close failures.

## Nit Findings

portfolio/avanza_orders.py:391: 🔵 nit: send_telegram call is inside try block wrapping order placement, so a Telegram timeout will cause the entire operation to be marked "error" even if the order succeeded. This is fail-safe (order succeeds silently rather than silently fails), but it could be clearer: move send_telegram outside the try/except or use a nested try so success is preserved regardless of notification outcome.

---

## Summary

- 1 risk: send_telegram failure inside order placement try block overwrites success status  
- 1 risk: exception variable shadowing in nested handler  
- 1 risk: verify_session checks resp.ok but not resp.status for auth failures  
- 1 risk: broad exception catch in verify_session swallows AvanzaSessionError  
- 1 risk: close_playwright swallows resource close failures, may cause port leak  
- 1 nit: send_telegram in try block could mark successful order as error

All risks are isolated to error handling paths. Stop-loss API path is correct (/_api/trading/stoploss/new), account ID whitelist is properly enforced (ALLOWED_ACCOUNT_IDS checked at placement), lock release is properly guarded (try/finally), order qty/price float math uses correct types (no explicit int casts that lose precision).
