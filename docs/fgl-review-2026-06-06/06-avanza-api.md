# Adversarial review — avanza-api (2026-06-06)

Reviewer: `caveman:cavecrew-reviewer` (returns text only — captured here verbatim).
Scope: broker integration (real orders, real money via Playwright BankID + REST).

## P0
- `portfolio/avanza_control.py:401`: P0: `delete_stop_loss_no_page` checks the non-existent
  `errorCode` field instead of the `ok` field that `api_delete` actually returns
  (`avanza_session.py:377` → `{"http_status", "ok"}`). A failed DELETE (e.g. HTTP 500 →
  `{"http_status":500,"ok":False}`) never trips the check → line 404 returns `True`. Caller
  believes the stop-loss was cancelled when it wasn't → old stop + new sell both live → overfill.
  **CONFIRMED independently by orchestrator.** → `if result.get("ok") is not True: return False, result`.

## P1
- `portfolio/avanza_orders.py:207-214`: P1: `OrderLockBusyError` (lock contention) is caught by the
  broad `Exception` handler at line 419, setting `status="error"`; next cycle skips it
  (`status != "pending_confirmation"`) → the user's confirmation is discarded and the order abandoned.
  → Catch `OrderLockBusyError` separately and preserve `status="confirmed"` for retry.

## P2
- `portfolio/avanza_order_lock.py:86-91`: P2: `FileLock.acquire()` 2s timeout is advisory, not
  guaranteed across platforms; under heavy concurrency with a crashed peer holding the lock, Windows
  may not reliably time out → potential hang.

## Verified-correct (no finding)
Auth failures properly close the session (`avanza_session.py:258-260,324-326,330-332,371-373`);
stop-loss uses `/_api/trading/stoploss/new` (not the regular order API); order lock released on
exception via context manager; account whitelist enforced (`ALLOWED_ACCOUNT_IDS`, ISK 1625505 only);
Telegram sender auth fail-closed (`avanza_orders.py:302-311`).

## Risk summary
The stop-loss deletion silent-failure is a P0 real-money path (position left unprotected / overfill on
the cancel-then-sell sequence). The lock-contention order abandonment strands confirmed trades. The
lock timeout is unreliable on Windows with a zombie holder.
