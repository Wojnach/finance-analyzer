# Avanza API Subsystem — Adversarial Review (2026-05-16)

## P1 (CRITICAL: Wrong output, crash, security hole, data loss)

### P1-1: POST to mutating endpoints is retried on browser death
**File:** portfolio/avanza_session.py:212–232  
**Bug:** `_with_browser_recovery()` retries the entire `_op()` lambda if a browser-dead error is caught. For POST operations like `place_order` and `place_stop_loss`, a retry can issue a duplicate order if the initial POST succeeded before the browser died.  
**Why it matters:**  
- Browser dies after Avanza confirms the order (201 posted) but before response is fully read.
- The retry re-executes the same `api_post()`, creating a second order identical to the first.
- Both orders execute, causing a double-buy (e.g. two 5000 SEK warrant buys when intending one).
- Caller has no idempotency protection — the order lock only guards against concurrent callers, not retry-induced duplicates.

**Fix:** Implement server-side order deduplication via request ID, or remove browser recovery from POST paths. Document that `_with_browser_recovery` is unsafe for mutations.

---

### P1-2: Session expiry check uses `<=` instead of `<`, creating a 1-second race
**File:** portfolio/avanza_session.py:89  
**Bug:** `if exp <= now:` marks a session expired at the exact second of expiry. A request at `now == exp` fails even though the session is still valid.  
**Why it matters:**  
- Avanza sessions expire at fixed timestamps (e.g. 2026-05-17 18:30:45 UTC).
- If a trade fires at exactly 18:30:45, the session appears expired, triggers re-login, blocks orders.
- The 30-minute buffer mitigates but doesn't eliminate the boundary condition.

**Fix:** Change line 89 to `if exp < now:`. Equality means valid; only reject if past expiry.

---

### P1-3: Account ID validation gap in avanza/client.py
**File:** portfolio/avanza/client.py:19  
**Bug:** `DEFAULT_ACCOUNT_ID = "1625505"` is hardcoded without validation against a whitelist. The `portfolio/avanza/trading.py` module uses `account_id or client.account_id` without enforcing `ALLOWED_ACCOUNT_IDS`.  
**Why it matters:**  
- A config change to `config.json["avanza"]["account_id"]` could point to an unvetted account (pension account, child ISK, etc.).
- The unified TOTP auth path (`portfolio/avanza/`) has no hard guard against wrong accounts.
- Orders route to the wrong account silently.

**Fix:** Duplicate `ALLOWED_ACCOUNT_IDS` validation into `portfolio/avanza/client.py`. In `AvanzaClient.get_instance()`, validate resolved account_id before caching.

---

## P2 (RISK: Edge case, race, leak, perf cliff, missing guard)

### P2-1: CSRF token validation after browser relaunch can create livelock
**File:** portfolio/avanza_session.py:315–332  
**Bug:** After browser relaunch, `_op` calls `_get_csrf(ctx)` on the fresh context. If the initial POST succeeded but Avanza returned 403 (CSRF stale), the retry fails again and closes the browser, creating a relaunch loop.  
**Why it matters:**  
- Rare edge case: browser dies mid-response, Avanza returns 403.
- Line 331 closes Playwright. Retry relaunch succeeds but hits 403 again.
- If the original order executed before dying, retry duplicates it.

**Fix:** Add retry counter to prevent infinite relaunch loops. Ideally, use server-side request deduplication.

---

### P2-2: Order lock timeout (2 seconds) may abort legitimate in-flight orders
**File:** portfolio/avanza_order_lock.py:51  
**Bug:** Default timeout is 2.0 seconds. If Avanza API is slow, a legitimate order placement taking 2.1s causes timeout, aborting the caller.  
**Why it matters:**  
- Metals loop retries next cycle; if the first attempt was in-flight and executed, retry creates duplicate.
- No backoff or jitter — 2s is rigid.

**Fix:** Increase default to 5–10 seconds or make configurable. Add telemetry to measure actual hold times.

---

### P2-3: Buying power not re-checked inside the order lock
**File:** portfolio/avanza_session.py:620  
**Bug:** `place_order()` acquires the lock but does NOT re-fetch buying power immediately before POST. A stale snapshot from 60ms earlier causes overdraft.  
**Why it matters:**  
- Metals loop reads 4000 SEK, decides 3500 SEK order. Golddigger reads same snapshot, places 3500 SEK order.
- Both execute sequentially (no race), but account is overdrawn.
- Lock guards concurrent reads, not stale snapshots.

**Fix:** Make explicit: all callers must re-check buying power immediately inside the lock, after acquiring it. Implement two-phase locking.

---

### P2-4: get_buying_power() returns None on failure; callers may not handle it
**File:** portfolio/avanza_session.py:539  
**Bug:** `get_buying_power()` returns `None` on any failure (network error, account not found, etc.). Callers may interpret `None` as "balance is zero."  
**Why it matters:**  
- Transient network error returns `None`.
- Caller may place 0-quantity order (if not guarded) or silently skip trading.

**Fix:** Audit all callers for `None` handling. Add explicit assertions: `if not buying_power: raise BalanceCheckFailed()`.

---

### P2-5: ResilientPage.evaluate() retry may apply mutations twice
**File:** portfolio/avanza_resilient_page.py:158–174  
**Bug:** `evaluate(script, arg)` retries after browser death. For stateful scripts, retry applies mutation twice.  
**Why it matters:**  
- `data/metals_avanza_helpers.place_order()` calls `evaluate()` with async fetch.
- Browser dies between fetch and parse; retry causes Avanza to receive order twice.

**Fix:** Document that only read-only `evaluate()` calls are retried. For writes, wrap in request ID or use server-side dedup.

---

### P2-6: cancel_all_stop_losses_for() poll loop breaks prematurely on transient error
**File:** portfolio/avanza_session.py:1050–1070  
**Bug:** Poll read error on line 1055 sets `poll_read_failed = True` and breaks. Lines 1108–1109 then drop all `cancelled` IDs, making re-arm impossible.  
**Why it matters:**  
- DNS flap causes `get_stop_losses_strict()` to raise exception mid-poll.
- Caller loses list of cancelled stops and cannot re-arm on sell failure.
- Position is naked and unprotected.

**Fix:** On poll read failure, return `status="PARTIAL"` with cancelled stops still in rollback set, so re-arm is possible.

---

### P2-7: Session verification conflates transient server errors with auth failure
**File:** portfolio/avanza_session.py:180–196  
**Bug:** `verify_session()` checks only `resp.ok`. A 500 error returns `False`, causing unnecessary re-login.  
**Why it matters:**  
- 500 (Avanza downtime) returns `False`, triggering re-login on transient server error.
- Wastes time and may hit BankID rate limits.

**Fix:** Distinguish `401` (session stale, re-login needed) from `5xx` (server down, retry later).

---

## P3 (NITS: Style, naming, micro-perf)

No P3 items escalated for this review.

---

## SUMMARY

P1=3 P2=7 P3=0

Critical: POST retry idempotency, session expiry boundary, account validation gap.
Recommended: Implement deduplication, fix expiry check (one line), add account whitelist to avanza/client.py.
