# Avanza API Adversarial Review — Findings

**Date**: 2026-05-28  
**Status**: 5 P0 (silent trading halt), 3 P1 (edge case fail), 1 P2 (latent race)

## Summary

The avanza-api worktree contains **5 critical bugs that can cause silent trading halt or wrong-account trades**. The most urgent: session expiry is checked on context init (line 139-140) but reused contexts skip expiry verification, allowing dead sessions to poison all trades for hours until process restart. Combined with fetch_failed downgrades (P0-2 below), this causes the system to believe auth is working when it is not. Callers proceed to place orders with expired credentials, which fail silently or return 401s that are logged but not acted upon.

---

## P0: Critical Money-Loss Bugs

### portfolio/avanza_session.py:139–140
**Bug**: Session expiry not re-verified on context reuse.

When `_get_playwright_context()` is called a second time, it returns the cached `_pw_context` immediately without calling `load_session()` to check if the session has expired. If the session was valid at t=0 but expires at t=3600, all API calls after expiry will use stale cookies.

**Fix**: Move `load_session()` outside the `if _pw_context is not None` block so it runs on every call.

**Impact**: Session dead since 2026-05-23 14:54 UTC. Metals_loop logs "proceeding" and trades with dead session.

---

### portfolio/avanza_account_check.py:225–248 + metals_loop.py:7052
**Bug**: Session expiry treated as transient outage; trading proceeds with dead session.

`verify_default_account()` catches ALL exceptions including AvanzaSessionError and downgrades to `reason="fetch_failed"`. Caller metrics_loop logs "proceeding" without raising. Orders subsequently fail silently.

**Fix**: Distinguish AvanzaSessionError (auth fatal) from transient errors (HTTP 5xx, timeout).

**Impact**: Daily critical_errors since 2026-05-23. System believes auth working, proceeds to trade.

---

### portfolio/avanza_session.py:620–630 (place_buy_order)
**Bug**: Order failure swallowed; result dict may be misinterpreted as success.

`api_post()` returns result dict regardless of status. If `orderRequestStatus != "SUCCESS"`, this is NOT raised as exception. Callers may misread the dict keys.

**Fix**: Log ERROR on non-success and ensure caller sees structured failure.

**Impact**: Orders fail but logs show "placed" at INFO. Caller thinks position open when not.

---

### portfolio/avanza_session.py:426–433 (get_buying_power)
**Bug**: API failure returns None; caller cannot distinguish "empty balance" from "session expired 401."

Bare except catches all errors and returns None. Grid_fisher uses static 6500 SEK cap fallback, can overleverage if account is lower.

**Fix**: Log specific error type at ERROR level so 401 expiry is visible.

**Impact**: On session expiry, grid_fisher uses wrong cash cap, may overleverage.

---

### data/place_stoploss_once.py:171–187
**Bug**: Script not idempotent; re-running places duplicate stops.

Script named `_once` but no deduplication. If user re-runs or process crashes mid-run, calling again creates new stops without checking if they exist.

**Fix**: Query existing stops at startup, skip orders already placed.

**Impact**: Re-running script creates 12 stops instead of 6.

---

## P1: Edge Case Failures

### portfolio/avanza_session.py:256–266 (api_get)
**Bug**: Non-JSON response (HTML error page) from 200 OK response raises JSONDecodeError uncaught.

If Avanza returns 200 with HTML body, `resp.json()` fails. Exception not caught in _op, bubbles up, browser NOT torn down. Poisoned context reused.

**Fix**: Catch JSONDecodeError in _op, close_playwright() on bad JSON.

**Impact**: 502 error page returns 200 OK, fails JSON parse, browser not reset.

---

### portfolio/avanza_orders.py:313–323 (token case)
**Bug**: CONFIRM token case-sensitive; uppercase "ABC" doesn't match lowercase "abc" token.

Regex at line 53 is `^[0-9a-f]+$` (lowercase only). User typing uppercase rejected.

**Fix**: Normalize token to lowercase before validation.

**Impact**: User confirms "CONFIRM ABC", order expires because token is "abc".

---

### portfolio/avanza_client.py:90–108 (TOTP creds)
**Bug**: TOTP credentials in config.json plaintext; "never commit" doesn't stop accidental leak.

TOTP fallback reads config.json which can be accidentally committed. No env var fallback.

**Fix**: Use env vars AVANZA_USERNAME, AVANZA_PASSWORD, AVANZA_TOTP_SECRET.

**Impact**: If config.json leaked, credentials exposed forever in git history.

---

## P2: Latent Races

### portfolio/avanza_order_lock.py:85–91
**Bug**: Stale lock file (process crash while holding) hangs all subsequent orders indefinitely.

If peer dies with lock held, lock file never released. Next caller waits 2s and raises OrderLockBusyError. All orders blocked until manual `rm data/avanza_order.lock`.

**Fix**: Implement mtime staleness check (remove lock >30s old).

**Impact**: Process crash mid-order leaves lock stuck forever.

---

## Root Cause Analysis

**Session Expiry Pipeline:**
```
metals_loop:7042  verify_default_account()
  → avanza_account_check:225  _api_get_categorized_accounts()
    → avanza_session:143  load_session()  [RAISES if expired]
  ← EXCEPT at line 226, logged as "fetch_failed"
← return ok=False
metals_loop:7043  if ok: [FALSE but proceeds anyway]
metals_loop:7052  log("proceeding — order guards still apply")
grid_fisher:69  places orders with dead session
  → api_post(...stoploss/new)
    → _with_browser_recovery(_op)
      → _get_playwright_context()
        → line 139: if _pw_context is not None: return _pw_context [NO EXPIRY CHECK]
        → returns stale context from t=0 (when session was fresh)
      ← stale context
    → ctx.request.post(...)  [Returns 401 because cookies are dead]
    → line 324: if resp.status == 401: raise AvanzaSessionError [CAUGHT by recovery]
    ← retry with same stale context [FAILS AGAIN]
  ← RuntimeError propagates
```

**Timeline:**
- 2026-05-23 14:54 UTC: User runs `avanza_login.py`, session recorded with expires_at
- 2026-05-23 14:54+24h = 2026-05-24 14:54 UTC: Session expires
- 2026-05-24 06:00 CET (05:00 UTC): Nightly cron runs account verification
  - Calls `verify_default_account()` → `_api_get_categorized_accounts()`
  - `api_get()` → `_get_playwright_context()` → `load_session()` raises (session expired)
  - Caught as `fetch_failed`, logged to critical_errors.jsonl
  - Returns `ok=False`
- metals_loop continues trading because `ok=False` just means "retry verification next cycle"
- All grid_fisher orders use stale `_pw_context` from initialization (which happened at process start, before session expired)
- 2026-05-28 06:00 CET: Same error still logging because process NEVER RESTARTED

**Why Session Persistence Across Restarts Doesn't Help:**
- `avanza_storage_state.json` contains Playwright cookies/auth state
- Cookies have their own expiry (24h from BankID auth)
- When context reloads cookies from storage state file, they are ALSO expired
- So both problems compound: stale context + stale cookies = 401 on every request

---

## Audit Checklist

- [ ] Verify no trades were placed 2026-05-23 14:54 UTC onwards (check journal for status="unknown" or error messages)
- [ ] Confirm grid_fisher did NOT place orders while session was dead (check warrants position vs grid_fisher state)
- [ ] Restart all loops to pick up fresh session: `.venv/Scripts/python.exe scripts/avanza_login.py && schtasks /run /tn PF-DataLoop`
- [ ] Deploy P0 fixes (139-140 and 225-248) to both main and any active branches
- [ ] Backtest grid_fisher with session-expiry scenario to ensure orders are NOT placed during outage

