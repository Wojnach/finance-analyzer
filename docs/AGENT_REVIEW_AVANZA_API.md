# Agent Review: avanza-api subsystem — FGL 2026-05-24

**Files reviewed**:
- portfolio/avanza_session.py
- portfolio/avanza_client.py
- portfolio/avanza_orders.py
- portfolio/avanza_control.py
- portfolio/avanza_tracker.py
- portfolio/avanza_account_check.py
- portfolio/avanza_order_lock.py
- portfolio/avanza_resilient_page.py

**Context**: Playwright BankID auth, ~24h session. ISK account 1625505 ONLY. Pension account 2674244 must NOT leak.

---

## Critical Findings

### 1. avanza_session.py:676-717: BUG — get_positions() returns pension account positions

**Severity**: BUG (security/data leak)

**Issue**: get_positions() calls api_get() and appends ALL entries from data.get("withOrderbook", []) without filtering by ALLOWED_ACCOUNT_IDS. The account_id field is set on line 714 but never checked. Callers receive pension account holdings (account 2674244) mixed with trading positions.

**Impact**: Dashboard, grid_fisher, fin_snipe, and position reconciliation sees pension portfolio. Memory rule feedback_isk_only.md explicitly forbids this.

**Fix**: Filter at line 701 — skip append if account.get("id", "") not in ALLOWED_ACCOUNT_IDS.

---

### 2. avanza_session.py:633-645: RISK — cancel_order() has no account whitelist guard

**Severity**: RISK (cross-account mutation)

**Issue**: cancel_order(order_id, account_id=None) accepts account_id parameter without verifying it's in ALLOWED_ACCOUNT_IDS. All other trading functions enforce the whitelist. This is the only mutating function without it.

**Impact**: A caller passing account_id="2674244" would cancel orders on the pension account.

**Fix**: Add if str(account_id or DEFAULT_ACCOUNT_ID) not in ALLOWED_ACCOUNT_IDS: raise ValueError(...) before line 638.

---

### 3. avanza_client.py:327-368: RISK — _place_order missing minimum order size guard

**Severity**: RISK (poor fee efficiency)

**Issue**: The TOTP path _place_order() skips the 1000 SEK minimum check that avanza_session._place_order() enforces at lines 595-597. Orders totaling 500-999 SEK bypass the guard and incur disproportionate brokerage costs.

**Impact**: Inefficient small orders placed; no hard rejection.

**Fix**: Add order_total = round(volume * price, 2); if order_total < 1000.0: raise ValueError(...) after line 348, matching avanza_session.

---

### 4. avanza_session.py:134-153: RISK — _get_playwright_context() has no exception cleanup

**Severity**: RISK (resource leak on init failure)

**Issue**: If _pw_browser.new_context() at line 149 throws (corrupted storage state, BankID re-auth required), _pw_instance and _pw_browser are assigned but never torn down. The lock is released, but the half-alive Playwright process remains. Subsequent calls see _pw_context is not None but may operate on a dead/stale context.

**Impact**: Browser zombie process; context corruption on repeated init failures.

**Fix**: Wrap lines 147-152 in try/except; on exception, call close_playwright() before re-raising.

---

### 5. avanza_session.py:89: RISK — Session expiry check uses <= instead of <

**Severity**: RISK (off-by-one allows expired token for 1 tick)

**Issue**: Line 89 checks if exp <= now: which treats the expiry instant as already expired. Should be < so a token valid until exactly 2026-05-24T12:00:00Z works until 11:59:59Z. Using <= rejects it 1 second early.

**Impact**: Sessions rejected 1 second before true expiry; inconsistent boundary behavior.

**Fix**: Change line 89 to if exp < now:. Optionally add EXPIRY_BUFFER_MINUTES drift.

---

## Design Notes (Lower Priority)

6. avanza_orders.py:355-365: place_buy/sell_order calls lack account_id parameter — correct by design but fragile for future multi-account features. Consider adding optional account_id field to order dict (set to DEFAULT_ACCOUNT_ID now).

7. avanza_client.py:123-134: find_instrument() overly permissive, returns all instruments. Document that search is global.

8. avanza_control.py:137: place_stop_loss() docstring missing parameter documentation.

---

## Verified Correct

- avanza_session.place_buy/sell_order account whitelist guards
- avanza_session.place_stop_loss account whitelist guard
- Stop-loss API uses /_api/trading/stoploss/new (line 801)
- avanza_client.get_positions (TOTP path) filters by ALLOWED_ACCOUNT_IDS
- avanza_order_lock cross-process semantics
- avanza_resilient_page.py browser recovery and RLock usage
- avanza_account_check.py multi-shape handling
- CONFIRM token validation and legacy backwards-compat

---

Totals: 1 bug, 4 risks, 3 nits
