# Agent Review: Avanza API Subsystem — FGL 2026-05-26

**Scope**: BankID session auth, TOTP fallback, ISK account 1625505 ONLY, pension 2674244 blocked.  
**Files**: avanza_session.py, avanza_client.py, avanza_orders.py, avanza_control.py, avanza_resilient_page.py, avanza_account_check.py, avanza_order_lock.py, fin_snipe*, fish_*

---

## Critical Findings

### 1. portfolio/avanza_session.py:701 — BUG: get_positions() returns pension account [REPEAT]

**Severity**: BUG (wrong data)

**Issue**: get_positions() iterates `data.get("withOrderbook", [])` without filtering by `ALLOWED_ACCOUNT_IDS`. Sets `account_id` on line 714 but never checks it. Callers see pension holdings (2674244) mixed with trading positions.

**Impact**: Dashboard, grid_fisher, fin_snipe, reconciliation all see pension portfolio. Memory rule `feedback_isk_only.md` explicitly forbids.

**Fix**: Insert at line 701: `if account.get("id", "") not in ALLOWED_ACCOUNT_IDS: continue`

---

### 2. portfolio/avanza_session.py:638 — RISK: cancel_order() lacks account whitelist [REPEAT]

**Severity**: RISK (cross-account mutation)

**Issue**: `cancel_order(order_id, account_id=None)` accepts arbitrary account_id without validation. Every OTHER trading function enforces `ALLOWED_ACCOUNT_IDS`: `place_buy_order` (591), `place_sell_order` (591), `place_stop_loss` (750). This is the only mutating function without it.

**Impact**: Caller passing `account_id="2674244"` would cancel pension orders.

**Fix**: Add before line 638:
```python
if str(account_id or DEFAULT_ACCOUNT_ID) not in ALLOWED_ACCOUNT_IDS:
    raise ValueError(f"Refusing to cancel on non-whitelisted account {account_id!r}")
```

---

### 3. portfolio/avanza_client.py:358 — RISK: _place_order TOTP path lacks 1000 SEK minimum [REPEAT]

**Severity**: RISK (poor fee efficiency, asymmetric guards)

**Issue**: Session path (`avanza_session._place_order`) enforces 1000 SEK minimum at lines 595–597. TOTP path (`avanza_client._place_order`) at 358–366 skips it. Orders totaling 500–999 SEK bypass the guard and incur disproportionate brokerage costs.

**Impact**: Small orders placed silently with inflated fees. Asymmetric behavior between two auth paths.

**Fix**: Add after line 348:
```python
order_total = round(volume * price, 2)
if order_total < 1000.0:
    raise ValueError(f"Order total {order_total:.2f} SEK below minimum 1000 SEK")
```

---

### 4. portfolio/avanza_session.py:147 — RISK: _get_playwright_context() resource leak on init failure [REPEAT]

**Severity**: RISK (resource leak, state corruption)

**Issue**: If `_pw_browser.new_context()` at line 149 throws (corrupted storage state, BankID re-auth required), `_pw_instance` and `_pw_browser` are assigned (lines 147–148) but never torn down. The lock is released but the half-alive Playwright process remains. Subsequent calls see `_pw_context is not None` but operate on dead/stale context.

**Impact**: Browser zombie process; context corruption on repeated init failures; silent errors in subsequent API calls.

**Fix**: Wrap lines 147–152 in try/except:
```python
try:
    _pw_instance = sync_playwright().start()
    _pw_browser = _pw_instance.chromium.launch(headless=True)
    _pw_context = _pw_browser.new_context(...)
except Exception:
    close_playwright()
    raise
```

---

### 5. portfolio/avanza_session.py:89 — RISK: Session expiry boundary off-by-one [REPEAT]

**Severity**: RISK (off-by-one, early rejection)

**Issue**: Line 89 checks `if exp <= now:` which treats the expiry instant as already expired. Should be `<` so a token valid until exactly `2026-05-24T12:00:00Z` works until `11:59:59Z`. Using `<=` rejects it 1 second early.

**Impact**: Sessions rejected 1 second before true expiry; inconsistent boundary behavior for automation.

**Fix**: Change line 89 to `if exp < now:` (or add EXPIRY_BUFFER_MINUTES safety margin).

---

## Design Issues (P2 — Lower Priority)

6. **avanza_orders.py:355–365** — `place_buy/sell_order()` calls omit `account_id` parameter. Correct by design (defaults to `DEFAULT_ACCOUNT_ID`), but brittle for future multi-account features. Consider adding optional `account_id` to pending order dict.

7. **avanza_client.py:123–134** — `find_instrument()` returns all results globally without filtering. Document that search is unconstrained.

8. **fin_snipe_manager.py:80+** — Dual state files (state.json + manager_log.jsonl + predictions.jsonl + fills.jsonl). Consider consolidating to single atomic source-of-truth for rollback safety.

---

## Verified Correct ✓

- `avanza_session.place_buy_order` account whitelist at line 591
- `avanza_session.place_sell_order` account whitelist at line 591
- `avanza_session.place_stop_loss` account whitelist at line 750
- Stop-loss API uses `/_api/trading/stoploss/new` (line 801) — NOT regular order API
- `avanza_client.get_positions` (TOTP path) filters `ALLOWED_ACCOUNT_IDS` (line 186)
- `avanza_order_lock` cross-process fail-fast semantics
- `avanza_resilient_page.py` browser recovery + RLock (no deadlock risk)
- `avanza_account_check.py` multi-shape handling for Avanza API drift
- `avanza_orders.py` CONFIRM token validation + legacy backwards-compat
- Order lock held across entire price-check–submit sequence (session path)
- `data/metals_avanza_helpers.fetch_positions` filters by account_id (line 88)

---

## Summary

**Totals**: 5 bugs (carried forward from May 24 review), 0 new findings  
**Top 3 severity**: Account data leak, unguarded cancel, undersized TOTP orders

All 5 issues from prior review remain unresolved. These are P0/P1 trades-safety issues requiring immediate fix before next order execution.

