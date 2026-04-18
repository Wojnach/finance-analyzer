# Adversarial Code Review: Avanza API Subsystem (Claude Reviewer)

## Executive Summary

**3 P1 findings, 5 P2 findings, 5 P3 findings.**

The Avanza integration is battle-hardened with clear incident-driven improvements. However, critical gaps remain: missing account whitelist on the Playwright order path (pension account unguarded), TOTP singleton never expires (stale sessions used forever), and a TOCTOU race where CONFIRM executes stale prices without re-validation.

---

## P1 Findings (Critical)

### P1-1: No account whitelist on Playwright order path
**File:** `data/metals_avanza_helpers.py:253`, `portfolio/avanza_control.py:130-134`

`place_order()` accepts any `account_id` with zero whitelist validation. Compare with `avanza_session.py:_place_order()` at line 586 which explicitly validates. Same gap in `place_stop_loss()` (line 330) and `delete_order()` (line 411). The pension account `2674244` could be used for trades.

**Fix:** Add `ALLOWED_ACCOUNT_IDS` validation at top of each function.

### P1-2: TOTP singleton never detects session expiry
**File:** `portfolio/avanza/auth.py:86-114`, `portfolio/avanza/client.py:47-72`

`AvanzaAuth` and `AvanzaClient` singletons created once, cached forever. No TTL, no validity check, no refresh. After ~24h Avanza session expiry, all API calls fail but singleton is never reset.

**Fix:** Catch 401/403, reset singleton, retry once.

### P1-3: TOCTOU race in order confirmation — CONFIRM executes stale price
**File:** `portfolio/avanza_orders.py:131-135`

Between order creation and CONFIRM (up to 5 minutes), market price can move significantly. No re-validation of current price or buying power at confirmation time. For 5x leveraged warrants, 5 minutes of price movement is substantial.

**Fix:** Re-fetch current quote and buying power at confirmation. Reject if price moved > 2%.

---

## P2 Findings (High)

### P2-1: No barrier proximity guard on stop-loss placement
**File:** `portfolio/avanza/trading.py:213-288`, `portfolio/avanza_session.py:706-787`

CLAUDE.md states "NEVER place stop-loss within 3% of bid". Memory warns about barrier proximity. No programmatic enforcement exists.

### P2-2: `get_positions()` returns all accounts unfiltered
**File:** `portfolio/avanza_session.py:662-703`

Returns positions from ALL accounts (ISK + pension). Downstream sizing uses wrong totals.

### P2-3: Browser recovery can duplicate non-idempotent orders
**File:** `portfolio/avanza_session.py:207-228`

If browser dies during order POST, `_with_browser_recovery` retries — potentially placing a duplicate real-money order.

### P2-4: `cancel_order` skips account whitelist
**File:** `portfolio/avanza_session.py:619-631`

### P2-5: Double-checked locking without memory barrier
**File:** `portfolio/avanza/auth.py:86`

Safe on CPython/GIL but breaks on GIL-free implementations.

---

## P3 Findings (Medium)

### P3-1: Tick rule cache unbounded, never invalidated
**File:** `portfolio/avanza/tick_rules.py:20`

### P3-2: Playwright subprocess leak on init failure
**File:** `portfolio/avanza_session.py:129-148`

### P3-3: Divergent position parsing between TOTP and BankID paths
**File:** `portfolio/avanza_client.py:161` vs `portfolio/avanza_session.py:662`

### P3-4: Module-level imports force both auth stacks
**File:** `portfolio/avanza_control.py:17-45`

### P3-5: `get_quote()` hardcodes "stock" instrument type
**File:** `portfolio/avanza_session.py:653-659`
