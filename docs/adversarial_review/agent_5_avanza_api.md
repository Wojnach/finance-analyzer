# Adversarial Review: Avanza API Subsystem

**Subsystem:** Avanza trading integration  
**File count:** 18 files (8 top-level + 10 in avanza/ package)  
**Findings:** 18 total

| Severity | Count |
|----------|-------|
| P0 (loses money / wrong account) | 3 |
| P1 (real bug) | 7 |
| P2 (latent) | 5 |
| P3 (minor) | 3 |

---

## Findings

### P0 — Account/Money Risk

**portfolio/avanza_session.py:676** P0 bug: `get_positions()` does NOT filter by account_id. Returns all positions from all accounts (ISK + pension + other). Callers using this to size trades hit wrong cash balance if pension account holds large warrants.

**portfolio/avanza_session.py:714** P0 bug: Position `account_id` extracted from `account.get("id", "")` but never validated against ALLOWED_ACCOUNT_IDS. No filter applied. Pension positions (2674244) leak into trading position list.

**portfolio/avanza_session.py:682-717** P0 risk: `get_positions()` endpoint is `/api/position-data/positions` (global view across all user accounts). Response aggregates ISK + pension + insurance holdings in `withOrderbook[]`. No mention of account filtering in docstring. Callers like metals_loop assuming ISK-only position view.

### P1 — Logic & Control Flow

**portfolio/avanza_session.py:89** P1 bug: Expiry check uses `<=` (correct: expired-at-or-before-now), but line 124 threshold check uses `<` (expiring-soon uses strict less-than). Session with 0 remaining minutes passes 60-min threshold check as NOT_expiring_soon. Asymmetric operator choice.

**portfolio/avanza_session.py:271-291** P1 bug: `_get_csrf(ctx=None)` accepts optional ctx but has two separate code paths (lines 280-283 with lock, lines 288-291 without lock). If ctx is None and concurrent access corrupts the context between `_get_playwright_context()` (line 287) and `ctx.cookies()` (line 288), the second call can hit a dead context. RLock protects the first path but not the split.

**portfolio/avanza_orders.py:373** P1 bug: Order `avanza_order_id` field stored in pending orders but ONLY set when status="executed" (line 372). If execution fails (status="failed" or "error"), no order ID is preserved. On retry/manual recovery, no record of which Avanza order ID to cancel.

**portfolio/avanza_orders.py:368** P1 bug: `order_id = result.get("orderId", "?")` — if Avanza API returns success=False but orderId is missing, the order may have been placed on broker but local state gets `"?"`. Orphaned order on Avanza, no local reference.

**portfolio/avanza_account_check.py:285** P1 bug: DISALLOWED_CATEGORY_FRAGMENTS is hardcoded as empty tuple `()` (line 52). Comment says "Empty by default" but justifies that ISK/pension are legal. If this tuple is ever edited, the verify_default_account() gate depends on grep-to-find-all-sites pattern — no compile-time check. Risk of silent disqualification without code review.

**portfolio/avanza/client.py:65** P1 bug: `account_id = str(avanza_cfg.get("account_id", DEFAULT_ACCOUNT_ID))` — config.json can override account_id, but there is NO whitelist check here. Only avanza_session.py enforces ALLOWED_ACCOUNT_IDS. If config has rogue `account_id: "2674244"` (pension), the TOTP path in avanza_client trades on pension.

### P2 — Latent / Edge Case

**portfolio/avanza_session.py:336-342** P2 risk: api_post() returns `{"raw": body}` on non-JSON response with resp.ok=true (line 342). Callers expect structured response with `orderRequestStatus` etc. Calling code never validates response shape before calling `.get("orderRequestStatus")`. Silent failure if API returns HTML (login page, Avanza outage).

**portfolio/avanza_session.py:620-630** P2 risk: place_order() result checked for status != SUCCESS but ONLY logs warning (line 624). No exception raised. Callers see "order placed" message in log but unknown to them result.get("orderId") is missing. Layer 2 agent proceeds assuming execution.

**portfolio/avanza_order_lock.py:86-91** P2 risk: FileLock.Timeout is caught and converted to OrderLockBusyError, but fileLock.acquire() on Windows may silently block forever if lock file is held by zombie process. No cross-platform test for lock expiry on dead process. 2s timeout is advisory, not guaranteed.

**portfolio/avanza_session.py:149-150** P2 risk: Playwright context created with `storage_state=str(STORAGE_STATE_FILE)` but no validation that file exists or is valid JSON. If storage_state.json is corrupted, context creation succeeds (Playwright init is lazy) but first request fails with confusing Playwright error.

**portfolio/avanza_session.py:265** P2 risk: api_get() raises RuntimeError on non-ok status (line 265) but does NOT call close_playwright() like it does for 401/403. If 502/503/timeout, caller retries but _pw_context may be stale. Next request may fail with TargetClosedError if Playwright hung.

### P3 — Minor

**portfolio/avanza_session.py:614** P3 nit: `valid_until or date.today().isoformat()` — time.today() is midnight UTC. If order expires at 20:00 CET and now is 23:00 CET, order already expired on same-day submit. Should check market hours before defaulting.

**portfolio/avanza_control.py:296-310** P3 nit: _v() helper defined inline in fetch_price_no_page() to unwrap `{"value": X}` — identical logic to types.py:_val(). Code duplication. If one path mutates, other becomes inconsistent.

**portfolio/avanza_session.py:32** P3 nit: EXPIRY_BUFFER_MINUTES=30 hard-coded. No config path. If Avanza session TTL changes or market hours shift, module requires edit + deploy.

---

## Root Causes (Patterns)

1. **Account filtering is scattered, not centralized.** ALLOWED_ACCOUNT_IDS defined in avanza_session.py + avanza_client.py (A-AV-2) but NOT checked in get_positions() or in avanza/client.py config override path.

2. **Order execution tracking is incomplete.** Pending order state machine tracks request→confirmation→execution but orphaned orders (placed but lost in crash before ID storage) cannot be recovered.

3. **Response shape assumptions.** Code assumes all API responses are JSON dict with known keys. Partial failures (connection drops, Avanza returning HTML) aren't explicitly handled.

4. **BankID session lifecycle unclear.** 24h validity assumed but hardcoded (L32). Storage state file (L28) has no validation or backup.

5. **Position filtering deferred to callers.** get_positions() returns all accounts; callers must filter. High risk of ISK leakage if filter logic is missed.

---

## No Fixes Included

Per review scope, this document identifies findings only. Fixes require architect decision on:
- Centralize account filtering in get_positions() vs push to callers
- Orphaned order recovery path (scan Avanza for unmatched fills?)
- Response shape validation (JSON schema or strict typing?)
