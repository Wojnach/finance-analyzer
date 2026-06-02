# Adversarial Review: avanza-api (Agent Findings)

Reviewer: caveman:cavecrew-reviewer subagent (fgl 8-subsystem audit, fresh pass)
Date: 2026-06-02
Repo: `Q:\finance-analyzer` (live, post-commit 8ca546dc)
Scope: 8 top-level avanza_*.py + portfolio/avanza/ dir (19 files). Add-only diff `review-base-2026-06-02..review-avanza-api-2026-06-02`.
Note: this subsystem was 100% [REPEAT] in the 2026-05-26 pass — but this pass surfaces a NEW P0 (session-expiry exception propagation). Account whitelist (ISK 1625505 only) confirmed correct.

## P0
- `portfolio/avanza_session.py:143`: **[NEW]** P0: `load_session()` raises `AvanzaSessionError` on expiry, but `_get_playwright_context()` has NO try/except. The exception propagates into callers' (`api_get`/`api_post`) retry loops; if a caller's broad `except` swallows it, the next cycle finds no context → silent auth outage (the March/May-2026 incident class). FIX: wrap in try/except, call `close_playwright()`, raise or return None explicitly.

## P1
- `portfolio/avanza_orders.py:419`: **[NEW]** P1: broad `except Exception` in `_execute_confirmed_order` swallows order-placement errors. If the Telegram notification path then also fails, the user gets ZERO feedback on a silently-failed order. FIX: log order_id before the Telegram attempt; ensure failure is always surfaced to caller.
- `portfolio/avanza_session.py:589-610`: **[NEW]** P1: `place_order()` has no guard that a stop-loss is NOT placed via the regular order API. Page-path routing via avanza_control→metals_avanza_helpers correctly uses `/_api/trading/stoploss/new`, but no validation prevents a legacy path from swapping to the wrong endpoint (instant-fill-at-bad-price, Mar 3 incident). FIX: assert / consolidated dispatch.
- `portfolio/avanza_resilient_page.py:126`: **[NEW]** P1: `page.goto(self._initial_url, wait_until="domcontentloaded")` has no explicit timeout (defaults 30s). A hung login/BankID interstitial blocks the entire first loop cycle and deadlocks subsequent requests. FIX: add `timeout=5000`.
- `portfolio/avanza_account_check.py:51`: **[REPEAT]** P1: session-expiry de-escalation (warning→critical on persistence, `_PERSISTENT_EXPIRY_MIN_PRIORS=2` + 24h) may not re-escalate a 3-day continuous outage before the auto-resolve window. FIX: validate thresholds against the May-Jun 2026 session-expiry incident data.

## P2
- `portfolio/avanza_session.py:342`: **[NEW]** P2: `api_post()` returns `{"raw": body}` when a 2xx response has a non-JSON body (HTML maintenance page / 503 masquerading as success). Callers `.get("status")` → None → silent failure, never surfaced. FIX: return `{"status":"FAILED","error":"unexpected_response_body"}`.
- `portfolio/avanza_session.py:764-767`: **[NEW]** P2/RISK: non-trailing stop with `sell_price ≤ 0` raises ValueError (good), but there is no guard that `trigger_price` is NOT within 3% of current bid (memory rule: never place a stop near a MINI barrier / within 3% of bid). A stop just above bid triggers immediately next tick. FIX: fetch current bid, reject if trigger within 3%.

## Top 3 must-fix
1. `avanza_session.py:143` — session-expiry exception not caught in `_get_playwright_context()`; propagates to retry loops → silent-outage pattern (Mar/May 2026).
2. `avanza_orders.py:419` — broad except in order execution; user blind to order failures if Telegram also fails.
3. `avanza_session.py:342` — silent non-JSON 2xx; maintenance pages / 503s masquerade as success.
