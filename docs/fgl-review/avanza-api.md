# FGL Review — avanza-api

## P1 Findings: Pension Account Leakage

- **portfolio/avanza_session.py:681** — P1 risk: `get_positions()` does NOT filter by ALLOWED_ACCOUNT_IDS. The endpoint `/_api/position-data/positions` returns positions from ALL accounts (including pension 2674244). Positions are extracted and returned without account filtering, unlike `data/metals_avanza_helpers.py:fetch_positions()` which filters at line 88 (`if (accountId && String(acc.id || '') !== accountId) continue;`). When this session-based path is called via `avanza_client.py:get_positions()`, pension holdings leak into the positions list unchecked. **Fix:** Add account filtering loop: `if account.get("id", "") not in ALLOWED_ACCOUNT_IDS: continue` before appending to positions list (mimic metals_avanza_helpers.py behavior).

## P0 Findings: None Found

All stop-loss calls correctly use `/_api/trading/stoploss/new` and `/_api/trading/stoploss/{account}/{id}` DELETE endpoints. All order calls correctly use `/_api/trading-critical/rest/order/new` and `/_api/trading-critical/rest/order/delete` POST endpoints. No findings for regular-order-API stop-loss placement (the Mar 3 incident guard is in place).

## P2 Findings: Edge Cases & Resilience

- **portfolio/avanza_session.py:89** — P2 edge: Session expiry check uses `exp <= now` (inclusive). If the expires_at timestamp rounds to exactly now, the session is expired. No off-by-one risk (correct operator), but document clearly if sub-second precision matters for BankID session re-auth timing.

- **portfolio/avanza_account_check.py:230-232** — P2 edge: Persistent expiry re-escalation uses `max(ages_h) >= _PERSISTENT_EXPIRY_MIN_AGE_H`. If the oldest entry is exactly 24.0 hours old and `_PERSISTENT_EXPIRY_MIN_AGE_H = 24.0`, the condition is True. Boundary-correct, but note that a session expiring at 23:59:00 on day 1 checked at 00:01:00 on day 2 is 24h 2m and will re-escalate. Intended behavior appears correct per design.

- **portfolio/avanza_session.py:1089** — P2 resilience: `cancel_all_stop_losses_for()` correctly filters cancelled set by remaining set to prevent re-arming dead stops. Logic is sound: `verified = cancelled - remaining`. No bypass possible.

## Summary

**Totals:** 1 P1 (pension leakage in avanza_session.get_positions), 0 P0, 2 P2 (boundaries OK, no action needed).

**Top Finding:** Portfolio/avanza_session.py:681 — avanza_session.get_positions() must filter by ALLOWED_ACCOUNT_IDS before returning, else pension account positions leak into trading callers.
