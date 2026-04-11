# Agent Review: avanza-api — Round 5 (2026-04-11)

**Agent**: feature-dev:code-reviewer
**Files reviewed**: 10 (avanza_session, avanza_orders, avanza_client, avanza_control,
avanza_tracker, scripts/avanza_login, avanza_metals_check, avanza_metals_ladder,
avanza_orders, avanza_smoke_test)
**Duration**: ~213s

---

## Findings (5 total: 0 P0, 3 P1, 2 P2)

### P1

**AV-R5-1** avanza_session.py:591-632 — get_positions() returns ALL accounts, no whitelist
- A-AV-2 fix applied to avanza_client.get_positions() but NOT avanza_session.get_positions()
- BankID session path returns pension account (2674244) positions
- Used by metals_check, fish scripts, position reconciliation
- Fix: Filter by ALLOWED_ACCOUNT_IDS in the position append block

**AV-R5-2** avanza_client.py:326-350 — _place_order missing 1000 SEK minimum guard
- avanza_session._place_order has the guard (lines 525-528), avanza_client doesn't
- Orders 500-999 SEK go through, incur disproportionate brokerage fees
- Fix: Add order_total >= 1000 check matching avanza_session

**AV-R5-3** avanza_session.py:551-560 — cancel_order no ALLOWED_ACCOUNT_IDS guard
- Only mutating function in the module without the whitelist check
- Could cancel orders on pension account if caller passes wrong ID
- Fix: Add whitelist guard matching place_buy_order/place_sell_order/place_stop_loss

### P2

**AV-R5-4** scripts/avanza_login.py:256-259 — Session file written with raw write_text
- Non-atomic write for critical auth file
- Crash during write corrupts session → system cannot authenticate
- Fix: Use atomic_write_json()

**AV-R5-5** avanza_session.py:126-145 — Expired session causes tight Chromium spawn loop
- 401 → close_playwright → next call re-creates from same stale storage state → 401 again
- No cooldown, counter, or backoff
- Spams Chromium launches until manual restart
- Fix: Add consecutive-401 counter with backoff

---

## Pension Account Reachability Summary

| Path | Pension Reachable? |
|------|-------------------|
| avanza_session.place_buy/sell_order | No — ALLOWED_ACCOUNT_IDS guard |
| avanza_session.place_stop_loss | No — ALLOWED_ACCOUNT_IDS guard |
| avanza_session.cancel_order | **YES — no guard (AV-R5-3)** |
| avanza_session.get_positions | **YES — no filter (AV-R5-1)** |
| avanza_client.place_buy/sell_order | No — get_account_id() whitelist |
| avanza_client.get_positions (TOTP) | No — ALLOWED_ACCOUNT_IDS filter |

## Stop-Loss API Audit
All stop-loss placements confirmed using /_api/trading/stoploss/new. No regular order API misuse found.

## Regression Verification
- A-AV-1 (_pw_lock RLock): CONFIRMED correct
- A-AV-2 (account whitelist): CONFIRMED in avanza_client, BUT NOT in avanza_session (AV-R5-1)
- C7 (buying power keys): CONFIRMED correct
- C8 (CONFIRM newest order): CONFIRMED correct
