# Subsystem 5: Avanza API — Round 5 Findings

## CRITICAL (P1)

None. Prior round fixes held:
- Account whitelist enforced in avanza_session._place_order (1625505 only)
- Stop-loss API correctly uses /_api/trading/stoploss/new everywhere
- Order lock prevents duplicates across cycles

## HIGH (P2)

**AV-R5-1** — metals_avanza_helpers.place_order has no account whitelist guard.
`data/metals_avanza_helpers.py:253-307`. Page-based order path bypasses ALLOWED_ACCOUNT_IDS.
Golddigger uses this path. Misconfigured config.json could trade on wrong account.
Fix: Add whitelist check matching avanza_session pattern.

**AV-R5-2** — api_delete does not handle 403 (CSRF stale). Stale session stays active.
`avanza_session.py:356-371`. api_get and api_post handle 403 → close_playwright(); api_delete only handles 401.
CSRF expiry during stop-loss delete leaves session broken for subsequent sell.
Fix: Add 403 handler to api_delete.

**AV-R5-3** — Golddigger places stop-loss when bid=0 (price fetch failure). No proximity guard.
`golddigger/runner.py:178-184`. Stop may be placed at arbitrary price relative to market.
Fix: Gate stop-loss placement on bid > 0.

## MEDIUM (P3)

**AV-R5-4** — avanza_orders.request_order accepts sub-1000 SEK orders. Fails at execution time.
**AV-R5-5** — portfolio/avanza/trading.py has no account whitelist or min order size.
**AV-R5-6** — portfolio/avanza/account.get_buying_power returns zero silently on account-not-found.
