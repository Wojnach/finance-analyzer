# Agent Review: avanza-api

## P1 Findings
1. **Telegram CONFIRM path uses TOTP** — bypasses order lock and BankID session (avanza_orders.py:17, avanza_control.py:43). Confidence: 95

## P2 Findings
1. **delete_stop_loss missing order lock** (metals_avanza_helpers.py:457). Confidence: 90
2. **get_portfolio_value/get_open_orders TOTP-only** — no BankID fallback (avanza_client.py:202,221). Confidence: 85
3. **Stale price execution** — no fill verification after confirmed orders (avanza_orders.py:130-135). Confidence: 80
4. **Session expiry check bypassed** on corrupt expires_at (avanza_session.py:89-90). Confidence: 82

## P3 Findings
1. delete_stop_loss treats 404 as failure (should be success) (metals_avanza_helpers.py:483)
2. priceType in stop-loss payload set to value_type (avanza_session.py:769)
3. ResilientPage __getattr__ no recovery
