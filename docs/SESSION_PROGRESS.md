# Session Progress — Avanza Pipeline Overhaul

**Date:** 2026-03-31
**Session:** "optimization" → "mrcrypto"
**Branch:** main (all merged and pushed)
**Last push:** 14ec331

---

## What Was Done

### Phase 1: New `portfolio/avanza/` Package (COMPLETE)
- 11 modules: auth, client, types, market_data, trading, account, search, tick_rules, streaming, scanner, __init__
- 170+ unit tests passing in 0.5s
- TOTP auth singleton (thread-safe, auto-renewable) — needs TOTP creds to test live
- `requests.Session` connection pooling (replaces Playwright for runtime API calls)
- WebSocket streaming client (CometD/Bayeux) — built, needs TOTP for pushSubscriptionId
- Instrument scanner — search + rank warrants/certs by spread/leverage/barrier. Works with BankID.
- Live-tested scanner: BULL OLJA (1.2s), MINI SILVER (0.4s), BULL GULD (0.4s)

### Phase 2: Playwright Migration (MOSTLY COMPLETE)
Files fully migrated (no more Playwright):
- `scripts/fin_fish_monitor.py` — all page.evaluate → api_get/api_post
- `data/place_stoploss_once.py` — ctx.request → api_post/api_delete
- `data/gold_sell_final.py` — page.evaluate → api calls
- `data/gold_sell_retry.py` — same
- `data/gold_sell_debug.py` — same
- `portfolio/fin_snipe_manager.py` — fully migrated via _no_page functions

New page-free functions in `avanza_control.py`:
- `fetch_price_no_page()`, `fetch_price_no_page_with_fallback()`
- `place_order_no_page()`, `place_stop_loss_no_page()`, `place_trailing_stop_no_page()`
- `delete_order_no_page()`, `delete_stop_loss_no_page()`

New functions in `avanza_session.py`:
- `place_stop_loss()` — hardware stop via /_api/trading/stoploss/new
- `place_trailing_stop()` — FOLLOW_DOWNWARDS (Avanza manages trail)
- `get_stop_losses()` — list active stop-loss orders
- `api_delete()` — authenticated DELETE requests

### Phase 3: Hardware Trailing Stops (WIRED IN)
- `avanza_session.place_trailing_stop(ob_id, trail_percent, volume)` — BankID path
- `portfolio.avanza.trading.place_trailing_stop(ob_id, trail_percent, volume)` — TOTP path
- **INTEGRATED into metals_loop.py** — `HARDWARE_TRAILING_ENABLED = True`
- On each BUY fill: places a single FOLLOW_DOWNWARDS 5% trailing stop via avanza_session
- Avanza manages the trail server-side — works even if process crashes
- Software trailing skipped when hardware trailing is active
- Stop ID stored in position state as `hw_trailing_stop_id`
- Live verified: stop-loss API reachable, functions callable, BankID session works

---

## What's NOT Done (Future Sessions)

### Needs TOTP Auth (user to configure)
1. WebSocket live test — `AvanzaStream` needs `pushSubscriptionId`
2. Smoke test — `scripts/avanza_smoke_test.py`
3. Parallel scanner — 6x faster with TOTP vs BankID

### Needs Careful Work (High Risk, separate session)
4. **metals_loop.py full Playwright removal** — 4,500 lines, ~50 page.evaluate() calls remain
   - Hardware trailing is wired in (done)
   - Remaining: replace page.evaluate() for price fetching, position reconciliation, order fills
   - Replace metals_avanza_helpers with avanza_session/_no_page functions
   - Remove sync_playwright() context manager + browser management
   - TEST EXTENSIVELY before deploying

5. **metals_avanza_helpers.py deprecation** — dead code once metals_loop migrated
6. **avanza_client.py simplification** — delegate to portfolio.avanza.client

---

## Key Files

### New Package
```
portfolio/avanza/__init__.py      — 30 exports
portfolio/avanza/auth.py          — TOTP singleton
portfolio/avanza/client.py        — HTTP wrapper
portfolio/avanza/types.py         — 18 dataclasses
portfolio/avanza/market_data.py   — quotes, depth, OHLC
portfolio/avanza/trading.py       — orders, stops
portfolio/avanza/account.py       — positions, cash
portfolio/avanza/search.py        — instrument search
portfolio/avanza/tick_rules.py    — price rounding
portfolio/avanza/streaming.py     — WebSocket CometD
portfolio/avanza/scanner.py       — search+rank instruments
```

### Design Docs
- `docs/superpowers/specs/2026-03-30-avanza-pipeline-overhaul-design.md`
- `docs/superpowers/plans/2026-03-30-avanza-pipeline-overhaul.md`

### Quick Resume
```bash
git log --oneline -5
.venv/Scripts/python.exe -m pytest tests/test_avanza_pkg/ -q
# Scanner test (needs BankID):
.venv/Scripts/python.exe -c "from portfolio.avanza.scanner import scan_instruments, format_scan_results; print(format_scan_results(scan_instruments('SILVER', direction='BULL', limit=5)))"
```
