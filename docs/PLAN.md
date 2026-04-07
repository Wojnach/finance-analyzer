# Fix: Fish Engine Integration — 6 Bugs From Live Test

## Date: 2026-04-07

## Problem

First live test of fish engine lost 590 SEK due to 6 integration bugs between
`data/fish_engine.py` (decision logic, works correctly) and `data/metals_loop.py`
(execution layer, broken wiring).

## Root Causes (priority order)

### Bug 1 — CRITICAL: fetch_price returns None, crashes execute functions
`fetch_price(_loop_page, ob_id, "warrant")` returns None for engine instruments.
Fix: use `fetch_price_with_fallback(_loop_page, ob_id)` which tries multiple
api_types, and guard against None return.

### Bug 2 — CRITICAL: trade_guard returns [] (empty list = no blocks), treated as False
`bool([])` = False, blocking ALL trades permanently.
Fix: `len(guard) == 0` means OK (already partially fixed on main, need to bring to worktree).

### Bug 3 — CRITICAL: EXIT_METALS_DISAGREE_COUNT too aggressive
Was 3 (3 minutes), sold at exact bottom twice. Already changed to 15 on main.
Additional fix: never trigger metals disagree exit when RSI < 30 (oversold) or RSI > 70
(overbought) — those are the exact zones where contrarian fishing positions should hold.

### Bug 4 — MODERATE: _loop_page scope / None guard
Module-level `_loop_page` global was added on main. Need proper None guards in both
execute functions. Already partially fixed.

### Bug 5 — MODERATE: Telegram send_telegram wrong module
Execute functions imported from `portfolio.telegram_notifications` instead of using
the metals loop's own module-level `send_telegram` function (which respects mute config).
Already fixed on main.

### Bug 6 — LOW: HOLD decisions not logged
Engine returns HOLD ~98% of ticks but nothing was logged. Already added periodic
HOLD logging on main.

## Implementation Plan

### Batch 1: Bring all main fixes into worktree + fix remaining bugs (data/metals_loop.py)

Changes to `_fish_engine_execute_buy`:
- Use `fetch_price_with_fallback(_loop_page, ob_id)` instead of `fetch_price`
- Guard against None return
- Use `fetch_account_cash(_loop_page, ACCOUNT_ID)` (add ACCOUNT_ID)
- Use `place_order(_loop_page, ACCOUNT_ID, ...)` for buy
- Add `_loop_page is None` guard

Changes to `_fish_engine_execute_sell`:
- Use `fetch_price_with_fallback(_loop_page, ob_id)` instead of `fetch_price`  
- Guard against None return
- Use `place_order(_loop_page, ACCOUNT_ID, ...)` for sell
- Add `_loop_page is None` guard
- Use module-level `send_telegram` (not import)
- Cancel any active stop-losses for the position before selling

Changes to `_run_fish_engine_tick`:
- Fix spread calculation to use `fetch_price_with_fallback` instead of `avanza_session`
- Fix trade_guard empty-list bug
- Add periodic HOLD logging

Module-level:
- Add `_loop_page = None` global
- Set `_loop_page = page` in Playwright init block
- Keep `FISH_ENGINE_ENABLED = False` (user enables manually)

### Batch 2: Smarter exit rules (data/fish_engine.py)

- `EXIT_METALS_DISAGREE_COUNT = 15` (already on main)
- Add RSI guard to metals disagree exit: skip when RSI < 30 or RSI > 70
  (oversold/overbought = contrarian zone, don't exit on disagreement)
- Add configurable `EXIT_METALS_DISAGREE_RSI_GUARD = True`

### Batch 3: Tests (tests/test_fish_engine_integration.py)

- Test _fish_engine_execute_sell with mock fetch_price returning None
- Test _fish_engine_execute_sell with mock place_order returning failure
- Test trade_guard empty-list handling
- Test metals_disagree doesn't fire when RSI < 30
- Test metals_disagree doesn't fire when RSI > 70
- Test HOLD logging frequency

### Non-goals
- NOT rewriting the engine tactics (they work, just need data)
- NOT adding new instruments to the catalog (separate task)
- NOT enabling FISH_ENGINE_ENABLED automatically (user flips it manually)
