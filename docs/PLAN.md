# Plan: Deduplicate Avanza Helpers in metals_loop.py

## Problem

`metals_loop.py` (the metals monitoring loop) contains 6 functions that are near-identical
copies of functions already in `metals_swing_trader.py`. Both files operate on a shared
Playwright `page` object, but neither imports from the other's helpers. This duplication
means bug fixes in one copy don't reach the other (e.g., the stop-loss API endpoint fix
on Mar 2 had to be applied twice).

## Duplication Map

| Function in metals_loop.py | Line | Original in metals_swing_trader.py | Line | Match % |
|---|---|---|---|---|
| `_get_csrf(page)` | 601 | `_get_csrf(page)` | 96 | 100% |
| `fetch_price(page, ob_id, api_type)` | 394 | `_fetch_price(page, ob_id, api_type)` | 104 | 95% (extra `underlying_name` field) |
| `_fetch_account_cash(page)` | 647 | `_fetch_account_cash(page)` | 129 | 100% |
| `_execute_order(page, order)` | 812 | `_place_order(page, ob_id, side, price, volume)` | 159 | 95% (different arg shape) |
| `_place_hardware_stop(page, ...)` | 857 | `_place_stop_loss(page, ...)` | 200 | 98% (hardcoded vs config days) |
| `_check_session_health(page)` | 677 | `avanza_session.verify_session()` | 149 | partial (different impl) |

Additionally, `portfolio/avanza_session.py` has `session_remaining_minutes()` and
`is_session_expiring_soon()` which overlap with the file-age-based expiry warning
in `_check_session_and_alert()`.

## What We'll Do

**Create `data/metals_avanza_helpers.py`** — a single shared module with the canonical
versions of all Avanza Playwright helpers used by both `metals_loop.py` and
`metals_swing_trader.py`.

### Why a new file instead of importing from metals_swing_trader.py?

`metals_swing_trader.py` is a monolithic class-based trader (the `SwingTrader` class). Its
helpers are module-level functions but tightly coupled to its constants (`ACCOUNT_ID`,
`STOP_LOSS_VALID_DAYS`). Extracting them into a standalone helper module is cleaner than
making metals_loop.py import from the trader.

### Why not use portfolio/avanza_session.py?

`avanza_session.py` manages its OWN Playwright instance (module-level `_pw_context`).
`metals_loop.py` and `metals_swing_trader.py` each pass a `page` argument from their own
Playwright sessions. The API pattern is fundamentally different — session.py creates its own
browser, while the loop/trader pass an existing page. Merging would require a large refactor
with high breakage risk. Instead, we'll note the overlap and unify later (TODO).

## What Could Break

1. **Import paths**: Both consumers must import from the new module correctly.
2. **Argument shape differences**: `_execute_order(page, order)` vs `_place_order(page, ob_id, side, price, volume)` — need to support both call patterns or standardize.
3. **Constant differences**: Loop hardcodes `days=8` for stop-loss, swing trader uses `STOP_LOSS_VALID_DAYS` config. Helper must accept it as a parameter.
4. **Return value differences**: `_place_order` returns `{**result, "parsed": body}`, `_execute_order` returns `{"http_status": ..., "parsed": body, "order_id": ...}`. Callers depend on specific keys.
5. **Logging**: Loop uses `log()`, swing trader uses `_log()`. Helper should accept a logger.
6. **`_check_session_and_alert`**: This is unique to the loop (Telegram alerting, global state). NOT a duplicate — keep it in metals_loop.py.

## Execution Order

### Batch 1: Create shared helpers module

1. Create `data/metals_avanza_helpers.py` with:
   - `get_csrf(page)` — extract CSRF token
   - `fetch_price(page, ob_id, api_type)` — fetch live price (superset of both versions)
   - `fetch_account_cash(page, account_id)` — fetch buying power
   - `place_order(page, account_id, ob_id, side, price, volume)` — place BUY/SELL
   - `place_stop_loss(page, account_id, ob_id, trigger_price, sell_price, volume, valid_days=8)` — place hardware stop
   - `check_session_alive(page)` — quick 401 health check

2. Each function takes explicit arguments (no module-level constants), returns consistent shapes.

### Batch 2: Update metals_loop.py

1. Import from `metals_avanza_helpers`
2. Delete duplicated functions: `_get_csrf`, `fetch_price`, `_fetch_account_cash`, `_execute_order`, `_place_hardware_stop`, `_check_session_health`
3. Update all callers to use new import names
4. Keep `_check_session_and_alert` (unique logic), update to call `check_session_alive()`

### Batch 3: Update metals_swing_trader.py

1. Import from `metals_avanza_helpers`
2. Delete: `_get_csrf`, `_fetch_price`, `_fetch_account_cash`, `_place_order`, `_place_stop_loss`
3. Keep: `_delete_stop_loss` (unique to swing trader)
4. Update all callers, adapting to new return value shapes if needed

### Batch 4: Verify & commit

1. Search for any remaining callers of old function names
2. Run a syntax check (python -c "import data.metals_avanza_helpers")
3. Verify metals_loop.py and metals_swing_trader.py both import correctly
4. Commit, merge to main, push

## Decisions

- **Name convention**: Drop leading underscore — these are now public module-level functions.
- **Account ID**: Passed as argument, not imported from config (keeps module dependency-free).
- **Logging**: Functions use `print()` by default; callers wrap in their own logger if needed. Or accept optional `log_fn` parameter. Keep it simple.
- **Return values**: Standardize on `(success: bool, result: dict)` for order functions. Both callers will need minor adaptation.
