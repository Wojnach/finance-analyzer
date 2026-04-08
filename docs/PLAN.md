# PLAN: Cancel stop-losses before sell — fix volume constraint bug

**Branch**: `fix/cancel-sl-before-sell`
**Worktree**: `/mnt/q/wt-cancel-sl-before-sell`
**Date**: 2026-04-08

## Problem

Avanza rejects sell orders with `short.sell.not.allowed` when the sum of
`(active_stop_loss_volume + sell_volume) > position_size`. The platform treats
the overlap as an attempted short-sale.

The metals fishing engine (`data/fish_engine.py` + `data/metals_loop.py`)
places cascade stop-losses immediately after a BUY fill, then later places
SELL orders for the same volume — without first cancelling the stops. The
sell rejects, the position stays open, and the loop logs a confusing error.

This is the same class of bug that contributed to the **-590 SEK loss on
2026-04-07** (per `memory/project_fish_engine_live_test.md`).

## Root cause map (from exploration)

| Path | Location | Cancel BEFORE sell? | Risk |
|---|---|---|---|
| Fish engine SELL | `metals_loop.py:2140-2186` (`_fish_engine_execute_sell`) | NO | **Primary bug** |
| Emergency L3 sell | `metals_loop.py:2780+` (`emergency_sell`) | NO (cancels AFTER, has reactive workaround) | Secondary |
| Trade queue SELL | `metals_loop.py:3197+` (queue execute path) | NO (cancels AFTER via `_handle_sell_fill`) | Secondary |

The reactive workaround in `emergency_sell` parses the `short.sell.not.allowed`
error and re-checks holdings. That is a band-aid — it does not retry the sell
after cancelling, it just gives up.

## Existing infrastructure to reuse

- `portfolio/avanza_session.py:476` — `place_stop_loss()` (works)
- `portfolio/avanza_session.py:576` — `get_stop_losses()` returns list of
  active SLs with `id`, `orderbook.id`, `account.id`, `status` fields
  (verified live against today's BEAR OLJAB X12 position)
- `portfolio/avanza_session.py:259` — `api_delete()` generic DELETE helper
- `portfolio/avanza_control.py:368` — `delete_stop_loss_no_page(account_id, stop_id)`
  uses `api_delete("/_api/trading/stoploss/{accountId}/{stopId}")` correctly
- `data/metals_loop.py:3492` — `_cancel_stop_orders(stop_state, key, csrf)`
  Playwright-based, cancels SLs tracked in `data/metals_stop_orders.json`
- `data/metals_loop.py:2928` — `_cleanup_stop_orders_for(page, key)` wraps
  the above + state cleanup, currently called AFTER sell

## Design

**Two layers** — server-side authoritative + state-file housekeeping:

### Layer 1: Server query + cancel + verify (in `avanza_session.py`)

```python
def cancel_stop_loss(stop_id, account_id=None) -> dict:
    """Cancel a single stop-loss by ID. Idempotent.

    Returns {"status": "SUCCESS"|"FAILED", "http_status": int, "stop_id": str}
    Treats HTTP 404 as success (already gone)."""

def cancel_all_stop_losses_for(orderbook_id, account_id=None, max_wait=3.0) -> dict:
    """Find all active SLs for the given orderbook, cancel them, and poll
    until they actually disappear from the SL list (or timeout).

    The polling is the critical part — Avanza's cancel API returns 200 OK
    immediately, but the encumbered volume isn't released until the SL
    is actually removed from the position's view. Polling every 0.5s with
    a 3s ceiling matches the propagation we have measured.

    Returns:
        {
            "status": "SUCCESS" | "PARTIAL" | "FAILED",
            "cancelled": [stop_id, ...],
            "remaining": [stop_id, ...],   # still showing after timeout
            "elapsed_seconds": float,
        }
    """
```

**Polling design** (the "timer" the user mentioned, with verification):
- t=0:    cancel each SL via DELETE
- t=0.0s: re-query, check if cleared (most cancels are immediate)
- t=0.5s: re-query
- t=1.0s: re-query
- t=1.5s: re-query
- t=2.0s: re-query
- t=2.5s: re-query
- t=3.0s: timeout, log warning, return PARTIAL/FAILED

### Layer 2: Loop integration (in `metals_loop.py`)

```python
def _ensure_stops_cancelled_for(page, key, ob_id, max_wait=3.0) -> bool:
    """Cancel all stops for ob_id BEFORE placing a sell.

    1. Cancel local cascade stops via existing _cancel_stop_orders (page-based,
       same Playwright session as the rest of the loop).
    2. Then call portfolio.avanza_session.cancel_all_stop_losses_for() as a
       belt-and-braces server-side check — catches any stops the loop did not
       know about (manual stops, leftover from a previous session, etc.).
    3. Returns True if both layers report success."""
```

Then wire `_ensure_stops_cancelled_for(...)` into:

1. `_fish_engine_execute_sell()` (line 2140) — **primary fix**
2. `emergency_sell()` (line 2780) — replace the reactive `short.sell.not.allowed`
   handler with proactive cancel-first
3. Trade queue SELL path (line 3197) — add proactive cancel-first

## Why polling, not a fixed timer

A fixed timer (`time.sleep(2)`) is fragile:
- If Avanza is slow (US market open volatility), 2s is not enough
- If Avanza is fast (most cases), 2s wastes time on every sell

Polling with a hard ceiling is the standard OMS pattern (per online research:
"cancel-then-replace with verification" — see Charles Schwab/Fidelity docs on
conditional orders). It is optimistic on the happy path and bounded on the
sad path.

## What could break

1. **Current oil position**: BEAR OLJAB X12 AVA 2 (id 2367798) has an active
   SL (id A2^1773297348702^1346781). The new code is in metals_loop, which
   does not manage the oil position — so this fix cannot affect it.
2. **Live silver/gold positions**: Any active fishing position would be
   touched by the new code path. As of right now (2026-04-08 09:00 CET), the
   fish engine state shows `position: null` — no live position.
3. **Test mocks**: The new helpers must be mockable for unit tests. Use
   monkeypatch on module-level `get_stop_losses` / `api_delete`.
4. **Race with stop-loss trigger**: If a stop fires DURING our cancel→sell
   sequence, the position is already gone. The next sell would fail
   harmlessly with "no position". Acceptable — fish engine reconciliation
   already handles this case via `_reconcile_fish_engine_position()`.
5. **Multiple stops for same instrument**: The cascade places up to 3 stops
   per position. `cancel_all_stop_losses_for` handles this by enumerating
   ALL active SLs for the orderbook, not relying on any local state.

## Execution batches

### Batch 1: Helpers in `avanza_session.py` + tests
- Add `cancel_stop_loss(stop_id, account_id=None)` to `portfolio/avanza_session.py`
- Add `cancel_all_stop_losses_for(orderbook_id, account_id=None, max_wait=3.0)` to same
- Add `tests/test_avanza_session_cancel_sl.py` covering:
  - happy path: 1 SL exists → cancelled → cleared on first poll
  - happy path: 0 SLs exist → returns SUCCESS immediately
  - retry: 1 SL exists → cancel API returns 200 → first poll still shows it → second poll cleared
  - timeout: 1 SL exists → cancel API returns 200 → polls never show clear → returns PARTIAL after max_wait
  - 404 cancel: SL already gone → treated as success
  - filter: 2 SLs across different orderbooks → only target ob_id is cancelled
- Run targeted tests
- Commit: `feat(avanza_session): add cancel_stop_loss with polling verification`

### Batch 2: Integration in `metals_loop.py`
- Add `_ensure_stops_cancelled_for(page, key, ob_id, max_wait=3.0)` near the
  existing `_cleanup_stop_orders_for` function (~line 2928)
- Call it BEFORE `place_order(..., "SELL", ...)` in:
  - `_fish_engine_execute_sell()` (line ~2150 area, before the place_order call at 2172)
  - `emergency_sell()` (line ~2790, before the direct API POST at 2807)
  - Trade queue SELL execution (line ~3195, before the place_order at 3197)
- Keep the existing AFTER cleanup (`_cleanup_stop_orders_for`) as a safety
  net for state file housekeeping.
- Commit: `fix(metals_loop): cancel stops before sell to avoid volume constraint`

### Batch 3: Verification & ship
- Run full test suite: `.venv/Scripts/python.exe -m pytest tests/ -n auto`
- Run codex adversarial review:
  `/codex:adversarial-review --wait --scope branch --effort xhigh`
- Address findings (or document why they are false positives)
- Re-run tests after any fixes
- Merge to main
- Push via Windows git
- Restart `PF-MetalsLoop` scheduled task to pick up the change

## Open question (will decide during impl)

Should `_ensure_stops_cancelled_for` block the sell on FAILED, or should it
log a warning and proceed? My read: **block on FAILED**. If we cannot cancel
the stops, the sell will reject anyway — better to surface the cancel failure
clearly than to attempt a doomed sell and confuse the operator. PARTIAL
(some cancelled, some not) → still block, investigate.

## Out of scope

- Extending the metals loop to handle oil positions (separate concern)
- Refactoring `_loop_page` global state (separate concern)
- Building a standalone oil position monitor (separate concern, may follow up)
