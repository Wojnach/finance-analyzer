# Stop-Loss Setup Guide — Metals Loop

How to activate the full multi-layer protection stack for a new warrant position.

## Protection Layers (7 total)

The metals loop provides 7 independent protection layers. **All 7 require the position to be
registered in `POSITIONS_DEFAULTS` in `metals_loop.py`** — hardware stops on Avanza are the
only layer that works independently.

| Layer | Name | Trigger | Action | Independent? |
|-------|------|---------|--------|--------------|
| HW | Hardware stop-loss | Price hits trigger on Avanza | Avanza auto-sells | Yes — broker-side |
| L1 | Software warning | Bid within 8% of `stop` price | Log + flag in context | Needs POSITIONS |
| L2 | Software alert | Bid within 5% of `stop` price | Telegram + force Claude invocation | Needs POSITIONS |
| L3 | Emergency auto-sell | Bid within 2% of `stop` price | Immediate sell via API | Needs POSITIONS |
| MOM | Momentum exit | Accelerating decline (velocity + accel) | Immediate sell | Needs POSITIONS |
| TRAIL | Trailing stop | Bid gained 2%+ then dropped 3% from peak | Cancel + re-place stops higher | Needs POSITIONS |
| AUTO | Auto-exit override | 5+ checks in L2 zone with declining trend | Immediate sell | Needs POSITIONS |

### How they cascade (silver_sg example, stop=46.0)

```
Bid 52.0 (entry)     — all clear
Bid 50.0 (L1 zone)   — L1 WARNING logged, flagged in context
Bid 48.4 (L2 zone)   — L2 ALERT: Telegram + Claude forced invocation
Bid 46.9 (L3 zone)   — L3 EMERGENCY: auto-sell immediately via API
Bid 46.0              — Hardware S1 fires (Avanza sells 221 units)
Bid 41.0              — Hardware S2 fires (Avanza sells 220 units)
```

The momentum exit can fire at any time if the derivative (velocity + acceleration) shows
an accelerating decline, but only when already in L1+ zone (`MOMENTUM_REQUIRE_L1 = True`).

## Setup Steps

### Step 1: Add to POSITIONS_DEFAULTS

In `data/metals_loop.py`, add the position to `POSITIONS_DEFAULTS` dict:

```python
POSITIONS_DEFAULTS = {
    # ... existing positions ...
    "silver_sg": {
        "name": "MINI L SILVER SG",   # display name for Telegram/logs
        "ob_id": "2043157",            # Avanza orderbook ID
        "api_type": "warrant",         # "warrant" or "certificate"
        "units": 441,                  # number of units held
        "entry": 52.0,                 # purchase price (SEK per unit)
        "stop": 46.0,                  # reference price for L1/L2/L3 calculations
        "active": True,                # must be True to enable monitoring
    },
}
```

**How to choose `stop`**: Set it to the first hardware stop-loss trigger price. The software
layers (L1 at 8%, L2 at 5%, L3 at 2%) measure distance from this price, so L3 fires just
before the hardware stop catches it.

**How to find the orderbook ID**: Go to Avanza, open the instrument page. The URL contains
the ID: `https://www.avanza.se/borshandlade-produkter/warranter-teckningsoptioner/om-warranten.html/XXXXXXX/...`

### Step 2: Add to positions state file

Create/update `data/metals_positions_state.json`:

```json
{
  "silver_sg": {
    "active": true,
    "units": 441,
    "entry": 52.0,
    "stop": 46.0
  }
}
```

This persists across restarts. The state file overrides DEFAULTS for fields it contains.
The loop now writes this file atomically via `portfolio.file_utils.atomic_write_json()`.
If you edit it manually, stop the loop first and replace the whole JSON document in one save.

### Step 3: Place hardware stop-loss orders

Use the `place_stop_loss()` helper from `portfolio.avanza_control.py`. The loop does this
automatically via `place_stop_loss_orders()`, but for manual placement:

```python
from portfolio.avanza_control import place_stop_loss

# Split units across 2-3 cascading levels
# Level 1: trigger near software L3, catches initial drop
success1, stop_id1 = place_stop_loss(
    page, ACCOUNT_ID, "2043157",
    trigger_price=46.0,   # when bid drops to this, order activates
    sell_price=45.5,       # sell at 0.5 below trigger (slippage buffer)
    volume=221,            # half the position
    valid_days=8,
)

# Level 2: deeper stop, catches flash crash
success2, stop_id2 = place_stop_loss(
    page, ACCOUNT_ID, "2043157",
    trigger_price=41.0,
    sell_price=40.5,
    volume=220,            # remaining units
    valid_days=8,
)
```

**CRITICAL**: Use `/_api/trading/stoploss/new` (the dedicated stop-loss API), NOT the regular
order API. The regular API causes "crossing prices" errors. See `memory/avanza.md`.

Save the stop order IDs to `data/metals_stop_orders.json`:

```json
{
  "silver_sg": {
    "date": "2026-03-03",
    "orderbook_id": "2043157",
    "units": 441,
    "orders": [
      {"trigger": 46.0, "sell": 45.5, "vol": 221, "id": "...", "status": "placed"},
      {"trigger": 41.0, "sell": 40.5, "vol": 220, "id": "...", "status": "placed"}
    ],
    "placed_ts": "2026-03-03T07:51:09Z"
  }
}
```

The metals loop now writes `data/metals_stop_orders.json` atomically as shared state.
Avoid partial manual edits while the loop is running.

### Step 4: Restart the loop

The loop reads POSITIONS on startup via `_load_positions()`. After restart:
- Price fetching activates for the new position (every 90s)
- L1/L2/L3 distance calculations start
- Momentum tracking starts collecting price history
- Trailing stop monitoring begins
- Holdings verification includes the new position

### Step 5: Verify

Check loop output (`data/metals_loop_out.txt`) for:
```
Position state loaded from data/metals_positions_state.json
```

After a few checks, you should see status lines like:
```
silver_sg:51.55(−0.9%)
```

## Removing a Position

When you sell: the loop's `emergency_sell()` or `_execute_sell()` sets `active: false` and
persists to the state file. Manual removal:

1. Set `"active": false` in `metals_positions_state.json`
2. Cancel hardware stop orders on Avanza (or let them expire after `valid_days`)
3. Position key stays in DEFAULTS (harmless when inactive) or can be removed later

## Config Reference

All tuning constants in `metals_loop.py` (lines 106-144):

| Constant | Default | Purpose |
|----------|---------|---------|
| `STOP_L1_PCT` | 8.0 | L1 warning zone (% of bid from stop) |
| `STOP_L2_PCT` | 5.0 | L2 alert zone |
| `STOP_L3_PCT` | 2.0 | L3 emergency auto-sell zone |
| `STOP_ORDER_ENABLED` | True | Enable hardware stop-loss placement |
| `STOP_ORDER_LEVELS` | 3 | Number of cascading stop orders |
| `STOP_ORDER_SPREAD_PCT` | 1.0 | Spread between stop levels (%) |
| `TRAIL_START_PCT` | 2.0 | Start trailing after this % gain |
| `TRAIL_DISTANCE_PCT` | 3.0 | Trail this % below peak bid |
| `TRAIL_MIN_MOVE_PCT` | 1.0 | Minimum move to update stop (avoid spam) |
| `MOMENTUM_ENABLED` | True | Enable derivative-based early exit |
| `MOMENTUM_LOOKBACK` | 5 | Checks to analyze (5 * 90s = ~7.5 min) |
| `MOMENTUM_MIN_VELOCITY` | -0.5 | Minimum decline rate (%/check) |
| `MOMENTUM_ACCEL_THRESHOLD` | -0.1 | Acceleration must be negative |
| `MOMENTUM_REQUIRE_L1` | True | Only trigger in L1+ danger zone |
| `AUTO_EXIT_L2_CHECKS` | 5 | Auto-sell after N checks in L2+ zone |

## What Saved silver301 on Mar 2, 2026

The L3 emergency auto-sell fired when MINI L SILVER AVA 301's bid dropped to 16.0 SEK,
which was within 2% of its stop price (15.73). The loop detected this in `check_triggers()`,
matched the "L3 EMERGENCY" pattern, and called `emergency_sell()` which placed an immediate
market sell via the Avanza API. The hardware stop orders were a backup but L3 caught it first.

Without software protection (the situation silver_sg was in before this fix), only the hardware
stops at 46.0 and 41.0 would have fired — meaning the position could have dropped 12-21%
before any automatic action.
