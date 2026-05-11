# Grid Market-Maker Runbook

## What it is

Continuous limit-order ladder on leveraged warrants. Each cycle the
fisher:

1. Reconciles its in-memory state against live Avanza (fills, external
   cancels).
2. Rotates any newly-detected buy fills into a paired sell limit at
   `+GRID_TARGET_PCT` and a stop-loss at `-GRID_STOP_PCT`.
3. For each active-direction instrument: arms missing tiers below the
   current bid; cancels armed buys on the opposite-direction
   instrument when the signal direction flips.
4. Near session close: cancels unfilled buys (10 min before) and
   force-flats inventory (5 min before).

Inspired by Marja Folcke's "many small streams" microcap strategy.
Adapted for warrants: smaller per-leg target (1.2 % vs her 5-10 %)
because warrant spreads are tighter and 5x leverage amplifies the
underlying move.

## Files

| Path                                   | Purpose                                  |
|----------------------------------------|------------------------------------------|
| `portfolio/grid_fisher_config.py`      | All tunable constants                    |
| `portfolio/grid_fisher.py`             | Orchestrator + state machine + tick      |
| `portfolio/grid_tiers.py`              | Pure tier math + exit-level math         |
| `data/grid_fisher_state.json`          | Runtime state (atomic JSON)              |
| `data/grid_fisher_decisions.jsonl`     | Append-only decision log                 |
| `data/fin_fish_config.py`              | Warrant catalog + PREFERRED_INSTRUMENTS  |
| `dashboard/app.py` `/api/grid-fisher`  | Ops endpoint for state + recent log      |

## Default settings

| Knob                                  | Value      | Notes                              |
|---------------------------------------|------------|------------------------------------|
| `GRID_FISHER_ENABLED`                 | `True`     | Master switch                      |
| `GRID_TIERS`                          | 3          | Number of buy-side rungs per side  |
| `GRID_TIER_SPACING_PCT`               | (0.3,0.8,1.5) | Tier offsets below bid          |
| `GRID_TARGET_PCT`                     | 1.2        | Sell limit above fill              |
| `GRID_STOP_PCT`                       | 3.5        | Stop-loss below fill               |
| `GRID_LEG_SEK`                        | 1200       | Per-leg SEK notional               |
| `GRID_PER_INSTRUMENT_MAX_SEK`         | 6000       | Inventory cap per instrument       |
| `GRID_PER_SESSION_LOSS_LIMIT_SEK`     | 500        | Per-instrument session loss floor  |
| `GRID_MIN_SIGNAL_CONFIDENCE`          | 0.56       | Matches metals swing trader floor  |
| `GRID_DIRECTION_FLIP_COOLDOWN_MIN`    | 30         | Min minutes between direction flips |
| `GRID_EOD_SWEEP_MINUTES_BEFORE`       | 10         | Cancel buys this far from close    |
| `GRID_EOD_MARKET_SELL_MINUTES_BEFORE` | 5          | Force-flat inventory               |
| `GRID_MAX_ORDERS_PER_MIN`             | 10         | Sliding-window rate limit          |

## Active instruments (2026-05-11)

| Underlying | LONG cert                | SHORT cert                |
|------------|--------------------------|---------------------------|
| XAG-USD    | BULL_SILVER_X5_AVA_4     | BEAR_SILVER_X5_AVA_12     |
| XAU-USD    | BULL_GULD_X5_AVA         | BEAR_GULD_X5_VON4         |
| OIL-USD    | BULL_OLJAB_X5_AVA_2      | BEAR_OLJAB_X5_AVA_2       |

OIL-USD is **seeded but idle** — agent_summary.json does not currently
publish OIL-USD signals, so each tick logs `no_direction` for both
oil orderbooks and moves on without placing orders. Bring oil online
by adding OIL-USD to the metals/oil signal pipeline; no changes are
needed in the grid fisher itself.

## How to disable in a hurry

Flip the master switch in `portfolio/grid_fisher_config.py`:

```python
GRID_FISHER_ENABLED = False
```

The next cycle will skip every placement and log
`skipped_reason: "disabled"`. Existing live orders are **not
cancelled** — that requires the EOD sweep or a manual run:

```python
from portfolio.grid_fisher import GridFisher
from portfolio import avanza_session
from data.fin_fish_config import FULL_CATALOG
g = GridFisher(session=avanza_session, catalog=FULL_CATALOG)
g.eod_cancel_buys()   # cancels resting buys only
g.eod_market_flat()   # force-flats inventory at bid - 1%
```

## Probe mode

Set `GRID_FISHER_PROBE_ONLY = True` to log intended placements as
`probe_placement` decision entries without posting to Avanza. Useful
for a single-cycle smoke test after a config tweak. Toggle back to
`False` once you've reviewed the log.

## Where decisions land

Every state mutation appends one line to
`data/grid_fisher_decisions.jsonl`. Categories you'll see:

| Category                | Meaning                                         |
|-------------------------|-------------------------------------------------|
| `place_buy`             | Buy limit placed at Avanza                      |
| `place_buy_rejected`    | Avanza returned non-SUCCESS                     |
| `place_buy_failed`      | Network / session error                         |
| `cancel_buy`            | We cancelled an armed buy                       |
| `cancel_failed`         | Cancel API errored                              |
| `fill_buy` / `fill_sell`| Reconcile detected a fill                       |
| `rotate`                | Sell limit + stop placed after a buy fill       |
| `external_cancel_buy`   | Order disappeared without filling (manual)     |
| `inventory_drift`       | Live volume disagrees with cached value         |
| `flip_direction`        | Direction switched (manual via `arm_direction`) |
| `eod_cancel_buys`       | Pre-close cancel sweep                          |
| `eod_market_sell`       | Pre-close market sell                           |
| `halt_global`           | Global drawdown limit breached                  |
| `skip_*`                | Placement skipped (cap, cooldown, loss, etc.)   |

## Read state via API

```
GET /api/grid-fisher?token=<dashboard_token>
```

Returns `{state, recent_decisions}`. `state` mirrors
`data/grid_fisher_state.json` and `recent_decisions` is the last 50
JSONL entries.

## Per-instrument lifecycle (concrete)

XAG signal flips BULL → BEAR with confidence 0.6:

1. `BULL_SILVER_X5_AVA_4` (active_direction = LONG)
   - signal direction (SHORT) != active_direction (LONG)
   - All armed BULL buys cancelled at Avanza
   - Existing inventory + sell ladder + stop preserved
   - Decision log: `cancel_buy` per tier, then nothing on subsequent
     ticks until signal flips back to LONG
2. `BEAR_SILVER_X5_AVA_12` (active_direction = SHORT)
   - signal direction (SHORT) == active_direction (SHORT)
   - Confidence (0.6) ≥ min (0.56) → arm
   - 3 tiers placed below bid

When BEAR ladder fills: rotation places sell at fill*1.012 plus stop
at fill*0.965 sized to the full current BEAR inventory.

## Risks (read before flipping live)

- **Spread tightens, edge disappears.** Warrant spreads can drop to
  0.05 % on calm days. Round-trip courtage on a 1200 SEK leg is ~2 SEK
  (0.17 %). Target 1.2 % gives a net edge of ~0.5 % per cycle in
  normal conditions; on a flash-tightening day this becomes ~0.3 %.
  Acceptable for now; watch decisions log for fill density to verify.
- **Trend regimes destroy MM.** ADX > 35 with signal against the
  trend → we keep filling buys that never revert. The signal_engine
  already biases toward with-trend during high-ADX windows, so the
  signal direction alone is the gate. If you observe a persistent
  counter-trend fish, drop confidence threshold or disable.
- **Knockout cliff.** 5x certs are vulnerable to sharp moves past
  the barrier. Tiers within 8 % of barrier are auto-skipped
  (`GRID_KNOCKOUT_SAFETY_PCT`). Watch for barrier drift on Avanza —
  it changes daily for MINIs; certs reset weekly.
- **Old fish-engine lost 12K SEK on 2026-04-15.** That engine had no
  signal direction gate (it placed both BULL and BEAR ladders
  simultaneously). This design only arms the with-signal direction
  and cancels the opposite side, which is the structural fix.
- **DST flip.** EOD uses `Europe/Stockholm` via zoneinfo. Watch the
  `eod_cancel_buys` decision around 03:00 on transition nights.

## Smoke test after merge

1. Restart `PF-MetalsLoop` (Windows cmd):
   `schtasks /run /tn PF-MetalsLoop`
2. Tail `data/metals_loop_out.txt`. Look for:
   `Grid fisher: ACTIVE (6 instruments, enabled=True)`
3. After ~2 cycles, inspect the decision log:
   `tail -20 data/grid_fisher_decisions.jsonl`
4. Hit the dashboard:
   `GET /api/grid-fisher?token=<token>`
5. Verify the state file:
   `cat data/grid_fisher_state.json`
6. Check Avanza for any new resting orders matching the catalog
   orderbook IDs.

If anything looks wrong, set `GRID_FISHER_ENABLED = False` and
restart the loop. Inventory continues to exit on its own ladder + stop.
