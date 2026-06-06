# Adversarial review — metals-core — 2026-06-06

Scope: grid_fisher + grid_tiers + grid_fisher_config, fin_snipe_manager/fin_snipe,
metals_swing_trader stop path, metals_ladder, kelly_metals, oil_grid_signal,
avanza_session/avanza_control plumbing, metals_loop grid integration. Files
verified pure/no-live-orders: exit_optimizer.py, iskbets.py, silver_monitor.py
(deprecated), metals_risk.py.

## P0

portfolio/grid_fisher.py:1940-1973: EOD market-flat cancels the broker stop
(`inst.stop_loss_id = None` set unconditionally at 1948, even if `cancel_stop_loss`
returned None/failed) BEFORE the replacement sell is confirmed. The replacement is a
limit at `bid*0.99` (NOT a true market order — Avanza warrants can't post market), which
during the illiquid 21:50–21:55 close auction may not fill. If `place_sell_order`
returns None/rejected (1966/1976 → `continue`) the position is left with NO stop AND NO
sell → guaranteed naked leveraged inventory carried overnight. Even on SUCCESS, an
unfilled resting limit + removed stop = naked overnight exposure if the auction doesn't
clear it. → Place/confirm the replacement sell (or re-place the stop) BEFORE removing
`stop_loss_id`; only null the stop after a confirmed working sell; on sell
failure/rejection, re-arm the stop rather than leaving the position unprotected.

portfolio/grid_fisher.py:1908 (`eod_market_flat` fill assumption): the EOD sell is
recorded as filled/rotated via the next tick's reconcile only if it actually executes,
but `eod_sell_order_id` is set on SUCCESS-acknowledged-but-unfilled orders. Across the
day boundary `roll_session_if_new_day` (line 517) clears `eod_sell_order_id` while
inventory_units may still be >0 (sell never filled). Next session the EOD guard at
1923 no longer sees an in-flight order and the stop is gone (nulled at 1948 prior day),
so the position sits unprotected until the next EOD window. → On session roll, if
inventory_units>0 and the prior EOD sell is unconfirmed, flag `stop_needs_rearm=True`
so the new session re-protects held inventory instead of waiting for EOD.

## P1

portfolio/grid_fisher.py:730-735 / 752-757: reconcile partial-fill path mutates
`tier.qty = int(delta)` IN PLACE on the persisted TierOrder, then calls record_fill.
The original ordered qty is lost. If the same order later fills the remainder (Avanza
keeps a reduced-volume partial in the open list, then completes it), the next reconcile
computes delta against the now-shrunk tier.qty and under-counts the additional fill,
leaving inventory_units below true live volume. The final drift block (765-793) papers
over the under-count by realigning to live_vol, but the realigned units get an *estimated*
avg entry from the cheapest armed tier — corrupting avg_entry_price and therefore every
downstream P&L and exit/stop level. → Track filled-qty separately (e.g. `filled_qty`
field) instead of overwriting `tier.qty`; keep original qty for delta math.

data/metals_swing_trader.py:2759-2760 / _set_stop_loss: stop trigger is anchored purely
to warrant % (`stop_anchor * (1 - sl_pct/100)`) with no validation against the warrant's
knockout barrier. On a deep-leverage cert a fixed warrant-% stop can land below the
knockout barrier price, where it is meaningless (the cert is already worthless / knocked
out). The selection-time barrier gate (2501) only checks the *entry* barrier distance,
not the stop trigger. → Clamp the computed trigger to stay a safe margin above (LONG) /
below (SHORT) the live barrier, or reject the stop and alert if no valid trigger exists.

portfolio/grid_fisher.py:1546-1547 vs barrier: rotate_on_buy_fill places the protective
stop at `stop_price = fill_price*(1-3.5%)` (grid_tiers.build_exit_levels) with NO
barrier-proximity check, unlike the buy-ladder which calls `_tier_skip_for_knockout`.
A 3.5% warrant-space stop on a 5x cert near its barrier can sit below the knockout. →
Run the same knockout-proximity guard on the stop trigger before placing it.

## P2

portfolio/avanza_session.py:653-669 + grid_fisher.py:1664: `get_open_orders` swallows
RuntimeError on BOTH endpoints and returns `[]` (not None). grid_fisher's degraded-fetch
guard only trips when the result is None. A transient orders-endpoint failure that still
lets `get_positions` succeed yields open_order_ids=∅, so reconcile marks every resting
buy as cancelled/filled by position delta — spurious fills/cancels and phantom rotations.
→ Make `get_open_orders` raise (or return None) on total failure so the existing
None-guard fails closed.

portfolio/grid_fisher.py:1839-1848 + avanza_session.py:678: Gate A and the bid fallback
fetch warrant/cert quotes through `/_api/market-guide/stock/{ob_id}/quote` (a *stock*
endpoint) for non-stock orderbooks. If the endpoint returns a divergent shape for
warrants, `timeOfLast` is absent → every instrument reads "stale" → grid never places
(silent no-op). Comment at 1138 claims empirical verification 2026-05-18; flag for
re-verification per active ob_id. → Use the instrument-typed market-guide path (as
fin_snipe._fetch_market_guide does) for quote/staleness.

portfolio/grid_fisher.py:1417-1425 global-cap per-tier check uses
`global_planned_notional(self.state)` recomputed each tier. Tiers placed earlier in the
SAME loop are appended to inst.buy_ladder (1463) so they are counted — good — but the
per-instrument loop iterates instruments sequentially within one tick and the global cap
is captured ONCE (`effective_cap` at 1864) before the per-instrument loop; placements on
instrument A within this tick are reflected via global_planned_notional, so ordering is
consistent. No cap breach found, but the cap is computed from cached buying power
(`GRID_BUYING_POWER_CACHE_SECS=60`) so two concurrent processes (metals_loop +
fin_snipe_manager both authenticated to acct 1625505) could each place up to the cap.
→ Confirm single-writer; the file `_state_lock` does not serialize across processes.

portfolio/fin_snipe_manager.py:1633-1648: naked-position detection runs only AFTER
execute_actions and only alerts (Telegram) — it does not re-place protection that cycle.
With two-phase staged replacement (1175-1191) a non-emergency stop reprice cancels this
cycle and places next cycle; the `emergency` flag keys off `len(open_stops)==0`, but if
the existing stop is present-but-mispriced the position spends a full interval (up to
`args.interval`, default 60s, or 5s fast-recheck) with only the stale stop. Acceptable
given a stop still exists, but the alert-without-action on a truly naked position is a
gap. → On detected naked position, schedule an immediate emergency stop placement.

## P3

portfolio/grid_fisher.py:1955/1959: EOD sell fallback price = `avg_entry_price` when the
quote fetch fails — for a losing position this is ABOVE market and will not fill,
defeating the "force flat" intent. → Fall back to last-known bid or a wider aggressive
offset, or escalate to alert when no live quote is available at EOD.

portfolio/oil_grid_signal.py:88/92: confidence floor `base = min(0.55, 0.5 + ...)` caps
the trend component at 0.55, then adds up to `rsi_distance*0.2`; with GRID_MIN_SIGNAL_
CONFIDENCE=0.56 the oil signal only ever arms when RSI is far from 50, making EMA spread
nearly irrelevant to gating. Likely unintended weighting. → Re-check intended confidence
curve vs the 0.56 arm threshold.

portfolio/kelly_metals.py:79-94: sqlite3 connection is not closed on the exception path
(connect succeeds, execute/fetch raises → `except` returns None without conn.close()).
Minor handle leak in a long-running loop. → Use `with contextlib.closing(conn)` /
try-finally.

## Risk summary

The dominant real-money risk is the grid fisher's EOD flatten sequence: it strips the
broker stop before confirming a replacement sell that may not fill at the illiquid close,
and the session-roll path can clear the in-flight-EOD-sell flag while inventory remains,
producing genuinely naked overnight leveraged exposure — the exact failure class the
subsystem invariants forbid. Secondary money-risk items are stop triggers (both grid
rotation and swing-trader) placed without any barrier-proximity validation, which on
near-barrier certs yield meaningless stops; everything else is robustness/accounting
(partial-fill qty corruption, fail-open order fetch) that degrades correctness rather
than guaranteeing loss.
