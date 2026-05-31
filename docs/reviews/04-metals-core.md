# Metals-Core Review

Adversarial review of the metals-core subsystem (live Avanza order placement for
silver/gold/oil leveraged certs + warrants). Read from isolated worktree
`Q:/fa-rev-0531`. Scope: `data/metals_loop.py`, `data/metals_swing_trader.py`,
`data/metals_risk.py`, `data/metals_execution_engine.py`,
`portfolio/grid_fisher.py`, `portfolio/oil_grid_signal.py`,
`portfolio/exit_optimizer.py`, `portfolio/fin_snipe.py`,
`portfolio/fin_snipe_manager.py`, `portfolio/iskbets.py`,
`portfolio/orb_predictor.py`. Supporting modules read for invariant
verification: `portfolio/avanza_control.py`, `portfolio/avanza_session.py`,
`data/metals_avanza_helpers.py`, `portfolio/grid_fisher_config.py`,
`portfolio/grid_tiers.py`, `data/metals_swing_config.py`,
`portfolio/fin_snipe_config.py`.

## Invariant verification summary (the good news first)

These critical invariants were checked and are HELD:

- **Stop-loss API**: every hardware stop in the metals path goes through
  `/_api/trading/stoploss/new` — verified in `avanza_session.place_stop_loss:806`,
  `metals_avanza_helpers.place_stop_loss:386`,
  `avanza_control.place_stop_loss`/`place_stop_loss_no_page`. The two raw
  `trading-critical/rest/order/new` call sites in `metals_loop.py`
  (`emergency_sell:3782`, `place_spike_orders:5324`) are SELL orders (emergency
  liquidation + profit-taking limit), NOT stops — correct usage. No stop is
  placed via the regular order API anywhere. The Mar-3 wrong-API class of bug
  is absent.
- **Grid global cap (6500 SEK) / per-leg (1200 SEK)**: enforced per-tier inside
  the placement loop, recomputed live (`grid_fisher.place_buy_ladder:1411-1425`,
  tick gate `:1863-1873`), not only at startup. Fail-closes to a 0 cap when live
  buying power can't be read (`_effective_global_cap:1006-1012`). `tick()` is
  called only from the 60s loop (`metals_loop.py:7587`), never from the 10s
  fast-tick, so no intra-process concurrent-fill cap breach.
- **3% stop-distance floor**: enforced for new stops in the cascade path
  (`metals_loop.place_stop_loss_orders:4903`, `_update_stop_orders_for:2460`)
  and in fin_snipe (`fin_snipe_manager._compute_stop_plan:545`,
  `MIN_STOP_DISTANCE_PCT=3.0`).
- **Fast-tick (10s) concurrency**: the embedded fast-tick is alert/candidate-only
  (`_silver_fast_tick`, `_entry_fast_tick`) — it places NO orders, and runs
  single-threaded inside `_sleep_for_cycle` (same thread as the 60s cycle), so it
  does not mutate position/state files concurrently with order logic.
- **iskbets / orb_predictor / exit_optimizer / oil_grid_signal / fin_snipe.py**:
  no live order placement (advisory/signal/manual-command only). oil_grid_signal
  fails to `direction=None` on fetch error (grid idles) — correct fail-safe.

---

## P0 findings

### P0-1 — Grid fisher reconciles against ALL accounts; pension holdings create phantom inventory → real SELL of unheld units + EOD short-sell
`portfolio/avanza_session.py:681` (`get_positions()` — no account filter) ·
`portfolio/grid_fisher.py:678` (`_position_volume_for` — first orderbook match,
no account filter) · wired at `metals_loop.py:7069-7073` (session=`avanza_session`,
which returns every account's positions).

**Causal chain.** `GridFisher.tick()` calls `self.session.get_positions()`
(`grid_fisher.py:1661`). `avanza_session.get_positions()` iterates
`data["withOrderbook"]` across **every** Avanza account (ISK 1625505 AND pension
2674244) and returns them merged; it records `account_id` per row but never
filters. `reconcile_against_live` then matches purely by orderbook id via
`_position_volume_for(positions, ob_id)`, which returns the FIRST orderbook
match regardless of account. If the user holds the same orderbook id in the
pension account:

1. `live_vol = pension_vol` while `inst.inventory_units = 0` in the ISK-only
   grid. For any ARMED buy tier whose order id is absent from `open_orders`
   (a normal mid-cycle cancel or a placement timeout), `delta = pension_vol -
   0 ≥ tier.qty` → `record_fill(...)` marks the tier FILLED at a price never
   paid → `rotate_on_buy_fill` places a **real SELL** (`place_sell_order`,
   `grid_fisher.py:1506`) for units the ISK account does not hold.
2. `eod_market_flat` (`grid_fisher.py:1913-1963`) reads `inst.inventory_units > 0`
   and places a full-inventory market-equivalent SELL of phantom units →
   **short-sell** / rejected order churn at the close auction.
3. Phantom planned notional inflates `hit_per_instrument_cap()` and the global
   cap, starving legitimate placement.

This directly contradicts `memory/feedback_isk_only.md` ("only ISK 1625505;
ignore pension 2674244"). The startup `verify_default_account()` gate
(`metals_loop.py:7042`) confirms the *default* account is the right ISK but does
NOT make `get_positions()` filter — the read path is still account-blind. This
is the same finding flagged in `docs/adversarial-review-2026-05-25/04-metals-core.md`
P0; it remains unfixed in this worktree.

→ **Fix.** Add an `account_id` parameter to `avanza_session.get_positions()`
that filters rows by `account.get("id")`, and pass `self.account_id` from the
grid_fisher reconcile call site (`grid_fisher.py:1661-1663`). Mirror the same
filter in `_position_volume_for` (match on both orderbook id AND account id).
Until fixed, the grid must not run on any account whose tradable orderbook ids
can also appear in a second account.

---

## P1 findings

### P1-1 — Swing trader sells BEFORE cancelling its hardware stop → broker rejects with `short.sell.not.allowed`, exit silently fails
`data/metals_swing_trader.py:3210` (SELL placed) vs `:3238-3254` (stop cancelled
after).

**Causal chain.** `_execute_sell` calls `place_order(... "SELL" ... units)` for
the FULL position size at line 3210, and only AFTER a successful fill cancels the
hardware stop (lines 3238-3254). The position carries a hardware stop covering the
same full `units` (set in `_set_stop_loss:2776`). The codebase's own authoritative
rule — `metals_loop._ensure_stops_cancelled_before_sell:4024-4030` — states Avanza
rejects a sell when `active_stop_loss_volume + sell_volume > position_size`. Here
that sum is 200% of the position, so the SELL is liable to be rejected with
`short.sell.not.allowed`. On rejection `success=False`, the code sets
`sell_failed_at` and returns (3212-3228); next cycle the same exit reason fires
and the same rejection repeats — the position **cannot exit** while its stop is
live (stop-loss, take-profit, time-limit, signal-reversal exits all route through
`_execute_sell`). This is inconsistent with `emergency_sell` and
`place_spike_orders`, which both cancel stops first via
`_ensure_stops_cancelled_before_sell`. Not a direct cash loss, but a
failure-to-honor-exit on a leveraged position is a P1 (the intended stop/TP
becomes a no-op; the position rides until the *hardware* stop independently
triggers, which may be far from the software exit level).

→ **Fix.** In `_execute_sell`, cancel the hardware stop(s) (and any
adopted/cascade stops) BEFORE placing the SELL — reuse the
`_ensure_stops_cancelled_before_sell` pattern (snapshot → cancel+poll → sell →
re-arm on failure). Do NOT place the sell until the encumbered volume is
confirmed released.

### P1-2 — Hardware trailing-stop placement failure at buy fill leaves the new position with NO broker stop and NO retry
`data/metals_loop.py:4802-4814` (`_handle_buy_fill`, hardware-trailing branch).

**Causal chain.** Production config is `STOP_ORDER_ENABLED=False` +
`HARDWARE_TRAILING_ENABLED=True` (lines 421/447), so the sole broker-side
protection for a queue/trade-queue fill is the single
`place_trailing_stop_no_page` call at line 4788. If it returns `ok=False`
(line 4802) or raises (line 4806), the code logs + sends a Telegram WARNING but
takes no corrective action: the position stays open, `hw_trailing_stop_id` is
never set, and there is **no retry loop** anywhere that re-attempts the stop on a
later cycle (unlike grid_fisher, which has `stop_needs_rearm` + a per-tick
retry). The position is then naked at the broker, protected only by the loop's
own software `emergency_sell` / `check_momentum_exit` — which require the Python
process to be alive, the Playwright session valid, and the price fetch working.
A transient stop-API hiccup at the wrong moment = an unprotected leveraged
position for the remainder of the hold.

→ **Fix.** On hardware-trailing-stop failure, set a `stop_needs_rearm` flag on
the position and re-attempt placement every cycle until it succeeds (mirror
`grid_fisher.rotate_on_buy_fill:1573` + the tick rearm block at `:1718-1737`).
Also write a `critical_errors.jsonl` entry so the fix-agent/operator is paged,
as grid_fisher does at `:1582`.

### P1-3 — grid_fisher rotation cancels the old stop before the new stop is confirmed → naked window on rearm failure
`portfolio/grid_fisher.py:1538-1543` (cancel old stop) vs `:1546-1574` (place new
stop; on failure keeps a stale `stop_loss_id` that points at the now-cancelled
stop).

**Causal chain.** In `rotate_on_buy_fill`, the existing stop is cancelled at the
broker (1538-1543) before the replacement stop is placed (1548-1553). If the new
placement fails, `new_stop_id` is None → `stop_needs_rearm=True` and
`stop_loss_id` is left pointing at the **already-cancelled** old stop (the comment
at 1576 says "keeping old stop_loss_id to avoid naked position", but that id no
longer corresponds to a live broker stop). The inventory is genuinely naked from
the cancel until a subsequent tick's rearm block (`:1718-1737`) succeeds — at the
60s cadence that is a multi-cycle exposure window on a leveraged cert during the
exact volatility that just filled the buy. This is acknowledged in-code as
"FGL P0-3" and partially mitigated by the rearm retry + critical-errors logging,
so it is downgraded to P1, but the cancel-before-place ordering is still the
wrong sequence.

→ **Fix.** Place the new stop FIRST, confirm success, then cancel the old stop —
or, if the API forbids two coexisting full-volume stops, hold the cancel until
the replacement is confirmed and skip the cancel entirely on rearm failure (do
not cancel a protective stop you cannot replace).

---

## P2 findings

### P2-1 — Cascade trailing-stop update cancels stops then skips re-placement when within 3% of bid (DORMANT under current config)
`data/metals_loop.py:2449-2463` (`_update_stop_orders_for`) and
`:5148-5168` (`update_trailing_stops` → `place_stop_loss_orders`, whose 3% skip
at `:4903-4906` can `continue` past a just-cancelled position).

Both functions cancel the existing stop orders and then, if the new ratcheted-up
stop would land within 3% of current bid, early-return / `continue` WITHOUT
re-placing — leaving the position naked until the next trailing update. The
trigger is narrow (bid dropped sharply between the ratchet decision and the
placement check) but real. **Dormant** because the entire cascade path is gated
by `STOP_ORDER_ENABLED=False` (line 421); production uses hardware trailing
stops. Becomes live the moment an operator flips `STOP_ORDER_ENABLED=True`.

→ **Fix.** Do not cancel the existing stop until the replacement passes the 3%
check and is confirmed placed. If the new stop is too close, keep the old one.

### P2-2 — Swing-trader `_set_stop_loss` has no minimum-distance / barrier guard
`data/metals_swing_trader.py:2714-2786`.

The stop is computed as `entry_price * (1 - sl_warrant_pct/100)` with
`sl_warrant_pct = SL_BASE_UNDERLYING_PCT(6.0) * leverage`. At the user's standard
5x that is a 30% warrant stop — safely far. But there is no explicit clamp: a
low-leverage instrument (≤0.5x) would yield a ≤3% warrant stop that could sit
within the 3% volatility floor, and there is no MINI-barrier-proximity check at
all in this path. The other order paths (cascade, fin_snipe) enforce a 3% floor;
this one relies entirely on leverage being high enough.

→ **Fix.** Add a `max(computed_stop, bid*(1-MIN_STOP_DISTANCE_PCT/100))` clamp
and a barrier-proximity guard (for MINI instruments) in `_set_stop_loss`,
matching `fin_snipe_manager._compute_stop_plan`.

### P2-3 — Swing trader has no EOD-flat (positions held overnight on 8-day stop)
`data/metals_swing_config.py:323` (`EOD_EXIT_MINUTES_BEFORE = 0`),
`metals_swing_trader.py:3073`.

The swing-trader EOD blind-exit is disabled by user override (documented), so
swing positions are NOT force-flattened at the 21:55 close — they ride overnight
protected only by the hardware stop (valid 8 days). This is an intentional config
choice, flagged only because it contradicts the "EOD-flat must not leave dangling
positions" premise: for the swing trader, dangling overnight positions are
**expected**. The grid_fisher DOES EOD-flat (`eod_market_flat`). No fix required
unless the overnight-hold policy changes; noted so the divergence between the two
subsystems is on the record.

---

## P3 / observations

- `place_spike_orders` and the grid use `bid*0.99` / `target_price` limit orders
  that can sit unfilled on an illiquid close; both have in-flight dedup guards
  (`eod_sell_order_id`, spike snapshot persistence) so they don't stack — OK.
- `metals_execution_engine.py` and `metals_risk.py` are recommendation/risk-
  reporting only (no order placement); sizing is capped at buying power
  (`_planned_units:266-290`) and all state writes use `file_utils` atomic I/O.
- Order-result checking is generally rigorous: HTTP 200 alone is never trusted —
  callers require `orderRequestStatus == "SUCCESS"` + a real `orderId`
  (`place_spike_orders:1349`, `metals_avanza_helpers.place_order:316`,
  `grid_fisher.place_buy_ladder:1456`). This closes the "silent HTTP-200-with-
  error-body" class.
- Session-expiry: `_check_session_and_alert` exists (`metals_loop.py:4391`) and
  grid/session calls fail-closed via `_safe_session_call` returning None
  (distinguished from empty), so a dead session skips placement rather than
  firing into the void.
