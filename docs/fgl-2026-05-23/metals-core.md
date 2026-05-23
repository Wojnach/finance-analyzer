# METALS-CORE Adversarial Review — 2026-05-23

Empty-baseline review. Scope: `data/metals_loop.py` (7880 lines), `portfolio/fin_snipe*.py`,
`portfolio/fin_fish.py`, `portfolio/fin_evolve.py`, `portfolio/fish_*`, `portfolio/metals_*`,
`portfolio/grid_fisher*.py`, `portfolio/grid_tiers.py`, `portfolio/oil_grid_signal.py`,
`portfolio/orb_*.py`, `portfolio/iskbets.py`.

Confidence ≥ 80 only. Project critical rules:
- NEVER place stop-loss within 3% of current bid (memory `feedback_mini_stoploss.md`).
- Stop-loss API: `/_api/trading/stoploss/new` only (NOT regular order API — Mar 3 instant-fill incident).
- Avanza commodity warrant hours: 08:15-21:55 CET, not 17:25 (memory `market_hours.md`).
- `todayClosingTime` must be read from API — do NOT hardcode 21:55 (DST-aware).
- Atomic I/O only via `file_utils.load_json` / `atomic_write_json` / `atomic_append_jsonl`.

---

## P0 — Critical, fix before next trading session

### P0-1 — `grid_fisher.rotate_on_buy_fill` can leave position naked after stop-rearm failure (conf 95)

`portfolio/grid_fisher.py:1424-1522` (`rotate_on_buy_fill`).

When a previously-filled buy tier rotates and the instrument already has
`inst.stop_loss_id` set, the code (1) places the new sell limit, (2)
cancels the OLD stop, then (3) attempts to place the NEW stop sized to
the *full current inventory*. If step (3) fails — `_safe_session_call`
returns `None` (timeout, exception, broker rejection, transient auth
glitch) — the assignment at line 1515 unconditionally writes
`inst.stop_loss_id = new_stop_id` where `new_stop_id` is `None`. The
position now has *increased* inventory (the new fill added units), the
old stop is gone, and no replacement stop exists at the broker. Naked
until the next tick at best, and forever if the loop crashes.

Worse: the next tick's `place_buy_ladder` doesn't re-arm a stop — it
only adds buy tiers. The stop is only revisited on the *next* fill via
`rotate_on_buy_fill` again, which will face the same conditions.

Fix: when `new_stop_id` is None, do NOT overwrite `inst.stop_loss_id`
— keep the previous value (or set a `stop_needs_rearm` flag and have
`tick()` re-attempt before the next buy ladder placement). Also log a
critical-error entry to `data/critical_errors.jsonl` so the auto-spawn
fix agent picks it up.

### P0-2 — `fin_snipe_manager` stop-distance gate is 1%, not 3% — direct memory violation (conf 95)

`portfolio/fin_snipe_manager.py:64` (`MIN_STOP_DISTANCE_PCT = 1.0`),
`fin_snipe_manager.py:529-563` (`_compute_stop_plan`).

Memory rule `feedback_mini_stoploss.md` (echoed in `.claude/rules/metals-avanza.md`):
"NEVER place a stop-loss within 3% of current bid. Silver warrants are
volatile." `_compute_stop_plan` rejects stops only when `distance_pct <
1.0` — and only for *new* stops (existing-stop branch keeps the close
stop regardless). The cascading `place_stop_loss_orders` in
`data/metals_loop.py:4903` and `_update_stop_orders_for` at line 2460
both check `distance_pct < 3.0`. fin_snipe_manager is the outlier and
will happily place stops 1.5-2.9% below the bid, well inside the
intraday volatility band on 5x leveraged metals certs.

Concrete failure: position averaged at 75 SEK, current bid 75.5 (up
0.7%). `trigger_price = 75 × 0.95 = 71.25`. `distance_pct = (75.5 -
71.25) / 75.5 × 100 = 5.6%`. Safe. But position averaged at 75 with
current bid 73.5 (down 2%): `distance_pct = (73.5 - 71.25) / 73.5 = 3.06%`
— passes the 1% gate but is right at the memory limit. Bid 73.0:
`distance_pct = 2.4%` — passes the 1% gate but violates the 3% rule. A
single noise tick triggers the stop instantly at 70.55 sell price, then
the broker dumps into the bid for ~9% slippage on a 5x cert.

Fix: change `MIN_STOP_DISTANCE_PCT` to `3.0` and apply the same gate to
the keep-existing branch (line 555-563) so a stop that becomes too
close is cancelled, not preserved.

### P0-3 — Hardcoded EOD 21:55 ignores `todayClosingTime` and DST drift (conf 90)

`portfolio/grid_fisher.py:277-279`:
```
_EOD_LOCAL_HOUR = 21
_EOD_LOCAL_MINUTE = 55
_EOD_TZ_NAME = "Europe/Stockholm"
```

`data/metals_loop.py:1574` (`is_market_hours`) hardcodes `8.25 <= h <= 21.92`
(08:15-21:55) and the FISHING_EOD_SELL_MINUTE_CET tuple at line 2032 is
`(21, 50)`. The metals-avanza ruleset (`.claude/rules/metals-avanza.md`)
explicitly requires reading `todayClosingTime` from the API. DST shifts
in late March/late October change Avanza warrant close times by an
hour; the hardcoded constant will fire EOD an hour late (positions held
through close) or an hour early (premature liquidation of profitable
positions). Confirmed historical pattern in `docs/SESSION_PROGRESS.md`
references for DST transitions.

Fix: query `/_api/market-guide/<inst>/<ob_id>` once per session for
`todayClosingTime` and persist it into the grid_fisher state. Fall back
to the hardcoded constant only when the API call fails.

### P0-4 — Loop crash inside EOD window (21:50-21:55) leaves grid_fisher inventory unsold and unsweeped (conf 85)

`data/metals_loop.py:7585-7591` (grid_fisher.tick wrapped in catch-all)
and `data/metals_loop.py:7850-7872` (`except Exception` falls through
to `_kill_claude()` + `release_singleton_lock()` without running EOD
sweep).

If the loop crashes (network blip, OS scheduler kill, OOM) anywhere
between 21:50 (`FISHING_EOD_SELL_MINUTE_CET`) and 21:55 (warrant
close), there is no graceful EOD handler. grid_fisher inventory
inherits ONLY the per-tier sell-limit + stop placed by
`rotate_on_buy_fill` — but the `eod_market_flat` "force-flat by
bid×0.99 limit" never runs, so any inventory that the per-tier sell
target wouldn't have reached will stay live overnight, breaking the
"EOD-flat" guarantee in CLAUDE.md ("EOD-flat" / grid_fisher description).
Worst case for legacy `POSITIONS`: `_eod_sell_fishing_positions` (line
2052) only runs from inside the main loop and never via a final
`finally:` block, so the fishing-EOD-sell is also skipped on a crash
after 21:50.

Also a P0 unique to grid_fisher: `eod_market_flat` runs only when
`eod_minutes_remaining <= GRID_EOD_MARKET_SELL_MINUTES_BEFORE` (5).
If the main loop is mid-`grid_fisher.tick` exactly at 21:50 and the
*previous* tick was at 21:49.7, the per-tick window can skip the
"<=5" check entirely (a single 60s cycle straddling 21:50 means
one tick ran at remaining=5.2, the next at remaining=4.2 — `<=5`
fires only on the second). That's fine on a healthy loop, but adds
latency to any subsequent crash.

Fix: register `atexit` / signal handler that calls
`grid_fisher.eod_market_flat()` + `_eod_sell_fishing_positions()`
during the EOD window. Persist a "EOD pending sweep" flag in
`grid_fisher_state.json` so the next process boot picks it up.

### P0-5 — `_handle_buy_fill` legacy fallback places stop via `place_stop_loss` with NO 3% gate (conf 80)

`data/metals_loop.py:4816-4827`.

The legacy cascade-stop branch runs when `STOP_ORDER_ENABLED=True and
not HARDWARE_TRAILING_ENABLED` — both defaults make this dead path
right now, but a single config flip re-enables it. Inside the branch
`place_stop_loss(page, ACCOUNT_ID, order["ob_id"], stop_trigger,
stop_sell, vol)` is called with `stop_trigger` straight from the
trade-queue order dict, no bid-distance check. If Layer 2 computed
`stop_trigger` at entry × 0.97 but the price gapped down to 0.98 between
queue and fill, the stop sits 1% below bid and trips on the first wick.
The two sibling paths (`place_stop_loss_orders` and
`_update_stop_orders_for`) both guard this with `distance_pct < 3.0`;
this branch should too.

Fix: add the 3% gate before line 4822, or refactor through a shared
`_place_stop_with_safety_gate(...)` helper that all three paths use.

---

## P1 — Important, fix this week

### P1-1 — `KNOWN_WARRANT_OB_IDS` extension uses raw `open() + json.load()` instead of `load_json` (conf 90)

`data/metals_loop.py:1978-1996`. Project rule #4: "Atomic I/O only.
Use `file_utils.atomic_write_json()`, `load_json()`,
`atomic_append_jsonl()`. Never raw `json.loads(open(...).read())`."

The fallback path swallows the exception so the loop doesn't crash, but
it silently produces an empty `_dyn_warrants` dict. Net effect: when
`metals_warrant_refresh.py` is mid-rewrite of
`data/metals_warrant_catalog.json` (atomic rename window), startup
loses the 100+ swing-managed warrants and `detect_holdings` logs them
as "unknown ob_id" until next process restart. Same bug class as the
TOCTOU issues called out elsewhere in the repo.

Fix: replace with `load_json("data/metals_warrant_catalog.json", default={})`.

### P1-2 — `grid_fisher.eod_market_flat` uses bid × 0.99 limit, only 1% slippage budget (conf 85)

`portfolio/grid_fisher.py:1832-1873`.

The EOD force-flat places a limit at `max(bid * 0.99, 0.01)`. On a 5x
leveraged BULL/BEAR cert at 21:55 CET in illiquid conditions (Friday
US close, low-cap warrant), the bid can be stale by tens of seconds
and the true tradable bid 2-3% below the displayed bid. The 1% buffer
isn't aggressive enough to guarantee a fill — `cancel_spike_orders`
uses 1% too in line 5444, but `emergency_sell` uses raw `bid` (zero
buffer). Inconsistent and likely to leak positions overnight on
illiquid days.

Also: when `quote` fetch fails (line 1834) the fallback price is
`inst.avg_entry_price` — for a position deep in the red this is far
ABOVE the real bid, the limit never fills, and the position carries
over with no broker-side stop.

Fix: use bid × 0.97 (3%) for the EOD force-flat limit, AND fall back
to bid × 0.95 (not avg_entry_price) when the live quote fetch fails.
Also: if the previous EOD sell didn't fill within 2 min, cancel and
re-place at bid × 0.95.

### P1-3 — `grid_fisher` per-instrument cap not enforced when caller bypasses `account_id` (conf 85)

`portfolio/grid_fisher.py:939-982` (`_effective_global_cap`).

When `account_id is None` (constructor default), `_effective_global_cap`
returns `(GRID_GLOBAL_MAX_SEK, ...)` — i.e. the bypass path. The
metals_loop integration must pass `account_id=ACCOUNT_ID` for the
live-buying-power clamp to work. Grep for "account_id" through the
metals_loop's `GridFisher(...)` construction:

```
data/metals_loop.py: grid_fisher = GridFisher(session=..., ...)
```

The construction site needs to be audited to confirm `account_id`
is actually passed. If it's missing, the global cap relies *solely*
on the hardcoded `GRID_GLOBAL_MAX_SEK = 6500` constant and never
clamps against the actual live cash balance — the exact failure
mode `GRID_CASH_SAFETY_BUFFER_SEK` was added to prevent
(2026-05-13 OLJAB-on-empty-cash incident, per the config comment).

Fix: verify the `GridFisher(...)` call in metals_loop passes
`account_id=ACCOUNT_ID`. Add a startup assertion in
`GridFisher.__init__` that warns loudly when `account_id is None`
and `GRID_FISHER_PROBE_ONLY is False` — a real-money cycle with
no live-cash gate is unsafe.

### P1-4 — `grid_fisher.cancel_armed_buys` falls through on `cancel_failed`, leaves zombie tier ARMED forever (conf 80)

`portfolio/grid_fisher.py:1238-1280`.

When `cancel_order` returns `None` (session timeout / playwright
hiccup), the tier stays in `inst.buy_ladder` with `status=ORDER_ARMED`.
On the next tick, `reconcile_against_live` will check whether the
order_id is in `open_order_ids`. If the cancel actually went through
on the Avanza side but the response was lost (network blip after
broker accepted), the order_id WON'T be in open_orders and the
reconcile code will misinterpret it as "filled or cancelled":

- `delta = live_vol - inst.inventory_units` — if Avanza did cancel,
  `live_vol == inst.inventory_units`, `delta == 0`, so the tier is
  marked CANCELLED. Fine.
- But if Avanza cancelled AND a *different* buy filled in the same
  interval (e.g. a deeper tier), `delta >= tier.qty` and the code
  records the wrong tier as FILLED at the wrong price. Inventory
  accounting then drifts and the rotation places a sell at the wrong
  level.

This is a logic edge case but on a high-volatility session with both
tiers near the bid, it's reachable.

Fix: on `cancel_failed` with `result is None`, do NOT keep the tier
ARMED on the next tick — mark it `pending_cancel_retry` with a
counter, and after N retries mark `force_cancelled` so reconcile can
treat it as cancelled regardless of broker state.

### P1-5 — `grid_fisher` direction flip cancel before reset is best-effort, can leak orders (conf 80)

`portfolio/grid_fisher.py:1877-1891` (`arm_direction`).

When direction flips, `cancel_armed_buys` is called first, then
`flip_direction` resets the buy ladder to empty. If `cancel_armed_buys`
fails to actually cancel one of the orders at the broker (cancel
rejected, session call returned None — see P1-4), the in-memory
ladder is cleared but the order is still live on Avanza. Next tick
places fresh tiers at the new direction's instrument while the OLD
direction's orphan order may still fill from a passing bid, creating
inventory in the wrong-direction cert. Reconcile won't catch this
because the order_id is no longer tracked in any tier.

Fix: `flip_direction` should only clear the ladder for tiers that
were successfully cancelled. Failed-cancel tiers should be moved to
a `pending_cleanup` list that `tick()` reconciles against the broker
on every cycle until they reach a terminal state.

### P1-6 — `_fish_engine_execute_buy` uses ask price as limit, no bid-distance gate (conf 80)

`data/metals_loop.py:3040`: `place_order(_loop_page, ACCOUNT_ID,
ob_id, "BUY", ask, volume)`.

Fishing buys are *contrarian* dip-buys, but the order is placed AT the
ask — that's a market-equivalent immediate fill. There's no
verification that `ask` is reasonable vs the bid (spread sanity), no
slippage gate, no max-budget enforcement beyond Kelly. If the
orderbook is wide (1.5% bid/ask spread, common on illiquid warrants),
fishing pays the entire spread on entry, immediately starting -1.5%
PnL. Memory `fishing_system.md` notes the fishing system is supposed
to use limit orders at computed dip levels — paying the ask defeats
the entire premise.

Fix: place the buy at `bid * 1.001` (penny over bid) or at the
computed fish level from `fin_fish.compute_fishing_plan`. Reject if
spread > 1.0%.

---

## P2 — Useful, when next touching this code

### P2-1 — `grid_fisher` rate-limiter persistence gap (conf 80)

`portfolio/grid_fisher.py:984-992`. `_recent_places` is a process-local
list, not persisted. A loop restart inside the burst window resets the
counter, allowing `2 × GRID_MAX_ORDERS_PER_MIN` placements in 1 min
across the restart boundary. With `GRID_MAX_ORDERS_PER_MIN=10`, that's
20 orders in <2 min after a crash — could flag the Avanza account.

Fix: persist `_recent_places` to grid_fisher_state on save_state, drop
items older than 60s on load.

### P2-2 — `grid_fisher` worker thread `ThreadPoolExecutor` lifecycle on GC (conf 80)

`portfolio/grid_fisher.py:1039-1048` (`__del__`). Relying on `__del__`
for the executor shutdown is brittle — CPython may not call it
deterministically on interpreter exit, and exception paths during
constructor can leak the executor. Use an explicit `close()` method
called from `finally:` in the metals_loop main(), or use a context
manager.

### P2-3 — `_compute_stop_plan` keep-existing branch doesn't surface "stop too close" risk (conf 80)

`portfolio/fin_snipe_manager.py:543-553`. Comment says "Hysteresis: if
we already have a managed stop, keep it regardless of distance." That
hides a real risk: when the position has rallied and the stop is now
0.5% below bid (very close), keeping it preserves the noise-triggering
risk the gate is supposed to prevent. A managed stop that's now too
close should be REPRICED (cancel + place new at -5% from current bid)
rather than kept.

This connects to P0-2: if `MIN_STOP_DISTANCE_PCT` is raised to 3.0,
the keep-existing branch should *also* cancel and reprice when the
stop drifts inside 3%.

### P2-4 — `eod_market_flat` does not re-attempt if quote fetch fails twice in a row (conf 80)

`portfolio/grid_fisher.py:1834-1842`. When `get_quote` fails, code falls
back to `inst.avg_entry_price`. If `_safe_session_call` returns None
for two cycles in a row, the EOD passes without flatten. There's a
hidden assumption that the session always recovers in time, but
nothing in the code asserts that. A 5-minute window (~5 ticks at 60s)
gives 5 chances; if Playwright session is dead, all 5 fail and the
position holds overnight unflattened.

Fix: if eod_market_flat can't fetch a fresh quote OR previous attempt
queued an order that didn't fill in 60s, escalate to bid × 0.95
(aggressive) or cancel the prior placement and re-place; finally send
Telegram CRITICAL alert so the operator can manually intervene.

---

## P3 — Nice-to-have / observations

### P3-1 — `_fetch_warrant_catalog_prices` writes barrier_distance_pct only when both und > 0 and barrier > 0 — silently None otherwise (conf 75 — borderline P3)

`data/metals_loop.py:4470-4476`. Catalog entries with `barrier=0`
(constant-leverage certs) get `barrier_distance_pct=None`. Downstream
consumers should explicitly handle None vs 0; minor display issue
otherwise.

### P3-2 — `iskbets.py` is notification-only — no live trading exposure (informational)

Confirmed: no `place_order`, `api_post`, `trading-critical`, or
`/_api/trading/...` references. The skill sends Telegram alerts and
the user trades manually. iskbets does NOT pose direct knockout/auto-
execution risk via this code path. The risk is purely user-side trade
discipline.

### P3-3 — `oil_grid_signal.py` uses `atomic_write_json` correctly; cache TTL 5min appropriate (informational)

No issues.

### P3-4 — `metals_ladder.py`, `metals_orderbook.py`, `metals_cross_assets.py`, `orb_*.py`, `fish_monitor_smart.py`, `fish_instrument_finder.py`, `fin_fish.py`, `fin_evolve.py` are read-only / planner code (informational)

None of these place live orders. Safe.

---

## 5-Line Summary

5 P0 findings: rotate_on_buy_fill can leave naked position when stop-rearm fails (grid_fisher:1515); fin_snipe_manager places stops 1% from bid violating the 3% memory rule (fin_snipe_manager:64); hardcoded EOD 21:55 ignores `todayClosingTime` + DST (grid_fisher:277); loop crash inside 21:50-21:55 window skips EOD sweep entirely (metals_loop:7850); legacy `_handle_buy_fill` cascade path has no 3% bid-distance gate when STOP_ORDER_ENABLED is flipped (metals_loop:4822). 6 P1 issues include raw `open()` reading metals_warrant_catalog.json, eod_market_flat using bid×0.99 with avg_entry_price fallback that won't fill on deep losers, missing `account_id` audit on GridFisher construction, cancel_failed leaving zombie ARMED tiers, direction-flip cancel-failure leak, and fish-engine executing buys at the ask. iskbets confirmed notification-only, no live execution. grid_fisher's global cap and Gate A/B silent-rejection back-off are well-designed; the stop-rearm gap and EOD-crash window are the highest residual risk to overnight inventory exposure.
