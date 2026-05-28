# Adversarial review — metals-core subsystem

Worktree: `Q:\fa-fgl\metals-core` (diff `fgl-baseline..HEAD`, all 10 files = additions).
Cross-referenced against live repo `Q:\finance-analyzer` for out-of-diff callees
(grid_tiers.py, avanza_control.py, metals_shared.py, avanza_session.py).

## Count summary

| Severity | Count |
|----------|-------|
| P0 (money loss / crash / corruption / wrong-direction) | 0 confirmed-live, 1 latent-but-flag-gated |
| P1 (wrong under realistic conditions) | 4 |
| P2 (latent) | 5 |
| P3 (minor) | 3 |

Net: the stop-loss API usage is correct everywhere (all stop paths use
`/_api/trading/stoploss/new`; only emergency/EOD market-equivalent sells use
the regular order API, which is the documented pattern). Stop *placement*
direction is correct everywhere (certs are always held long → SELL stop below
cert bid, which is right for BULL and BEAR alike). The real direction-blindness
lives in (a) the catalog barrier-distance computation and (b) the swing trader's
SHORT exit-trigger P&L math, the latter gated behind a default-off flag.

---

## P0 / latent-catastrophic

P0-1 (latent, flag-gated) — data/metals_swing_trader.py:2974-2977 — WRONG-DIRECTION:
SHORT warrant exit-trigger P&L is inverted. For a held BEAR/MINI_S cert you
BOUGHT (entry=ask) and SELL to exit, so realised P&L is always
`(current_bid/entry - 1)` regardless of LONG/SHORT — exactly what
`_execute_sell` computes at line 3169. But `_check_exits` computes
`warrant_pct_change = (entry - current_bid)/entry` for SHORT (the inversion).
Consequence once `SHORT_ENABLED=True`: a *winning* BEAR position (cert price
rising) reports a negative `warrant_pct_change`, which (a) never trips
TAKE_PROFIT (line 2984) and (b) trips HARD_STOP (line 3047
`warrant_pct_change <= -pos_sl_pct`) — i.e. it force-sells winners as if they
were catastrophic losses, and lets losers ride. This contradicts line 3169 in
the very same flow. Currently latent: `SHORT_ENABLED=False` and
`SHORT_CANARY_WARRANTS=frozenset()` (lines 161-162). It is P0-class the instant
the documented flag is flipped. fix: delete the SHORT branch at 2976-2977 —
warrant_pct_change for a held cert is `(current_bid - entry)/entry` for BOTH
directions. The underlying-side direction-awareness (und_change_pct, peak/trough
tracking, momentum/signal-reversal) is already correct and should stay; only the
*warrant-price* P&L must not be inverted.

---

## P1 (wrong under realistic conditions)

P1-1 — data/metals_loop.py:4474 — WRONG-DIRECTION barrier distance:
`entry["barrier_distance_pct"] = round((und - barrier) / und * 100, 1)` is
direction-blind. The live catalog holds 47 SHORT (MINI_S / BEAR) warrants whose
barrier sits ABOVE the underlying (e.g. barrier 98.26 / 80.97 while silver ~30).
For every SHORT warrant this yields a large NEGATIVE distance. That value feeds
`metals_execution_engine._summary_filters` (line 257-260), whose gate
`barrier_distance_pct < MIN_BARRIER_DISTANCE_PCT (15.0)` is then always true →
**every short-warrant BUY recommendation is silently filtered out** with a bogus
"barrier too close" reason, even when the short cert is perfectly safe. The
correct formula already exists in this codebase at
metals_swing_trader.py:2490-2498. fix: branch on the entry's `direction`
(catalog rows carry `direction` LONG/SHORT) — `LONG: (und-barrier)/und`,
`SHORT: (barrier-und)/und`.

P1-2 — data/metals_loop.py:6822-6880 / 7850-7880 — EXIT-0-ON-CRASH:
On a fatal exception the `while True` (inside the try at 7166) breaks to
`except Exception` (7852), logs FATAL + Telegram, runs `finally`, and `main()`
falls off the end returning `None` → `sys.exit(None)` = **exit code 0 on a
crash**. This is the exact failure class CLAUDE.md's STARTUP CHECK warns about
(the March-April auth outage: process exited 0 while broken, so nothing
restarted it). A supervisor keyed on exit code cannot distinguish crash from
clean stop. fix: have `main()` return a non-zero code on the
`except Exception` path (e.g. set `rc = 1` before the finally and
`return rc`), and reserve 0 only for KeyboardInterrupt / clean shutdown.

P1-3 — data/metals_loop.py:7166-7848 — NO PER-CYCLE ISOLATION (loop-reliability,
#1 priority): the entire 680-line `while True` cycle body is one try with the
only `except` OUTSIDE the loop (7852). Several top-level steps are NOT wrapped in
their own try (e.g. `fetch_underlying_from_binance()` 7176,
`_accumulate_orderbook_snapshots()` 7185, the holdings/stop block 7187-7215,
the EOD-fishing block 7217-7232). Any uncaught raise there kills the whole loop
(→ P1-2 exit-0) rather than skipping one cycle. Per CLAUDE.md "the loop must run
100% of the time". fix: wrap the per-cycle body in an inner
`try/except Exception` that logs + continues to the next cycle (keeping the outer
handler only for truly unrecoverable startup/teardown failures).

P1-4 — portfolio/grid_fisher.py:1871-1916 (eod_market_flat) — naked window at
EOD: the function sets `inst.stop_loss_id = None` (1879, after cancelling the
broker stop) and cancels all armed sell tiers BEFORE attempting the aggressive
market-equivalent sell. If that sell then fails (`result is None` 1897, or
rejected 1907 → `continue`) the position is left with NO stop and NO sell until
a future tick retries. If the loop dies in that window (or the close auction is
illiquid for the rest of the session) the position is naked overnight with the
stop already deleted. fix: place the replacement aggressive sell FIRST and only
cancel the existing stop after the sell is confirmed accepted; or re-arm the
stop on the failure branches the way `rotate_on_buy_fill` does (it sets
`stop_needs_rearm=True` + logs a critical naked-position entry; eod_market_flat
does neither).

---

## P2 (latent)

P2-1 — data/metals_loop.py:7224 — EOD fishing-sell window miss: the trigger fires
only when `_h_int == 21 and _m_int >= 50` (i.e. the 21:50-21:59 minute window).
If the loop is down, restarting, or a cycle overruns across that 10-min window,
the once-per-day `_eod_fishing_sold_today` guard is never set and the hour rolls
to 22 → the EOD sell is skipped and "intraday-only" fishing positions are held
overnight (hardware stops at validDays=8 are the only remaining protection). fix:
use a "past cutoff and not yet sold today" condition
(`(_h_raw*60) >= (21*60+50) and _eod_fishing_sold_today != today`) so a late or
restarted cycle still flattens.

P2-2 — portfolio/fin_snipe_manager.py:340-373 (_estimate_entry_underlying) —
direction-unaware back-calc: `underlying_return = instrument_return / leverage`
(366) assumes a positive correlation. For a BEAR cert (cert up when underlying
down) the recovered entry-underlying is reflected the wrong way unless
`leverage` carries a negative sign. Only matters when `entry_underlying` is
absent and must be reconstructed; affects exit-optimizer EV estimates, not the
stop itself. fix: thread the instrument direction sign into the divisor, mirroring
metals_execution_engine `_instrument_price_from_underlying` (line 162,
`direction_sign * leverage * und_return`).

P2-3 — data/metals_swing_trader.py:550-581 + cross-process — STATE_FILE
read-modify-write has no cross-process lock. `metals_swing_state.json` is loaded
once into `self.state` (682) and saved with atomic writes, but the load→mutate→
save sequence is not atomic; any other process that opens the same file (e.g. a
manual tool, dashboard write, or a second swing instance) can lose updates
(last-writer-wins). In-process this is safe (single-threaded loop; fast-ticks run
sequentially in the sleep, not a thread — verified no `threading.Thread` in
metals_loop). fix: if any out-of-process writer exists, guard STATE_FILE writes
with the same `avanza_order_lock`-style file lock used for orders, or a dedicated
state lock; otherwise document the single-writer invariant explicitly.

P2-4 — portfolio/grid_fisher.py:285-310 (minutes_until_eod) — EOD disabled on
missing tzdata: returns `float("inf")` when zoneinfo/tzdata is unavailable, so
`tick()` never enters the EOD sweep/flat branches (1691) and positions are never
auto-flattened at close. Fail-OPEN for EOD on a Windows host where tzdata can be
absent. The grid relies on per-order `validUntil=today` to expire buys, but
inventory + sells + stops would persist. fix: fall back to a fixed UTC-offset
cutoff (CET/CEST with a DST guess) instead of inf, and emit a critical log so the
operator knows EOD handling is degraded.

P2-5 — data/metals_loop.py:6242 / 3736 — emergency drawdown acts on cached
POSITIONS: `check_portfolio_drawdown(POSITIONS, ...)` and the L3 path use the
in-memory POSITIONS units/entry, which can be stale if a broker stop already
fired between the 30s holdings reconcile windows. emergency_sell DOES re-verify
holdings on the short-sell-not-allowed branch (3842-3899), which is the main
guard, but the drawdown % itself can be computed off a position that no longer
exists, producing a spurious EMERGENCY trigger. fix: gate the drawdown breach on
a fresh positions fetch (or skip keys whose last reconcile is older than N
cycles).

---

## P3 (minor)

P3-1 — data/metals_loop.py:415 — comment/code mismatch: "Stop levels (distance
from barrier as % of bid)" but `check_triggers` (6164) computes
`dist_stop = (bid - pos["stop"])/bid` — distance from the STOP, not the barrier.
Misleading for the next reader. fix: correct the comment to "distance from stop".

P3-2 — portfolio/grid_fisher.py:1731-1737 — dead ADX trend filter: the
`if adx is not None and adx > self._adx_trend_filter:` block body is just `pass`
with a comment — it computes nothing and never gates. Either implement the
counter-trend skip or remove the no-op so it doesn't read as an active filter.

P3-3 — data/metals_avanza_helpers.py:358-359 — stop validUntil uses naive local
`datetime.now()` + N days for `validUntil`/`validDays`. Harmless for an 8-day
window but inconsistent with the UTC timestamps used elsewhere; on a host whose
local date differs from exchange date near midnight the broker could interpret
the date a day off. fix: anchor to the Stockholm exchange date explicitly.

---

## Verified-correct (no action)

- All stop placements (`metals_avanza_helpers.place_stop_loss`,
  `_update_stop_orders_for`, `place_stop_loss_orders`, swing `_set_stop_loss`,
  fin_snipe `_compute_stop_plan`, grid `rotate_on_buy_fill`) use the stop-loss
  API and trigger BELOW the held cert's bid — correct for BULL and BEAR (cert is
  always held long). 3% min-distance-from-bid guards are present
  (metals_loop 2460, fin_snipe 545, swing not-near-barrier via select).
- emergency_sell / EOD market-flat correctly use the regular order API (these are
  market-equivalent sells, not stops) and re-arm stops on every failure branch.
- grid_fisher global/per-instrument cap is single-tick-threaded (no concurrent
  tick); fills are polled, both legs were within the placed budget, so two fills
  cannot race past the cap. buying-power gate fails CLOSED (cap=0) when balance
  unfetchable.
- grid_fisher P&L (`record_fill`) and metals_execution_engine
  (`_instrument_price_from_underlying`, `_position_direction`,
  `fill_drift = direction_sign*leverage*drift`) are fully direction-aware.
- grid buy orders carry `validUntil=today` so they self-expire EOD even if the
  loop is down at close (orphaned-buy cleanup is broker-side).
