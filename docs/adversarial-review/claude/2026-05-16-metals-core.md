# Adversarial Review: Metals-Core Subsystem (2026-05-16)

Scope: data/metals_loop.py, portfolio/grid_fisher.py, grid_fisher_config.py, grid_tiers.py, exit_optimizer.py, price_targets.py, orb_predictor.py, fin_fish.py, fish_instrument_finder.py, fish_monitor_smart.py, metals_ladder.py, oil_grid_signal.py, warrant_portfolio.py, silver_precompute.py, gold_precompute.py, metals_precompute.py, metals_orderbook.py.

---

## [P1] Race condition: unprotected global state shared between fast-tick sub-loop and main loop
**File:** data/metals_loop.py:872,878,1087-1095,1437-1438,1468,1472-1493
**Bug:** Globals `_silver_fast_prices`, `_silver_alerted_levels`, `_silver_underlying_ref`, `POSITIONS` are accessed by both the embedded fast-tick sub-loop (10s cadence) and the main loop (60s) without locking. Fast-tick appends to the deque while main resets it; `alerted_levels` checked and added non-atomically.
**Why it matters:** Duplicate alerts fire (same level alerted twice). `POSITIONS` mutated mid-read produces stale entry prices, miscomputing P&L for orders. Worst case: triple-buy if fast-tick sees a stale "no entry" snapshot and races main.
**Fix:** Wrap all access in a single `threading.Lock`; or use a single-writer pattern where only the fast-tick reads from a snapshot the main loop publishes.

---

## [P1] Knockout-distance logic inverted for SHORT warrants
**File:** portfolio/grid_tiers.py:94-99
**Bug:** Distance is computed as `(barrier - underlying) / barrier`. For SHORT warrants, barrier lies above underlying; as underlying rises *toward* the barrier, the formula returns smaller positive values, but when underlying drops *away* from the barrier the value can become negative. The check `distance_pct < 8.0` is true for negatives, so SHORT tiers are skipped while they are actually furthest from knockout.
**Why it matters:** SHORT tiers systematically under-deploy in the safest part of the move. The Bull/Bear symmetry is broken: BULL works, BEAR under-positions when it should be aggressive.
**Fix:** Use `abs(implied - barrier) / barrier` for both directions, or branch on `direction` and compute `(underlying - barrier) / barrier` for SHORT.

---

## [P1] Global cap bypass via hardcoded fallback when buying-power fetch fails
**File:** portfolio/grid_fisher.py:795-801, ~L1300+
**Bug:** When `fetch_buying_power()` fails repeatedly, the cache eventually ages past 300s. The fallback substitutes a hardcoded `GRID_GLOBAL_MAX_SEK` (6500) rather than refusing to operate. After a manual top-up the operator expects more headroom, but the hardcoded value silently caps below the new buying power.
**Why it matters:** When the API recovers and the operator has deposited more cash, the grid keeps under-utilizing. Conversely, if buying power falls but cache stays stale-fresh, the grid can over-deploy.
**Fix:** Store `(value, fetch_ts)` separately. If `fetch_ts` is older than max-stale, return 0 (fail-closed) — never substitute a hardcoded constant.

---

## [P1] exit_optimizer crashes / returns NaN on past session_end
**File:** portfolio/exit_optimizer.py:464-550 (esp L506-510)
**Bug:** If `session_end < current_time`, `remaining_minutes` is negative. At L215, `int(remaining_minutes / dt)` yields a negative `n_steps`. Path simulation produces NaN; `CandidateExit.fill_prob` is NaN, unchecked by caller.
**Why it matters:** Exit ladder caller receives malformed candidate; downstream `max(..., key=fill_prob)` orders by NaN and behavior is undefined. Worst-case: agent silently doesn't exit.
**Fix:** Guard at entry: `if session_end <= market.asof_ts: return _immediate_market_exit(...)`.

---

## [P1] ORB morning-window UTC bounds computed once at __init__ — DST stale
**File:** portfolio/orb_predictor.py:35-46, 126-130
**Bug:** `_morning_window_utc()` is invoked from `__init__` and cached as instance state. After a DST transition, the window is 1 hour off in UTC but the cached value persists for the life of the loop process.
**Why it matters:** ORB prediction reads the wrong candles on the day(s) after the transition. Predictions degrade silently for weeks until process restart.
**Fix:** Re-evaluate `_morning_window_utc()` on every `calculate_morning_range()` call, or invalidate on date change.

---

## [P2] Velocity-alert key rollover allows double-fire on boundary
**File:** data/metals_loop.py:1488-1502
**Bug:** `vel_key = f"vel_{int((time.time() - 2) // 300)}"`. The `-2` second offset still permits two alerts to fire within ~1-2 seconds when `time.time()` straddles a 5-min boundary.
**Why it matters:** Duplicate Telegram notifications for the same velocity event.
**Fix:** Use a TTL dict keyed by the alert *content*; or `vel_key = f"vel_{int(time.time() // 300)}"` with no offset and tighten dedupe window.

---

## [P2] Oil RSI(14) off-by-one warm-up
**File:** portfolio/oil_grid_signal.py:50-62
**Bug:** Returns RSI as soon as `len(series) >= period + 1`. Wilder's RSI needs ~`2*period` bars to stabilize; at 15 bars the result has high variance.
**Why it matters:** Early-session signal is unstable; first three bars give a noisy buy/sell, leading to bad fills at open.
**Fix:** Require `len(series) >= 2 * period` or return neutral 50.0 below that threshold.

---

## [P2] EOD market-sell duplicate guard incomplete on failure
**File:** portfolio/grid_fisher.py:1545-1552, 1607
**Bug:** `eod_sell_order_id` is only stored on a successful place. If `place_sell_order()` raises (Avanza halt), the id remains `None`; next tick the same EOD sweep retries and posts a second sell.
**Why it matters:** Inventory goes net short. The grid is not configured for short exposure.
**Fix:** Set `eod_sell_order_id = "ATTEMPTED"` *before* the network call, then update with the real id on success; only clear on confirmed failure (HTTP 4xx).

---

## [P2] Warrant leverage drag ignores financing/decay cost
**File:** portfolio/price_targets.py:327-330
**Bug:** `warrant_move = pct_move * leverage` is used as the exit price scaling. Avanza certs include daily financing/decay (~0.1-0.3%/hour intraday for 5x).
**Why it matters:** Realized P&L on intraday holds is overstated by ~50 bps per 1200 SEK leg over a 6h hold.
**Fix:** Multiply expected exit by `exp(-daily_decay_rate * hours_held / 24)` (or equivalent linear approximation).

---

## [P2] Fishing stop ignores barrier buffer
**File:** portfolio/fin_fish.py:197-199
**Bug:** Stop placed at fixed % from entry. On a normal -2.5% underlying move, stop triggers though underlying is still 8%+ from barrier.
**Why it matters:** Trades exit on routine pullback instead of waiting for bounce; system gives up edge it has on multi-day swings.
**Fix:** Require `stop_distance_pct` such that the underlying at stop trigger remains >= 5% from the barrier.

---

## [P2] DST fold ambiguity in `minutes_until_eod()`
**File:** data/metals_loop.py:262-281, portfolio/grid_fisher.py:269-295
**Bug:** When called during the "fold" hour on autumn DST, `astimezone()` may pick the wrong side; cutoff drifts by 1 hour.
**Why it matters:** EOD sweep fires twice or not at all once a year.
**Fix:** After computing local time, inspect `local.fold` and pick the explicit branch.

---

## [P3] BEAR cert leverage sign is conventional but unflagged
**File:** portfolio/fin_fish.py:104-118
**Bug:** Catalog stores `leverage: 5.0` (always positive); `direction: "SHORT"` carries the sign semantically.
**Why it matters:** Future maintainer might assume BEAR `leverage` should be `-5.0` and double-negate.
**Fix:** Add a comment at the catalog write site clarifying that `leverage` is always positive.

---

## SUMMARY
P1=5 P2=6 P3=1
