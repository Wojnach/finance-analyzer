# Claude adversarial review: metals-core

## Summary

Reviewed grid_fisher, fin_snipe_manager, mstr_loop, golddigger, elongir, oil_grid_signal, fin_fish, ORB predictor/backtest, metals_loop stop-loss paths, plus the crypto/oil swing loops and their precompute caches. Subsystem is the highest-risk surface (Mar-3 stop incident territory) and several Mar-3-class defects remain.

The single most dangerous finding: **`portfolio/mstr_loop/config.py:19` flips PHASE to "live" from a single env var with zero confirmation gate**, contradicting the documented 90-day shadow + manual approval requirement. Set `MSTR_LOOP_PHASE=live` and `_live_place_buy`/`_live_place_sell` route real Avanza orders on next cycle.

Second tier of concerns are stop-loss placements that ignore the warrant barrier (the original Mar-3 pattern persists in `fin_snipe_manager._compute_stop_plan` and `portfolio/grid_fisher.rotate_on_buy_fill`), cascading-stop placement that doesn't verify the prior cancel succeeded (volume-overfill risk), and an ORB backtest with intraday look-ahead bias that overstates simulated P&L. Several smaller issues around session windows, EOD timing, cache TTL across session boundaries, fee math, and tier ranking are itemised below.

## P0 — Blockers

### P0-1. `mstr_loop` live phase has no confirmation gate — single env var unlocks real orders
File: `portfolio/mstr_loop/config.py:19`, `portfolio/mstr_loop/execution.py:165-173, 234-243, 311-317, 463-523`.

```
PHASE: Phase = (os.environ.get("MSTR_LOOP_PHASE") or "shadow").strip()
```

Setting `MSTR_LOOP_PHASE=live` is the only requirement. Nothing checks for an approval file, Telegram confirmation, or the 90-day shadow log the CLAUDE.md explicitly requires (`docs/MSTR_LOOP_NOTES.md`). `_handle_buy` then routes through `_live_place_buy` (`execution.py:492-506`) which calls `place_buy_order` from `avanza_session` immediately. A stray export in a `.bat`, scheduled task definition, or any restart script silently transitions paper→live.

Fix: gate live with (a) `MSTR_LOOP_LIVE_APPROVED` sentinel file containing operator-signed confirmation, (b) cross-check shadow-log age ≥ 90 days, (c) Telegram round-trip "confirm LIVE" reply before first order. The CLAUDE.md memory `feedback_isk_only.md` and the rule "live trading requires manual approval" is being violated by the code path as written.

### P0-2. `mstr_loop` live path mutates state.cash_sek before order fills — drift forever
File: `portfolio/mstr_loop/execution.py:165-170`.

```python
elif config.PHASE == "live":
    ok = _live_place_buy(decision, cert_ask, units)
    if not ok:
        return False
    state.cash_sek -= total_cost  # live cash will re-sync next cycle
```

The comment claims "live cash will re-sync next cycle" but there is no reconciler reading Avanza's balance back into `state.cash_sek`. `total_cost` is `units × cert_ask` (the limit), not the realised fill price. Partial fills, missed fills, and slippage all create a permanent divergence between `state.cash_sek` and Avanza reality. The bookkeeping is used for sizing (`_notional_for_entry:101-114`), so once cash drifts low the bot starves itself; if it drifts high it overbids cash it doesn't have.

### P0-3. Grid Fisher rotation places sell + stop using warrant-price math only, no barrier check
File: `portfolio/grid_fisher.py:1151-1249`, `portfolio/grid_tiers.py:208-223`.

`rotate_on_buy_fill` computes `sell_price = fill_price * (1 + GRID_TARGET_PCT/100)` and `stop_price = fill_price * (1 - GRID_STOP_PCT/100)` (`build_exit_levels`). `GRID_STOP_PCT = 3.5` on a 5x cert → 0.7% underlying move. There is NO check that the implied underlying level at `stop_price` sits outside the knockout barrier. On a BEAR_GULD or MINI cert, a 3.5% cert drop can land the underlying inside the knockout zone, so the stop trigger fires after the cert has already been knocked out (worthless). This is the exact Mar-3 pattern the codebase has a memory file warning about (`feedback_mini_stoploss.md`). The barrier check in `_tier_skip_for_knockout` runs only on the **buy ladder**, not the rotated stop.

Additionally, per the project rule "5x leverage certificates need -15%+ stops, not -8%, to survive intraday wicks" (`.claude/rules/metals-avanza.md`), `GRID_STOP_PCT = 3.5` is dangerously tight on a 5x cert. Realised XAG/XAU volatility gives 3.5% cert moves multiple times per session — every grid rotation eats a stop-out on noise.

### P0-4. `fin_snipe_manager._compute_stop_plan` ignores barrier entirely — Mar-3 pattern
File: `portfolio/fin_snipe_manager.py:529-563`.

```python
trigger_price = _round_order_price(position_avg * (1.0 - HARD_STOP_CERT_PCT))
sell_price = _round_order_price(trigger_price * (1.0 - HARD_STOP_SELL_BUFFER_PCT))
```

`HARD_STOP_CERT_PCT = 0.05` ⇒ entry-minus-5% cert stop. The only safety check (`MIN_STOP_DISTANCE_PCT = 1.0`, line 545) is bid-vs-trigger distance, not barrier-vs-implied-underlying. For a 5x silver MINI with barrier at $76 and current silver $78, a 5%-below-entry stop on the warrant price can sit inside the knockout zone. Same Mar-3 incident pattern: stop set inside barrier → barrier hits first → instant zero, not a 5%-down sell.

Plus the same -15% rule violation as P0-3 — 5% is too tight for 5x certs.

### P0-5. `place_stop_loss_orders` does not verify cancel succeeded before placing new stop — volume overfill
File: `data/metals_loop.py:4892-4948`.

```python
# Cancel any existing orders first
if existing.get("orders"):
    _cancel_stop_orders(page, key, existing, csrf)

# ... later, unconditionally:
ok, stop_id = place_stop_loss(page, ACCOUNT_ID, pos["ob_id"], ...)
```

`_cancel_stop_orders` (`data/metals_loop.py:4979-5020`) swallows cancel failures silently into a warning log and continues. The caller (`place_stop_loss_orders`) does not check the return — it places new stops regardless. If the cancel failed (CSRF stale, Avanza 5xx, stop already in transition), the old stop and new stop both sit on the position. Combined volume can exceed position size, triggering `short.sell.not.allowed` rejection on the new stop (best case) or executing both stops back-to-back (worst case — double the intended exit, position flipped short). Project rule `.claude/rules/metals-avanza.md`: "Cancel existing stops BEFORE placing a sell (prevents overfill). Use rollback if cancel fails."

### P0-6. `_cancel_stop_orders` falls back to regular order cancel on 404 — Mar-3 pattern direction
File: `data/metals_loop.py:4992-5007`.

```javascript
let resp = await fetch('https://www.avanza.se/_api/trading/stoploss/' + orderId, {method: 'DELETE', ...});
if (resp.status === 404) {
    resp = await fetch('https://www.avanza.se/_api/trading-critical/rest/order/delete/' + orderId, {method: 'DELETE', ...});
}
```

This is the inverse direction of the Mar-3 incident: there, the regular-order API was used for stop-loss placement (caused instant fill). Here, the regular-order delete endpoint is used as a fallback when the stop-loss delete returns 404. A 404 from the stoploss API generally means the order is GONE (not a stop-loss). But if Avanza has migrated the order to a regular order state (e.g., stop triggered, awaiting fill), this fallback will try to cancel a triggered order. If the regular order is sitting waiting to fill, the cancel succeeds and the user is left naked with no stop AND the position never exits. In all cases, the 404 itself should be treated as "already gone, mark cancelled locally" — the regular-order fallback adds risk without benefit.

## P1 — High

### P1-1. ORB backtest has intraday look-ahead bias — claimed wins not achievable
File: `portfolio/orb_backtest.py:173-211`, `portfolio/orb_postmortem.py:85-101`.

```python
for day in days:
    if not day.buy_target_hit:    # actual_low <= predicted_low_median
        continue
    ...
    if day.sell_target_hit:        # actual_high >= predicted_high_median
        # Sell filled at predicted_high_median
        winning_trades += 1
```

The backtest only sees (high, low) per day, not the path. It assumes if `low ≤ buy_target` AND `high ≥ sell_target`, both fills happened in the correct order (buy first, then sell). On many days the high happens BEFORE the low (morning rally then afternoon flush). In those cases, a real buy-at-low limit would not fill until after the high — sell-at-high target would never trigger that day. Backtest reports a win; reality reports no trade or a buy-and-hold loss.

`orb_postmortem.run_postmortem:89-92` repeats the same logic on live data. Reported `simulated_pnl_pct` is therefore systematically optimistic. To fix, the backtest needs minute-level path data and a sequential fill simulator.

Secondary bug at `orb_backtest.py:226-227`: `morning_high = day.predicted_high_conservative` is approximated as morning_high but `predicted_high_conservative = morning_high * (1 + up_25/100)` which is strictly **above** morning_high. The `upside = day.actual_high - morning_high` computation is therefore biased downward, breaking `_directional_accuracy`.

### P1-2. Grid Fisher EOD market-sell does not decrement inventory after place — re-fires duplicate sells
File: `portfolio/grid_fisher.py:1514-1572`.

The code already has a TODO at line 1567-1570 acknowledging this:

```
# TODO: MANUAL REVIEW — should decrement inst.inventory_units
# here to prevent duplicate sells if eod_market_flat() runs
# again before the order fills. Current code re-sells full
# inventory on each call.
```

If `eod_market_flat` runs at minute T, then again at T+1 (next cycle, still inside `eod_minutes_remaining <= GRID_EOD_MARKET_SELL_MINUTES_BEFORE`), it re-issues a full-inventory aggressive sell. Avanza accepts it as a new order — now there are **two** sell limits on the full position. First fills the position (now flat), second fires as a short sell — `short.sell.not.allowed` rejection at best, accidental short position at worst.

### P1-3. Grid Fisher live cash gate consults stale buying-power cache up to 5 minutes after a fail — order-then-balance race
File: `portfolio/grid_fisher.py:800-864`, `grid_fisher_config.py:91`.

`_fetch_buying_power_sek` returns a cached value within `GRID_BUYING_POWER_STALE_GRACE_SECS = 300`. The hot path:

1. T0: bp fetch returns 4000 SEK, cached
2. T1: buy ladder fills, cash drops to 1500 SEK
3. T2 (within 60s window): bp cache still 4000 SEK, no refresh
4. T3 (60s < t < 5min): bp fetch fails; stale 4000 SEK reused
5. Tick places more buys against the stale 4000 cap — overshoots real 1500.

`GRID_BUYING_POWER_CACHE_SECS = 60` already lets a fill happen inside the window without updating cap. The grace fallback compounds it. The "fail-closed" claim at line 894-900 only kicks in after 5 minutes — that's not closed enough when a recent fill silently invalidated the cache.

Fix: invalidate `_buying_power_cache` immediately after every successful place/rotate; or fetch live bp before each placement decision instead of relying on cache; or use Avanza's `accountOverview` balance from the same call that fetches positions in reconcile.

### P1-4. Grid Fisher `flip_direction` cooldown not set on direction-mismatch cancels — repeated cancel churn
File: `portfolio/grid_fisher.py:1424-1430`.

```python
if inst.active_direction != direction:
    cancelled = self.cancel_armed_buys(inst)
    ...
    continue
```

Direction mismatch only cancels the buys — does not set a cooldown via `flip_direction`. On the next tick, the same mismatch is detected again; if any new tier was placed in between by another path, it gets cancelled again. Each cancel is an API call. With signals oscillating near consensus boundary (a known mode per memory file `proven signal patterns`), this becomes a cancel storm.

Also: `flip_direction` (line 457-477) clears `inst.buy_ladder = []` — but in the mismatch case, `cancel_armed_buys` only marks tiers CANCELLED, so the next tick's `prune_terminal_orders` (line 1497) drops them. Direction mismatch with seeded instruments never actually changes the instrument's `active_direction` (it stays bound to the cert's natural side), so cooldown isn't logically needed for the same instrument. But the rate-limit consequence remains.

### P1-5. Crypto/oil loops declare `run_loop() -> int` but fall off function end returning None
File: `data/crypto_loop.py:284-352`, `data/oil_loop.py:297-363`.

Docstring claims "Returns 0 on graceful shutdown (SIGINT/SIGTERM)". Implementation only has explicit `return EXIT_LOCK_CONFLICT` on lock conflict; normal shutdown falls off the bottom of the `finally:` block returning `None`. `main()` does `return run_loop(notify=notify)` then `sys.exit(main())` — Python converts None to exit code 0, masking the issue. But the contract says 0; if any caller asserts the int return type (mypy strict, test harness), it breaks. Worse: if `EXIT_LOCK_CONFLICT` is meant to signal "stop the supervisor restart loop" via exit code 11 (per `.bat` wrapper), a path that returns None will look identical to a clean shutdown — masking real lock conflicts.

### P1-6. Grid Fisher fee math ignores Avanza courtage and warrant spread
File: `portfolio/grid_fisher_config.py:45-50`, real catalog spreads in `portfolio/fin_fish.py:103,117,131,145,158`.

Config comment: "Target must clear courtage (~2 SEK round-trip on 1200 SEK = 0.17%) plus warrant spread (~0.5%) with margin. 1.2% gives ~0.5% net edge per cycle."

Real numbers on Avanza ISK for warrant trades:
- Courtage minimum ~1 SEK + ~0.25% of trade value → 3 SEK per leg, 6 SEK round-trip (0.5% on 1200 SEK)
- Spread on BEAR_GULD_X5_VON4: **2.2%** per catalog
- 1.2% target − 0.5% courtage RT − 2.2% spread = **NEGATIVE 1.5% per cycle**

Even on the 0.5%-spread AVA certs (BULL_SILVER_X5_AVA_3, BULL_GULD_X5_AVA, BEAR_SILVER_X5_AVA_12), the math is:
- 1.2% gross − 0.5% courtage − 0.5% spread = +0.2% per cycle (not the claimed 0.5%)

For VON-issued certs (2.2% spread), every grid cycle is structurally unprofitable. The cert ranking in `fish_instrument_finder.py:170-176` sorts by spread first, which helps, but the grid_fisher catalog hard-codes a VON cert in `GRID_ACTIVE_INSTRUMENTS["XAU-USD"]["SHORT"] = "1047859"` (BEAR_GULD_X5_VON4 per `fin_fish.py:135-147`).

### P1-7. Oil grid signal uses Brent BZ=F but warrants may be WTI-backed — direction can be inverted
File: `portfolio/oil_grid_signal.py:42`, `data/oil_loop.py:163-167`, `portfolio/oil_precompute.py:131-148`.

```python
UNDERLYING = "BZ=F"  # OLJAB warrants track Brent.
```

The comment asserts OLJAB tracks Brent but the codebase elsewhere fetches WTI (CL=F) for oil context (`oil_precompute._fetch_oil_futures("CL=F", "6mo")`). If a single OLJAB instrument in `GRID_ACTIVE_INSTRUMENTS["OIL-USD"]` (ob 2367797 or 2367803) actually tracks WTI, the Brent signal can be wrong-direction in periods of WTI/Brent divergence (which happens during US driving season, Cushing inventory shocks, etc.). Recommend verifying each ob_id's underlying via Avanza market-guide and routing the signal to the matching underlying.

### P1-8. Oil grid signal caches null direction on fetch failure for 5 minutes
File: `portfolio/oil_grid_signal.py:122-148, 171-176`.

```python
except SourceUnavailableError as exc:
    ...
    return {"ts": _utcnow_iso(), "direction": None, ...}
...
fresh = compute_signal()
try:
    atomic_write_json(SIGNAL_FILE, fresh)  # writes None on fail
```

A single 5-minute API blip pins oil grid into "no direction" for the full TTL even if the underlying market is active and the next call would succeed. Better: only persist on success; on failure, leave the previous cached signal in place (subject to a max staleness like 30 min). This is the Mar-XX precompute pattern fix that's already applied in `oil_precompute.py:213-223` (refresh state keeps last good value on failure) — `oil_grid_signal` regressed.

### P1-9. `fish_monitor_smart` cross-asset prices fetch DAILY bars and compare to once-set session baseline — dead signal
File: `portfolio/fish_monitor_smart.py:147-154, 640-642, 676-679`.

```python
df = fetch_klines(ticker, interval="1d", limit=2)
```

Cross-asset values are daily closes. Baselines are set ONCE at session start (`run:642`). Subsequent refreshes (`CROSS_ASSET_INTERVAL = 120s`) re-fetch the same daily close. All "cross-asset move %" alerts compare daily-close to a stale baseline of the previous daily close — they reflect overnight gap, not intraday move. The status display lies to the operator about live cross-asset behaviour. For a tool framing itself as "Smart fishing monitor — signal-aware position tracking", this is a credibility hole.

Fix: use `interval="1m"` or `"5m"` with limit ≥30, set baseline at FIRST fetch (entry-time price), and refresh against that baseline.

### P1-10. `_compute_stop_plan` skip-too-close logic fails when bid has dropped below trigger
File: `portfolio/fin_snipe_manager.py:541-553`.

```python
distance_pct = ((current_bid - trigger_price) / current_bid * 100.0) if current_bid > 0 else None
if not has_existing_stop and distance_pct is not None and distance_pct < MIN_STOP_DISTANCE_PCT:
    return {"skip": True, ...}
```

If `current_bid < trigger_price` (price already fell past entry-5%), `distance_pct` is negative — passes the `< MIN_STOP_DISTANCE_PCT = 1.0` check and is logged as "stop_too_close". The plan returns `skip=True`. This means: at the exact moment the position is most in danger (already at trigger level, deserving an immediate exit), the manager refuses to place a stop. There's no fallback emergency-sell branch in `plan_instrument` for this case. The position sits naked.

### P1-11. fin_snipe_manager uses raw open() to read postmortem JSONL
File: `portfolio/orb_postmortem.py:145-156`.

```python
with open(path, encoding="utf-8") as f:
    for line in f:
        ...
```

CLAUDE.md rule: "Atomic I/O only. Use file_utils.atomic_write_json(), load_json(), atomic_append_jsonl(). Never raw json.loads(open(...).read())". Read path uses raw open; this can read a partial line written by a concurrent atomic_append (atomic_append is per-line atomic, but reading WHILE the append is mid-flight returns the same line twice or 0 times depending on filesystem). Use `load_jsonl()` from `file_utils`.

### P1-12. mstr_loop weekend-survivor risk: position can survive Friday→Monday with no monitor
File: `portfolio/mstr_loop/session.py:54-79`, `portfolio/mstr_loop/loop.py:100-106`.

```python
if not in_window and not session.in_eod_flatten_window():
    _log_poll(None, "outside_session_window", cycle_count)
    state.save_state(bot_state)
    return
```

`in_session_window` and `in_eod_flatten_window` both return False on Saturday/Sunday (line 63-64, 74-75). If the Friday EOD flatten fails (broker error, exception, kill-switch toggled), the bot returns immediately at every weekend cycle and **never** retries the flatten. Position is naked from Friday 22:00 CET to Monday 15:30 CET (65+ hours) with no exit logic running. Monday's gap-open absorbs the full weekend news cycle uncontrolled.

Fix: EOD-flatten on Saturday morning if any position remains — at minimum, log a critical-error journal entry that triggers the auto-fix-agent.

## P2 — Medium

### P2-1. Grid Fisher hardcodes EOD at 21:55 CET, violating "do NOT hardcode" rule
File: `portfolio/grid_fisher.py:257-287`, `data/metals_loop.py:7220` (FISHING_EOD_SELL_MINUTE_CET).

The `.claude/rules/metals-avanza.md` rule states: "Check API for `todayClosingTime` — do NOT hardcode 21:55. Varies with DST." Grid fisher uses `_EOD_LOCAL_HOUR = 21, _EOD_LOCAL_MINUTE = 55` (constants), and `FISHING_EOD_SELL_MINUTE_CET = (21, 50)`. During DST gap weeks or US/EU DST mismatch, real warrant close shifts ±1h. The bot would either sweep too early (giving up live action) or too late (no time for orders to round-trip — risk of unsweep'd positions).

### P2-2. `mstr_loop` EOD time hardcoded 21:45 CET assumes 22:00 close — wrong by 1h during DST gap
File: `portfolio/mstr_loop/config.py:75-81`.

The code admits this in comments (line 78-81): NASDAQ DST transitions create ±1h offset that the hardcoded EOD doesn't handle. The "21:45 still works in 20:45 buffer week" assumption is true for shadow-mode, but in live mode a 21:45 EOD when the actual close is 20:00 means positions are held 1h45 past close — broker will let limit orders expire untouched.

### P2-3. `golddigger` hardware stop check uses 3% bid distance, no barrier check
File: `portfolio/golddigger/runner.py:189-209`.

```python
if bid > 0 and (bid - stop_price) / bid < 0.03:
    logger.warning("Stop too close to bid (%.2f vs %.2f), skipping HW stop", stop_price, bid)
```

Distance-to-bid check exists (good) but no distance-to-barrier check. For gold MINI L/S certs the barrier matters; for `1069606` (BULL_SILVER_X5_AVA_3, no barrier) it doesn't. Catalog mismatch — some certs in `_DEFAULT_CATALOG` (fin_fish.py:99-160) report `barrier: 0` but the underlying instrument may have a barrier set by issuer that's not reflected in the local catalog. Verify against live `keyIndicators.barrierLevel` rather than relying on local config.

### P2-4. `fish_instrument_finder` ranks no-barrier certs as least-safe due to None coercion
File: `portfolio/fish_instrument_finder.py:170-174`.

```python
def sort_key(c: dict) -> tuple:
    spread = c["spread_pct"]
    barrier = -(c["barrier_distance_pct"] or 0)  # None → 0 → ranks AS IF distance is 0%
    return (spread, barrier)
```

A constant-leverage cert (no knockout barrier) has `barrier_distance_pct = None`. `None or 0 = 0`, then negated to `0`. A MINI cert with 1% barrier distance (extremely dangerous) gets sort key `(spread, -1)` — lower than the safe constant-leverage's `(spread, 0)`. So MINI-1%-distance ranks SAFER than no-barrier in the ascending sort. The intent is the opposite — no-barrier should be safest.

Fix: `barrier = -(c["barrier_distance_pct"] if c["barrier_distance_pct"] is not None else 999)` (treat None as max-safe).

### P2-5. Grid Fisher `_safe_session_call` swallows ALL exceptions to default — silent broker failures
File: `portfolio/grid_fisher.py:921-964`.

```python
except Exception as exc:  # noqa: BLE001
    self._log("session_call_error", method=..., error=str(exc))
    return default
```

Every Avanza error (rate limit, session expired, instrument suspended) returns the default (None) silently. Place/cancel/quote calls treat None as "not-success" and continue; reconcile path treats None as "fetch failed, retry next cycle". The grid fisher would keep ticking with mostly None responses while Avanza is wedged. Catastrophic failures (e.g. session totally expired) need to surface to the operator via Telegram, not just journal-log.

### P2-6. Grid Fisher `__del__` cleanup of executor — not reliable on Windows process kill
File: `portfolio/grid_fisher.py:966-975`.

`__del__` only fires on graceful GC. Windows Task Scheduler `taskkill /T /F` (used by PF-DataLoop auto-restart) bypasses Python finalisation. The single-worker thread + cached Playwright sync context can leak file handles or zombie Chrome processes across restarts. Combine with the metals_loop's `atexit` (also not run on hard kill, `data/metals_loop.py:6835`) and you can accumulate dangling browser processes after enough restart cycles.

### P2-7. `reconcile_against_live` partial-fill detection mutates tier.qty in place — distorts future log analysis
File: `portfolio/grid_fisher.py:654-660`.

```python
original_qty = tier.qty
tier.qty = int(delta)
record_fill(inst, tier.tier, tier.price, side="buy")
res.inventory_drift.append((ob_id, original_qty, int(delta)))
```

The tier object is mutated to reflect the partial fill qty. The full ladder snapshot in `state.by_instrument[ob].buy_ladder` no longer matches what was originally ARMED — anyone reading the saved state later (analytics, postmortem, audit) sees the reduced qty without context. Fix: keep the original qty on tier and add a `filled_qty` field.

### P2-8. ORB predictor day_window inclusive-end vs morning_window half-open — boundary candle treatment differs
File: `portfolio/orb_predictor.py:206-207, 250-252`.

```python
morning = [c for c in day_candles if self.morning_start_utc <= c["hour"] < self.morning_end_utc]
full_day = [c for c in day_candles if self.day_start_utc <= c["hour"] <= self.day_end_utc]
```

Morning is half-open `[start, end)`; day is closed `[start, end]`. The 11:00 CET candle (UTC 10:00 winter, 09:00 summer) is in `full_day` but not in `morning`. The 22:00 candle is in `full_day` but real Avanza warrant trading ends at 21:55. Result: `day_high`/`day_low` can be set by a candle that traded after Avanza is closed — predictions are calibrated against unhittable extremes.

### P2-9. `_directional_accuracy` computation chain-of-approximations breaks accuracy stat
File: `portfolio/orb_backtest.py:226-238`.

The function admits it's approximating morning high/low via the conservative prediction (which is `morning_high * (1 + p25/100)` — strictly above morning_high). It also computes `upside = day.actual_high - morning_high` which, with the inflated baseline, biases upside downward. The reported `directional_accuracy` is therefore biased.

### P2-10. mstr_loop `_handle_partial_sell` shadow-mode accounting decrements `pos.units` but no broker order — units can divorce from broker reality if phase changes
File: `portfolio/mstr_loop/execution.py:302-323`.

If a shadow-mode session runs a partial-exit tranche, `pos.units -= units_to_sell` at line 319 even though no real sell happened. If PHASE flips to live mid-day (e.g. operator changes env var and restart picks up the same `state.json`), the `pos.units` is now smaller than Avanza's actual holding. Reconciliation is missing. The PHASE transition path is not safe with a non-empty position state. (P0-1 makes this transition trivially exploitable.)

### P2-11. mstr_loop fallback CET-offset logic is dead code (never reached when zoneinfo present)
File: `portfolio/mstr_loop/session.py:23-34, 37-46`.

`_cet_now` uses `try/except` around `from zoneinfo import ZoneInfo`. On Python 3.9+ this always succeeds. The fallback path with `_last_sunday` heuristic is unreachable on the production environment (Win11/Python 3.12 per CLAUDE.md). Either remove dead code, or test the fallback path explicitly — currently it's untested AND unused.

### P2-12. Grid Fisher buying-power cache shared across instruments — micro-race
File: `portfolio/grid_fisher.py:818-844`.

Inside one `tick()`, multiple instruments call `_effective_global_cap` which calls `_fetch_buying_power_sek`. The first call populates `_buying_power_cache`. Subsequent calls within the same tick re-use that value — which is correct given the buying-power is a per-cycle snapshot. But: if the FIRST instrument's place_buy_ladder fills a tier (synchronous via `_safe_session_call`), the cache value is now stale for the SECOND instrument's evaluation in the same tick. Not catastrophic if `GRID_CASH_SAFETY_BUFFER_SEK = 500` covers a single buy, but a worst-case sequence of placements within one tick can over-commit by ~`GRID_LEG_SEK × (n_instruments-1) = 2400 SEK`.

### P2-13. Fee/courtage not modeled in elongir P&L
File: `portfolio/elongir/bot.py:279-283`.

```python
fee = proceeds * self.cfg.commission_pct
```

Uses `commission_pct` — a single multiplicative fee. Avanza's structure is min-fee (1 SEK) + percentage; on a small position you may pay the minimum on every leg even though pct math says less. For small sims this is OK as a P&L overestimate; doesn't impact correctness, but means backtested edge appears better than live.

## P3 — Low

### P3-1. `_silver_fast_tick` uses module-globals for state — not thread-safe
File: `data/metals_loop.py:1392-1502`.

`_silver_session_low`, `_silver_session_high`, `_silver_alerted_levels`, etc. are module globals mutated without lock. Currently called from the same thread as the main loop (no race), but the comment chain suggests intent to broaden — a future call site from a different thread would race.

### P3-2. `oil_loop.fetch_live_prices` uses 1m bar 'close' as live price — 60s stale
File: `data/oil_loop.py:163-172`.

`limit=5` of 1m bars gives the last completed minute, not real-time tick. For 60s loop with fast-tick monitoring that's fine; for sub-cycle decision logic it's stale.

### P3-3. `orb_postmortem.run_postmortem` uses raw open for `orb_predictions_today.json` read
File: `portfolio/orb_postmortem.py:251`.

```python
with open(PREDICTIONS_TODAY_PATH, encoding="utf-8") as f:
    pred_data = json.load(f)
```

Same atomic-I/O rule violation as P1-11. Use `load_json()`.

### P3-4. Grid Fisher `_recent_places` is wall-clock based — TimeShift / monotonic-vs-time inconsistency
File: `portfolio/grid_fisher.py:911-919`.

Rate limiter uses `time.time()` (wall clock) but elsewhere `time.monotonic()` is used for cache (`_fetch_buying_power_sek`). NTP step or admin clock change can collapse the rate-limit window — minor.

### P3-5. `WARRANT_CATALOG` barrier=0 sentinel ambiguous with real-barrier=0
File: `portfolio/fin_fish.py:99,113,127,141,154`, `portfolio/grid_fisher.py:1487`.

```python
barrier=cat.get("barrier") if cat.get("barrier") else None
```

Treats `0` and `None` identically. AVA constant-leverage certs have no barrier, hence `0` is correct sentinel. But if the catalog ever lists a real cert with barrier exactly 0 (unlikely but possible after data ingest error), it would silently disable knockout checks. Use explicit `barrier=None` for "no barrier" instruments.

### P3-6. `oil_grid_signal._rsi` returns 100 when avg_loss==0 — divides-by-zero guard but loses sign info
File: `portfolio/oil_grid_signal.py:60-63`.

Standard Wilder RSI behaviour. Fine for trend; just note: 0 losses doesn't mean unbounded uptrend, it means insufficient down moves in the window.

### P3-7. `elongir._send_telegram` returns truthy from message_store but doesn't surface failures
File: `portfolio/elongir/runner.py:67-74`.

Failures only log to logger.warning. The bot continues without alert that ops visibility is degraded.

### P3-8. ORB fetch_klines on first batch passes no startTime — depends on Binance returning "most recent N"
File: `portfolio/orb_predictor.py:141-167`.

If Binance ever changes the default to "oldest N" (unlikely but unverified), the backtest reads ancient data instead of recent. Pin `endTime = now_ms` explicitly.

### P3-9. `grid_fisher_config.GRID_FISHER_ENABLED = True` with `PROBE_ONLY = False` — no manual gate on first run
File: `portfolio/grid_fisher_config.py:25-29`.

Comment says "Flipping PROBE_ONLY back to False so real placements route to the configured account." The flip happened. Anyone pulling this branch into a fresh environment starts placing live orders on the user's ISK without further config. Consider a `data/grid_fisher.enabled` sentinel file as a runtime gate (mirroring `mstr_loop.disabled` kill-switch pattern).

## Tests missing

- **Stop-loss-vs-barrier safety test**: assert that on a MINI cert with known barrier, neither `fin_snipe_manager._compute_stop_plan` nor `grid_fisher.rotate_on_buy_fill` produces a stop whose implied underlying lies inside the knockout zone. This would catch P0-3, P0-4.
- **Mar-3 regression: cancel API for stop-loss**: assert grid_fisher and metals_loop _cancel_stop_orders call only `/trading/stoploss/{id}`, no fallback to `/trading-critical/rest/order/delete/`. Would catch P0-6.
- **mstr_loop live phase confirmation gate**: assert that `MSTR_LOOP_PHASE=live` without `MSTR_LOOP_LIVE_APPROVED` sentinel causes `execute()` to refuse all BUY/SELL. Would catch P0-1.
- **Grid Fisher EOD market-flat idempotency**: call `eod_market_flat()` twice in succession with same state, assert second call is a no-op (not a second sell). Would catch P1-2.
- **ORB backtest intraday path test**: provide a synthetic path where high precedes low, assert simulated_pnl_pct == 0 (no buy fill before high). Would catch P1-1.
- **`reconcile_against_live` partial-fill drift**: simulate partial fill where `delta < tier.qty`, assert the tier's original `qty` is preserved (only filled_qty changes) and inventory accounting matches Avanza's reported volume.
- **Grid Fisher buy-power cache invalidation after place**: after a successful place_buy_order, assert `_buying_power_cache` is cleared so the next placement re-fetches. Would catch P1-3.
- **Cascade-stop cancel-failure test**: monkeypatch `_cancel_stop_orders` to fail, run `place_stop_loss_orders`, assert no new stops are placed (i.e. the existing-stop volume isn't doubled). Would catch P0-5.
- **`_compute_stop_plan` below-trigger guard**: feed `current_bid < trigger_price`, assert the manager produces an emergency-sell action, not a "skip — too close" decision. Would catch P1-10.
- **`fish_instrument_finder` sort safety**: feed mix of (no-barrier, barrier-1%, barrier-10%), assert no-barrier ranks safest. Would catch P2-4.

## Cross-cutting observations

1. **Barrier safety is the systemic weak spot.** Five separate stop-loss placement sites (grid_fisher.rotate_on_buy_fill, fin_snipe_manager._compute_stop_plan, metals_loop.place_stop_loss_orders, golddigger.runner._execute_order HW stop, elongir.compute_stop) all compute stops in warrant-price space without consulting the knockout barrier. The shared safety helper should live in one place (`portfolio.stop_loss_safety.compute_safe_stop(entry_cert, underlying, barrier, leverage, direction, pct)`) and refuse to return a stop whose implied underlying is within `MIN_BARRIER_DISTANCE_PCT` of barrier. The Mar-3 incident memory exists; the code defence does not.

2. **PHASE/live mode is one env-var away from production orders** in mstr_loop. Compare to metals_loop's `EMERGENCY_SELL_ENABLED` config and `MIN_BARRIER_DISTANCE_PCT` — there's a culture of defensive runtime sentinels, but mstr_loop missed it. Apply the kill-switch-file pattern (`data/mstr_loop.disabled` already exists for the opposite — operator-disable) symmetrically for operator-enable of live phase.

3. **Cache TTLs cross session boundaries silently.** `oil_grid_signal` (5min), `grid_fisher` buying-power (60s + 5min grace), `fish_monitor_smart` cross-asset (2min refresh of daily bars), `oil_precompute` (4h per-source) — none of them re-evaluate at session-open or DST-transition boundaries. A signal computed Friday 21:54 from Brent klines is read Monday 15:30 as if fresh. Add a "cache is invalid across session-boundary" predicate to the lookup path or include the session-day in the cache key.

4. **EOD timing is hardcoded in three places** with slightly different values: grid_fisher 21:55, metals_loop FISHING 21:50, mstr_loop 21:45. Per project rule, these should query `todayClosingTime` from Avanza. Especially during DST transitions, three different hardcoded values guarantee at least one of them is wrong.

5. **Reconciliation only happens partially.** mstr_loop has no reconciler against broker (P0-2). Grid_fisher reconciles open_orders + positions but does not reconcile `state.cash_sek` against broker buying power (P1-3). The metals_loop has multi-layer reconciliation (holdings_check, _reconcile_fish_engine_position, _reconcile_swing_orphans) — that pattern should be lifted into a shared helper and applied to ALL stateful subsystems.

6. **Loop lifecycle: SIGTERM is not handled in metals_loop or fin_snipe_manager.** Only `atexit` and `KeyboardInterrupt`. On Windows scheduled-task auto-restart (`taskkill /T /F`), atexit doesn't fire — singleton lock files survive, browser handles leak. crypto_loop/oil_loop register SIGTERM/SIGINT (good); metals_loop does not (P3 but worth noting given how often it restarts).

7. **Fee math is structurally over-optimistic** in grid_fisher_config and elongir. Real Avanza ISK courtage + warrant spread frequently exceeds the configured target. Recommend integrating `data.fin_fish_config` per-issuer spread + a minimum-courtage SEK value into the target-edge calculation, and refusing to place a tier whose `target_pct - 2*spread_pct - 2*courtage_pct` is negative.

8. **Test coverage for the most dangerous code paths is thin.** From the file tree under `tests/`, there's coverage for signal-engine and `iskbets`, but the metals stop-loss → cancel-old → place-new pipeline doesn't have a dedicated test asserting volume conservation. Given Mar-3 was an exec-path bug with no test that would have caught it, replicating that gap for new exec paths (grid_fisher rotate, mstr_loop live BUY) is the highest-leverage testing work.
