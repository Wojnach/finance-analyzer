# Claude adversarial review: metals-core (2026-05-12)

Scope: grid_fisher + metals_loop + crypto/oil loops + fin_fish / fin_snipe / ORB
+ precompute pipelines + MSTR/elongir/golddigger sub-bots.

## Summary

Two of the three prior P0 items are still live:

- `grid_fisher.rotate_on_buy_fill` still arms a stop_loss whose price is
  derived purely from `fill_price × (1 − stop_pct)`. There is no
  consultation of `barrier` / `financing_level` despite catalog entries
  carrying that data. Buy-tier construction in `grid_tiers.build_buy_ladder`
  does check the knockout barrier — but the matching exit pair
  `grid_tiers.build_exit_levels` does not, and the stop placement in
  `grid_fisher.py:1097-1121` calls `place_stop_loss` directly with that raw
  level. This still violates the project rule "MINI warrant stop-loss: NEVER
  near financing level (barrier)" for any inherited or non-MINI position
  whose drop math happens to put the stop inside the barrier band.

- MSTR live trading remains gated by a plain env var only
  (`MSTR_LOOP_PHASE=live`). There is no shadow-day counter, no manual
  approval token file, no signed approval. Setting the env var on Task
  Scheduler is sufficient to flip from `shadow` to `live`. The rule "90 days
  shadow + manual approval" is documentation-only.

The first prior P0 — stop-loss cancel falling back to generic
`cancel_order` — is **resolved**, but the new code path silently no-ops if
`session.cancel_stop_loss` is missing AND does not check the return status
of the cancel. The end state is the prior bug class becomes a different
failure mode: dangling stop-losses can survive into the next stop
placement, allowing duplicate stops on the same orderbook.

Additional independently-found P0/P1 issues are listed below.

## P0 — Blockers

### P0-1 — grid_fisher stop-loss is placed without barrier consultation

`portfolio/grid_fisher.py:1029-1121` rotates a filled buy by computing
`sell_price, stop_price = build_exit_levels(filled.fill_price, target_pct,
stop_pct)` (line 1046) and then calls `self.session.place_stop_loss(ob_id,
stop_price, stop_sell_price, inventory_units)` at line 1099-1104. The
`build_exit_levels` helper at `portfolio/grid_tiers.py:208-223` only knows
the fill price, target % and stop %. It never looks at the catalog
`barrier`, `financing_level`, or `leverage` for the orderbook.

Compare with `_tier_skip_for_knockout` in `grid_tiers.py:71-106` which
deliberately rejects BUY tiers that would imply a knockout band, using
`underlying_price`, `barrier`, `leverage`, and `direction`. The exit half
of the pair has no such guard.

Consequence: for any leveraged warrant that does carry a non-zero barrier
in `data.fin_fish_config.WARRANT_CATALOG` (e.g. future MINI silver/gold),
a buy fill rotates straight into a stop_loss order whose trigger sits at
warrant-price `fill × 0.965`. With 5× leverage that's a 0.7% underlying
move, well inside the typical intraday wick — and the underlying price at
which it triggers is **never compared against barrier proximity** before
the order is sent. Project rule violation.

The `_catalog_for(ob_id)` lookup at line 1354 does retrieve `barrier` and
`leverage` for buy-ladder construction, but those values are not threaded
into the rotate-on-fill path. The fix is to (a) feed catalog+spot into
`build_exit_levels`, and (b) refuse stop placement when the implied
underlying trigger price lands within `GRID_KNOCKOUT_SAFETY_PCT` of the
barrier (or move the stop to a barrier-aware floor).

### P0-2 — MSTR live trading gated only by env var

`portfolio/mstr_loop/config.py:19`:
```python
PHASE: Phase = (os.environ.get("MSTR_LOOP_PHASE") or "shadow").strip()
```
And `portfolio/mstr_loop/execution.py:165-170`:
```python
elif config.PHASE == "live":
    ok = _live_place_buy(decision, cert_ask, units)
    if not ok:
        return False
    state.cash_sek -= total_cost  # live cash will re-sync next cycle
    _record_trade("BUY", ...)
```
There is no:
- shadow-day counter check (rule: "90 days shadow + manual approval")
- approval sentinel file (e.g. `data/mstr_loop.live.approved` with
  a signed-or-dated payload)
- runtime confirmation that `data/mstr_loop_scorecard.json` has met an
  internal goal before flipping
- safety check that `BEAR_MSTR_OB_ID` is None for SHORT entries (it is —
  but only inside `mean_reversion` strategy, see config.py:33; nothing
  prevents a future strategy from arming SHORT on a missing ob_id)

Anyone with permission to set `MSTR_LOOP_PHASE=live` on the scheduled
task can promote to live. This contradicts `docs/MSTR_LOOP_NOTES.md` Phase
A requirement and is exactly the failure mode the project rule was
written to prevent.

Fix sketch: require BOTH (a) `MSTR_LOOP_PHASE=live` and (b) a sentinel
file `data/mstr_loop.live.approved` containing a date > 90 days after
the first shadow log entry in `data/mstr_loop_shadow.jsonl`, AND emit a
loud Telegram on every live-mode startup so silent promotion is visible.

### P0-3 — fin_snipe_manager stop plan ignores barrier and financing levels

`portfolio/fin_snipe_manager.py:529-563` (`_compute_stop_plan`) computes
`trigger_price = position_avg * (1 - HARD_STOP_CERT_PCT)` and
`sell_price = trigger * (1 - HARD_STOP_SELL_BUFFER_PCT)` with
`HARD_STOP_CERT_PCT = 0.05`. The function reads `current_bid` and
`position_avg` but never consults `barrier_level` or `financing_level`
even though `_summarize_market` extracts both at lines 605-607.

For a 5× MINI silver warrant near the barrier, a 5% cert-price drop is
about a 1% underlying move. If the position was opened in a tight
barrier window (e.g. financing 75.03 SEK and silver around 78), the
stop trigger price can sit below the financing distance, which is the
exact scenario the memory `feedback_mini_stoploss.md` (CRITICAL: NEVER
place stop-losses near MINI warrant barriers) warns against. This
manager runs on real Avanza orders (`place_stop_loss_no_page`,
`delete_stop_loss_no_page`).

Fix: gate the stop plan on `(position_avg - financing_level) /
position_avg >= MIN_BARRIER_DISTANCE_PCT` and refuse / widen the stop
otherwise. Mirror `portfolio/fin_fish.py:737-755` which already does the
right thing for tier placements.

## P1 — High

### P1-1 — grid_fisher does not check `cancel_stop_loss` return status

`portfolio/grid_fisher.py:1089-1094` and `1404-1412` call
`self._safe_session_call(cancel_fn, inst.stop_loss_id, default=None)` and
discard the result. `portfolio/avanza_session.cancel_stop_loss` (line 888)
returns `{"status": "SUCCESS"|"FAILED", "http_status": int, ...}`. A
FAILED response is silently swallowed; `rotate_on_buy_fill` then proceeds
to call `place_stop_loss`, producing a second stop on the same orderbook
volume.

Worst case: stop A from a prior fill plus stop B from the new rotation
both target the same `inventory_units`. If the price ladder hits both
trigger levels in sequence, the second sell oversells the position by up
to `units` units → short position. The "Sell + stop-loss volume must NOT
exceed position size" rule from `.claude/rules/metals-avanza.md` is
violated.

Fix: check `result.get("status") == "SUCCESS"`, log on failure, and
refuse to place a replacement stop when the cancel failed (or place with
volume reduced by the still-live stop's volume).

### P1-2 — grid_fisher missing cancel_stop_loss method = silent dangling stop

In the same lines 1089-1094 / 1404-1412, the code uses `getattr(self.session,
"cancel_stop_loss", None)` and silently does nothing if the attribute is
missing. The avanza_session module exposes the function (line 888), but
unit-test fakes and mock sessions injected via the constructor parameter
(line 727 — `session: Any`) often won't. A test session that places stops
but lacks the cancel helper leaves orphan stops on Avanza on the next
rotation and the test surface won't catch it.

Fix: raise (or set state to halted) when `cancel_stop_loss` is missing
AND `inst.stop_loss_id` is set. Don't silently skip a mutating step that
will trigger duplicate-stop bugs.

### P1-3 — EOD-flat sweep dumps at bid×0.99 without barrier-distance check

`portfolio/grid_fisher.py:1414-1424`:
```python
aggressive = round(max(bid * 0.99, 0.01), 2)
result = self._safe_session_call(
    self.session.place_sell_order, inst.ob_id, aggressive,
    inst.inventory_units, default=None,
)
```
On a position where bid is already near the barrier, `bid × 0.99` can be
below the warrant's intrinsic floor. The order won't fill at the wanted
price and the catalog's barrier proximity isn't even checked. Combined
with EOD being a fire-and-forget sweep (the result of `place_sell_order`
is logged but no retry / verification), a tail-end barrier touch could
leave the position open into close.

Fix: floor `aggressive` at `max(bid × 0.99, financing_level × 1.005)` for
LONG MINIs and the symmetric ceiling for SHORTs. Or fall back to an
end-of-day market-only "best effort" sell that explicitly accepts the
slippage but logs it loudly.

### P1-4 — `grid_fisher.minutes_until_eod` returns +inf when zoneinfo is absent

`portfolio/grid_fisher.py:261-286`. When zoneinfo is unavailable (test
host, stripped runtime), the function returns `float("inf")`. The caller
at `data/metals_loop.py:7548` then compares `_eod_min <= 10` and
`_eod_min <= 5`, both False — so the EOD sweep NEVER fires. With
`GRID_FISHER_ENABLED=True` and `GRID_FISHER_PROBE_ONLY=False`, that means
no force-flat at the end of the session in any environment that's lost
its tzdata. The project rule "EOD-flat" silently breaks.

Fix: fall back to a fixed UTC schedule (compute the 21:55 Stockholm
cutoff using current UTC + DST detection from `portfolio.market_timing`)
rather than disabling the sweep. Loudly log "tzdata missing — falling
back to fixed schedule" on first hit.

### P1-5 — ORB walk-forward shares `MIN_HISTORY` baseline; no purging or embargo

`portfolio/orb_backtest.py:131-164` (`walk_forward_backtest`) is correct
in only feeding past days into the predictor (`training_data =
day_results[:i]`), but the predictor itself (`ORBPredictor.predict_daily_range`,
line 305-384) just sorts the historical extension percentages and takes
quartiles. There is no:
- gap / embargo between training and test days (a 15m bar from yesterday's
  late afternoon can correlate strongly with today's morning bar; ORB
  ignores this leak)
- adjustment for autocorrelated extensions
- per-regime split (DST transition days, FOMC days, weekends-after-extreme
  flush — all flow into the same sample)
- handling of the "morning_direction" filter when the test day's `direction`
  matches a very small filtered sample (filtered set ≥ `min_sample=5` is
  the only floor; one or two outliers dominate the quartile)

Worse, the `leave_n_out_validation` function at line 304-374 is **not
walk-forward** — it picks N random eligible days globally and trains on
"all days except holdout, that come before this day". With N=5 and 50
iterations, the same training set is reused with tiny perturbations, and
the metrics aggregate across these effectively-overlapping trials.

Reported `directional_accuracy`, `win_rate`, `total_pnl_pct` from
`run_backtest` are therefore optimistic. Anyone reading the report would
overestimate stability.

Fix: add a fixed-window training set (rolling 60 days), enforce an
embargo (skip predicting day D when training set contains day D-1's
afternoon — or equivalently drop the most-recent K days), and replace
leave-N-out with proper k-fold blocked CV.

### P1-6 — `_simulate_trades` assumes the buy fill happens before the sell

`portfolio/orb_backtest.py:173-211`. `_simulate_trades` increments
winning_trades whenever `day.sell_target_hit`, regardless of intraday
order. If actual_high happens BEFORE actual_low on the same day, the
"buy at predicted_low_median" never fires, but the simulator still
credits the day as a win. The data has `high_hour_utc` and `low_hour_utc`
available on `DayResult` (line 81-82) but they are not threaded through
to `BacktestDay` (line 22-46) so the simulator cannot check ordering.

This silently overstates win-rate and P&L. The bias is correlated with
volatile up-trending days.

Fix: capture `high_hour_utc` / `low_hour_utc` on BacktestDay and skip
the trade unless `low_hour_utc < high_hour_utc`.

## P2 — Medium

### P2-1 — oil_grid_signal cache has only a 5-minute TTL but uses 1h klines

`portfolio/oil_grid_signal.py:39, 43`. `REFRESH_INTERVAL_SEC = 300` but
`INTERVAL = "1h"` and the signal is derived from EMA9/21 on hourly bars.
Within a single 1-hour bar, the signal will flip between LONG/SHORT/None
based on intra-hour wick noise; the cache merely throttles a fetch that
returns the same closed-bar EMA most of the time. Effectively the grid
fisher re-arms direction every 5 minutes on stale 1h bar data. Better:
align cache TTL with the bar interval (60 min) or store the last closed
bar's signal and only refresh after that bar closes.

### P2-2 — `oil_precompute._should_refresh` clamps OK status on read failure

`portfolio/oil_precompute.py:131-223`. When a source fetch fails, the
state stays `{"last_error_ts": now}` but **never resets `ok` to False**
because the original `{"ts": ..., "ok": True}` from a prior success is
left unchanged via `{**refresh_state.get(...,{}), "last_error_ts": now}`.
The output's `result["refresh_status"]` then reports the source as
`ok=True` even after consecutive failures. Downstream consumers reading
`oil_deep_context.json` cannot distinguish fresh-and-good from
stale-and-failing.

Fix: explicitly write `"ok": False` on the failure branch and surface
"degraded" in the output context.

### P2-3 — Grid fisher tier global cap does not include sell ladders

`portfolio/grid_fisher.py:176-183` (`planned_notional_sek`) and
`global_planned_notional` (line 561-565). The "armed buys + inventory at
avg entry" formula tracks deployed capital, but the in-flight ROTATE
state (a buy just filled, sell+stop not yet placed) and the resting sell
ladder are not subtracted from headroom even when the sell limit would
reduce exposure. This isn't a correctness issue but it can cause the
cap to be artificially binding right after a rotation, blocking the
next buy tier that would otherwise be within budget.

### P2-4 — `fin_snipe_manager._compute_stop_plan` `MIN_STOP_DISTANCE_PCT = 1.0`

`portfolio/fin_snipe_manager.py:64`. On 5× MINIs, a 1% underlying move
is a 5% cert move. Combined with `HARD_STOP_CERT_PCT = 0.05` (line 61)
the stop sits exactly at the breakeven; bid noise within the typical
0.5% intraday wick can chop the stop into oblivion. This is an
operational concern (false stop-outs) and likely costs more in lost
P&L than the barrier proximity scenario.

### P2-5 — Crypto/oil loops fetch live prices but never gate market hours

`data/crypto_loop.py:227-247` and `data/oil_loop.py:248-268` run
`run_one_cycle` unconditionally — no `is_market_hours()` gate, no
sleep-until-open. For oil, this is correct (futures trade ~24x5). For
crypto it's also correct (24/7). But the swing-trader inside both is
not "warrants are tradeable" aware — when DRY_RUN flips to False and
the warrant catalog is in place, both will keep emitting orders even
during Avanza warrant off-hours (22:00-08:15 CET). Today the DRY_RUN
defaults save us. There is no defence-in-depth.

### P2-6 — fast-tick `_silver_fast_tick` mutates global state with no lock

`data/metals_loop.py:1393-1505`. `_silver_session_low`, `_silver_session_high`,
`_silver_consecutive_down`, `_silver_prev_price`, `_silver_alerted_levels`,
`_silver_fast_prices` are module-level globals mutated without locks.
Today this is called only from `_sleep_for_cycle` on the main thread so
there's no race. BUT the GridFisher runs `place_buy_ladder` on its own
persistent worker thread (`portfolio/grid_fisher.py:799-836`) and that
thread can call back into shared modules. The risk is low because the
worker only calls `avanza_session`, not metals_loop globals — but the
implicit contract "metals_loop globals are single-threaded" should be
documented or enforced with a lock so future changes don't silently
introduce a race.

### P2-7 — crypto_precompute does not write `_REFRESH_STATE_FILE`

`portfolio/crypto_precompute.py:110-236`. Unlike `oil_precompute`,
`crypto_precompute._fetch_market_data` does not honour per-source TTLs —
on every successful precompute, every source is re-fetched. This is
inefficient (CoinGecko free-tier rate limit is real) and inconsistent
with the doc-string promise of refreshing on the configured intervals
(line 36-45). Either implement the same gating as oil_precompute or
remove `_REFRESH_INTERVALS` from the file.

## P3 — Low

### P3-1 — `_eod_sell_fishing_positions` uses CET-hour rounding

`data/metals_loop.py:7184-7196` uses
```
_h_raw, _, _ = get_cet_time()
_h_int = int(_h_raw)
_m_int = round((_h_raw % 1) * 60)
```
Rounding minutes from a float-hour is brittle near the boundary (e.g.
21.4999 rounds to 30, missing the 50-minute trigger). On a slow loop
cycle the `_eod_h == _h_int and _m_int >= _eod_m` window can miss
entirely if the cycle takes longer than 60s and the loop transitions
from 21:49 to 21:51. Use `datetime.datetime.now(stockholm_tz)` and
compare directly. The legacy guard `_eod_fishing_sold_today` makes
this idempotent, so impact is bounded.

### P3-2 — `ORBPredictor.translate_to_warrant` documentation is wrong

`portfolio/orb_predictor.py:411-414`:
```python
factor = intrinsic_target / (silver_target - fl)  # This simplifies but...
# Actually: factor = new_intrinsic / current_intrinsic
factor = intrinsic_target / intrinsic_entry
```
The first assignment is overwritten by the second, and the comment claims
`factor = new_intrinsic / current_intrinsic` but the implementation uses
`entry` not `current`. Cosmetic — but consumers reading the field will
get the wrong factor when current diverges from entry.

### P3-3 — `_morning_window_utc` hard-codes EU DST 1-hour shift

`portfolio/orb_predictor.py:36-46`. Uses `_is_eu_dst` from
`portfolio.market_timing` to flip between UTC start hours 7/8. For the
XAG-USD silver instrument the underlying is global, not EU-bound — the
DST shift correctly maps to local 09:00-11:00 CET/CEST. The
implementation is correct for that goal but the function name suggests
EU DST, which would be wrong if the symbol changes. Cosmetic.

### P3-4 — Elongir uses constant `financing_level = 75.03`

`portfolio/elongir/config.py:33` hard-codes the MINI Long silver
financing level. Real MINI financing drifts daily (overnight cost). The
bot is paper / signal only, so trade math is purely indicative, but the
fix would be to read the live `financingLevel` keyIndicator on each
Avanza market_guide pull.

### P3-5 — `_seed_state_for_active_instruments` silently writes `unknown_ob_<id>`

`portfolio/grid_fisher.py:299-322`. When the catalog doesn't carry a
matching ob_id, the instrument is created with `cert_name=
f"unknown_ob_{ob}"` and proceeds. Better to refuse and log a critical
config mismatch, since trading an instrument whose catalog metadata
(barrier, leverage, parity) is unknown is exactly what the knockout
guard depends on.

## Status of prior P0s (2026-05-11)

- **grid_fisher.py:1089-1094 stop-loss cancel falls back to generic
  cancel_order — RESOLVED for the fallback path.** The code now uses
  `getattr(self.session, "cancel_stop_loss", None)` and only calls if
  present. However:
  - The return status is discarded (now P1-1 above).
  - Missing `cancel_stop_loss` method results in a silent no-op (now
    P1-2 above), so duplicate stops can still occur in environments
    where the helper isn't injected.

- **grid_fisher.py:1097-1098 + grid_tiers.py:208-223 stop placement
  skips knockout check on inventory — STILL PRESENT.** `build_exit_levels`
  has no barrier parameter; the `_tier_skip_for_knockout` guard exists
  only on the BUY side. The catalog is fetched at line 1354 but its
  `barrier` field is not threaded into the rotate path. See P0-1 above.

- **mstr_loop/config.py:19 + execution.py:165-169 PHASE=live env-gated
  only — STILL PRESENT.** No shadow-day counter, no approval sentinel,
  no Telegram acknowledgment required. See P0-2 above.

## Tests missing

- `tests/test_grid_fisher_stop_barrier.py` — table-driven test: given a
  MINI catalog with `barrier=80.0`, `leverage=5.0`, and a fill at warrant
  price 100, rotate must NOT place a stop whose implied underlying lies
  within `GRID_KNOCKOUT_SAFETY_PCT` of 80.

- `tests/test_grid_fisher_cancel_stop_failure.py` — inject a fake session
  whose `cancel_stop_loss` returns `{"status": "FAILED"}`. Assert no new
  stop is placed AND a halt entry / journal warning fires.

- `tests/test_grid_fisher_missing_cancel_stop.py` — inject a fake session
  without `cancel_stop_loss`. Assert the rotation refuses to place a new
  stop (state.halted=True OR no `place_stop_loss` call).

- `tests/test_mstr_loop_live_gate.py` — set `MSTR_LOOP_PHASE=live`
  without the proposed approval sentinel; assert `_handle_buy` returns
  False with a "live mode not approved" rationale.

- `tests/test_fin_snipe_manager_barrier_stop.py` — feed a snapshot whose
  `barrier_level` is within 3% of `position_average_price`; assert
  `_compute_stop_plan` returns `skip=True` with reason `"barrier_too_close"`.

- `tests/test_grid_fisher_minutes_until_eod_no_zoneinfo.py` — monkeypatch
  zoneinfo to raise `ImportError`; assert `minutes_until_eod` returns a
  finite value derived from UTC (NOT `float("inf")`) so EOD still fires.

- `tests/test_orb_backtest_simulate_trade_ordering.py` — synthesize a
  day where actual_high precedes actual_low. Assert `_simulate_trades`
  records zero trades (buy never filled), not a win.

- `tests/test_orb_backtest_leave_n_out_no_overlap.py` — sanity check that
  consecutive iterations of `leave_n_out_validation` operate on
  non-overlapping training sets, or replace with k-fold blocked CV.

- `tests/test_oil_precompute_refresh_state_failure.py` — make a fetcher
  raise; assert the next `result["refresh_status"]` for that key shows
  `ok=False`, not the stale `ok=True` from prior success.

- `tests/test_crypto_precompute_refresh_ttl.py` — verify that consecutive
  calls within `_REFRESH_INTERVALS["fear_greed"]` do not re-fetch the
  Fear & Greed endpoint.
