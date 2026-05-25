# Metals-Core Adversarial Review

Worktree: `Q:/finance-analyzer-worktrees/review-metals-core`
Date: 2026-05-25
Scope: metals warrant trading core (grid market-maker, fish engine, EOD,
microstructure, oil/silver/gold precompute, fin_snipe_manager, ISKBETS,
crypto swing trader).

## P0 findings

### P0-1 — grid_fisher: duplicate orders on session-call timeout
`portfolio/grid_fisher.py:1392-1424` (`place_buy_ladder`) +
`portfolio/grid_fisher.py:997-1040` (`_safe_session_call`).

`_safe_session_call(self.session.place_buy_order, ob_id, price, qty)` uses
`future.result(timeout=30)`. If Avanza accepts the order but the HTTP
response is slow (or Playwright stalls), the future times out, the helper
returns `default=None`, and the code logs `place_buy_failed`. The tier is
**not** appended to `inst.buy_ladder` (append is gated on
`status == "SUCCESS"` at line 1405-1418).

On the NEXT tick the reconciler builds
`existing_tiers = {t.tier for t in inst.buy_ladder if t.status == ARMED}`
which still excludes the missing tier; `place_buy_ladder` then issues a
FRESH order for the same tier index. Avanza sees both orders as
independent placements (they have different order_ids), and the original
order is invisible to grid_fisher because we never recorded its
`order_id`. Result: **two resting buys for one intended tier** until the
silent fill is reconciled (which only happens when `live_vol -
inst.inventory_units >= tier.qty` — i.e. AFTER a fill, by which point
both may have already filled). Real-money risk: 2× notional on a single
tier, blowing past per-instrument cap and (because cap is checked AFTER
this happens) past the global cap as well.

Fix: store a pending placement marker (e.g. an idempotency key persisted
before the call, status `PENDING`) and require a positive
"order_not_found-on-Avanza" confirmation before retrying. Or skip the
tier for one cycle after a placement timeout and let the next reconcile
adopt it via order-id discovery.

### P0-2 — emergency_sell sends price=0 sell orders to Avanza
`data/metals_loop.py:2074-2075` and `data/metals_loop.py:3736-3797`.

`_eod_sell_fishing_positions` explicitly calls
`emergency_sell(page, key, pos, 0)` when no bid is available; the comment
says "attempting emergency sell at 0". `emergency_sell` then constructs
a Playwright `page.evaluate` POST to `/_api/trading-critical/rest/order/new`
with `"price": 0` and `"side": "SELL"`. There is **no client-side
price>0 guard** in this code path — unlike `portfolio.avanza_session._place_order`
(line 586-588) which raises on `price<=0`.

Avanza will almost certainly reject a price=0 limit sell, but the
defense is broker-side only — and `emergency_sell` itself enforces no
guards, which means a malformed price (e.g. NaN passed in by upstream
math) is shipped directly. Today `EMERGENCY_SELL_ENABLED=False`
(line 427), so the path is dormant; the moment that flag flips, the
zero-price sell is a latent footgun.

Fix: refuse to construct a SELL payload when `bid <= 0` (or any
non-positive). For the legitimate "we MUST flat now" case, switch to a
market-style aggressive limit (e.g. last seen price × 0.95) like
`grid_fisher.eod_market_flat` does on its own quote-failure path.

### P0-3 — grid_fisher cross-account position confusion
`portfolio/grid_fisher.py:666-675` + `portfolio/avanza_session.py:676-717`.

`reconcile_against_live` calls `self.session.get_positions()`, which is
`portfolio.avanza_session.get_positions()`. That function ignores
`account_id` and returns positions across **every Avanza account on the
session** (ISK 1625505 + pension 2674244). The
`_position_volume_for(positions, ob_id)` helper iterates and returns the
FIRST orderbook_id match — there is no account-id filter.

If the user holds the same orderbook_id in the pension account
(2674244) and grid trades in the ISK account (1625505), grid_fisher
will see `live_vol > inst.inventory_units` and:

1. For any ARMED buy tier whose order_id is missing from open_orders
   (e.g. a placement-timeout from P0-1, or a normal cancel mid-cycle):
   `delta = pension_vol - 0 ≥ tier.qty` → `record_fill` at the tier's
   limit price → marks tier FILLED at a price NEVER paid → triggers
   `rotate_on_buy_fill` which places a SELL for shares the ISK account
   doesn't have. Avanza will reject the sell with `short.sell.not.allowed`,
   but grid_fisher's state now claims phantom inventory.
2. Subsequent buy ladders see `hit_per_instrument_cap()` triggered by
   phantom notional → ladder placement starves.
3. EOD market-flat reads `inst.inventory_units > 0` → attempts to sell
   units never owned in the ISK account.

The fix lives in `avanza_session.get_positions` — it must accept and
filter by `account_id`. The grid_fisher call site must also pass
`self.account_id` through.

Memory `feedback_isk_only.md` is explicit: only ISK 1625505 should be
touched; pension 2674244 must be ignored. The current code does not
enforce this for the grid path.

## P1 findings

### P1-1 — fish_engine peak/trough state not persisted (Bug 3 silently re-introduced on restart)
`data/fish_engine.py:1006-1048` (`to_dict`/`from_dict`) vs
`data/fish_engine.py:185-188` and `2026-04-13 Bug 3` fix.

`underlying_peak_price`, `underlying_peak_ts`, `underlying_trough_price`,
`underlying_trough_ts` (added 2026-04-13 to veto ORB continuation votes
during pullbacks) are NOT serialized. After every process restart the
peak/trough resets to 0/inf, the veto logic disables, and the engine can
again fire LONG entries -4.7% below the intraday peak — the exact loss
pattern Bug 3 was written to prevent. `_hold_tick_count` is also not
persisted (cosmetic side effect: HOLD-spam suppression resets too).

Fix: add the four fields to `to_dict`/`from_dict`. They're cheap to
serialize.

### P1-2 — microstructure_state double-records OFI per persist cycle
`portfolio/microstructure_state.py:175-213`.

`get_microstructure_state(ticker)` calls `record_ofi(ticker, ofi)`
exactly once — the docstring at `record_ofi` (line 92-100) says "Called
once per cycle from get_microstructure_state to avoid double-appending".
But `persist_state()` (line 205-213) ALSO calls `get_microstructure_state`
on every persisted ticker, which means each persist appends an extra
OFI sample to the rolling history.

`metals_loop._accumulate_orderbook_snapshots` (line 1830-1834) calls
`persist_state()` every 5th cycle. So on every 5th cycle: main signal
path records OFI once, then persist_state records OFI again for every
ticker. The `_ofi_history` deque is bounded at 120, but the
distribution is skewed — recent values get over-represented, compressing
the z-score and biasing the orderbook_flow signal toward HOLD.

Fix: split state read from state mutation. `get_microstructure_state`
becomes a pure read; a separate `record_cycle_ofi` is the only mutator
and is called explicitly once per main cycle.

### P1-3 — grid_fisher inventory drift never auto-corrected
`portfolio/grid_fisher.py:752-757`.

`reconcile_against_live` logs inventory_drift when `live_vol !=
inst.inventory_units` but does NOT update `inst.inventory_units` from
live. Comment claims "the caller decides whether to forcibly align" —
no caller does. Result:

- User manual partial sell of a grid-managed position → grid's cached
  `inventory_units` stays high.
- `hit_per_instrument_cap()` and `planned_notional_sek()` use the stale
  count, so `_effective_global_cap` and per-instrument cap are wrong.
- `eod_market_flat` attempts to sell stale-count units → broker
  short-sell rejection → naked position into close.

Fix: either (a) trust live as ground truth and align inst.inventory_units
on drift > N units, or (b) emit a critical_errors.jsonl entry and refuse
further placements on that instrument until reconciled by an operator.

### P1-4 — minutes_until_eod returns inf when zoneinfo unavailable, silently disables EOD-flat
`portfolio/grid_fisher.py:285-310`.

`minutes_until_eod()` returns `float("inf")` when `import zoneinfo`
fails OR when `ZoneInfo("Europe/Stockholm")` raises. The docstring
frames this as fail-safe ("caller never triggers EOD on the failure
path"). In practice this is fail-DANGEROUS: the EOD market-flat path is
the only thing that closes intraday-only grid positions before
overnight gap risk. A missing tzdata install (Windows-without-tzdata
ships exactly this failure mode) means **positions are silently held
overnight**.

The grid trades 5x leveraged certificates on metals/oil. Overnight gap
risk on these is material. Fail-DANGER, not fail-safe.

Fix: if zoneinfo is unavailable, log a critical_errors.jsonl entry AND
fall back to a UTC-based cutoff (e.g. CET=UTC+1, CEST=UTC+2) — or refuse
to operate the grid at all.

### P1-5 — fin_snipe_manager stop placement has no barrier-distance check
`portfolio/fin_snipe_manager.py:529-563` (`_compute_stop_plan`).

`trigger_price = position_avg * (1 - 0.05)` — 5% below entry. No check
against the warrant's knockout barrier. For MINI warrants
(per `data/fin_fish_config.py:190-553`, leverage 10-30x, barriers within
3-14% of underlying), a 5% trigger drop can place the stop AT or PAST
the knockout level. If the warrant has already been knocked out, the
stop never fires; the position decays to ~0.

`MIN_STOP_DISTANCE_PCT = 3.0` (line 64) is distance from current bid,
not from barrier. Memory `feedback_mini_stoploss.md` is explicit:
"Never place stop-losses near MINI warrant barriers". Not enforced.

Fix: when `pos.api_type == "warrant"` and `barrier > 0`, compute the
implied underlying level at trigger_price and refuse placement if
within X% of barrier.

### P1-6 — _safe_session_call retries hide silent broker rejects from per-tick cap math
`portfolio/grid_fisher.py:1392-1411`.

When `place_buy_order` returns a non-SUCCESS response (e.g. silent
broker reject, insufficient buying power, halted instrument), the tier
is NOT added to `inst.buy_ladder`. The next tick's `existing_tiers`
check will retry the placement. The OUTER global cap calculation uses
`global_planned_notional(self.state)` which only sees ARMED tiers — so
a rejected placement does NOT consume cap budget. This means a
chronically-rejecting instrument (e.g. quote-stale window where Gate A
doesn't fire) can spam-place into the rate-limit ceiling, denying ALL
instruments their cap allocation that cycle.

Gate B (rapid_cancel_count) catches the cancel-side version of this
once 2 cancels happen within 120s. It does NOT catch silent-reject
which never produces a cancel.

Fix: add a `place_rejected_count` counter on `InstrumentState`, with
its own cooldown after N consecutive rejects. Mirror Gate B's design
but on the placement-side.

### P1-7 — metals_avanza_helpers.place_order has no price>0 or volume>0 guard
`data/metals_avanza_helpers.py:253-327` (`place_order`).

The Playwright-based `place_order` validates CSRF token and warns on
sub-1000 SEK orders but does NOT validate `price > 0` or `volume > 0`.
This contrasts with `portfolio.avanza_session._place_order` (line
584-588) which raises on either. Any caller that hits the Playwright
helper with malformed inputs (e.g. `bid=0`, `units=0` from a
fetch_price failure that falls through to placement) sends the broken
order to Avanza.

Today the consumers (metals_swing_trader, emergency_sell) mostly
pre-check, but the defense-in-depth gap is the bug pattern that
historically bit in March (cancel-stop-loss API mismatch).

Fix: copy the avanza_session validations (price>0, volume>=1,
order_total within MAX_ORDER_TOTAL_SEK, account whitelist) into the
Playwright helper. Single helper, single guard surface.

### P1-8 — silver_fast_tick OFI snapshot increments under empty-best-book gates poorly
`portfolio/microstructure.py:118-148` (`compute_ofi`).

When `prev_bid_vol=0` and `curr_bid > prev_bid`, OFI adds
`curr_bid_vol` — fine. But when both volumes are 0 (FAPI returns empty
top-of-book during a flash crash), `delta_bid = curr_bid_vol - prev_bid_vol
= 0` for the equal-price branch but doesn't degrade gracefully if best_bid
itself is missing. Code accesses `prev["bids"][0][1] if prev["bids"] else 0`
which only protects against empty list, NOT against malformed level
like `[None, None]` from a transient API hiccup.

Not actively bleeding money, but the OFI feeds the orderbook_flow
signal which is one of the gating inputs for swing_trader entries.
Garbage in → spurious signal.

### P1-9 — grid_fisher `eod_market_flat` falls back to avg_entry_price as bid
`portfolio/grid_fisher.py:1882-1891`.

When `get_quote` returns None, code uses `bid = inst.avg_entry_price`,
then `aggressive = round(max(bid * 0.99, 0.01), 2)`. If the cert price
has moved up 20% intraday, the EOD-flat order goes in at -21% from
current market — locks in a worse exit than the (just-failed) live
quote would have given. Not a blowup but a definite money leak when
EOD coincides with a quote API blip.

Fix: cache the last successful quote per ob_id (TTL e.g. 60s) and use
THAT as the bid fallback, not entry price.

### P1-10 — oil_grid_signal cache has no upper bound on age before refuse
`portfolio/oil_grid_signal.py:151-176`.

`get_cached_or_refresh` returns the cached value when `age <
REFRESH_INTERVAL_SEC (300)`. On `compute_signal` failure (FAPI/yfinance
down for hours), the cached file ages past 300s and the next call
attempts a fresh fetch which also fails — returning a fresh dict with
`direction=None, confidence=0.0` (correct).

But the cached file is NEVER deleted, and a stale `direction` from
hours ago could survive a fetch failure if the write-on-fresh path
fails (line 173-175 swallows exceptions). The next caller reads the
stale cache. There's no hard "if last successful fetch > X hours ago,
return None" gate. P1 because oil is leveraged 5x and trading on a
stale signal direction is real-money exposure.

Fix: add a max-age cutoff (e.g. 30 minutes) past which the cache is
discarded regardless.

## P2 findings

### P2-1 — grid_fisher worker thread leaked on instance abandon
`portfolio/grid_fisher.py:997-1051`.

`_safe_session_call` lazily creates a single-worker `ThreadPoolExecutor`
held on the instance. Cleanup is in `__del__`. If GridFisher is held
elsewhere (e.g. metals_loop main process) and the process is killed
hard (SIGKILL, OOM), `__del__` won't run and the executor's worker
thread leaks — but in that case the process dies too, so the leak is
inert. Defensive: switch to `weakref.finalize` or an explicit
`close()` method that the metals_loop calls on shutdown.

### P2-2 — metals_accuracy_review uses raw open()+json.loads, hardcoded chdir
`data/metals_accuracy_review.py:14, 28-37`.

`os.chdir(r"Q:/finance-analyzer")` at module top — breaks portability
and silently relocates cwd of any importing process. Raw
`open(DECISIONS_FILE).readlines()` + `json.loads` violates CLAUDE.md
Critical Rule 4 ("Atomic I/O only").

Fix: switch to `load_jsonl` from `file_utils`; drop chdir.

### P2-3 — fin_fish_config inline catalog drift vs runtime catalog
`portfolio/fin_fish.py:91+` vs `data/fin_fish_config.py:9+`.

`fin_fish.py` carries an inline `_DEFAULT_CATALOG` used when
`data.fin_fish_config` import fails. The inline copy is incomplete
(missing OIL warrants, missing several MINI/TURBO entries). If
`fin_fish_config.py` import fails for any reason (e.g. syntax error
during a merge), fin_fish silently falls back to a half-truth catalog.
Better to fail-fast.

### P2-4 — grid_fisher direction-mismatch cancel doesn't reset rapid_cancel counter
`portfolio/grid_fisher.py:1747-1753`.

When the signal flips direction and we cancel armed buys, the cancel
fires `cancel_buy_tier` → `prune_terminal_orders`. The next reconcile
won't see those tiers (already pruned). Then `external_cancel_buy`
won't fire for them. But manually-cancelled-due-to-flip tiers don't
reset `rapid_cancel_count` either — it only resets via a fill or
session roll. A run of legitimate signal flips can pile up
rapid_cancel_count and trip a 6h cooldown for unrelated reasons.

Fix: clear `rapid_cancel_count` on direction flip.

### P2-5 — fin_snipe_manager critical alerts read config.json on every alert
`portfolio/fin_snipe_manager.py:99-108`.

Each critical alert re-reads `config.json` from disk. config.json is a
symlink to the user's secret store. High frequency disk hits +
unnecessary symlink resolution. Cache the config once at module load.

### P2-6 — metals_cross_assets has no fallback when yfinance hangs
`portfolio/metals_cross_assets.py:55-73`.

`fetch_klines` failure returns empty DataFrame, which becomes None
upstream. But `_yf_download` itself wraps in `try/except` with only
`logger.warning` — no rate-limit / retry backoff. If yfinance is hung
(common during their outages), every cross-asset getter blocks for the
yfinance default timeout (~10s) per call. With 8 getters (copper, gvz,
gs_ratio, oil, spy, plus intraday variants), one cycle can spend 80
seconds blocked on dead yfinance — exceeds the 60s loop budget.

Fix: explicit 5s timeout on price_source.fetch_klines for cross-asset
calls, OR cache last successful response and serve stale during outage.

### P2-7 — grid_fisher state schema bumped without migration script
`portfolio/grid_fisher_config.py:185`.

`GRID_STATE_SCHEMA_VERSION = 2` and the loader migrates v1→v2 via
`.get()` defaults. Fine for the current diff (Gate B fields default to
0/None). No problem today, but if the schema grows non-defaultable
fields later, the silent migration becomes a silent corruption.

Fix: when bumping the version, require an explicit migration function
keyed on (from_version, to_version).

### P2-8 — fish_engine straddle mode lacks per-tick deadline
`data/fish_engine.py:213-232`.

The session-end auto-disable fires at 21:55 CET. If `hour_cet` /
`minute_cet` are passed stale by the caller (e.g. metals_loop computes
them once at cycle start and reuses), a cycle that began at 21:54 could
miss the 21:55 cutoff entirely. Not super-likely (60s cycles), but the
engine should call `time.time()` itself for time-critical guards
instead of trusting state-dict inputs for the cutoff.

## P3 findings

### P3-1 — _round_price clamps stop_sell_price floor to 0.01 SEK
`portfolio/grid_tiers.py:49-53` and grid_fisher line 1496.

`_round_price` returns 0.0 for `price <= 0`. `stop_sell_price = stop_price * 0.995` can theoretically go to 0 if stop_price is already 0. Then the floor `max(bid * 0.99, 0.01)` only applies in `eod_market_flat`, not in `rotate_on_buy_fill`. Defensive: floor stop_sell_price to a sensible min (e.g. 0.05 SEK) in rotate path too.

### P3-2 — silver/gold_precompute are 35-line delegation wrappers
`portfolio/silver_precompute.py`, `portfolio/gold_precompute.py`.

Both files only delegate to `metals_precompute`. Could be removed
entirely if no external scripts still reference them. Codebase clutter.

### P3-3 — `_silver_session_low` etc. are unbounded session globals
`data/metals_loop.py:870-1505` area.

Several module-level globals used by `_silver_fast_tick` reset only on
process restart. A multi-day silver session keeps `_silver_session_high`
from days ago. Doesn't affect trading directly but the
`pct_change_from_ref` calculation drifts.

### P3-4 — magic 0.995 stop-sell offset
`portfolio/grid_fisher.py:1496, 1655`.

`stop_sell_price = round(stop_price * 0.995, 2)` — the 0.5% offset
below trigger to ensure fill is hardcoded in two places. Extract to
`GRID_STOP_SELL_OFFSET_PCT = 0.5` in config.

## Cross-cutting observations

1. **Two cooldown universes** — grid_fisher has its own
   per-instrument cooldown + direction-flip cooldown + rapid-cancel
   cooldown, completely independent of `trade_guards.check_trade_guard`
   which the metals_loop fish engine uses. The same instrument can be
   in grid cooldown but NOT in trade_guards cooldown, allowing
   metals_swing_trader to open a position grid_fisher just bailed out of.
   No de-conflict layer exists.

2. **Multiple order-placement code paths with different validation
   surfaces** — `portfolio.avanza_session._place_order` (validated),
   `data.metals_avanza_helpers.place_order` (unvalidated Playwright
   path), `data.metals_loop.emergency_sell` (raw page.evaluate, no
   validation), `portfolio.avanza_control.place_order_no_page` (used by
   fin_snipe_manager). At least four code paths can issue a buy/sell to
   Avanza. Validation drift between them is inevitable; the safer
   pattern is a single helper that everyone routes through.

3. **DRY_RUN flags are checked at the trader level, not at the
   helper level**. `crypto_swing_trader._place_buy` checks `cfg.DRY_RUN`,
   `fish_engine` is decision-only (caller checks), `grid_fisher` has
   `_probe_only` (defaults FALSE — live!). There is no single
   "safe-mode" kill-switch — adding new traders requires duplicating
   the DRY_RUN check pattern, which is exactly the kind of defense
   that quietly degrades.

4. **`get_positions` returns cross-account positions silently**. This
   is the root cause of P0-3 and affects every consumer that doesn't
   manually filter by account_id (grid_fisher, fin_snipe_manager probably
   safer because it explicitly checks account_id on positions it fetches,
   but the grid path is broken). The fix is upstream in
   `portfolio.avanza_session.get_positions`.

5. **No round-trip recording for grid fills**. The grid_fisher logs
   `place_buy`, `fill_buy`, `rotate`, `fill_sell` to the decisions
   JSONL but does not build round-trip P&L records compatible with
   the rest of the system's accuracy/journal pipeline. The
   `session_pnl_sek` counter is in-memory only and resets daily; no
   weekly/monthly aggregation. Layer 2 / dashboard can't see grid P&L
   evolution.

6. **The pension-account exclusion is policy-only, not code-enforced**
   in the grid path. CLAUDE.md / memory feedback_isk_only.md state it
   clearly. `_place_order` enforces ALLOWED_ACCOUNT_IDS but the
   reconcile path does not. A defense-in-depth review of every
   "account_id" reference in the metals subsystem would be worthwhile.

## Files reviewed

- `data/metals_loop.py` (7880 lines — partial review focused on grid
  hook, fast-tick, emergency_sell, momentum_exit)
- `data/metals_execution_engine.py` (559 lines)
- `data/crypto_swing_trader.py` (744 lines — entry/exit + execution
  paths)
- `data/crypto_swing_config.py` (201 lines)
- `data/metals_avanza_helpers.py` (516 lines)
- `data/metals_accuracy_review.py` (240 lines)
- `data/fish_engine.py` (1048 lines)
- `data/fin_fish_config.py` (657 lines)
- `portfolio/metals_ladder.py` (188 lines)
- `portfolio/metals_orderbook.py` (121 lines)
- `portfolio/metals_precompute.py` (1218 lines — partial: API surface
  + refresh state)
- `portfolio/metals_cross_assets.py` (310 lines)
- `portfolio/fin_fish.py` (1457 lines — overview only)
- `portfolio/fin_snipe.py` (277 lines)
- `portfolio/fin_snipe_manager.py` (1765 lines — partial: stop_plan +
  reconcile)
- `portfolio/fish_instrument_finder.py` (213 lines)
- `portfolio/fish_monitor_smart.py` (746 lines — read-only monitor;
  surface review)
- `portfolio/grid_fisher.py` (1970 lines — FULL review)
- `portfolio/grid_fisher_config.py` (185 lines)
- `portfolio/grid_tiers.py` (233 lines)
- `portfolio/iskbets.py` (916 lines — partial: entry/exit/L2 gate)
- `portfolio/oil_grid_signal.py` (182 lines)
- `portfolio/silver_precompute.py` (34 lines)
- `portfolio/gold_precompute.py` (34 lines)
- `portfolio/microstructure.py` (227 lines)
- `portfolio/microstructure_state.py` (236 lines)
- `portfolio/avanza_session.py` (cross-reference for place_order,
  place_stop_loss, get_positions signatures)
