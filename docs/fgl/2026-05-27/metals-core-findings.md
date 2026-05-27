# Metals-Core Adversarial Review — 2026-05-27

## Summary

- **P0**: 3 — barrier-blind stop placement on knockout warrants, EOD-exit disabled with no expiry, idempotency hole in `_with_browser_recovery` order POST retry.
- **P1**: 6 — half-day/holiday EOD ignorance, swing-trader stop-anchor barrier check missing, LONG-only `barrier_distance_pct` formula in `_fetch_warrant_catalog_prices`, account-isolated stop-loss/sell volume not cross-checked, dashboard `market_close_cet` mislabel, `_select_warrant` does not re-check `tradable` at place-time.
- **P2**: 4 — silver fast-tick starvation on long cycles, stale catalog if `page=None` from cron, no cross-process lock between grid_fisher + swing_trader on same warrant, hardcoded `EOD_HOUR_CET=17.0` for `write_context` (still in `hours_remaining`).
- **P3**: 2 — naming inconsistencies, deprecated module still bundled.

**Top 3 themes**: (1) EOD/closing-hour assumptions hardcoded instead of read from `todayClosingTime`; (2) idempotency missing on broker mutation retries (place_order, stop-loss); (3) barrier safety checks computed at catalog-build time but NOT re-checked at stop-loss placement — silver crash + ill-placed stop = instant knockout.

**Biggest single risk**: `_with_browser_recovery` (`portfolio/avanza_session.py:212`) retries the POST after browser death. On a flaky network where the order POST reached Avanza's REST tier successfully but the browser-side response read died, the retry places a **duplicate order**. Combined with `MAX_ORDER_TOTAL_SEK = 50_000` per call, two duplicate buys = 100K SEK exposure on a single signal — far above grid-fisher's intentional 6500 SEK budget and metals_swing_trader's 30% allocation cap.

---

## Critical (P0)

### [P0] `_with_browser_recovery` retries mutating POSTs — duplicate orders on transient browser death
**File:** `portfolio/avanza_session.py:212-232` (consumed by `api_post` at `portfolio/avanza_session.py:294-340`)
**Issue:** `_with_browser_recovery` wraps every `api_post` (including `/_api/trading-critical/rest/order/new` and `/_api/trading/stoploss/new`). On `TargetClosedError` / `is_browser_dead_error`, it tears down Playwright and **retries the same POST once**. Browser-dead errors are detected by exception type/message but cannot distinguish "browser died before POST sent" from "POST landed at Avanza successfully but response read failed". In the latter case Avanza creates the order, the retry creates a second identical order, and the caller logs the *retry's* `orderId` as the only one to track. The first orphan order sits resting until day-end auto-cancel and can fill independently of all internal state.
**Impact:** Up to 2× position size on a single user-intended order placement. With `MAX_ORDER_TOTAL_SEK = 50_000` (`avanza_session.py:602`), worst-case duplicate = 100K SEK warrant exposure — bypasses every per-instrument and per-strategy notional cap. For a stop-loss POST, you could end up with two active broker stops on the same volume; one fires and triggers a phantom short the moment the position is sold.
**Fix:** Restrict retry to GETs (or any idempotent path). For mutating endpoints — `order/new`, `order/delete`, `stoploss/new`, `stoploss/delete` — either don't retry, or after the relaunch read open orders to detect whether the original request actually landed before retrying. Cheapest patch: pass an `idempotent: bool` flag from `api_get` (`True`) and `api_post` (`False`) into `_with_browser_recovery` and gate the retry on it.
**Confidence:** 90

### [P0] Swing-trader stop-loss anchor never validated against warrant barrier
**File:** `data/metals_swing_trader.py:2714-2786` (`_set_stop_loss`)
**Issue:** `_set_stop_loss` computes `trigger_price = round(stop_anchor * (1 - warrant_drop_pct), 2)` where `warrant_drop_pct = sl_pct / 100` (default 30% on 5x). It does NOT consult the warrant's barrier. With a 5x LONG MINI on XAG-USD with `barrier = 30.5` and underlying at 33: a 30% warrant drop is ~6% underlying drop → underlying at 31.02. Barrier sits at 30.5. A second 1.7% underlying tick wipes the warrant entirely AND the stop never fires (stop price would be below barrier-knockout-zeroed value). `MIN_BARRIER_DISTANCE_PCT = 10` is only enforced at WARRANT SELECTION (`_select_warrant` line 2501); once a warrant is in inventory the broker stop placement is barrier-blind. Memory `feedback_mini_stoploss.md` explicitly: *"Never place stop-losses near MINI warrant barriers"*. This file is RED-FLAGGED in MEMORY.md.
**Impact:** Stop placed inside the knockout zone on volatile metals warrants → silent stop-loss bypass on the move that knocks the warrant out → full position lost without exit. Probable knockout event during XAG flash-crash windows (US open, FOMC). Memory documents prior occurrences.
**Fix:** Inside `_set_stop_loss`, after computing `trigger_price`: convert back to implied underlying via leverage, compute `barrier_distance_pct` for the warrant's barrier+direction, and require >= `MIN_BARRIER_DISTANCE_PCT / 2` (5%). If breached, raise the stop towards entry so the implied underlying sits at `barrier × (1 + buffer/100)` for LONG (or `barrier × (1 − buffer/100)` for SHORT). Send Telegram critical alert if no safe stop fits.
**Confidence:** 92

### [P0] EOD swing exits permanently disabled — overnight holds unlimited
**File:** `data/metals_swing_config.py:323` (`EOD_EXIT_MINUTES_BEFORE = 0`)
**Issue:** Set to 0 with comment *"REVERT to 25 after current position closes"* (dated 2026-04-13). With value 0, the consumer at `metals_swing_trader.py:3073` `if minutes_to_close <= EOD_EXIT_MINUTES_BEFORE` only fires *past* close. Effectively the EOD exit never fires. The "current position" mentioned in the comment closed long ago; nobody re-enabled it. Combined with `MAX_HOLD_HOURS = 24` the only time-based exit is the 24h safety net — positions are held overnight by default. User explicitly says (`memory.md`): *"Does NOT want to hold warrants for a full day"*.
**Impact:** Active swing positions carry overnight gap risk on every trade. Silver gap-down on Monday open (Sunday news, Mideast escalation) wipes 30-50% of a 5x leveraged position before any stop can react during EU pre-open. The user's risk profile rejects this.
**Fix:** Set `EOD_EXIT_MINUTES_BEFORE = 25` (matches existing DST-gap-safe comment block). Verify by inspection that `_check_exits` then forces EOD sells when `minutes_to_close <= 25`. Telegram alert when value becomes 0 again (config drift guard).
**Confidence:** 95

---

## Important (P1)

### [P1] EOD detection hardcodes 21:55 — half-day closures (Christmas Eve, Midsummer Eve, New Year's Eve) over-hold
**File:** `portfolio/grid_fisher.py:280-310` (`_EOD_LOCAL_HOUR = 21`, `_EOD_LOCAL_MINUTE = 55`); `data/metals_swing_trader.py:2448` (`close_cet = 21.0 + 55/60`); `data/metals_execution_engine.py:53` (`close_cet = ...replace(hour=21, minute=55,...)`)
**Issue:** Three independent places hardcode 21:55 CET as the warrant market close. Avanza commodity warrants close at 13:00 CET on half-days (Christmas Eve, New Year's Eve, Midsummer Eve, sometimes Maundy Thursday). Although metals_loop's `is_swedish_market_holiday()` calendar treats these as full closures (which suppresses placement entirely — safe), the `metals_swing_trader._check_exits` runs whenever the loop reaches it; if half-day handling slips, the EOD logic would not fire until 21:30 — 8.5 hours past actual close. The `.claude/rules/metals-avanza.md` says explicitly: *"Check API for `todayClosingTime` — do NOT hardcode 21:55."*
**Impact:** Conditional on holiday-calendar drift. Today low-probability because the holiday list treats half-days as full closures. Becomes P0 if anyone removes 12/24 / 6/20 / 12/31 from `swedish_market_holidays()` without also adding a dynamic close-time path.
**Fix:** Add `portfolio.avanza_session.get_session_close_cet(orderbook_id)` that reads `todayClosingTime` from `/_api/market-guide/.../order-depth-level-1`. Replace the three hardcodes. Until then, document the half-day dependency on `is_swedish_market_holiday()` and add a test that asserts every half-day appears in that set.
**Confidence:** 85

### [P1] `_fetch_warrant_catalog_prices` computes `barrier_distance_pct` LONG-only — wrong sign for SHORT certs
**File:** `data/metals_loop.py:4473-4476`
**Issue:** `entry["barrier_distance_pct"] = round((und - barrier) / und * 100, 1)` is correct for LONG warrants. SHORT certs (`direction="SHORT"`, `barrier > underlying`) get a negative number (`(33 - 91.94)/33 = -178.6%`). The result is fed into `cached_warrant_catalog` and consumed by `build_execution_recommendations` → `_summary_filters` (`metals_execution_engine.py:259`) which checks `barrier_distance_pct < MIN_BARRIER_DISTANCE_PCT`. A SHORT cert correctly far from its barrier is silently dropped from execution recommendations because its computed distance is negative. Inversely, a SHORT cert that has already crossed its barrier (`und > barrier`) reports a positive distance and would pass safety filters. The static `WARRANT_CATALOG` today has only one SHORT entry (with `barrier=None`, skipping the formula), so the bug is **latent** — but `metals_swing_config.WARRANT_CATALOG` is editable and `metals_warrant_refresh._barrier_distance_pct` (correct, direction-aware) is in a *different* code path.
**Impact:** If/when a SHORT warrant with a barrier is added to the static catalog (likely first SHORT silver short the user wants to scalp), it would be silently dropped from execution rec — system says "no SHORT candidates". Worse, a knocked-out SHORT (post-barrier) appears as a high-distance candidate.
**Fix:** Replace the formula at `metals_loop.py:4474` with the direction-aware `_barrier_distance_pct` already shipped in `metals_warrant_refresh.py:195-207`. Or import that helper and call it. Add unit test covering one SHORT with `barrier > und`.
**Confidence:** 90

### [P1] `write_context` reports `hours_remaining` using legacy `EOD_HOUR_CET = 17.0` when execution engine unavailable
**File:** `data/metals_loop.py:402` (`EOD_HOUR_CET = 17.0`), `data/metals_loop.py:5954-5956`
**Issue:** Fallback expression `EOD_HOUR_CET + 25/60 - cet_hour()` = `17.42 - cet_hour()`. This is the legacy 17:25 close (EU equity close, NOT Avanza commodity warrants 21:55). When `EXECUTION_ENGINE_AVAILABLE = False` (any import failure in metals_execution_engine), Layer 2 receives `hours_remaining` claiming the market closes at 17:25 — 4.5 hours earlier than reality. Layer 2 then refuses to enter trades that have plenty of intraday runway, or rushes scalps it should hold. Also: `market_close_cet` field hardcoded to `"21:55"` in the same dict (line 5964) directly contradicts the fallback `hours_remaining` math.
**Impact:** Layer 2 mis-estimates remaining trading window during any metals_execution_engine import failure → either skips valid intraday trades or forces premature exits. Silent: there is no log when the fallback path engages.
**Fix:** Replace the fallback expression with one that targets 21:55 CET: `(21.0 + 55/60) - cet_hour()`. Better: lift `hours_to_metals_close` into a module that has no Avanza-side dependencies so the fallback is never needed.
**Confidence:** 88

### [P1] `_select_warrant` does not verify `tradable` status at order placement time
**File:** `data/metals_swing_trader.py:2473-2540`
**Issue:** `_select_warrant` iterates `self.warrant_catalog` (refreshed every 6h per `TTL_HOURS`) and calls `fetch_price(self.page, w["ob_id"], w["api_type"])` to get live bid/ask, leverage, and barrier. It never re-checks `tradable == "BUYABLE_AND_SELLABLE"`. If a warrant gets knocked out, suspended, or delisted between catalog refreshes (worst case: a 6-hour-old catalog entry), the system can try to buy a dead instrument. The bid/ask zero-check is the only line of defense, but Avanza may still return stale last-trade data for an instrument that's now untradable.
**Impact:** Possible BUY POST on a knocked-out warrant. Avanza rejects → courtage spend wasted, swing-trader logs a placement-failure but continues. Lower risk because `place_order` requires `volume * price >= 1000 SEK` and Avanza returns explicit errors for delisted instruments. Realistic but rarely catastrophic.
**Fix:** Inside `_select_warrant`, after `fetch_price`, also call market-guide to read `tradable` and skip non-BUYABLE_AND_SELLABLE candidates. Cache the tradable status in `cached_warrant_catalog` with short TTL (~5 min).
**Confidence:** 80

### [P1] Sell volume + stop-loss volume not cross-checked against position size
**File:** `portfolio/grid_fisher.py:1455-1502` (`rotate_on_buy_fill`); `.claude/rules/metals-avanza.md` (Order Safety)
**Issue:** After a buy fills, the code places a `place_sell_order` for `filled.qty` AND a `place_stop_loss` for `inst.inventory_units`. If `filled.qty` includes the tier's units but the rotation runs while a previous unfilled sell ladder rung is still resting, total resting sell + stop volume can exceed the live position. The `.claude/rules/metals-avanza.md` rule says *"Sell + stop-loss volume must NOT exceed position size. Check existing orders before placing."* — the rotation does not enumerate other resting sells to verify. The `prune_terminal_orders` runs only at end of tick, so within one tick the sell and stop can both be placed against the freshly-rotated tier while a previous tier's sell still rests.
**Impact:** On a multi-fill burst (two tiers fill in same minute), total resting sell volume can exceed cached `inventory_units`. If both fills' rotation runs in the same tick and the prior tier's sell hasn't been pruned, accidental short-sell on the second fill via overlapping sell limits. Avanza will likely reject the second sell with "shortSellingAllowed=False" — but if the stop's `shortSellingAllowed=False` is enforced inconsistently across endpoints (legacy issue), one stop could short-sell.
**Fix:** Before `place_sell_order` and `place_stop_loss`, re-read `get_open_orders` (already done at tick start) and compute `sum(armed_sell_qty) + sum(stop_qty) <= inventory_units`. Reject placement otherwise and log `skip_overcommit`.
**Confidence:** 80

### [P1] Dashboard `/api/golddigger` reports `market_close_cet: "21:30"` — inconsistent with all other paths
**File:** `dashboard/app.py:626`
**Issue:** Every other place in the metals subsystem says 21:55 (metals_loop, metals_execution_engine, metals_swing_trader, grid_fisher). `golddigger` payload returns `"21:30"`. Looks like a copy-paste-after-DST-change leftover.
**Impact:** Layer 2 reading golddigger context interprets 25 min before real close as "EOD now" and would execute force-flat 25 min too early when running on golddigger context.
**Fix:** Update to `"21:55"` and route via the same `MARKET_CLOSE_CET` constant from `metals_execution_engine.py:38`.
**Confidence:** 90

---

## Important (P2)

### [P2] Silver fast-tick starves when main cycle overruns 60s
**File:** `data/metals_loop.py:1023-1080` (`_sleep_for_cycle`)
**Issue:** Sub-loop only runs ticks while `remaining > min_remaining = 5s` (half a fast-tick interval). When the main cycle takes ≥55s of the 60s window (LLM batching, slow Avanza fetch), zero or one fast tick fires. When it overruns 60s entirely, `time.sleep(remaining)` returns immediately and the next cycle starts — zero fast-tick coverage. There is no Telegram alert when fast-tick coverage drops below threshold. The exit-side silver tick (downside velocity alert protecting an active silver position) is therefore *most* likely to be skipped exactly when the loop is busiest (high vol → many LLM invocations → cycle overrun → no fast tick → silver crash → broker stop fires at lag).
**Impact:** On the day silver crashes -8% in 90s during US open (historical pattern), the fast-tick that would surface a -3% / -5% / -7% Telegram alert is most likely to be absent — exactly when the user wants it most.
**Fix:** Detach the fast-tick into a dedicated thread (or asyncio task) that runs independently of the main cycle. Add `_fast_tick_health` heartbeat metric to `health_state.json` so silent skips surface in `/api/health`.
**Confidence:** 82

### [P2] `metals_warrant_refresh.load_catalog_or_fetch(page=None)` returns whatever-is-cached without staleness warning
**File:** `data/metals_warrant_refresh.py:390-392`
**Issue:** When called with `page=None` (e.g., from a cron diagnostic or `__main__` test), the function returns the cached catalog "as-is" with a `WARNING` log only — never raises or returns empty. If the cache is 48 hours old (process crashed for 2 days), callers get a stale catalog and proceed to trade based on stale leverages, barriers, and is-AVA flags. The cache's `_is_stale` returns True past `TTL_HOURS = 6` but the page=None branch ignores that check.
**Impact:** Diagnostic / repair scripts running with `page=None` get unbounded-age catalog data and may report decisions to operators based on it.
**Fix:** When `page is None` and the cache is stale (`_is_stale(cached)`), return `{}` instead of stale data. Or surface the age via the return shape: `{"warrants": ..., "age_hours": N}`.
**Confidence:** 78

### [P2] grid_fisher + metals_swing_trader can both place orders on the same warrant ob_id with no cross-process volume check
**File:** `portfolio/grid_fisher.py:1392-1396` (grid place), `data/metals_swing_trader.py:2625` (swing place); both gated by `avanza_order_lock` for I/O atomicity but not by a logical "this warrant is already held by another strategy" check.
**Issue:** `GRID_ACTIVE_INSTRUMENTS` (`grid_fisher_config.py:140`) lists `XAG-USD: LONG=1650161, SHORT=2286417` and metals_swing_config `BEAR_SILVER_X5_AVA_12` also has `ob_id=2286417`. If the grid fisher arms a SHORT direction on ob 2286417 at the same time the swing trader takes a SHORT signal on XAG-USD via the same cert, both place independently-sized orders against the same Avanza account. Each respects its own per-instrument cap; their *sum* can exceed total cash. The 50K-per-order ceiling in `_place_order` doesn't catch aggregated dual-strategy exposure.
**Impact:** Double-allocation on the *same* warrant ob_id when both strategies trade the same instrument. Real money exposure 2× the intended cap.
**Fix:** Maintain a single shared `position_owner` registry. Before placing, check `owner == self.strategy_name`. Reject if owned by another. The simplest version: hash `(ob_id) → strategy` written to `data/warrant_ownership.json` with atomic I/O.
**Confidence:** 80

### [P2] `EOD_HOUR_CET = 17.0` still in metals_loop scope even though "legacy"
**File:** `data/metals_loop.py:402, 5955, 6256`
**Issue:** Comment says "legacy summary trigger" but the constant is still used: (1) the `hours_remaining` fallback path described above, (2) the `end_of_day_summary` trigger at line 6256. The 17:00 CET trigger predates the 21:55 close. With check ≥10, every 17:00:00-17:03:00 CET produces an "end_of_day_summary" trigger that invokes Layer 2 — even though the real EOD is 4.5 hours later. Layer 2 sees a misleading prompt context.
**Impact:** Wasted Layer 2 invocation at 17:00 daily; potentially wrong "EOD" decision rationale logged to journal. Lower harm because Layer 2 reads its own context to determine real close, but trigger naming itself misleads the operator.
**Fix:** Rename trigger to "eu_equity_close_summary" or remove it. Clean up the dead `EOD_HOUR_CET` from `write_context` fallback.
**Confidence:** 78

---

## Smell (P3)

### [P3] `silver_monitor.py` still bundled despite "DEPRECATED" header
**File:** `data/silver_monitor.py:1-21`
**Issue:** Docstring says merged into metals_loop. File still imports `process_lock`, still has main entry, is 827 lines. Risk of someone restarting it as a competing process (it does have a singleton lock so it would refuse) but logs would look identical to the working subsystem and confuse triage.
**Impact:** Operational confusion.
**Fix:** Move to `archive/` or delete. Replace contents with a `raise SystemExit("Merged into data/metals_loop.py")`.
**Confidence:** 80

### [P3] Three independent EOD constants risk drift
**File:** `data/metals_execution_engine.py:38` (`MARKET_CLOSE_CET = "21:55"`), `data/metals_loop.py:5964` (literal "21:55"), `portfolio/grid_fisher.py:280-281` (numeric 21/55), `data/metals_swing_trader.py:2448` (float 21+55/60)
**Issue:** Four ways of expressing the same value across the metals subsystem. Inevitably drift on the next DST/policy change.
**Fix:** Define `MARKET_CLOSE_CET_LOCAL = (21, 55)` once (e.g., in `portfolio/market_timing.py`) and import everywhere.
**Confidence:** 78
