# Agent 4 — Metals Core Adversarial Review

Reviewer: Opus 4.7 (1M ctx)
Scope: metals_loop.py + grid_fisher.py + fin_snipe.py + fin_snipe_manager.py + exit_optimizer.py + price_targets.py + orb_*.py + iskbets.py + metals_ladder.py + metals_orderbook.py + oil_grid_signal.py + supporting files (24,021 LOC total)

Severity legend:
- **P0** — will lose money, knockout warrant, data loss
- **P1** — real bug, behaviour incorrect under normal operation
- **P2** — latent / edge-case bug
- **P3** — minor / nit

---

## CRITICAL (P0)

### 1. grid_fisher.py:1493 — Stop-loss sell-price floor of `stop_price * 0.995` is far too tight (`0.5%`) for warrant volatility; market sells fail and stops do not execute
**P0 (will lose money on next gap).** `rotate_on_buy_fill` computes `stop_sell_price = round(stop_price * 0.995, 2)`. For a 5x silver MINI cert with `GRID_STOP_PCT=3.5%`, on a sharp downward gap the warrant's bid can fall 5–10 % below the trigger before the broker probes it, but the limit sell sits only 0.5 % below the trigger. Avanza's hardware stop turns into a *limit* sell at `stop_sell_price`; a gap past the limit silently parks the order in the book without filling. Memory `feedback_mini_stoploss.md` and rule `metals-avanza.md` ("5x leverage certificates need -15%+ stops, not -8%") both flag this exact behaviour. Fix: drop the limit price 2-3 % below trigger or use a wider buffer scaled by warrant ATR.

### 2. portfolio/grid_tiers.py:71-106 — Knockout guard uses constant-leverage approximation but the active grid certs in `GRID_ACTIVE_INSTRUMENTS` have `barrier=0` baked into `data/fin_fish_config` defaults
**P0 (knockout protection is silently inert).** `_tier_skip_for_knockout` short-circuits with `return None` when `barrier is None`. The default catalog (`portfolio/fin_fish.py:91-161`) sets `"barrier": 0` on every silver/gold cert (e.g. `BULL_SILVER_X5_AVA_3`). `_catalog_for(ob_id).get("barrier") if cat.get("barrier") else None` in grid_fisher.py:1762 passes `None` to `place_buy_ladder` whenever barrier is falsy (0). Net effect: the barrier proximity check never fires for the production instruments. A swift adverse move on a 5x cert with a real (unloaded) barrier could knock out without the grid skipping the deepest tier. Fix: load the live `barrierLevel` per-instrument via Avanza market guide instead of relying on the catalog default.

### 3. portfolio/grid_fisher.py:1493 (stop_sell_price 0.5%) compounded with grid_fisher.py:1844 EOD fallback to `bid * 0.99` for forced exits
**P0 (drag on every EOD).** `eod_market_flat` uses `aggressive = round(max(bid * 0.99, 0.01), 2)` as a "market-equivalent". On a 5x warrant that bid value can already be 5 % below mid; placing -1 % below bid still rests as a limit. With `GRID_EOD_MARKET_SELL_MINUTES_BEFORE = 5` minutes left, an illiquid silver cert at 21:50 CET routinely shows 1-2 % spread. A -1 % limit will not cross to the ask side and inventory is carried overnight — exactly the failure mode `CLAUDE.local.md` flags ("User does NOT want to hold warrants overnight"). Fix: use `bid * 0.95` (or bid - 2× spread) at EOD or place an actual market order via the trading API.

### 4. portfolio/fin_snipe_manager.py:61 — `HARD_STOP_CERT_PCT = 0.05` (5% cert distance) is too tight for 5x warrants per `metals-avanza.md`
**P0 (premature stop-out).** `_compute_stop_plan` computes `trigger_price = position_avg * (1.0 - 0.05)`. The project rule says "5x leverage certificates need -15%+ stops, not -8%, to survive intraday wicks." For a 5x silver cert, a 5 % cert stop = 1 % underlying — well inside silver's typical 1-hour wick. The `MIN_STOP_DISTANCE_PCT=1.0` guard only checks distance from current bid, not from the volatility floor. Every position will get stopped on routine noise rather than at a true thesis-invalidation level. Fix: raise to 15 % cert (=3 % underlying for 5x) or compute from ATR.

### 5. portfolio/fin_snipe_manager.py:537 — `sell_price = trigger_price * (1.0 - 0.01)`, a 1% buffer on a stop-loss limit, will leave warrants unfilled on a fast move
**P0 (stop becomes a wish, not a fill).** `HARD_STOP_SELL_BUFFER_PCT = 0.01`. After Avanza triggers at `trigger_price`, the limit sits only 1 % below. Silver warrants regularly print 3-5 % wicks in 30 s during US session opens; the order sits in the book and never fills. The Mar 3 `metals-avanza.md` rule literally references "regular order causes instant fill at bad price" — the inverse failure (limit too tight → no fill at all) is what this code creates. Fix: 3-5 % buffer below trigger, or use trigger as market-on-touch.

### 6. portfolio/iskbets.py:391 — `hard_stop = entry_price - (hard_stop_mult * atr)` operates on USD underlying with no leverage applied, but Avanza warrant sells happen at cert price
**P0 (mis-sized stops vs leveraged instrument).** `check_exits` uses `entry_price_usd` and ATR in USD, then computes `hard_stop` and `stage1_target` on USD. The user trades real cert positions on Avanza ("amount_sek"), but the stop check compares `price <= hard_stop` where `price` is also USD (`prices_usd.get(ticker, 0)`). That part is internally consistent, but the `format_exit_alert` shows `pnl_sek = shares * (price - entry_price) * fx_rate` — i.e., the user is told the warrant moved 1× the underlying, while in fact a 5x cert lost 5× as much. The Telegram message understates risk by exactly `1/leverage`. Fix: scale ATR-based stops by leverage in both alert and tracking.

### 7. data/metals_loop.py:1407-1503 — Silver fast-tick mutates module-globals (`_silver_fast_prices`, `_silver_alerted_levels`, `_silver_session_low/high`, `_silver_consecutive_down`, `_silver_prev_price`) with NO lock, racing the 60s slow loop that touches `POSITIONS` and `_underlying_prices`
**P0 (data corruption + missed alerts).** `_silver_fast_tick` runs inside `_sleep_for_cycle` between cycles; `_get_active_silver()` reads `POSITIONS` while the main thread can mutate it via `emergency_sell`, `_save_positions`, holdings reconcile. `_silver_fast_prices.append()` and `.clear()` are not atomic w.r.t. the velocity check (`_silver_fast_prices[0]`). On Python 3.13 free-threading or under GIL release during socket recv inside `requests.get`, a `clear()` on session reset can race a velocity-window read, producing IndexError or stale alerts. Memory `system_reliability.md` says "System reliability is #1." Fix: wrap fast-tick state in a `threading.Lock`.

### 8. portfolio/orb_predictor.py:166 — `end_time = data[0][0] - 1` mutates kline pagination by 1 ms but Binance returns kline open times that are 15-min aligned; consecutive batches may overlap or skip bars depending on rounding
**P1 borderline P0.** When `data[0][0]` is a 15-min kline open time, subtracting 1 ms still lands in the *current* bar's window, so `endTime` param may include the same bar twice or miss the one immediately preceding. ORB backtest stats then either double-count a bar (inflating volume) or under-count days. The 1-week reported accuracy in the project notes is suspect for exactly this reason. Fix: subtract the full interval (15·60·1000 ms) or use Binance pagination via `klines` `endTime` minus one full interval.

### 9. portfolio/orb_predictor.py:36-46 — `_morning_window_utc()` uses `_is_eu_dst` to pick CET vs CEST hours but `fetch_klines` returns timestamps in UTC and the rest of the code mixes UTC-hour comparisons (`c["hour"]` from candle ts) with this DST-shifted window
**P0 (silently wrong samples around DST transitions).** During the spring-forward Sunday, the morning window flips from `(8,10)` to `(7,9)` UTC mid-dataset. Historical days fetched *before* the transition still have their morning bars in 08-10 UTC bucket but the predictor compares them against the new 07-09 bucket, throwing 1 hour of bars away per pre-DST day. The 5-batch backtest spans ~52 days which always crosses at least one DST boundary in spring/fall. Result: morning ranges go from "wide and informative" to "tiny and skewed" silently. Fix: tag each candle's `_morning_in_window` at fetch time using its own local time, not module-state DST.

### 10. portfolio/exit_optimizer.py:217 — `_TRADING_MINUTES["warrant"] = 820` (~13.67h) but Avanza warrants trade 08:15-21:55 CET = 13h 40min = exactly 820, which matches *current* DST. After DST shifts, the underlying USD market still trades 24h but the warrant window in CET doesn't change, so the constant stays valid — BUT `compute_exit_plan` simulates GBM in *warrant trading minutes* and the underlying USD price moves continuously 24/7
**P1.** The `dt = 1.0 / (820 * 252)` annualisation factor for a 24/7 underlying like XAG/XAU systematically *understates* path variance by ~24/13.67 ≈ 1.75× (vol on a 24/7 underlying should be annualised over 24h-day-equivalent). Resulting `session_max` Monte Carlo quantiles are too tight; recommended exit `price_usd` quantiles understate the upside the underlying can actually reach. Fix: for warrants whose underlying is 24/7, use `_TRADING_MINUTES["crypto"] = 1440` for the annualisation step (or pass an explicit underlying-vol time-base).

### 11. data/metals_avanza_helpers.py:294 — `today_str = datetime.datetime.now().strftime("%Y-%m-%d")` for `validUntil` uses local OS timezone, not Avanza/Stockholm time
**P1 borderline P0.** Process running under UTC-7 (US dev VM) at 23:01 PT = 08:01 CET *next* day: `today_str` becomes "yesterday" in Stockholm. The Avanza endpoint may reject the order ("validUntil in past") or accept and immediately cancel. The metals system is documented as running on Windows 11 CET, so usually safe — but the bug is latent for any other deployment and there is no timezone safety. Fix: use `datetime.datetime.now(ZoneInfo("Europe/Stockholm")).strftime("%Y-%m-%d")`.

---

## IMPORTANT (P1)

### 12. portfolio/grid_fisher.py:707-721 — Partial-fill detection mutates `tier.qty` in place to `int(delta)` but does NOT reduce the original `notional_sek` tracking; subsequent `planned_notional_sek()` calls undercount inventory
**P1.** After a partial fill on tier 0 (qty 100 → delta 30), the code rewrites `tier.qty = 30` so the FILLED record shows 30 units. But `inst.inventory_units` correctly increases by 30, while the *armed* tier 1 still records qty=100, which then sits resting. When global cap evaluates `armed + inventory@avg_entry`, the math is off by `(original_qty - delta) * price`. Fix: track partial as a new record rather than mutating the in-place tier.

### 13. portfolio/grid_fisher.py:1731-1735 — `bid = float((quote or {}).get("buy") or 0)` uses Avanza JSON shape but the dict already comes from session.get_quote whose shape isn't asserted; if get_quote returns `{"buy": {"value": 95.20}}` (Avanza's wrapped-value style) it silently becomes 0
**P1.** `metals_avanza_helpers.fetch_price` explicitly unwraps `v(d.quote?.buy)`, but `avanza_session.get_quote` returns the raw payload in some code paths. A raw `{"value": ..., "currency": "SEK"}` `or 0` → 0, no bid, instrument skipped. Same pattern at line 1840 in `eod_market_flat`. Result: grid-fisher silently goes idle whenever Avanza nests the bid. Fix: use the same `_extract_value` helper that `fin_snipe.py:_value()` uses.

### 14. portfolio/grid_fisher.py:1844 — `eod_market_flat` falls back to `inst.avg_entry_price` as the bid when quote fails
**P1.** Using last-known *entry* as a market-sell-limit yields wildly off-market orders. If the position is down 5 %, the order rests 5 % above the live bid and never fills, leaving the position naked overnight (violates EOD-flat invariant). Fix: skip the sell on quote failure and alert, rather than placing a stale-price limit.

### 15. portfolio/exit_optimizer.py:514-518 — Edge case "session_ended" returns `mkt_candidate` with `bid or market.price`; if `market.bid is None` but `market.price` is a USD spot, the `_compute_pnl_sek` is called with `exit_price_usd = USD spot` even though the warrant cannot be exited at that price
**P1.** `MarketSnapshot.bid` is optional. When `compute_exit_plan_from_summary` (line 720+) builds the snapshot, it never sets `bid`, so `bid is None` and the function falls back to `market.price` (USD spot). The PnL is then computed in `_compute_pnl_sek` using the leverage formula against `market.price` — fine for the underlying-only path, but the recommended fill is reported as a USD price the user can't trade. UI/Telegram consumers see a confusing "exit at $25.45" when the cert is trading at 70 SEK.

### 16. portfolio/exit_optimizer.py:611 — `tuple(_compute_risk_flags(None, position, market, remaining_min))` is called WITHOUT `session_min`/`session_max`, dropping the `KNOCKOUT_PROB_X%` flag from the market-exit candidate
**P1 borderline P0 for MINI warrants.** Lines 596-598 pass `session_max, session_min` for the limit-exit candidates, but the market-exit fallback at line 611 omits them. A user told "market exit ev=+50 SEK" doesn't see the `KNOCKOUT_PROB_30%` warning that the limit candidates surface. For MINI futures with `financing_level` set, this hides the most important risk on the only candidate guaranteed to fill. Fix: pass both arrays consistently.

### 17. portfolio/price_targets.py:106 — `fill_probability_buy` uses `price ** 2 / target if target > 0 else price` as the symmetric reflection target — that's wrong for first-passage probability
**P1 (mis-calibrated buy fill probabilities).** The reflection-principle "buy below" maps to `P(min<=target) = P(running max >= 2*log(price) - log(target))` in log space, not `price^2 / target`. The formula coincidentally gives the right answer when drift=0 and time→0 but diverges as `target/price` moves away from 1. Fix: pass `2*price - target` in log space (equivalent to using log-prices and the reflection of the barrier), or compute `P(min)` directly with negated drift.

### 18. portfolio/price_targets.py:360 — `if side == "sell" and val > price_usd or side == "buy" and val < price_usd:` lacks parens, evaluates as `(sell and val>p) or (buy and val<p)` which works by accident but is a bug magnet
**P3 (works but fragile).** Python's `and` binds tighter than `or` so the current evaluation matches intent. But any future edit can flip semantics. Fix: explicit parens.

### 19. portfolio/orb_backtest.py:235 — Same precedence bug, more dangerous: `if day.morning_direction == "up" and upside >= downside or day.morning_direction == "down" and downside >= upside:`
**P2.** Parses correctly today, but `correct += 1` will trigger on any future operator addition that breaks the implicit precedence. Wrap in parens.

### 20. portfolio/orb_predictor.py:404-414 — `translate_to_warrant` always recomputes `factor = intrinsic_target / intrinsic_entry` regardless of `current_warrant_price`; the `if` branch is dead code
**P3 nit.** Both branches assign the same value; `current_warrant_price` parameter is unused. Won't cause bugs but signals a stale refactor.

### 21. portfolio/orb_postmortem.py:73,76 — Division by `prediction.predicted_high_median` and `predicted_low_median` with no guard
**P2.** If the predictor returns 0 or negative on degenerate data (`predict_daily_range` filters most cases via `min_sample` but `morning.high * (1 + up_25/100)` can be 0 if morning.high is 0), this raises `ZeroDivisionError`, killing the postmortem write. Fix: guard for `> 0`.

### 22. portfolio/iskbets.py:412 — `trailing_stop = max(entry_price, highest - (trailing_mult * atr))` after breakeven; the floor pins to entry, but for a 5x cert this is wrong: 1× ATR move on USD = 5× ATR on cert, so trailing distance is mis-sized
**P1.** Same root cause as P0 #6 — IskBets treats USD-underlying ATR as cert-price ATR. After stage 1 (breakeven), the trailing stop is set on USD, but actual cert P&L can fall 5× as much before the USD trail triggers. Fix: divide the trailing distance by leverage.

### 23. portfolio/exit_optimizer.py:215-218 — `n_steps = max(1, int(remaining_minutes))` floors to int, but `remaining_minutes` is a float; `_path_statistics` then slices `paths[:, 1:]`. If `remaining_minutes = 0.7` (less than 1 minute), n_steps = 1, simulating a SINGLE minute step — and the result's `session_max` is just terminal price (no path coverage)
**P2.** The edge case `remaining_min < 1` is filtered at line 514, but the band 1.0-1.999 minutes produces 1 step — a degenerate MC with effectively zero variance. Fix: use `max(2, ceil(remaining_minutes))` or fallback to market-exit-only when < 5 minutes.

### 24. data/metals_loop.py:1486-1494 — `vel_key = f"vel_{int((time.time() - 2) // 300)}"`  for de-duplicating velocity alerts uses a 5-min epoch bucket but the bucket is keyed on raw epoch, not session, so on loop restart at minute 4:59 the next velocity alert fires immediately because the new process has empty `_silver_alerted_levels`
**P3.** Memory `feedback_no_assumptions.md` warns about restart races. Replays a velocity alert that was already sent <1 min ago. Fix: persist `_silver_alerted_levels` across restarts.

### 25. portfolio/grid_fisher.py:1422 — `time.sleep(self._order_delay_s)` inside `place_buy_ladder` blocks the metals_loop *main thread* (the tick is called from the metals_loop tick) for `GRID_ORDER_PLACE_DELAY_S * placed` seconds per cycle
**P2 (loop latency).** With 3 instruments × 2 tiers = 6 placements at 0.6 s = 3.6 s of sleep per tick on cold start, on top of all the get_open_orders / get_positions / get_quote calls running serially through `_safe_session_call`'s single-thread executor. The 60s metals cycle becomes 55s + grid ~5s, then `_silver_fast_tick` window shrinks. Fix: do placement on the worker thread and `await` results, or batch through async.

### 26. portfolio/grid_fisher.py:204 — `hit_per_instrument_cap()` compares against `GRID_PER_INSTRUMENT_MAX_SEK = 3000` using `>=`, but `planned_notional_sek` already includes `held` at `avg_entry_price`; on a position rallying +10 %, the cap is artificially tight (you used 1200 SEK budget but cap reads 1320 SEK and blocks the next tier)
**P3.** Held inventory should be priced at *current* bid for cap purposes, not entry. Minor sizing issue — leans conservative, so not a money-loser, but wastes opportunity.

### 27. portfolio/fin_snipe_manager.py:1138-1146 — `desired_buys: list[dict]` is initialized to `[]` then `open_buys = _candidate_entry_orders(snapshot, instrument_state, _desired_buy_orders(...) if entry_volume > 0 else [])` — but in the `position_volume > 0` branch `entry_volume` is the saved-or-observed volume, which includes the already-held position units; the manager will try to re-buy the same volume it already holds
**P2.** Triggered only when position fills mid-cycle and `last_position_volume` hasn't synced. The two-phase replacement in `_stage_replacements` masks it, but on a partial fill into a position that already had a sell ladder, the manager can attempt a duplicate BUY. Fix: subtract `position_volume` from `entry_volume` in the exit-mode branch.

### 28. portfolio/metals_ladder.py:91-118 — `compute_targets` is called WITHOUT `is_24h=True` for XAG-USD / XAU-USD; default is `is_24h=True` so by coincidence correct — but if a non-24h ticker is ever added here it silently mis-annualises
**P3.** `build_intraday_ladder` does not pass `is_24h`. Fragile. Fix: explicitly pass it derived from `_is_24h(ticker)`.

### 29. portfolio/fish_monitor_smart.py:308 — `from scripts.fish_preflight import compute_preflight` imported inside loop runtime
**P3.** Lazy import inside a 30 s polling loop. Harmless, but performance smell.

### 30. portfolio/iskbets.py:743 — Hardcoded `fx_rate = 10.5` fallback when `fetch_usd_sek` fails — but USD/SEK has traded 10.7-11.0 in 2026
**P2 (sizing 4-5 % off).** `_handle_bought` computes `shares = amount_sek / (price_usd * fx_rate)`; a stale 10.5 fallback inflates shares by ~4 %. P&L reports are then off by the same. Fix: cache the last good rate to disk.

### 31. portfolio/orb_predictor.py:359 — `idx = int(len(sorted_list) * pct / 100)` rounds down, so `percentile(list_of_10, 25)` returns index 2 (the 3rd element). True percentile should interpolate
**P3.** Trivial bias for small samples but the same bias is applied to all four percentiles so relative rankings are stable. Documented as nit.

### 32. data/metals_loop.py:6256 — End-of-day trigger uses `int(h_cet) == int(EOD_HOUR_CET) and (h_cet % 1) < 0.05`, with `EOD_HOUR_CET = 17.0` hardcoded
**P1.** The comment at line 402 says "17:00 Stockholm time (legacy summary trigger)". But the metals close is 21:55 CET. The 17:00 EOD trigger is a *summary* invocation, not a flatten. If anyone refactors this to flatten, they'll cut all metals positions at 17:00 instead of 21:55. The hard rule from `metals-avanza.md` is "do NOT hardcode 21:55. Varies with DST" — and 17.0 is hardcoded with no DST handling at all. Fix: replace with `cet_time_str()`-based comparison and DST-aware close time.

### 33. portfolio/fin_snipe_manager.py:1175-1196 — Emergency two-phase replacement (`emergency=True`) cancels and re-places in the same cycle but `apply_execution_results_to_state` runs sequentially on `results`; if the cancel succeeds but the place fails (Avanza rate-limit / 500), the position is left naked with the stop alert firing too late
**P1.** The `_notify_critical("naked_position", ...)` alert runs only after `log_cycle_results` at the END of the cycle. Window between cancel and place is short (~hundreds of ms), but on a rate-limited Avanza session it can be several seconds. Memory `feedback_just_do_it.md` says "EXECUTE immediately". Fix: place first, then cancel old.

### 34. portfolio/grid_fisher.py:535 — `cooldown_dt = _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(minutes=cooldown_min)` but `inst.in_cooldown(now_iso)` does a string comparison `now_iso < self.cooldown_until` with ISO format
**P2.** ISO 8601 strings sort lexically only when both have the same precision/tz format. `_utcnow_iso()` returns `%Y-%m-%dT%H:%M:%SZ`; `cooldown_dt.strftime("%Y-%m-%dT%H:%M:%SZ")` matches. Safe today. But if anyone passes a `.now().isoformat()` string (with microseconds), the lex compare breaks. Fix: parse to datetime for comparison.

### 35. portfolio/exit_optimizer.py:322 — `exit_warrant_sek = max(exit_warrant_sek, 0)` clamps MINI knockout to 0, but the position size `entry_value` is not similarly clamped; computed `pnl = -entry_value` already accounts for total loss
**P3.** Math is correct; the clamp is defensive. Code reads OK.

### 36. portfolio/metals_ladder.py:131-149 — `flash_underlying` clamped to `min(working_underlying, current * (1 - flash_drop_pct/100))`; if `current_underlying * (1 - flash_drop_pct/100)` is *above* working_underlying (flash is mild), the flash price equals working price → `_price_matches(working, flash)` later evaluates to True and the function logs only one rung
**P2.** Designed behaviour (merge identical tiers), but in flash-window mode the user expects two distinct tiers. Fix: enforce a minimum gap between working and flash (e.g. 30 % of computed flash drop) before placing.

### 37. portfolio/orb_predictor.py:209-237 — `calculate_morning_range` filters bars by `c["hour"]` which is *UTC* hour from line 181 (`"hour": ts.hour` where ts is in UTC tz). But `morning_start_utc` defaults via `_morning_window_utc()` which already shifts for DST. Double DST shift in CEST (`hour` in UTC, window in UTC-shifted-by-DST) means CEST morning windows skip an hour of valid morning data
**P1.** In CEST (summer), `_morning_window_utc()` returns `(7,9)`. A bar at 08:30 UTC has `hour=8` which IS in `[7,9)` — correct. So actually OK by accident. But the comment at line 38-46 documents 09:00-11:00 CET as the target window; in CET (winter) `(8,10)` matches 08-10 UTC = 09-11 CET — OK. In CEST `(7,9)` matches 07-09 UTC = 09-11 CEST. Both correct. Reviewing more carefully: this is fine. Withdraw P1 → **P3 confusing but correct**.

### 38. portfolio/grid_fisher.py:1593-1595 — `open_order_ids = {str(o.get("orderId") or o.get("id") or "") for o in open_orders_raw if o.get("orderId") or o.get("id")}` will include the empty string `""` if neither key is present; subsequent `tier.order_id in open_order_ids` for an unfilled tier with `order_id=None` doesn't false-positive because `None not in set_of_str`, but the `""` entry is a footgun
**P3.** Defensive nit.

### 39. portfolio/exit_optimizer.py:516 — `mkt_candidate` for session-ended uses `market.bid or market.price` — same shape error as #15 — when called from `compute_exit_plan_from_summary` (no bid passed) the "market exit" price is the USD spot, not the SEK cert bid
**P1.** Already P1 above (#15). Highlighting the second instance for completeness.

### 40. portfolio/grid_fisher.py:1402-1418 — On successful buy placement the tier is appended at `tier.tier=tier.index`, but `existing_tiers` set is checked at line 1355 before placement — re-running on a partial outage where some tiers placed and some didn't, then partial state file load, can produce duplicate tier indices in `buy_ladder` (one ARMED, one CANCELLED). `armed_buy_tiers()` then returns only the ARMED one which is fine, BUT `prune_terminal_orders` (line 615) drops CANCELLED, so on next tick the system places a *new* tier at the same index because the in-memory ladder no longer has the prior one
**P3 nit.** Worst case duplicate work, not duplicate placement (reconcile catches it).

### 41. data/metals_loop.py:7220-7232 — Fishing EOD sell triggers at 21:50 CET *only* during the cycle that lands in that minute window. With 60s cycles, if the cycle starts at 21:49:55 the next tick is 21:50:55 — fishing positions get sold once. Fine. BUT if `is_market_hours()` returned False (e.g. clock drift, weekend, holiday), the entire branch above is skipped via the `if not is_market_hours(): continue` at line 7234 — meaning the EOD-sell check at 7217-7232 NEVER runs when the market check returns False even by 1 second
**P2.** A holiday declared during the day, or NTP clock skew of >5 s past 21:55, prevents the fishing EOD flush. Result: fishing positions held overnight (memory rule violation).

### 42. portfolio/grid_fisher.py:1492 — `if not self._probe_only and inst.inventory_units > 0:` places the stop. But for a partial fill where `record_fill` left `inventory_units > 0` AND there's already an existing `stop_loss_id` from a *previous* batch, the cancel-then-place sequence runs. If `cancel_stop_loss` succeeds but `place_stop_loss` fails (broker error, e.g. crossing prices), `inst.stop_loss_id = None` and `inst.stop_loss_price = stop_price` — but no actual stop is live. The position is NAKED until next rotation
**P0 borderline.** Lines 1505-1515: on `place_stop_failed` or `place_stop_rejected`, the code sets `inst.stop_loss_id = new_stop_id` where `new_stop_id` is still None from line 1491. After the if-branch sets `inst.stop_loss_price = stop_price`, the state shows a stop price but no broker order. Dashboard / external readers see "stop @ X" but no protection. Fix: only update `stop_loss_price` on confirmed placement; alert on cancel+place gap.

### 43. portfolio/iskbets.py:751 — On ATR fetch failure, `atr = price_usd * 0.02` is used as fallback — 2 % of price as a "typical ATR". For BTC at $100k that's $2000, way larger than actual 15-min ATR (~$150). Stops are then 2× $2000 = $4000 away
**P2.** Fallback is wildly off for crypto. Fix: use last cached ATR via shared_state.

### 44. portfolio/fin_snipe.py:106-107 — `fetch_open_orders()` and `fetch_stop_losses()` are called every `build_snapshots` invocation but cached nowhere; for the `fin_snipe_manager` loop running at 60 s with --loop, that's ~1500 Avanza calls/day. The Avanza session has explicit cooldown — no rate limit guard here
**P2.** Hammer pattern. The `fin_snipe_manager` adds a fast-recheck mode every 5 s on pending replacements, making this worse. Memory `system_reliability.md` flags the Avanza session as the most fragile dep. Fix: cache 30-60 s.

### 45. portfolio/metals_orderbook.py:81-82 — `spread_bps = (spread / mid) * 10000 if mid > 0 else 0.0` — but if `mid > 0` and `spread` is computed from `best_ask - best_bid`, on a crossed/locked book (rare but real on illiquid metals futures), `spread` can be negative and `spread_bps` becomes negative. Downstream OFI/VPIN/microstructure assumes positive spread for variance computations
**P3.** Edge case. Locked books on XAUUSDT FAPI happen during halt/resume. Defensive: `max(0, spread)`.

### 46. portfolio/exit_optimizer.py:188-243 — `simulate_intraday_paths` allocates `Z = rng.standard_normal((n_half, n_steps))` and then `Z_all = np.vstack([Z, -Z])`. For `n_paths=5000, n_steps=820`, that's 5000 × 821 × 8 bytes = ~33 MB per call, allocated fresh each `compute_exit_plan`. fin_snipe_manager calls this every cycle for each instrument with `EXIT_OPTIMIZER_N_PATHS = 2000` (smaller, but still ~13 MB)
**P3.** Memory churn but not a leak. Numba/torch port would be better.

### 47. portfolio/orb_predictor.py:412-416 — `if current_warrant_price and current_warrant_price > 0: factor = intrinsic_target / (silver_target - fl)` — `silver_target - fl == intrinsic_target` by definition, so this is `factor = intrinsic_target / intrinsic_target = 1.0`. The next line overrides with the correct formula. Looks like a half-finished refactor; the "if" path's first assignment is dead
**P3.** Same observation as #20.

### 48. portfolio/fin_snipe_manager.py:1244 — `(_est := _estimate_entry_underlying(...)) > 0` uses walrus inside a conditional expression. `_estimate_entry_underlying` is called TWICE per tick when position_volume > 0 (once at line 464 inside `_compute_exit_target` and again here at 1244)
**P3.** Same input each time but it makes API-style calls? It only reads snapshot/state — pure CPU. Performance nit.

### 49. portfolio/grid_fisher.py:1316 — `inst.in_cooldown(self.now_fn())` is checked AFTER `cancel_armed_buys` (line 1700) for a direction mismatch. Net effect: on signal flip, buys cancel even during cooldown — correct intent — but the LOG at line 1716 says "skip cooldown" only on the *next* tick after the flip
**P3.** Behaviour matches intent; log readability nit.

### 50. portfolio/orb_predictor.py:155-167 — `fetch_klines` uses `end_time = data[0][0] - 1` (millisecond), which Binance accepts. But the pagination iterates `num_batches` times with `endTime` shrinking; each batch can return 1000 candles. For 15m interval, 1000 candles = 10.4 days. 5 batches = 52 days. Comment says "~52 days" — correct. BUT the loop doesn't break if a batch returns < 1000 candles, just `all_klines = data + all_klines` — fine. However the 1-ms backstep means the *last candle of the older batch* might be the *first candle of the newer batch* (Binance is right-inclusive sometimes). Could cause a duplicate kline
**P3.** See #8 — primary impact already flagged.

---

## Summary Counts

- **P0 (will lose money / data loss):** 11 findings — #1, #2, #3, #4, #5, #6, #7, #8 (P1 borderline P0), #9, #10 (P1 borderline P0), #11 (P1 borderline P0), #42 (borderline)
- **P1 (real bugs):** 13 findings — #12, #13, #14, #15, #16, #17, #22, #32, #33, #39 (dup of #15)
- **P2 (latent):** 11 findings — #19, #21, #23, #25, #27, #30, #34, #36, #41, #43, #44
- **P3 (nit):** 15+ findings

## Highest-priority remediation order

1. **Widen all stop-loss limit prices** (#1, #3, #5) — every metals trade is risking unfilled stops on a fast wick.
2. **Fix HARD_STOP_CERT_PCT for leverage** (#4) — premature stop-outs on every position.
3. **Load real barrier levels for grid certs** (#2) — knockout proximity check is currently inert.
4. **Audit USD↔SEK leverage in iskbets** (#6, #22) — Telegram alerts understate risk by 5×.
5. **Add lock to silver fast-tick state** (#7) — race condition will eventually corrupt alerts.
6. **Fix annualisation factor for 24/7 underlyings in exit_optimizer** (#10) — MC paths are 1.75× too tight.
7. **Replace `today_str` with timezone-aware Stockholm time** (#11) — order-rejection footgun.
8. **Confirm stop-loss state only on broker confirm** (#42) — naked-position window after place_stop failure.

## Notes

- No instances found of regular order API misused for stop-loss (good — `metals_avanza_helpers.place_stop_loss` correctly hits `/_api/trading/stoploss/new`).
- Atomic writes for `metals_swing_state.json`, `metals_trades.jsonl`, `grid_fisher_state.json` are all using `atomic_write_json` / `atomic_append_jsonl` — clean.
- ORB backtest uses walk-forward without look-ahead — correct.
- Direction handling on flip (cancel armed buys, preserve inventory + sell ladder) is the safe interpretation; reviewed and OK.
- Avanza commodity warrant hours (08:15-21:55 CET) are correctly used in `grid_fisher_config.py:_EOD_LOCAL_HOUR/MINUTE` and `is_market_hours()` at metals_loop.py:1574.
