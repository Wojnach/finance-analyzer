# Adversarial Code Review — Metals Core Subsystem
**Date:** 2026-05-26
**Reviewer:** Claude Opus 4.7 (1M context, adversarial mode)
**Scope:** `data/metals_loop.py`, `data/metals_swing_trader.py`, `data/metals_swing_config.py`, `data/metals_execution_engine.py`, `data/metals_warrant_refresh.py`, `data/metals_history_fetch.py`, `data/metals_signal_tracker.py`, `data/metals_accuracy_review.py`, `data/metals_risk.py`, `data/metals_avanza_helpers.py`, `data/metals_llm.py`, `data/metals_shared.py`, `portfolio/metals_orderbook.py`, `portfolio/metals_cross_assets.py`, `portfolio/metals_ladder.py`, `portfolio/metals_precompute.py`, `data/silver_monitor.py`, `portfolio/silver_precompute.py`

## TOP 3 (money-loss potential)

1. **Silver fast-tick is BLIND to swing-managed positions** — `_has_active_silver()` only reads legacy `POSITIONS` (data/metals_positions_state.json). All threshold and 3-min velocity alerts (−3% / −5% / −7% / −10% / −12.5%) silently no-op when silver is held by the swing trader (data/metals_swing_state.json), which is the canonical post-April-2026 entry path. A 10% silver crash during US open on a swing-managed MINI L SILVER position produces ZERO Telegram alerts and ZERO accelerated Claude analysis.

2. **`data/metals_swing_trader.py:2760` legacy 1% sell buffer on the swing trader's stop-loss** — `sell_price = round(trigger_price * 0.99, 2)`. Same 1% wick-bypass class of bug flagged in the May-24 review for grid_fisher and fin_snipe_manager (those were widened to 3%). The swing trader is now THE primary metals entry path and places this stop on every BUY. On a 5x XAG cert with 30% SL anchor, a US-open volatility spike past the trigger can fire the stop without filling the sell — naked position drops 5-10% further before the next cycle re-arms.

3. **Fast-tick gets ZERO ticks on cycle overrun** — `_sleep_for_cycle` (data/metals_loop.py:1051) returns immediately with an "overran by Xs" log when main cycle work exceeds 60s. Ministral+Chronos+Qwen3 inference chains routinely take 30-90s during LLM-heavy cycles (memory `gpu_loop_reliability.md` warns this). During an overrun the 10s silver fast-tick fires 0 times that cycle, blowing through threshold alerts and the 3-min velocity window. There is no catch-up mechanism on the next cycle either.

---

## Critical (P0)

data/metals_loop.py:1088: P0: `_has_active_silver()` only checks legacy POSITIONS dict; swing-trader silver positions (data/metals_swing_state.json) get NO 10s fast-tick monitoring (threshold alerts, velocity alerts, opportunistic XAG orderbook snapshot at 10s). Fix: also probe self.swing_trader.state["positions"] for any pos with underlying=="XAG-USD" and active flag/units>0.

data/metals_loop.py:1051: P0: `_sleep_for_cycle` overrun branch returns immediately with no fast-tick sub-loop — when LLM inference pushes main cycle past 60s the silver fast-tick fires 0 times for that cycle. Velocity deque (18 readings × 10s = 3 min) is also corrupted because successive overruns gap the price history without resetting. Fix: when overrun detected, still run at least one tick before returning; or detect gaps in the velocity window and force `_silver_fast_prices.clear()`.

data/metals_swing_trader.py:2760: P0: `sell_price = round(trigger_price * 0.99, 2)` — only 1% buffer below stop trigger on the swing trader's hardware stop. On a 5x XAG cert this is ~0.2% underlying — a single US-open wick prints past trigger without filling the sell. Memory `feedback_mini_stoploss.md` mandates 3%+ on 5x products. Fix: `round(trigger_price * 0.97, 2)` matching the prior batch widening for grid_fisher and fin_snipe_manager. [REPEAT — same pattern as May-24 metals_loop:2469/4913 ladder finding, but here it's on the active swing path, not the legacy gated ladder.]

data/metals_llm.py:443: P0: Ministral-8B prompt for XAG/XAU metals decisions begins `"You are an expert cryptocurrency trader"`. The model is being asked to score metals using a crypto persona — biases drift, sentiment framing, and asset-class reasoning are wrong. This explains some of the Ministral accuracy regression noted in MEMORY.md. Fix: replace with `"You are an expert metals trader specialising in silver and gold futures"`.

---

## Important (P1)

data/metals_warrant_refresh.py:316,322: P1: catalog stale-cache fallback never alerts beyond DEBUG. If refresh fails completely and the cache is stale (TTL=6h), `load_catalog_or_fetch` returns the stale dict with a WARNING log only — no Telegram, no critical_errors.jsonl. The current on-disk catalog is already 5+ days old (`refreshed_ts: 2026-05-21` per git status). A weekly cert rebase between then and now means `barrier` / `barrier_dist_pct` could place a trade into a near-knockout zone. Fix: gate trading entirely when catalog staleness > 24h, or write critical_errors.jsonl entry. [REPEAT — flagged in May-24 review's "Notes" section, not actioned.]

data/metals_swing_trader.py:2796: P1: hardcoded `close_cet = 21.0 + 55/60` (21:55) in `_check_exits`. DST gap weeks (~Mar 8-29, Oct 25-Nov 1 when EU/US DST offsets misalign) push the real Avanza commodity close to 21:00 CET. With `EOD_EXIT_MINUTES_BEFORE = 0` user-override, positions are held 55 min past actual market close, bleeding into off-hours liquidity. Same hardcode at metals_swing_trader.py:2448 (entry gate) and data/metals_loop.py:1574 (`is_market_hours`). Fix: ship the `get_session_close_cet()` lookup that metals_swing_config.py:319-321 already TODOs.

data/metals_signal_tracker.py:643,644: P1: `best_key = max(report.keys(), key=lambda k: report[k]["accuracy"])` — no minimum-sample gate. The "best" / "worst" signals exposed via `get_accuracy_for_context()` (consumed by Claude) can be a single-sample 100% signal beating a 1000-sample 60% signal. This is the exact small-sample inflation pattern that produced the "XAG 82% from 34 samples" illusion (memory). Fix: filter `report` to entries with `total >= 30` before max/min selection.

data/metals_loop.py:1485-1503: P1: silver velocity alert (3-min flush) deduplicates by `vel_key = f"vel_{int((time.time() - 2) // 300)}"` — a 5-min coarse bucket. After a single rapid drop alerts and the price keeps falling another 1% in the next 4 minutes (entirely possible during US open), no second alert fires until the next 5-min epoch. Also: if the fast-tick missed ticks during a cycle overrun, the velocity computation uses a non-contiguous window (oldest is 10-20 minutes old) and may report misleading % moves. Fix: time-tag each tick in the deque (price, ts) and only compute velocity when oldest ts is within window.

data/metals_swing_trader.py:2487,2509,2547: P1: `live_leverage = data.get("leverage") or w.get("leverage")` can be None on a stale Avanza response where keyIndicators.leverage is missing; line 2509 then crashes on `abs(None - TARGET_LEVERAGE)`. Catalog refresh populates leverage, but legacy `WARRANT_CATALOG` entries in metals_swing_config.py have it set, and the fallback path uses `w.get("leverage")` which is OK there — but a partial fetch from `fetch_price` returning leverage=None and overriding the catalog value brings the crash live. Fix: `live_leverage = data.get("leverage") or w.get("leverage") or 0`; explicit `if not live_leverage: continue` guard.

data/silver_monitor.py:1-9: P1: file is marked DEPRECATED in the docstring but is 800+ lines, has its own singleton lock, and is still actively maintained (last edit 2026-05-26 — same day as this review). Two competing silver monitors exist: the merged metals_loop fast-tick (10s) and this standalone (10s + 5-min Claude). If both run (and the singleton lock at data/silver_monitor.singleton.lock isn't paired against metals_loop's lock), they double-tick the Binance FAPI from the same host — risk of rate-limit hits AND duplicate Telegram alerts on threshold breaches. Fix: actually delete this file or harden the deprecation to refuse-to-run with sys.exit when the metals_loop is detected (e.g. via metals_loop singleton lock probe).

data/metals_swing_trader.py:1862: P1: `direction = "SHORT" if sig.get("action") == "SELL" else "LONG"` — but the assignment is conditioned on `SHORT_ENABLED=False` at metals_swing_config.py:161. With SHORT disabled, a SELL action arriving here is fall-through into `_select_warrant(underlying, "SHORT")` if (and only if) `_evaluate_entry` already returned True for direction="SHORT" — which it doesn't because `_evaluate_entry` returns `(False, "SHORT disabled")` at line 2214. Net: dead branch. The bug-shaped concern: when SHORT_ENABLED is flipped to True in an emergency, `_select_warrant` returns a BEAR cert and `_execute_buy` is called with `direction="SHORT"` — but `_execute_buy` ONLY records direction in the position dict; the Avanza order is BUY (line 2625 hardcodes "BUY"). The BEAR cert IS bought (correct for SHORT exposure), but the swing trader's per-leverage TP/SL math at line 2650-2651 uses positive `live_lev` — TP/SL signed correctly only because `_check_exits` flips the und_change_pct for SHORT (line 2868). However, the warrant-side `WARRANT_TAKE_PROFIT`/`WARRANT_TRAILING` block at line 3011 has `if direction == "LONG"` — SHORT BEAR-cert positions get NO warrant-side TP. This was the entire reason for adding WARRANT_TAKE_PROFIT (MM-side spikes), so SHORT positions lose that protection silently. The TODO comment at 2999 acknowledges this — un-actioned. Fix: port the block for SHORT with sign-inverted peak tracking before flipping SHORT_ENABLED.

data/metals_llm.py:386-388: P1: `_query_chronos_server_inner` returns None on empty/timeout response and calls `_stop_chronos_server()`. The Chronos server is then restarted on the very next query — but if the underlying issue is VRAM exhaustion or GPU hang, restart hits the same wall and silently returns None. The persistent server holds 673MB; combined with Ministral (4.5GB) and Qwen3 (4GB) on a 10GB GPU, contention is plausible. The `_log` is INFO-level; no critical_errors.jsonl entry. Caller treats None as "fall back to subprocess" — but that subprocess loads Chronos on CUDA AGAIN. Fix: log critical when restart sequence repeats N times within 10 min.

data/metals_avanza_helpers.py:283: P1: `today_str = datetime.datetime.now().strftime("%Y-%m-%d")` uses naive local time. On a Windows-CET host this matches Avanza's expected day, but on a non-CET host (e.g. cloud-hosted UTC failover), `validUntil = today_str` may be "yesterday" from Avanza's POV, making the order immediately expired. Same risk at line 358 for stop-loss `valid_until`. Fix: use `datetime.datetime.now(ZoneInfo("Europe/Stockholm")).strftime("%Y-%m-%d")`.

---

## Important (P2)

data/metals_swing_trader.py:2110: P2: `units = int(alloc / ask_price)` floors fractional units. For a SEK 750 ask price on a 5x cert with alloc=3000, units=4 → total_cost=3000 — clean. But with alloc=1000 (MIN_TRADE_SEK floor) and ask=720, units=1 → 720 SEK actual spend, BELOW the 1000 SEK courtage threshold. `place_order` then logs a WARNING and proceeds — courtage on a 720 SEK trade eats ~1.5% round-trip alone. Fix: gate `if total_cost < MIN_TRADE_SEK * 0.95: continue` AFTER the int() truncation.

data/metals_loop.py:1492: P2: silver velocity dedup key `vel_{int((time.time() - 2) // 300)}` adds a 2-second offset comment-explained as "stable for the full window, avoiding a double-fire when time.time() rolls over the 5-min epoch". But the offset doesn't actually fix double-fire — it just shifts the boundary 2s earlier. If a velocity alert fires at t=298 with key vel_0 (since (298-2)//300 = 0), the next alert at t=302 has key vel_1 (since (302-2)//300 = 1) — both fire within 4 seconds. Fix: use a monotonic last_vel_alert_ts with explicit cooldown (e.g. 300s).

data/metals_risk.py:286-293: P2: `simulate_warrant_risk` is called with `direction_prob` derived from LLM consensus direction "up"/"down". For a BEAR cert (SHORT position via inverse warrant), `direction == "up"` for the UNDERLYING means the BEAR cert price goes DOWN — but `simulate_warrant_risk` treats `direction_prob > 0.5` as bullish for the WARRANT (entry_price_warrant rises). Risk metrics are inverted for SHORT positions, biasing VaR/stop-hit-probability low when the underlying outlook is unfavorable for the position. Fix: detect SHORT positions (key contains 'bear' or pos has direction="SHORT") and invert `direction_prob = 1.0 - direction_prob` before calling.

data/metals_accuracy_review.py:18-21: P2: hardcoded `POSITIONS = {"gold": {"entry": 972.4}, "silver79": {"entry": 65.13}, "silver301": {"entry": 20.70}}` — these are entries from MONTHS ago. The "per-position PNL" output and HOLD/SELL accuracy buckets are anchored to phantom entries, not the current state. Anyone running `data/metals_accuracy_review.py` to validate the system gets a misleading report. Fix: read entries from data/metals_positions_state.json AND data/metals_swing_state.json at runtime.

data/metals_accuracy_review.py:14: P2: `os.chdir(r"Q:/finance-analyzer")` at module load (also data/metals_history_fetch.py:17). Side-effect on import; if any other module imports these helpers (the relative `data/` paths suggest they might), the CWD silently changes. Fix: replace with explicit absolute paths via `Path(__file__).resolve().parents[1]`.

data/metals_loop.py:7204-7210: P2: silver fast-tick activation gated on `_has_active_silver()` AND `_silver_underlying_ref is None`. If the loop restarts and reads a swing-managed silver position via `detect_holdings`, `POSITIONS` may not contain it (swing positions live in metals_swing_state.json, not the legacy POSITIONS dict). The activation message never logs. Compounds the P0 above. Fix: make `_has_active_silver()` swing-aware.

data/metals_swing_trader.py:2625: P2: BUY order uses `ask_price` as the limit price. If by the time the order reaches Avanza the ask has moved up, the order rests unfilled and the swing trader has already subtracted `total_cost` from `cash_sek` (line 2673). `fill_verified=False` plus `_verify_recent_fills` rollback path covers this AFTER timeout, but during the gap (FILL_VERIFY_MAX_AGE_S) the trader believes cash is committed. Sizing for any concurrent ticker's BUY in the same cycle is wrong. Fix: subtract cash only inside the `if success and fill_verified:` path of `_verify_recent_fills`; or use an "open_orders_committed_sek" sidecar that's reconciled per cycle.

data/metals_swing_config.py:323-329: P2: `EOD_EXIT_MINUTES_BEFORE = 0` user-override means positions are held all the way to (and past) 21:55 CET. After 22:00 CET, `minutes_to_close = (21.917 - 22.5) * 60 = -35` — EOD_EXIT fires. But after midnight CET, `_cet_hour() ≈ 0` returns `minutes_to_close = +1315` (positive) — EOD doesn't fire. Between 22:00 and 23:59 CET the system attempts an EOD sell, but Avanza is CLOSED — the sell order rests unfilled into the next session, then sells at next morning's opening fix. Fix: include weekday/closed-session check in EOD_EXIT (skip if outside market hours since order won't fill).

portfolio/metals_cross_assets.py:139-145: P2: `get_gold_silver_ratio` uses GC=F and SI=F (US futures, closed 22:00 CET Fri to 23:00 CET Sun). On weekends and after-hours, the ratio is stale by up to 65 hours, but the function returns it without a freshness flag. Downstream metals signal weighting treats this as live. Fix: tag `as_of_ts` from `df.index[-1]` and let callers reject stale data.

data/metals_avanza_helpers.py:401: P2: `place_stop_loss` success check is `body.get("status") == "SUCCESS"` — but `place_order` uses `body.get("orderRequestStatus") == "SUCCESS"`. Inconsistent — if Avanza changes the stop-loss response shape (it has changed twice in 2026 per the file's own comments at line 121-128), the helper silently reports success=False and the position becomes naked while the caller logs "stop placed" (line 2783 in swing trader writes the stop_id but never checks for empty string).

data/metals_loop.py:1086-1088: P2: `_has_active_silver` iterates `POSITIONS.items()` checking `"silver" in key.lower()`. Case-sensitive subset string match — if a key is `XAG_TRACKER` or `BEAR_SILVER_X5_AVA_12`, "silver" is in the second but not the first. Inconsistent detection. Fix: also check `pos.get("underlying") == "XAG-USD"`.

data/metals_warrant_refresh.py:281: P2: `"parity": probe.get("parity") or 10` — fallback to 10 when Avanza doesn't return parity. For warrants with parity 1 (single-unit certs like BEAR cert config entry), this would overstate price-per-underlying-unit by 10x and break any cross-check math. Per-warrant parity must be authoritative. Fix: skip the candidate when parity is missing rather than defaulting.

data/metals_swing_trader.py:3011-3041: P2: warrant-side TAKE_PROFIT/TRAILING block guarded by `if direction == "LONG"`. SHORT BEAR-cert positions get NO market-maker-spike TP protection. The 2026-04-20 incident this block was added for (warrant peaked +5.9% while underlying only +1.26%) can happen symmetrically on BEAR certs when underlying crashes. Fix: port with sign-inverted peak tracking when SHORT is enabled (TODO comment at 2999 is unactioned).

portfolio/metals_orderbook.py:35: P2: `_fetch_fapi_json` calls `_binance_limiter.wait()` from `portfolio.shared_state`. The metals_loop fast-tick (10s cadence) and main cycle (60s) both call this through different paths — under contention the limiter may serialize fast-tick calls behind a hung main-cycle request, adding 5-10s latency to silver alerts. Fix: dedicated `_metals_orderbook_limiter` or bypass rate-limiting for the fast-tick orderbook snapshot since it's only XAG-USD at 10s.

data/metals_loop.py:1407-1503: P3 / [REPEAT — explicit not-a-race documented in May-24 review]: silver fast-tick mutates `_silver_session_low/high`, `_silver_consecutive_down`, `_silver_prev_price`, `_silver_fast_prices` deque, `_silver_alerted_levels` set without locks. Single-threaded today (May-24 review verified). Add a doc comment that future threading additions require lock. Not actionable as a bug today.

data/metals_swing_trader.py:2697: P3: BUY Telegram references DEPRECATED `TAKE_PROFIT_WARRANT_PCT` and `STOP_LOSS_WARRANT_PCT` globals (5%, 30%) — for non-5x positions these print wrong numbers (a 10x cert is 10%/60% per-position pct, Telegram shows 5%/30%). Cosmetic; doesn't affect trading. Fix: format from `tp_warrant_pct`/`sl_warrant_pct` already stored on the position dict.

---

## Verification against May-24 review

| Item | State |
|---|---|
| metals_loop.py legacy stop ladder 1% sell buffer (lines 2469, 4913) | NOT RE-CHECKED in current scope (outside top-3-priority paths) but presumed still present per May-24 finding |
| silver fast-tick "race" with main loop | [REPEAT] Confirmed not a race (single-threaded). But P0 here: fast-tick gets 0 ticks on overrun, which May-24 didn't catch |
| Catalog staleness (>24h) | [REPEAT] still degrades silently; on-disk catalog is currently 5+ days old |

---

## Recommended remediation order

1. **Today** — top-3 #1 (silver fast-tick swing-blindness, metals_loop.py:1088), #2 (swing trader 1% stop buffer, line 2760), #4 (Ministral metals prompt at metals_llm.py:443).
2. **This week** — fast-tick overrun handling (`_sleep_for_cycle`); accuracy-tracker min-sample gate (P1); catalog staleness gate (>24h refuse-trade).
3. **Next sprint** — DST-aware close lookup (P1); SHORT warrant-side TP block (P2); kill or harden silver_monitor.py.

End of review.
