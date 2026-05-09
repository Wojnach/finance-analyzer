# Critique of Codex Review — Metals Core Subsystem

Date: 2026-05-09
Reviewer: Claude Opus 4.7 (1M)
Source: data/fgl-logs/codex-metals-core.txt (codex/gpt-5.4)
Reference: docs/fgl/claude-metals-core.md

## Verdicts on Codex Findings

[CONFIRM] data/metals_loop.py:2456 — Existing stops cancelled before "too close" guard runs (P0) | At line 2456 `_cancel_stop_orders(...)` runs unconditionally; the distance gate at 2466 then `return`s without re-placing. Position is left bare during fail-closed window. Bug confirmed in actual code.

[CONFIRM] data/metals_loop.py:4893 — Same-day fast path treats any non-empty `orders` list as "already placed" (P0) | Lines 4893-4895 check only `date == today_str AND stop_base == pos["stop"] AND existing.get("orders")`. No filter for `status == "failed"` records, so a single bad placement poisons the day. Confirmed.

[CONFIRM] data/metals_loop.py:4900 — Initial stop placement cancels existing before 3% distance gate (P0) | Line 4900 `_cancel_stop_orders(...)`, then 4912 distance check `continue`s without replacement. Same fail-closed issue as finding 1 but in the cold-start path. Confirmed.

[CONFIRM] data/metals_loop.py:2499 — Rebuilt trailing-stop state writes `volume` instead of `units` (P0) | Line 2499 in `_rebuild_stop_orders_for` literally writes `"volume": level_units`, while `place_stop_loss_orders` at line 4951 writes `"units": level_units`. Schema mismatch is real.

[CONFIRM] data/metals_loop.py:5066 — Stop-fill accounting reads `order["units"]`, breaks on rebuilt records (P0) | Line 5066 `total_filled_units += order["units"]` will KeyError on records written by `_rebuild_stop_orders_for`. The broad `except` at 5070-5078 swallows it as a `warning` log line; broker-sold position remains marked active locally. Confirmed.

[CONFIRM] data/metals_loop.py:5016 — Stop orders marked `cancelled` regardless of HTTP status (P1) | Line 5015 logs `result.get('status')` from page.evaluate, but line 5016 sets `order["status"] = "cancelled"` unconditionally inside the try block — no check on the 200/404/500 return. State desyncs from broker on any non-2xx. Confirmed.

[CONFIRM] portfolio/fin_fish.py:360 — SHORT fishing feeds `1 - p_up` into `drift_from_probability` (P1) | Line 360: `drift = drift_from_probability(1.0 - p_up, vol)` for SHORT direction. `drift_from_probability` (monte_carlo.py:68 docstring) takes "probability of price being higher at horizon" — passing the inversion produces drift opposite to the intended directional belief, mis-ranking BEAR setups. Confirmed.

[CONFIRM] portfolio/fin_fish.py:735 — Breached BEAR MINI barriers fall through with `pass` (P1) | Line 732-735 literally:
```
if direction == "SHORT" and spot >= barrier:
    # ... skip if too close
    pass
```
Then code falls through to barrier_distance check at 736 `abs(spot - barrier) / spot * 100` which is 0+ even when knocked out. Already called out in claude-metals-core.md P0 #6. Confirmed.

[CONFIRM] portfolio/fin_snipe.py:160 — Ladder generation never passes `direction_sign=-1` for BEAR/MINI S (P1) | Line 160 calls `build_intraday_ladder(...)` with no `direction_sign` parameter. `metals_ladder.build_intraday_ladder` defaults to LONG translation. Same bug claude flagged at P0 #3 in claude-metals-core.md (cross-flagged from 2026-04-16 review). Confirmed; codex correctly identifies it as still unfixed.

[CONFIRM] portfolio/fin_snipe_manager.py:1246 — Flat-state planning carries forward stale `entry_underlying` (P1) | Line 1246 — `entry_underlying` keeps `instrument_state.get("entry_underlying")` when `position_volume == 0`. Next entry into same orderbook reuses old basis. Confirmed.

[CONFIRM] portfolio/fin_snipe_manager.py:1311 — Planning exception only logs and skips (P1) | Lines 1305-1321 — `try plan_instrument` then `except Exception: logger.error(...) ; _notify_critical(...) ; continue`. A live position whose plan_instrument raises will miss its sell/stop maintenance for the cycle. The `_notify_critical` softens the impact but core risk holds. Confirmed.

[CONFIRM] portfolio/fin_snipe.py:49 — Stop-loss inventory failures collapsed to `[]` (P2) | `fetch_stop_losses` lines 45-54: `except Exception: logger.warning(...); return []`. Caller cannot distinguish "no stops exist" from "API unreadable". Confirmed.

[CONFIRM] portfolio/fin_snipe_manager.py:1237 — `next_state` never persists `entry_ts` (P2) | Line 1237 dict literal lacks `entry_ts` key. While line 460 mutates `instrument_state["entry_ts"]` in-place, line 1324 (`current_state["instruments"][orderbook_id] = plan["state"]`) overwrites with the new `next_state` that lacks the field. Mutation persists only mid-cycle then is wiped. HOLD_TIME_EXTENDED never accumulates across cycles. Confirmed (despite the in-cycle mutation hack documented in lines 434-448 comments).

[CONFIRM] portfolio/exit_optimizer.py:617 — Hold EV from 5 quantile percentiles instead of full distribution (P2) | Lines 617-621: `terminal_pnls = np.array([_compute_pnl_sek(...) for p in np.percentile(terminal, [10, 25, 50, 75, 90])])` then `hold_ev = float(np.mean(terminal_pnls))`. Mean of 5 quantile-sampled paths, not mean over all simulated terminal paths. Bias kicks in on non-linear PnL (knockout floor at 0). Confirmed.

[CONFIRM] portfolio/metals_precompute.py:149 — Off-cycle precompute runs overwrite deep-context with None (P2) | `_fetch_market_data` (line 137) initializes all keys to None. Each source is only set when `_should_refresh(...)` passes (lines 149, 162, 175, 188, 201, ...). Off-cycle calls return a dict with valid sections set to None for non-refreshed sources. If callers atomically rewrite contexts using this output, last-good payloads are clobbered. Confirmed.

[CONFIRM] portfolio/orb_predictor.py:32 — Session windows hardcoded to winter UTC (P2) | Lines 32-33: `MORNING_START_UTC = 8`, `MORNING_END_UTC = 10`. Comment explicitly says "winter". During CEST (late March – late October), 09:00 CET = 07:00 UTC, so the predictor reads 10:00-12:00 CET instead of 09:00-11:00 CET. Off-by-one for ~7 months. Already flagged in claude-metals-core.md P1. Confirmed.

[CONFIRM] portfolio/iskbets.py:313 — Layer-2 gate defaults to approved=True on failures (P2) | Line 313 `approved = True` initialization, then timeout (345), FileNotFoundError (348), and generic exception (351) handlers all leave `approved` at the default. Only auth failure (340) and explicit reject (342) flip it. Operational failures silently approve buys. Confirmed.

[CONFIRM] data/metals_loop.py:1836 — Microstructure persisted every 5 cycles vs 120s readers (P3) | Line 1836: `if _microstructure_persist_counter % 5 == 0:`. Main loop ~60s → persist every ~5 minutes. Reader staleness threshold is 120s (microstructure_state.py:227). Persisted data self-invalidates between writes. Comment at 1836 says "~2.5-5 min" which contradicts the 120s threshold. Confirmed.

[CONFIRM] portfolio/microstructure_state.py:227 — `load_persisted_state` drops snapshots >120s (P3) | Line 227: `if age_ms > 120_000: return None`. Companion to finding 18; readers see None most of the time. Confirmed.

## MISSED BY CODEX (P0/P1 from claude-metals-core.md)

[CONFIRM-MISSED] portfolio/fin_snipe.py:69 — `fetch_positions_by_orderbook()` does NOT filter on ISK account 1625505 (P0) | Line 69 records `account_id` but never filters; the dict is keyed solely on `orderbook_id` (line 66), so a pension holding (account 2674244) of the same warrant overwrites the ISK record. Project rule (`memory/feedback_isk_only.md`): only show ISK 1625505. Catastrophic for downstream order routing. Codex did not flag this.

[CONFIRM-MISSED] portfolio/fin_snipe_manager.py — entire module has zero direction handling (P0) | `Grep "SHORT|BEAR|direction"` on the file returns 0 occurrences. Combined with codex finding #9 (ladder direction) and `_compute_exit_target` calling `translate_underlying_target` with default `direction_sign=1`, every BEAR/MINI S position is mis-priced at every exit/stop computation. Codex caught the ladder symptom (#9) but missed the broader structural blindness across the manager.

[CONFIRM-MISSED] portfolio/fin_snipe_manager.py:61,536 — `HARD_STOP_CERT_PCT = 0.05` anchored to entry, never trails (P0) | Line 61 sets the constant; line 536 anchors trigger to `position_avg * (1.0 - HARD_STOP_CERT_PCT)`. For 5x leverage that's only 1% on underlying — well inside silver's intraday wick range. Project memory `feedback_mini_stoploss.md` requires -15%+ stops for 5x leverage. Codex flagged this stop neither for sizing nor for trailing.

[CONFIRM-MISSED] portfolio/fin_snipe_manager.py — no MINI knockout-barrier proximity check anywhere (P0) | `_compute_stop_plan` only checks distance from current bid (`MIN_STOP_DISTANCE_PCT`), never from `barrierLevel`. A `MINI L SILVER` stop at `entry × 0.95` could be inches above the barrier; trigger + slippage = knockout-zone fill at near zero. Project rule: "Never place stop-losses near MINI warrant barriers". Codex missed.

[CONFIRM-MISSED] data/metals_loop.py:4869 + 2486 — `place_stop_loss_orders` and trail cascade lack barrier-proximity check (P0) | The 3% distance check is from `cur_bid`, not from `barrierLevel`. No `keyIndicators.barrierLevel` lookup before placing cascading stops. Codex missed (only flagged the cancel-before-validate ordering, finding #1/#3).

[CONFIRM-MISSED] portfolio/fin_fish.py:730-735 — for SHORT direction barrier-knockout check is `pass` (P0) | Codex finding #8 (line 735) catches the SHORT pass-through bug but classifies as P1; claude classifies as P0. Severity disagreement: a knocked-out instrument being sized for fishing CAN result in real money flowing into a dead instrument. Both reviews caught it, but codex understated severity.

[CONFIRM-MISSED] data/metals_loop.py:1455 — `silver_pos.get("leverage", 4.76)` hardcoded default (P0) | Line falls back to 4.76 always since `POSITIONS_DEFAULTS` lacks `leverage`. For BULL X8 N positions, fast-tick warrant_pct estimate is wrong by ~70%. Codex did not flag.

[CONFIRM-MISSED] data/metals_loop.py:1450,1457,1478 — `_silver_fast_tick` LONG-only thresholds (P0) | `pct_change` and `warrant_pct = pct_change * leverage` with negative thresholds. A BEAR silver position would never alert on rising XAG. The catalog HAS `SHORT_INSTRUMENTS["bear_silver_x5"]`, so this isn't theoretical. Codex missed.

[CONFIRM-MISSED] portfolio/exit_optimizer.py:320-340 — `_compute_pnl_sek` LONG-only (P1) | Lines 320-340 read in code: `exit_warrant_sek = (exit_price_usd - position.financing_level) * fx` for MINI futures; `pct_move * leverage` for daily certs. No `Position.direction` field exists; `dataclass Position` has no SHORT branch. Codex missed.

[CONFIRM-MISSED] portfolio/exit_optimizer.py:373-378 — `_compute_risk_flags` LONG-only knockout proximity (P1) | Line 374: `distance_pct = (market.price - position.financing_level) / market.price * 100`. For BEAR MINIs the financing level sits ABOVE underlying; expression goes negative; `distance_pct < 3` is always true → forces market exit on every cycle for any held BEAR position. Codex missed.

[CONFIRM-MISSED] portfolio/fin_snipe_manager.py:921-929 — `_budgeted_entry_volume` no available-cash check (P1) | Manager sizes per-instrument `budget_sek` independently; multiple fin_snipe instruments can each consume their full budget; no `/_api/customer/cash` lookup. Codex missed.

[CONFIRM-MISSED] portfolio/fin_snipe_manager.py:918 — order sizing can fall below 1000 SEK courtage minimum (P1) | `int(budget_sek // working_price)` × `working_price` is unenforced; flash-leg at L949-955 takes 30% of an already-too-small volume. Project rule (`metals-avanza.md`): every leg must be ≥1000 SEK. Codex missed.

[CONFIRM-MISSED] portfolio/microstructure_state.py:205-213 (`persist_state`) — double-records OFI per cycle (P1) | `persist_state` calls `get_microstructure_state(ticker)` for each ticker; that function calls `record_ofi(ticker, ofi)`. So every persistence cycle DOUBLE-RECORDS OFI into rolling history, polluting the z-score baseline. Codex missed despite reviewing the persist/load symmetry (findings 18-19).

[CONFIRM-MISSED] data/metals_loop.py:3786-3803 — `emergency_sell` uses `condition: NORMAL` at current bid (P1) | For an L3 emergency (within 3% of stop), placing a limit at the current bid rather than `bid × (1 - slippage)` may leave the order unfilled if bid retreats. Codex missed.

## Counts

CONFIRM=19 DISPUTE=0 PARTIAL=0 UNVERIFIED=0 MISSED=14
