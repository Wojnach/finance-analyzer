# Adversarial Code Review — Metals Core Subsystem

Date: 2026-05-09
Reviewer: Claude Opus 4.7 (1M)
Scope: data/metals_loop.py, portfolio/{metals_orderbook,metals_cross_assets,metals_ladder,metals_precompute,exit_optimizer,price_targets,orb_predictor,orb_postmortem,silver_precompute,iskbets,fin_fish,fin_snipe,fin_snipe_manager,microstructure,microstructure_state}.py

Methodology: cold read of every file in scope; cross-checked against project memory (feedback_mini_stoploss, feedback_bounce_reentry, fishing_system, market_hours, feedback_isk_only) and against `.claude/rules/metals-avanza.md`.

## P0 — Catastrophic / Real-money risk

[P0] portfolio/fin_snipe.py:60-74 — `fetch_positions_by_orderbook()` does NOT filter on ISK account 1625505. The `withOrderbook[]` payload includes ALL accounts (ISK + pension 2674244). Map keyed only by `orderbook_id` so a pension holding of the same warrant overwrites the ISK record, and downstream `account_id` (line 138) is whatever was on the last record processed. Project rule: "Only show ISK account 1625505. Ignore pension account 2674244." Catastrophic — fin_snipe_manager will then route SELL/BUY actions at the pension account. | FIX: filter `if str((item.get("account") or {}).get("id")) != "1625505": continue` (and constant the value).

[P0] portfolio/fin_snipe_manager.py — entire module has ZERO direction handling. No reference to "SHORT", "BEAR", "LONG", or "direction" anywhere in the manager. Manager assumes every position is LONG. For BEAR certs this means: (a) `_compute_exit_target` at L491-497 calls `translate_underlying_target(...)` with default `direction_sign=1` (mirrored target), (b) `compute_exit_plan` from exit_optimizer assumes warrant price moves WITH underlying, so for a BEAR cert held while underlying rises the simulator produces "winning" exits at higher underlying levels (which is exactly when the BEAR cert is LOSING). Stops, exit, sizing all wrong direction. | FIX: derive `direction_sign` from snapshot (instrument name "BULL"/"BEAR"/"MINI L"/"MINI S" or market-guide direction), thread through to translate_underlying_target and exit_optimizer, and add a SHORT branch to `_compute_pnl_sek`.

[P0] portfolio/fin_snipe.py:160-172 — `build_intraday_ladder(...)` is called WITHOUT `direction_sign`, so it defaults to `+1` (LONG). For BEAR/MINI S products in the managed orderbook list this produces working/flash/exit prices on the WRONG side of current price (mirrored). Same bug previously called out in `docs/reviews/2026-04-16/codex/metals-core.txt:9` and `data/adv-2026-05-08/codex-metals-core.md:541`; still unfixed. | FIX: detect direction from `market.underlying`/instrument name and pass `direction_sign=-1` for SHORT instruments.

[P0] portfolio/fin_snipe_manager.py:61,536 — `HARD_STOP_CERT_PCT = 0.05` produces a stop trigger at `position_avg × 0.95`. For 5x leverage that is only `1%` on underlying — well inside silver's intraday wick range (project memory `feedback_mini_stoploss`: "Never place stop-losses near MINI warrant barriers" and "5x leverage certificates need -15%+ stops, not -8%, to survive intraday wicks"). Stop will trigger on every dip and re-enter via working buy → death-by-1000-spreads. | FIX: scale the cert stop by leverage — e.g. compute target underlying drawdown (10-15% leverage-adjusted) and translate via `translate_underlying_target`, with a hard floor at 15% on cert price for 5x and 25% for 10x. Mirror logic for SHORT.

[P0] portfolio/fin_snipe_manager.py: ENTIRE FILE — no barrier-proximity check anywhere when sizing the stop. `_compute_stop_plan` only checks distance from current bid (`MIN_STOP_DISTANCE_PCT = 1.0`) — not distance from MINI warrant knockout barrier. For a "MINI L SILVER" with barrier at $X, a stop at `entry × 0.95` could be inches above the barrier; if it triggers and the SELL slippages, the broker may fill below the barrier (knockout) at near-zero. Project memory `feedback_mini_stoploss`: "Never place stop-losses near MINI warrant barriers." | FIX: read `barrierLevel` from market-guide indicators (already extracted at fin_snipe_manager.py:606), compute `(trigger_price - barrier × parity × fx) / current_price`, and reject if < 5%. For SHORT MINIs invert (barrier above current).

[P0] data/metals_loop.py:4869-4983 (`place_stop_loss_orders`) and :2486 (cascade in `_update_stop_orders_for`) — places cascading stop orders without checking distance from MINI warrant barrier. Only check is `distance_pct < 3.0` from CURRENT BID, not from BARRIER. With a sequence of cancel/place under hardware trail enabled, a stop could be placed at the knockout zone. | FIX: load barrier from `key_indicators.barrierLevel`, refuse to place a stop whose trigger underlying-equivalent is within 3% of barrier; for SHORT MINIs invert (barrier above stop trigger).

[P0] portfolio/fin_fish.py:730-735 — for SHORT direction the barrier-knockout check is `pass` (literally a no-op):
```python
if direction == "SHORT" and spot >= barrier:
    # BEAR MINIs get knocked out if underlying goes above barrier
    # (depends on product, but skip if too close)
    pass
```
A BEAR MINI whose underlying has crossed the barrier (i.e. already knocked out) is NOT skipped — code proceeds to size positions and emit fishing levels for a dead instrument. | FIX: `if direction == "SHORT" and spot >= barrier: continue` and apply the same `MIN_BARRIER_DISTANCE_PCT` check using `(barrier - spot) / spot * 100`.

[P0] data/metals_loop.py:1455 — `leverage = silver_pos.get("leverage", 4.76)` defaults to a hardcoded 4.76. The `POSITIONS_DEFAULTS` dict (L521-534) DOES NOT carry a `leverage` field, so this default is ALWAYS used unless something else stuffs the field in later. Means warrant P&L estimate in the fast-tick alerts is wrong for any non-4.76 instrument (BULL X8 N has leverage ≈ 8). | FIX: load leverage from market-guide `keyIndicators.leverage` at position-establish time, persist it in `POSITIONS[key]`, and require non-default before using.

[P0] data/metals_loop.py:1450,1457,1478 — `_silver_fast_tick` computes `pct_change = (price - ref) / ref * 100` and `warrant_pct = pct_change * leverage`, then alerts when `pct_change <= threshold` (negative thresholds). This assumes LONG silver. If a BEAR silver position is ever active (`SHORT_INSTRUMENTS["bear_silver_x5"]` is in the catalog), an XAG RISE — bad for the BEAR position — never crosses the negative threshold so NO alert fires. Position is unmonitored. | FIX: detect direction from POSITIONS[key], invert thresholds and `warrant_pct = pct_change * leverage * direction_sign`.

## P1 — High severity / direction or math errors

[P1] portfolio/exit_optimizer.py:320-340 — `_compute_pnl_sek` for warrants always assumes LONG: `exit_warrant_sek = (exit_price_usd - position.financing_level) * fx` and `pct_move = (exit - entry)/entry; warrant_move = pct_move * leverage`. For BEAR/SHORT warrants a rising underlying must produce NEGATIVE warrant move. Position has no `direction` field. | FIX: add `direction: str = "LONG"` to `Position`, branch `pct_move *= -1 if SHORT` before applying leverage; for MINI futures the formula is `(financing_level - exit)` for BEAR (intrinsic = barrier_above_underlying - underlying).

[P1] portfolio/exit_optimizer.py:373-378,396-400,431-434 — `_compute_risk_flags` and `_apply_risk_overrides` compute `(market.price - financing_level) / market.price` to gauge knockout proximity. For BEAR MINIs the financing level is ABOVE underlying (knockout when underlying RISES), so this expression is negative → distance_pct is negative → `distance_pct < 3` always true → forces market exit on every cycle for any held BEAR position. | FIX: branch on direction; SHORT case is `(financing_level - market.price) / market.price * 100`. Also extend `session_min` knockout check at L396-400 to use `session_max <= stop_buffer` for SHORT (which barrier sits ABOVE).

[P1] portfolio/fin_snipe_manager.py:536-537 — stop is anchored to `position_avg` (entry price), never trails. On a winning trade gradually drifting up, the stop stays at `entry × 0.95`. Combined with `MIN_STOP_DISTANCE_PCT = 1.0` hysteresis (line 545), once it gets > MIN distance from bid the stop becomes effectively immune to repricing. So a +20% winning trade still has its stop at -5% from entry instead of trailing to lock in gains. | FIX: anchor to `max(position_avg, recent_high) × (1 - HARD_STOP_CERT_PCT)` once in profit, à la metals_loop hardware trail.

[P1] portfolio/fin_snipe_manager.py:_budgeted_entry_volume:921-929 / `_entry_volume_from_budget`:913-918 — sizes orders from per-instrument `budget_sek` without verifying actual available cash. Multiple fin_snipe instruments (silver + gold + warrants) each consume their budget independently — sum can exceed account balance. Manager has zero awareness of `available_cash_sek` from `/_api/customer/cash`. | FIX: at top of `plan_cycle`, fetch live ISK balance and proportionally clip `budgets[orderbook_id]` so total ≤ available_cash × 0.95.

[P1] portfolio/fin_snipe_manager.py:918 — `int(budget_sek // working_price)` can produce orders below the 1000 SEK Avanza minimum courtage threshold (`.claude/rules/metals-avanza.md`: "Every Avanza order must be ≥1000 SEK to avoid minimum courtage; applies per-leg on laddered orders"). The flash-leg sizing at L949-955 then takes 30% of that already-too-small volume, so flash-leg can be 200 SEK. | FIX: enforce `qty * working_price >= 1000` after sizing; if not, drop the flash leg or merge into working.

[P1] portfolio/fin_snipe_manager.py — no cross-process lock with metals_loop's swing trader. fin_snipe_manager has its own LOCK_FILE (L52) but metals_loop runs separately and the swing trader reads/writes Avanza orders for similar/identical orderbook_ids. Race window: fin_snipe places a working buy → metals_loop's swing executes a separate buy on the same instrument → over-budget. | FIX: introduce a per-orderbook coordination file (or merge avanza order placement behind a shared lock).

[P1] portfolio/fin_snipe.py:108 — `fetch_positions_by_orderbook()` is called BEFORE `verify_session()` at fin_snipe.py:260 in `main()`, but `build_snapshots` at L96 has no such guard. If the manager's `run_cycle` (fin_snipe_manager.py:1574) calls `verify_session` and gets True, then the underlying API call inside `fetch_positions_by_orderbook` may still 401 if the session expired between calls. No retry/refresh. | FIX: wrap `api_get` in a session-renewal helper that refreshes BankID storage state on 401.

[P1] portfolio/exit_optimizer.py:312-330 — for warrants without `financing_level` (the daily-cert path L326-333), `_compute_pnl_sek` uses `pct_move * leverage` and CAPS at zero (`max(exit_warrant_sek, 0)`). For LONG cert with falling underlying: ok. For SHORT cert: `pct_move` is positive (underlying rose), `warrant_move = +pct_move × leverage` — gives POSITIVE return when the SHORT cert should be losing. The `max(.,0)` cap doesn't help. | FIX: same direction-sign branch as P1 above.

[P1] portfolio/price_targets.py:62-66 — `if hours_remaining <= 0 or price <= 0 or vol_annual <= 0: return 1.0 if _on_easy_side(price, target, "sell") else 0.0` — always passes `"sell"` as the side argument, even when the caller is computing a BUY fill probability. For a buy with target above price (impossible), code returns 0 (correct). For a buy with target below price (instant fill), `_on_easy_side(price, target, "sell")` evaluates `target <= price` → True → returns 1.0 (also coincidentally correct). For a sell with target above price, returns 0.0 (correct). Net: works by accident, but the `"sell"` literal is misleading and will break if `_on_easy_side` semantics change. | FIX: pass `side` parameter through, hard-code each branch correctly.

[P1] portfolio/price_targets.py:106-107 — `fill_probability_buy` uses `price ** 2 / target` symmetry. If `target == 0`, returns `price` (silently — caller's fill_probability returns 1.0 for "easy side"). Safe-by-luck. | FIX: explicit `if target <= 0: return 0.0`.

[P1] portfolio/orb_predictor.py:32-35 — `MORNING_START_UTC = 8`, `MORNING_END_UTC = 10` are hardcoded UTC offsets corresponding to 09-11 CET in WINTER. In SUMMER (CEST = UTC+2) the morning window in CET is 09-11, which maps to 07-09 UTC, NOT 08-10. So during DST the entire morning observation period is OFF BY 1 HOUR. This silently corrupts every prediction during ~7 months of the year. | FIX: compute window from `Europe/Stockholm` zoneinfo each day; do not hardcode UTC offsets.

[P1] portfolio/orb_predictor.py:373-404 — `translate_to_warrant` hardcodes `entry_price=90.55, leverage=4.76`. The financing-level formula `fl = entry_price - entry_price/leverage` is an approximation (real FL is set by issuer, includes financing carry, drifts daily). Fishing/postmortem reports built off this will diverge from reality after a week. | FIX: take live FL from market-guide `keyIndicators.financingLevel` instead of inferring from entry/leverage.

[P1] portfolio/orb_predictor.py:391-397 — `factor` is computed twice. First branch (`if current_warrant_price`) sets factor from `intrinsic_target/(silver_target - fl)` (= 1.0 always since denominator equals numerator), then immediately overwrites with `intrinsic_target/intrinsic_entry`. Dead code, but the comment "let's keep explicit" suggests confusion. | FIX: remove the dead first assignment.

[P1] portfolio/orb_postmortem.py:73,76 — `high_error_pct = high_error_abs / prediction.predicted_high_median * 100` and analogous for low. No zero-check on `predicted_high_median` / `predicted_low_median` — if a prediction came back with median=0 (degenerate sample), divides by zero. | FIX: guard `if median > 0 else 0.0`.

[P1] portfolio/microstructure_state.py:205-213 (`persist_state`) — calls `get_microstructure_state(ticker)` for each ticker, which itself calls `record_ofi(ticker, ofi)` (line 185). So every persistence cycle DOUBLE-RECORDS the current OFI value into the rolling history (once during the main cycle's `get_microstructure_state` call, once during persist). Pollutes the OFI z-score distribution: extreme readings get artificially compressed because the mean shifts toward each computed OFI. | FIX: in `persist_state` read snapshots directly without going through `get_microstructure_state`, OR add a `record: bool = True` flag and pass `record=False` in persist.

[P1] portfolio/microstructure_state.py:216-229 (`load_persisted_state`) — `data[ticker]` is accessed directly after `if not data or ticker not in data: return None`, but no shape validation. If file was written partially (atomic_write_json should prevent this, but corruption can happen on disk full or crash mid-write), `entry.get("ts", 0)` could fail if `entry` is a string/int. | FIX: `if not isinstance(entry, dict): return None`.

[P1] data/metals_loop.py:4480 — `barrier_distance_pct = round((und - barrier) / und * 100, 1)` — assumes barrier BELOW underlying (LONG). For BEAR/MINI S products the barrier sits ABOVE underlying; this returns negative values that are NOT clamped, are used in downstream telemetry, and could feed into L3 emergency triggering logic in misleading ways. | FIX: detect direction; for SHORT use `(barrier - und) / und * 100`. Add unit tests with a SHORT example.

[P1] data/metals_loop.py:7460,7475 — L3 EMERGENCY and AUTO-EXIT both compute `dist = ((bid - pos["stop"]) / bid * 100)` — this is the distance from STOP (not barrier, despite the comment at L414 saying "Stop levels (distance from barrier as % of bid)"). Misleading constant naming. Not a bug per se but accelerates incorrect mental models. | FIX: rename `STOP_L1_PCT` etc. to `STOP_DIST_FROM_BID_*_PCT` and update comment.

[P1] data/metals_loop.py:2977 — hardcoded fallback `"1650161" if direction == "LONG" else "2286417"`. ob_id 1650161 is NOT in either POSITIONS_DEFAULTS or the WARRANT_CATALOG dynamic fishing list inspected in this review. If decision dict omits `instrument_ob`, places order on a possibly stale/wrong instrument. | FIX: assert decision contains explicit `instrument_ob` and fail closed if missing; or look up from catalog by direction+leverage.

[P1] data/metals_loop.py:2979 — `leverage = 5.0  # BULL/BEAR SILVER X5` — hardcoded leverage. If decision picks ob_id 856394 (BULL GULD X8), Kelly sizing at L3014-3023 uses 5x — undersizes by ~38%. | FIX: read leverage from `keyIndicators.leverage` of the market-guide for the chosen `ob_id`.

[P1] data/metals_loop.py:3786-3803 (`emergency_sell`) — uses `condition: NORMAL` regular order at the bid. For a fast-falling silver this places a limit at the CURRENT bid that may not fill if the bid retreats further. For a true L3 emergency (3% from stop → catastrophic risk), should use a WIDER limit (e.g. bid × 0.97) or a marketable IOC. | FIX: place at `bid × (1 - EMERGENCY_SLIPPAGE_PCT)` to ensure execution; fall back to log+telegram if still unfilled.

## P2 — Medium severity

[P2] portfolio/fin_fish.py:367 — `chronos_annual = (chronos_pct / 100.0) * 252`. If `chronos_pct` is a 24h forecast in percent, annualizing by ×252 (trading days) over-scales by ~1 year for what's a 1-day forecast — dimensionally a daily drift, not annual. The signal-derived `drift_from_probability` is annual; blending 0.7 × annual + 0.3 × wrongly-annualized produces ~70% wrong drift contribution. | FIX: use `chronos_annual = chronos_pct / 100.0 / (24/8760) ≈ chronos_pct/100 * 365` (or for trading hours-aware drift, scale by 252×24/24).

[P2] portfolio/fin_fish.py:1359-1383 — `get_positions()` is called without account filter inside fin_fish; matched purely on instrument name keywords. Pension positions (account 2674244) holding the same name keywords would be falsely treated as fishing positions. | FIX: filter by `pos["accountId"] == "1625505"` (or use `fetch_positions_by_orderbook` with a fixed ISK constant).

[P2] portfolio/fin_fish.py:744-754 — barrier distance for LONG is `(level - barrier) / level * 100` (signed; can be negative if level < barrier i.e. already knocked out at the fishing target). For SHORT it's wrapped in `abs(...)`, so a SHORT level above barrier (already knocked out) reports a LARGE positive distance — false safety. | FIX: for SHORT use `(barrier - level) / level * 100` and reject if negative (level above barrier).

[P2] portfolio/fin_fish.py:715-741 — `evaluate_warrants` config leverage is "stale (set when cert was added, not at current price)". For non-daily-cert (`barrier > 0`) it recomputes `leverage = spot / dist`. But for daily-cert with `barrier == 0` it KEEPS config leverage. That config leverage decays daily for real daily certs (rebalance drift) — over 30+ days the config drifts by ~3-5% from actual. Sizing/EV stay slightly off. | FIX: read live leverage from market-guide `keyIndicators.leverage` for daily certs too.

[P2] portfolio/exit_optimizer.py:617-621 — `terminal_pnls` is computed by sampling 5 quantile values of `terminal` and averaging their PnL. This is a 5-point approximation of the expected hold-to-close P&L, which IS biased relative to `np.mean([_pnl(t) for t in terminal])` whenever the P&L function is non-linear (e.g. knockout floor at 0). | FIX: compute mean PnL over all terminal samples, not just 5 quantiles.

[P2] portfolio/exit_optimizer.py:284-296 (`_first_hit_times`) — `argmax` returns 0 on a row where no element is True. Code masks that case by `never_hit` mask + setting `result[never_hit] = -1`. But then in `compute_exit_plan:577` `hitting_times = hit_times[hit_times > 0]` correctly filters; if NONE of the paths hit, `len(hitting_times) == 0` and falls through to `expected_time = remaining_min` — an UNDERESTIMATE of the time-to-fill (since by definition we never filled). Better to SKIP the candidate entirely in that case (fill_prob is already low). | FIX: `if fill_prob == 0: continue`.

[P2] portfolio/exit_optimizer.py:537-548 — `simulate_intraday_paths` uses GBM with constant volatility. For metals near a US-open flash-crash window this produces unrealistically thin tails; combined with the optimizer's quantile candidates, recommended targets cluster too close to current price. Project memory `feedback_mini_stoploss` mentions "5x leverage need -15% stops to survive intraday wicks" — implies fat tails. | FIX: add a regime/vol-shock multiplier (e.g. 1.5x vol during 13:30-15:00 CET if XAG fast-tick velocity > threshold) or switch to t-distribution.

[P2] portfolio/orb_predictor.py:130-139 — `endTime = data[0][0] - 1` to walk backwards. If Binance returns out-of-order responses (rare but possible), this produces gaps. No deduplication. | FIX: dedupe by `ts` in `_parse_klines`.

[P2] portfolio/orb_postmortem.py:144-156 — uses raw `open(path)` instead of project's atomic file_utils. If postmortem file is being appended to while read, can return partial JSON line. | FIX: use `load_jsonl` from `portfolio.file_utils`.

[P2] portfolio/metals_orderbook.py:65 — `bids = [[float(p), float(q)] for p, q in data["bids"]]` raises `ValueError` if any element is malformed (e.g. Binance returns a non-numeric placeholder). Caller catches it via `_cached`'s implicit try, but the cache miss is silent. | FIX: try/except around the float casts, return None on parse failure.

[P2] portfolio/metals_cross_assets.py:138-162 — `get_gold_silver_ratio` computes ratio on `gold_close.index.intersection(silver_close.index)`. If the two series have non-aligned timestamps (e.g. weekend gap on one but not the other), intersection drops bars. The 20-bar sma/std then run on a possibly-discontinuous series. | FIX: forward-fill both before computing ratio.

[P2] portfolio/metals_cross_assets.py:107 — `vs_sma20_pct = (close.iloc[-1] / close.rolling(20).mean().iloc[-1] - 1) * 100`. If sma20 is NaN (insufficient data), result is NaN, falls through to caller as a feature. | FIX: `if pd.isna(sma): return None`.

[P2] data/metals_loop.py:1112-1120 — `_silver_fetch_xag` does no retry on connection error and no timeout on rate-limit-429. A FAPI hiccup causes the fast-tick to silently use the cached price for arbitrarily long. | FIX: track consecutive failures, alert telegram after N (already done via 1-in-30 throttle, but for fast-tick should surface).

[P2] data/metals_loop.py:417 `STOP_L3_PCT = 2.0` — emergency auto-sell triggers when bid is within 2% of stop. Combined with `STOP_L2_PCT = 5.0` and 5x leverage warrants, this is a 10% cert price drop from the stop level. After a typical ATR wick the position will hit STOP_L3 first (auto-sell) before the actual stop-loss runs. Means the cascade is invocation-order-dependent (if L3 fires first, no chance to recover). | FIX: gate auto-sell on STOP_L3 zone PERSISTING for N checks, not just being entered.

[P2] portfolio/fin_snipe.py:107 — `fetch_open_orders` returns ALL orders across ALL accounts — same account-filter bug as positions, but for orders. Snipe manager could try to manage a pension order. | FIX: same fix as P0 for positions.

[P2] portfolio/fin_snipe_manager.py:902-910 (`_seed_entry_volume`) — falls back to `open_buy_volume + position_volume` when no saved volume. On first encounter with a manual order (existing BUY at $X), this adopts that volume. If the user placed a small test order, the manager will keep replacing it with a same-volume working bid forever. | FIX: cap `entry_volume` at `budget_sek / working_price` even when seeded from observed.

[P2] portfolio/fin_snipe_manager.py:1330-1347 — `_validate_action` does not validate that `volume * price` >= 1000 SEK (Avanza minimum courtage). Approved actions can be sub-minimum. | FIX: add `if volume * price < 1000: return "below minimum order size"`.

[P2] portfolio/fin_snipe_manager.py:541 — when `current_bid <= 0` (illiquid silver pre-market), `distance_pct = None` and the "stop too close" check at L545 is bypassed. Can place stops at unsafe levels. | FIX: when bid<=0, refuse to place a NEW stop at all (only re-arm existing).

[P2] portfolio/exit_optimizer.py:679-752 — `compute_exit_plan_from_summary` accepts `instrument_type="warrant"` default but no direction. Same direction blindness as the rest. Even with `financing_level` provided, the SHORT case is broken. | FIX: add `direction` parameter, thread through.

[P2] portfolio/microstructure.py:151-165 (`spread_zscore`) — when std≈0 returns ±10.0 sentinel. That sentinel can leak into downstream signal logic and trigger spurious "extreme spread" alerts. | FIX: return `None` instead of sentinel; let caller decide.

[P2] data/metals_loop.py:2466 — `if distance_pct < 3.0` to skip stop update. But `distance_pct` is computed from `cur_bid` not from BARRIER. If cur_bid wicks down momentarily past stop_base, distance is negative — code SKIPS the update (preserving old stop), which is correct. But during normal operation if the trail is intentional and just close to bid (< 3%), we silently fail to update. | FIX: log a WARNING when skipping; consider relaxing to 1.5% during low-vol regimes.

[P2] portfolio/iskbets.py:441 — `pos["stop_loss"] = trailing_stop if stop_at_breakeven else hard_stop` — only assigns LONG-side stop. ISKBETS targets crypto/stocks (not metals warrants directly), but with no direction guard, a bear-thesis ticker would have stops on the wrong side. | FIX: add direction check / restrict to LONG-only by config.

[P2] portfolio/iskbets.py:763 — fallback `atr = price_usd * 0.02` (2% of price) when ATR computation fails. For high-vol crypto this drastically under-estimates ATR; stop = price - 2 × 0.02 × price = -4% — way too tight. | FIX: use a per-ticker fallback table or refuse to size if ATR unavailable.

[P2] portfolio/orb_postmortem.py:289-290 — `actual_high = max(...); actual_low = min(...)` — but `day_candles` can include incomplete current hour. If the function is called pre-close, "actual" high/low are partial. | FIX: filter `if hour < 22` (full hours only) or use Binance closed-bar timestamp.

## P3 — Low severity / hygiene

[P3] portfolio/fin_snipe.py:154 — `current_price = float(_value(quote.get("sell")) or _value(quote.get("last")) or 0.0)` — uses ASK as `current_instrument_price`. Then `current_ask` (L185) re-uses `sell`. So there's no `current_mid`. P&L estimates will skew. | FIX: derive `current_price = (bid+ask)/2` if both > 0.

[P3] portfolio/exit_optimizer.py:54 — `usdsek: float = 10.85` is a hardcoded fallback inside the dataclass default. If caller forgets to populate, every SEK calc uses 10.85 statically. Already noted in comments at fin_snipe_manager:430-433 but the dataclass default itself is a footgun. | FIX: use `usdsek: float | None = None` and assert non-None at compute time.

[P3] portfolio/metals_orderbook.py:38-44 — `_nocache` decorator is a no-op for tests; for production the function is wrapped with `_cached` but `_nocache` adds another wrap. Not harmful, mildly confusing. | FIX: comment that `_nocache` ONLY exposes `__wrapped__` and does not bypass `_cached` at runtime.

[P3] portfolio/metals_ladder.py:52 — `if current_instrument_price <= 0 or current_underlying_price <= 0: return 0.0`. No check on `target_underlying_price > 0` — if caller passes 0, returns `current_instrument_price` (instant fill = wrong). | FIX: also require `target_underlying > 0`.

[P3] portfolio/metals_ladder.py:56 — `round(max(0.01, ...), 4)` floors at 0.01 SEK. For a knocked-out warrant the real price is 0; clamping at 0.01 hides the dead-instrument case from the snipe planner. | FIX: return None (or 0) and let caller filter.

[P3] portfolio/metals_ladder.py:119-122 — `working_underlying = min(recommended.price, extremes.p25)` — for SELL targets `min` produces the LOWER (more conservative) of two possible exit prices — but for BUY-side ladder this is misnamed (the lower of buy-targets is the better/cheaper one). Variable `working_underlying` lacks side context. | FIX: rename to `working_buy_underlying`; mirror separately for sell.

[P3] portfolio/orb_predictor.py:262 — `morning.range_abs / mid * 100 if mid > 0 else 0` — same expression repeated at L242 (`d_range_pct`) without zero-guard on `d_mid`. | FIX: same zero-guard.

[P3] portfolio/orb_postmortem.py:251-252 — uses raw `open(path)`; not file_utils. Postmortem state file is small but inconsistent with house style. | FIX: `load_json(PREDICTIONS_TODAY_PATH)`.

[P3] portfolio/exit_optimizer.py:444-454 — `_apply_risk_overrides` returns `market_exit` on knockout danger but the candidate list may already CONTAIN a market exit ranked higher; either way the override fires before normal sort. The behavior is correct but the early-return prevents recording the override in `provenance`. | FIX: log override reason to `ExitPlan.provenance["override_reason"]`.

[P3] portfolio/microstructure.py:159-164 — magic constant `10.0` for spread_zscore sentinel when std≈0. Document the choice. | FIX: name a constant.

[P3] portfolio/microstructure_state.py:227 — staleness threshold `120_000 ms` (2 min) is a magic number. Should reference the loop cadence (~60s) so it's clearly "2x the cycle". | FIX: name a constant and comment.

[P3] data/metals_loop.py:1111-1120 — `_silver_fetch_xag` returns the cached value on failure. If that cached value is itself stale (older than several minutes), the fast-tick alerts can fire on "frozen" prices. No staleness check. | FIX: track `_silver_xag_ts` and refuse to alert on data older than 60s.

[P3] data/metals_loop.py:1497-1499 — `vel_key = f"vel_{int((time.time() - 2) // 300)}"` — uses 5-min epoch buckets to dedup velocity alerts. The "now - 2" trick prevents double-fire near the bucket boundary but if two velocity events legitimately happen 4 min apart in the same 5-min window, the second is silently dropped. | FIX: use a true cooldown timestamp instead of bucketed key.

[P3] data/metals_loop.py:2419 — `trail_dist = TRAIL_TIGHTEN_ACCEL` — same condition `velocity < -0.02 and acceleration < 0` is hard-coded magic; thresholds should be config. | FIX: top-of-file TRAIL_TIGHTEN_VEL_THRESH, TRAIL_TIGHTEN_ACCEL_THRESH constants.

[P3] data/metals_loop.py:447 `HARDWARE_TRAILING_PCT = 5.0` — hardware trail at 5% AVANZA percentage. Avanza %-trail uses warrant price not underlying; for 5x leverage this is 1% on underlying — too tight for silver wicks. (Same pattern as P0 above but for the hardware path.) | FIX: scale by leverage; e.g. for 5x use 12-15%.

[P3] portfolio/fin_snipe_manager.py:217-228 (`_price_abs_tolerance` / `_price_matches`) — tolerance tiers `< 1`, `< 20`, `< 100` are magic. Silver warrants often trade ~5-15 SEK; the `< 20` tier (0.02 SEK abs tolerance, 0.25% rel tolerance) is fine for 10 SEK warrants but generates false-match for small warrants. | FIX: align to instrument tick size from market-guide.

[P3] MAYBE: portfolio/fin_snipe.py:128 — `managed_orderbooks = set(grouped_orders) | set(grouped_stop_losses) | set(positions_by_orderbook)`. If user has a stale stop on an instrument they no longer hold (Avanza state drift), this STILL appears in the managed set and the manager will try to size entry orders against a position that doesn't exist. | FIX: only include orderbook_id if it has a position OR an active BUY working order that matches.

[P3] data/metals_loop.py:5003 — JS template-string interpolation: `'https://www.avanza.se/_api/trading/stoploss/' + orderId` — orderId injected into URL. If somewhere a malformed orderId reaches this code path (ever), URL-injection. Inside Avanza session not externally exploitable but bad style. | FIX: encodeURIComponent on JS side.

[P3] portfolio/exit_optimizer.py:221 `vol = max(volatility, _MIN_VOLATILITY)` — silently floors zero ATR at 5% annual. For a delisted/halted instrument this masks the data error. | FIX: log a warning when flooring kicks in.

[P3] portfolio/price_targets.py:329 — same pattern: when `atr_pct <= 0` returns empty result, but doesn't log. Operator can't see why targets are empty. | FIX: logger.debug when silenced.

[P3] portfolio/fin_fish.py:255 — `range_pct = round((high - low) / low * 100, 2) if low > 0 else 0`. Should divide by `(high+low)/2`, not `low`. | FIX: `((high-low) / ((high+low)/2)) * 100`.

[P3] portfolio/fin_fish.py:1226 — `_DEFAULT_BUDGET_SEK = 20_000` — high default for an autonomous fish; sub-1000 SEK or unconfigured operator could trigger a 20k SEK BUY. | FIX: keep default but require explicit opt-in via config.

[P3] portfolio/fin_fish.py:354 — `daily_sigma = avg_range / 1.5 / 100.0` — magic 1.5 (range-to-sigma ratio). For high-kurtosis days this is way off. | FIX: name constant and document derivation.

[P3] portfolio/fin_snipe_manager.py:209-214 (`_round_order_price`) — rounds to 3 decimals if price < 1, else 2. Avanza requires specific tick-size per instrument (often 0.001 / 0.01 / 0.05). Wrong rounding → broker rejects with `tick.size.invalid`. | FIX: read `keyIndicators.tickSize` and snap to it.

[P3] portfolio/iskbets.py:226 — `hold_c = extra.get("_total_applicable", 21) - buy_c - sell_c` — hardcoded 21 total applicable signals. Project has 33 active. | FIX: read from signal_engine.

[P3] portfolio/orb_predictor.py:340 — `idx = int(len(sorted_list) * pct / 100)` — incorrect percentile. For `pct=25, len=4`, idx=1 (correct). For `pct=75, len=4`, idx=3 (last element); statisticians expect interpolation. | FIX: use `numpy.percentile` for consistency with other modules.

[P3] portfolio/exit_optimizer.py:54 — `usdsek: float = 10.85` static fallback diverges from `fin_snipe_manager.py:433` live fetch. If exit_optimizer is invoked outside fin_snipe_manager (testing, ORB postmortem), uses 10.85. | FIX: align via shared `fx_rates.get_usdsek_or_fallback`.

[P3] data/metals_loop.py:7587-7593 — reads `metals_trades.jsonl` with `open()` not `load_jsonl`. Single-line read, not a corruption risk, but inconsistent. | FIX: trivial.

[P3] portfolio/fin_snipe_manager.py:225-228 `_price_matches` — `tol = max(_price_abs_tolerance(ref), ref * 0.0025)`. The 0.25% relative tolerance bumped against tiny absolute tolerance (0.002 for price<1) means for sub-1-SEK warrants two prices 0.0025 SEK apart are matched as equal — could mask a real reprice. | FIX: explicit cap `tol = min(tol, ref * 0.01)`.

[P3] portfolio/exit_optimizer.py:177-185 `_estimate_volatility` — `atr_frac * sqrt(252/14)` is hourly-ATR annualization. If `atr_pct` is daily (varies by upstream), conversion is off by a factor of sqrt(14) ≈ 3.7. | FIX: take `atr_period` parameter, document.

[P3] portfolio/microstructure.py:88-104 (`compute_vpin`) — bucket fill loop: when a single huge trade exceeds `bucket_size`, the inner `while` correctly splits across buckets but the resulting `imbalance` for that synthetic split assigns the FULL trade direction to each split — overstating VPIN. | FIX: weight by `fill / qty`.

[P3] data/metals_loop.py:520-534 `POSITIONS_DEFAULTS` — hardcoded ob_ids and warrant names. As of 2026-05 there's a separate dynamic catalog at `data/oil_warrant_catalog.json` mentioned in the file header — manual sync risk. | FIX: load defaults from catalog file with hardcoded values as fallback only.

## MAYBE — Uncertain

[MAYBE P2] portfolio/fin_snipe_manager.py:1583-1593 — `verify_session()` is called once at the start of `run_cycle`. If session expires mid-cycle (between snapshot fetch and order placement), actions will fail. Not necessarily a bug if `delete_order_no_page` / `place_order_no_page` re-auth, but I haven't traced that path. | MAYBE FIX: confirm avanza_control re-authenticates on 401 mid-cycle.

[MAYBE P2] portfolio/fin_fish.py:367 `chronos_annual` annualization — depends on whether `chronos_24h_pct` is "% over 24h" or "% per 24h-bar of an annualized model". If the latter, current code is right; if the former, ×252 is wrong by orders of magnitude. | MAYBE FIX: trace `forecast_indicators.chronos_24h_pct` semantics in `signals/forecast_*`.

[MAYBE P2] portfolio/exit_optimizer.py:284-296 (`_first_hit_times`) — `argmax` semantics: numpy's argmax on bool array returns 0 if no True (which is then masked by `never_hit`). However, if the FIRST element IS True, also returns 0 — masked incorrectly as never-hit. | MAYBE FIX: `result = first_idx + 1; result[never_hit] = -1`. Actually the code does `+ 1` so first-hit-at-step-1 returns 1, but if `paths[:, 1:]` first column is True at index 0, `argmax` returns 0, `result = 1` (correct), and `never_hit` is False (since `np.any(hits, axis=1) == True`) — so this is actually correct. Confirming it's NOT a bug, just confusing.

[MAYBE P3] data/metals_loop.py concurrency — fast-tick (10s) and main cycle (60s) both touch `_underlying_prices` (line 1117) without explicit lock. CPython GIL makes simple dict assigns safe but multi-step read-modify-write (`if not in: init` patterns) could race. Did not find an obvious data corruption. | MAYBE FIX: defensive lock if extending logic.

[MAYBE P3] portfolio/fin_snipe_manager.py:451-460 (entry_ts bootstrap) — when `entry_ts` first encountered, defaults to `now()`. Comment notes this. Means HOLD_TIME_EXTENDED flag (5h hold) becomes available only ~5h after restart for already-open positions. Operationally acceptable but invisible to the user — if they restart fin_snipe_manager mid-trade they LOSE the hold-time clock. | MAYBE FIX: persist entry_ts in fin_snipe_state.json as soon as a position is observed; restart preserves.

## Counts

- P0: 9
- P1: 19
- P2: 22
- P3: 28
- MAYBE: 4
- Total: 82 findings
