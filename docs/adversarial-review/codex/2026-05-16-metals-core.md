## [P1] Trade Queue Only Runs After Claude Exits
**File:** data/metals_loop.py:7702
**Bug:** `process_trade_queue(page)` is nested under `if claude_proc and claude_proc.poll() is not None` at line 7758.
**Why it matters:** If Claude is disabled, skipped by cooldown, fails before spawning, or orders are queued by another component, pending BUY/SELL intents are never processed. A queued SELL can sit indefinitely while the position keeps moving.
**Fix:** Drain the trade queue every market cycle under the existing queue lock, and also immediately after Claude completion.

## [P1] Transient Session Outage Permanently Drops Pending Trades
**File:** data/metals_loop.py:4537
**Bug:** When `check_session_alive(page)` fails, every pending order is marked `"failed"` and persisted.
**Why it matters:** A temporary 401/session refresh issue permanently discards Layer 2 trade intents. A risk-reducing SELL becomes a dead queue item instead of retrying after re-login.
**Fix:** Leave orders pending with a retryable error state, track retry count/last_error, and only expire by age or explicit operator action.

## [P1] BUY Fill Hardware Stop Path Uses Wrong Return Contract
**File:** data/metals_loop.py:4787
**Bug:** The hardware trailing path treats `place_stop_loss(...)` as a dict (`result.get(...)`), while the same imported function is unpacked as `(ok, stop_id)` elsewhere in this file.
**Why it matters:** A successful BUY can hit `AttributeError: 'tuple' object has no attribute 'get'`, get caught, and leave the new position without broker-side trailing protection.
**Fix:** Normalize `place_stop_loss` to one return type and update this path to handle the actual `(ok, stop_id)` contract.

## [P1] Stop Cancels Are Marked Cancelled Regardless Of Broker Status
**File:** data/metals_loop.py:5009
**Bug:** `_cancel_stop_orders` sets `order["status"] = "cancelled"` after any HTTP response, including 401, 403, 500, or non-404 stop-loss delete failures.
**Why it matters:** Local state can lose a live broker stop. Later sells or replacement stops are planned against corrupted state, risking rejected orders or duplicate protection.
**Fix:** Only mark cancelled on confirmed 2xx/known terminal-not-found responses; otherwise keep `"placed"` and retry/escalate.

## [P2] Stop Fill Polls The Regular Order Endpoint
**File:** data/metals_loop.py:5047
**Bug:** `check_stop_order_fills` queries `/_api/trading-critical/rest/order/{accountId}/{orderId}` for stop-loss order IDs.
**Why it matters:** Stop-loss orders live on Avanza’s stop-loss API surface, so fills can be missed or treated as 404. Local positions can stay active after broker stop execution until a later holdings diff catches it.
**Fix:** Poll the stop-loss status endpoint for stop-loss IDs, or reconcile by orderbook holdings before any further exit action.

## [P1] Emergency Sell Marks Position Sold On Order Acceptance
**File:** data/metals_loop.py:3828
**Bug:** `emergency_sell` sets `pos["active"] = False` when Avanza returns `orderRequestStatus == "SUCCESS"`, which means order accepted, not necessarily filled.
**Why it matters:** An aggressive limit sell at bid can rest unfilled. The loop then deactivates the position and cleans stops, leaving a live holding unmonitored and potentially unprotected.
**Fix:** Track the sell order as open, keep the position active until fill confirmation or holdings reconciliation proves the units are gone.

## [P1] Fishing EOD Sell Is Marked Done Even If Nothing Sold
**File:** data/metals_loop.py:7234
**Bug:** `_eod_fishing_sold_today` is set immediately after `_eod_sell_fishing_positions(page)` without checking whether any emergency sell succeeded.
**Why it matters:** `emergency_sell` returns `False` when disabled or blocked, but the EOD retry flag still advances. Intraday fishing positions can be held overnight with no retry.
**Fix:** Set the EOD-done flag only after all fishing positions are confirmed closed; otherwise retry or escalate.

## [P1] Timed-Out Grid Broker Call Poisons All Future Broker Calls
**File:** portfolio/grid_fisher.py:969
**Bug:** `_safe_session_call` uses a persistent single-worker executor and returns on timeout without cancelling/replacing the stuck worker.
**Why it matters:** One hung Avanza call occupies the only worker forever. Later cancels, stop placements, quotes, and EOD sells queue behind it and silently return defaults/time out.
**Fix:** On timeout, rebuild the executor or isolate calls so a stuck broker request cannot block subsequent trading actions.

## [P1] Grid Stop Replacement Removes Old Stop Before New Stop Exists
**File:** portfolio/grid_fisher.py:1225
**Bug:** `rotate_on_buy_fill` cancels the existing stop before confirming the replacement stop, then sets `inst.stop_loss_id = new_stop_id` even when placement failed.
**Why it matters:** If cancel succeeds and placement fails, the full inventory has no stop. If cancel fails, the live stop ID is still erased locally.
**Fix:** Place replacement protection first where possible, or only clear the old stop ID after confirmed cancel and confirmed replacement.

## [P1] Grid EOD Can Double-Sell Inventory After Failed Cancel
**File:** portfolio/grid_fisher.py:1560
**Bug:** EOD cancellation of armed sell tiers ignores the cancel result and marks each tier `ORDER_CANCELLED` before placing a full-inventory sell.
**Why it matters:** If an old sell order remains live and the new full-inventory EOD sell also fills, Avanza can reject or over-sell relative to holdings.
**Fix:** Require confirmed cancel before placing the full-volume EOD sell; otherwise retry cancel or reduce sell size by still-open order volume.

## [P1] Missed Grid EOD Window Rolls To Tomorrow
**File:** portfolio/grid_fisher.py:293
**Bug:** `minutes_until_eod` returns minutes until tomorrow’s cutoff once today’s cutoff has passed.
**Why it matters:** If the loop misses the small sweep window around 21:55, `tick` no longer sees `<= GRID_EOD_MARKET_SELL_MINUTES_BEFORE`; grid inventory can remain overnight.
**Fix:** Return a negative/expired value after today’s cutoff and have the caller run the EOD flatten path until positions are confirmed flat.

## [P2] Daily Cert Plans Have Zero Executable Quantity
**File:** portfolio/fin_fish.py:789
**Bug:** Daily certificate candidates return `qty = 0` and `warrant_price = 0.0` while still reporting nonzero `invest_sek`, `gain_sek`, and `ev_sek`.
**Why it matters:** A consumer can select the top EV plan but has no valid price or volume to place. This can become a no-op order or a malformed trade request.
**Fix:** Fetch live cert bid/ask for daily certs before emitting executable plans, or exclude them from orderable output.

## [P2] Exit Optimizer Prices All Financing-Level Warrants As LONG
**File:** portfolio/exit_optimizer.py:320
**Bug:** `_compute_pnl_sek` always uses `(underlying - financing_level) * fx` for MINI warrants and has no direction field.
**Why it matters:** MINI SHORT/BEAR products are valued with the wrong sign. A profitable short as underlying falls can be ranked as a loss, producing wrong exit recommendations.
**Fix:** Add explicit position direction and use `financing_level - underlying` for short products.

## [P2] Warrant Portfolio Inverts BEAR P&L
**File:** portfolio/warrant_portfolio.py:96
**Bug:** `warrant_pnl` computes `underlying_change * leverage` without considering LONG vs SHORT direction.
**Why it matters:** For a BEAR certificate, XAG moving from 30 to 27 should be positive P&L; this code reports roughly `-50%` at 5x leverage.
**Fix:** Store direction on holdings and negate the underlying return for short/BEAR instruments.

## [P2] Precompute Erases Live External Data Between Refreshes
**File:** portfolio/metals_precompute.py:137
**Bug:** `_fetch_market_data` initializes every source to `None` and only fills sources due for refresh; non-due cached payloads are not loaded.
**Why it matters:** Runs between refresh intervals rebuild `silver_deep_context.json`/`gold_deep_context.json` without live futures, ETF, COT, FRED, or G/S data, because overlays only happen when `market.get(...)` is present.
**Fix:** Persist and reload full source payloads, not just timestamps, or preserve prior context fields when a source is not due.

## [P2] ORB Backtest Uses Today’s DST Offset For Historical Days
**File:** portfolio/orb_predictor.py:43
**Bug:** `_morning_window_utc()` computes the CET/CEST offset from `datetime.now(UTC)` and the constructor applies that same window to all historical candles.
**Why it matters:** A May run uses 07:00-09:00 UTC for all backtest days; days before the EU DST change should use 08:00-10:00 UTC. The morning range and predicted highs/lows are shifted by one hour.
**Fix:** Determine the Stockholm offset per candle date, not once at process start.

## [P3] Raw JSON File Reads Bypass File Utilities
**File:** data/metals_loop.py:747
**Bug:** The scoped code still uses raw `open`/`json.load` and raw JSONL parsing in several places instead of `file_utils.load_json`/`load_jsonl`.
**Why it matters:** Concurrent writers can expose partial files or malformed tail lines, and callers silently fall back to defaults or drop entries.
**Fix:** Replace JSON state reads with `load_json`/`load_jsonl`; keep raw text reads only for non-JSON prompt/log files.

## SUMMARY P1=10 P2=6 P3=1