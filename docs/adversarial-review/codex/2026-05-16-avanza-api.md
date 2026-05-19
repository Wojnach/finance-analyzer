## [P1] Non-idempotent session POST retry can duplicate live orders
**File:** portfolio/avanza_session.py:344
**Bug:** `api_post()` retries mutating POSTs after browser-dead errors; `place_order()` and `place_stop_loss()` use this path.
**Why it matters:** If Avanza accepts a BUY/SELL/stop-loss POST and Chromium dies before the response is read, the retry submits the same real-money order again.
**Fix:** Do not auto-retry non-idempotent POSTs. Reconcile broker state first, or add an idempotency/order-intent key and only retry reads.

## [P1] ResilientPage retries arbitrary mutating JavaScript
**File:** portfolio/avanza_resilient_page.py:168
**Bug:** `evaluate()` retries every script after relaunch, with no distinction between read scripts and order-placement/delete scripts.
**Why it matters:** A `page.evaluate()` POST can reach Avanza, then fail locally with `TargetClosedError`; the retry can place/cancel the order a second time.
**Fix:** Make retries opt-in for read-only calls, or require mutating callers to handle recovery by checking broker state before resubmitting.

## [P1] Pending orders file has lost-update races
**File:** portfolio/avanza_orders.py:132
**Bug:** `request_order()` performs load/append/atomic-write with no process/thread lock; `check_pending_orders()` also rewrites the same file.
**Why it matters:** Concurrent order requests or confirmation checks can drop pending orders, resurrect executed orders as pending, or overwrite failure/execution state.
**Fix:** Guard the entire read-modify-write cycle with a cross-process file lock and validate the loaded shape before writing.

## [P1] Telegram offset is persisted before execution state
**File:** portfolio/avanza_orders.py:343
**Bug:** `_check_telegram_confirm()` advances the Telegram offset before `_execute_confirmed_order()` runs and before `_save_pending()` persists the result.
**Why it matters:** A crash after Avanza executes but before pending state is saved leaves the order pending while the CONFIRM update is unreplayable; a manual retry can duplicate the trade.
**Fix:** Persist an `executing` state before advancing the offset, then persist the final broker result immediately after the Avanza call.

## [P1] Page-based mutating facade bypasses account whitelist
**File:** portfolio/avanza_control.py:132
**Bug:** `place_order()`, `place_stop_loss()`, and delete helpers accept caller-supplied `account_id` and forward it without checking `ALLOWED_ACCOUNT_IDS`.
**Why it matters:** Any caller bug or LLM-supplied account can place real orders or delete stops on a non-approved Avanza account.
**Fix:** Enforce the same whitelist used by `avanza_session` before every mutating page-based operation.

## [P1] Stop-loss delete reports success on HTTP failures
**File:** portfolio/avanza_control.py:404
**Bug:** `delete_stop_loss_no_page()` returns `True` for any `_api_delete()` dict unless it has `errorCode`; `_api_delete()` returns dicts for 500/403 with `ok=False`.
**Why it matters:** A failed stop-loss cancel can be treated as cleared, so a dependent sell can proceed against still-encumbered volume or leave stale protection in place.
**Fix:** Require `result["ok"]` or explicit `http_status` 2xx/404 before returning success.

## [P1] Modular Avanza client accepts arbitrary configured account
**File:** portfolio/avanza/client.py:65
**Bug:** `account_id` is loaded from config without any whitelist.
**Why it matters:** The newer `portfolio.avanza.trading` path uses this account for live trades, bypassing the hardcoded account guard in the legacy path.
**Fix:** Centralize and enforce `ALLOWED_ACCOUNT_IDS` in `AvanzaClient.get_instance()` before caching the singleton.

## [P1] Modular trading path has no order lock
**File:** portfolio/avanza/trading.py:84
**Bug:** `place_order()`, `modify_order()`, `cancel_order()`, and stop-loss mutations call Avanza directly without `avanza_order_lock`.
**Why it matters:** This path can race the legacy/session/page paths and submit overlapping orders from the same buying-power snapshot.
**Fix:** Wrap every mutating broker call in `avanza_order_lock` with operation-specific labels.

## [P1] Modular order path lacks max-order guard
**File:** portfolio/avanza/trading.py:74
**Bug:** `place_order()` enforces the 1000 SEK minimum but not the legacy 50,000 SEK maximum exposure guard.
**Why it matters:** A malformed volume/price from Layer 2 can place an outsized real-money order through the modular API.
**Fix:** Apply the same max-order total guard as `avanza_session._place_order`, preferably from shared config.

## [P1] Modular stop-loss allows dangerous zero/invalid values
**File:** portfolio/avanza/trading.py:213
**Bug:** `place_stop_loss()` does not validate `volume >= 1`, positive trigger values, or `sell_price > 0` for non-trailing monetary stops.
**Why it matters:** A `LESS_OR_EQUAL` stop with `sell_price=0` can become an unintended market sell, or invalid values can silently fail after local logic assumes protection exists.
**Fix:** Port the validation from `avanza_session.place_stop_loss()` into this path.

## [P1] Barrier distance uses warrant price instead of underlying
**File:** portfolio/avanza/scanner.py:243
**Bug:** `barrier_dist_pct` is computed as `abs(last - barrier) / last`, where `last` is the instrument quote, not the underlying price.
**Why it matters:** A mini future with underlying 100, barrier 95, warrant price 5 is reported as 1800% from barrier instead of 5%, causing unsafe instrument ranking.
**Fix:** Compute barrier distance from `underlying_price`, and make direction-aware distance checks for bull/bear products.

## [P2] Session positions are not account-filtered
**File:** portfolio/avanza_session.py:682
**Bug:** `get_positions()` returns every position from `/_api/position-data/positions` without filtering to `ALLOWED_ACCOUNT_IDS`.
**Why it matters:** Risk, exposure, and sell sizing can include holdings from other Avanza accounts while trades still route to the default account.
**Fix:** Filter by whitelisted account IDs by default, with an explicit opt-in for cross-account reads.

## [P2] BankID price shape is parsed as zero
**File:** portfolio/avanza_tracker.py:67
**Bug:** `fetch_avanza_prices()` reads `lastPrice` and `changePercent`, but the BankID session path returns market-guide data under nested `quote` fields.
**Why it matters:** With session auth active, tracked Avanza instruments can be recorded as `0.0` price/change, corrupting downstream signals.
**Fix:** Parse both legacy flat fields and nested `quote.last.value` / `quote.changePercent.value`.

## [P2] Missing account is silently converted to zero buying power
**File:** portfolio/avanza/account.py:92
**Bug:** `get_buying_power()` returns `AccountCash(0,0,0)` when the account is absent.
**Why it matters:** API shape drift or wrong account ID is indistinguishable from a real zero-cash account, causing silent wrong sizing/halts.
**Fix:** Return `None` or raise a typed exception for account-not-found and malformed overview responses.

## [P2] Account-check timeout can still hang indefinitely
**File:** portfolio/avanza_account_check.py:174
**Bug:** `_api_get_categorized_accounts()` calls `future.result(timeout=30)` inside a `ThreadPoolExecutor` context manager; on timeout, context shutdown waits for the stuck worker.
**Why it matters:** A stuck Playwright/API call can block startup despite the 30-second timeout.
**Fix:** On timeout, cancel the future and shut down with `wait=False`, or run the check in a reusable worker with hard process-level timeout.

## [P2] Tick rounding still uses floating-point floor math
**File:** portfolio/avanza/tick_rules.py:87
**Bug:** `price_int = price * multiplier` remains a float and is passed into `math.floor`.
**Why it matters:** Exact valid prices like `0.29` can become `28.999999...` after scaling and round down one extra tick, producing worse or invalid limit prices.
**Fix:** Use `Decimal` or convert from formatted integer strings before floor/ceil.

## [P2] Scanner parallelizes a shared TOTP client
**File:** portfolio/avanza/scanner.py:310
**Bug:** The scanner marks the TOTP Avanza client as thread-safe and uses one shared client/session across worker threads.
**Why it matters:** Shared HTTP session state can race under parallel detail fetches, returning mixed or failed instrument details used for ranking.
**Fix:** Use one client/session per worker or serialize calls behind a lock.

## [P2] BankID scanner ignores requested instrument type
**File:** portfolio/avanza/scanner.py:77
**Bug:** The BankID fallback `_search()` ignores `itype_str` and posts only `{"query", "limit"}`.
**Why it matters:** `instrument_type="warrant"` can return certificates or other products, and the later pipeline does not enforce the requested type.
**Fix:** Pass the type filter supported by the endpoint, or filter returned hits by type before fetching details.

## [P2] Stream stop can orphan a live daemon thread
**File:** portfolio/avanza/streaming.py:108
**Bug:** `stop()` joins for 5 seconds and then sets `_thread = None` regardless of whether the thread actually exited.
**Why it matters:** A blocked read loop can keep running while `start()` creates a second stream, duplicating callbacks for order/deal events.
**Fix:** Only clear `_thread` after `is_alive()` is false; otherwise report failure and prevent restart until the old thread exits.

## SUMMARY
P1=11 P2=8 P3=0