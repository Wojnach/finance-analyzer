## [P1] Timeframe timeout cannot stop hung worker
**File:** portfolio/data_collector.py:323
**Bug:** The `ThreadPoolExecutor` context manager still waits for running futures after `as_completed(..., timeout=...)` times out and `f.cancel()` runs.
**Why it matters:** One hung yfinance/API fetch can stall the trading loop indefinitely despite `_TF_POOL_TIMEOUT=60`.
**Fix:** Manage executor shutdown explicitly with `shutdown(wait=False, cancel_futures=True)` on timeout and add hard timeouts around yfinance calls.

## [P1] Failed refresh can overwrite good candle history
**File:** portfolio/data_refresh.py:85
**Bug:** `df.to_feather(path)` writes directly to the target even when `download_klines` stopped early after `fetch_with_retry` returned `None`.
**Why it matters:** A transient Binance outage can replace a good history file with partial or empty data.
**Fix:** Validate coverage before writing; write to a temp file, fsync, then atomic replace only on complete refresh.

## [P1] Earnings fetch failures disable the earnings gate for 24h
**File:** portfolio/earnings_calendar.py:177
**Bug:** `None` results are cached for the full 24h TTL.
**Why it matters:** If providers fail or rate-limit before NVDA earnings tomorrow, `should_gate_earnings()` returns `False` and BUY signals are allowed into a binary event.
**Fix:** Do not cache `None` as a valid no-earnings result; use a short retry TTL or fail closed for unknown earnings proximity.

## [P1] FX fallback returns unsafe prices as if live
**File:** portfolio/fx_rates.py:60
**Bug:** Stale cached rates are returned indefinitely, and cold-start failure returns hardcoded `10.50`.
**Why it matters:** SEK sizing/valuation can be materially wrong with no machine-readable stale flag, so order sizes can be wrong.
**Fix:** Return structured `{rate, stale, age}` or raise after a max stale age; block real-money sizing on hardcoded fallback.

## [P1] Config loading fails open with stale settings
**File:** portfolio/api_utils.py:30
**Bug:** Config uses raw `open`/`json.load`, catches all exceptions, and returns the previous `_config_cache` if parsing/stat/read fails.
**Why it matters:** A bad config edit intended to disable trading or reduce risk can be ignored while the process continues using old live settings.
**Fix:** Use the project file utility for JSON, log failures, and fail closed for critical config instead of returning stale settings silently.

## [P2] Futures refresh downloads spot candles
**File:** portfolio/data_refresh.py:31
**Bug:** Futures files are populated from `BINANCE_BASE` spot `/klines`, not Binance FAPI futures klines.
**Why it matters:** Files named `*-futures.feather` contain spot data, so futures backtests/signals silently use the wrong market.
**Fix:** Use `BINANCE_FAPI_BASE` for futures refreshes or rename/output as spot data.

## [P2] Alpha Vantage budget accounting is wrong
**File:** portfolio/alpha_vantage.py:281
**Bug:** `_daily_budget_used` increments only on successful normalized responses and is only in memory.
**Why it matters:** Failed/rate-limited calls still burn Alpha Vantage quota, and restarts reset the local counter, causing repeated quota exhaustion and stale fundamentals.
**Fix:** Count attempted requests, persist the daily budget state, and reset by date from persisted state.

## [P2] Earnings calls bypass Alpha Vantage budget
**File:** portfolio/earnings_calendar.py:49
**Bug:** The earnings endpoint explicitly bypasses `alpha_vantage.py` budget tracking.
**Why it matters:** A stock universe refresh can consume the same 25/day quota that fundamentals expects to own, making both datasets stale.
**Fix:** Centralize Alpha Vantage budget reservation across OVERVIEW and EARNINGS before making any request.

## [P2] Partial on-chain fetches are cached as fresh
**File:** portfolio/onchain_data.py:224
**Bug:** Any single successful metric sets `any_success=True`, then the partial result is saved with a fresh 12h timestamp.
**Why it matters:** If five of six BGeometrics metrics fail, missing NUPL/SOPR/netflow data will not be retried for 12 hours.
**Fix:** Cache per metric or mark partial fetches with a short TTL and missing-field retry policy.

## [P2] Futures cache keys ignore requested limit
**File:** portfolio/futures_data.py:83
**Bug:** History cache keys include ticker/period but omit `limit`; funding history also omits `limit`.
**Why it matters:** A first call with `limit=5` can poison a later `limit=100` call with only five rows, weakening trend calculations.
**Fix:** Include all behavior-changing parameters in cache keys.

## [P2] Yahoo/yfinance news fetch bypasses shared yfinance lock
**File:** portfolio/sentiment.py:172
**Bug:** `_fetch_yahoo_headlines()` calls yfinance without the shared `yfinance_lock`.
**Why it matters:** Concurrent ticker workers can hit yfinance’s non-thread-safe paths and produce missing or failed sentiment inputs.
**Fix:** Wrap yfinance calls in the same shared lock used by other modules.

## [P2] A/B sentiment entries are dropped before durable write
**File:** portfolio/sentiment.py:473
**Bug:** `flush_ab_log()` snapshots and clears `_pending_ab_entries` before `_log_ab_result()` writes.
**Why it matters:** Disk/full/file-lock/write errors lose the pending shadow results permanently.
**Fix:** Clear entries only after successful append, or requeue failed entries for the next flush.

## [P2] Stock shadow backfill only gives MSTR stock-calendar tolerance
**File:** portfolio/sentiment_shadow_backfill.py:95
**Bug:** `_STOCK_TICKERS` contains only `MSTR`; all other stock tickers get a 2h tolerance.
**Why it matters:** Friday signals for NVDA/AMD/AAPL often target weekend timestamps and get skipped instead of matching Monday prices.
**Fix:** Use the full stock universe or market-calendar-aware lookup tolerances for all equities.

## [P2] Backfill idempotency races across processes
**File:** portfolio/sentiment_shadow_backfill.py:188
**Bug:** Existing keys are loaded once, then appends happen later with no process lock around check-and-write.
**Why it matters:** Two backfill runs can both miss the same key and append duplicate outcome rows, corrupting model accuracy.
**Fix:** Guard the whole backfill with a file/process lock or rewrite outcomes atomically after dedupe.

## [P2] Macro event times ignore ET and DST
**File:** portfolio/econ_dates.py:155
**Bug:** CPI/NFP/GDP/FOMC event times are hardcoded as `14:00 UTC`.
**Why it matters:** 8:30 AM ET releases are 13:30 UTC in standard time and 12:30 UTC in daylight time, so blackout windows can miss the actual release.
**Fix:** Store event-specific local ET times and convert with `zoneinfo.ZoneInfo("America/New_York")`.

## SUMMARY
P1=5 P2=10 P3=0