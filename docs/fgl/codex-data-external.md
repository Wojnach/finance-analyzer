# Codex Adversarial Review — data-external subsystem

Reviewer: codex / gpt-5.4 (xhigh reasoning)
Date: 2026-05-09
Branch: review/2026-05-08-data-external (off empty-baseline)

Format: `[Pri] file.py:line — problem | FIX: repair`

---

[P1] portfolio/data_collector.py:322 — The BUG-179 timeout is ineffective because leaving the `ThreadPoolExecutor` context calls `shutdown(wait=True)`, so cancelled hung fetches still block the collection cycle. | FIX: manage the executor lifecycle explicitly and shut it down with `wait=False, cancel_futures=True` or move each fetch into a killable worker/process.
[P2] portfolio/data_collector.py:173 — `fetch_vix()` touches yfinance outside the shared `yfinance_lock`, so concurrent ticker/news/VIX calls can race and silently return empty or corrupted data. | FIX: wrap the full `^VIX` `Ticker/history` block in `portfolio.shared_state.yfinance_lock`.
[P2] portfolio/alpha_vantage.py:31 — The Alpha Vantage daily-budget counter lives only in RAM, so any intraday restart resets usage to zero and can blow past the hard daily quota without warning. | FIX: persist the budget date and used-count to disk and reload it on startup before issuing requests.
[P2] portfolio/alpha_vantage.py:281 — `_daily_budget_used` increments only on successful parses even though failed or empty Alpha Vantage requests still consume quota, so the tracker undercounts and keeps sending calls after the plan is exhausted. | FIX: reserve or increment budget on every outbound AV request and refund only when no request was actually sent.
[P2] portfolio/sentiment.py:192 — NewsAPI calls are counted only when articles are returned, so empty-result searches silently consume quota while `newsapi_quota_ok()` still thinks budget remains. | FIX: track every successful NewsAPI request regardless of article count.
[P1] portfolio/sentiment.py:134 — Yahoo headline fetches use yfinance outside the shared lock, so concurrent sentiment workers can intermittently lose all stock headlines and vote on partial news. | FIX: guard the entire `Ticker/news` access with `yfinance_lock` and let hard failures trigger the existing fallback path.
[P1] portfolio/bert_sentiment.py:447 — Per-text inference failures are converted into neutral placeholders instead of raising, which silently biases the primary sentiment vote toward HOLD and prevents `sentiment.py` from falling back to the subprocess model. | FIX: re-raise after logging or fail the whole call so the caller can use its existing subprocess fallback.
[P2] portfolio/earnings_calendar.py:53 — Earnings Calendar sends Alpha Vantage `EARNINGS` requests outside `alpha_vantage.py`’s budget accounting, so the shared quota can be exhausted while fundamentals refresh still believes capacity remains. | FIX: route AV earnings calls through a shared budget/reservation helper or persist a cross-module counter that this path increments.
[P1] portfolio/earnings_calendar.py:102 — The yfinance fallback hits `Ticker.calendar` without `yfinance_lock`, so concurrent calls can fail and drop the earnings gate exactly when it is supposed to block trades around reports. | FIX: wrap the entire yfinance calendar fetch/parsing block in the shared yfinance lock.
[P1] portfolio/earnings_calendar.py:177 — A transient fetch miss is cached as `None` for 24 hours, which silently disables the earnings BUY gate for the rest of the day after a single network or parser hiccup. | FIX: cache negative results for a very short TTL or only cache concrete earnings dates.
[P2] portfolio/macro_context.py:230 — `avg20` includes the same completed candle stored in `last_vol`, which dilutes true volume spikes and suppresses legitimate volume-confirmation BUY/SELL signals. | FIX: compute the comparison average from the 20 candles preceding `vol.iloc[-2]`, not from a window that includes it.
[P1] portfolio/macro_context.py:313 — The 2-year leg uses `2YY=F` (a Treasury futures price) as if it were a yield, so `yield_pct`, `change_5d`, and `spread_2s10s` are materially wrong. | FIX: fetch an actual 2-year yield series such as FRED `DGS2` or a proper yield index and keep units consistent with the 10y/30y legs.
[P1] portfolio/market_health.py:255 — `detect_ftd_state()` reprocesses the same daily bar on every hourly refresh and increments `rally_day` again, allowing false FTD confirmation intraday without any new trading day. | FIX: persist the last processed market date and skip state transitions when the newest daily bar date has not changed.
[P2] portfolio/market_health.py:450 — Persisting `ftd_day_offset` as a rolling-array index instead of a date makes the failure-window aging wrong once the 90-day window shifts, so FTD states do not mature or expire on real calendar days. | FIX: persist the actual FTD session date or timestamp and compare dates when evaluating the failure window.
[P3] portfolio/futures_data.py:83 — The open-interest-history cache key omits `limit`, so a small earlier request can poison later larger requests with truncated history until TTL expiry. | FIX: include `limit` in the cache key.
[P3] portfolio/futures_data.py:112 — The long/short-ratio cache key omits `limit`, so callers requesting different history lengths can receive stale truncated data from a prior smaller fetch. | FIX: include `limit` in the cache key.
[P3] portfolio/futures_data.py:141 — The top-trader-position-ratio cache key omits `limit`, so later callers can be served the wrong window length from cache. | FIX: include `limit` in the cache key.
[P3] portfolio/futures_data.py:170 — The top-trader-account-ratio cache key omits `limit`, so cached results can silently return fewer rows than the caller asked for. | FIX: include `limit` in the cache key.
[P3] portfolio/futures_data.py:198 — The funding-history cache key omits `limit`, so one short request can silently truncate later funding-history reads until the cache expires. | FIX: include `limit` in the cache key.
P0=0 P1=7 P2=7 P3=5
