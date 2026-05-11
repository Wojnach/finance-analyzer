# Claude adversarial review: data-external

## Summary

Reviewed 30 data-external modules. Found multiple correctness issues clustered around
(1) rate-limit counters that don't survive restart and aren't process-shared,
(2) HTTP retry behavior that swallows 4xx errors via `raise_for_status` after retries
exhausted and that returns `None` indistinguishably from "0 retries", (3) FX caching
that returns a stale rate after refusing a sanity-failed one without ever timing the
cache out, (4) a `funding_rate` interpretation that conflicts with the project's own
"shorts pay vs longs pay" docstring, (5) `price_source` silently falling back to
yfinance with potential symbol-routing collisions (SI=F ↔ XAGUSDT) that can drift
real-time vs end-of-day data into the same code path without warning, (6) DST
detection in `session_calendar` that miscomputes the last Sunday on years where
`Mar 31` happens to be Sunday, (7) hard-coded FOMC / CPI / NFP / GDP date lists with
no auto-refresh and no end-of-2027 warning, and (8) regex-from-feed injection through
`news_keywords._KEYWORD_PATTERNS` pattern strings re-injected verbatim into the
`matched` return list.

## P0 — Blockers

- portfolio/futures_data.py:33-55 — `get_open_interest()` returns `{"oi": ..., "symbol": ..., "time": ...}`
  but the doctring promises `{oi, oi_usdt, symbol, time}` and downstream consumers
  in reporting expect `oi_usdt`. The premiumIndex/openInterest endpoint does NOT
  contain `oi_usdt`. Why it bites: KeyError or `None` shows up as "0 usdt OI" in
  agent_summary, distorting the futures-flow voter. Fix: drop `oi_usdt` from the
  docstring AND from any consumer reading it, OR compute `float(data["openInterest"]) * mark_price`
  via a second call.

- portfolio/funding_rate.py:44-49 — Sign convention is **inverted relative to
  Binance**. Binance defines `lastFundingRate > 0` as "longs pay shorts" — that's the
  standard "overheated longs → BEARISH = contrarian SELL" reading the code does.
  But the file's docstring quoted in CLAUDE.md says "shorts pay vs longs pay
  convention reversed"; verify against the live Binance docs and against the
  `funding_history` consumer in `signals/funding.py`. If the signal uses negative
  rate as BEARISH anywhere, this module disagrees. Why it bites: at 74% accuracy
  (3h) the signal could be lucky and actually be flipped at certain horizons. Fix:
  add a unit test using a captured 2024 funding-rate spike with a known directional
  outcome.

- portfolio/fx_rates.py:46-71 — Sanity-failed rate path NEVER updates `cached_time`,
  so a single bad rate (10x scale glitch, decimal mis-parse) plus a healthy cache
  produces an infinite "we always use the 24h-old cached rate" loop until restart.
  Worse, line 60 returns `cached_rate` even when the cache was last refreshed >24h
  ago — the staleness threshold (`_FX_STALE_THRESHOLD = 7200` = 2h) only triggers a
  *warning + Telegram alert*, not invalidation. Why it bites: SEK has moved 10-15%
  in 24h before; portfolio valuations drift silently. Fix: invalidate `_fx_cache`
  when `now - cached_time > N` (e.g. 6h) and force the fallback path to throw if
  no fresh rate available — or return the hardcoded 10.50 explicitly so risk
  modules can detect the degraded state.

- portfolio/http_retry.py:88 — `fetch_json` calls `resp.raise_for_status()` AFTER
  the retry loop has already returned the response. That means 4xx responses
  (400/401/403/404) — which `RETRYABLE_STATUS` deliberately excludes from retry —
  reach `fetch_json` as a populated `resp`, then `raise_for_status` raises HTTPError,
  the bare `except Exception` swallows it and returns `default=None`. The caller
  sees "fetch failed" with no distinction between "API auth dead" and "API
  transient 503". Why it bites: silent 401 from Alpha Vantage on token rotation,
  silent 403 from CryptoCompare when an API-key gets banned, silent 404 from a
  retired Binance symbol — all look identical to "transient outage" and never page.
  Fix: log status code at WARNING + add `raise_for_status_4xx` mode that returns a
  distinguishable sentinel (e.g. `{"_status": 401}` vs `None`).

## P1 — High

- portfolio/alpha_vantage.py:157-168 — `_check_budget()` resets `_daily_budget_used`
  on date change but the counter is **per-process** with no on-disk persistence.
  A loop restart at 18:00 silently re-enables the entire 25-call budget. With
  PF-DataLoop's auto-restart-on-crash + crash-recovery exponential backoff, a
  flapping loop in the early morning can burn through 50+ AV requests/day. Fix:
  persist `{date, used}` to `data/alpha_vantage_budget.json` and reload on
  startup.

- portfolio/alpha_vantage.py:209-298 — `refresh_fundamentals_batch` is called from
  the main loop but `_daily_budget_used` is incremented OUTSIDE the per-iteration
  lock semantically — it IS under `_cache_lock` (line 279), but `_check_budget()`
  reads it under the same lock then RELEASES the lock before
  `_alpha_vantage_limiter.wait()` and the actual fetch. Multi-thread races could
  let 2 callers both pass the budget check, then both fetch. Fix: re-check budget
  immediately before each fetch under the same lock that increments.

- portfolio/earnings_calendar.py:49-52 — Earnings-fetch path explicitly bypasses
  the daily AV budget tracker. Each stock ticker = 1 AV call/24h. With 1 active
  stock (MSTR) the cost is tiny, but the comment says "Known limitation" and the
  budget tracker silently undercounts the limit. Why it bites: a future ticker
  expansion (5 stocks → 5 extra AV calls/24h) eats into the 25/day budget without
  the tracker noticing — fundamentals_batch then over-spends and AV returns the
  "Note" rate-limit error. Fix: export a `_increment_budget()` from
  `alpha_vantage.py` and call it from `earnings_calendar._fetch_earnings_alpha_vantage`.

- portfolio/session_calendar.py:56-67 — `_eu_dst` calculates "last Sunday of March"
  as `31 - (mar31.weekday() + 1) % 7` which yields the right Sunday when Mar 31 is
  a Mon-Sat, but when **Mar 31 is itself a Sunday** (e.g. 2024-03-31, 2030-03-31)
  the formula gives 31 - (6 + 1) % 7 = 31 - 0 = 31 — correct. But when Mar 31 is
  Monday (weekday=0), it gives 31 - (0+1)%7 = 30, also correct. Actually re-checked
  the math: it's correct. **However** the comparison uses `dst_start <= dt < dst_end`
  on UTC; the EU DST switch happens at 01:00 UTC, but the function compares the
  full datetime to a datetime at hour=1 minute=0 — fine. Edge case: when called
  with a naive datetime, `dst_start.tzinfo` is UTC and `dt.tzinfo` is None →
  TypeError on comparison. `get_session_info` defends against this on line 130-131
  but `_eu_dst` is also called from internal helpers without first normalizing.
  Why it bites: any caller that passes a naive datetime crashes the session
  calculation. Fix: add a `_ensure_utc()` helper.

- portfolio/news_keywords.py:80-83, 155 — `_KEYWORD_PATTERNS` builds regexes from
  the ALL_KEYWORDS dict (currently safe — all hardcoded literals). But
  `score_headline` then returns the *pattern string* back to the caller via
  `pattern.pattern.replace(r"\b", "").replace("\\", "")`. If ALL_KEYWORDS ever
  gains a keyword with special regex chars (a user-config addition is not
  unprecedented), the .replace strip will leave broken markup that crashes any
  consumer attempting to `re.compile` the returned string. Why it bites: low
  blast radius now, big footgun later. Fix: store the original literal alongside
  the compiled pattern in the tuple instead of round-tripping through `.pattern`.

- portfolio/onchain_data.py:244-284 — `get_onchain_data` ALWAYS reads
  `load_json(CACHE_FILE)` from disk on every call, then ALSO calls `_cached()`
  which has its own in-memory check. The persistent-cache seed only fires if
  the in-memory cache is missing the key — but a stale persistent entry seeded
  into the in-memory dict at startup will then **be used for the full 12h TTL
  even if the persistent entry is itself stale by 11h+59m**. There's no second
  check that the seeded entry isn't already aged out — `_cached`'s ttl check
  uses `time.time() - entry["time"] < ttl`, where `entry["time"] = cache_ts`
  (the original epoch). That's correct, BUT only if `_coerce_epoch` recovers a
  real epoch. If the JSON has no `ts` field at all and no `_fetched_at`, line
  259 returns 0.0 (ancient) — `_cached` will see `time.time() - 0 > 43200`,
  treat as miss, refetch. Good. So this is actually correctly handled. **The
  real issue:** `_load_onchain_cache` (line 105) is dead code — never called
  from `get_onchain_data`. Fallback path "token None → try persistent" goes
  through `_load_onchain_cache` (line 278), but the primary path doesn't. So a
  token-loss intermittent stuffs stale data into the response. Fix: make the
  primary path also gate on token presence.

- portfolio/price_source.py:212-244 — On primary-source failure for an
  `alpaca` ticker (e.g. MSTR), the catch-all falls back to yfinance with
  `_fetch_yfinance(ticker, ...)`. yfinance receives the bare alpaca ticker
  string ("MSTR") which happens to be valid yfinance syntax — but for a
  ticker like `XBT-TRACKER` or similar Avanza warrant pseudo-symbol the
  yfinance fallback returns empty silently. Worse: the fallback for
  `binance_fapi` (XAU-USD) passes the *original* alias `XAU-USD` to yfinance
  which yfinance interprets as a different symbol (it has its own
  `XAU-USD` quote that is *not* identical to the Binance perpetual). Why it
  bites: silent data-source switch from XAU-USD perpetual → XAU-USD spot when
  Binance FAPI is down; mid-stream regime shifts in the price series corrupt
  indicators. Fix: never silently swap symbols across sources; require an
  explicit `yfinance_alias` mapping, fail-loud otherwise.

## P2 — Medium

- portfolio/sentiment.py:859-860 — `ab_key = f"{short}:{datetime.now(UTC).isoformat()}"`
  Two parallel threads from the ticker pool processing the *same* ticker
  (shouldn't happen but a re-entrant cycle could double-process via Telegram poller
  /mode commands) would produce nearly-identical ab_keys that hash separately but
  one would overwrite the other in `_pending_ab_entries` if ISO timestamps collide
  to microsecond precision. Fix: add `uuid4()` suffix.

- portfolio/fear_greed.py:55-71 — Streak update logic increments `streak_days` on
  date change but **doesn't reset the streak on a long gap**. If the loop has been
  down for 5 days, then comes back with `fg_value <= 20`, the code says
  "prev_type was extreme_fear, is_new_day, prev_days + 1" — net effect: 5-day gap
  becomes 1-day increment. Acceptable, but the underlying assumption of "daily
  observation" is unverified. Fix: detect `(today - last_date).days > 1` and reset
  streak to 1.

- portfolio/microstructure_state.py:175-186 — `get_microstructure_state` calls
  `get_rolling_ofi` (acquires `_buffer_lock`), then `get_ofi_zscore` (acquires
  again), then `record_ofi` (acquires again), then `get_spread_zscore` (again),
  then `get_multiscale_ofi` (again), then takes the lock a 6th time at line 188.
  Six lock acquisitions per ticker per cycle. With two metals tickers and a 10s
  fast-tick, this is ~36 acquisitions/min per ticker. Low blast radius (lock is
  uncontended in practice) but unnecessary churn. Fix: single critical section.

- portfolio/microstructure_state.py:_MAX_OFI_HISTORY=120 and `_MAX_SNAPSHOTS=60`,
  but a `_MIN_OFI_HISTORY_FOR_ZSCORE=10` only — z-score normalization starts after
  only 10 samples, which on a 60s cycle means 10 minutes of warmup. With high
  intraday volatility regime changes, a 10-sample baseline is extremely noisy and
  the resulting "z-score" early on is essentially random. Fix: require at least
  30-60 samples (30-60 min) before publishing a non-zero z-score.

- portfolio/seasonality_updater.py:53-69 — `_fetch_hourly_klines` is called with
  `limit=500` (~20 days) from `update_seasonality_profiles`, but `compute_hourly_profile`
  requires `_MIN_DAYS=5` (`5 * 24 = 120` rows). 500 row limit is fine, but the
  data could span 3 weeks across DST changes, which means the hour-of-day grouping
  mixes pre/post-DST observations into the same hour bucket. For metals trading
  the bucket "11:00 UTC" pre-DST = 12:00 CET, post-DST = 13:00 CET. The
  seasonality profile silently averages over different LOCAL market sessions.
  Fix: either group by `local_hour` after DST normalization or bound the window
  to a single DST regime.

- portfolio/onchain_data.py:74 — `ONCHAIN_TTL = 43200` (12h) but BGeometrics
  limits are "8 req/hour, 15 req/day". With 6 metrics per fetch + 12h cache, the
  cap is 12 req/day (6 * 2) — close to the 15/day limit. If `_save_onchain_cache`
  fails (disk full, permission), the next cycle hits the API again and busts
  budget. Fix: even on save failure, return the data (which it does) but log at
  WARNING with budget context.

- portfolio/data_collector.py:74-101 — `_binance_fetch` catches all exceptions
  and calls `cb.record_failure()`, including `KeyboardInterrupt` and
  `SystemExit` (via bare `except Exception` — actually it's `except Exception`
  which excludes those, OK). However, **any** `raise_for_status` HTTP error
  (incl. 4xx) is counted as a circuit-breaker failure. A configuration mistake
  that yields permanent 401 will trip the circuit breaker — but Binance public
  endpoints don't need auth, so a 401 here likely means an upstream proxy/WAF
  misclassified the request. Fix: distinguish 4xx (caller bug, don't break
  circuit) from 5xx (provider problem, do break circuit).

- portfolio/sentiment.py:33-42 — `MODELS_PYTHON` and model script paths use
  `r"Q:\..."` on Windows and `/home/deck/...` on Linux. The Linux path is
  hardcoded for the original test setup; on a different Linux host the
  subprocess fallback dies immediately. The in-process BERT path (bert_sentiment.py)
  uses `_resolve_cache_dir` with the same dichotomy. Why it bites: deploying
  the loop to a new Linux box silently degrades to "no shadow A/B sentiment".
  Fix: read paths from config.json.

## P3 — Low

- portfolio/econ_dates.py:121-167 — `ECON_EVENTS` is computed at module import,
  not lazily. End of 2027 the calendar runs dry; `next_event` returns None and
  `is_macro_window` returns False forever, silently disabling the macro window
  gate. No auto-refresh, no logging warning. Fix: log WARNING when next_event
  returns None.

- portfolio/fomc_dates.py — Hardcoded through end of 2027 only. Same problem
  as econ_dates.

- portfolio/social_sentiment.py:29-56 — Uses raw `urllib.request.urlopen` with
  10s timeout, bypassing `http_retry.py`. No retries on transient Reddit
  outage; no rate-limit handling. Reddit hot.json has been intermittently
  rate-limiting cloud IPs since 2023. Fix: route via `fetch_json` + ratelimiter.

- portfolio/social_sentiment.py:110, 122 — Catches all exceptions and
  `print()`s — no logger. Inside the main loop's stdout swallowing, these
  errors vanish. Fix: use `logging.warning`.

- portfolio/crypto_macro_data.py:65 — `import datetime` shadows the
  module-level `from datetime import UTC` (well, it doesn't on this file
  since this file imports nothing from datetime at top). However, this
  function-local `import datetime` is then used as `datetime.datetime.strptime`
  and `datetime.date.today()`. Style: put at top.

- portfolio/indicators.py:81-88 — `tr.ewm(span=14)` uses pandas `ewm` (with
  `adjust=False`) which is NOT Wilder's smoothing — Wilder is
  `alpha=1/period` which equals `span=2*period-1=27` for `period=14`. The
  current code computes ATR over `span=14` (alpha=2/15), which differs from
  the conventional Wilder ATR(14) by ~20% short-window. Comments on line 99
  "Wilder's smoothing ≈ EWM with alpha=1/14" disagree with the implementation
  (`span=14` ≠ `alpha=1/14`). Fix: use `alpha=1/14` explicitly. Drift vs
  pandas-ta/talib visible in any signal comparing ATR thresholds.

- portfolio/indicators.py:73-74 — Bollinger Band std uses default ddof=1
  (sample std). pandas-ta and TradingView use population std (ddof=0).
  Bands will be ~3% wider than TV reference. Minor but affects BB-squeeze
  triggers.

- portfolio/data_collector.py:122-125 — Alpaca `start` is computed as
  `end - pd.Timedelta(days=lookback_days)`. For `1Month` interval the
  lookback is 1825 days = 5 years. Alpaca IEX has data starting ~2015,
  fine for now, but the request size for 5y of 1Month bars is small. No bug,
  just verbose.

- portfolio/futures_data.py:42-53 — `get_open_interest` uses `data["openInterest"]`
  unconditionally. If Binance returns the documented `{"symbol": ..., "openInterest": ..., "time": ...}`,
  fine; if a transient error returns `{"code": -1121, "msg": "Invalid symbol."}`,
  KeyError propagates. The other fetchers in this file gracefully handle empty
  `data` but this one does `data["openInterest"]` before checking. Fix: same
  defensive .get() pattern.

## Tests missing

- FX rate sanity-fail loop: feed a sequence (good, bad, good) and assert the
  middle bad rate doesn't pin the cache forever.
- HTTP retry 4xx behavior: assert that 401/404 do NOT retry and DO surface
  status to caller.
- Alpha Vantage budget persistence across restart (currently zero coverage
  for the most expensive paid quota).
- Earnings calendar bypass-counter: assert AV budget includes earnings calls.
- `_eu_dst` boundary tests for Mar 31 falling on Sunday/Monday/Saturday and
  Oct 31 falling on Sunday.
- `funding_rate` sign convention: feed a known +0.001 (0.1%) rate from
  historical Binance data and assert SELL.
- `seasonality.compute_hourly_profile` across DST boundary (assert
  pre/post-DST buckets aren't mixed).
- `price_source.fetch_klines` fallback symbol-collision: when binance_fapi
  XAU-USD fails, the yfinance fallback should EITHER use a distinct alias
  OR raise — not silently return spot data.

## Cross-cut observations

1. **Per-process counters everywhere.** Alpha Vantage budget, NewsAPI budget,
   BGeometrics budget, fear-greed streak — all track quotas in-memory and
   only some persist. With the loop's exponential-backoff restart cycle, a
   crashing loop can burn an API budget multiple times. Persist all daily
   counters to disk with last-reset-date.
2. **`fetch_json` is the choke point** for transient-vs-permanent failure
   distinction. Right now `None` ↔ "any error". Three modules (alpha_vantage,
   futures_data, funding_rate) all swallow status codes. Add a
   `fetch_json_strict()` variant that returns `(status, body|None)`.
3. **Symbol routing is ambiguous between Binance FAPI / yfinance for metals.**
   `SI=F` is aliased to `XAGUSDT` in `price_source._BINANCE_FAPI` but
   `metals_cross_assets.get_gold_silver_ratio` fetches `SI=F` then operates
   on daily bars where Binance FAPI returns perpetual data — mixing perp and
   spot data into the ratio. Audit every `_yf_download("SI=F"|"GC=F")` call.
4. **Hardcoded date lists with no auto-refresh** (FOMC, CPI, NFP, GDP). Add a
   `data_refresh.py`-style updater that pulls from FRED's
   `ALFRED` releases-calendar or BLS's static schedule once a month, and
   alert if next_event > 90 days out (likely calendar exhausted).
5. **Mixed tz-naive / tz-aware datetime** across `forecast_signal.py`
   (`datetime.now(UTC).replace(tzinfo=None)` for Prophet), `session_calendar`
   (`now.tzinfo` checks), `econ_dates` (UTC anchor at 14:00). Each module
   handles it locally; one upstream caller passing the wrong shape will
   produce a TypeError deep in the stack.
6. **Hardcoded paths for sentiment subprocess scripts** create silent
   degradation on new hosts. Combined with the silent in-process→subprocess
   fallback in `_run_model`, a missing script on Linux means "no shadow
   sentiment" with only DEBUG logging.
