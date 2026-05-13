# Claude adversarial review: data-external

Read-only review at HEAD 59ad394e. Scope: portfolio/data_collector.py,
fear_greed.py, sentiment.py, bert_sentiment.py, alpha_vantage.py,
futures_data.py, funding_rate.py, fx_rates.py, onchain_data.py,
news_keywords.py, social_sentiment.py, crypto_macro_data.py,
crypto_scheduler.py, earnings_calendar.py, econ_dates.py, fomc_dates.py,
seasonality.py, seasonality_updater.py, session_calendar.py,
price_source.py, http_retry.py, api_utils.py, data_refresh.py,
forecast_signal.py, indicators.py, metals_orderbook.py,
microstructure.py, microstructure_state.py, metals_cross_assets.py,
tickers.py.

## Summary

Three findings rise to **P0** because they directly violate the
"live prices first" rule or corrupt data that every signal consumes:

1. **`http_retry.fetch_with_retry` parses `retry_after` from JSON body
   (Telegram-only) and never reads the HTTP `Retry-After` header.** Every
   other rate-limited API in scope - Binance, Alpha Vantage, NewsAPI,
   Frankfurter, BGeometrics, CryptoCompare - sends `Retry-After` as a
   header. `_binance_limiter` is a soft client-side bucket that can
   under-count during burst cycles, so when Binance enforces 429 we ignore
   the server-provided cooldown and back off using the default exponential
   schedule, often re-firing the next loop and getting banned.

2. **`portfolio/funding_rate.py` SELL threshold is mis-set.** Comment says
   "Normal funding ~0.01% (0.0001) ... `> 0.03%` -> SELL" but the literal is
   `rate > 0.0003` which is 0.03% expressed as a fraction. Worse, the BUY
   threshold is asymmetric: `< -0.0001` (-0.01%) vs `> 0.0003` (+0.03%).
   The signal will SELL on every minor positive funding tilt and BUY
   hardly ever. The asymmetry is unjustified.

3. **`price_source.fetch_klines` silent fallback chain serves
   yfinance 15-min-delayed data labelled as "live".** When Binance FAPI
   raises (rate limit, transient 5xx) we emit `logger.error` and call
   `_fetch_yfinance` - but the returned `pd.DataFrame` has no source tag,
   so every downstream consumer (signal_engine, microstructure
   accumulators, exit_optimizer) computes signals from `SI=F` / `GC=F`
   bars that are 10-15 min stale, against book/trade snapshots that are
   live. The comment block (lines 13-19) cites a 7.7s vs 655.4s
   freshness gap; the fallback path silently re-introduces that gap.

The remaining list is long but the data path here has a lot of
defensive cleanup (UTC clamping, NaN guards, dogpile prevention),
which is genuinely good. The flagged items are real defects, not
theatrical churn.

## P0 - Blockers

### P0-1. `Retry-After` HTTP header ignored on every 429

`portfolio/http_retry.py:43-49` - the 429 branch only tries
`resp.json().get("parameters", {}).get("retry_after", wait)`. That JSON
shape is Telegram-specific (`{"ok":false,"parameters":{"retry_after":N}}`).
Every other provider returns either:

- the standard `Retry-After: <seconds>` HTTP header (Binance, Alpha
  Vantage, NewsAPI when throttled);
- a JSON `Note`/`Information` field but no parameters block.

When Binance's spot or FAPI returns 429 with a `Retry-After: 120`
header, `wait` stays at `backoff * factor^attempt` ~ 1-4 s, blows past
the server's cooldown, and we get banned. Compare with the explicit
Binance-ban risk the docstring acknowledges in `shared_state.py:291`
("Binance: 1200 weight/min - very generous, but space out slightly"):
the client-side limiter assumes 600/min but actual weight cost varies
by endpoint and burst.

Fix: read `resp.headers.get("Retry-After")` first (parse as int or
HTTP-date), then fall through to Telegram JSON shape, then to the
default backoff. While there: cap `Retry-After` so a hostile/erroneous
`Retry-After: 999999` doesn't lock the loop for 11+ days.

### P0-2. Funding-rate thresholds wrong + asymmetric

`portfolio/funding_rate.py:42-49`:

```
if rate > 0.0003:       # 0.03 %
    action = "SELL"
elif rate < -0.0001:    # -0.01 %
    action = "BUY"
```

`0.0003` is the Binance default *funding rate cap* on most perps - not
0.03% above the "normal ~0.01%" docstring reference. So:

- **Asymmetry**: SELL threshold (3x normal) vs BUY threshold (-1x
  normal). The signal is structurally biased to SELL in any positive
  funding regime, and that's most of the time in bull markets.
- **Magnitude**: the comment block claims "Normal funding ~0.01%
  (0.0001). Thresholds: > 0.03% -> overleveraged longs". Actual funding
  on BTCUSDT routinely prints 0.005%-0.015% in calm regimes. A
  threshold of 0.03% requires *3-6x normal* to fire - too high for a
  signal you've tagged as "74.2% at 3h, 535 samples" (CLAUDE.md).
- **Sign confusion**: Binance reports the funding rate *per 8h
  interval* unannualised. The thresholds in the file appear to be in
  per-8h fraction. That matches Binance's `lastFundingRate` field -
  good - but no comment clarifies that, and `rate_pct = round(rate *
  100, 4)` will print `0.03` for a 0.03% funding cap, which a human
  could easily misread as "3 %".

Fix: use symmetric thresholds (+/-0.015% or so), and add a comment
explicitly stating the rate is per-8h-interval not annualised. Cross-
reference the per-ticker accuracy in `accuracy_stats` before promoting
to BUY/SELL: a XAG signal at 67% and a BTC signal at 35% should not
share a hardcoded threshold.

### P0-3. yfinance fallback serves stale data without source tagging

`portfolio/price_source.py:223-243` - on primary-source exception
(Binance FAPI 429/timeout, Alpaca 5xx) we call `_fetch_yfinance` and
return the resulting DataFrame *transparent to the caller*. The
docstring (`lines 13-19`) literally says yfinance is "85x stale" - yet
when the fallback fires:

- no source attribute is set on the returned DataFrame;
- no field on the returned bars indicates "this is degraded data";
- `microstructure_state.persist_state()` will still write
  `ts: int(time.time() * 1000)` against book/trade data that is fresh
  but klines that lag 10-15 min;
- `forecast_signal._load_candles` calls go straight through Chronos.

Result: when Binance is intermittently flaky (which it is - Mar/Apr
incidents documented in code comments), the loop generates signals
mixing live microstructure and 10-min-stale OHLCV. That's exactly the
"BTC 12h BUY phantom" pattern that MEMORY.md flags as "always fades -
proven noise - ignore."

Fix options: (a) raise `SourceUnavailableError` rather than transparent
fallback, force the signal to HOLD; (b) annotate the returned
DataFrame with a `source` attribute / `_meta` column and have
`compute_indicators` refuse to compute if source != binance/alpaca; or
(c) at minimum tag `agent_summary.json` with a `degraded_sources` list
so downstream voting can apply a penalty.

### P0-4. `_cached()` LRU eviction default TTL collides with timeframe cache entries

`shared_state.py:54-58` evicts entries based on `v.get("ttl", 3600) *
_MAX_STALE_FACTOR`. But `data_collector._fetch_one_timeframe` writes
to `_ss._tool_cache` directly (line 309) without ever setting a `ttl`
field on the cached entry. So a 24h-TTL `6mo` timeframe entry is
treated as if its TTL were 3600 s for eviction purposes; after 3 *
3600 = 3 hours of disuse it can be evicted, forcing a re-fetch that
the TIMEFRAMES table explicitly tried to avoid (cache 24h).

Fix: pass `ttl` when writing the cache entry (`_tool_cache[cache_key]
= {"data": entry, "time": time.time(), "ttl": ttl}`). Same fix needed
in any other call site that pokes `_tool_cache` directly - grep for
`_tool_cache[` writes outside `shared_state.py`.

## P1 - High

### P1-1. Alpha Vantage daily budget reset is wall-clock UTC, but free-tier reset is provider-specific

`alpha_vantage.py:163` resets `_budget_reset_date` based on
`datetime.now(UTC).strftime("%Y-%m-%d")`. Alpha Vantage's free tier
documents a daily reset at **00:00 US Eastern**, not UTC. Between
00:00 UTC and 05:00 ET (winter) or 04:00 ET (summer) the in-memory
counter resets, the loop fires its first refresh batch, but AV is
still on yesterday's quota. The 25/day budget will silently burn.

### P1-2. Alpha Vantage `_normalize_overview` swallows AV's rate-limit response

`alpha_vantage.py:93-99` - returns `None` whenever response has `"Note"`
or `"Error Message"`. AV's classic rate-limit response is `{"Note":
"Thank you for using Alpha Vantage..."}` with HTTP 200. `_fetch_overview`
*also* checks for `"Note"` and returns None (lines 150-153), so both
layers swallow the same condition and the *outer caller*
`refresh_fundamentals_batch` increments nothing, never marks the
circuit-breaker failure, never persists the budget exhaustion to
config or telemetry. Net effect: on rate-limit you record `_cb.record_failure()`
3 times (line 274-276) and then the cb opens for 300 s, but the
`Note` itself never logs as a budget event. Operations cannot tell
"Alpha Vantage burned out at 13:00 UTC" from "Alpha Vantage stopped
responding at 13:00 UTC".

### P1-3. `_check_budget` increments under lock but reads outside the lock loop

`alpha_vantage.py:157-168` locks `_cache_lock` to read+reset the daily
counter, but `refresh_fundamentals_batch` (line 250-251) computes
`remaining_budget = daily_budget - budget_used` outside the lock. Two
concurrent batch-refresh callers (this should never happen if the
loop is single-threaded, but Layer 2 / dashboard / manual scripts can
all import the module) will both see the same `budget_used` and both
attempt 25 calls. With 5/min limiter that's a 50-call window that
exceeds the daily quota.

### P1-4. `earnings_calendar._fetch_earnings_alpha_vantage` bypasses the daily-budget counter

Explicit in the comment at `earnings_calendar.py:49-52`: "earnings
calls bypass alpha_vantage.py's `_daily_budget_used` counter". With 5
US stock tickers (just MSTR today, but the function iterates
`STOCK_SYMBOLS`) and 24h cache, that's 5 hidden AV calls per day
outside the budget. Currently safe because of small STOCK_SYMBOLS,
but the budget guard is a footgun - if MSTR + 5 more tickers ever come
back (the comment notes "Removed Mar 15: AMD, GOOGL..."), the 25/day
budget is silently +6 over what `refresh_fundamentals_batch` thinks.

### P1-5. `fear_greed.get_crypto_fear_greed` lacks cache; relies on `signal_engine` wrapping

`fear_greed.py:95-123` makes an unconditional HTTP call to
alternative.me on every invocation. The only thing keeping us from
hammering the API at 60s cadence x N tickers is the `_cached()`
wrapper at `signal_engine.py:3140`. If any other caller imports
`get_crypto_fear_greed` directly (golddigger, after-hours research,
dashboards, `daily_digest.py`), they bypass the cache. There IS no
in-module rate limiter.

Confirmed at least one direct caller: `if __name__ == "__main__"`
block (line 181) - fine. But any new code that imports the helper
will burn API calls. Reasonable fix: lift the cache into the module
itself, so the wrapping is invariant to caller location.

### P1-6. `fx_rates.fetch_usd_sek` race between sanity-check and cache write

`fx_rates.py:36-71` - under high concurrency the timeline is:

1. Thread A reads cache (line 31-32), sees stale, calls Frankfurter.
2. Thread B reads cache (line 31-32) at the same moment, also sees
   stale, also calls Frankfurter.

Both pass the sanity check, both write to `_fx_cache`. Frankfurter
publishes mid-rate. ECB midpoint. Fine for valuation, *wrong for any
hedge calculation* that needs bid/ask spread - but the rate is used
exclusively for portfolio valuation in this codebase, so the bid/ask
question is moot. The race in the fetch is the actionable bug - two
Frankfurter hits per minute is wasteful but not budget-breaking.

More serious: when the API returns rate **outside** 7-15 (the sanity
floor/ceiling), the code logs ERROR (good) and *returns the cached
stale rate* (line 56-65) without alerting Telegram unless the cached
rate is also >2h old. So if the API has a known-bad data outage that
returns 16.4 (e.g. brief glitch), and our cached rate is 5min old,
we'll silently use 5min-old data forever - there's no proactive
re-test interval; we only retry when `time.time() - cached_time` >=
900 s. So a 1-hour brief glitch will lock us on a single cached value
for 15 minutes after each call, which is fine, but a sustained glitch
keeps us on stale data until eventually the cache passes the 2h
threshold and we alert.

### P1-7. `seasonality_updater._fetch_hourly_klines` does not normalize the index timezone vs `seasonality.compute_hourly_profile`

`seasonality_updater.py:75-85` builds an index with `pd.Timestamp(k[0],
unit="ms", tz="UTC")`. Good. But `seasonality.compute_hourly_profile`
uses `df.index.hour` (line 45). For a TZ-aware DatetimeIndex,
`df.index.hour` returns the **UTC hour**, which is what we want.
However, the profile is keyed by `str(hour)` ("0".."23"). The metals
warrant trades 08:15-21:55 CET (= 07:15-20:55 UTC winter / 06:15-19:55
UTC summer). The DST flip will shift the "active hour" pattern by +/-1
across the year, so any profile built right after a DST change
contains mixed-hour data. `compute_hourly_profile` averages this
silently - the resulting profile has a partial 1-2hr smear around the
metals open/close transitions.

This is a known issue in academic intraday-seasonality literature.
The defensive fix is to compute profiles in **local market time**
(CET-fixed, not CET/CEST DST-aware) rather than UTC; or to recompute
profiles right after each DST transition with a 2-week lookback to
exclude pre-DST data. Currently neither is done.

### P1-8. `crypto_macro_data._fetch_deribit_options` max-pain locale fragility

`crypto_macro_data.py:130-167` - the inline comment ping-pongs three
times about the formula then settles on the right one (minimize
total intrinsic-value payout across candidate strikes). The actual
implementation matches the "min total payout" definition.

But: `_parse_expiry` uses `datetime.datetime.strptime(s, "%d%b%y")`,
which is locale-sensitive. On a non-English-locale runtime, `%b`
won't match "MAR"/"APR" and every expiry parse returns None. The
fallback (line 117-121) picks "first expiry with most OI" - usually
the long-dated 6-month, not the nearest weekly - giving a useless
max-pain estimate. The code runs on Windows where `LC_TIME` defaults
to the OS locale; on Swedish locale, English month abbreviations are
"mars"/"apr"/etc, not "MAR"/"APR".

Fix: parse manually with a {month_abbr: 1..12} table, or set
`LC_TIME=C` for the parse, or use `dateutil.parser` with explicit
month names.

### P1-9. `microstructure_state.persist_state` double-calls `get_microstructure_state` which mutates `_ofi_history`

`microstructure_state.py:175-202` - `get_microstructure_state(ticker)`
internally calls `record_ofi(ticker, ofi)` (line 185), which APPENDS
to `_ofi_history`. `persist_state()` (line 205-213) iterates over
`_snapshot_buffers` and calls `get_microstructure_state(ticker)` for
each - pushing **another** synthetic OFI value into the rolling
history even though no new snapshot arrived. Result: every
`persist_state()` call inflates the OFI-history denominator by len(tickers),
biasing z-scores toward zero (the "self-contamination" pattern the
docstring claims to fix at line 108).

Fix: split into a pure read function (`peek_microstructure_state`) and
a record function (`update_microstructure_state`). `persist_state`
should peek, not update.

### P1-10. `forecast_signal._get_chronos_pipeline` exception clause shadows ImportError

`forecast_signal.py:97` uses `except (ImportError, Exception)`. Since
`Exception` is the supertype of `ImportError`, this catches everything
- including a real `KeyboardInterrupt` if it were to fire during
import (it isn't a subclass of Exception, so actually safe, but the
intent is obscured). More importantly, a non-import failure (e.g.
CUDA OOM, model file corruption) is silently treated as "Chronos-2
not available" and falls back to v1 - at which point v1 may not even
be installed in `Q:/finance-analyzer/.venv` since the codebase has
moved to Chronos-2. v1 import will fail too, the function returns
None, and the forecast signal silently goes HOLD without an alert.

Fix: tighten the except to `ImportError` only for the "feature
detection" path; let real failures escape with an `exc_info=True`
log so we know the GPU/model is bricked.

### P1-11. `data_refresh.download_klines` uses spot endpoint for "futures" data

`data_refresh.py:1-15` writes feathers to
`user_data/data/binance/futures/` - the path strongly implies FAPI
data. But line 31 calls `f"{BINANCE_BASE}/klines"` which is *spot*,
not FAPI. Spot BTCUSDT and futures BTCUSDT diverge by basis (and
during contango can differ by 1-3%). Any backtest that consumes these
feathers as "futures" and trades against perp pricing in real time is
running on the wrong dataset.

Fix: switch to `BINANCE_FAPI_BASE`, or rename the directory to
`spot/`.

### P1-12. `onchain_data._fetch_all_onchain` `time.sleep(1)` blocks the worker thread under the cache-lock-free path

`onchain_data.py:223-235` calls `time.sleep(1)` 5 times in sequence
inside a `_cached()` wrapper. Under dogpile prevention
(`shared_state._cached:79-88`), only one thread runs the fetcher; the
others get stale data. But that one thread holds the loading-key for
*at least 5 s* of cumulative sleep, which is fine. **The latent issue**
is that `_LOADING_TIMEOUT = 120` in `shared_state.py:29` is the
stuck-key eviction; if the BGeometrics API stalls during a fetch the
key sits for up to 120 s and **no other thread** can refresh. The
serialized sleeps make this more likely, not less.

More urgent: the BGeometrics budget is documented at 15 req/day. The
function makes 6 calls per fetcher invocation. If `_cached` evicts the
key during the 120s loading window and another thread re-enqueues the
fetcher, that's 12 calls in a single retry cycle - *80% of daily
budget*. The current code does have the "in-memory seed from
persistent cache" path (line 252-273) that prevents this on cold
start, but if persistent cache is missing/corrupt and three workers
hit `get_onchain_data` concurrently before the loading-key gets set,
all three will start the 6-call sequence. Stuck-key eviction at 120s
makes this worse, not better.

Fix: tighten `_LOADING_TIMEOUT` for budget-sensitive keys, or add a
budget guard in `_fetch_all_onchain` similar to `alpha_vantage`'s.

### P1-13. `fx_rates._fx_cache` carries an alert sentinel under the same key

`fx_rates.py:78` writes `_fx_cache["_last_fx_alert"] = now`. The cache
dict is keyed on `"rate"` / `"time"` / `"_last_fx_alert"`. Tests or
future migrations that iterate `_fx_cache.items()` will hit the
sentinel as a "rate". Reasonable defensive fix is to keep the alert
state in a separate dict.

### P1-14. `data_collector.collect_timeframes` `future.cancel()` is a no-op once worker is in HTTP call

`data_collector.py:329-338` - on `TimeoutError` we call `f.cancel()`
for each unfinished future. Python's `Future.cancel()` only cancels
work that hasn't started yet; once `_fetch_one_timeframe` is inside
`fetch_with_retry`, the cancel returns False and the thread continues
running until the network call returns (or our 10s HTTP timeout
fires). For a 60s `_TF_POOL_TIMEOUT` with 4 in-flight HTTP timeouts
of 10s each, that's another 10s of worker waste before the next
cycle. The visible symptom is "stuck timeframes" in the logs that
silently disappear next cycle but billed full HTTP latency in the
meantime.

Fix: lower per-request timeout for the parallel timeframe fetches
specifically, or pass `requests.Session` with a connect/read pair.

### P1-15. `econ_dates` event time hardcoded to 14:00 UTC

`econ_dates.py:155-156` & `:180` & `:225-227`:

```
evt_dt = datetime.combine(evt["date"], datetime.min.time().replace(hour=14), tzinfo=UTC)
```

BLS releases (CPI/NFP) print at **08:30 ET** - that's 12:30 UTC (EST)
or 13:30 UTC (EDT), NOT 14:00 UTC. GDP is also 08:30 ET. FOMC
announcements are 14:00 ET = 18:00 UTC (EST) or 19:00 UTC (EDT),
NOT 14:00 UTC.

So:

- CPI/NFP/GDP `hours_until` is **off by 30-90 minutes** depending on
  DST.
- FOMC is off by **4-5 hours**.

The downstream consumer `is_macro_window` uses `lookback_hours=24`,
which is wide enough to dwarf the 4-hour FOMC error - so the regime
gate still fires correctly. But the user-facing "hours_until" value
in crypto_scheduler reports, Telegram digests, and dashboard is
wrong by hours on FOMC days. Pre-FOMC positioning ("don't trade in
the 2h before announcement") is undermined.

Fix: use 12:30 UTC for CPI/NFP/GDP (08:30 ET, adjust for DST in
ref_time), and 18:00/19:00 UTC for FOMC.

### P1-16. `crypto_scheduler._build_crypto_report` reads `agent_summary_compact.json` snapshot, but consumes fields across cycles

`crypto_scheduler.py:110` - `load_json(DATA_DIR / "agent_summary_compact.json")`
is atomic-safe. BUT: the report is built from a **mix** of fields -
`signals`, `macro`, `futures_data`, `prophecy`, `monte_carlo`,
`price_levels` - that all sit on disk in the same file. If a
subsequent write of `agent_summary_compact.json` lands mid-report-build,
the report's fields are internally inconsistent (e.g.
`signals.BTC-USD.price_usd` from cycle N, `prophecy.beliefs` from
cycle N+1). The atomic write prevents file corruption but not value
drift across multiple `load_json` calls.

The report only does ONE `load_json` here, so this particular path is
actually consistent. But `fund_cache = load_json(DATA_DIR /
"fundamentals_cache.json", default={})` (line 277) is a separate file
read, and there's no contract that it matches the agent_summary
generation cycle. In practice this is fine - probabilistically tiny.
Worth a note.

### P1-17. `metals_orderbook._DEPTH_TTL = 10` permits noisy zero-depth from Binance with no validation

`metals_orderbook.py:28` sets 10s TTL on depth snapshots. The fetcher
returns `None` only when bids OR asks lists are empty (line 67-68).
But Binance has been observed to return books with a single quote on
one side (e.g. one bid level, zero ask levels) during illiquid moments
on XAUUSDT - the existing check passes, but downstream
`depth_imbalance` (microstructure.py:36) sees `ask_vol <= 0` and
returns 0.0, which signals "balanced" when the book is actually
broken. Same path in `compute_ofi` - `prev["asks"]` may have one
level but `[0][1]` qty could be 0.01 vs `curr["asks"][0][1] = 0`
giving `delta_ask = -prev_ask_vol` regardless of price.

The microstructure features are sensitive to thin books. There should
be a minimum-depth gate (`min(bid_levels, ask_levels) >= 3` say)
before computing imbalance. Currently any single-quote book votes
"flat" in microstructure.

### P1-18. `bert_sentiment._predict_per_text` placeholder returns `confidence=0.0, neutral=0.34` even after PyTorch errors

`bert_sentiment.py:443-452` - when a per-text forward pass throws,
we append `{"confidence": 0.0, "scores": {"positive": 0.33,
"negative": 0.33, "neutral": 0.34}}`. That's a perfectly valid
sentiment entry that downstream `_aggregate_sentiments` treats as a
real vote - diluting the per-headline majority. The fallback message
in the docstring claims this is "a final safety net", but the impact
is bias toward neutral.

Worse: per the docstring (line 220-227) this was the *exact*
mechanism by which a meta-tensor race poisoned the A/B log for hours.
The meta-tensor detection fix (line 238-257) is good; the
per-text-failure placeholder is still there. A single torch-related
failure on one headline doesn't just lose that headline - it makes
the whole batch's aggregator vote drift toward "neutral".

Fix: drop the failed entry from results rather than substituting a
fake neutral; let the aggregator process the surviving subset.

### P1-19. `news_keywords.score_headline` discards multi-keyword count

`sentiment.py:590-601` & `news_keywords.py:139-159` - `score_headline`
returns `(max_weight, matched)`, but `_compute_weights` discards
`matched` (line 597 `_, _`). So we only ever see "highest single
keyword found in title", not "how many high-severity keywords are in
this title". A headline with "war tariff crash invasion sanctions"
gets weight 3.0; one with "tariff" gets weight 3.0. The aggregator
sees them identically.

Trivial fix: return `sum(weights) / len(weights)` or
`1 + log(count of matched + 1)` to reward multi-keyword headlines.

### P1-20. `_normalize_overview` divides on AV's pre-stringified pct values

`alpha_vantage.py:106-107` - `revenue_growth_yoy` and
`earnings_growth_yoy` come from AV's `QuarterlyRevenueGrowthYOY` /
`QuarterlyEarningsGrowthYOY`. AV returns these as decimal fractions
(e.g. `"0.215"` for 21.5%). The `_float` helper just `float()`s the
string. Downstream consumers might expect percent (21.5) or fraction
(0.215). No comment clarifies. Any caller mixing assumptions silently
mis-scales fundamentals by 100x. Worth a docstring note + a guard
test.

## P2 - Medium

### P2-1. `social_sentiment._fetch_subreddit` uses raw `urllib.request.urlopen`, no `http_retry`

`social_sentiment.py:30-56` - bypasses `fetch_with_retry`,
`_binance_limiter`-equivalent throttling, and the 429-friendly retry
logic. Reddit is famously throttled on UAs that don't match a real
browser; we hit `Mozilla/5.0`-equivalent but use the
"finance-analyzer/1.0" UA which Reddit's anti-bot frequently flags.
On 429 the `urlopen` call raises `HTTPError`; the except clause
swallows it as `print(...)` (not even `logger`). Loop continues but
sentiment from reddit silently disappears.

### P2-2. `social_sentiment` uses `print(...)` for errors

`social_sentiment.py:110, 122` - `print(...)` not `logger.warning`.
These show up in `agent.log` only via stdout capture and are
unstructured.

### P2-3. `fear_greed.get_stock_fear_greed` reads `^VIX` directly via yfinance and then `data_collector.fetch_vix` does the same independently

`fear_greed.py:126-171` and `data_collector.py:168-204` are two
independent yfinance fetches of `^VIX` with no shared cache and
different freshness budgets. Both serialise on `yfinance_lock` but
under load they can fire 2x per cycle. Minor wasted bandwidth.

### P2-4. `crypto_macro_data._fetch_deribit_options` `expiry_data` defaultdict uses lambda

`crypto_macro_data.py:67-71` - `defaultdict(lambda: {...})`. If
`_tool_cache` is ever migrated to a disk-backed serialization format,
the lambda is not serialisable. Today not a problem; flagging as a
maintainability cliff if cache backing changes.

### P2-5. `metals_cross_assets._yf_download` routes via `price_source`, but then re-capitalizes columns

`metals_cross_assets.py:67-73` - receives lowercase columns from
`price_source.fetch_klines`, then renames to uppercase for backward
compat. Every signal in this module then uses `df["Close"]`. The
double-rename is a maintenance footgun: any future caller in this
module that uses `df["close"]` (lowercase) will mysteriously fail.

### P2-6. `seasonality_updater._fetch_hourly_klines` symbol map import drags `metals_orderbook` namespace

`seasonality_updater.py:10` imports `SYMBOL_MAP` from
`metals_orderbook`. That map is for **order book** symbols (includes
BTC-USD, ETH-USD), not metals-only. If we ever pass `["BTC-USD"]` to
`update_seasonality_profiles`, it will try to fetch FAPI klines for
BTCUSDT - but the function is purely for metals. Type-narrow the
import to a metals-only set in tickers.py.

### P2-7. `data_collector.collect_timeframes` `tf_order` uses 999 sentinel - wrong-key warnings absent

`data_collector.py:341-342` - if a future result returns a label
that's not in the timeframes table, `tf_order.get(x[0], 999)` sorts it
to the end. No log warning. A typo or schema change downstream
won't be caught.

### P2-8. `crypto_macro_data._append_netflow_history` race between `_load_netflow_history` and `atomic_append_jsonl`

`crypto_macro_data.py:414-426` - reads file, checks last timestamp,
then appends. Two threads can both observe "latest is >6h old" and
both append, producing duplicate samples 0.1s apart. The history file
is then used by `_load_netflow_history` for `sum_7d` computations -
duplicate samples bias the sum.

The same pattern exists in `_append_ratio_history` (line 292-308).
Both should use an in-memory write-mutex.

### P2-9. `onchain_data._fetch_all_onchain` `_save_onchain_cache` writes even on partial success

`onchain_data.py:236-241` - if `any_success` is True we save the
whole partial result. The next cache load sees the stale missing
fields (e.g. `mvrv_zscore` from yesterday) interpreted as fresh
because `ts` is current. The `interpret_onchain` function silently
classifies those zones from old data.

### P2-10. `funding_rate.get_funding_rate` `_cached` doesn't cache None, retries every cycle

`funding_rate.py:62` - `_cached(...)` doesn't cache `None` (per
`shared_state.py:99`), but `_fetch_funding_rate` returns `None` on
KeyError. The next cycles will retry, hammering Binance FAPI. Not
the worst - `_binance_limiter` will throttle - but defeats the
purpose of a 15min TTL.

### P2-11. `alpha_vantage._is_stale` uses `max_stale_days=5` default but `cache_ttl_hours` from config is 24

`alpha_vantage.py:171-185` - `_is_stale` defaults to 5 days, called
nowhere in this codebase. `_cache_age_hours` is what's actually used
by `refresh_fundamentals_batch` (line 242). Dead code path that
divergent staleness threshold could mislead future readers.

### P2-12. `econ_dates.NFP_DATES_2026` April 2nd entry is a Thursday

`econ_dates.py:61` - `date(2026, 4, 2)` with comment "Apr 3 = Good
Friday, market closed". 2026-04-02 is a Thursday. The comment claims
this is correct because BLS released early. **In reality**: 2026 NFP
April 2 is a Thursday because BLS releases NFP on Friday and Apr 3
was Good Friday. The comment looks correct. **False alarm but worth
auditing the 2027 dates** - none of those have explanatory
comments, so any holiday-shifted date is invisible to a future reader.

### P2-13. `fomc_dates.py` should be cross-checked against the Fed's published 2026/2027 calendar

`fomc_dates.py:13-22` lists the 2026 meeting dates. The Fed's 2026
schedule on federalreserve.gov should be the source of truth.
Recommendation: write a one-shot test that fetches the Fed's
published calendar (or pins a known-good snapshot) and verifies our
list matches. Today this is hand-maintained and silently drifts.

### P2-14. `seasonality.compute_hourly_profile` 5-day minimum is below academic guidance

`seasonality_updater.py:32` - fetches 500 hourly bars (~20 days),
which is well above the 5-day minimum but below the academic
recommendation of 30+ days for stable hour-of-day mean estimates.
With 500 bars / 24 hours = ~20 samples per hour-bucket, the standard
error of each hour's mean return is large enough that detrending
adds noise more than it removes a deterministic seasonal cycle.

### P2-15. `microstructure_state` cold-start blackout for ~10 minutes after every restart

`microstructure_state.py:216-229` - `load_persisted_state` only reads
the latest dict snapshot from disk. After a process restart, all
rolling buffers start empty; the cold-start period (until
`_MIN_SPREADS_FOR_ZSCORE = 10` spreads have been accumulated, which
at 60s cadence is 10 minutes) produces `spread_zscore = None` and
`ofi_zscore = 0.0`. The orderbook_flow signal during those 10 minutes
votes effectively HOLD. Cold-start is a recurring event (every
metals_loop restart = once per couple of hours when the loop
crashes/restarts per CLAUDE.md's recovery scheme).

Fix: persist the deques themselves and reload them.

### P2-16. `bert_sentiment._load_model` `local_files_only=True` for CryptoBERT/Trading-Hero - HF hub fallback disabled

`bert_sentiment.py:88, 97` - `local_files_only=True`. If the local
cache dir is corrupted, the model load fails entirely rather than
re-downloading. Fine for production (no network egress), bad for a
new dev machine.

### P2-17. `forecast_signal.run_forecasts` `_load_candles` uses fixed `periods=168` (7 days hourly)

`forecast_signal.py:36-63` - 168 hourly bars for Chronos context.
Chronos-2 ingest typically benefits from 512-2048 context window.
168 is below the model's training context; forecast accuracy
documented at 25.6% recent (per tickers.py:165-172). The disabled
status confirms the production accuracy collapse; the 168-context
choice is plausibly a contributor (along with the regime change
the comment cites).

### P2-18. `crypto_macro_data._fetch_deribit_options` no rate limiter

`crypto_macro_data.py:46-54` - Deribit's public API allows ~20 r/s,
which we don't approach, but two concurrent BTC+ETH options fetches
hit Deribit's endpoint without coordination. If the loop later
expands to per-strike options analytics, this will need throttling.

### P2-19. `data_collector.binance_klines` interval validation absent

The default `interval="5m"` in `_binance_fetch` is correct. The
issue: no validation that callers pass a Binance-supported interval.
`fetch_with_retry` would surface the -1120 error to the worker as
exception. Defensive `if interval not in {"1m","3m","5m","15m","30m",
"1h","2h","4h","6h","8h","12h","1d","3d","1w","1M"}: raise` at the
top would catch typos at the source.

## P3 - Low

### P3-1. `crypto_scheduler` uses `datetime.now().astimezone().isoformat()` for log_entry timestamp - host-local, not UTC

`crypto_scheduler.py:310` - `datetime.now().astimezone().isoformat()`
captures local timezone offset. Every other JSONL writer in the
codebase uses `datetime.now(UTC).isoformat()`. Inconsistent
timestamping makes log correlation across files painful.

### P3-2. `social_sentiment` `_fetch_subreddit` and `_search_subreddit` duplicate the trivial post-shape parsing

40 lines duplicated between these two functions; minor maintenance
weight.

### P3-3. `news_keywords._KEYWORD_PATTERNS` cleanup uses ad-hoc strip

`news_keywords.py:155` - `pattern.pattern.replace(r"\b", "").replace("\\", "")`
to recover the original keyword from a compiled pattern. Brittle; if a
keyword ever contained `\b` it would scramble. Better to keep the
(pattern, keyword) tuples explicit.

### P3-4. `fomc_dates.FOMC_DATES_ISO` is derived but `FOMC_ANNOUNCEMENT_DATES` is hand-written

`fomc_dates.py:37, 40` - `FOMC_DATES_ISO` is a derived list; the
announcement-only list is hand-curated. If the calendar table is
edited but `FOMC_ANNOUNCEMENT_DATES` is forgotten, the two diverge.
Derive both from a single (start, announce) tuple list.

### P3-5. `data_collector._YF_INTERVAL_MAP` daily period - yfinance allowed-period matrix changes silently

`data_collector.py:35-41` - the period-vs-interval matrix in yfinance
shifts with library upgrades. A test that verifies each entry returns
>=1 row would catch breaks.

### P3-6. `metals_cross_assets.get_gvz()` `_GVZ_TTL = 600` is overkill

`metals_cross_assets.py:32` - 10-min TTL on a daily-published index
is unnecessary. 4-6h would be fine.

### P3-7. `sentiment._STOPWORDS` and `_SIGNIFICANT_KEYWORDS` defined as plain sets

`sentiment.py:559-571` - defined as `set` literals; should be
`frozenset` for immutability. Microscopic.

### P3-8. `indicators.compute_indicators` `horizon` parameter only handles `"3h"`, no other horizons

`indicators.py:13-17` - the function silently uses the 14-period RSI
default for anything other than `"3h"`. No validation of the horizon
string. A caller passing "30m" gets the same indicators as the
default - confusing if they expected horizon-adapted values.

### P3-9. `bert_sentiment._models` is process-global and never bounded

`bert_sentiment.py:117` - fine because only 3 models exist; pinned
intentionally. Worth a comment.

### P3-10. `fear_greed.update_fear_streak` `streak_started` is rewritten on every "neutral" case

`fear_greed.py:69-71` - when current `fg_value` is neutral, the
`streak_started` is rewritten to `now_str` every cycle. Should keep
its previous value (or null it out) so future debugging can trace
when the neutral streak began. Trivial.

### P3-11. `econ_dates.events_within_hours` returns events in calendar order, not chronological order from ref_date

`econ_dates.py:170-191` - events are sorted by date in
`_build_events` (line 136), so within a small window this orders
correctly. Not actively a bug; worth documenting.

### P3-12. `seasonality.compute_hourly_profile` empty-hours zero-fill biases the profile mean

`seasonality.py:67-71` - hours with zero samples are filled as
mean_return=0.0, mean_volatility=0.0. If a 24h cycle has 3 hours with
no data (e.g. low-volume Sun midnight), the global mean across all
hours is dragged toward zero. Better to leave the hour absent and
have `detrend_return` use the global mean as default.

### P3-13. `news_keywords._PATTERN_CACHE` is module-global, never cleared

`news_keywords.py:319` - fixed small set, fine. Documented in comment.

### P3-14. `crypto_scheduler` `MIN_GAP_SECONDS = 3000` (50 min)

`crypto_scheduler.py:40` - comments say "prevents double-fires within
same hour". 50 min is safely less than the 60 min cadence. Trivial
note.

### P3-15. `metals_orderbook._nocache` decorator is a no-op wrapper

`metals_orderbook.py:38-44` - `@_nocache` decorates `get_orderbook_depth`
and `get_recent_trades`, but those functions DO call `_cached(...)`
internally. The `__wrapped__` attribute lets tests reach in. Confusing
name - it doesn't disable cache; it exposes the wrapped function.
Cosmetic.

### P3-16. `forecast_signal._prophet_cache` declared but never read or written

`forecast_signal.py:33` - module-level dict never used. Dead code.

### P3-17. `data_collector.fetch_vix` is duplicate of `fear_greed.get_stock_fear_greed`'s VIX path

Two yfinance reads of `^VIX` with different return shapes. See P2-3.

### P3-18. `microstructure.detect_trade_throughs` threshold reasonable but never tuned

`microstructure.py:172` - XAGUSDT 0.001 USDT tick over ~$33 reference
price = 0.30 bps; XAUUSDT 0.1 USDT over $3000 = 0.33 bps. A 5 bps
threshold corresponds to a tick-jump of ~16 ticks on silver, which
is rare. The threshold is reasonable for noise filtering, but worth a
sensitivity test on the actual flagged trade-through rate.

### P3-19. `tickers.STOCK_SYMBOLS` iteration order non-deterministic in some call sites

Verified `alpha_vantage.py:238` does `sorted(STOCK_SYMBOLS)`. Other
iterations (`earnings_calendar.py:206`) don't sort. Order matters for
AV's 5 r/min limiter - same-order traversal means same-priority
ticker exhausts the budget first.

### P3-20. `api_utils.load_config` `_config_cache` global never expires when the file is deleted

`api_utils.py:21-36` - `os.stat` raises on missing file, except
clause keeps stale cache. Defensive but means a delete-and-recreate
config sequence won't be picked up until process restart. Symlink
target swaps would also miss.

### P3-21. `session_calendar._eu_dst` works on UTC datetime but doesn't validate tz-awareness

`session_calendar.py:50-67` - relies on `dt.year` arithmetic against
`tzinfo=UTC` constants. If the caller passes a naive datetime, the
boundary check may be off by an hour during the transition window
(01:00-02:00 UTC). The `get_session_info` wrapper does
`now.replace(tzinfo=UTC)` defensively, so practically safe; worth a
short docstring on `_eu_dst` saying "caller must pass UTC-aware dt".

## Tests missing

These are the gaps I'd open issues for. Order is by load-bearing-ness.

1. **`http_retry.fetch_with_retry` HTTP-header `Retry-After` handling.**
   No test verifies that a 429 with `Retry-After: 30` header waits 30s.
   Should be a parameterized test against a mock server. Critical
   for the rate-limit P0.

2. **`funding_rate` symmetric threshold test.** Property-based: for any
   `rate` in `[-1e-3, +1e-3]`, verify action distribution matches
   expectation. Currently no symmetry check.

3. **`price_source` fallback observability.** Test that a primary
   Binance failure results in a degraded-source flag on the returned
   DataFrame. Today the test set verifies the routing decision; not
   the fallback contract.

4. **`alpha_vantage._check_budget` daylight saving / TZ rollover.** Time-
   travel test (`freezegun`) verifying budget resets at provider-local
   midnight, not UTC midnight.

5. **`fear_greed.get_crypto_fear_greed` cache integrity.** Verify no
   direct API call when invoked twice within FEAR_GREED_TTL via
   different callers (or document the wrap contract loudly).

6. **`microstructure_state.persist_state` does not mutate
   `_ofi_history`.** Today the read function records, which the
   persist function calls, which double-counts. Test that
   `len(_ofi_history[ticker])` is unchanged after `persist_state()`.

7. **`crypto_macro_data._parse_expiry` locale independence.** Test
   under `LC_TIME=sv_SE.UTF-8` to verify month parsing still works.

8. **`econ_dates` event time accuracy.** Verify CPI/NFP show
   `hours_until = 12.5` (for 08:30 ET at 20:00 UTC ref), NOT
   `hours_until = 14:00 ref`. FOMC verify 18:00/19:00 UTC.

9. **`session_calendar` DST boundary integration.** Around each DST
   flip (2nd Sun Mar / 1st Sun Nov for US; last Sun Mar/Oct for EU),
   verify `get_session_info` returns the correct remaining minutes
   and `is_open` flag for the metals warrant session.

10. **`bert_sentiment._predict_per_text` failure handling.** Test that
    a partial-batch failure does NOT inject a fake neutral entry.

11. **`onchain_data` budget bound.** Stress test: 3 concurrent
    `get_onchain_data` calls with no persistent cache; assert at most
    6 BGeometrics calls fire.

12. **`metals_orderbook` thin-book defense.** Test that a book with 1
    bid level / 0 ask levels returns None, not a degraded snapshot.

13. **`data_collector._fetch_one_timeframe` cache eviction.** Verify
    that a 24h-TTL entry survives an LRU eviction pass (proves the
    `ttl` key is written).

14. **`news_keywords.score_headline` multi-keyword count vs max.**
    Verify that "war tariff crash" scores higher than "tariff" alone
    (currently they tie).

15. **`indicators.compute_indicators` zero-price guard idempotency.**
    Verify that a single zero close ANY where in the series triggers
    None - currently only `close <= 0` after the ffill is checked.

## Cross-cutting observations

**1. Live-vs-cached is not auditable end-to-end.**
The pricing layer makes routing decisions (`price_source.resolve_source`),
logs degradation (`logger.error` on fallback), but the resulting
DataFrame travels through the pipeline without a provenance attribute.
A signal that fires "BTC BUY at $67,000 with 87% confidence" doesn't
say "the OHLCV came from yfinance 12 minutes ago and the order book
came from Binance 3 seconds ago." That's exactly the bug class that
caused the "BTC 12h BUY phantom" memory entry - *and the data
collector's silent fallback is the most plausible cause*.

The fix is structural: every fetched DataFrame should carry a
`{source: "binance_fapi", fetched_at: <epoch>, fresh: bool}` metadata
block, and the signal-engine should propagate it into
`agent_summary_compact.json`. Today, downstream cannot distinguish
"live primary" from "stale fallback" inside the signal vote.

**2. Cache TTL is inconsistent and the eviction mechanism is
ttl-aware but the writers aren't.**
`_tool_cache` entries written from `_cached()` carry `ttl`; entries
written from `data_collector._fetch_one_timeframe` and direct
`_update_cache` callers may not. The LRU pass uses
`v.get("ttl", 3600)` so long-TTL entries get prematurely evicted.
Standardize on always-include-ttl on every write.

**3. Daily-budget tracking is per-module and fragile.**
Each provider has its own state machine: `alpha_vantage._daily_budget_used`,
`shared_state._newsapi_daily_count`, BGeometrics has none (relies on
12h TTL math + persistent cache seeding). Earnings_calendar bypasses
the AV budget entirely. Reset times are inconsistent (UTC vs
provider-local). A single `budget_registry` module that all providers
register against would eliminate the off-by-one risk in
`refresh_fundamentals_batch + earnings_calendar`, and make budget
exhaustion observable on the dashboard.

**4. Threading contracts are mostly correct, with one race in the
microstructure persist path.**
The `_buffer_lock` / `_cache_lock` / `yfinance_lock` separation is
sensibly designed. The bug is at the orchestration layer
(`persist_state` re-entering `get_microstructure_state` which records
OFI). Not a lock-correctness issue - a logical idempotency issue. The
broader codebase pattern of "pure read + explicit record" should be
applied consistently.

**5. The "MSTR-as-stock" is the only AV-budget-relevant ticker, but
all the AV scaffolding still runs.**
`STOCK_SYMBOLS = {"MSTR"}` per tickers.py:33. The
`refresh_fundamentals_batch` + earnings_calendar machinery is sized for
the 12-ticker pre-Apr-09 era. Either retire the unused budget machinery
or proactively put it back in service with a future ticker list - leaving
it half-engaged is the worst of both worlds.

**6. `forecast_signal` is disabled per tickers.py but the module still
runs Chronos forward passes if invoked.**
`run_forecasts()` writes to `forecast_predictions.jsonl` regardless of
the `DISABLED_SIGNALS` set in tickers.py. The signal-engine skips the
signal, but the forecast module has no awareness of its disabled
status - any caller that imports `run_forecasts` (the dashboard? a
manual script?) will incur ~5-15s of Chronos inference per ticker for
data that downstream voting ignores. Add an early-exit guard in
`run_forecasts` that consults `DISABLED_SIGNALS`.

**7. The `_cached` dogpile-prevention works correctly, but downstream
callers don't all know to handle `None` returns.**
Several call sites assume the return is "fresh-or-stale". For
example, `metals_cross_assets.get_copper_data()` returns whatever
`_cached` returns - including None during dogpile. Downstream
`get_all_cross_asset_data()` will then have `copper: None`. Whether
the consumer handles that gracefully varies by signal. Audit each
consumer for `if data is None: vote_hold()` patterns.

**8. The `Retry-After` JSON-only parse is the canary for a broader
issue: provider-shape coupling in shared infrastructure.**
`http_retry.py` is shared infra but encodes Telegram's response
schema. The right shape is: read the standard HTTP header first, then
let providers register custom parsers via a small dict (Telegram,
Binance's error JSON, AV's `Note`). Today the only handled case is
Telegram.

**End of review.** Counted: 4 P0, 20 P1, 19 P2, 21 P3.
