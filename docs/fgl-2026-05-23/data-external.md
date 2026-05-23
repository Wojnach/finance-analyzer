# Adversarial Review — DATA-EXTERNAL Subsystem

Date: 2026-05-23
Reviewer: empty-baseline adversarial review
Scope: portfolio/{data_collector, data_refresh, futures_data, fx_rates,
alpha_vantage, fear_greed, sentiment, onchain_data, news_keywords,
crypto_macro_data, earnings_calendar, econ_dates, fomc_dates, funding_rate,
microstructure, microstructure_state, social_sentiment, price_source,
bert_sentiment}.py

Focus areas: rate-limit safety, silent stale fallback, HTTP/JSON
robustness, symbol/interval correctness, FX direction, concurrency,
microstructure math.

---

## P0 — Critical (correctness / budget burn / silent live-data violation)

### P0-1 — `fx_rates.fetch_usd_sek` silently returns stale FX rate without surfacing in return value

File: `portfolio/fx_rates.py` lines 28-71

Behaviour: When the live API fails, the function returns
`cached_rate` directly. The age-staleness check only **logs** a warning and
sends Telegram — the returned value is indistinguishable from a fresh one.
Callers (`risk_management`, `monte_carlo_risk`, `portfolio_mgr`) cannot tell
they got a 7h-old rate vs the 10s-old rate.

This violates the live-prices-first rule in CLAUDE.md. A 12h-stale SEK rate
during a real currency shock (SEK has moved 8% in single weeks historically)
would mis-value the entire portfolio. The Telegram cooldown (14400s = 4h)
means the first stale read may not even fire an alert.

Fix: Return `(rate, is_stale, age_seconds)` tuple — or add a sibling
`fetch_usd_sek_with_metadata()` that callers performing portfolio valuation
can use to refuse stale rates entirely.

### P0-2 — `data_collector._fetch_klines` only takes yfinance_lock for the Alpaca branch when market is closed

File: `portfolio/data_collector.py` lines 253-268 and 280-294

`_fetch_klines` calls `yfinance_klines()` for stock tickers when the market
is closed, but only the **outer** `_fetch_one_timeframe` wraps it under the
shared `_yfinance_lock` (line 291). The dispatcher path on line 265 hits
yfinance with **no lock**:

```python
return yfinance_klines(ticker, interval=interval, limit=limit)  # NO LOCK
```

`_fetch_klines` is called from 5+ other code paths (e.g. via golddigger,
direct `binance_fapi_klines` callers, oil_grid_signal). Anything not going
through `_fetch_one_timeframe` enters yfinance from up to 8 worker threads
unserialized. yfinance is documented thread-unsafe (cookie/session state
race) — silent data corruption / empty DataFrames observed in past
incidents. The comment at line 274-277 explicitly states "yfinance is not
thread-safe; serialize calls with a shared lock" yet line 265 does not.

Fix: move the lock acquisition into `_fetch_klines` itself (or into
`yfinance_klines`) so ALL paths through this module are serialized.

### P0-3 — `crypto_macro_data._fetch_deribit_options` parses Deribit instrument names with `len(parts) != 4`, dropping all non-vanilla instruments silently

File: `portfolio/crypto_macro_data.py` lines 79-82

Deribit periodically introduces instrument variations (combos, daily expiries
with extra suffixes). Anything with `!=4` parts is silently dropped. If
Deribit changes its naming scheme (already happened in 2023), `expiry_data`
becomes empty and the function returns None — and the max-pain signal goes
dead silently. No log, no alert, no telemetry.

Worse: the max-pain calc loop (lines 139-166) is **O(strikes²)**. For a busy
expiry with 200+ strikes, this is 40K iterations per fetch — fine, but it
runs synchronously in the cycle hot path; check `OPTIONS_TTL=900` is
respected for ETH-USD too (it is, line 194).

Fix: log a WARNING when `expiry_data` ends up empty after parsing; add a
counter for dropped instruments.

### P0-4 — `data_refresh.download_klines` advances start_time using a hard-coded ms-per-candle table that lacks `1w`, `1M`, `5m`, etc.

File: `portfolio/data_refresh.py` lines 25-27

```python
ms_per_candle = {"1h": 3600000, "4h": 14400000, "1d": 86400000}[interval]
```

Any caller passing `5m`, `15m`, `1w`, `1M` triggers `KeyError`. The
`refresh_all()` function (line 78) only calls with `1h`, but `download_klines`
is exported and callable externally. More importantly, advancing by exactly
one candle interval can skip candles if Binance returns batches of 1000 with
gaps — the original `start_time = batch[-1][0] + ms_per_candle` will miss
the next candle when the last bar's open_time was already the most-recent
bar. Better: `start_time = batch[-1][0] + 1` and dedupe (which is done at
line 71, so the skip is the only real bug).

Fix: derive ms-per-candle from the actual interval string parsing OR raise a
clearer error for unsupported intervals.

### P0-5 — `earnings_calendar` Alpha Vantage path silently bypasses the daily-budget tracker

File: `portfolio/earnings_calendar.py` lines 49-52 (explicit comment
acknowledges this)

```python
# NOTE: earnings calls bypass alpha_vantage.py's _daily_budget_used counter
# because there is no public increment function exported from that module.
# Known limitation — earnings fetches consume 1 AV call each but are not
# reflected in the budget tracker.
```

AV free tier = **25 req/day**. STOCK_SYMBOLS has at least 1 ticker (MSTR);
historically up to 17. Each `_fetch_earnings_alpha_vantage()` is 1 call
**every 24h per ticker** — so 1-17 hidden calls/day. The `alpha_vantage.py`
budget check at line 232 (`if budget_used >= daily_budget: return 0`) only
sees its own counter, so on heavy-rebuild days the fundamentals batch can
burn the remainder of a budget already consumed by earnings, blocking the
fundamentals refresh entirely.

Fix: export an `alpha_vantage.increment_budget()` and call it from
earnings_calendar, OR have earnings_calendar call directly through
`alpha_vantage._cache_lock`-protected increment.

### P0-6 — `data_collector.binance_klines` returns DataFrame indexed by integer with `"time"` column NOT set as the index

File: `portfolio/data_collector.py` lines 93-98

```python
df = pd.DataFrame(data, columns=_BINANCE_KLINE_COLS)
for col in ...
df["time"] = pd.to_datetime(df["open_time"], unit="ms")
```

`open_time` is in milliseconds since epoch — `pd.to_datetime(..., unit="ms")`
correctly produces UTC-naive datetimes. But the parallel paths (Alpaca line
157, yfinance line 245) produce tz-aware datetimes (Alpaca returns ISO with
"Z"; yfinance returns tz-aware index). Downstream `compute_indicators` and
`technical_signal` may mix tz-aware and tz-naive timestamps depending on
source. This produces silent comparison failures (TypeError when comparing,
or wrong sort order). Search for any `df["time"] < other_ts` or `.between_time()`
usage downstream — they will behave differently across sources.

Fix: explicitly localize to UTC in all three paths
(`pd.to_datetime(...).tz_localize("UTC")` for the Binance path).

---

## P1 — Important (signal-correctness or rate-limit risk)

### P1-1 — `sentiment._fetch_newsapi_with_tracking` may under-count quota on partial responses

File: `portfolio/sentiment.py` lines 225-235

```python
if result:  # only count against budget when we actually got data
    newsapi_track_call()
```

If NewsAPI returns `200 OK` with an empty `articles: []` (no results for the
query), the call **still consumes the daily quota** on NewsAPI's side but is
NOT counted locally. Empty responses are common for stock tickers with rare
news days. Over time the local counter drifts below NewsAPI's actual count
and the 100/day cap will hit 429s before the local counter realizes.

Fix: increment on every successful HTTP 200, regardless of result content.

### P1-2 — `onchain_data._fetch_all_onchain` retries=0 but partial success is cached

File: `portfolio/onchain_data.py` lines 134-135, 224-243

`_api_get` uses `retries=0` to conserve the 8 req/hour budget — good. But if
4 of 6 metric fetches succeed and 2 fail (typical scenario at BGeometrics
quota boundary), `_save_onchain_cache` persists the partial dict (only 4
metrics) with the SAME `ts: time.time()`. The 12h TTL then prevents retry of
the failed metrics until tomorrow morning, even though only 1-2 metrics are
missing. Worse: `_safe_float(data.get("netflow"))` returns None for the
missing keys, and `interpret_onchain` silently produces an interp dict
without `netflow_signal`, `mvrv_zone`, etc. — downstream signals see "this
metric is not available today" for 12h after a brief outage.

Fix: persist `_fetched_keys: [...]` alongside `ts`, and skip the TTL check
for missing keys so they get retried at the next hour.

### P1-3 — `futures_data.get_open_interest` reads `data["openInterest"]` without `.get()`

File: `portfolio/futures_data.py` lines 60-65

```python
return {
    "oi": float(data["openInterest"]),
    "symbol": data["symbol"],
    ...
}
```

`fetch_json` returns None on failure, but if Binance returns a malformed
response (e.g. `{"code":-1121,"msg":"Invalid symbol"}` during a temporary
delisting glitch or symbol-renaming event), `data["openInterest"]` raises
`KeyError`. The inner `_fetch` raises, escapes `_cached()`'s try/except (it
catches Exception, line 109) — so under `_cached`, the function returns
stale-cached or None. **But** the circuit breaker sees `data is None` (line
38), records success on the actual HTTP success path, doesn't fire — so the
loop will retry every cycle, burning rate budget.

Fix: use `data.get("openInterest")` and check `is None` before float()
conversion. Same applies to `_fetch_open_interest_history`,
`_fetch_long_short_ratio`, etc. — all rely on direct dict subscript inside a
list comprehension.

### P1-4 — `econ_dates.is_macro_window` hardcodes 14:00 UTC for ALL event types

File: `portfolio/econ_dates.py` lines 272-274

```python
evt_dt = datetime.combine(
    evt["date"], datetime.min.time().replace(hour=14), tzinfo=UTC,
)
```

CPI/NFP release at 08:30 ET (12:30/13:30 UTC depending on DST). FOMC
announcement at 14:00 ET (18:00/19:00 UTC). GDP at 08:30 ET. Using a
universal 14:00 UTC means the lookback/lookahead window is offset by 1-6
hours from reality. With `lookback_hours=24`, the post-FOMC volatility
hangover window starts 5h late, missing the highest-vol period right after
the announcement.

Fix: per-event-type release-time map (FOMC=18:30 UTC, CPI/NFP/GDP=13:30 UTC
during US DST, 14:30 outside DST).

### P1-5 — `crypto_macro_data._fetch_deribit_options` max-pain confusion

File: `portfolio/crypto_macro_data.py` lines 136-167

The comment block at lines 148-162 includes commentary that conflicts with
the implementation. The code computes:
```python
call_pain = call_oi * max(0, candidate - strike)  # ITM calls
put_pain = put_oi * max(0, strike - candidate)    # ITM puts
total_pain = sum(call_pain + put_pain)
```
then picks the candidate that **minimizes** total_pain (line 164: `if ... total_pain < max_pain_value`). This is actually the **OTM expiry** (where most options expire worthless = max pain to **buyers**) — the standard formula picks the candidate that minimizes the seller payout, which is the strike where the most options expire OTM. The code labels this "max pain" but the variable name `max_pain_value` is updated only when we find a SMALLER value, which is correct but counterintuitive given the variable name. Initialization `max_pain_value = -1` is fragile — if `total_pain == 0` for every candidate (no OI), we'd pick the last candidate.

Fix: rename to `min_payout_value`; initialize to `float("inf")`.

### P1-6 — Avanza Tier 2 (price-only) tickers have no live-price fetch path

CLAUDE.md states Tier 2 = `SAAB-B, SEB-C, INVE-B` get Avanza price-only.
`price_source.resolve_source` defaults to Alpaca for bare uppercase tickers
(line 185) — Alpaca will 404 these symbols every cycle. This was likely a
removal cleanup oversight but I cannot find code in the reviewed scope that
fetches these. If any downstream code calls
`fetch_klines("SAAB-B")` the result is the yfinance fallback with WARNING
spam at the loop tempo, and an Alpaca circuit-breaker open event.

Fix: either add explicit yfinance routing for `.ST`-suffix Swedish tickers,
or remove them from any ticker iteration.

### P1-7 — `microstructure_state.persist_state` is not lock-protected at iteration

File: `portfolio/microstructure_state.py` lines 205-213

```python
def persist_state() -> None:
    state = {}
    for ticker in _snapshot_buffers:   # ← iterates dict without _buffer_lock
        ms = get_microstructure_state(ticker)
        ...
```

`_snapshot_buffers` is mutated under `_buffer_lock` in `_ensure_buffer` (line
47-52), which the fast-tick (10s) silver monitor also calls. Iterating it
without holding the lock can raise `RuntimeError: dictionary changed size
during iteration` when a brand new ticker first enters the buffer. Rare but
deterministic during cold-start.

Also: every call to `get_microstructure_state` inside the loop calls
`record_ofi`, **mutating** the history. So `persist_state()` has a side
effect of appending the current OFI value to the rolling history — calling
it on demand inflates the history vs the snapshot-tick path. The OFI z-score
denominator can be perturbed by spurious persist calls.

Fix: snapshot the ticker keys under the lock first; have a non-mutating
read-only variant of `get_microstructure_state` for persistence.

### P1-8 — `funding_rate._fetch_funding_rate` thresholds: BUY at `< -0.0001` is wrong sign

File: `portfolio/funding_rate.py` lines 44-49

```python
if rate > 0.0003:
    action = "SELL"
elif rate < -0.0001:
    action = "BUY"
```

The comment says ">0.03% = overleveraged longs = SELL" — correct (positive
funding means longs pay shorts, indicates long-crowding). But "-0.01% =
overleveraged shorts = BUY" — `-0.0001 = -0.01%`. So a funding rate of
-0.005% (`-0.00005`, very slight short crowding) returns HOLD; a funding
rate of -0.011% returns BUY. This is asymmetric: longs need 3x normal
funding to flag, shorts need only 1x normal. Normal funding is ~0.01%
positive; -0.01% is barely shorted, not overleveraged. The actual contrarian
BUY threshold should be ~-0.0003 (mirror of long side).

Fix: symmetric thresholds `rate < -0.0003 -> BUY`.

### P1-9 — `data_collector.fetch_vix` returns NO timezone/timing info, used for fear_greed signal

File: `portfolio/data_collector.py` lines 168-204

VIX closes 16:15 ET, so during the overnight Asian session the "current"
value is yesterday's close — but the function returns it as if it were live.
For the F&G signal computed in fear_greed.get_stock_fear_greed (line
126-171), this stale VIX dominates the score across the entire 17h overnight
window. Combine with `prepost=False` (default for `.history()`), and the
overnight VIX can mismatch reality after an Asian session crash by 10+
points.

Fix: include a `last_updated` timestamp; downstream caller should reject
stale data older than 24h on weekends.

### P1-10 — `fear_greed.get_stock_fear_greed` uses `period="5d"` regardless of weekend/holiday

File: `portfolio/fear_greed.py` line 135

On a 3-day weekend or Christmas week, `period="5d"` may return only 2 bars
(open + most recent), making `h.iloc[-1]` a holiday-stale value. No
freshness check. Used as primary stock F&G — signal silently bases decisions
on Wednesday's VIX on Christmas Eve.

Fix: check `(now - h.index[-1]).total_seconds()` and refuse if >36h.

### P1-11 — `data_collector.yfinance_klines` does not apply `_yfinance_limiter` rate limit

File: `portfolio/data_collector.py` lines 207-247

The function calls `yf.download(...)` directly. Rate limiting is applied
only at the dispatcher (line 264). External callers using `yfinance_klines`
directly (any code that imports it from `data_collector`) bypass the 30/min
limit. yfinance has aggressive anti-bot rate limiting and has been observed
to return empty DataFrames + cookie-rotated 401s under burst load.

Fix: acquire `_yfinance_limiter.wait()` inside `yfinance_klines` itself.

### P1-12 — `social_sentiment._fetch_subreddit` uses bare `requests.get` (no retry, no User-Agent rotation, no rate limit)

File: `portfolio/social_sentiment.py` lines 32, 65

Both `_fetch_subreddit` and `_search_subreddit` call `requests.get` directly
with no retry, no `fetch_with_retry`, no shared session. Reddit aggressively
rate-limits unauthenticated requests (60 req/10min) and returns 429
intermittently. Exception is caught at lines 110, 122 only as a print to
stdout — silently degrades to "no Reddit data" with no logger, no circuit
breaker, no quota tracking. Print statements go to stdout, not the logger,
so they vanish in Task Scheduler logs.

Fix: use `fetch_json` from `http_retry`; add a circuit breaker; use the
logger.

### P1-13 — `earnings_calendar._fetch_earnings_alpha_vantage` no timeout fallback if call hangs

File: `portfolio/earnings_calendar.py` lines 53-61

timeout=10 is set, but `fetch_with_retry`'s default retries=3 with backoff
2.0 means a hung AV call worst-case = 10 + 2 + 4 + 8 = 24s × 3 attempts =
72s. Called once per ticker per 24h — but during cache-miss after a process
restart, ALL stock tickers iterate together. With 17 tickers historically,
that's 17 × 72s = ~20 minutes of potentially blocked cycle time on a single
AV outage.

Fix: pass `retries=1` since this is a daily-cached metric.

### P1-14 — `data_collector.alpaca_klines` `lookback_days` is fixed per interval but does not adapt to `limit`

File: `portfolio/data_collector.py` lines 26-32, 123-125

```python
ALPACA_INTERVAL_MAP = {
    "15m": ("15Min", 5),   # 5 days lookback
    "1h": ("1Hour", 10),   # 10 days
    ...
}
```

Caller passes `limit=100` for 15m: 100 candles × 15min = 25h = ~1 trading
day, fits in 5d. Fine.
Caller passes `limit=100` for 1h: 100 candles × 1h ≈ 4 trading days, fits in
10d. Fine.
Caller passes `limit=200` for 15m: 200 candles × 15min = 50h ≈ 7-10 trading
days — exceeds 5-day lookback. `df.tail(limit)` silently returns 100 actual
bars instead of 200. Caller gets fewer bars than requested with no warning.

Fix: derive `lookback_days` from `max(default, candles_needed × interval_in_days × 1.5)`.

---

## P2 — Material concerns

### P2-1 — `crypto_macro_data._load_ratio_history` and `_load_netflow_history` use raw `open()` instead of `file_utils`

File: `portfolio/crypto_macro_data.py` lines 275, 397

CLAUDE.md rule 4 says "Atomic I/O only. Use file_utils.atomic_write_json,
load_json, atomic_append_jsonl. Never raw json.loads(open(...))." Append
already uses `atomic_append_jsonl` (good), but reads use raw `open()`.
Concurrent crash during a write+read can return a half-line. The
`json.JSONDecodeError` exception swallows it (line 284) so the impact is
silent data loss — the partial-line tail of the file silently disappears
from history.

Fix: factor a `load_jsonl(path)` helper into file_utils with safer parsing.

### P2-2 — `data_collector.binance_klines` interval validation absent

File: `portfolio/data_collector.py` lines 74-105

No validation that `interval` is a legal Binance interval. The infamous
`10m` bug (CLAUDE.md: "Binance: 10m interval does NOT exist (error -1120)")
would still crash silently here — Binance returns 400, fetch_with_retry's
`raise_for_status` raises, circuit breaker records failure, the call is
None. The error is logged but the calling code (signal modules) sees
"insufficient data" not "invalid interval". Same applies to the
`_binance_interval` shim in `price_source.py` lines 100-106 — `90m → 1h`
silently down-samples.

Fix: add an interval whitelist constant; raise a clearer error.

### P2-3 — `bert_sentiment._models` cache survives meta-tensor failures

File: `portfolio/bert_sentiment.py` lines 238-257, 314-319

The double-checked-locking pattern caches `entry` only if `_load_model`
returns successfully. The retry path in `_load_model` does raise on the
second meta-tensor detection (line 252) — good. But once a model IS cached
on GPU and a later CUDA OOM or weight corruption happens at predict time,
the model stays cached (no eviction path). Subsequent calls keep hitting the
corrupt model. The per-text fallback writes 0.33/0.33/0.34 neutral
placeholders for every headline — silent A/B-log pollution as documented in
the comment at line 220-225.

Fix: detect repeated per-text failures and evict the model from `_models`
to force a clean reload.

### P2-4 — `sentiment._fetch_yahoo_headlines` returns articles with `datetime.now(UTC)` when no `pubDate`

File: `portfolio/sentiment.py` lines 182-198

```python
pub = content.get("pubDate") or content.get("displayTime", "")
...
"published": pub or datetime.now(UTC).isoformat(),
```

If Yahoo returns articles with missing pubDate (it does — old archive
articles), the function stamps the **current time** as the published time.
Downstream dissemination_score (news_keywords.py line 244) detects "60%+ of
articles in 1h window" → returns 1.5x clustering multiplier — every fetch
where Yahoo serves stale articles triggers a false breaking-news amplifier.

Fix: drop articles with missing pubDate, or stamp them with a sentinel like
"unknown" that downstream code can filter.

### P2-5 — `data_collector.yfinance_klines` `prepost=True` but Alpaca path uses `feed="iex"` (no extended hours)

File: `portfolio/data_collector.py` lines 133, 226

When Alpaca path active: `feed="iex"` only — no extended hours.
When yfinance fallback (market closed): `prepost=True` — includes
extended hours.

This creates **regime discontinuity at market open/close**. The same
ticker's last bar at 16:00 ET (Alpaca closing print) vs 16:01 ET (yfinance
including the 16:00-20:00 ET extended session) can differ by 3-5% on
earnings days. Downstream indicators computed across this boundary produce
spurious crossovers.

Fix: pick one — either always include extended hours or never.

### P2-6 — `microstructure.detect_trade_throughs` mid_price uses arithmetic mean of two consecutive trades

File: `portfolio/microstructure.py` lines 207-210

```python
mid_price = (prev["price"] + curr["price"]) / 2.0
...
gap_bps = abs(curr["price"] - prev["price"]) / mid_price * 10000
```

For trade-through detection, the appropriate denominator is the BBO mid (the
quote at the time of the trade), not the average of two consecutive trade
prices. Using consecutive-trade mean means a large trade-through itself
inflates the denominator and shrinks its own gap_bps, biasing detection
DOWN. Trade-throughs that should fire at 5bps may be measured at 4.7bps
post-bias.

Fix: pass in the order book depth and use `(best_bid + best_ask)/2` as the
mid at the time of `curr`.

### P2-7 — `alpha_vantage._normalize_overview` returns dict even when ALL numeric fields are None

File: `portfolio/alpha_vantage.py` lines 100-126

If AV returns a valid response with `Symbol` set but every metric value as
`"None"` (happens during AV's intraday outages — they return a stub
response), the function returns a dict with all-None numeric values. This
gets cached for 24h. Downstream signals checking `pe_ratio is not None` get
nothing for the whole day.

Fix: require at least 2-3 metrics to be present, otherwise return None and
skip the cache write.

### P2-8 — `news_keywords.score_headline` returns max weight rather than aggregated

File: `portfolio/news_keywords.py` lines 139-159

Headline matching multiple critical keywords ("war AND tariff AND sanctions"
in a Russia/China headline) returns weight=3.0 same as a headline with one
word "war". Downstream sentiment weighting under-amplifies stacked critical
news.

Fix: use sum or sum-capped-at-Nx for additive amplification.

### P2-9 — `crypto_macro_data._append_netflow_history` calls `_load_netflow_history` (full read) every time, just to check latest timestamp

File: `portfolio/crypto_macro_data.py` lines 414-424

Reads the whole 30-day JSONL file every 6 hours just to check `latest_ts`.
O(N) for an O(1) check. Same anti-pattern at `_append_ratio_history` line
295. For 30 days of 6h netflow = 120 entries, negligible — but the pattern
scales badly and reads the same file twice in one logical operation.

Fix: maintain an in-memory `_last_append_ts` cache.

---

## P3 — Style / robustness nits

### P3-1 — `data_collector.fetch_vix` divides by `prev_close` with `> 0` check but no `is None` check

Line 184: `if prev_close > 0` — but pandas may return NaN; `NaN > 0` is False
so the branch is safe, but the explicit type wasn't documented.

### P3-2 — `econ_dates.py` hardcoded calendar for 2026 only

CPI/NFP/GDP dates are hand-curated for 2026-2027. Calendar must be manually
extended before Jan 2028 or all date proximity logic returns no-events
silently. No alarm fires when the calendar runs out.

Fix: add a startup assertion that the calendar extends ≥ N days into the
future; alert when < 90 days remain.

### P3-3 — `fx_rates.fetch_usd_sek` 15-min cache TTL is hard-coded inline (line 33)

Should be a module constant like `_FX_CACHE_TTL` for consistency with
`_FX_STALE_THRESHOLD`.

### P3-4 — `data_refresh.download_klines` uses `time.sleep(0.2)` instead of `_binance_limiter.wait()`

Bypasses the shared rate limiter, no concurrency safety with the main loop's
Binance budget.

### P3-5 — `onchain_data._fetch_exchange_netflow` ambiguous "latest" selection

Line 178: `latest = data[0] if isinstance(data[0], dict) else data[-1]`.
This conditional is unreachable — `data[0]` is always a dict if it's a list
of dicts. It just picks `data[0]` always, but the intent looks like
"newest". BGeometrics may return in either chronological direction depending
on endpoint; should explicitly sort by timestamp.

### P3-6 — `bert_sentiment._predict_per_text` swallows tokenizer errors silently

Line 442-453: catches Exception, writes a neutral placeholder, continues.
Good for resilience, but no counter / rate of failures is tracked. After 30%
of headlines silently fail, the aggregate sentiment vote is meaningless. No
operator visibility.

### P3-7 — `microstructure.compute_ofi` silently treats missing `best_bid`/`best_ask` as 0.0

Line 122-125: `prev["best_bid"]` raises KeyError if missing. Calling code
(`microstructure_state.compute_ofi(snapshots)`) trusts the buffer is
well-formed — but the snapshot accumulator (line 65-71) only stores keys
that exist. If `metals_orderbook` ever returns a depth with missing
`best_bid`, the accumulator stores `snapshot["best_bid"] = depth["best_bid"]`
and raises right there. Defensive code missing.

### P3-8 — `social_sentiment` not in price_source routing & uses `print()` for errors

File: `portfolio/social_sentiment.py` lines 110, 122 — `print(f"...")` to
stdout. In Task Scheduler logs these are typically discarded.

---

## 5-line Summary

1. **P0-1 / FX**: `fetch_usd_sek` returns stale-cached rate identically to
   live rate — caller cannot distinguish, portfolio valuations silently use
   7h+ old rates during outage; violates live-first rule.
2. **P0-2 / yfinance lock**: `_fetch_klines` only takes `yfinance_lock` via
   the outer wrapper when market is closed; the dispatcher path at line 265
   and direct callers of `yfinance_klines` race across 8 threads, risking
   silent corruption.
3. **P0-5 / Alpha Vantage budget burn**: `earnings_calendar` makes 1-17
   hidden AV calls/day that bypass the 25/day budget tracker, so daily
   fundamentals refresh can be silently blocked on quota exhaustion no one
   sees.
4. **P0-3 / Deribit parser**: `_fetch_deribit_options` silently drops every
   instrument with `parts != 4`; a Deribit naming-scheme change kills the
   max-pain signal with no log, no alert, returning None silently for the
   whole crypto macro voter.
5. **P1-8 / funding_rate threshold asymmetry**: BUY fires at -0.01% (barely
   shorted), SELL fires at +0.03% (3x normal) — the contrarian BUY side is
   3x more sensitive than SELL, biasing the funding signal long.
