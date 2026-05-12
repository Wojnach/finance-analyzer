# Claude adversarial review: data-external (2026-05-12)

## Summary

The data-external subsystem is large (30 files, ~4.5k LOC) and shows
evidence of repeated post-incident hardening (atomic I/O, dogpile gates,
meta-tensor recovery, `_coerce_epoch`, decisive-margin sentiment gates).
Many of the obvious adversarial vectors enumerated in the prompt are
already mitigated.

That said, several real risks remain:

- **Cache-error silent fallback** (`shared_state._cached`,
  `portfolio/shared_state.py:110-126`) returns *stale* cached data on
  exception by rewinding the timestamp; combined with a wide
  `_MAX_STALE_FACTOR=3`, this means a 12 h on-chain cache can be served
  as if fresh for **36 h** without a single log line at WARNING level.
  This directly violates the "live prices first" rule for the on-chain
  voter and gold/BTC ratio history.
- **Timezone-mixing inside Chronos-2 context build**
  (`portfolio/forecast_signal.py:196`) constructs a tz-aware
  `pd.date_range` from `pd.Timestamp.now(tz="UTC")` but the upstream
  Binance candle times are *naive* (`portfolio/data_collector.py:96`,
  no `unit="ms", utc=True`). The two never meet, but Chronos-2's
  internal feature engineering can resample on the synthetic UTC index
  while the model was trained on a different cadence convention,
  silently degrading forecasts. The same naive↔aware mismatch in
  `forecast_prophet` is hidden by an explicit `.replace(tzinfo=None)`.
- **`http_retry` 429 handling is Telegram-specific only**
  (`portfolio/http_retry.py:43-49`). For every other 429 (Binance,
  Alpaca, Alpha Vantage, NewsAPI, BGeometrics, Frankfurter) the standard
  `Retry-After` HTTP header is ignored — falls back to exponential
  backoff, which can hammer a rate-limit-exhausted endpoint and burn
  the daily quota faster.
- **NewsAPI daily counter never resets to today's midnight**: the reset
  guard at `portfolio/shared_state.py:335-337` sets
  `_newsapi_daily_reset = now` once on the first call of the day, but
  the *threshold* it compares against is `today_start`. Restarts cross
  the date boundary correctly, but a process running across midnight
  UTC sees `_newsapi_daily_reset` get set to e.g. 03:00 UTC on day 1,
  and the next-day check `_newsapi_daily_reset < today_start` is then
  `false` only on the first call after midnight (correct) — actually
  works, but is brittle and could be silently broken by any caller that
  reads `_newsapi_daily_count` before `newsapi_quota_ok()`.
- **Alpha Vantage earnings calls (`portfolio/earnings_calendar.py:48-54`)
  bypass the AV daily-budget tracker entirely** by design. Up to 6
  earnings refresh attempts per restart can silently consume the AV
  quota, then `refresh_fundamentals_batch` runs and finds itself out of
  budget without ever logging why.
- **Sentiment ticker-synonym pattern leakage on stocks without a
  synonym list** (`portfolio/news_keywords.py:307-314`): falls back to
  `re.compile(r"\b" + ticker + r"\b")` for the bare ticker symbol. For
  short tickers (`MU`, `AMD`) this matches inside normal English
  phrasing (`MU` appears nowhere as English but `AMD` matches in
  "AMD-based", *and* the `bare uppercase isalnum` test accepts e.g.
  `"AI"`, `"USA"` if those were ever tickers). Adversarial focus item
  7 ("XAG matching X-A-G") is already mitigated for crypto/metals (the
  explicit synonym list takes precedence and ALWAYS uses `\b` word
  boundaries), so the asserted leak is not present, but a related leak
  for short-symbol stocks IS reachable.
- **Microstructure rolling state lives only in memory** for
  `portfolio/microstructure_state.py`. `persist_state` is called only
  by metals_loop, and the in-memory ring buffers (`_snapshot_buffers`,
  `_spread_buffers`, `_ofi_history`) are never seeded from disk on
  restart. After a metals_loop crash, the OFI z-score and multiscale
  OFI return 0 / insufficient data for at least 10 cycles. The
  `load_persisted_state` reader exists but is never wired into the
  accumulator's startup.
- **Price-source emergency fallback to yfinance is recursive-failure
  blind** (`portfolio/price_source.py:228-243`): when primary
  (Binance/Alpaca) fails, the emergency yfinance call is NOT logged at
  ERROR, just at WARNING (or DEBUG for last-resort tickers). The
  downstream consumer cannot distinguish "real-time Binance data" from
  "10-15 min stale yfinance data" — they receive the same DataFrame
  shape with no provenance flag.

## P0 — Blockers

None. Every P0-class hazard I checked (stale-while-revalidate without
upper bound, infinite retry, GPU lock deadlock, RSI bug from
incompatible ticker symbol map) has at least partial mitigation already
in place.

## P1 — High

### P1-1: `_cached` error path can serve 3×TTL-stale data without a WARN

`portfolio/shared_state.py:110-126` (the `except Exception` branch):

```python
_tool_cache[key]["time"] = now - ttl + _RETRY_COOLDOWN
return _tool_cache[key]["data"]
```

This deliberately rewinds the cache timestamp by `_RETRY_COOLDOWN=60s`
(so the next fetch retries in 60s) but **the stale data is returned**.
The age check `if age > max_stale` only refuses if the data is older
than `ttl * 3`. For on-chain BTC (`ONCHAIN_TTL=43200` s = 12 h),
`max_stale` is **36 h**, and the only log is `WARNING` at the boundary —
not during normal stale-serve.

Combined with `_save_onchain_cache` being called only when ≥1 metric
succeeded (`portfolio/onchain_data.py:236-241`), a full BGeometrics
outage will silently feed stale MVRV / NUPL into the on-chain voter for
up to 36 h. The voter then weights its decisions on 36 h-old data which
the system labels as fresh.

**Recommendation**: on the exception path, *return* the stale data
once, but stop serving it from future calls until success — i.e. mark
the cache entry as "served once after failure" and never serve again
without a successful refresh. At minimum log every stale-after-error
serve at WARNING with `age=`, not just the boundary case.

### P1-2: `forecast_signal` Chronos-2 builds tz-aware index from naive Binance times

`portfolio/data_collector.py:96`:
```python
df["time"] = pd.to_datetime(df["open_time"], unit="ms")  # NAIVE
```

`portfolio/forecast_signal.py:194-201` (Chronos-2 path):
```python
timestamps = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="h")
context_df = pd.DataFrame({"timestamp": timestamps, "target": prices, "id": ticker})
```

The `prices` list is built by `_load_candles` which discards `df["time"]`
entirely (only `df["close"].values.tolist()`). Chronos-2 then sees a
synthetic UTC index spaced exactly 1 h apart, ending at `now`. If
`_load_candles` runs at e.g. 23:47 UTC, the synthetic last-timestamp is
23:47 but the real last Binance candle closes at 23:00 (next at 00:00).
Chronos-2's 1 h-ahead prediction is therefore for **00:47 UTC**, not
**00:00 UTC** — a 47-minute phase offset that systematically shifts the
forecast horizon.

**Note**: forecast_chronos is currently re-enabled in `tickers.py`
(2026-04-21 note). The phase offset would explain part of its 45.4%
1 h-accuracy that the comment attributes to "Kronos pollution".

### P1-3: `http_retry` ignores `Retry-After` on non-Telegram 429s

`portfolio/http_retry.py:43-49` only parses Telegram's JSON
`parameters.retry_after`. Standard HTTP `Retry-After` header (Alpaca,
Binance, NewsAPI, BGeometrics all return it on 429/503) is not read.
The exponential backoff (`backoff_factor=2`, `DEFAULT_BACKOFF=1.0`,
`DEFAULT_RETRIES=3`) tops out at ~7s; if Binance signals "back off for
60s" via header, we hammer it 3 more times in 7s. For BGeometrics
(8 req/hour limit) this will burn the budget on each retry.

**Recommendation**: read `resp.headers.get("Retry-After")` first, fall
back to `parameters.retry_after`, then to computed backoff.

### P1-4: `data_collector._binance_fetch` returns naive timestamps despite Binance epoch being UTC

`portfolio/data_collector.py:96`:
```python
df["time"] = pd.to_datetime(df["open_time"], unit="ms")
```

This produces a **tz-naive** Series. Downstream code that does
`df.set_index("time").tz_localize("UTC")` would be safe, but any code
that compares `df["time"]` against an aware `datetime.now(UTC)` raises
`TypeError`. `portfolio/seasonality_updater.py:75` correctly uses
`pd.Timestamp(k[0], unit="ms", tz="UTC")` — `data_collector` does not.
Result: the "Now" timeframe DataFrame stashed in
`entry["_df"]` (`portfolio/data_collector.py:306`) is consumed by
enhanced signals that may then break or silently compare across naive
vs aware.

Within `compute_hourly_profile` the index `.hour` attribute works fine
on a naive index, but seasonal alignment assumes UTC — naive index
without explicit tz means CET-local server clocks would silently shift
the profile.

### P1-5: BGeometrics request burst can exceed 8 req/hr free-tier

`portfolio/onchain_data.py:206-241` `_fetch_all_onchain` calls 6
fetchers with `time.sleep(1)` between them. The free tier is **8 req/hr**
(not per minute) — 6 calls in 6 seconds is well within the *rate* limit
but if `_fetch_all_onchain` is called twice within an hour (e.g. due to
a cache miss during restart-warmup followed by `_save_onchain_cache`
failing because no metric succeeded → `any_success=False` → no cache
written), the next call hits the API again. Result: 12 req in <1 hour,
exceeding tier.

The `H12/DC-R3-5` restart seed (`portfolio/onchain_data.py:249-273`)
mitigates this, but only if the *file* cache is fresh. If the file
cache is stale and BGeometrics is in a slow degradation (responding
slowly enough that `fetch_json` returns None for 4 of 6 metrics),
`any_success` may be True for ≥1 metric, the cache gets re-written
with a *partial* dict, and the next fetch sees only 1-2 fresh entries
plus None for the missing 4. `_safe_float(None)` → `None`, and
`interpret_onchain` only emits zones for the metrics it has — masking
the partial outage.

### P1-6: Microstructure ring buffers never seeded on restart

`portfolio/microstructure_state.py:39-52` initializes empty deques
lazily. `load_persisted_state` exists (line 216) but is only used by
the orderbook_flow signal *reader* — never by the accumulator. After
metals_loop crash + restart:

- `_MIN_OFI_HISTORY_FOR_ZSCORE = 10` snapshots required before OFI
  z-score returns non-zero
- `_MIN_SPREADS_FOR_ZSCORE = 10` snapshots required before spread
  z-score is non-None
- At 10 s metals_loop fast-tick cadence that's ~100 s of warmup, but at
  60 s main loop cadence it's **10 minutes** before the orderbook_flow
  signal contributes anything but zeros.

During warmup the signal silently votes HOLD (`return 0.0` /
`return None`) — but `orderbook_flow` is currently DISABLED
(`portfolio/tickers.py:127`), so the impact is limited. If re-enabled
without fixing this, the first 10 cycles post-restart will be a
data blackout the operator has no visibility into.

### P1-7: Sentiment ticker-synonym fallback creates a `\b<TICKER>\b` regex for any short alpha symbol

`portfolio/news_keywords.py:307-314`:
```python
if not syns:
    if not short or not short.isalnum():
        return None
    return re.compile(r"\b" + re.escape(short) + r"\b", re.IGNORECASE)
```

For tickers in `STOCK_SYMBOLS` that aren't in the explicit
`_TICKER_SYNONYMS` dict, this builds `\bSHORT\b` case-insensitive. The
*current* STOCK_SYMBOLS is just `{MSTR}` and MSTR has explicit synonyms,
so the bug is dormant — but the design accepts ANY ticker. If
`STOCK_SYMBOLS` ever readds 2- or 3-char symbols (the historical
`MU`, `AMD`, `T`, `F`), random English text containing those substrings
or initialisms will trigger relevance. For the duration of this risk
window, sentiment is noisy on those tickers without it being measurable
through accuracy stats (looks like normal degradation).

Mitigation: require explicit synonym list before considering a stock
ticker's headlines relevant.

### P1-8: `fetch_vix` and `get_stock_fear_greed` use yfinance directly without `portfolio.price_source`

`portfolio/data_collector.py:168-204` (`fetch_vix`) and
`portfolio/fear_greed.py:127-171` (`get_stock_fear_greed`) both go
direct to yfinance. `^VIX` is on the `_CBOE_VOL_INDICES` allowlist so
this is technically allowed, but bypassing `fetch_klines` means:

1. No circuit breaker (price_source's primary→fallback path doesn't
   apply because there's no primary).
2. `fetch_vix` is not under any lock — yfinance is not thread-safe per
   the comments in `data_collector.py:274-277`, yet `fetch_vix` calls
   `vix.history(period="5d")` *outside* the shared `yfinance_lock`.

Result: from the 8-worker ThreadPoolExecutor, two concurrent
`fetch_vix()` calls race inside yfinance internals. The Yahoo network
session is module-global, and concurrent requests can produce
`Connection aborted` or empty frames intermittently. The function
swallows the exception with `return None`.

`get_stock_fear_greed` correctly uses the lock (`fear_greed.py:133`).
Fix is symmetric for `fetch_vix`.

## P2 — Medium

### P2-1: `_cached` "don't cache None" rule races with `_loading_keys`

`portfolio/shared_state.py:99-100`:
```python
if data is not None:
    _tool_cache[key] = {"data": data, "time": now, "ttl": ttl}
```

If func legitimately returns None (e.g. funding rate signal saw a
missing `lastFundingRate`), the key is removed from `_loading_keys`
(line 101) but never written to `_tool_cache`. The next cycle's call
will re-acquire `_loading_keys` and call `func` again — which is
correct behaviour for transient errors but burns rate-limit budget if
the upstream is in a steady-state-None mode (e.g. Binance maintenance).

For Alpha Vantage's earnings endpoint that's 6× rate-limit consumption
per ticker per cycle. The `_check_budget` guard in `alpha_vantage.py`
limits *batch* refresh to 25/day but doesn't gate ad-hoc calls from
`earnings_calendar.py`.

### P2-2: `_save_onchain_cache` writes ts but doesn't clear stale entries

`portfolio/onchain_data.py:206-241`: on partial success, the result
dict has the new `ts`, but if metric X failed, the old value for X is
*not in the dict* (the result is built from scratch each call). On
disk, `atomic_write_json` clobbers the whole file — so the stale value
for X is *gone*. But the in-memory `_tool_cache["onchain_btc"]` may
still hold the previous full dict from the prior successful call. The
two caches diverge: in-memory has metric X, on disk does not. Next
restart, `load_json(CACHE_FILE)` returns the on-disk partial version
and metric X is missing forever until the next full success.

### P2-3: `compute_gold_btc_ratio` reads from a sibling-process file with no lock

`portfolio/crypto_macro_data.py:208-262`: reads
`data/agent_summary_compact.json` to extract BTC and XAU prices.
`agent_summary_compact.json` is rewritten atomically by `main.py` each
cycle. Atomic rename is safe for readers — they always see either old
or new — but during the moment when both BTC and XAU are stale (e.g.
the writer is mid-cycle, signals not yet computed for one of them),
`get_signals("BTC-USD", {}).get("price_usd")` may return 0 / None, and
the ratio computation returns None → `_cached("gold_btc_ratio")` caches
that None for 1h. Actually `_cached` won't cache None (per P2-1) — but
this is the right behavior here. No real bug, just confusing dual
behavior between `_save_onchain_cache` (no None guard) and `_cached`
(None guard).

### P2-4: Funding-rate thresholds don't account for ETH vs BTC convention

`portfolio/funding_rate.py:44-49`:
```python
if rate > 0.0003: action = "SELL"
elif rate < -0.0001: action = "BUY"
```

These thresholds were tuned for BTC perp. ETH perp funding distribution
is wider (typical 0.005% - 0.04% range vs BTC's tighter 0.005% - 0.025%).
Asymmetric BUY (-0.01%) vs SELL (+0.03%) threshold means ETH triggers
SELL much more often than BUY on the same statistical "extreme" event.

Note this is consistent with the project memo about funding signal
being horizon-gated to 3h/4h only, where the threshold mismatch
matters less than at 1d.

### P2-5: `_fetch_yahoo_headlines` swallows the `news` API contract change

`portfolio/sentiment.py:131-156`: yfinance `Ticker.news` has changed
shape multiple times in 2024-2026 (top-level dict → nested `content`
dict). The code handles both with `content = item.get("content", item)`
but no telemetry on which shape was seen. If yfinance changes again,
the silent failure mode is `articles=[]` and the function returns
empty → sentiment for stocks degrades silently.

### P2-6: `metals_orderbook` returns None when bids[0] or asks[0] empty, masking maintenance windows

`portfolio/metals_orderbook.py:67-68`:
```python
if not bids or not asks:
    return None
```

Caller in `microstructure_state.accumulate_snapshot` then skips the
snapshot (`if depth is None: return`). During Binance FAPI maintenance
the order book can come back empty for tens of minutes — the buffer
stagnates, and after maintenance ends the OFI z-score is computed
against an outdated mean/std. No log at WARN level.

### P2-7: `economic events` use 14:00 UTC hard-coded; CPI/NFP actually release at 13:30 UTC summer / 14:30 winter

`portfolio/econ_dates.py:155, 180, 224, 272`:
```python
evt_dt = datetime.combine(evt["date"], datetime.min.time().replace(hour=14), tzinfo=UTC)
```

CPI and NFP are released at 8:30 AM ET. In US DST that's 12:30 UTC, in
standard time 13:30 UTC. Using 14:00 UTC for both creates a 30-90 min
phase error in the `hours_until` calculation that propagates into the
econ_calendar signal's proximity gates. For a 24-hour lookback window
this is small; for a `≤6h` pre-event gate it can mistakenly include or
exclude bars from the actual release moment.

### P2-8: `data_collector.fetch_vix` silently swaps yfinance multi-index without locking

(`portfolio/data_collector.py:178-184`) — see P1-8.

### P2-9: `dissemination_score` ISO timestamp parsing inconsistent

`portfolio/news_keywords.py:222-230`:
```python
pub_str = str(pub).replace("Z", "+00:00")
ts = datetime.fromisoformat(pub_str)
```

If `pub` is a string like `"2026-05-12T10:30:00.123Z"`,
`fromisoformat` in Python <3.11 raises `ValueError` for the millisecond
+ Z combination. The code catches `ValueError` so the whole article's
timestamp is dropped from the clustering check — which can be a
material amount on Bloomberg/Reuters feeds.

### P2-10: BERT in-process predict swallows model corruption per-text

`portfolio/bert_sentiment.py:408-453` (`_predict_per_text`): on any
exception per text, emits a `(neutral, 0.0)` placeholder and continues.
This is documented as a safety net, but the comment in `_load_model`
(line 217-237) acknowledges this exact pattern was a silent A/B-log
corruption mode in 2026-05-03. The meta-tensor detection now runs at
*load* time, but a load-time-pristine model can still raise during
forward pass (CUDA OOM mid-inference if `BERT_SENTIMENT_USE_GPU=1` was
set and llama-server reloads concurrently). The per-text path then
fills 100% of headlines with `(neutral, 0.33)` and the primary sentiment
result becomes neutral with high confidence — without a single WARN
above the per-call `_predict_per_text` line.

### P2-11: `seasonality.compute_hourly_profile` requires `klines_1h.index.hour` but data_collector returns reset index

`portfolio/data_collector.py:246`:
```python
df["time"] = df.index
df = df.reset_index(drop=True)
```

After this the DataFrame has a RangeIndex, not a DatetimeIndex. The
`compute_hourly_profile` precondition `if hasattr(df.index, "hour")`
returns False on RangeIndex and the function returns None. In practice
this is bypassed because `seasonality_updater._fetch_hourly_klines`
explicitly does `df.set_index("time")` (line 84) so the issue is
contained, but anything outside `seasonality_updater` that passes a
data_collector DataFrame to `compute_hourly_profile` will silently fail.

## P3 — Low

### P3-1: `_fetch_subreddit` uses raw `urllib.request` instead of `http_retry`

`portfolio/social_sentiment.py:30-34, 65-67`: Reddit's public JSON API
will return 429 + `Retry-After` if abused. The current code uses
`urllib.request.urlopen` with no retry, no User-Agent rotation. The
Telegram report path calls `get_reddit_posts` for at most BTC/ETH/PLTR/
NVDA — minimal risk, but inconsistent with the rest of the codebase's
`fetch_json` discipline.

### P3-2: Hard-coded `time.sleep(0.2)` in `data_refresh.download_klines`

`portfolio/data_refresh.py:48`: a 200ms sleep between Binance kline
chunks during the historical backfill loop. With 1000-candle batches
over 365 days at 1h interval that's ~9 batches → 1.8s total — fine for
the backfill, but doesn't honour `_binance_limiter`. If two backfills
run concurrently (e.g. operator runs `python -m portfolio.data_refresh`
while the main loop is also fetching Binance), the limiter is bypassed.

### P3-3: `funding_rate` ETH threshold asymmetry (see P2-4) is not a bug per se but worth documenting in the code

### P3-4: `econ_dates.next_event` defaults `hours_until` to `max(0.0, ...)`

`portfolio/econ_dates.py:160`: events whose 14:00 UTC anchor is in the
past (but whose date is today) report `hours_until=0.0`. Combined with
the 14:00 hardcode (P2-7), this means any event at 13:30 UTC effectively
"happens" at 14:00 UTC in our calculation — calls to
`events_within_hours(0.5)` after the real release but before our
synthetic anchor report no current event.

### P3-5: `price_source._fetch_yfinance` calls `df.columns.droplevel(1)` unconditionally

`portfolio/price_source.py:150-151`: `droplevel(1)` raises if columns
aren't MultiIndex. Currently guarded by
`isinstance(df.columns, pd.MultiIndex)` so it's safe, but the
`_fetch_yfinance` two-tail pattern (one for non-MultiIndex flatten in
`yfinance_klines`, one for MultiIndex droplevel here) is inconsistent.

### P3-6: `_pct_change` returns float('nan'), callers don't always check

`portfolio/metals_cross_assets.py:85-89`: returns `float("nan")` when
insufficient data. The callers (`get_copper_data`, etc.) include the
NaN in the result dict. Downstream consumers (`portfolio/signals/
metals_cross_asset.py`) may not all check for `math.isnan`, and signal
voting on NaN compared with thresholds is False for both sides → forced
HOLD with no log line. Indirect noise rather than a real failure.

### P3-7: `fx_rates._fx_cache` uses non-monotonic time

`portfolio/fx_rates.py:30-33`: `now = time.time()` for both staleness
math and Telegram alert cooldown. If wall-clock is set backwards (NTP
adjustment), the staleness check `now - cached_time < 900` can be
negative — that's still `< 900` so a stale rate is still served, but
the alert cooldown also moves and could spam Telegram.

### P3-8: `forecast_signal._load_candles` returns None silently on every kline error

`portfolio/forecast_signal.py:60-63`: catches all exceptions, logs at
DEBUG. If Binance is down for an extended window, `run_forecasts`
silently produces no entries for the affected tickers — no WARN,
no entry in `forecast_predictions.jsonl`, no metric to alert on. The
`forecast_accuracy` tooling will report "no data" rather than "fetch
failed".

### P3-9: `crypto_macro_data._load_ratio_history` opens file without `file_utils.atomic_*`

`portfolio/crypto_macro_data.py:275`: direct `open(... encoding="utf-8")`
read of `gold_btc_ratio_history.jsonl`. Reads are not write-atomic but
JSONL is line-atomic, so partial writes from a sibling process can
leave a half-written final line that `json.loads` rejects. The except
just `continue`s, dropping that line silently. Minor, given the file
is append-only and one bad final line is unlikely to misrepresent the
trend.

### P3-10: `session_calendar._eu_dst` uses naive last-Sunday math without honoring tz

`portfolio/session_calendar.py:50-67`: the last-Sunday-of-March/October
calculation uses UTC midnight, which is correct for Europe (DST flips at
01:00 UTC). But the EU DST rule is "last Sunday at 01:00 UTC" — the
code anchors at exactly 01:00 UTC. For the single hour `00:00-01:00`
on the DST flip day, the function may return the wrong offset. Edge
case only relevant for warrants trading between 02:15 and 03:15 CET on
DST-flip mornings (which is during warrant open at 08:15 CET → no
overlap, so dormant bug).

## Tests missing

1. **`_cached` stale-after-error window**: no test verifies that data
   stored ≥ `_MAX_STALE_FACTOR * ttl` ago is rejected. Specifically
   need a test that mocks `func` to raise, advances time past
   `_MAX_STALE_FACTOR * ttl`, then asserts `_cached` returns None and
   logs WARN.

2. **`http_retry` Retry-After header parsing**: no test for non-Telegram
   429 with `Retry-After: 60` header. Currently the code ignores it
   (P1-3) — once fixed, regression test needed.

3. **NewsAPI quota reset across UTC midnight**: no test for
   `newsapi_quota_ok` reset across the date boundary. Time-freeze
   tests should set `_newsapi_daily_count = 89` at 23:59:59 UTC, then
   advance time to 00:00:01 UTC and assert counter resets to 0 and
   `newsapi_quota_ok()` returns True.

4. **Alpha Vantage earnings bypassing daily budget**: no test verifying
   that 6 `_fetch_earnings_alpha_vantage` calls don't bleed into the
   `refresh_fundamentals_batch` budget. Should fail today (P1-5
   adjacent).

5. **Binance kline tz-awareness**: no test that
   `binance_klines(...)["time"].dt.tz` is UTC. Currently it's `None`
   (naive). A test asserting tz-awareness would surface P1-4.

6. **Forecast Chronos-2 time alignment**: no test for the
   synthetic-vs-real timestamp phase offset (P1-2). A test that mocks
   `pd.Timestamp.now` to 23:47 UTC and verifies the first forecast
   timestamp equals the candle close time, not `now`.

7. **Microstructure ring-buffer restart hydration**: no test that
   `accumulate_snapshot` populates buffers from
   `load_persisted_state(ticker)` on first call after process restart.
   Currently the function silently warms up from zero.

8. **Sentiment relevance for short-symbol stocks**: no test that
   `is_relevant_headline("AMD-based research", "AMD")` correctly rejects
   noise. The current synonym list for AMD is `["amd"]` so the test
   would pass today; need a test for a *new* short ticker without an
   explicit synonym (e.g. `is_relevant_headline("...the F...", "F")`).

9. **`_save_onchain_cache` partial-merge regression**: no test for the
   in-memory ↔ on-disk divergence under partial fetch failure (P2-2).
   A test should pre-populate the in-memory cache with all 6 metrics,
   simulate a fetch where only 1 metric succeeds, then restart and
   verify the on-disk + reload state contains 6 metrics (current
   behavior loses 5).

10. **Funding-rate ETH threshold asymmetry**: no test on per-ticker
    threshold tuning. A test using historical ETH funding distribution
    quantiles to assert the BUY/SELL thresholds map to roughly
    equivalent percentile extremes for BTC and ETH.

11. **`fetch_vix` thread-safety**: no test that 8 concurrent
    `fetch_vix()` calls don't race. The `get_stock_fear_greed` test
    suite covers VIX via the locked path; nothing covers the unlocked
    `data_collector.fetch_vix`.

12. **Econ-event 14:00 UTC hardcode**: no test verifying
    `next_event(...)` for a CPI day returns the correct
    `hours_until` (13:30 UTC actual vs 14:00 UTC anchor).

13. **`price_source` provenance flag**: no test asserts the consumer
    can distinguish "real Binance data" from "yfinance emergency
    fallback". The DataFrame shape is identical; ideally the function
    would attach an attribute (e.g. `df.attrs["source"]`).

14. **`is_relevant_headline` cache invalidation**: `_PATTERN_CACHE`
    grows unbounded if a process sees many distinct uppercase tickers
    over a long run. No test for cache eviction / growth.

15. **`dissemination_score` ISO ms-Z parsing**: no test for
    `2026-05-12T10:30:00.123Z`-style timestamps on Python 3.10.
    `fromisoformat` rejects this on older Pythons → article timestamp
    silently dropped from clustering check.
