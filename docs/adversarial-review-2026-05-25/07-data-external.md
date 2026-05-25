# Data-External Adversarial Review

Worktree: `Q:/finance-analyzer-worktrees/review-data-external/`
Scope: 21 modules in `portfolio/` that fetch external data (Binance, Alpaca,
yfinance, FRED, Alpha Vantage, NewsAPI, BGeometrics, CryptoCompare,
Deribit, Reddit, alternative.me F&G). ~5,860 LOC.

## P0 findings

### portfolio/shared_state.py:304-336 — NewsAPI daily quota burns on every restart
`_newsapi_daily_count` and `_newsapi_daily_reset` are module-level globals
with NO disk persistence. On every Layer 1 process restart (auto-restart
on crash, scheduled task, manual loop bounce), the counter resets to 0
even if the loop already burned 90 calls today. The reset gate at line
333 only triggers on a calendar-day boundary, never on restart, so a loop
that crashes/restarts twice could send 270 NewsAPI calls in a 24h window
(3× the 100/day free-tier budget) and trigger account suspension. Fix:
persist `(count, last_reset_utc_date)` to `data/newsapi_quota.json` and
reload on `newsapi_quota_ok()` first call after import. The metals
priority signals (XAU/XAG NewsAPI fetches every 20 min during 08:00–22:00
CET) would be the first casualty of suspension.

### portfolio/alpha_vantage.py:30-34, 162-168 — Alpha Vantage daily-budget counter not persisted
Same failure mode as NewsAPI: `_daily_budget_used` and `_budget_reset_date`
are in-memory only. `_check_budget()` only resets when the in-memory
`_budget_reset_date` doesn't match `today` — on restart, both are empty
strings, so the first call resets to 0. The 25/day free-tier limit will
be exceeded if the loop restarts mid-day after already consuming
fundamentals. Saving grace is `cache_ttl_hours=24` on the persisted
fundamentals cache, so most tickers won't re-fetch immediately, BUT
`earnings_calendar._fetch_earnings_alpha_vantage` (line 31-94) ALSO
consumes AV calls AND bypasses the budget counter entirely (acknowledged
in its own comment at line 49-52). Restart pattern: cache file >24h old
on 8 stock tickers + 8 stock tickers needing earnings = 16 AV calls
immediately on first cycle, with the budget counter reading "0/25 used".

### portfolio/onchain_data.py — BGeometrics has NO daily-budget or hourly-rate tracking
The module documents the 8 req/hour, 15 req/day free-tier limit at line 6
but implements neither. `_fetch_all_onchain` fires 6 sequential requests
with 1s sleep between (so 6 req in ~5s, well under 8/hr per call but
no enforcement across calls). On startup with stale cache, the call burns
6 requests immediately. If the cache becomes corrupted or the 12h TTL
expires while two threads concurrently call `get_onchain_data()`, the
`_cached()` dogpile guard helps but won't prevent restart-burn. A loop
restart loop (crash → restart → fetch → crash) on a corrupt cache could
exhaust the 15/day budget in 2–3 cycles. Fix: add a persisted budget
counter analogous to alpha_vantage, plus a per-hour rate limiter.

## P1 findings

### portfolio/http_retry.py:48-55 — HTTP retry ignores standard Retry-After header
The 429 handling at line 49-54 only knows about Telegram's body-payload
form (`resp.json()["parameters"]["retry_after"]`). The standard HTTP
`Retry-After` response header (used by NewsAPI, Alpha Vantage rate-limit
responses, Binance ban responses) is completely ignored — exponential
backoff is used instead. For Binance, ignoring `Retry-After` after a
418/429 can extend the IP ban (Binance increases ban duration when
client keeps requesting during an active ban). Fix: read
`resp.headers.get("Retry-After")` first, fall back to body parse for
Telegram-style responses, then fall back to exponential. Also the
random-jitter line 55 (`wait += random.uniform(0, wait)`) makes the
wait 1.0×–2.0× the spec'd value — fine for jitter on backoff, but
should NOT be applied on top of an explicit Retry-After value.

### portfolio/fx_rates.py:47-53 — Sanity-rejected FX rate is silently dropped without alert
When the API returns a rate outside `[7.0, 15.0]`, line 48 logs ERROR
and execution falls through to the stale-cache fallback. But
`_fx_alert_telegram` is only called if `cached_rate` is missing or
ancient — so an API returning persistently bad data (e.g. 5.0 SEK/USD
due to broken parser) would log errors every 15min but never Telegram-
alert because the cached value is still being returned silently. Add
a "rejected_due_to_sanity" alert path so the user knows the API is
broken even when caching masks it. Same code path also silently
discards a `rate == 0.0` (which would pass `not (7 <= 0 <= 15)` and
fall to stale cache).

### portfolio/data_refresh.py:1-87 — File saved as "futures" but fetched from spot endpoint
Line 6 imports `BINANCE_BASE` (the spot API) and line 31 calls
`{BINANCE_BASE}/klines`. The output is written to
`user_data/data/binance/futures/{symbol}_USDT_USDT-1h-futures.feather`.
If any ML training / backtest pipeline reads these files trusting the
"futures" label, the model is trained on spot data labeled as futures
— a silent label-swap bug. Fix: either rename the directory to
`spot/` (truthful) or switch to `BINANCE_FAPI_BASE`.

### portfolio/social_sentiment.py:32, 65 — No retry, no circuit breaker, no shared session
Direct `requests.get(...)` with `timeout=10` (line 32, 65). Reddit can
return 429 or 503 during traffic spikes; this code raises and the
caller's try/except just `print()`s the error (line 110, 122). No
backoff, so the next cycle hits Reddit again 60s later with the same
result. Also re-opens a fresh TCP connection per call — no
`requests.Session()`. Symptom is correlated burst-failures during
Reddit rate-limit events. Fix: route through `fetch_with_retry` like
the rest of the data-external modules; replace `print()` with
`logger.warning()`.

### portfolio/data_collector.py:230-247 — yfinance fallback masks empty-data failures as exceptions
`yfinance_klines` raises `ValueError("No yfinance data for ...")` on
empty (line 231). Inside `_fetch_one_timeframe` (line 312), the caller
catches the exception, returns `{"error": str(e)}`. Compute pipeline
treats this as a transient miss. There's no retry, no fallback to
Alpaca / Binance for the same ticker. yfinance returning empty for an
otherwise-valid ticker (rate-limit, intraday gap, prepost edge) =
permanent miss for that cycle. For TIMEFRAMES `("Now","15m"…)` (most
recent), this drops the most actionable signal silently. Fix: when
yfinance returns empty and the ticker has an Alpaca route available
(market_state was just-flipped-to-closed but Alpaca still has yesterday's
last bar), fall back to Alpaca for the most recent N bars.

### portfolio/macro_context.py:140-144 — DXY synth uses fake constant 58.0 for `value`
The synth path at line 143 builds `synth = 58.0 * (eurusd ** -0.576)`.
The 58.0 constant is acknowledged-arbitrary (line 137: "does NOT match
real DXY levels (~99)") and only the `change_*_pct` fields are usable.
But the returned dict at line 144 includes `value=round(synth.iloc[-1], 4)`
which downstream consumers might read. If anything reads `.value` (not
`.change_*`) from the intraday-DXY result and that ticker just happens
to be on the fallback path (primary DX-Y.NYB empty), the result is
~58 instead of ~99 — half the real DXY, completely misleading. Either
nullify `value` in the synth result, or document "synth value is
unreliable" explicitly in the returned dict (a `synth=True` flag).

### portfolio/earnings_calendar.py:27-28, 168-178 — Earnings cache not persisted to disk
`_earnings_cache` is in-memory only. On restart the 8-stock cache
rebuilds from scratch, calling Alpha Vantage 8 times (and AV's budget
counter doesn't see these calls per the comment at line 49-52). Combined
with the P0 NewsAPI/AV budget issues, a flapping loop could exhaust the
25/day AV budget purely on earnings refreshes. Fix: write to
`data/earnings_cache.json` with mtime, load on first access.

### portfolio/fear_greed.py:43-79 — `update_fear_streak` re-reads file every call without atomic compare-and-swap
Each F&G fetch (every 5min via FEAR_GREED_TTL=300) calls
`update_fear_streak()` which `load_json()` → mutate → `atomic_write_json()`.
There's no lock. With 8 worker threads possibly all hitting F&G near-
simultaneously after a cache TTL expire, two threads can both read
prev `streak_days=5`, both detect `is_new_day=True`, both write
`streak_days=6` — but a third worker could read between these writes
and double-increment. The H26 fix (line 41-42) prevents per-fetch
inflation but not concurrent-write races. Add a `threading.Lock`
around the read-modify-write or use file locking.

### portfolio/crypto_macro_data.py:208-310 — Gold/BTC ratio history file has read-modify-append race
`_append_ratio_history` (line 292) reads the file, checks latest
timestamp, then writes a new entry. Two cycles 1h apart could both
see "no entry in last 1h" and both append — doubling samples. Mostly
cosmetic given the 1h cadence but worth a lock or "only one writer per
day" semantics. Same pattern at `_append_netflow_history` (line 414).

### portfolio/data_collector.py:104, 108 — Binance default interval mismatch
`binance_klines(symbol, interval="5m"…)` and `binance_fapi_klines(...,
interval="5m"…)` default to `5m`. The `TIMEFRAMES` list passes explicit
intervals so this default isn't reached in practice from the
multi-timeframe collector, but any caller that omits the kwarg gets
`5m`. This is a footgun (cf. the CLAUDE.md note that Binance
`10m` doesn't exist) — `5m` is fine, but the function signature
suggests interval is optional when in practice every caller specifies
it. Document or make required.

## P2 findings

### portfolio/api_utils.py:21-36 — `load_config` raises on first call if config missing, but silently swallows mtime errors thereafter
The `try: ... except Exception` at line 27-35 covers the entire `stat()
+ open + json.load` block. If config is deleted while running, the
`Exception` is swallowed and the OLD `_config_cache` is returned. Good
for resilience, but a deleted config silently using stale credentials
could be confusing. Add a WARNING log on stat failure when cache exists.

### portfolio/onchain_data.py:42-67 — `_coerce_epoch` returns 0.0 silently for unknown timestamp formats
Line 67 falls through to `return 0.0` for any unparseable value with
only a DEBUG-level log. The comment correctly notes this forces a
cache miss (extra API call), but in the context of the P0 BGeometrics
budget issue, a single corrupted timestamp = one extra burned
request. Worth promoting the log to WARNING so it shows in normal
operation.

### portfolio/fx_rates.py:14-15 — Note disclaims migration but `_FX_STALE_THRESHOLD` not configurable
2h staleness threshold hardcoded; alert cooldown 4h hardcoded. Config
section for `fx` would let ops tune these without code changes.

### portfolio/metals_cross_assets.py:68-73 — Re-capitalizing column names is a fragile workaround
The legacy `df["Close"]` consumers depend on capitalized names while
the routed `fetch_klines` returns lowercase. Module rewrites in-place
on every call. Better to migrate the in-module getters to lowercase
once.

### portfolio/sentiment.py:33-42 — Hardcoded model script paths per OS
Lines 33-42 hardcode `Q:\models\...` (Windows) and
`/home/deck/models/...` (Linux). Any deployment outside these two
paths breaks. Config-ify under `config.models.{cryptobert_script, ...}`.

### portfolio/news_keywords.py:155 — Pattern → keyword string reconstruction is fragile
`pattern.pattern.replace(r"\b", "").replace("\\", "")` to reconstruct
the matched-keyword string is brittle. Store keyword alongside compiled
pattern in `_KEYWORD_PATTERNS` to avoid the string-mangle round-trip.

### portfolio/data_collector.py:44-52 — TIMEFRAMES uses non-standard `"1mo"` label with `"3d"` Binance interval
The label `"1mo"` historically meant "1 month of data" but the interval
is `"3d"` candles × 100 = 300d (~10mo). Comment at line 49 says
"~300d data, cache 4hr" which contradicts the "1mo" label. Either
rename to `"10mo"` or reduce limit. This is a documentation/clarity
issue — the actual data fetched is fine.

## P3 findings

### portfolio/social_sentiment.py:110, 122 — `print()` instead of `logger.warning()`
The two exception handlers use `print(f"    [Reddit r/{sub}] error: {e}")`
which bypasses the logging framework — won't be captured by
`agent.log` or filtered by log levels.

### portfolio/data_collector.py:262 — `_current_market_state` accessed without lock
Module-level state read from `_ss._current_market_state` at line 262
without holding `_cache_lock`. Stale read is fine functionally (off-by-
one cycle) but worth a comment that this is intentionally racy.

### portfolio/fear_greed.py:127-128 — Lock acquisition for VIX yfinance call is necessary, but lock release semantics around `h.empty` check could allow concurrent re-fetch
The lock is released after the `vix.history(...)` call returns (line 134-135
is inside the `with yfinance_lock` block). After release, line 136 checks
`if h.empty` outside the lock. Two threads can both call, both get empty,
both return None — fine. But two threads can also both succeed and both
proceed to compute VIX value redundantly. Cache the result instead of
recomputing.

### portfolio/futures_data.py:212-228 — `get_all_futures_data` doesn't parallelize 6 API calls
Sequential `get_open_interest` + `get_open_interest_history` + ... = 6
round-trips back-to-back. With each TTL-cached (5min OI, 15min funding),
amortized cost is low, but on first call after restart all 6 happen
sequentially. Pool them.

### portfolio/macro_context.py:13 — Hard-coded `CONFIG_FILE` path bypasses `load_config()` cache
Line 13 reads `config.json` directly inside `_fred_10y_fallback` (line
287) instead of using `portfolio.api_utils.load_config()`. Inconsistent
with rest of codebase and skips the mtime cache.

### portfolio/sentiment.py:34-37 — Model paths use raw strings; only `MODELS_PYTHON` is `r"..."`
Windows paths in lines 34-37 all use `r"..."` raw-string prefix
consistently, which is good. Linux paths don't need raw-string (no
backslashes). Cosmetic.

## Cross-cutting observations

1. **Daily-budget tracking is inconsistent.** Three external APIs have
   daily quotas (NewsAPI 100, Alpha Vantage 25, BGeometrics 15) and
   each implements (or doesn't implement) budget tracking differently:
   - NewsAPI: in-memory only, NOT persisted (P0)
   - Alpha Vantage: in-memory only, NOT persisted (P0); earnings path
     bypasses the counter entirely
   - BGeometrics: no tracking at all (P0)
   The right pattern is a single
   `portfolio/quota.py` module backed by `data/api_quotas.json` with
   `(api_name, calls_today, last_reset_utc_date)` rows and per-API
   `check()` / `record()` helpers. Persisted, locked, single source of
   truth.

2. **HTTP `Retry-After` header is universally ignored** in
   `http_retry.py`. Only Telegram's body-parameter form is parsed. This
   matters most for Binance (which uses HTTP headers + IP ban
   escalation if ignored) and NewsAPI/Alpha Vantage (standard headers).

3. **Earnings calendar / FX / earnings cache are in-memory only.**
   These rebuild on every restart. Combined with the budget issues,
   restart-flap = quota exhaustion.

4. **`requests.get()` direct calls bypass retry/CB infrastructure.**
   `social_sentiment.py` does this for Reddit. A grep over the broader
   codebase would likely find more — worth a Glob for `requests\.(get|post|put)\b`
   to find other un-retried callers.

5. **Stale-data fallback semantics differ per module.** `fx_rates` has
   explicit `_FX_STALE_THRESHOLD` + Telegram alert; `_cached()` allows
   `ttl * _MAX_STALE_FACTOR` silently; `onchain_data` allows
   `ONCHAIN_TTL * 2` silently. Inconsistent — a trader looking at any
   given data point can't tell if it's fresh or stale without checking
   the specific module's policy.

6. **Symbol-mapping is duplicated.** `futures_data.SYMBOL_MAP` and
   `funding_rate.SYMBOL_MAP` both define `{"BTC-USD": "BTCUSDT", ...}`.
   `price_source._BINANCE_SPOT` defines essentially the same map.
   `data_collector` indirectly uses `TICKER_MAP`. Five different
   ticker-mapping dicts; if XRP-USD is added tomorrow, all five need
   updates. Consolidate in `portfolio/tickers.py`.

7. **DataFrame column-casing convention is split.** Binance/Alpaca
   pipelines normalize to lowercase (`open, high, low, close, volume`).
   yfinance pipeline emits capitalized (`Open, High, ...`). The
   `price_source._fetch_yfinance` normalizes (line 156-159), but
   `metals_cross_assets._yf_download` re-capitalizes on the way out
   for backward compat. Net effect: every consumer must guess what
   casing it'll get. Pick one (lowercase, since most code already uses
   it) and migrate.

8. **Module-level test/example code at end of files.** All of
   `funding_rate`, `futures_data`, `data_collector`, `fear_greed`,
   `social_sentiment`, `sentiment`, `macro_context` end with
   `if __name__ == "__main__": ...` blocks that hit live APIs without
   any safeguards. If someone runs `python portfolio/macro_context.py`
   on a production loop machine, they consume real API budget. Move to
   `scripts/` or guard with an explicit `--live` flag.

## Files reviewed

- portfolio/data_collector.py (344 LOC)
- portfolio/alpha_vantage.py (321 LOC)
- portfolio/fear_greed.py (191 LOC)
- portfolio/futures_data.py (245 LOC)
- portfolio/fx_rates.py (91 LOC)
- portfolio/onchain_data.py (345 LOC)
- portfolio/sentiment.py (1,051 LOC)
- portfolio/social_sentiment.py (137 LOC)
- portfolio/bert_sentiment.py (471 LOC)
- portfolio/news_keywords.py (353 LOC)
- portfolio/earnings_calendar.py (216 LOC)
- portfolio/crypto_macro_data.py (461 LOC)
- portfolio/funding_rate.py (68 LOC)
- portfolio/http_retry.py (99 LOC)
- portfolio/api_utils.py (60 LOC)
- portfolio/metals_cross_assets.py (310 LOC)
- portfolio/econ_dates.py (282 LOC)
- portfolio/fomc_dates.py (57 LOC)
- portfolio/macro_context.py (403 LOC)
- portfolio/price_source.py (263 LOC)
- portfolio/data_refresh.py (90 LOC)

Cross-reference (out of scope but relevant):
- portfolio/shared_state.py — quota state lives here
- portfolio/golddigger/data_provider.py — `fetch_us10y` referenced by `_fred_10y_fallback`
