# Adversarial Review — Data-External Subsystem

**Date:** 2026-05-17
**Reviewer:** Claude (Opus 4.7)
**Scope:** portfolio/{data_collector, fear_greed, sentiment, social_sentiment,
alpha_vantage, futures_data, onchain_data, fx_rates, crypto_macro_data,
funding_rate, news_keywords, earnings_calendar, econ_dates, fomc_dates,
price_source, http_retry, microstructure_state, metals_orderbook,
metals_cross_assets}.py

**Verdict:** 3 P1 issues, 6 P2 issues, 2 P3 notes. The credential-leak path
in `http_retry.py` (via Telegram URLs containing bot tokens) is the most
urgent finding. The bare-`requests` calls in `social_sentiment.py` and the
silently-empty yfinance DataFrame fallback in `price_source.py` both have
production-impact failure modes that are easy to fix.

---

## P1 — Critical

### P1-1. Telegram bot token leaked to logs on every retry / failure

**File:** `portfolio/http_retry.py:50–55`, `64`, `67–68`
**Confidence:** 95

```python
50:            logger.warning("HTTP %s from %s, retry %d/%d in %.1fs",
51:                           resp.status_code, url, attempt + 1, retries, wait)
...
54:            logger.error("HTTP %s from %s after %d retries",
55:                         resp.status_code, url, retries)
```

```python
63:            logger.warning("%s from %s, retry %d/%d in %.1fs",
64:                           e.__class__.__name__, url, attempt + 1, retries, wait)
...
67:            logger.error("Request failed after %d retries: %s - %s",
68:                         retries, url, e)
```

`http_retry.fetch_with_retry()` logs the full request URL at WARNING and ERROR
on every retryable status and on connection/timeout failures. Telegram
sender callers pass the URL as `f"https://api.telegram.org/bot{token}/sendMessage"`
(confirmed in `portfolio/message_store.py:137-142, 160-165`,
`portfolio/telegram_notifications.py:55, 75`, `portfolio/telegram_poller.py:130, 375`,
`portfolio/avanza_orders.py:275`). Every Telegram 429/500/503 or connection
flake — and 429s are routine, the code at line 44-49 explicitly handles
Telegram's `retry_after` — writes the bot token into the log file.

`agent.log` is rotated to disk, sometimes shared during debug sessions or
adversarial reviews. The repo history already records one config.json key
leak (CLAUDE.md "exposed API keys on Mar 15, 2026"). This is the same class
of bug.

**Fix:** in `fetch_with_retry`, redact `/bot<token>/` segments before
logging. One liner:
```python
def _redact(url):
    return re.sub(r"/bot[0-9]+:[A-Za-z0-9_-]+/", "/bot***/", url)
```
Apply to every `url` reference in logger calls. Alternatively, accept a
`log_url` kwarg defaulting to `url` and have Telegram callers pass a
redacted form.

---

### P1-2. `social_sentiment.py` bypasses retry / rate-limit / circuit breaker

**File:** `portfolio/social_sentiment.py:29–33`, `59–66`, `103–110`, `113–122`
**Confidence:** 90

```python
29:def _fetch_subreddit(sub, keywords, dedicated, per_sub):
30:    posts = []
31:    url = f"https://www.reddit.com/r/{sub}/hot.json?limit={per_sub + 5}&raw_json=1"
32:    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
33:    resp.raise_for_status()
```

```python
103:    for sub, dedicated in subreddits:
104:        try:
105:            fetched = _fetch_subreddit(sub, keywords, dedicated, per_sub)
106:            for p in fetched:
107:                if p["title"] not in seen:
108:                    seen.add(p["title"])
109:                    posts.append(p)
110:        except Exception as e:
111:            print(f"    [Reddit r/{sub}] error: {e}")
```

Both `_fetch_subreddit` and `_search_subreddit` use raw `requests.get` rather
than `fetch_with_retry` / `fetch_json`. Consequences:

1. **No exponential backoff** — Reddit's unauthenticated API caps at ~60/min
   per IP. A burst of failed retries from the cycle loop's 8-worker pool can
   immediately push past the rate limit and cause hour-long bans (the
   convention list at the top of this brief explicitly calls out
   "Retry without backoff → ban or rate-limit explosion").
2. **No circuit breaker** — repeated Reddit 5xx will burn ~1.6s per cycle
   (4 subs × 2 calls × 0.2s connect; longer if Reddit is hanging) regardless
   of upstream health.
3. **`print()` not `logger`** — failures emit to stdout and are invisible to
   `data/critical_errors.jsonl`, the health module, and dashboards. The
   sentiment voter then silently runs without Reddit input.
4. **`raise_for_status()` lets 429 propagate** — but the calling `try` in
   `get_reddit_posts()` only `print`s; no Retry-After handling, no cooldown.

**Fix:** route all Reddit calls through `fetch_json` (gives retry + Retry-After
handling + structured logging). Either add a `_reddit_limiter` in
`shared_state.py` similar to `_binance_limiter`, or wrap the calls in
`@_cached` with a generous TTL (sentiment voter polls infrequently
anyway).

---

### P1-3. `price_source._fetch_yfinance` returns empty DataFrame instead of raising

**File:** `portfolio/price_source.py:146–160`, `213–243`
**Confidence:** 85

```python
146:    df = yf.download(
147:        ticker, period=p, interval=interval,
148:        progress=False, auto_adjust=True,
149:    )
150:    if isinstance(df.columns, pd.MultiIndex):
151:        df.columns = df.columns.droplevel(1)
152:    if df.empty:
153:        return df
```

```python
228:        if source in ("binance_fapi", "binance_spot", "alpaca"):
229:            logger.error(
230:                "price_source: primary source %s FAILED for %s (%r). "
231:                "Falling back to yfinance. Investigate the primary outage.",
232:                source, ticker, exc,
233:            )
234:            try:
235:                return _fetch_yfinance(ticker, interval, period=period, limit=limit)
```

The yfinance fetcher returns an empty DataFrame (line 152-153) instead of
raising. `fetch_klines` wraps `_fetch_yfinance` in `try/except` (line 234)
but the `if df.empty: return df` path never raises, so an empty DataFrame
propagates to callers. The brief's exact wording: *"yfinance returning
empty DataFrame treated as zero price (would HOLD all decisions silently
OR worse, cause divide-by-zero)."*

Confirmed leak path: `metals_cross_assets._yf_download` (lines 56-73) calls
`fetch_klines` and on empty DataFrame returns `pd.DataFrame()`. The callers
in the same file (`get_copper_data`, `get_gvz`, `get_gold_silver_ratio`,
etc.) check `if df.empty or "Close" not in df.columns: return None` — so
those handle it. But other callers do NOT:

- `portfolio/macro_context.py:39-41, 114-118, 311-317` calls
  `fetch_klines(...)` without consistent empty checks.
- `portfolio/fish_monitor_smart.py:149, 259, 284` calls
  `fetch_klines(...)` and the downstream consumer assumes a non-empty
  frame.
- `portfolio/market_health.py:78-82` — fetches and then asserts via
  `.iloc[-1]` which would `IndexError` on empty.

`SourceUnavailableError` (defined line 84-85) is precisely the type intended
for this case; the function chooses to silently degrade instead. This is
the worst possible failure mode for a price router because every downstream
indicator computation either crashes with confusing tracebacks or
silently treats `nan`/0 as a real signal.

**Fix:** in `_fetch_yfinance`, replace `if df.empty: return df` with
`if df.empty: raise SourceUnavailableError(f"yfinance returned empty for {ticker}")`.
Then `fetch_klines`'s outer except will catch it, log, and either fall
back or re-raise per the existing logic.

---

## P2 — Important

### P2-1. `econ_dates.py` uses 14:00 UTC for all event types

**File:** `portfolio/econ_dates.py:154–156`, `180–181`, `224–225`, `272–274`
**Confidence:** 85

```python
154:            evt_dt = datetime.combine(evt["date"], datetime.min.time().replace(hour=14),
155:                                      tzinfo=UTC)
```

Hard-codes 14:00 UTC for *every* event type. Actual release times:

- **CPI / NFP:** 08:30 ET → 12:30 UTC (DST) or 13:30 UTC (winter)
- **FOMC announcement:** 14:00 ET → 18:00 UTC (DST) or 19:00 UTC (winter)
- **GDP advance:** 08:30 ET, same as CPI/NFP

The 14:00 UTC value is wrong by ~5h for FOMC and ~5h for CPI/NFP. With
`lookback_hours=24` / `lookahead_hours=72` this is still inside the window,
so the `is_macro_window` gate keeps working. But finer-grained logic in
`recent_high_impact_events(hours=24)` could report a CPI as ~5h fresher
or older than it actually was, mis-classifying the post-event hangover.

Brief flags this explicitly under P2 ("Earnings calendar timezone confusion
(US ET vs UTC)"). Same root cause applies here.

**Fix:** parameterize per event type (a dict of `{"FOMC": time(18,0,
tzinfo=UTC), "CPI": time(13,0), …}`) and dispatch from `_build_events()`.

---

### P2-2. FOMC / CPI / NFP / GDP date lists are hard-coded and will drift

**File:** `portfolio/fomc_dates.py:13–22, 25–34, 40–57`,
`portfolio/econ_dates.py:23–51, 57–85, 91–103`
**Confidence:** 90

```python
13:FOMC_DATES_2026 = [
14:    date(2026, 1, 28), date(2026, 1, 29),
15:    date(2026, 3, 17), date(2026, 3, 18),
...
22:]
```

```python
23:CPI_DATES_2026 = [
24:    date(2026, 1, 14),   # Dec 2025 CPI
25:    date(2026, 2, 12),   # Jan 2026 CPI
26:    date(2026, 3, 11),   # Feb 2026 CPI
```

Brief explicitly flags this under P2 ("FOMC date hard-coded list (drifts
annually)"). Confirmed: every economic event in the system is stored as a
literal list. 2027 dates are provisional / forward-projected; the BLS
release schedule for 2027 is not yet final. When dates shift (BLS revises
mid-year, FOMC adds an unscheduled meeting, GDP timing changes), the
signal-engine silently uses stale dates and `is_macro_window` lies.

**Fix:** either (a) fetch from FRED's release calendar endpoint (free,
already configured) with persistent cache, or (b) add a startup check
that fails loudly when any date list extends beyond `today + N` so a
human is forced to update it.

---

### P2-3. `_fetch_yfinance` calls bypass the `yfinance_lock`

**File:** `portfolio/price_source.py:127–160`
**Confidence:** 85

```python
127:def _fetch_yfinance(
128:    ticker: str, interval: str, period: str | None = None, limit: int | None = None,
129:) -> pd.DataFrame:
...
146:    df = yf.download(
147:        ticker, period=p, interval=interval,
148:        progress=False, auto_adjust=True,
149:    )
```

The function does NOT acquire `portfolio.shared_state.yfinance_lock`,
which `data_collector.py:277` explicitly documents as required:

> H11/DC-R3-4: yfinance is not thread-safe; serialize its calls with a shared lock.

`data_collector.yfinance_klines` and `fear_greed.get_stock_fear_greed`
acquire the lock; `price_source._fetch_yfinance` does not. Since
`metals_cross_assets._yf_download` (and many other callers) route through
`fetch_klines` → `_fetch_yfinance`, those calls race against the lock-
holding callers and can corrupt yfinance's internal caches. Symptoms:
`MultiIndex` columns when single-ticker (the file's own line 150-151
comments document the workaround for this exact symptom).

**Fix:** wrap the `yf.download` call with `with yfinance_lock:`.

---

### P2-4. `_fetch_yfinance` ignores caller's `limit` for `tail` only

**File:** `portfolio/price_source.py:160`
**Confidence:** 80

```python
160:    return df.tail(limit) if limit else df
```

`yf.download(period=p)` returns a fixed-size frame based on `period`, not
`limit`. So callers that pass `limit=100, period="5d"` get whatever 5
days of `interval=` bars produces, then `.tail(100)`. If the caller really
wanted 100 bars (the `data_collector.TIMEFRAMES` spec is `(label,
interval, num_candles, ttl)`), and the period yields fewer than 100, the
result is silently short. This causes the `compute_indicators` 20-bar
minimum requirement to silently fail on weekends/holidays for
`get_copper_intraday`, `get_oil_intraday`, etc. (the metals_cross_assets
intraday functions check `len(close) < 4`, so 4 bars is OK; but tighter
indicators get NaN).

**Fix:** if `limit > len(df)`, log a WARNING so we know upstream callers
asked for more data than available.

---

### P2-5. `alpha_vantage` `_check_budget` is racy with `_daily_budget_used` increment

**File:** `portfolio/alpha_vantage.py:157–168`, `279–283`
**Confidence:** 80

```python
157:def _check_budget():
158:    """Check and reset daily budget counter. Returns current usage count.
159:
160:    BUG-108: Protected by _cache_lock for thread safety.
161:    """
162:    global _daily_budget_used, _budget_reset_date
163:    today = datetime.now(UTC).strftime("%Y-%m-%d")
164:    with _cache_lock:
165:        if _budget_reset_date != today:
166:            _daily_budget_used = 0
167:            _budget_reset_date = today
168:        return _daily_budget_used
```

```python
279:            with _cache_lock:
280:                _cache[ticker] = normalized
281:                _daily_budget_used += 1  # BUG-108: increment under lock
282:            _cb.record_success()
283:            success_count += 1
```

The check (line 231: `budget_used = _check_budget()`) and the increment
(line 281) are not atomic. A second thread can pass the
`budget_used >= daily_budget` gate while the first is in the middle of
the API call. Free tier is 25/day and the Alpha Vantage `Note:` rate-limit
response is documented at line 150-152; if multiple workers race past the
check, the actual call burns the budget regardless of the local counter.

In practice `refresh_fundamentals_batch` is called once per day from a
single thread, so the race is rarely hit. But `earnings_calendar._fetch_
earnings_alpha_vantage` also uses the same AV limiter (line 48) without
touching `_daily_budget_used` at all (acknowledged in the docstring at
line 49-52: *"earnings calls bypass alpha_vantage.py's _daily_budget_used
counter"*). When earnings runs concurrently with the batch refresh, the
budget tracker undercounts.

**Fix:** export an `increment_budget()` helper from `alpha_vantage.py` and
have `earnings_calendar._fetch_earnings_alpha_vantage` call it post-fetch.

---

### P2-6. `crypto_macro_data._load_ratio_history` / `_load_netflow_history` use raw `open()`

**File:** `portfolio/crypto_macro_data.py:275–286`, `397–408`
**Confidence:** 78

```python
275:        with open(RATIO_HISTORY_FILE, encoding="utf-8") as f:
276:            for line in f:
277:                line = line.strip()
278:                if not line:
279:                    continue
280:                try:
281:                    entry = json.loads(line)
282:                    if entry.get("ts", 0) >= cutoff:
283:                        entries.append(entry)
284:                except (json.JSONDecodeError, KeyError):
285:                    continue
```

Convention rule (CLAUDE.md): *"Atomic I/O only. Use file_utils.atomic_write_json(),
load_json(), atomic_append_jsonl(). Never raw json.loads(open(...).read())."*

The write paths use `atomic_append_jsonl` correctly (lines 308, 424), but
the read paths use raw `open()` + `json.loads(line)`. JSONL line reads are
less corruption-prone than full-JSON reads (a single bad line is skipped,
not the whole file), so the practical impact is low. Convention requires
the wrapped helpers anyway.

**Fix:** add an `iter_jsonl()` helper to `file_utils.py` and use it.

---

## P3 — Notes

### P3-1. Magic URLs scattered across modules

`https://api.alternative.me/fng/` (`fear_greed.py:96`),
`https://min-api.cryptocompare.com/data/v2/news/?lang=EN` (`sentiment.py:70`),
`https://newsapi.org/v2/everything` (`sentiment.py:204`),
`https://www.alphavantage.co/query` (`alpha_vantage.py:26`,
`earnings_calendar.py:54`),
`https://api.frankfurter.app/latest` (`fx_rates.py:37`),
`https://bitcoin-data.com` (`onchain_data.py:73`),
`https://www.deribit.com/api/v2/public` (`crypto_macro_data.py:36`),
`https://www.reddit.com/r/...` (`social_sentiment.py:31, 62`).

No central registry. Switching a host (e.g., when Frankfurter goes down,
or moving Reddit to a paid mirror) requires touching every file.

**Fix:** add `portfolio/api_utils.py` constants (it already houses
`BINANCE_BASE` / `ALPACA_BASE`).

---

### P3-2. Hardcoded `User-Agent: "Mozilla/5.0"` in sentiment + earnings paths

`sentiment.py:121, 207`. Hardcoding a browser UA on a script makes it easy
for Yahoo/CryptoCompare/NewsAPI to block the entire fleet by UA pattern.
`social_sentiment.py:11` uses the better convention
`"finance-analyzer/1.0 (portfolio intelligence bot)"`.

**Fix:** unify on a single `USER_AGENT` constant.

---

## Verifications performed

- **Binance `10m` interval grep** (P1 from brief): `Grep "10m"` and
  `interval=.10m.` across `portfolio/` — zero matches. Convention adhered to.
- **Funding-rate sign** (`portfolio/funding_rate.py:44-49`): positive
  funding → longs pay shorts → `SELL` (contrarian). Convention correct.
- **Atomic cache writes**: `onchain_data._save_onchain_cache` (line 102),
  `alpha_vantage._save_persistent_cache` (line 55), `fear_greed.update_fear_streak`
  (line 77), `microstructure_state.persist_state` (line 213) all use
  `atomic_write_json`. P1 convention satisfied for caches.
- **Timeouts on `requests.get`**: every direct `requests.get` call in
  the in-scope files passes `timeout=`. The only modules using bare
  `requests` without `fetch_with_retry` are `social_sentiment.py` (P1-2);
  timeouts are set there too.
- **Microstructure look-ahead**: `microstructure_state.accumulate_snapshot`
  (line 55-75) appends `depth` as-is with a `ts` from the snapshot itself;
  `get_rolling_ofi` reads only past snapshots; `get_ofi_zscore` explicitly
  excludes current value from history (line 122-124 comment block).
  No tomorrow-data leak.
- **Fear & Greed maintenance-window guard** (`fear_greed.py:104-112`):
  `alternative.me` empty-`{"data": []}` is correctly handled (cited:
  P1-13 in the 05-01 review, P1-2 in the 04-29 review).
- **Funding rate FAPI schema drift guard** (`funding_rate.py:31-39`):
  `.get()` + None-return chain prevents `KeyError` crash propagation.

---

## Files reviewed (clean, no issues found)

- `portfolio/futures_data.py` — clean. Uses `fetch_json` + `_cached`
  + `_binance_limiter` correctly.
- `portfolio/metals_orderbook.py` — clean. Same pattern.
- `portfolio/microstructure_state.py` — clean. Look-ahead correctly
  prevented per the inline comments.
- `portfolio/news_keywords.py` — clean. All regex patterns use
  `re.IGNORECASE`; ticker synonym patterns memoized.

