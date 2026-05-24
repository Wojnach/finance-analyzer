# Adversarial Code Review — data-external subsystem

Reviewer: opus 4.7 (1M) /fgl audit, empty-baseline
Date: 2026-05-24
Worktree: `Q:\finance-analyzer\finance-analyzer-reviews\2026-05-24` (branch `review/fgl-2026-05-24`)
Scope: 20 modules (Binance, Alpaca, yfinance, Alpha Vantage, NewsAPI, FRED, BGeometrics, etc.)

---

## Top 5 (most impactful, sorted by blast radius × silence)

1. **`data/crypto_data.py:223-231`** — `get_onchain_summary()` reads keys (`zone`, `bias`, `summary`) that the underlying `interpret_onchain()` **never emits** (it returns `mvrv_zone` / `sopr_zone` / `nupl_zone` / `netflow_signal`). Result: every on-chain summary returned to the crypto/metals loop is permanently `{"zone": "neutral", "bias": "neutral", "summary": ""}` regardless of real on-chain state. Silent — no exception, no log. CLAUDE.md rule #3 violation (downstream consumers think they're getting on-chain context).
2. **`data/crypto_data.py:75-82, 184-198`** — `get_fear_greed()` masks empty Binance/alternative-me responses as `value=50, classification=Neutral` (silently fabricated baseline); `compute_mstr_btc_nav()` uses hard-coded MSTR_BTC_HOLDINGS=499,096 and SHARES=229M from "early 2026". As of 2026-05-24 MSTR has continued to accumulate BTC — NAV premium calc is structurally biased low.
3. **`portfolio/forecast_accuracy.py:282, 297, 303-304`** — `datetime.fromisoformat(ts_str)` produces a naive datetime when `ts` lacks tz info. Then compared to `datetime.now(UTC)` (aware) → raises `TypeError: can't compare offset-naive and offset-aware datetimes`. Backfill silently aborts mid-loop. Same naive/aware mismatch exists in `local_llm_report.py:48-51` (naive `ts < cutoff` comparison).
4. **`portfolio/forecast_signal.py:97`** — `except (ImportError, Exception)` — `Exception` swallows `ImportError`, but more importantly it swallows **every** exception (OOM, CUDA driver, model file corruption, etc.) and logs them all as "Chronos-2 not available". The fallback to v1 may then crash on the same root cause but with no signal of what actually broke.
5. **`portfolio/data_collector.py:168-204` (`fetch_vix`)** — Calls `yf.Ticker(...).history()` **without acquiring `_yfinance_lock`**. Every other yfinance entry point in the codebase (`fear_greed.get_stock_fear_greed`, `data_collector._fetch_one_timeframe` market-closed path, `golddigger/data_provider`) acquires the lock. Concurrent VIX fetch with sentiment yfinance call from another worker → known yfinance non-thread-safety segfault / corrupted SSL session.

---

## Critical (P0/P1)

### P0-1 — `data/crypto_data.py:228-231` | P0 | API contract mismatch — silent
**Issue**: `interpret_onchain(raw)` returns `{mvrv_zone, sopr_zone, nupl_zone, netflow_signal}` (see `portfolio/onchain_data.py:300-345`). `get_onchain_summary()` reads non-existent keys `zone`, `bias`, `summary` and falls through to default `"neutral"` / `""`. The metals/crypto loop has been receiving permanently-neutral on-chain signal since this code was written.
**Fix**: Read the actual keys:
```python
result = {
    "mvrv": raw.get("mvrv"),
    "sopr": raw.get("sopr"),
    "nupl": raw.get("nupl"),
    "mvrv_zone": interpretation.get("mvrv_zone"),
    "sopr_zone": interpretation.get("sopr_zone"),
    "nupl_zone": interpretation.get("nupl_zone"),
    "netflow_signal": interpretation.get("netflow_signal"),
}
```

### P0-2 — `data/crypto_data.py:73-85` | P0 | Fabricated fallback data
**Issue**: Lines 75 `r.json().get("data", [{}])[0]` — if API returns `{"data": []}` (maintenance window — see fix in `fear_greed.py:104-108`), this code creates `{}` then defaults to `value=50, classification="Neutral"`. The metals loop then treats "50 Neutral" as a real reading. CLAUDE.md rule #3: live prices first.
**Fix**: Mirror `portfolio/fear_greed.py:104-108` guards — return `None` on empty.

### P0-3 — `data/crypto_data.py:184-203` | P0 | Hard-coded MSTR holdings staleness
**Issue**: `MSTR_BTC_HOLDINGS = 499_096` and `MSTR_SHARES_OUTSTANDING = 229_000_000` — comment says "as of early 2026". Today is 2026-05-24. MSTR's NAV premium calc is structurally biased because BTC holdings have moved (MSTR announces purchases monthly). User trades MSTR as leveraged BTC proxy per memory — wrong NAV will steer real decisions.
**Fix**: Either source these from Avanza/Alpaca/SEC filings on a refresh, or pin them to a `_FETCHED_AT` and warn loudly if >30 days stale.

### P0-4 — `portfolio/forecast_accuracy.py:282-304` | P0 | Naive/aware datetime comparison crash
**Issue**:
```python
entry_time = datetime.fromisoformat(ts_str)        # ← naive if ts has no tz
...
now = datetime.now(UTC)                            # aware
...
horizon_time = entry_time + timedelta(hours=hours) # still naive
if now < horizon_time:                             # ← TypeError on aware vs naive
```
Any `forecast_predictions.jsonl` row written without `+00:00` suffix kills the backfill loop on contact. The `sentiment_shadow_backfill.py:211-213` peer correctly handles this with `if entry_time.tzinfo is None: entry_time = entry_time.replace(tzinfo=UTC)`. Apply same.
**Fix**:
```python
entry_time = datetime.fromisoformat(ts_str)
if entry_time.tzinfo is None:
    entry_time = entry_time.replace(tzinfo=UTC)
```
Same fix needed in `local_llm_report.py:48-51`.

### P0-5 — `portfolio/forecast_signal.py:97` | P0 | Exception swallowing hides real failures
**Issue**: `except (ImportError, Exception) as e:` — `Exception` matches everything (CUDA OOM, GPU driver crash, model file IO error, NaN in tensor, etc.). All get logged as "Chronos-2 not available" and the code falls through to v1, which then fails for the same root cause silently because there's no targeted recovery. Operators see "Chronos-2 not available" forever without knowing whether it's a missing package (one-time install fix) or a transient CUDA OOM (recoverable).
**Fix**: Catch only `ImportError`. Let other exceptions propagate so the outer `forecast_chronos()` `except Exception as e: logger.warning("Chronos forecast failed for %s: %s", ...)` reports the real cause once.

### P0-6 — `portfolio/data_collector.py:168-204` | P0 | yfinance thread-safety violation
**Issue**: `fetch_vix()` calls `yf.Ticker("^VIX").history(period="5d")` **without holding `_yfinance_lock`** (defined in `shared_state.py:286` precisely for this reason). All other yfinance callers (`fear_greed.get_stock_fear_greed:133`, `data_collector._fetch_one_timeframe:291`, etc.) acquire it. Concurrent calls during a busy cycle (8-worker pool) → yfinance's shared session state corrupts, manifests as random `JSONDecodeError`, empty DataFrame, or SSL retries that occupy the full 60s loop budget.
**Fix**:
```python
from portfolio.shared_state import yfinance_lock as _yfinance_lock
def fetch_vix():
    try:
        import yfinance as yf
        with _yfinance_lock:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="5d")
        ...
```

### P1-7 — `portfolio/earnings_calendar.py:64-65` | P1 | Bypasses fetch_json error handling
**Issue**: `data = r.json()` directly after `fetch_with_retry` — no `r.raise_for_status()`, no check for AV's `{"Note": "..."}` or `{"Information": "..."}` rate-limit responses. If AV returns 200 with `{"Information": "thank you for using..."}` (the newer daily-limit response — `alpha_vantage.py:150` only checks `"Note"`), `data.get("quarterlyEarnings", [])` returns `[]` cleanly and the function returns None — earnings gate silently disabled for the day. Use `fetch_json` instead (which handles raise_for_status), and add the `"Information"` key check shared with `alpha_vantage._fetch_overview`.
**Fix**: Replace with `fetch_json(...)`. Then check both `"Note"` AND `"Information"` keys.

### P1-8 — `portfolio/alpha_vantage.py:150` | P1 | Misses AV's newer rate-limit key
**Issue**: AV's daily-limit response uses `"Information"` as of late 2024 / 2025. This code only checks `"Note"`. When the 25/day budget is exhausted, AV returns 200 OK with `{"Information": "We have detected..."}` — passes the `"Note"` check, then `_normalize_overview` sees no `"Symbol"` and returns None, marked as failure. Circuit breaker accumulates failures on rate-limit responses (line 274) → spurious cooldown.
**Fix**:
```python
if isinstance(data, dict) and ("Note" in data or "Information" in data):
    logger.warning("Alpha Vantage rate limit hit: %s",
                   (data.get("Note") or data.get("Information"))[:100])
    return None
```

### P1-9 — `portfolio/social_sentiment.py:32, 65` | P1 | Bypasses http_retry + prints errors
**Issue**: Raw `requests.get(...)` — no retry, no circuit breaker, no Telegram-token redaction in URLs (not needed here, but pattern-broken). Errors go to `print(f"...")` (stdout), not `logger.warning`. CLAUDE.md rule #2 ("Search before writing — reuse `http_retry`"). When Reddit rate-limits (HTTP 429 with no-Retry-After), this code will hammer them and get banned.
**Fix**: Migrate to `fetch_json(url, headers=..., timeout=10, label=f"reddit:{sub}")`. Replace `print()` with `logger.warning()`.

### P1-10 — `portfolio/crypto_macro_data.py:381-382` | P1 | Truthiness bug rounds 0 to None
**Issue**:
```python
"sum_7d": round(sum_7d, 1) if sum_7d else None,
"sum_14d": round(sum_14d, 1) if sum_14d else None,
```
`sum_7d` can legitimately be `0.0` (perfectly balanced inflow/outflow) — falsy in Python. The condition skips the round and stores `None`, which downstream readers interpret as "no data" rather than "exactly zero netflow". Either `is not None` or directly compare.
**Fix**:
```python
"sum_7d": round(sum_7d, 1) if sum_7d is not None else None,
```

### P1-11 — `portfolio/crypto_macro_data.py:202-225` | P1 | CLAUDE.md rule #3 violation (live prices)
**Issue**: `compute_gold_btc_ratio()` reads `agent_summary_compact.json` for BTC and Gold prices. CLAUDE.md rule #3 ("Live prices first. Never base analysis on cached/precomputed data."). Worse, the cache is intentionally rolled into a 1h `_cached("gold_btc_ratio", RATIO_TTL, ...)` (line 450) on top of an already-stale source — staleness can compound to many hours. Trading signals derived from this ratio can be wildly out of sync with reality.
**Fix**: Route through `portfolio.price_source.fetch_klines("BTC-USD", "1h", 2)` and `fetch_klines("XAU-USD", "1h", 2)` for live prices; preserve the JSONL ratio history (already correct) but compute the ratio from live data.

### P1-12 — `portfolio/data_collector.py:296-300` | P1 | "insufficient data" error swallows real failures
**Issue**: When `compute_indicators(df)` returns None, code logs `insufficient data ({rows} rows)`. But `compute_indicators` can return None for **many** reasons besides "not enough bars" — including a malformed dataframe, NaN-saturated column, or specific indicator failure. The label is misleading. Operators reading logs assume "API returned too few bars, harmless"; the actual cause might be a Binance schema change that broke `astype(float)` for one column.
**Fix**: Either inspect `compute_indicators`' failure mode or change the label to `"compute_indicators returned None ({rows} rows)"` — make the path clearly distinguishable from "API returned 5 bars" vs "API returned 100 bars but they're broken".

### P1-13 — `portfolio/seasonality.py:60` | P1 | `grouped.loc[hour]` returns Series — silent default on multi-row
**Issue**: When `df.groupby("hour")` produces rows where one hour has only 1 observation, `grouped.loc[hour]` returns a Series. But if `hour` duplicates somehow (shouldn't normally happen), `.loc[hour]` returns a DataFrame and `float(row["mean_return"])` fails with TypeError. The function catches nothing and propagates. Wrap defensively or use `.iloc` after explicit sort.
**Fix (minor)**: Already low-impact since groupby keys are unique; add a defensive `if isinstance(row, pd.DataFrame): row = row.iloc[0]`.

### P1-14 — `portfolio/fear_greed.py:152-154` | P1 | "Belt-and-suspenders" only works for 2D
**Issue**: The defensive comment correctly notes that yfinance may return a DataFrame for `h["Close"]`, but the squeeze logic only handles `ndim>1`. If yfinance returns a 1D Series (the normal case), `.iloc[-1]` is fine. But if yfinance returns a scalar (rare degenerate case for `^VIX` with 1 row), `float(close_series.iloc[-1])` raises AttributeError. Add a final `try/except` shield with logger.warning instead of letting the exception propagate to the worker.

### P1-15 — `portfolio/data_collector.py:230-231` | P1 | yfinance empty-DataFrame → ValueError → caught → silent skip
**Issue**: `yfinance_klines` raises `ValueError("No yfinance data for ...")` on empty result. This is caught by `_fetch_one_timeframe` and stored as `{"error": str(e)}`. The error is logged at DEBUG only (line 298). When MSTR's Alpaca path fails and falls through to yfinance, and yfinance also fails (e.g. invalid ticker, rate-limited), **the operator sees nothing**. CLAUDE.md "system reliability is #1" — silent yfinance failures during US market hours have been a recurring source of multi-hour data gaps.
**Fix**: Log at WARNING when yfinance is the primary fallback (i.e., when market is closed for a stock ticker).

### P1-16 — `portfolio/alpha_vantage.py:30-32` | P1 | Module-level state not lock-protected on init
**Issue**: `_cache = {}` and `_daily_budget_used = 0` are module-level. `load_persistent_cache()` (line 36) is called from main.py startup. If multiple worker threads import this module simultaneously (during the rare PF-DataLoop + dashboard race), the `_cache` reference may be replaced inside `load_persistent_cache()` without the `_cache_lock`. Actually checking: line 44 does `with _cache_lock: _cache = data`. The lock IS held — but Python module-level `_cache = data` rebinds the name, which is atomic. Other readers using `with _cache_lock: return _cache.get(ticker)` will see the new dict. **Actually OK.** No bug — marking as verified-safe.

---

## Important (P2/P3)

### P2-17 — `portfolio/funding_rate.py:44-49` | P2 | Hardcoded thresholds with no documentation source
**Issue**: Thresholds `> 0.0003` (SELL) and `< -0.0001` (BUY) are asymmetric and not documented (comment says "Normal funding ~0.01% (0.0001)"). Bias makes BUY trigger 3x more easily than SELL. May or may not be intentional. Either document the rationale or symmetrize.

### P2-18 — `portfolio/futures_data.py:60-64` | P2 | Bare `data["openInterest"]` after `None` check
**Issue**: `if data is None: return None` then immediately `float(data["openInterest"])` — if Binance changes the schema or returns `{}` (rare error path), KeyError propagates uncaught into the worker. Mirror the defensive pattern from `funding_rate.py:31-39`.

### P2-19 — `portfolio/seasonality_updater.py:53-69` | P2 | Bypasses circuit breaker
**Issue**: Calls `fetch_json(BINANCE_FAPI_BASE/klines, ...)` directly instead of `binance_fapi_klines()` (which uses circuit breaker). If Binance FAPI is degraded and `binance_fapi_cb` is OPEN, this fetcher still pounds the API. Use `binance_fapi_klines(symbol, "1h", limit)` and convert to DataFrame from there.

### P2-20 — `portfolio/onchain_data.py:282` | P2 | "stale cache without token" path violates rule #3
**Issue**: When `_load_config_token()` returns None, the function returns the persistent cache (2x normal TTL — 24h instead of 12h). Operator with a missing/revoked BGeometrics token gets day-old on-chain data silently. Should at minimum log WARNING that we're serving 2x-stale data; better: log ERROR and let the on-chain voter abstain.

### P2-21 — `portfolio/sentiment.py:329-338` | P2 | 120s subprocess timeout in fallback path
**Issue**: When in-process BERT fails, the legacy subprocess fallback has a 120s timeout. The ticker pool timeout is 500s (per comment elsewhere). A wedged sentiment subprocess can hang the worker for 2 minutes, multiplied across multiple tickers in the same cycle. If GPU lock is held by ministral/qwen3, the subprocess may sit waiting indefinitely. Drop subprocess timeout to 30s; in-process fallback is sufficient.

### P2-22 — `portfolio/sentiment.py:32-42` | P2 | Hardcoded paths break dev environments
**Issue**: Linux paths are `/home/deck/models/...`. The repo has both WSL and Windows paths elsewhere. New contributors on macOS or generic Linux see import-like failures with no helpful message. Read paths from config (`config.local_models.bert_paths`) with these as fallback.

### P2-23 — `portfolio/fx_rates.py:54` | P2 | Bare `except Exception` swallows specific errors
**Issue**: `except Exception as e: logger.warning("FX rate fetch failed: %s", e)` then falls to cached/fallback. Hides JSONDecodeError, KeyError on `r.json()["rates"]["SEK"]`, etc. Won't crash the loop but operators see "FX rate fetch failed: 'SEK'" with no context to debug. Log `exc_info=True` so the traceback is captured at WARNING level.

### P2-24 — `portfolio/crypto_macro_data.py:108` | P2 | `now = datetime.date.today()` is system-local
**Issue**: Uses `date.today()` which is *system-local*. The host is CET; Deribit expiries are UTC. Around midnight CET/UTC switch this can pick the wrong nearest expiry by 1 day, particularly for short-dated weeklies. Use `datetime.now(UTC).date()`.

### P2-25 — `portfolio/forecast_accuracy.py:357-369` | P2 | Linear scan of snapshot file on every backfill row
**Issue**: `_lookup_price_at_time` iterates entire `price_snapshots_hourly.jsonl` for **every** prediction entry — quadratic cost. For N predictions and M snapshots, O(N×M). Once the file grows past a few thousand rows this becomes the dominant cost of `backfill_forecast_outcomes()`. Build an in-memory index per (ticker, hour bucket) on first call.

### P3-26 — `portfolio/futures_data.py:63` | P3 | Field name typo / partial implementation
**Issue**: `get_open_interest` returns `{oi, symbol, time}` but the docstring says `{oi, oi_usdt, symbol, time}` — `oi_usdt` is missing in the returned dict. Either rip from docstring or compute from `mark_price` if available.

### P3-27 — `portfolio/sentiment.py:233` | P3 | Truthy check on result list
**Issue**: `if result:` — empty list is falsy. If NewsAPI returns 200 with `articles=[]` (legitimate, no news), `newsapi_track_call()` is NOT incremented. That's actually a feature per the comment (H9/DC-R3-2). But the comment claims "only count against budget when we actually got data" — silently the rate limit isn't being respected if a ticker consistently returns empty arrays. Subtle quota leak.

### P3-28 — `portfolio/data_collector.py:262` | P3 | Implicit string membership check
**Issue**: `_ss._current_market_state in ("closed", "weekend", "holiday")` — does not handle case-sensitivity issues. If market_state is set as "Closed" anywhere (a future refactor risk), this silently routes to Alpaca which then fails. Defensive `.lower()` on read.

### P3-29 — `data/crypto_data.py:247-259` | P3 | `is_us_market_hours` ignores holidays
**Issue**: Returns True for any weekday 09:30-16:00 NY time — including July 4, Thanksgiving, MLK Day, etc. Stock signals get derived from holiday data. Compare with `econ_dates.py` pattern (hard-coded list).

### P3-30 — `portfolio/alpha_vantage.py:266-269` | P3 | Circuit breaker break loop premature
**Issue**: When `_cb.allow_request()` returns False, the loop breaks immediately — but `_alpha_vantage_limiter.wait()` may have already paid the rate-limit cost for this iteration. Move the CB check above the limiter wait.

### P3-31 — `portfolio/fomc_dates.py` and `portfolio/econ_dates.py` | P3 | Hard-coded calendars expire
**Issue**: Calendars only cover 2026-2027. By Q4 2027 the system will silently degrade — `next_event()` returns None for queries after 2027-12-08. Add a startup assertion that the calendar covers `now()+lookahead_hours`.

### P3-32 — `portfolio/data_collector.py:323` | P3 | ThreadPoolExecutor not closed on TimeoutError
**Issue**: `with ThreadPoolExecutor(...)` context manager handles shutdown, but inside the timeout `except` block we call `f.cancel()` on futures — `cancel()` only works on **not-yet-started** futures (running ones are uncancellable in stdlib executors). The pool will block in `__exit__` until those running futures finish (which is the original timeout cause). The `with` may sit beyond `_TF_POOL_TIMEOUT`. Consider `shutdown(wait=False, cancel_futures=True)` on the timeout path.

### P3-33 — `portfolio/fear_greed.py:31` | P3 | "extreme_fear" streak resets on missing data
**Issue**: `get_sustained_fear_days()` returns 0 if `load_json` raises. If the streak file is missing/corrupt during a brief disk error, the signal engine sees `sustained=0` and contrarian gating disengages — exactly when high contrarian conviction is most valuable. Log at WARNING when this branch fires.

### P3-34 — `portfolio/forecast_signal.py:196` | P3 | Wall-clock timestamp doesn't match price candle timing
**Issue**: `timestamps = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="h")` — generates hourly timestamps ending NOW. But `prices` came from Binance klines whose last bar's `close_time` may be 0-59 minutes ago. Chronos-2 receives prices with synthetic timestamps that don't reflect when the data was actually observed. Practical impact small (Chronos doesn't strictly need calendar alignment) but documentation lies.

### P3-35 — `portfolio/local_llm_report.py:48-51` | P3 | Naive datetime same-class as P0-4
**Issue**: Same naive/aware issue as forecast_accuracy.py but in the report builder. If `entry["ts"]` lacks tz, the `ts < cutoff` comparison raises TypeError. Caught at line 49 but as `(TypeError, ValueError)` — wait, it's `(TypeError, ValueError)`. OK, caught. But silently dropping every row with no-tz timestamp means the report shows zero recent predictions. Add WARNING.

### P3-36 — `portfolio/futures_data.py:88-91` | P3 | `sumOpenInterestValue` missing in returned dict for `get_open_interest`
**Issue**: `get_open_interest_history` returns `{oi, oi_usdt, timestamp}` with `oi_usdt = sumOpenInterestValue`. But `get_open_interest` (the singular version) doesn't return oi_usdt (see P3-26). Consumers using the singular path get a different schema than the history path. Inconsistent.

---

## Notes on prior P0/P1s (2026-05-19) — verification

1. **Binance error responses returned as garbage candles (`r.json()` → DataFrame on dict)**: `data_collector.py:88-93` now has `if not data: raise ConnectionError` guard. Dict error responses pass the truthy check, then `pd.DataFrame(data, columns=_BINANCE_KLINE_COLS)` raises ValueError on scalar values → caught and CB-recorded as failure. **Verified fixed (mitigated, not eliminated).**
2. **yfinance error handling silently returns empty DataFrame**: `yfinance_klines` raises `ValueError` on empty (line 230). Caught at `_fetch_one_timeframe` (line 312) and logged at DEBUG. **Partially fixed** — see P1-15 above; should be WARNING when yfinance is the primary fallback.
3. **Binance 10m interval doesn't exist — use 5m**: `TIMEFRAMES` and `STOCK_TIMEFRAMES` (data_collector.py:44-62) use `15m`, `1h`, `4h`, `1d`, `3d`, `1w`, `1M` — no `10m`. **Verified fixed.** `_binance_interval` map in `price_source.py:101-106` maps `60m→1h, 90m→1h, 120m→2h` correctly.

---

## Files reviewed (clean — no P0/P1)

- `portfolio/fomc_dates.py` — date constants only, no I/O
- `portfolio/econ_dates.py` — handles tz correctly via UTC anchor at 14:00 (P3-31 calendar staleness only)
- `portfolio/alert_budget.py` — clean thread-safe token bucket
- `portfolio/news_keywords.py` — pure functions, no I/O
- `portfolio/bert_sentiment.py` — well-designed with meta-tensor recovery, per-model locks
- `portfolio/price_source.py` — clean router with fallback chain
- `portfolio/sentiment_shadow_backfill.py` — correctly handles naive/aware datetimes (line 212)

---

## Recommendations (in priority order)

1. **Fix P0-1 immediately** — the on-chain summary bug is silent and downstream signals depend on it. One-line dict-key fix.
2. **Fix P0-3** — refresh MSTR holdings/shares constants from Avanza/Alpaca or pin with staleness warning.
3. **Fix P0-4 and P3-35 together** — apply the `tzinfo is None → UTC` guard pattern from `sentiment_shadow_backfill.py:212` everywhere `datetime.fromisoformat` reads JSONL ts.
4. **Fix P0-6** — add yfinance_lock to `fetch_vix()`. Trivial.
5. **Fix P0-5** — narrow the `except` in forecast_signal.py to ImportError only.
6. **Fix P1-7, P1-8** — coordinate the AV "Note"/"Information" key fix across earnings_calendar.py and alpha_vantage.py.
7. **Migrate `data/crypto_data.py` off raw `requests.get`** to `fetch_json` — three call sites at lines 73, 102, 148.
8. **P2 follow-ups** — circuit-breaker hygiene in seasonality_updater.py and subprocess timeout reduction in sentiment.py.

External APIs are unreliable — the patterns of `fetch_json` + circuit breaker + rate limiter that the main `data_collector.py` and `alpha_vantage.py` follow are sound. The biggest risk vectors are **silent semantic bugs** (P0-1 wrong dict keys, P0-3 stale constants) that produce plausible-looking-but-wrong outputs, and **silent quota / staleness leaks** (P1-8 missing rate-limit key, P2-20 stale on-chain cache without token). Less common is outright crash — the codebase has matured well around the prior P0s.
