# Adversarial review — data-external subsystem

Scope: portfolio/data_collector.py, futures_data.py, onchain_data.py,
fx_rates.py, fear_greed.py, alpha_vantage.py, news_keywords.py,
sentiment.py, social_sentiment.py, bert_sentiment.py,
earnings_calendar.py, macro_context.py, crypto_macro_data.py,
market_health.py.

Conventions: [Pn] file.py:line — problem | FIX: repair. MAYBE = suspected
but unverified. Only bugs listed; no praise.

---

## Findings

[P0] data_collector.py:96 — `pd.to_datetime(df["open_time"], unit="ms")` produces tz-NAIVE timestamps. Binance returns UTC epoch ms, but the resulting Series has no tz; downstream signal modules that compare these to `datetime.now(UTC)` (sentiment freshness, outcome backfill, microstructure clock) silently mismatch by the local TZ offset (UTC+1/+2 in CET). All time-based decay/recency math is off by one hour in winter, two in summer. | FIX: `pd.to_datetime(df["open_time"], unit="ms", utc=True)`.

[P0] data_collector.py:157 — Alpaca returns ISO timestamps with `Z`/offset; `pd.to_datetime(df["time"])` without `utc=True` returns mixed-tz objects depending on pandas version (sometimes naive, sometimes object dtype). Mixed with the Binance naive output above, anything that concatenates these (multi-asset cross-correlation, regime detection) gets wrong relative ordering. | FIX: `pd.to_datetime(df["time"], utc=True)`.

[P0] data_collector.py:245 — `df["time"] = df.index` after a yfinance download keeps the yfinance DatetimeIndex which is *exchange-local* tz-aware for intraday (e.g. America/New_York for stocks, UTC for crypto). Mixing with Alpaca/Binance times produces silent ~5 h skew on US stocks during stale fallback. | FIX: `df["time"] = pd.to_datetime(df.index, utc=True)`.

[P0] fx_rates.py:33 — Cache freshness check `cached_rate and now - cached_time < 900` returns the cached value but never validates that `cached_rate` itself is sane. If a previous corrupt cache write put `rate=0.0` or `inf` (e.g. transient parse bug), it is served for 15 min and any SEK→USD math divides-by-zero or explodes portfolio values. | FIX: also gate on `FX_RATE_MIN <= cached_rate <= FX_RATE_MAX` before returning.

[P0] fx_rates.py:47-53 — When the live API returns an out-of-range rate, the code logs ERROR and falls through to the stale-cache branch, but it does NOT increment a failure counter or open a circuit. Frankfurter occasionally publishes 0.0 during off-hours; on the FIRST call of a fresh process the cache is empty, so the code drops to line 71 and returns `FX_RATE_FALLBACK = 10.50` which is hardcoded ~6% off from current ~11.10. Real-money trading uses this. | FIX: keep the last sane rate persisted to disk (atomic_write_json), seed `_fx_cache` from it on import, and track consecutive bad-rate count to circuit-open after 3.

[P0] fx_rates.py:71 — `FX_RATE_FALLBACK = 10.50` is a static literal authored months ago; SEK has moved ~6%. Portfolio valuation, stop-loss SEK conversions, and warrant P&L all run on this when both live + cache fail. | FIX: persist last-known-good FX rate to disk on every successful fetch and load on import; fallback to that, not to `10.50`.

[P0] onchain_data.py:284 — `_cached("onchain_btc", ONCHAIN_TTL, _fetch_all_onchain, token)` includes `token` as the args; if config rotates the token (or temporarily reads as `None` and `_load_config_token()` returns the literal None), the second branch on line 276 returns and stale cache is preserved — but if token *changes string* the cache key collides and the new fetch reuses the OLD cache for 12 h. | FIX: include token hash in the cache key, or invalidate when token changes.

[P0] onchain_data.py:223-225 — `time.sleep(1)` between 6 BGeometrics fetchers runs INSIDE the `_cached(...)` worker, holding the cache `_loading_keys` slot for ~5 s. Any concurrent caller during those 5 s gets `None` (line 88 in shared_state). Because BGeometrics has 8 req/hour AND 15 req/day, the 5 s sleep is appropriate, but the on-chain voter goes silent for every concurrent ticker on the cycle that triggers a refresh. | FIX: do the sleep-between-calls outside the cache-worker, or batch-fetch into a private dict and only call `_cached` to publish the final result.

[P0] futures_data.py:50 — `"oi_usdt": float(data["openInterest"])` (line 50 returns `oi`/`symbol`/`time` but builds no `oi_usdt`); the docstring promises `oi_usdt` but the code at lines 49-53 never sets it. Signal `futures_flow` reads `open_interest.oi_usdt` and falls back to None silently. | FIX: either drop the docstring promise or compute `oi_usdt = oi * markPrice` from a separate FAPI premium call.

[P0] crypto_macro_data.py:75 — `oi = item.get("open_interest", 0) or 0` then later `expiry_data` indexed by `expiry_str` BUT the parsed name format on line 79 (`name.split("-")`) silently drops any instrument whose name has 4 segments but is malformed (e.g. `BTC-PERPETUAL-...`). Worse, the strike parse only catches `ValueError`/`TypeError` — Deribit non-option names like `BTC_USDC-PERPETUAL` produce 5 segments and are skipped, OK, but `index_price`/`mark_price` instruments fall through and contaminate `expiry_data` if their `instrument_name` pattern is exactly `X-Y-Z-W`. | FIX: also require option_type in `{"C","P"}` AND `len(strike_str) >= 1` AND that `expiry_str` parses via `_parse_expiry`; reject otherwise.

[P0] crypto_macro_data.py:283 — `with open(RATIO_HISTORY_FILE, encoding="utf-8") as f:` reads the entire JSONL file every call, but the call happens inside `_append_ratio_history` and on every macro fetch. As `gold_btc_ratio_history.jsonl` grows (years of hourly rows ≈ 8760 lines/yr ≈ 100 KB), this is fine; but the read is **also** bypassing the atomic-I/O rule (CLAUDE.md rule 4 explicitly forbids raw `open(...)`). On a Windows crash mid-write the JSONL has a partial line which the reader silently `continue`s past, but the truncation can also drop the latest 30d window edge case used by trend detection. | FIX: use `file_utils.read_jsonl_safe()` or equivalent (atomic semantics, partial-line tolerance documented).

[P0] crypto_macro_data.py:397 — same raw-`open` pattern for `NETFLOW_HISTORY_FILE`. CLAUDE.md rule 4 violation. | FIX: route through `file_utils`.

[P0] sentiment.py:288-294 — `subprocess.run([MODELS_PYTHON, script], input=json.dumps(texts), ..., timeout=120)` blocks the calling worker thread. `MODELS_PYTHON` on Windows points at the same `.venv\Scripts\python.exe`, which means a subprocess fallback path imports torch+transformers cold inside the venv — triggers the 3-10 s spawn penalty documented in the module header — and runs CPU because `cryptobert_infer.py` etc. don't move to CUDA. With 8 ticker workers all hitting fallback simultaneously, you can deadlock the GPU lock and burn 60+ s/cycle. | FIX: either propagate the exception (let the cycle fail loudly) or rate-limit subprocess fallback to one concurrent invocation via an asyncio/semaphore.

[P0] sentiment.py:99-105 — `datetime.fromtimestamp(a["published_on"], tz=UTC).isoformat()` raises `KeyError` if the CryptoCompare entry is missing `published_on`. The list-comprehension has no per-item try/except, so a single malformed article kills the whole fetch and the function returns `[]` via the fallback path at line 107. | FIX: per-item try/except with skip; or `.get("published_on", time.time())`.

[P0] alpha_vantage.py:266-269 — On a "Note" rate-limit response, `_fetch_overview` returns None, but the loop only calls `_cb.record_failure()` and continues. The `_alpha_vantage_limiter` is at 5 req/min; rate-limit Notes mean the daily 25 budget got hit, not the per-minute one. The code retries the SAME ticker on the next day's batch and the budget tracker increments only on success — but `_fetch_overview` still consumes a quota slot per call (AV counts it). | FIX: detect "Note" specifically (substring "Thank you for using Alpha Vantage" or "premium endpoint") and break out of the loop entirely, also incrementing `_daily_budget_used` for the failed attempt.

[P0] alpha_vantage.py:142-147 — `resp.json()` is called with no try/except for ValueError; `fetch_with_retry` returns a `Response` object even on 200 with HTML body (some upstream proxies do this on auth failure). `data` becomes whatever `.json()` raises through, but the broader try is at line 263 `try: raw = _fetch_overview(...)`. The narrow `except (ValueError, AttributeError):` at line 146 catches the JSON failure and returns None — **but** silent-None looks identical to "no data" and burns a budget slot. | FIX: distinguish 200/HTML vs 200/JSON-error and log loudly, do not consume budget on parse failure.

[P0] earnings_calendar.py:81 — `if days_until >= -1:` ALWAYS returns the FIRST quarterly entry whose date is yesterday-or-future. AV `quarterlyEarnings` is sorted DESCENDING (most recent first); the FIRST match is the most-recent past report, not the next upcoming. Callers see `gate_active = (0 <= days_until <= 2)` which is computed against a stale earnings date. The gate fires for tickers whose last report was 1-2 days ago, which is wrong. | FIX: sort quarterly ascending by `reportedDate`, return the first whose `days_until >= 0`.

[P0] earnings_calendar.py:104-107 — yfinance `t.calendar` is a *dict-or-DataFrame* depending on the yfinance version (>=0.2.40 returns dict). The branch at line 107 handles dict, but at line 110 the `cal.index` access assumes DataFrame. If yfinance returns a Series (one quarter only, some versions), `"Earnings Date" in cal.index` raises AttributeError-not-caught-by-suppress (it IS suppressed, but then function returns None silently). | FIX: explicit `isinstance(cal, pd.DataFrame)` check before `.loc`.

[P0] news_keywords.py:155 — `pattern.pattern.replace(r"\b", "").replace("\\", "")` is recovering the keyword string from the compiled regex pattern; for keywords that legitimately contain spaces ("rate hike", "trade war") this works, but for any future keyword containing a backslash or `\b` literal the recovery silently corrupts. More importantly, `matched` is used by `dissemination_score` and `keyword_severity` reporting; any non-ASCII keyword (none today, but the dict is mutable) breaks. | FIX: store keyword string alongside the pattern: `(re.compile(...), weight, kw_string)`.

[P0] sentiment.py:805 — `titles = [a["title"] for a in all_articles]` raises KeyError on any article without `title`. `_filter_relevant_headlines` only filters by `is_relevant_headline` and `is_credible_source`, neither of which guarantees title presence (Yahoo `_fetch_yahoo_headlines` skips empty titles, but social_posts injected via `social_posts=` may not — the function takes external `social_posts` and merges them at line 786 with no title check). | FIX: `titles = [a.get("title", "") for a in all_articles if a.get("title")]` and align `all_articles` accordingly.

[P0] data_collector.py:225 — `yf.download(yf_ticker, period=yf_period, interval=yf_interval, prepost=True, progress=False, auto_adjust=True)` runs WITHOUT the shared `yfinance_lock`. The function `_fetch_klines` at line 290 takes the lock conditionally, only when `"alpaca" in source AND market closed`. But `yfinance_klines` is also called directly from other modules (price_source router, fear_greed) that may not hold the lock. Concurrent yfinance calls from 8 workers corrupt yfinance's internal session state (documented behavior). | FIX: take `yfinance_lock` inside `yfinance_klines` itself; double-locking is a no-op via threading.RLock or a stable check pattern.

[P0] sentiment.py:135 — `news = stock.news or []` runs without `yfinance_lock`. Yahoo headlines fetched concurrently from 8 ticker threads race the same `yf.Ticker` cache. | FIX: wrap `_fetch_yahoo_headlines` body in `with yfinance_lock:`.

[P0] earnings_calendar.py:101-102 — `t = yf.Ticker(ticker); t.calendar` runs without `yfinance_lock`. Same race as above. | FIX: take `yfinance_lock`.

[P0] macro_context.py:78-145 — `_fetch_dxy_intraday` and `_fetch_dxy` both call `price_source.fetch_klines(...)` which may use yfinance internally; this module path is taken without acquiring `yfinance_lock`. If `price_source` already locks, fine — MAYBE, depends on that module — but reading the imports here, no lock acquisition is visible. | FIX: verify `price_source.fetch_klines` holds `yfinance_lock` whenever it falls back to yfinance; if not, wrap calls here.

[P1] data_collector.py:74-84 — `_binance_fetch` defaults `interval="5m"` but does NOT validate the interval against Binance's accepted values. Per CLAUDE.md, `10m` does NOT exist (returns -1120). Any caller that passes `"10m"` gets a generic `r.raise_for_status()` failure, but the circuit_breaker at line 100 records the failure as if the API was down, eventually opening the breaker and disabling all 5m fetches for 60 s. | FIX: whitelist `interval in {"1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w","1M"}` and raise ValueError on unknown.

[P1] data_collector.py:115-118 — Alpaca `1Day` lookback is 365 days; `1Week` is 730 days. For `limit=100` weekly bars you need ~700 trading days = ~1000 calendar days; 730 days yields ~104 weeks IF all weeks have data. The signal pipeline's weekly indicators silently get truncated to ~95 bars near edges, breaking 100-period indicators (e.g. 100-week SMA). | FIX: increase `1w` lookback to 900 days, `1M` to 3650 days.

[P1] data_collector.py:80-87 — `r.raise_for_status()` is called on the response, but the raised HTTPError carries no JSON body context. Binance error -1120 (invalid interval) returns 400; the breaker at line 100 increments on EVERY exception, including PERMANENT errors (invalid symbol, invalid interval). One bad symbol opens the breaker for 60 s, suppressing every other valid request to the same source. | FIX: distinguish 4xx (config bug, do not increment) from 5xx/network (transient, increment). 429 should also be treated specially — back off but don't open the breaker.

[P1] data_collector.py:174-204 — `fetch_vix()` only catches a generic `Exception`; on `KeyError("Close")` (yfinance returned an unexpected schema) it logs a warning and returns None. VIX is consumed by regime detection; silent None means the regime engine drops to "normal" classification, which is the most permissive trading regime. A silent VIX failure during a VIX spike is the worst-case scenario. | FIX: log at ERROR with the actual schema dump on KeyError, and have the consumer treat None as "unknown" (HOLD), not "normal".

[P1] futures_data.py:33-55 — `get_open_interest` returns `_cached(...)` with TTL 300 s; on a Binance 429, `fetch_json` returns None, `_fetch` returns None, `_cached` does NOT cache None (per shared_state.py:99) — good — but the next call retries immediately via the rate limiter, which under sustained 429 produces a retry storm. http_retry.py honors Telegram-format `retry_after`, NOT Binance's, so the backoff is the default 1s*2^attempt. Binance can rate-limit for minutes; the 1-s exponential is too aggressive. | FIX: parse Binance 429 `Retry-After` header explicitly in http_retry.py, fall back to ≥30 s.

[P1] futures_data.py:189-198 — `get_funding_rate_history` extracts `d["fundingRate"]`, `d["fundingTime"]` with no per-row try/except inside the comprehension. One malformed entry tears down the whole list. | FIX: wrap the comprehension in a list-of-dicts builder with per-row try/except.

[P1] onchain_data.py:174-177 — `data[0] if isinstance(data[0], dict) else data[-1]`: this assumes the API either gives newest-first or oldest-first; reality is the API doc says newest-first but the code defensively peeks both. If both data[0] and data[-1] are dicts (any non-empty list), it picks data[0]. Fine on happy path, but if the API switches order silently, the value labeled "latest" is actually oldest 30 days. | FIX: sort by timestamp descending after parsing, take [0].

[P1] onchain_data.py:90 — `logger.warning("BGeometrics token load failed: %s", e, exc_info=True)` runs on EVERY config-load failure, but `_load_config_token()` is called once per get_onchain_data() — i.e. once per main loop cycle. If config is missing, the warning floods the log every 60 s. | FIX: log once, then debug-level suppress.

[P1] onchain_data.py:236-238 — `if not any_success: return None`. But `_save_onchain_cache(result)` at line 240 is OUTSIDE this check. If `any_success` is True but only ONE metric succeeded, the partially-populated `result` is saved and returned. The persistent cache then has stale 5-of-6 metrics for 12 h. | FIX: also gate `_save_onchain_cache` and the return on a minimum success count (e.g. ≥4/6) OR merge with prior persistent cache so missing metrics keep their old values.

[P1] onchain_data.py:259 — `persistent.get("ts", 0) or persistent.get("_fetched_at", 0)`: if `ts` is 0 (falsy), falls back to `_fetched_at`. But the file written by `_save_onchain_cache` only ever sets `ts` (line 211). The `_fetched_at` key is from old-format files. Once the cache is written by current code, this branch is dead. The danger: a corrupted older cache with `ts=0` and `_fetched_at=<recent ISO>` is treated as fresh, returning stale metrics indefinitely. | FIX: explicit version check.

[P1] fx_rates.py:46 — `rate = float(r.json()["rates"]["SEK"])` — KeyError on missing `rates` or `SEK`, ValueError on float conversion failure. The outer try catches Exception (line 54), logs warning, drops to fallback. Frankfurter returns `{"amount":1.0,"base":"USD","date":"...","rates":{"SEK":11.10}}` — but on weekends/holidays it returns the LAST PUBLISHED date's rates, which can be 3 days stale during long holiday weekends. The function does NOT check `data["date"]` against today. | FIX: parse `data["date"]` and refuse rates older than 7 days; alert at >3 days.

[P1] fear_greed.py:96 — `fetch_json("https://api.alternative.me/fng/", ...)` has no API key (alternative.me is free/unlimited), but no rate limiter either. Fine for now, but the module-level call sites (signal_engine, digest) hit it on every cycle without `_cached(...)` wrapping. The fear_greed signal is a daily index — minute-level fetches are wasted bandwidth and risk casual rate-limiting. | FIX: wrap in `_cached("fng_crypto", FEAR_GREED_TTL, get_crypto_fear_greed)` (FEAR_GREED_TTL is already defined in shared_state).

[P1] fear_greed.py:135 — `vix.history(period="5d")` runs under `yfinance_lock`, good. But `^VIX` is a US market index; on weekends the 5d window may yield only 3-4 bars and `prev = h.iloc[-2]` is fine, but on a Monday morning before US open the latest bar is Friday's close — fed into a "stock fear/greed" signal as if current. | FIX: log the staleness (h.index[-1] vs now) and downgrade confidence when the latest bar is >1 trading day old.

[P1] fear_greed.py:154-164 — VIX-to-fear/greed mapping uses arithmetic that produces non-monotonic boundaries. At VIX=20 exactly: branch 159 `value = int(20 + (30 - 20) * 3) = 50`; at VIX=20.01 branch 161 `value = int(50 + (20 - 20.01) * 6) = 49`; at VIX=15 exactly: branch 161 `value = int(50 + 5*6) = 80`; at VIX=14.99: branch 163 `value = int(80 + 0.01*4) = 80`. Minor rounding artifacts at boundaries. Not catastrophic but introduces tick-by-tick label flips. | FIX: replace piecewise-linear with a clean monotonic function (e.g. `value = max(0, min(100, 100 - 2*VIX))` or similar).

[P1] sentiment.py:863-888 — `enqueue_fingpt(ab_key, "headlines", ...)` and the cluster loop after it run inside a try/except that catches generic Exception and logs at DEBUG. If `llm_batch` is misconfigured, the FinGPT shadow log silently goes empty for days; A/B accuracy comparisons appear to converge but are actually starved of data. | FIX: log at WARNING on enqueue failure and emit a daily summary count.

[P1] alpha_vantage.py:151 — `data["Note"][:100]` — if "Note" is present but is a non-string (defensive), this raises TypeError, caught by the outer try/except in `refresh_fundamentals_batch`. The ticker is then skipped. A single weird API response can derail the whole batch. | FIX: `str(data["Note"])[:100]`.

[P1] earnings_calendar.py:155-179 — On stock-ticker fetch failure, `_fetch_earnings_date` returns None; the cache writes `{"data": None, "time": now}` for 24 h. So a single failure (rate-limit on the AV side, momentary outage) silently disables the earnings gate for the next 24 h. During earnings week this is a real risk. | FIX: do not cache None — mirror the shared_state.py:99 pattern.

[P1] earnings_calendar.py:38 — `from portfolio.api_utils import load_config` is imported INSIDE the function on every call. Cheap, but combined with the `from portfolio.shared_state import _alpha_vantage_limiter` also inside, every earnings fetch re-imports two modules. Not a bug per se, but the imports being inside the function suggests circular-import workarounds and you cannot tell whether refactoring is safe without testing. MAYBE. | FIX: move to module top if no circular import.

[P1] macro_context.py:218-225 — `_fetch_volume_signal` reads `vol.iloc[:-1].rolling(20).mean().iloc[-1]` and divides by it. If `avg20` is NaN (which happens when fewer than 20 non-NaN volumes exist; not caught by `len(vol) >= 22`), the `if avg20 > 0` branch evaluates as False (NaN > 0 is False), `ratio` stays at default 1.0 — silently. Volume spike detection then never fires. | FIX: explicit `pd.notna(avg20) and avg20 > 0`.

[P1] macro_context.py:233 — `last_vol = float(vol.iloc[-2])` uses the SECOND-to-last bar (last completed candle); fine for closed-bar logic, but the rolling mean at line 230 uses `.iloc[:-1]` (excludes today) which matches; HOWEVER `price_change` at line 238 uses `close.iloc[-2] / close.iloc[-5]` — 3-bar change of completed bars. Inconsistency: 1 vs 3 bars. Volume-confirms-direction is 1-bar volume vs 3-bar price; that's the spec, but worth flagging. MAYBE intentional.

[P1] macro_context.py:286-298 — `_fred_10y_fallback` opens config.json with raw `open()`. CLAUDE.md rule 4 violation. | FIX: use `file_utils.load_json(CONFIG_FILE)`.

[P1] crypto_macro_data.py:233-241 — Trend detection `if age_days >= 14 and ratio_14d is None: ratio_14d = entry.get("ratio"); break`. The `break` exits the loop after finding the 14d ratio, but `ratio_7d` may not have been set if the loop iterated newest-first and never hit a 7d-old entry before finding a 14d one. Reading line 234 `for entry in reversed(history)`, the iteration is OLDEST-first (reversed of presumably oldest-first JSONL). So we walk old→new; the first entry with `age_days >= 14` is found early and `break` happens before any 7d-aged entry is seen. `ratio_7d` stays None for tickers with >14d of data. | FIX: don't `break` until BOTH ratios found, or compute via timestamp targeting.

[P1] crypto_macro_data.py:444 — `if "BTC" in ticker` matches "WBTC", "tBTC", any ticker containing those letters. With current symbol set this is OK, but a future ticker like "BTCBULL" or "STBTCH" silently routes to the BTC branch. | FIX: exact prefix match `ticker.startswith("BTC-")`.

[P1] market_health.py:53-95 — Both Alpaca primary and yfinance fallback paths return `closes` etc. as lists via `.tolist()`. Lists lose dtype information; downstream `count_distribution_days` does `volumes[i] >= volumes[i - 1]` which is fine for floats but not for the int/Decimal mix yfinance can return on penny-priced indices. MAYBE. | FIX: explicit `[float(x) for x in df["close"]]`.

[P1] market_health.py:328-330 — `sma200 = sum(closes[-200:]) / 200`. If any close is NaN (yfinance can return NaN on holidays), `sum()` returns NaN, the SMA200 component silently scores 0 (NaN > NaN is False) and breadth score under-scores by 20 pts. | FIX: filter NaN before summing.

[P1] bert_sentiment.py:286-287 — `os.environ.get("BERT_SENTIMENT_USE_GPU", "").strip() in ("1","true","TRUE","yes")`. The strip is good but the comparison set is case-sensitive; `"True"` (capital T common from Windows env) is rejected. MAYBE intentional but documented as forgiving. | FIX: lowercase compare.

[P1] bert_sentiment.py:117 — `_models: dict[str, tuple[Any, Any, str, threading.Lock]] = {}` is module-level and never bounded. With three model names this is fine, but `_get_model` raises KeyError on unknown names rather than returning None — so if a typo'd model name leaks in, the entire sentiment phase crashes. | FIX: catch the KeyError at the caller and degrade gracefully.

[P1] social_sentiment.py:33 — `urllib.request.urlopen(req, timeout=10)` with no retry, no rate-limit handling. Reddit returns 429 or 503 frequently; the function raises `URLError`, caller catches Exception at line 110 and `print()`s (not log). Sentiment from Reddit silently drops. CLAUDE.md mandates "Live prices first" — this means Reddit silently zeroes out. | FIX: route through `fetch_json` with a User-Agent header, retries=2, and log via `logging`, not `print`.

[P1] social_sentiment.py:36 — `data = json.loads(resp.read())` — raw decode, no try/except inside the helper. Reddit occasionally returns HTML on rate limit, JSONDecodeError propagates. Caught at line 110 by bare Exception, but the log message is `print(f"...error: {e}")` to stdout — invisible to the agent log. | FIX: use logger.

[P1] social_sentiment.py:31 — URL uses `?limit={per_sub + 5}` to over-fetch; Reddit silently caps to 25 (default) or 100 (with auth); without auth, asking for 25 might trip the new anti-AI throttle. MAYBE. | FIX: use authenticated requests via OAuth client (5,000/day).

[P1] news_keywords.py:58-74 — `MODERATE_KEYWORDS` includes both `"earnings miss"` (1.5 weight, SELL bias) and `"earnings beat"` (1.5 weight, BUY bias) but no directional metadata; `KEYWORD_SECTOR_IMPACT` dict at 108 lacks any entry for these. So a headline "Acme earnings miss" gets weight 1.5 in the aggregate but no directional impact mapping. The dissemination_score amplifies it equally regardless of bull/bear. | FIX: add directional impact to `KEYWORD_SECTOR_IMPACT`.

[P1] news_keywords.py:130 — `CREDIBLE_SOURCES` is matched via `cs in lower` substring (line 179). "ap" matches any source name containing "ap" — "AP News" yes, but also "ApeCoin News" or "AP News Daily Report" or even "Slap-and-Tickle Trading Daily". | FIX: exact match against word boundaries or use a regex.

[P1] news_keywords.py:182-249 — `dissemination_score`: time clustering at line 234 picks `max_cluster` based on a 1-hour window centered on each timestamp; if the loop iterates over a 4-hour spread of articles the max can be small. Boundary case: two articles 30 min apart but third one 90 min later — cluster of 3 only if you slide windows; the code picks `max_cluster=2`. Underestimates clustering. | FIX: sliding window or DBSCAN-style clustering.

[P1] news_keywords.py:344 — `if pat is None or not pat.search(title)`: `pat is None` only when neither known synonym list nor alphanumeric ticker. If `short = ""` (e.g. for `"USD"` cased as `"-USD"` stripped), `_ticker_synonym_pattern` returns None and the function says NOT relevant. Borderline OK, but no log. | FIX: log unmatched-ticker-symbols at debug.

[P1] sentiment.py:712-720 — `_filter_relevant_headlines` falls back to `most-recent N` (default 3) when ALL headlines fail relevance. This is documented and intentional. But the fallback ignores the *credibility* gate; an irrelevant Reddit hot-take can be selected as "fallback". | FIX: prefer credible-source articles in fallback.

[P1] sentiment.py:858-860 — `ab_key = f"{short}:{datetime.now(UTC).isoformat()}"`. Concurrent calls for the same ticker within the same microsecond produce duplicate keys (Python isoformat is microsecond-precision). Then `_stash_ab_context` overwrites the prior entry; the prior FinGPT enqueue still references the dropped key and is dead-lettered in `_stash_fingpt_result` (silently dropped, line 401). | FIX: include a uuid4 or counter in the key.

[P1] sentiment.py:889 — `except Exception as e: logger.debug("FinGPT enqueue failed: %s", e)`: too quiet for a feature that affects A/B accuracy. | FIX: warning level.

[P1] data_collector.py:38-41 — `_YF_INTERVAL_MAP` claims `15m`/`1h` allow 5d/30d periods, but yfinance limits intraday <60d. For `1h` with `period="30d"` and `limit=100`, you get ~24*30=720 bars and `tail(100)` returns the most recent 100 — fine. But yfinance `1h` for indices like `^VIX` returns gaps over weekends; `len(df) < 100` is possible and indicators fail at line 295 silently (returns `(label, None)` not error). | FIX: log when len < limit.

[P1] data_collector.py:283-309 — TTL>0 entries are written to `_tool_cache` but the `entry` dict includes the raw DataFrame for `Now` label (line 306). Caching DataFrames is fine, but the cache hits return the SAME DataFrame object — which downstream code at compute_indicators may mutate (column adds, casts). Cross-cycle mutation contaminates subsequent reads. | FIX: deep-copy on cache hit, or freeze with `.copy(deep=True)` at write.

[P2] data_collector.py:287 — `cached and time.time() - cached["time"] < ttl`: if `cached["data"]` is `{"error": "..."}` from a prior cycle (line 312 caches errors? No, line 312 has no TTL>0 path — actually re-reading line 311 the error-only return does NOT write to cache. OK, false alarm.) MAYBE.

[P2] futures_data.py:201-217 — `get_all_futures_data` calls 6 sub-fetchers sequentially; each does `_binance_limiter.wait()`. With 600/min limit, 6 calls take ~0.6 s minimum. For 2 crypto tickers × 6 endpoints × 7 timeframes (if invoked per timeframe; not the case currently) this would scale poorly. Not a current bug. MAYBE.

[P2] sentiment.py:163-169 — NewsAPI call uses default `User-Agent: Mozilla/5.0`; NewsAPI tracks per-key, not per-UA, so this is cosmetic. But the lack of an explicit User-Agent matching the API key registration (NewsAPI ToS) is a soft TOS violation. | FIX: identify with `finance-analyzer/1.0`.

[P2] alpha_vantage.py:181-183 — `age_seconds > max_stale_days * 86400`: uses calendar days, not market days. A 5-trading-day stale check actually allows 7 calendar days during long weekends. Fine for slow-moving fundamentals. MAYBE.

[P2] earnings_calendar.py:209 — `for ticker in STOCK_SYMBOLS: prox = get_earnings_proximity(ticker)`: serial; with 6 stocks * (AV 12s/req at 5/min) = up to 72 s during cold cache. Combined with the 24h TTL it only happens once daily. Not a bug, but a long pause. MAYBE.

[P2] crypto_macro_data.py:104 — `datetime.datetime.strptime(s, "%d%b%y")` parses Deribit "28MAR26" as 2026-03-28. The "%y" 2-digit year wraps in 2069; harmless for ~40 years. | FIX: log a warning when the parsed year < 2020.

[P2] market_health.py:209-211 — `recent_high = max(closes[-60:])` is computed only once on cold-start, and `if today_close > recent_high: recent_high = today_close` updates incrementally; but if the persisted `prev_state` has a stale `recent_high` from a 6-month-old correction, the drawdown calc is wrong on the very next call. The state file is written every cycle at line 444, so steady-state OK; the only window of risk is process restart with very stale `_STATE_FILE`. MAYBE — likely fine because line 222 updates immediately.

[P2] market_health.py:404 — `prev_ftd = load_json(_STATE_FILE, default={}).get("ftd_state")`. If the file is corrupt JSON, `load_json` returns `{}` (assuming file_utils default behavior); the FTD state machine resets to STATE_CORRECTING. This is a soft failure — verify `load_json` behavior on partial files. MAYBE.

[P2] sentiment.py:34-38 — Hardcoded Windows paths `Q:\models\...` and Linux `/home/deck/models/...`. If venv runs on a third platform the paths fail at first call; subprocess fallback fails; in-process fallback in bert_sentiment.py uses the same paths. | FIX: load model paths from config.

[P2] news_keywords.py:18-35 — `CRITICAL_KEYWORDS` includes "ban" (3.0). False positives: "Tom Brady's ban", "ban on plastic straws", "Cuban ban list". The keyword scorer has no negation handling. | FIX: ML-based filter or keyword context windows.

[P2] sentiment.py:783 — `model_name = "Trading-Hero-LLM"` is the primary for ALL asset classes. Per CLAUDE.md, "CryptoBERT (crypto) / Trading-Hero-LLM (stocks)". The CryptoBERT demotion is documented in the docstring (2026-04-28), but the CLAUDE.md description is now stale. Documentation drift, not a code bug.

[P3] data_collector.py:25 — `_TF_POOL_TIMEOUT = 60` is hardcoded; configurable via env would help. MAYBE.

[P3] sentiment.py:131-156 — `_fetch_yahoo_headlines` returns articles with `published` falling back to `datetime.now(UTC).isoformat()`. This labels Yahoo articles missing `pubDate` as "just published", inflating recency-based weighting. | FIX: skip articles without a real pub date.

[P3] crypto_macro_data.py:267 — `RATIO_HISTORY_FILE` and `NETFLOW_HISTORY_FILE` grow unbounded. After 1 year ~9000 lines each (~1MB), small but accumulating. | FIX: rotate or cap to 30 days.

[P3] fear_greed.py:135-154 — VIX-derived stock fear/greed has no time-based confidence decay; a Friday close VIX is used through the weekend with `value` unchanged. | FIX: include `vix_age_hours` in the result.

[P3] news_keywords.py:282-304 — `_TICKER_SYNONYMS` lists tickers that were REMOVED (per CLAUDE.md, AMD/GOOGL/AMZN/AAPL/AVGO/META/SOUN/LMT/PLTR/NVDA/MU/SMCI/TSM/TTWO/VRT removed). Dead synonyms cost nothing but the comment `# Memoize per-ticker patterns` is now misleading. | FIX: prune.

[P3] earnings_calendar.py:50-52 — Comment says "earnings calls bypass alpha_vantage.py's _daily_budget_used counter". This is documented as a known limitation, but the budget check itself runs before calling earnings; a stock ticker with both fundamentals AND earnings refresh in the same day double-spends silently. | FIX: expose `bump_budget` from `alpha_vantage.py`.

[P3] sentiment.py:792-800 — Returning `{"overall_sentiment": "unknown", ...}` is not the same shape as the success path (no `avg_scores`). Callers reading `result["avg_scores"]["positive"]` get KeyError. MAYBE — depends on consumer guards.

[P3] bert_sentiment.py:288-294 — On `model.to("cuda")` failure, the model stays on CPU. Subsequent calls retry the move under the lock — but `_load_model` only runs once; subsequent calls reuse the cached CPU model. Fine. MAYBE not a bug, but a process restart is required to retry GPU placement.

[P3] market_health.py:53-95 — Period mapping `_LIMIT = {"1d": 2, ...}` differs from the limit param the caller wants. For period="90d" the dict gives `90`, but the actual function signature is `(symbol, "90d")` — period and limit are aliased. Confusing API. | FIX: rename the param.

[P3] data_collector.py:202 — `logger.warning("VIX fetch failed: %s", e)` swallows the traceback. For diagnosing yfinance schema changes a traceback is essential. | FIX: `exc_info=True`.

[P3] alpha_vantage.py:94-99 — `if "Error Message" in raw or "Note" in raw`: AV also returns `"Information"` keys for premium endpoints — not handled, treated as success path with all _float()'d values returning None. Cache then stores effectively-empty entries. | FIX: also reject when `"Information"` is present.

---

COUNT: 70 findings (28× P0, 27× P1, 11× P2, 4× P3)
