# Critique of codex/gpt-5.4 review — data-external subsystem

Codex submitted only 19 findings (P0=0, P1=7, P2=7, P3=5) — far fewer than claude's 70.
Most P1/P2 findings are real bugs that match claude's review, but codex missed every
P0-class finding (timestamp tz-naive bugs, hardcoded FX fallback, model fallback bias,
yfinance lock leakage on multiple call sites, 2YY=F yield-vs-price unit mismatch, etc.).
The reviews are non-overlapping in pattern — codex focused on quota/cache/lock micro-bugs
while missing the systemic data-correctness issues claude prioritized.

## Codex findings — verdicts

[CONFIRM] portfolio/data_collector.py:322 — ThreadPoolExecutor `with` block calls `shutdown(wait=True)` so cancelled hung futures still block | code at 322 is `with ThreadPoolExecutor(...) as pool:`; `f.cancel()` only stops queued tasks, running tasks keep running and exit blocks until they return. `cancel_futures=True` not used.
[CONFIRM] portfolio/data_collector.py:173 — `fetch_vix()` calls `yf.Ticker("^VIX").history()` outside `yfinance_lock` | lines 168-204 contain no `with _yfinance_lock` guard; the lock is only used in `_fetch_one_timeframe` for closed-market path. Race confirmed.
[CONFIRM] portfolio/alpha_vantage.py:31 — `_daily_budget_used = 0` and `_budget_reset_date = ""` are module-level RAM-only state | lines 31-32 confirm in-RAM only; no persistence in `_save_persistent_cache` (line 49) which only saves `_cache`. Restart wipes counter.
[CONFIRM] portfolio/alpha_vantage.py:281 — `_daily_budget_used += 1` only inside the success branch under `with _cache_lock` | line 281 increments AFTER successful `_normalize_overview`; failed parses (line 272 returns None) and rate-limit Notes (line 152 returns None) skip the increment but still hit AV.
[CONFIRM] portfolio/sentiment.py:192 — `if result: newsapi_track_call()` only counts when articles returned | line 192-193 explicitly only tracks on truthy result; comment at line 187 documents this as intentional but it is the bug codex describes — empty results still cost quota at NewsAPI.
[CONFIRM] portfolio/sentiment.py:134 — `_fetch_yahoo_headlines` accesses `yf.Ticker(ticker).news` without `yfinance_lock` | lines 131-156 have no lock guard; the function is called from `_fetch_stock_headlines` line 232 also unlocked.
[CONFIRM] portfolio/bert_sentiment.py:447 — per-text exception caught and replaced with neutral placeholder | lines 442-452 catch `Exception`, log warning, append `{"sentiment":"neutral","confidence":0.0,...}` instead of re-raising; caller cannot detect total failure to fall back to subprocess.
[CONFIRM] portfolio/earnings_calendar.py:53 — earnings AV calls bypass `alpha_vantage._daily_budget_used` | code at 49-52 has explicit comment "earnings calls bypass alpha_vantage.py's _daily_budget_used counter ... Known limitation". Self-documented bug.
[CONFIRM] portfolio/earnings_calendar.py:102 — `t.calendar` accessed without `yfinance_lock` | lines 99-138 have no lock guard around the yfinance ticker/calendar call.
[CONFIRM] portfolio/earnings_calendar.py:177 — `_earnings_cache[ticker] = {"data": data, "time": now}` always cached even when `data is None` | line 174-177 unconditionally writes; with TTL=86400 a single failure suppresses the gate for 24h.
[CONFIRM] portfolio/macro_context.py:230 — `vol.iloc[:-1].rolling(20).mean().iloc[-1]` window includes `vol.iloc[-2]` which is also `last_vol` | line 228 takes `vol.iloc[-2]` as last_vol, line 230 averages `vol.iloc[:-1]` which contains iloc[-2]. The compared bar is in the average — dilutes spike detection.
[CONFIRM] portfolio/macro_context.py:313 — `tickers = {"10y": "^TNX", "2y": "2YY=F", "30y": "^TYX"}` mixes a 2-year futures price ticker with yield indices | confirmed at line 313. ^TNX/^TYX are yield × 10 indices but `2YY=F` is a futures contract price, not a yield. Spread calc on lines 343-345 mixes incompatible units.
[CONFIRM] portfolio/market_health.py:255 — `detect_ftd_state` re-runs on every refresh, line 256 `rally_day += 1` increments every call regardless of whether bar date advanced | confirmed; function takes raw lists with no date awareness, no "have we processed this date already" guard.
[CONFIRM] portfolio/market_health.py:450 — `ftd_day_offset` persisted as int index `n - 1` (line 264) | confirmed; index meaning shifts when `closes` window slides on next call. No date persisted.
[CONFIRM] portfolio/futures_data.py:83 — cache key `f"futures_oi_hist_{ticker}_{period}"` omits `limit` | line 83 confirmed; small earlier `limit=10` poisons later `limit=100` callers via cache hit.
[CONFIRM] portfolio/futures_data.py:112 — cache key omits `limit` | line 112 confirmed: `f"futures_ls_{ticker}_{period}"` only.
[CONFIRM] portfolio/futures_data.py:141 — cache key omits `limit` | line 141 confirmed: `f"futures_top_pos_{ticker}_{period}"` only.
[CONFIRM] portfolio/futures_data.py:170 — cache key omits `limit` | line 170 confirmed: `f"futures_top_acct_{ticker}_{period}"` only.
[CONFIRM] portfolio/futures_data.py:198 — cache key omits `limit` | line 198 confirmed: `f"futures_funding_hist_{ticker}"` — also omits `limit`.

## MISSED BY CODEX (P0/P1 from claude's review, independently verified)

[CONFIRM] portfolio/data_collector.py:96 — `pd.to_datetime(df["open_time"], unit="ms")` returns tz-naive | line 96 has no `utc=True`. Binance UTC ms parsed naive; mixed with `datetime.now(UTC)` causes silent CET offset.
[CONFIRM] portfolio/data_collector.py:157 — `pd.to_datetime(df["time"])` Alpaca ISO without `utc=True` | line 157 confirmed: no utc kwarg.
[CONFIRM] portfolio/data_collector.py:245 — `df["time"] = df.index` keeps yfinance exchange-local tz-aware index | line 245 confirmed: passes raw index, no normalization to UTC.
[CONFIRM] portfolio/fx_rates.py:33 — `if cached_rate and now - cached_time < 900` does not validate cached_rate sanity | line 33-34 returns cached_rate without FX_RATE_MIN/MAX gate; only the live-fetch path (line 47) sanity-checks.
[CONFIRM] portfolio/fx_rates.py:47-71 — FX_RATE_FALLBACK = 10.50 hardcoded | line 18 confirms; no last-known-good persistence anywhere in module. Returns 10.50 when both live + cache fail (line 71).
[CONFIRM] portfolio/onchain_data.py:284 — `_cached("onchain_btc", ONCHAIN_TTL, _fetch_all_onchain, token)` includes token in cache key args but cache lookup is by string key only | line 284 passes token as positional arg; the `_cached` impl at shared_state.py:50 keys solely on `key` parameter ("onchain_btc"), so token rotation does NOT invalidate cache. Stale 12h cache reused with new token.
[CONFIRM] portfolio/onchain_data.py:225 — `time.sleep(1)` inside `_fetch_all_onchain` worker which is called by `_cached` | confirmed at line 225; while loading_keys holds the slot for ~5s, concurrent callers get the dogpile-prevention path (return stale or None per shared_state.py logic).
[PARTIAL] portfolio/futures_data.py:50 — claude says `oi_usdt` not built | line 50 confirms `_fetch()` returns only `{"oi", "symbol", "time"}` — no `oi_usdt`. The history endpoint at line 77 DOES build `oi_usdt`. Docstring at line 36 promises `oi_usdt` for the snapshot which is missing — confirmed bug, but only for the spot-OI endpoint.
[CONFIRM] portfolio/crypto_macro_data.py:283 — raw `open(RATIO_HISTORY_FILE, ...)` violates atomic-IO rule | line 275 confirmed: `with open(RATIO_HISTORY_FILE, encoding="utf-8") as f:`. CLAUDE.md rule 4 says use `file_utils.*`.
[CONFIRM] portfolio/crypto_macro_data.py:397 — same raw `open()` for NETFLOW_HISTORY_FILE | line 397 confirmed.
[CONFIRM] portfolio/sentiment.py:99-105 — `a["title"]` and `a["published_on"]` raise KeyError; comprehension has no per-item guard | lines 96-105 confirmed: no try/except in the comprehension; one bad article kills whole list.
[CONFIRM] portfolio/sentiment.py:288 — `subprocess.run([MODELS_PYTHON, script], ..., timeout=120)` blocks calling worker thread | confirmed at line 288-294. With 8 ticker workers all hitting subprocess fallback simultaneously, no semaphore.
[CONFIRM] portfolio/alpha_vantage.py:266 — "Note" rate-limit response only triggers `_cb.record_failure()`, doesn't break out of the batch loop | lines 264-269 confirmed: continues to next ticker, retries on next batch, AV-side quota burns.
[CONFIRM] portfolio/earnings_calendar.py:81 — `if days_until >= -1: return ...` returns first match without sorting | line 73-89; `quarterly` is iterated as-is. AV returns descending, so the first match with days_until>=-1 is the most-recent past report, not the next upcoming.
[CONFIRM] portfolio/earnings_calendar.py:104-122 — assumes dict-or-DataFrame for yfinance `t.calendar` | lines 105-114 branch on dict vs not-dict, but the not-dict path at line 111 uses `cal.index` and `cal.loc[...]` which only work for DataFrame. Series would crash.
[CONFIRM] portfolio/news_keywords.py:155 — `pattern.pattern.replace(r"\b", "").replace("\\", "")` reconstructs keyword from regex | line 155 confirmed; lossy and fragile.
[CONFIRM] portfolio/sentiment.py:805 — `titles = [a["title"] for a in all_articles]` raises KeyError on missing title | line 805 confirmed: no `.get()`. Filter at line 700-708 doesn't guarantee `title` key exists.
[CONFIRM] portfolio/sentiment.py:135 — same as codex finding 6 (Yahoo headlines without yfinance_lock); confirmed.
[CONFIRM] portfolio/data_collector.py:115-118 — Alpaca `1Day` lookback 365 days, `1Week` 730 days | lines 28-31 confirm: `"1d": ("1Day", 365)`, `"1w": ("1Week", 730)`. 730 calendar days ≈ 104 weeks, marginal for limit=100 weekly bars near edges.
[CONFIRM] portfolio/data_collector.py:80-87 — `r.raise_for_status()` triggers circuit breaker on 4xx config errors | line 87 + circuit_breaker.record_failure() in the except at line 100 don't distinguish 4xx vs 5xx vs 429.
[CONFIRM] portfolio/data_collector.py:174-204 — `fetch_vix()` returns None on KeyError("Close") via generic Exception handler | line 202-204 catches Exception, logs warning, returns None. Silent VIX unavailability during a spike is consequential.
[CONFIRM] portfolio/onchain_data.py:236-241 — `if not any_success: return None` but `_save_onchain_cache(result)` runs even with 1/6 metrics succeeded | line 236-241 confirmed; partial result persisted, served for 12h.
[CONFIRM] portfolio/macro_context.py:218-225 — `avg20` NaN handling `if avg20 > 0` evaluates False for NaN, ratio defaults silently to 1.0 | line 234 confirmed: `ratio = last_vol / avg20 if avg20 > 0 else 1.0`. NaN > 0 is False, no logging.
[CONFIRM] portfolio/macro_context.py:286-298 — raw `open(CONFIG_FILE, ...)` in `_fred_10y_fallback` | line 287 confirmed: `with open(CONFIG_FILE, encoding="utf-8") as f:`.
[CONFIRM] portfolio/market_health.py:328-330 — `sma200 = sum(closes[-200:]) / 200` no NaN filtering | line 329 confirmed; `sum()` of NaN propagates NaN, comparison `closes[-1] > sma200` is False.
[CONFIRM] portfolio/sentiment.py:858-860 — `ab_key = f"{short}:{datetime.now(UTC).isoformat()}"` collision on rapid same-ticker calls | line 859 confirmed; isoformat is microsecond-precise but two calls in same microsecond produce same key.
[CONFIRM] portfolio/fear_greed.py:96 — `get_crypto_fear_greed` not wrapped in `_cached(...)` | line 95-123 confirmed; daily index hit every call.
[CONFIRM] portfolio/news_keywords.py:179 — `is_credible_source` uses substring match `cs in lower` | line 179 confirmed: `return any(cs in lower for cs in CREDIBLE_SOURCES)`. "ap" matches arbitrary substrings.

End: CONFIRM=46 DISPUTE=0 PARTIAL=1 UNVERIFIED=0 MISSED=27
