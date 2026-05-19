# Adversarial Review — data-external subsystem

Reviewer: agent_7 / Opus 4.7 (1M context)
Date: 2026-05-19
Scope: portfolio/{data_collector,data_refresh,fear_greed,sentiment,sentiment_shadow_backfill,social_sentiment,bert_sentiment,alpha_vantage,futures_data,onchain_data,fx_rates,funding_rate,earnings_calendar,econ_dates,fomc_dates,macro_context,market_health,crypto_macro_data,news_keywords,microstructure,microstructure_state,price_source,indicators}.py

Severity legend:
- P0: bad data → bad signals → bad trade
- P1: real bug
- P2: latent / edge
- P3: minor

---

## portfolio/microstructure_state.py

**portfolio/microstructure_state.py:205-213: P0 (side-effect-in-read, double-counts OFI history).**
`persist_state()` calls `get_microstructure_state(ticker)` for every ticker. `get_microstructure_state` at line 185 unconditionally calls `record_ofi(ticker, ofi)`, which appends to `_ofi_history`. So every persist tick re-records the SAME OFI value (the snapshot buffer didn't change between this call and the previous one inside the cycle). With the metals_loop calling both the consumer and `persist_state` on each cycle, `_ofi_history` doubles — compresses variance → inflated z-scores → orderbook_flow false signals. Fix: separate `read_only` from `record`, or have `persist_state` snapshot the dict from `_ofi_history[-1]` directly. Same bug flagged in 3 prior reviews; still open.

**portfolio/microstructure_state.py:175-186: P1 (z-score self-contamination on first cycle).**
`get_ofi_zscore` uses the in-history population (excluding the passed `current_ofi`), but `record_ofi` is called UNCONDITIONALLY at line 185 even when the snapshot buffer hasn't grown. After warm-up, the same OFI value keeps appending to `_ofi_history`, which crushes `std` toward zero and pushes the z-score either to 0 (line 122 short-circuit) or to infinity when there's any real change. Pairs with the P0 above.

**portfolio/microstructure_state.py:205-213: P1 (persist_state takes no lock).**
`persist_state` reads `_snapshot_buffers` keys outside `_buffer_lock`. `metals_loop`'s 10s fast-tick can be inside `accumulate_snapshot` mutating the same dict while `persist_state` iterates → `RuntimeError: dictionary changed size during iteration`. Fix: `with _buffer_lock: tickers = list(_snapshot_buffers.keys())`.

**portfolio/microstructure_state.py:75: P2 (KeyError if depth missing `spread`).**
`_spread_buffers[ticker].append(depth["spread"])` will KeyError if upstream `metals_orderbook` returns a depth dict without the `spread` key (it doesn't always populate it on partial book updates). Use `depth.get("spread", 0)` or guard.

---

## portfolio/microstructure.py

**portfolio/microstructure.py:107-148: P1 (compute_ofi missing book-side KeyError guard).**
`compute_ofi` accesses `prev["best_bid"]`, `curr["best_bid"]`, `prev["best_ask"]`, `curr["best_ask"]` without `.get()` defaults. If `accumulate_snapshot` ever stores a snapshot from a partial book response (e.g. when only one side returned), the loop raises KeyError → entire `get_microstructure_state` cycle fails silently in the calling thread. Snapshots come from `metals_orderbook`; review of that module needed but the consumer must be defensive.

**portfolio/microstructure.py:23-38: P2 (depth_imbalance returns 0 on one-sided book — could be a real BUY/SELL pressure signal).**
When `bid_vol > 0 and ask_vol == 0` (extreme one-sided depth), the function returns 0.0 — same as a balanced book. The actual signal would be "infinite buying pressure." Returning 0 understates the signal during liquidity-cliff events (XAG fast moves) precisely when microstructure matters most. Consider returning a saturated value (e.g. ±5.0).

**portfolio/microstructure.py:172-227: P3 (trade-through detection mid-price denominator).**
`mid_price = (prev["price"] + curr["price"]) / 2.0` uses two trade prices, not bid/ask mid. After a big trade-through, the second price is far from the prior mid, biasing `gap_bps` downward (underdetects throughs after the first one). Minor — should use a rolling reference.

---

## portfolio/data_collector.py

**portfolio/data_collector.py:74-101: P0 (Binance error responses returned as garbage candles).**
`_binance_fetch` calls `r.raise_for_status()` and `r.json()`, then immediately tries `pd.DataFrame(data, columns=_BINANCE_KLINE_COLS)`. Binance error responses come back as 200 OK with a JSON dict like `{"code": -1121, "msg": "Invalid symbol."}` (e.g. on a typo, or for `10m` interval per CLAUDE.md). That dict gets shoved into the DataFrame constructor — the resulting frame has 1 row of `code`/`msg` strings cast to float later → ValueError or silent garbage. Fix: `if isinstance(data, dict) and "code" in data: raise ConnectionError(...)`.

**portfolio/data_collector.py:96: P1 (Binance timestamps are tz-naive but represent UTC).**
`df["time"] = pd.to_datetime(df["open_time"], unit="ms")` produces tz-naive datetimes. Binance is UTC. When downstream code (indicators, microstructure, prophecy) mixes these with `datetime.now(UTC)` it either raises a comparison error or silently rounds wrong. Use `pd.to_datetime(..., unit="ms", utc=True)`.

**portfolio/data_collector.py:222-247: P1 (yfinance fallback strips signal of extended-hours flag inconsistently).**
`yfinance_klines` passes `prepost=True`. `price_source._fetch_yfinance` (price_source.py:146) does NOT pass `prepost`. Modules that route through `price_source` (market_health, macro_context, mstr) get RTH-only data; the breadth score's volume comparisons therefore drop pre-market institutional activity that the original yfinance path captured. Pick one and stick to it.

**portfolio/data_collector.py:316-344: P1 (cancel-on-timeout doesn't actually cancel running futures).**
On `as_completed` TimeoutError at line 334, the `for f in futures: f.cancel()` only cancels PENDING futures — running yfinance/Binance requests continue to completion and burn the ThreadPoolExecutor's worker threads. With `_TF_POOL_TIMEOUT = 60` (line 25) and 8 tickers × 7 timeframes, a single slow yfinance call can poison the pool across cycles. No `pool.shutdown(wait=False, cancel_futures=True)` either.

**portfolio/data_collector.py:280-313: P2 (cache key uses source_key not full source — cross-pollution risk).**
`cache_key = f"tf_{source_key}_{label}"` uses only the ticker symbol (e.g. `BTCUSDT`). If two different source dicts (`{"binance": "BTCUSDT"}` vs `{"binance_fapi": "BTCUSDT"}` — possible if a future ticker uses both) hit the same key, FAPI candles would get returned for a SPOT request. Currently BTC is only via SPOT so no collision, but it's a footgun for the next addition.

---

## portfolio/alpha_vantage.py

**portfolio/alpha_vantage.py:31-32, 161-168: P0 (daily budget not persisted across restarts).**
`_daily_budget_used` is a module-level int reset to 0 on every process start. Restart at noon → budget effectively becomes 50/day instead of 25 → quota exhaustion at AlphaVantage end → 429s → no fundamentals for the rest of the day. The 25/day free-tier cap is a HARD external limit; we have no visibility into it. Persist `{"date": ..., "used": ...}` to disk.

**portfolio/alpha_vantage.py:149-154: P1 (`Note` detection misses the new "Information" rate-limit field).**
AlphaVantage rolled out a second rate-limit message field "Information" (returns "API call frequency is..." instead of "Note"). The check `if isinstance(data, dict) and "Note" in data` ignores `Information`, so rate-limit responses get treated as success → 0 fields populated → cached as bogus "empty fundamentals" for 24h. Combine: `for key in ("Note", "Information"): ...`.

**portfolio/alpha_vantage.py:289-294: P1 (no breakout on circuit-breaker open).**
The `if not _cb.allow_request(): break` lives ONLY in exception handler and `if normalized is None` paths. But after a successful `_cb.record_failure()` from `raw is None`, the loop calls `_cb.allow_request()` — fine — but the next ticker proceeds to `_fetch_overview` even though the previous one failed. The check protects subsequent tickers; OK on paper. However a `_fetch_overview` that returns `None` from `resp.json()` parsing exception (line 146-147) returns silently → no `record_failure()` call → circuit never opens on JSON-parse storms.

---

## portfolio/earnings_calendar.py

**portfolio/earnings_calendar.py:49-52, 31-94: P0 (earnings AV calls bypass the daily 25-call budget).**
The function explicitly comments "earnings calls bypass alpha_vantage.py's `_daily_budget_used` counter." With STOCK_SYMBOLS sized at ~10 active tickers and 24h TTL, every fresh start can immediately burn the entire budget on earnings before fundamentals get a chance. Combined with the "budget not persisted" P0 above this is a recipe for silent AV blackout. Add a shared counter API and call it.

**portfolio/earnings_calendar.py:172-178: P1 (negative caching of `None` traps stale "no earnings" for 24h).**
On fetch failure, `data = _fetch_earnings_date(ticker)` returns `None`, then `_earnings_cache[ticker] = {"data": None, "time": now}`. Now `should_gate_earnings` returns False for a full 24h based on a transient network blip. If earnings really are in 2 days, the gate stays disabled — BUY signals fire into the binary event. Don't cache None.

---

## portfolio/fx_rates.py

**portfolio/fx_rates.py:67-71: P0 (hardcoded fallback 10.50 SEK never adjusts to spot regime).**
`FX_RATE_FALLBACK = 10.50` is from spring 2024-era SEK levels. Spot has been 10.85-11.30 in 2025-2026. Portfolios valued at hardcoded 10.50 understate USD-priced holdings by ~5-7%, which feeds into the equity-curve and Monte Carlo VaR. The `_fx_alert_telegram` warning fires but trading still proceeds on the bad number. Either widen FX_RATE_MIN/MAX or kill trades when no fresh rate is available.

**portfolio/fx_rates.py:33-34, 50-53: P1 (TOCTOU window between read and write).**
Reads cached_rate/time, releases lock, fetches, then re-acquires lock to write. Two threads can both pass the freshness gate (line 33) and both call `fetch_with_retry` → 2× requests to frankfurter (free tier with limits). Not catastrophic since the limit is 60/hour, but defeats the cache.

---

## portfolio/onchain_data.py

**portfolio/onchain_data.py:208-243: P1 (no quota tracking on BGeometrics 15/day budget).**
The module documents "8 req/hour, 15/day, budget 12/day" but enforces ONLY a 12h cache + 1s sleep between requests. Nothing increments a daily counter or short-circuits when budget is approaching exhaustion. Restart during a 12h-stale cycle = 6 fresh API calls. Two restarts/day = budget burned by lunch.

**portfolio/onchain_data.py:107-119: P1 (cache load uses `data.get("ts", 0)` but `_save_onchain_cache` does NOT set "ts" — only inside `_fetch_all_onchain` line 213).**
If a future code path writes via `_save_onchain_cache` without going through `_fetch_all_onchain` (e.g. external script), the cache file has no `ts` field and `_load_onchain_cache` treats it as ancient → forces refresh → budget burn. Add an explicit `ts` injection in `_save_onchain_cache`.

**portfolio/onchain_data.py:246-286: P2 (persistent cache seeding short-circuits via `_tool_cache` without TTL check).**
Line 269 seeds `_tool_cache["onchain_btc"]` with the persistent cache. The seed sets `"time": cache_ts`, so `_cached()` will see the data as already-old at seed time. Should be safe. But the seed bypasses `_loading_keys` so concurrent threads could still race into a fresh fetch. Minor.

---

## portfolio/sentiment.py

**portfolio/sentiment.py:329-338: P1 (subprocess fallback eats stdout JSON-parse failures).**
`_run_model` fallback: `return json.loads(proc.stdout)`. If the inference script prints anything (a `print`, a warning, a deprecation notice) to stdout in front of the JSON, `json.loads` raises and the WHOLE sentiment cycle returns nothing. The legacy subprocess scripts under `Q:\models\*_infer.py` are the only writers we control; an upstream `transformers` update could add a stdout warning. Capture and strip non-JSON noise.

**portfolio/sentiment.py:118-149: P2 (CryptoCompare API key never validated; silent 401 → Yahoo fallback always taken).**
On 401 (bad/expired API key), CryptoCompare returns `{"Response": "Error", "Message": "Authentication failed"}`. The code catches "Response == 'Error'" and falls back to Yahoo — fine — but never alerts that the API key is dead. Sentiment quality silently degrades. Telegram-warn when fallback is triggered N consecutive times.

**portfolio/sentiment.py:941: P2 (ab_key uses `now.isoformat()` — collisions possible at 60s cycle with multiple workers).**
`ab_key = f"{short}:{datetime.now(UTC).isoformat()}"` — with 8 ThreadPoolExecutor workers, two threads for the SAME ticker calling get_sentiment in the same microsecond would collide. Practically rare (the SENTIMENT_TTL deduplicates) but a `uuid.uuid4()` suffix is free insurance.

---

## portfolio/social_sentiment.py

**portfolio/social_sentiment.py:32, 65: P1 (raw requests.get — no retry, no backoff, no rate-limit handling).**
Bypasses `fetch_with_retry`. Reddit aggressively rate-limits scrapers (429 with Retry-After header). On 429 the call raises in the caller; no exponential backoff means we hammer Reddit on every cycle, hastening a permanent IP ban from Reddit's edge cache. Route through `fetch_json`.

**portfolio/social_sentiment.py:14-19: P3 (subreddit list still includes removed tickers).**
PLTR/NVDA are no longer in TIER 1 instruments per CLAUDE.md (removed Apr 09), but social_sentiment still defines subreddits for them. Dead code.

---

## portfolio/fear_greed.py

**portfolio/fear_greed.py:126-171: P1 (VIX fetch bypasses `price_source` router).**
Uses `yfinance` directly. VIX is on the `_YFINANCE_LAST_RESORT` whitelist, so OK on paper, but means the circuit-breaker / rate-limit infrastructure doesn't apply. Route through `price_source.fetch_klines("^VIX", interval="1d", period="5d")` for consistency with the router contract that CLAUDE.md mandates.

**portfolio/fear_greed.py:155-165: P3 (VIX-to-F&G mapping has integer truncation discontinuities).**
The piecewise `int(...)` mapping creates step-jumps at VIX boundaries (20, 30, 40). Round, don't truncate.

---

## portfolio/futures_data.py

**portfolio/futures_data.py:42-55: P1 (drops `oi_usdt` from response).**
The docstring claims "Returns: {oi, oi_usdt, symbol, time}" but the function body only returns `oi`, `symbol`, `time` — `oi_usdt` is dropped. The Binance `/openInterest` endpoint doesn't return oi_usdt (only `/openInterestHist` does), so the docstring is wrong AND any downstream consumer expecting `oi_usdt` from `get_open_interest()` gets KeyError.

**portfolio/futures_data.py:27-30: P2 (rate limiter shared between two distinct Binance endpoints).**
`_binance_limiter` is the same one used by spot klines (600/min). FAPI and futures-data both bill against the same weight bucket per Binance docs but a single `_RateLimiter` doesn't model weight-based limits — futures kline calls cost 5 weight, openInterestHist costs 1, etc. Under load you can still get 429s before the rate limiter triggers.

---

## portfolio/funding_rate.py

**portfolio/funding_rate.py:44-49: P1 (asymmetric thresholds: SELL at +0.03%, BUY at -0.01%).**
`0.0003` for SELL but `-0.0001` for BUY. Normal funding sits around ±0.0001%. The asymmetry causes BUY (-0.01%) to trigger far more often than SELL (+0.03%) under symmetric noise distributions — embeds a structural long bias into the funding signal. Either symmetric (±0.0002) or document the rationale.

---

## portfolio/macro_context.py

**portfolio/macro_context.py:138-144: P1 (synthetic DXY constant 58.0 produces meaningless `value`).**
The docstring at line 136-139 admits "The constant 58.0 does NOT match real DXY levels (~99) — it is arbitrary." The `value` field is then passed to `_dxy_features_from_close` and returned in the result dict — anyone using `value` from the EURUSD-synth path gets a garbage absolute level. Downstream `signals/dxy_cross_asset.py` currently only reads `change_*_pct`, but the contract isn't enforced. Either omit `value` from the synth path or set it to None to fail loudly.

**portfolio/macro_context.py:360-392: P1 (FOMC date comparison string-based — fragile to format mismatch).**
`upcoming = [d for d in FOMC_DATES if d >= today]` works because both are `YYYY-MM-DD` ISO strings. But `FOMC_DATES_ISO` is built once at module import via list comprehension on `date.isoformat()` — fine. A more subtle issue: `is_meeting_day = today in FOMC_DATES` only matches the EXACT day; the `FOMC_DATES_ISO` list is the announcement+start day pairs from `fomc_dates.py:13-22` — both days are in the list, so this is correct. But `meetings_remaining = len(upcoming) // 2` assumes every meeting is a 2-day; if a single-day announcement is added (rare but possible per Fed schedule changes), the math is off-by-one.

---

## portfolio/econ_dates.py, fomc_dates.py

**portfolio/fomc_dates.py:25-34, portfolio/econ_dates.py:38-103: P1 (hard-coded calendar EXHAUSTS end of 2027).**
After Dec 8 2027 the lists are empty. `next_event()` returns None, `is_macro_window()` returns False — silent calendar starvation. By 2027-Q4 onwards, `recent_high_impact_events` returns [] for all queries, meaning the post-event blackout fires never, meaning sentiment/momentum signals stop being suppressed in the volatility hangover window. Production-affecting in ~18 months. Schedule an annual refresh or fetch from BLS/Fed JSON.

**portfolio/econ_dates.py:155-156, 180-181, 224-225, 272-273: P2 (release time hardcoded to 14:00 UTC).**
"assume 14:00 UTC release" — CPI/NFP/FOMC each release at DIFFERENT times (CPI 13:30 UTC; NFP 13:30 UTC; FOMC announcement 18:00 UTC during normal time, 19:00 UTC during DST). The hardcoded 14:00 means `hours_until` is off by 0.5-5h, causing the proximity gate to fire too early/late. The `is_macro_window` lookback/lookahead buffers (24h/72h) probably absorb this, but tight FOMC same-day windows misclassify.

---

## portfolio/crypto_macro_data.py

**portfolio/crypto_macro_data.py:108: P1 (`datetime.date.today()` uses local time — Deribit expiry off by a day near midnight).**
On a Windows host in CET, `date.today()` returns local-time date. Deribit expires in UTC. Right at midnight CET → midnight UTC, options expiring 28MAR26 still in OI on Deribit but `today()` already says 28MAR26, so the filter `if d and d >= now` could either include or exclude the same-day expiring contract inconsistently. Use `datetime.datetime.now(UTC).date()`.

**portfolio/crypto_macro_data.py:218-220: P2 (compute_gold_btc_ratio reads from `agent_summary_compact.json` — stale data risk).**
Reads BTC and Gold prices from a precomputed summary file, NOT live. CLAUDE.md mandates "Live prices first." If the loop's report writer crashed or skipped a cycle, this ratio uses N-cycles-old prices and trends fire on stale data. Either pull live or document the staleness.

---

## portfolio/data_refresh.py

**portfolio/data_refresh.py:25-48: P1 (pagination loop has no safety limit; broken end-time causes infinite loop).**
`while start_time < end_time` — if Binance returns `batch[-1][0] >= end_time` (off-by-one or duplicate), the next iteration starts at the same `start_time` and never makes progress. Add `max_iterations = days * 50` or break when `start_time` doesn't advance.

**portfolio/data_refresh.py:23-26: P3 (interval dict missing keys → KeyError on un-mapped intervals).**
`ms_per_candle = {"1h": ..., "4h": ..., "1d": ...}[interval]` — pass "5m" or "15m" and you get KeyError. Caller currently only invokes "1h", but a one-line robustness fix.

---

## portfolio/market_health.py

**portfolio/market_health.py:445-454: P2 (FTD state persisted as JSON — restart resets rally counters if file is corrupt).**
`atomic_write_json(_STATE_FILE, state_to_save)` is correct, but `load_json(_STATE_FILE, default={}).get("ftd_state")` returns None silently if the file is missing/corrupt. The FTD state machine then re-initializes from `prev_state=None`, possibly losing weeks of rally tracking. Telegram-warn on initial load failure.

---

## portfolio/sentiment_shadow_backfill.py

**portfolio/sentiment_shadow_backfill.py:126-141, 191-194: P1 (full-file `read_text()` on potentially-multi-MB JSONL).**
`outcomes_path.read_text(encoding="utf-8").splitlines()` loads the entire sentiment_ab_log.jsonl into memory. After months of run-time this can hit hundreds of MB. The reviewer claim "Three years' history is already there" makes this load O(n²) per backfill run (each invocation walks every line × every model row inside). Use `file_utils.load_jsonl` with streaming.

**portfolio/sentiment_shadow_backfill.py:118-120: P2 (TICKER_EXPAND missing recent tickers).**
The map has BTC/ETH/XAU/XAG/MSTR. If a new instrument is added (recently or in future), shadow backfill silently drops outcomes for it because price snapshots key by the long form. Defensive: log a warning when `_expand_ticker` returns the input unchanged AND the input isn't already a long form.

---

## portfolio/bert_sentiment.py

**portfolio/bert_sentiment.py:286-288: P3 (env-var opt-in for GPU but `BERT_SENTIMENT_USE_GPU=` empty string isn't excluded).**
`os.environ.get("BERT_SENTIMENT_USE_GPU", "").strip() in ("1", "true", "TRUE", "yes")` — fine. But `"True"` (Python capitalization) isn't in the list, so a user setting `BERT_SENTIMENT_USE_GPU=True` (most common Python convention) silently stays on CPU. Use case-insensitive compare.

**portfolio/bert_sentiment.py:208-212: P3 (tokenizer load doesn't apply `low_cpu_mem_usage` retry from line 244-251).**
The meta-tensor recovery retry only re-loads `model`, not `tokenizer`. If tokenizer ever lands in meta state (rare but documented), it stays broken. Symmetric recovery for tokenizer.

---

## portfolio/indicators.py

**portfolio/indicators.py:155-207: P1 (regime cache keyed on `is_crypto` but not on `horizon`).**
`compute_indicators(df, horizon="3h")` produces different RSI/MACD coefficients than the default, so a cached regime for the same `(close, atr_pct, ema9, ema21, rsi, adx)` tuple could be re-used across horizons. Inputs differ between horizons, so the key changes most of the time, but for borderline-stable signals it could return the wrong regime. Add `horizon` (or a coarser proxy) to the cache key.

**portfolio/indicators.py:44-47: P2 (zero-price detection returns None — kills cycle of valid candles).**
If a Binance maintenance/glitch produces ONE zero-price candle in the middle of a 100-candle window, the entire indicator computation returns None and the whole timeframe is lost. Better: drop the bad row, log a warning, continue with the rest.

---

## portfolio/news_keywords.py

**portfolio/news_keywords.py:139-159: P2 (score_headline doesn't normalize "rate hike"/"rate cut" Tier-2 phrases against the dictionary).**
The pre-compiled `_KEYWORD_PATTERNS` is sorted by length descending so "rate hike" matches before "hike" — but "hike" isn't in any dict, fine. However "rate" appears in NO dict, and a headline "Fed considers rate" doesn't match anything: false negative. Add "rate decision" / "fed rate" phrases.

---

## Summary

Total findings: 31

- **P0 (4):** Binance error JSON returned as candles; AV daily budget not persisted across restarts; AV earnings calls bypass budget; FX hardcoded 10.50 SEK is stale; OFI history double-recorded by `persist_state`.
- **P1 (16):** Binance tz-naive timestamps; yfinance prepost flag inconsistency; pool cancel doesn't cancel running futures; AV `Note`-only rate-limit detection; circuit-breaker not triggered on JSON-parse failures; earnings negative-caching; fx TOCTOU; BGeometrics no daily quota; onchain cache `ts` injection; subprocess JSON-parse fragility; raw requests.get to Reddit; VIX bypasses price_source router; FAPI oi_usdt drop; funding-rate asymmetric thresholds; DXY synth `value` is garbage; FOMC calendar exhausts EoY 2027; econ-dates 14:00 UTC hardcoding (when CPI/NFP are 13:30 and FOMC 18:00/19:00); Deribit `date.today()` local-time; backfill full-file read_text; regime cache missing horizon key; microstructure_state's read-side-effects double bug pair.
- **P2 (8):** depth_imbalance saturation; trade-through mid-price denominator; data_collector cache key collision risk; CryptoCompare key silent fallback; ab_key uniqueness; FAPI weight-based limits not modeled; FTD state file silent-corrupt; sentiment_shadow_backfill missing tickers; compute_gold_btc_ratio reads cached file; zero-price candle drops entire window; "rate" not in keyword dict.
- **P3 (3):** social_sentiment dead subreddits; trade-through approximation; bert_sentiment env-var case sensitivity; tokenizer retry path; data_refresh interval dict KeyError; F&G integer truncation.
