# Adversarial review — data-external subsystem (2026-06-06)

Scope: market-data fetch + external API modules. Reviewed as a fresh PR (full current state).

## P0 — stale-as-live / loop-stall / secret leak

portfolio/http_retry.py:53-62: `retry_after` from a 429 body is used verbatim as sleep, then `wait += random.uniform(0, wait)` is applied on top (line 59), and there is NO upper bound. A server (or spoofed/garbage `parameters.retry_after`) returning e.g. 600 makes one retry sleep 600-1200s, blocking the calling worker far past the 60s cycle = loop stall. fetch_with_retry has no total-time budget; with retries=3 and backoff doubling, worst case is minutes even on well-behaved 429s. → Clamp every computed `wait` to a hard ceiling (e.g. `min(wait, MAX_BACKOFF)` with MAX_BACKOFF ~10-20s) and cap cumulative sleep across attempts; validate `retry_after` is a sane positive number before use.

portfolio/metals_cross_assets.py:225,243,267,285 (and 116,139,170,188): all intraday + daily cross-asset fetchers pass `interval="60m"`. `fetch_klines` runs `_binance_interval("60m") → "1h"` only on the Binance path; HG=F/CL=F/SI=F/GC=F/^GVZ/SPY route to yfinance/Alpaca where `60m` is used directly — but if the Binance primary fails for SI=F/GC=F and the emergency fallback in price_source.fetch_klines fires, it calls `_fetch_yfinance(..., "60m")` which yfinance accepts, OK. Not a stall, but note GC=F/SI=F now resolve to Binance FAPI (XAU/XAG perps) in `_BINANCE_FAPI`, so `get_gold_silver_ratio_intraday` silently fetches *perp* gold/silver, not COMEX futures — a wrong-instrument substitution for a ratio signal that was calibrated on futures. → confirm perp-vs-futures basis is acceptable for the G/S ratio z-score; if not, pin these to yfinance.

portfolio/data_collector.py:74-101 + shared_state.py:37-125: on a fetch exception inside a `_cached`-wrapped fetcher, `_cached` (line 109-125) silently returns the prior cached value (refreshing its `time` to `now - ttl + _RETRY_COOLDOWN`) with NO staleness marker propagated to the caller. Futures/funding/onchain/cross-asset/treasury/dxy all flow through `_cached`. A consumer cannot distinguish a fresh quote from data served after a live-fetch failure — direct violation of "LIVE PRICES FIRST". The bound (`ttl * _MAX_STALE_FACTOR`) limits age but the served value is still presented as live. → attach a `_stale`/`_as_of` marker to cached payloads on the error path (as price_source already does with `df.attrs["_source"]`) so downstream signal/decision code can suppress or discount.

## P1 — quota burn / fail-open / wrong-instrument

portfolio/earnings_calendar.py:48-61: AV EARNINGS calls go through `_alpha_vantage_limiter.wait()` (5/min) but DELIBERATELY bypass the 25/day `_daily_budget_used` counter in alpha_vantage.py (acknowledged in the line 49-52 comment). With STOCK_SYMBOLS earnings refreshed every 24h plus `refresh_fundamentals_batch` both drawing on the same undocumented 25/day key, earnings fetches can silently push total daily AV usage over 25, causing OVERVIEW refreshes to start failing with rate-limit Notes. → route earnings through a shared AV budget counter (export an increment fn from alpha_vantage.py) so both consumers debit one quota.

portfolio/shared_state.py:331-344 + sentiment.py:233: NewsAPI quota is only debited via `newsapi_track_call()` when a fetch returns non-empty (`if result:` at sentiment.py:233). A NewsAPI 200-with-empty-articles response, or any path where `_fetch_newsapi_with_tracking` returns `[]`, consumes a real API call but is NOT counted. Under the 90/day budget this under-counts and can overrun the true 100/day free cap. The H9 comment frames this as intentional ("only count when we got data") but the failure mode is quota overrun, not the spurious-exhaustion it claims to prevent. → count every HTTP request that actually hit NewsAPI (move the debit into `_fetch_newsapi_headlines` after the request returns, regardless of article count).

portfolio/fx_rates.py:33-72: cache returned on the happy path (line 33) is the last good rate up to 15 min old — fine. But on live-fetch failure (line 61-72) the stale cached rate is returned with only a log warning + throttled Telegram; callers (risk_management, monte_carlo_risk, portfolio valuation) receive a number indistinguishable from live. A 2h+ stale SEK rate silently mis-values the whole SEK portfolio. The Telegram alert is the only signal and it is 4h-cooldown'd. → return the rate with an age/stale flag, or have callers query `_fx_cache["time"]`; at minimum surface staleness in agent_summary.

portfolio/social_sentiment.py:32,65: raw `requests.get` (timeout=10 — OK) with `resp.raise_for_status()` then `resp.json()`; no retry/backoff and `print()` instead of logger on error (lines 110,121-122). Reddit 429s are common and will throw, caught only by the broad `except` that prints to stdout. Not a stall (timeout present) but quota/robustness: no shared rate-limit on reddit.com across 8 workers. → route through `fetch_json`/`http_retry` and a limiter; replace `print` with `logger`.

## P2 — robustness / edge

portfolio/onchain_data.py:277-286: when no token is configured, `_load_onchain_cache(max_age_seconds=ONCHAIN_TTL*2)` (24h) returns and serves cache up to 24h old as a live result with only a DEBUG log. On-chain zones (mvrv/sopr/nupl) drive the BTC-only voter; 24h-old MVRV during a fast move is materially wrong but presented as current. → mark as stale in the returned dict.

portfolio/data_collector.py:347-351: on timeframe-pool timeout the code logs stuck labels then calls `f.cancel()` on already-running futures (which cannot cancel a running thread) and proceeds with partial `raw_results`. Correct by design (documented OR-I-001) but the stuck worker thread leaks and keeps holding a Binance/Alpaca limiter slot until its own socket timeout — under repeated upstream hangs this slowly starves the shared limiters. → acceptable given fetch_with_retry's bounded socket timeout, but consider a hard per-fetch deadline.

portfolio/crypto_macro_data.py:275,397: `_load_ratio_history`/`_load_netflow_history` read JSONL with raw `open(...)` + line-by-line `json.loads`, not the project's atomic IO helpers. Tolerant of corruption (per-line try/except) so low risk, but violates CLAUDE.md rule #4 (atomic IO only). → use file_utils helpers for consistency.

portfolio/futures_data.py:62-64,87-91,114-121: list-comprehension parsers do `float(d["sumOpenInterest"])`, `float(d["longShortRatio"])`, `d["timestamp"]` with bracket access. Binance futures/data endpoints are stable, but a partial/schema-changed row raises KeyError/ValueError inside the comprehension, killing the whole batch for that cycle (the funding_rate.py:31-39 path was already hardened against exactly this; futures_data was not). → mirror the `.get()` + skip-row guard used in funding_rate.py.

portfolio/sentiment.py:329-335: subprocess fallback in `_run_model` has `timeout=120` — a 120s blocking call inside a ticker worker. Bounded (won't hang forever) but 120s >> 60s cycle; if the in-process BERT path is broken across all 8 workers the cycle blows its budget every time. → lower subprocess timeout or gate it behind the pool timeout.

## P3 — maintainability

portfolio/macro_context.py:143: EURUSD→DXY synth constant `58.0` is arbitrary and `value` field is meaningless (documented), relying on every downstream consumer reading only the change fields. Fragile coupling. → emit `value: None` on the synth path so a future consumer can't accidentally trust it.

portfolio/data_collector.py:49: TIMEFRAMES "1mo" uses Binance interval `"3d"` (valid) — no `10m` usage found anywhere in the subsystem; the documented Binance `10m`→`5m` invariant is respected. (no action)

portfolio/api_utils.py:33-35: `load_config` swallows all exceptions and re-raises only if no cache exists; a corrupted config edit silently keeps serving the stale in-memory config across a key rotation. Low risk for a symlinked external file but worth a WARNING log on the swallowed-exception branch.

## Risk summary

The most dangerous gaps are silent stale-as-live serving via `_cached` on the fetch-error path and the unbounded `retry_after` sleep in http_retry, either of which can put old prices or a stalled worker into the 60s decision loop without any caller-visible signal. Quota accounting for Alpha Vantage (earnings bypass) and NewsAPI (empty-response under-count) can quietly overrun the tight free-tier daily caps, degrading the news/fundamentals voters mid-session; no hardcoded or logged secrets were found.
