# Adversarial review — data-external subsystem (2026-05-29-fgl)

Scope: market-data collectors, external API clients, precompute caches.
Reviewed at HEAD. 26 files. Focus: stale-data-served-as-fresh, silent API
failures, wrong source/unit, rate-limit handling, atomic I/O, validation.

## Counts
- P0: 0
- P1: 3
- P2: 7
- P3: 5

Overall the subsystem is in good shape. The shared `_cached()` primitive caps
staleness at `ttl * _MAX_STALE_FACTOR` and returns None beyond that (good), the
price-source router keeps yfinance to a documented last-resort list with ERROR
logs on primary-source fallback, FX has explicit staleness alerting, and the
hot Binance/Alpaca kline paths use circuit breakers + http_retry (429 handled).
No `10m` Binance interval usage anywhere. The findings below are mostly
data-quality / robustness issues on secondary (precompute / sentiment / macro)
paths, not the live Tier-1 price path.

---

## P1 — incorrect unit/source, missing staleness guard, swallowed failure

- portfolio/macro_context.py:297: P1: `_fred_10y_fallback` hardcodes
  `"change_5d": 0.0` when the yfinance `^TNX` path is down and FRED is used.
  The 2s10s curve classification (lines 343-351) and any consumer reading
  `change_5d` then sees a fabricated "no change" for the 10y leg during the
  exact outage windows this fallback exists for (the docstring cites 16h ^TNX
  staleness). A flat 0 can mask a real yield move feeding the treasury/curve
  signal. → Compute 5d change from a second FRED observation
  (`fetch_us10y` history / DGS10 5 obs ago) or mark the field `None` so
  downstream treats it as missing rather than "unchanged".

- portfolio/mstr_precompute.py:142-162: P1: MSTR price + history fetched
  directly from yfinance (`yf.Ticker("MSTR").history`), bypassing
  `price_source`/Alpaca entirely. CLAUDE.md mandates MSTR via Alpaca IEX and
  "live prices first"; yfinance lags 10-15 min and is the documented last
  resort. The BTC leg (line 162) has the same issue. Even though this is a 4h
  shadow-phase informational precompute, it silently establishes a 15-min-stale
  MSTR/BTC price with no staleness flag in `mstr_deep_context.json`. → Route
  through `portfolio.price_source.fetch_klines("MSTR", ...)` (Alpaca primary,
  yfinance only on documented fallback) so the freshness contract matches the
  rest of the system.

- portfolio/sentiment.py:211-212 (and 200-222): P1: `_fetch_newsapi_headlines`
  returns `[]` for *any* `fetch_json` failure, which includes a 429
  rate-limit / auth failure, indistinguishable from a legitimately empty result.
  Combined with `_fetch_newsapi_with_tracking` (line 233, `if result:`), a
  rate-limited/失败 call neither surfaces the quota exhaustion nor counts
  against the 100/day budget — the news signal silently goes dark with no
  WARNING. → Inspect the response status in `fetch_json` path (or return a
  sentinel) and log a WARNING when NewsAPI returns 429/401 so quota exhaustion
  is observable.

---

## P2 — robustness

- portfolio/crypto_precompute.py:185 & 195: P2: `float(r.json().get("lastFundingRate", 0))`
  and `float(r.json().get("openInterest", 0))` substitute `0.0` when the field
  is absent (partial/changed Binance schema), writing a plausible-but-fake
  neutral funding rate / zero OI into `crypto_deep_context.json` instead of
  `None`. Downstream cannot distinguish "funding is 0" from "field missing".
  → Use `.get(...)` without the `0` default and gate on `is not None` before
  `float()`, mirroring `funding_rate.py:31-39`.

- portfolio/crypto_precompute.py:159-234: P2: All Binance / Binance-FAPI /
  CoinGecko fetches use raw `requests.get` with no circuit breaker, no shared
  rate limiter (`_binance_limiter`), and no `http_retry` backoff — unlike the
  rest of the subsystem. On a Binance hiccup this can contribute to retry
  pressure and ignores the spot/FAPI circuit breakers the loop relies on.
  → Route through `data_collector` / `futures_data` helpers or at least
  `fetch_json` + `_binance_limiter`.

- portfolio/data_collector.py:94-98 (and 155-157, 242-244): P2: OHLCV columns
  are cast to float but never validated for negative / zero / NaN close or
  zero volume before flowing into `compute_indicators` and signals. A bad
  upstream row (e.g. a 0 or negative print) propagates silently into RSI/BB/
  momentum. → Add a sanity check (close > 0, no NaN in OHLC) and drop/raise on
  violation so corrupt prints don't reach the voting engine.

- portfolio/onchain_data.py:280-283: P2: When the BGeometrics token is missing,
  `_load_onchain_cache(max_age_seconds=ONCHAIN_TTL*2)` returns up to 24h-old
  on-chain data at DEBUG level with no staleness marker in the returned dict.
  The BTC on-chain voter then treats 24h-old MVRV/SOPR/NUPL as current.
  → Stamp an `age_seconds`/`stale` flag onto the returned dict (or log WARNING)
  so the consumer/interpretation layer can down-weight clearly old data.

- portfolio/social_sentiment.py:32,65: P2: Reddit fetch uses raw `requests.get`
  with no `http_retry`/circuit breaker, and errors are emitted via `print()`
  (lines 110,122) rather than the logger — invisible to log-based monitoring.
  → Use `fetch_json` and `logging.getLogger(...).warning`.

- portfolio/earnings_calendar.py:48-52: P2: Alpha Vantage EARNINGS calls
  consume a 25/day AV request each but bypass `alpha_vantage._daily_budget_used`
  (acknowledged in the comment). Under many stock tickers this can silently
  exhaust the AV quota that `refresh_fundamentals_batch` depends on, with no
  shared accounting. → Export an increment hook from `alpha_vantage.py` and
  count earnings calls against the same daily budget.

- portfolio/crypto_precompute.py:210 (DX-Y.NYB/SPY/GLD) & metals seed: P2:
  `gold_close` is sourced from the GLD ETF (~1/10 of spot XAU). Only
  `change_pct` is currently consumed, so it is presently safe, but the field
  is named `gold`/`close` in the shared context with no note that the absolute
  level is ETF-scaled — a future consumer reading `shared.gold.close` as a
  spot gold price would be off by ~10x. → Rename to `gld_close` or annotate the
  scale in the schema.

---

## P3 — nits

- portfolio/price_source.py:343-351 / macro_context.py:343: P3: 2s10s spread
  mixes `^TNX` (yfinance, percent) with `2YY=F` (yield-futures pseudo-ticker).
  Both are in percent units so the math is correct today, but the mixed-source
  assumption is undocumented and brittle if `2YY=F` quoting ever changes.
  → Add a comment asserting both legs are percent-yield.

- portfolio/metals_precompute.py:1058-1209: P3: `_load_silver_seed_research`
  / `_load_gold_seed_research` embed static analyst targets dated 2026-03-14/15.
  These are clearly research context (not prices), but there is no max-age
  warning if the external cache is never refreshed — the seed can persist
  indefinitely as "current". → Log when the seed (not the live cache) is used.

- portfolio/crypto_precompute.py:166: P3: `float(d.get("lastPrice", 0)) or None`
  converts a legitimately-zero (impossible for BTC/ETH but pattern-wise) price
  to None via truthiness; harmless here but the `or None` idiom would also nuke
  a real 0. → Prefer explicit None checks.

- portfolio/data_refresh.py:41: P3: `download_klines` `break`s on a single
  `fetch_with_retry` None and silently returns a partial/empty DataFrame which
  is then written to the feather cache (line 85) with no row-count guard. This
  is an offline backfill utility, not the live path, but a truncated history
  file could later mislead a backtest. → Guard `len(df)` before `to_feather`.

- portfolio/fear_greed.py:174-177: P3: `get_fear_greed(ticker=None)` returns
  the crypto F&G for a None ticker; callers passing an unknown stock-like
  ticker that doesn't end in `-USD` will route to the VIX-derived path which is
  fine, but there is no logging when a ticker falls through classification.
  → Minor; log at DEBUG on unrecognized ticker.
