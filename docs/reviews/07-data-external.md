# Data-External Review

Adversarial read-only review (caveman:cavecrew-reviewer) of the market-data
ingestion subsystem in worktree `Q:/fa-rev-0531`. One line per finding.
**Totals: 1 P0, 9 P1, 2 P2.**

## P0
- `portfolio/fx_rates.py:44` — P0 wrong-data: FX rate sanity check [7.0,15.0] SEK/USD fails through WITHOUT updating cache; caller gets None and on subsequent calls an *old cached rate* is returned anyway (line ~65). A persistently bad upstream → every SEK conversion silently uses a stale rate. → Cache the bad rate with a `_bad_data` flag, or raise so `fetch_with_retry` logs it; never silently serve stale FX.

## P1
- `portfolio/price_source.py:252` — P1 staleness: yfinance fallback returns a DataFrame with no freshness marker; downstream cannot detect a price stale by >10m from a failed Binance/Alpaca. → Attach `_age_seconds`/`_source` and have decision code reject stale.
- `portfolio/fear_greed.py:115` — P1 wrong-data: `data["value"]` indexed after only an isinstance(dict) check; empty inner field → KeyError swallowed by the try at ~113. → `data.get("value")` + None-check before int().
- `portfolio/onchain_data.py:114` — P1 staleness: `_load_onchain_cache()` returns full dict with no age flag; consumer can't tell fresh-5min from 11.9h-just-under-TTL. → add `_age_seconds`.
- `portfolio/alpha_vantage.py:95` — P1 budget race: `r.json()` on a 429 rate-limit HTML/JSON page; `_check_budget()` not incremented but fetch retried → burns the 25/day budget untracked. → check `r.status_code` before `r.json()`, increment on 429.
- `portfolio/metals_orderbook.py:63` — P1 silent failure: empty depth `{"bids":[],"asks":[]}` parses to 0.0 imbalance instead of None → feeds neutral to microstructure signal. → return None when either side empty.
- `portfolio/metals_cross_assets.py:58` — P1 wrong-data: `_yf_download()` ignores `_primary_failed`/`_source=yfinance_fallback` attrs → can mix ~655s-stale yfinance with fresh Binance FAPI silently. → warn/return None on fallback attrs.
- `portfolio/funding_rate.py:31` — P1 silent failure: missing `lastFundingRate`/`markPrice` → None, never cached → re-fetch storm then stale; funding_rate signal silently drops from consensus for N cycles. → log ERROR on missing fields.
- `portfolio/onchain_data.py:232` — P1 silent failure: blanket `_safe_float` over all fields caches None for failed conversions (e.g. netflow); downstream signal may not None-check. → return None / don't cache partial.
- `portfolio/data_collector.py:88` — P1 silent failure: Binance error response `{"code":-1000,...}` is a dict not a list → `r.json()` succeeds, then `pd.DataFrame(data, columns=...)` TypeError. → check `code`/shape before building frame.
- `portfolio/sentiment.py:136` — P1 staleness: reused `yf.Ticker("^VIX")` internal cache may serve >1h-old VIX with no TTL enforcement. → fresh Ticker per call or log bar timestamp.

## P2
- `portfolio/microstructure_state.py:123` — P2 concurrency: `get_ofi_zscore()` can include the just-appended current value (contradicts the "WITHOUT current value" comment at ~107) if called after `record_ofi()` from another thread → z-score self-contaminates. → snapshot history excluding current, or call zscore before record.
- `portfolio/crypto_macro_data.py:223` — P2: `_load_ratio_history()` uses raw `open()` not `load_json()` → corruption risk on a racing persist write. → use `load_json()`.

## Cross-cutting theme
A recurring **"stale-but-silent" pattern**: caches/fallbacks return last-known
data without a freshness flag, violating the live-prices-first invariant. The
fix is uniform: every cache/fallback return should carry an age/source marker
and decision-time code should reject stale.
