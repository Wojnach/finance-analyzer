# FGL Review — data-external

## P0: Critical (Wrong/stale/zero price silently fed to signals/trades)

- **portfolio/futures_data.py:61** — KeyError on malformed Binance FAPI response. When API returns `{... but missing "openInterest" key}`, line 61 crashes unguarded: `data["openInterest"]`. Exception propagates through _cached() as None (safe fallback), but if API returns partial data (e.g., {"symbol": "X", ...} without "openInterest"), the function silently fails. **Fix:** wrap in try/except: `"oi": float(data.get("openInterest", 0.0))` or check keys upfront with `if "openInterest" not in data: return None`.

- **portfolio/futures_data.py:87-89, 115-118, 144-147, 174-177, 202-204** — Same KeyError pattern in list comprehensions for `get_open_interest_history`, `get_long_short_ratio`, `get_top_trader_position_ratio`, `get_top_trader_account_ratio`, `get_funding_rate_history`. If any dict in the API response list is missing expected keys (e.g., `sumOpenInterest`), the comprehension raises KeyError, _cached() catches it, returns None, and the signal/report silently loses data for that ticker. **Fix:** Replace comprehensions with explicit try/except per item: `try: ... except KeyError: continue` or use `.get()` with explicit None checks.

## P1: Risk (Stale-as-live, swallowed failures returning bad data, rate-limit silent degrade)

- **portfolio/data_collector.py:289-290** — Cache hit returns data without staleness signal. While this is mitigated by the "Now" (15m) timeframe having TTL=0, longer timeframes (12h@300s, 2d@900s, etc.) return cached data without metadata. Consumers reading from `tfs` list do not know if a cached entry is stale or fresh. A long-running loop that misses several API calls could serve week-old cached 7d data without indication. **Fix:** Attach `_cached_age_sec` or `_source` metadata to every cache hit: `cached["data"]["_cached_age_sec"] = time.time() - cached["time"]`.

- **portfolio/onchain_data.py:286** — On successful token load, `_cached()` is called but if BGeometrics API returns partial metrics (e.g., {"ts": ..., "mvrv": 1.5} missing "sopr"/"nupl"), the fetcher aggregates whatever succeeded. No error is raised, but subsequent interpret_onchain() calls see incomplete data. If 5 of 6 metrics fail, the signal still fires with 1 metric, potentially giving false confidence. **Fix:** Track success count in _fetch_all_onchain and return None if fewer than 4 of 6 metrics succeeded (majority gate).

- **portfolio/alpha_vantage.py:149-152** — Rate-limit exhaustion is detected but logged as WARNING only. Once the 25/day budget is hit, refresh_fundamentals_batch() logs "budget exhausted" and returns 0. Subsequent loops attempt 0 refreshes quietly without notifying operators. If a multi-day outage occurs and stock signals depend on aged fundamentals, no alert surfaces until the 5-day cache TTL expires. **Fix:** Send Telegram alert the first time daily budget is exhausted: `logger.error("Alpha Vantage daily budget exhausted, no refreshes possible")`; escalate to critical_errors.jsonl if persists.

- **portfolio/fx_rates.py:44-55** — FX rate falls back to cached/hardcoded silently on Frankfurter API failure. If the rate diverges 10% (e.g., SEK weakens from 10.5 to 9.45), and the API is down for 4 hours, portfolio valuations will be off by 10% using the hardcoded fallback. The function logs ERROR but continues, and P&L calculations blindly use the stale rate. **Fix:** No code change needed (already addressed with ERROR logging), but Telegram alert (already in place via _fx_alert_telegram) should fire immediately rather than on 4h+ staleness.

- **portfolio/fear_greed.py:96-123** — `get_crypto_fear_greed()` is defensive and returns None on empty/malformed data, but consumers in signal_engine.py (line 3349) cache the result via _cached(). A None result is not cached (by design in shared_state.py:94-95), which forces a re-fetch every 30s. However, if the API returns `{"data": []}` (maintenance), the function returns None cleanly, but the 60s signal engine cycle will retry fetch every 60s, burning no budget. **Status:** Working as designed. No issue.

- **portfolio/price_source.py:239-257** — Fallback from primary to yfinance on Binance/Alpaca failure is logged and marked with `df.attrs["_source"] = "yfinance_fallback"`. Consumers should check this. However, the attrs are only present on the returned DataFrame; if a consumer accesses a cached entry via _fetch_one_timeframe (data_collector.py:290), the entry is a dict of indicators, not a DataFrame, and the "_source" metadata is lost. **Fix:** Attach _source metadata at the indicator-level cache entry, not just the DataFrame.

## P2: Edge Case / Race Condition

- **portfolio/microstructure_state.py:147-151** — Flow acceleration is computed as `fast_per_snap - slow_per_snap` (normalized by snapshot counts). During warmup (< 5 snapshots), `n < _OFI_WINDOW_FAST`, so both fast and slow are set to the same value (ofi_medium or ofi_slow fallback), and flow_acceleration=0. However, line 147-148 uses `ofi_fast / max(_OFI_WINDOW_FAST - 1, 1)` which assumes _OFI_WINDOW_FAST >= 2 (true: it's 5). The fallback is correct. **Status:** No issue.

- **portfolio/microstructure.py:94** — VPIN computation uses `bucket_vol >= bucket_size - 1e-12` as the bucket-full threshold, a floating-point epsilon guard. If trades have qty values < 1e-12, buckets may never fill and the function returns an empty imbalances list, then None (line 102-103). For most crypto/metals trades (qty >> 1e-12), this is safe. **Status:** Edge case only; no issue for production data.

## P3: Maintainability / Consistency

- **portfolio/futures_data.py:60-64, 72, 88, 117, 146, 175** — Inconsistent field-access patterns. Some use `.get()` with defaults (line 63: `data.get("time", ...)`), others use bare bracket access (line 61: `data["openInterest"]`). The mix makes it easy to miss guards. **Fix:** Standardize: use `.get()` everywhere with explicit None/0 defaults.

- **portfolio/onchain_data.py:144-146, 154, 162, 170, 179, 189-191** — Similar inconsistency: `_fetch_mvrv` returns `data.get("mvrv")` (may be None), but signal_engine.py (line 3496+) consumes the result without null checks. If BGeometrics API response has `{"mvrvZScore": 1.5, ...}` but no "mvrv" key, interpret_onchain() receives {"mvrv": None, "mvrv_zscore": 1.5}, and the zone logic at line 304 checks `zscore is not None` but not `mvrv is not None`. **Fix:** Return None from _fetch_mvrv if critical fields are missing: `if "mvrv" not in data or "mvrvZScore" not in data: return None`.

---

## Summary

**Total findings:** 8 (2 P0, 5 P1, 1 P2 edge-case noted as safe)

**Top issue:** Unguarded KeyError in futures_data.py list comprehensions (lines 87-206) will silently degrade Binance derivatives feeds when API returns partial/malformed data. Propagates through _cached() as None, but the loop layer doesn't know a metric is unavailable vs. just unleveraged. Recommend immediate: add try/except per-item or .get() with None checks.

**Secondary:** data_collector.py cached timeframes (TTL > 0) return data without age metadata, and FX rate fallback to hardcoded 10.50 during API outages can silently inflate portfolio valuations by 10%+ — both require metadata attachment or alerting to surface to operators.

**Not an issue:** fear_greed, price_source fallback, microstructure edge cases all have working guards or are designed correctly.
