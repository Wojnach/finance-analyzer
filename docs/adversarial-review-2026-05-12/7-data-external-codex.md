# Codex adversarial review: data-external
## Summary
8 material issues. The main blocker is `price_source.py` silently swapping live exchange data for delayed/unnormalized Yahoo data on primary-source failure. High-severity issues also exist in macro-event timing, metals headline normalization, microstructure restart behavior, and stale FX handling. Medium issues are concentrated in request-budget accounting and retry bypasses.

## P0 — Blockers
- `portfolio/price_source.py:223` — `fetch_klines()` unconditionally falls back to `_fetch_yfinance()` after any Binance/Alpaca failure and returns the fallback frame with no provenance marker. For aliases the router explicitly advertises as valid (`XAGUSDT`, `XAUUSDT`, `BTCUSDT`, `ETHUSDT` at lines 37-50 and 197-198), the fallback reuses the raw Binance symbol as the Yahoo ticker (lines 235 and 146-149), so a primary outage can silently degrade from live exchange data to delayed or unsupported/wrong-symbol data while callers keep treating it as canonical market data.

## P1 — High
- `portfolio/econ_dates.py:155` — all blackout helpers hard-code event time to `14:00 UTC` (`next_event`, `events_within_hours`, `recent_high_impact_events`, `is_macro_window` at lines 155-159, 180-183, 224-226, 272-275). That is wrong for CPI/NFP/GDP (08:30 ET), wrong for FOMC (14:00 ET), and wrong across US DST changes. The gating window moves by 1.5 to 7 hours.
- `portfolio/sentiment.py:771` — non-crypto tickers are normalized by stripping `-USD` (`XAG-USD` -> `XAG`, `XAU-USD` -> `XAU`) and passed to `_fetch_stock_headlines()` (lines 771-779). The Yahoo fallback then does `yf.Ticker(ticker)` on that short code (lines 227-233 and 131-135), bypassing the repo’s canonical mappings. For metals this can resolve to the wrong Yahoo symbol or no proper metals news feed at all.
- `portfolio/microstructure_state.py:36` — restart persistence is incomplete. The real rolling inputs live only in in-memory `deque`s (`_snapshot_buffers`, `_spread_buffers`, `_ofi_history` at lines 36-42), but `persist_state()` writes only derived aggregates (lines 205-214). After restart, `load_persisted_state()` can expose old aggregates for 2 minutes (lines 216-229), but producer-side OFI/spread history is gone and z-scores cold-start until enough fresh snapshots accumulate.
- `portfolio/fx_rates.py:56` — stale USD/SEK is accepted indefinitely. After fetch failure, the module returns the cached rate regardless of age (lines 56-65); after 2 hours it only warns. There is no max-age cutoff before valuations/trading continue on arbitrarily old FX.

## P2 — Medium
- `portfolio/alpha_vantage.py:31` — AV daily-budget tracking is process-local and success-counted, not request-counted. `_daily_budget_used` / `_budget_reset_date` live only in RAM (lines 31-32, 157-168), so restarts and multi-process use reset the budget. The counter is incremented only after a successful normalized response (lines 264-281), so failed/empty/rate-limited requests still burn provider quota without moving the local budget.
- `portfolio/sentiment.py:184` — NewsAPI tracking undercounts quota. `_fetch_newsapi_with_tracking()` only calls `newsapi_track_call()` when `result` is truthy (lines 184-193). Empty-but-successful searches still consume NewsAPI quota.
- `portfolio/earnings_calendar.py:49` — this module explicitly bypasses Alpha Vantage budget accounting for the `EARNINGS` endpoint (lines 49-53). Current blast radius is small because `STOCK_SYMBOLS` is just `MSTR`, but the subsystem’s “25/day” contract is already false.
- `portfolio/social_sentiment.py:29` — Reddit fetches bypass `http_retry.py` entirely and use `urllib.request.urlopen()` directly (lines 29-33 and 59-66). No centralized backoff, no 429 handling, no shared retry discipline.

## P3 — Low
- None.

## Tests missing
- Restart/multi-process tests for Alpha Vantage and NewsAPI quota accounting.
- DST boundary tests for `econ_dates.py` around US DST start/end with event-specific release times.
- Ticker-normalization tests for `sentiment.py` covering `XAG-USD`, `XAU-USD`, `SI=F`, and `GC=F`.
- Failover tests for `price_source.py` that force primary failure and assert fallback provenance plus alias remapping.
- Restart tests for `microstructure_state.py` proving OFI/spread rolling state survives process restart.
- Max-staleness tests for `fx_rates.py`.