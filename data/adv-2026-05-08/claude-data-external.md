# Adversarial Review: data-external subsystem (2026-05-08)

[P1] portfolio/fx_rates.py:42-48
**FX bounds-check rejects out-of-range rate but returns it anyway.**
Problem: When Frankfurter returns an out-of-range rate, the code logs ERROR but
returns the rate, bypassing cache. Next cycle then triggers stale fallback. Bounds
rejection should refuse the value, not pass it through.
Fix: On out-of-range, raise to the caller and let the cache fallback handle it
explicitly; do not return invalid rate.

[P1] portfolio/sentiment.py:233-237
**Headline dedupe only on title, ignores source/date.**
Problem: Same headline from Reuters + AP + Bloomberg counts as 1, suppressing
`dissemination_mult`. Underestimates actual news intensity.
Fix: Dedupe on `(title, published_date)` tuple, keep source list for dissemination.

[P1] portfolio/data_collector.py:96, 157
**Kline timestamps timezone-naive on both Binance and Alpaca paths.**
Problem: Cross-source merges risk misalignment (Binance comes UTC, Alpaca local). A
silent off-by-an-hour at DST boundary corrupts indicator calculations.
Fix: Add `utc=True` to both `pd.to_datetime()` calls; assert tz-aware downstream.

[P1] portfolio/shared_state.py:95-103
**None results not cached; stale fallback returned asymmetrically.**
Problem: API returning `None` skips cache update; next call returns stale data without
re-fetching. Flaky APIs cause oscillation between fresh-fail and stale-return that
distorts signal stability.
Fix: Cache `None` with short TTL (e.g., 30s) to break the oscillation; or expose retry
metric so callers can decide.

[P1] portfolio/earnings_calendar.py:49-52
**Alpha Vantage earnings calls bypass the daily budget counter.**
Problem: Each ticker fetch hits the Alpha Vantage 25/day quota but the budget tracker
isn't incremented. 100+ tickers of earnings refresh = silent 4x quota overrun.
Fix: Wrap every AV request through the same `_daily_budget_used` accounting.

[P1] portfolio/sentiment.py:804
**Headline filter fallback ignores per-source noise profiles.**
Problem: CryptoCompare press-wire (high noise) and NewsAPI metals (pristine) share the
same fallback threshold. Crypto fires noise-driven sentiment more often.
Fix: Per-source threshold; tune crypto floor higher than metals.

[P1] portfolio/alpha_vantage.py:281
**Budget reset at midnight not under lock.**
Problem: `_budget_reset_date` checked outside lock; race between reset and increment
at midnight boundary can lose 1–5 calls of accounting.
Fix: Move the reset check inside the same critical section as the increment.

[P1] portfolio/sentiment.py:876-888
**FinGPT enqueue failure silently drops A/B entry.**
Problem: Exception only logged at DEBUG; downstream `shadow[]` ends up missing FinGPT
votes. Accuracy tracking sees fewer A/B comparisons than expected.
Fix: Promote to WARNING; emit a metric on dropped FinGPT entries.

[P1] portfolio/data_collector.py:296-299
**Raw DataFrame discarded when indicator computation returns None.**
Problem: Whole timeframe entry dropped. Downstream modules needing raw OHLCV (orderbook
context, volume profile) get nothing for that timeframe.
Fix: Keep `_df` even when indicators fail; emit warning so consumers know indicators
are missing but raw data is available.

[P1] portfolio/futures_data.py:22-24
**Open Interest cache TTL 300s too long for momentum signals.**
Problem: Binance OI updates every second; a 5-min-old snapshot can flip the sign of a
3h trade signal at the boundary.
Fix: Drop TTL to 30–60s; or version the cache with last-update-tick and let signal
caller override.

[P1] portfolio/onchain_data.py:112, 260
**Malformed timestamp coerces to 0.0 → cache miss every restart.**
Problem: Persisted cache stores ISO string ts in some path, float in another; parse
mismatch returns 0.0 (ancient), forcing BGeometrics 15/day budget burn on every
restart.
Fix: Pick one format (float epoch), validate on read, recover gracefully.

[P2] portfolio/macro_context.py:141-144
**EURUSD synth fallback for DXY returns meaningless `value`.**
Problem: Comment admits value is arbitrary. Downstream code that ever reads it as a
real DXY level will silently produce wrong z-scores or thresholds.
Fix: Set `value=None` and add `value_is_synth=True` flag so consumers must opt in.

[P2] portfolio/sentiment.py:757-758
**Trading-Hero-LLM has known permabull bias on crypto, no de-rating.**
Problem: Despite documented bias, no correction factor is applied to its votes; per-
asset accuracy gate may already mask this, but the input is biased.
Fix: Apply per-source bias correction during sentiment aggregation, or rotate to a
calibrated model for crypto.

## Summary

11 P1 + 2 P2. Themes: cache-staleness lying about freshness, rate-limit accounting
gaps, timezone-naive datetimes, and per-source noise profiles being applied
indiscriminately. Several issues are silent failures that pass tests but degrade
signal quality slowly.
