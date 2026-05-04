# Codex Review — 7-data-external

## Summary

The patch introduces multiple correctness problems in core data-generation paths. In particular, metals precompute drops previously fetched live data on routine runs, the timeframe timeout does not actually prevent hangs, and the crypto options max-pain metric is computed incorrectly.

Full review comments:

- [P1] Preserve prior market data when a source skips refresh — Q:\fa-review\portfolio\metals_precompute.py:137-145
  `_fetch_market_data()` initializes every payload slot to `None` and only fills it inside `_should_refresh(...)`. Because `maybe_precompute_metals()` runs every 2 hours while most sources refresh every 4 hours, 24 hours, or 7 days, the intermediate runs rewrite `silver_deep_context.json`/`gold_deep_context.json` with missing futures/ETF/COT/FRED overlays instead of carrying forward the last successful values. On the default schedule the deep-context files will oscillate between complete and mostly empty every other cycle.

- [P1] Make the timeframe timeout actually stop stuck workers — Q:\fa-review\portfolio\data_collector.py:334-339
  If `as_completed(..., timeout=...)` raises here, the code is still inside a `with ThreadPoolExecutor(...)` block. Calling `f.cancel()` does not stop futures that are already running, and exiting the `with` immediately calls `shutdown(wait=True)`, so a hung fetch (for example a yfinance call that never returns) will still block `collect_timeframes()` until that worker finishes. In other words BUG-179 does not actually cap the hang it is meant to prevent.

- [P2] Initialize max-pain search for a minimum, not -1 — Q:\fa-review\portfolio\crypto_macro_data.py:137-137
  `total_pain` is always non-negative, so seeding `max_pain_value` with `-1` makes the `total_pain < max_pain_value` check below false for every candidate after the first one. For any expiry with multiple strikes, `max_pain_strike` stays pinned to the first entry in `all_strikes`, so every caller of `get_deribit_options()` gets a bogus max-pain level.

- [P2] Pull futures backfill from the FAPI endpoint — Q:\fa-review\portfolio\data_refresh.py:30-31
  `download_klines()` writes into `.../binance/futures` and the output filenames are `*-futures.feather`, but this request goes to the spot `BINANCE_BASE` endpoint. Any downstream consumer treating these files as futures bars will silently mix spot candles with futures-only metrics such as funding/open interest, which makes the backfill inconsistent.

- [P2] Use per-event release times instead of 14:00 UTC — Q:\fa-review\portfolio\econ_dates.py:155-156
  This hardcodes every macro event to 14:00 UTC, but the calendar mixes 08:30 ET releases (CPI/NFP/GDP) with 14:00 ET FOMC decisions. Around event days `next_event()`, `events_within_hours()` and `is_macro_window()` will therefore open or close their blackout windows several hours early or late; on FOMC days the error is 4-5 hours. Using one UTC anchor defeats the proximity checks these helpers are supposed to provide.
The patch introduces multiple correctness problems in core data-generation paths. In particular, metals precompute drops previously fetched live data on routine runs, the timeframe timeout does not actually prevent hangs, and the crypto options max-pain metric is computed incorrectly.

## Full review comments

- [P1] Preserve prior market data when a source skips refresh — Q:\fa-review\portfolio\metals_precompute.py:137-145
  `_fetch_market_data()` initializes every payload slot to `None` and only fills it inside `_should_refresh(...)`. Because `maybe_precompute_metals()` runs every 2 hours while most sources refresh every 4 hours, 24 hours, or 7 days, the intermediate runs rewrite `silver_deep_context.json`/`gold_deep_context.json` with missing futures/ETF/COT/FRED overlays instead of carrying forward the last successful values. On the default schedule the deep-context files will oscillate between complete and mostly empty every other cycle.

- [P1] Make the timeframe timeout actually stop stuck workers — Q:\fa-review\portfolio\data_collector.py:334-339
  If `as_completed(..., timeout=...)` raises here, the code is still inside a `with ThreadPoolExecutor(...)` block. Calling `f.cancel()` does not stop futures that are already running, and exiting the `with` immediately calls `shutdown(wait=True)`, so a hung fetch (for example a yfinance call that never returns) will still block `collect_timeframes()` until that worker finishes. In other words BUG-179 does not actually cap the hang it is meant to prevent.

- [P2] Initialize max-pain search for a minimum, not -1 — Q:\fa-review\portfolio\crypto_macro_data.py:137-137
  `total_pain` is always non-negative, so seeding `max_pain_value` with `-1` makes the `total_pain < max_pain_value` check below false for every candidate after the first one. For any expiry with multiple strikes, `max_pain_strike` stays pinned to the first entry in `all_strikes`, so every caller of `get_deribit_options()` gets a bogus max-pain level.

- [P2] Pull futures backfill from the FAPI endpoint — Q:\fa-review\portfolio\data_refresh.py:30-31
  `download_klines()` writes into `.../binance/futures` and the output filenames are `*-futures.feather`, but this request goes to the spot `BINANCE_BASE` endpoint. Any downstream consumer treating these files as futures bars will silently mix spot candles with futures-only metrics such as funding/open interest, which makes the backfill inconsistent.

- [P2] Use per-event release times instead of 14:00 UTC — Q:\fa-review\portfolio\econ_dates.py:155-156
  This hardcodes every macro event to 14:00 UTC, but the calendar mixes 08:30 ET releases (CPI/NFP/GDP) with 14:00 ET FOMC decisions. Around event days `next_event()`, `events_within_hours()` and `is_macro_window()` will therefore open or close their blackout windows several hours early or late; on FOMC days the error is 4-5 hours. Using one UTC anchor defeats the proximity checks these helpers are supposed to provide.
