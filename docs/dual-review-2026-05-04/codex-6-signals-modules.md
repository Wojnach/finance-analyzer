# Codex Review — 6-signals-modules

## Summary

The patch introduces multiple correctness issues in active signal modules, including a systematic January BUY bias in `calendar_seasonal`, a missed pre-holiday case, a runtime failure path in `futures_flow`, and dead seasonality logic for metals. These materially change outputs or suppress intended behavior, so the patch should not be considered correct.

Full review comments:

- [P2] Remove January from the Sell-in-May BUY branch — Q:\fa-review\portfolio\signals\calendar_seasonal.py:157-163
  When `month == 1`, `_sell_in_may()` votes `BUY` here and `_january_effect()` votes another `BUY` at lines 183-184. Because the composite only requires two aligned active votes, an otherwise neutral mid-January session becomes a hard `BUY` at 0.6 confidence, giving the calendar signal a standing January-long bias across every ticker.

- [P2] Look past the next calendar day for Monday holidays — Q:\fa-review\portfolio\signals\calendar_seasonal.py:222-223
  This only checks `last_date + 1 day`. On the main multi-day-gap cases the docstring describes—Friday sessions before Monday holidays like MLK, Presidents', Memorial, or Labor Day—`next_day` is Saturday, so the pre-holiday `BUY` never fires. The signal therefore misses the exact long-weekend setups it is meant to capture.

- [P2] Guard the indicator pass against sparse OI history rows — Q:\fa-review\portfolio\signals\futures_flow.py:286-287
  The OI sub-signals above already handle missing fields with `d.get("oi", 0)`, but this reporting block switches back to `d["oi"]`. If Binance omits `oi` in any sample, `compute_futures_flow_signal()` raises `KeyError` after the vote is computed, and `signal_engine` will turn the whole `futures_flow` signal into a failed/HOLD result for that ticker instead of returning a degraded signal.

- [P2] Use the candle timestamp column for metals seasonality — Q:\fa-review\portfolio\signals\mean_reversion.py:460-460
  The standard OHLCV shape in this repo keeps timestamps in `df["time"]` and leaves the index as a `RangeIndex` (`data_collector` and `tests.make_ohlcv_df()` both do this). With those inputs, `hasattr(df.index, "hour")` is false, so the metals seasonality profile is never applied here and the signal silently runs on raw prices in production. The same `df.index.hour` assumption is duplicated in `momentum_factors._apply_seasonality()`.
The patch introduces multiple correctness issues in active signal modules, including a systematic January BUY bias in `calendar_seasonal`, a missed pre-holiday case, a runtime failure path in `futures_flow`, and dead seasonality logic for metals. These materially change outputs or suppress intended behavior, so the patch should not be considered correct.

## Full review comments

- [P2] Remove January from the Sell-in-May BUY branch — Q:\fa-review\portfolio\signals\calendar_seasonal.py:157-163
  When `month == 1`, `_sell_in_may()` votes `BUY` here and `_january_effect()` votes another `BUY` at lines 183-184. Because the composite only requires two aligned active votes, an otherwise neutral mid-January session becomes a hard `BUY` at 0.6 confidence, giving the calendar signal a standing January-long bias across every ticker.

- [P2] Look past the next calendar day for Monday holidays — Q:\fa-review\portfolio\signals\calendar_seasonal.py:222-223
  This only checks `last_date + 1 day`. On the main multi-day-gap cases the docstring describes—Friday sessions before Monday holidays like MLK, Presidents', Memorial, or Labor Day—`next_day` is Saturday, so the pre-holiday `BUY` never fires. The signal therefore misses the exact long-weekend setups it is meant to capture.

- [P2] Guard the indicator pass against sparse OI history rows — Q:\fa-review\portfolio\signals\futures_flow.py:286-287
  The OI sub-signals above already handle missing fields with `d.get("oi", 0)`, but this reporting block switches back to `d["oi"]`. If Binance omits `oi` in any sample, `compute_futures_flow_signal()` raises `KeyError` after the vote is computed, and `signal_engine` will turn the whole `futures_flow` signal into a failed/HOLD result for that ticker instead of returning a degraded signal.

- [P2] Use the candle timestamp column for metals seasonality — Q:\fa-review\portfolio\signals\mean_reversion.py:460-460
  The standard OHLCV shape in this repo keeps timestamps in `df["time"]` and leaves the index as a `RangeIndex` (`data_collector` and `tests.make_ohlcv_df()` both do this). With those inputs, `hasattr(df.index, "hour")` is false, so the metals seasonality profile is never applied here and the signal silently runs on raw prices in production. The same `df.index.hour` assumption is duplicated in `momentum_factors._apply_seasonality()`.
