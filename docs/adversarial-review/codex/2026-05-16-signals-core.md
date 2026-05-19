## [P1] Forecast backfill deletes unprocessed predictions
**File:** portfolio/forecast_accuracy.py:322  
**Bug:** The loop breaks when `updated >= max_entries`, then `_write_predictions(modified_entries, path)` overwrites the whole JSONL with only the processed prefix.  
**Why it matters:** A file with 10,000 predictions can be truncated to a few hundred rows after a normal capped backfill. Historical forecast data is permanently lost.  
**Fix:** Always preserve and write the unprocessed tail, or update in-place via a full transformed list before applying the cap.

## [P1] Stock outcome backfill uses future daily closes
**File:** portfolio/outcome_tracker.py:273  
**Bug:** For `YF_MAP` tickers, historical lookup selects the daily row with `date <= target_date`, then returns that row’s close. For intraday horizons, that close occurs after the target time.  
**Why it matters:** A 3h MSTR outcome at 17:30 UTC can be scored using the same day’s market close, leaking future price movement into accuracy stats and poisoning live signal weights.  
**Fix:** Use intraday candles with timestamps and select the last completed bar at or before `target_ts`.

## [P1] Core-signal gate uses pre-persistence counts
**File:** portfolio/signal_engine.py:4119  
**Bug:** The final weighted-consensus gate checks `core_active`, but `core_active` was computed before `_apply_persistence_filter()` can turn core votes into `HOLD`.  
**Why it matters:** A stock can trade on enhanced-only consensus if the only core vote was filtered out but `core_active` still says one core signal exists.  
**Fix:** Recompute `core_active` from `consensus_votes` and gate on the post-persistence core count.

## [P1] MSTR BTC proxy bypasses accuracy tracking
**File:** portfolio/signal_engine.py:3697  
**Bug:** `btc_proxy` is injected into `votes` but is not in `SIGNAL_NAMES`, so outcome logging and accuracy stats do not track it. `_weighted_consensus()` then gives it default ungated 0.5 accuracy.  
**Why it matters:** A synthetic MSTR voter can influence real trades indefinitely without samples, accuracy gates, or postmortem visibility.  
**Fix:** Register it as a real signal included in logging/accuracy, or keep it out of consensus until it has tracked outcomes.

## [P1] Signal log rewrite is not durable
**File:** portfolio/outcome_tracker.py:559  
**Bug:** `backfill_outcomes()` manually writes a temp JSONL and `os.replace()`s it without `flush()` + `os.fsync()` before replacement.  
**Why it matters:** A crash or power loss can replace `signal_log.jsonl` with a non-durable temp file and lose the primary accuracy history.  
**Fix:** Use `file_utils.atomic_write_jsonl` semantics under the JSONL sidecar lock, or explicitly flush/fsync the temp file before replace.

## [P2] Binance outcomes use the candle after the target
**File:** portfolio/outcome_tracker.py:220  
**Bug:** Binance `klines` is queried with `startTime=target_ts` and `limit=1`, which returns the first 1h candle opening at or after the target.  
**Why it matters:** If a 3h target lands at 10:37, the scored close can be from the 11:00 candle, using future movement and corrupting crypto accuracy.  
**Fix:** Floor to the containing candle or fetch surrounding candles and choose the last bar at or before the target timestamp.

## [P2] Metals voter floor is overridden later
**File:** portfolio/signal_engine.py:2928  
**Bug:** Metals lower `min_voters` to 2, but `apply_confidence_penalties()` recomputes `dynamic_min` from regime only and forces `HOLD` below 5 voters in ranging/unknown regimes.  
**Why it matters:** The intended XAG/XAU 2-voter intraday path is silently disabled by a later penalty stage.  
**Fix:** Pass the asset-specific `min_voters` into the penalty cascade or make `_dynamic_min_voters_for_regime()` ticker-aware.

## [P2] ADX cache can return stale values for different candles
**File:** portfolio/signal_engine.py:2783  
**Bug:** `_compute_adx()` keys the cache by `(len, first_close, last_close)` only, ignoring high/low series and `period`.  
**Why it matters:** Two different OHLC frames with the same first/last close can reuse the wrong ADX, incorrectly triggering or skipping volume/ADX gates.  
**Fix:** Include `period` and high/low/close fingerprints in the key, or remove this cache.

## [P2] Accuracy loader ignores JSONL once SQLite has any rows
**File:** portfolio/accuracy_stats.py:151  
**Bug:** `load_entries()` returns SQLite entries whenever `snapshot_count() > 0`, without checking whether JSONL has newer rows.  
**Why it matters:** `log_signal_snapshot()` writes JSONL first and only best-effort writes SQLite; if SQLite lags, accuracy stats silently ignore valid signal snapshots.  
**Fix:** Reconcile latest timestamps/counts and fall back to JSONL or merge when SQLite is stale.

## [P2] Signal utility cache refreshes stale horizons
**File:** portfolio/accuracy_stats.py:138  
**Bug:** `_write_signal_utility_disk()` updates one global `"time"` for all horizons. `_load_signal_utility_disk()` uses that global timestamp for every horizon.  
**Why it matters:** Recomputing `1d` can make an old `3h` utility block look fresh, so live weights can use stale horizon data indefinitely.  
**Fix:** Store and validate per-horizon timestamps.

## [P2] Snapshot append gate is racy
**File:** portfolio/cumulative_tracker.py:42  
**Bug:** `maybe_log_hourly_snapshot()` reads the last timestamp and then appends without holding the JSONL sidecar lock across the check-plus-write.  
**Why it matters:** Concurrent loops can both decide the last snapshot is old and append duplicate hourly snapshots.  
**Fix:** Hold the same lock used by `atomic_append_jsonl` across last-line read, interval check, and append.

## [P2] Rolling changes anchor to wall clock, not latest snapshot
**File:** portfolio/cumulative_tracker.py:110  
**Bug:** `compute_rolling_changes()` uses `datetime.now(UTC)` for the lookback target instead of the timestamp of `snapshots[-1]`.  
**Why it matters:** After downtime, a “1d” change compares the latest stored price to a price 24h before now, not 24h before that latest price.  
**Fix:** Parse `latest["ts"]` and anchor all windows to that timestamp.

## [P2] Econ blackout is bypassed by throttle replay
**File:** portfolio/accuracy_degradation.py:351  
**Bug:** `check_degradation()` returns cached violations on the throttle path before checking `_is_econ_blackout()`.  
**Why it matters:** If a blackout starts within the 55-minute throttle window, stale pre-blackout violations are replayed and can escalate.  
**Fix:** Check blackout before throttle replay, or suppress cached violations while blackout is active.

## [P2] Forecast accuracy cache stays stale after backfill
**File:** portfolio/forecast_accuracy.py:327  
**Bug:** `backfill_forecast_outcomes()` writes new outcomes but never invalidates `_forecast_accuracy_cache`.  
**Why it matters:** Callers using `cached_forecast_accuracy()` can keep using pre-backfill accuracy for up to an hour after outcomes changed.  
**Fix:** Call `invalidate_forecast_accuracy_cache()` after a successful write.

## SUMMARY P1=5 P2=9 P3=0