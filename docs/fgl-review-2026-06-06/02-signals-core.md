# Adversarial review — signals-core (2026-06-06)

Reviewed full current state of the 21 named files. Findings below; only defensible correctness issues included.

## P0
(none — the live consensus path is heavily hardened against NaN/None/poisoned-cache and silent failure; no wrong-trade-causing P0 found.)

## P1
portfolio/signal_engine.py:4679-4698: P1: seasonal BUY multiplier (1.10-1.15x Jan-Apr, applied at line 4689) runs AFTER the documented global confidence cap `conf = min(conf, 0.80)` (line 4664) and the 3h cap (4669), with NO clamp before `return action, conf` (4698). A metals BUY pinned at the 0.80 cap returns conf 0.88-0.92 in Jan-Apr — violating the stated calibration cap and feeding inflated confidence to Layer 2 / sizing. → Re-apply `conf = min(conf, 0.80)` (and the 3h cap) after the seasonal multiplier, or move the seasonal mult above line 4664.

## P2
portfolio/signal_engine.py:525,516-530: P2: `_accuracy_tier_mult()` and `_ACCURACY_TIER_THRESHOLDS` (the "1.25x for 65%+, 1.15x for 60%+" boost documented in CLAUDE.md and the module docstring) are defined but never called anywhere in the module. The advertised accuracy-tier boost is dead code; high-accuracy regime signals do not actually get the extra weight the spec claims. → Either wire it into `_weighted_consensus` after direction-specific weight assignment, or delete it and correct CLAUDE.md so the documented behavior matches reality.

portfolio/forecast_signal.py:150: P2: `samples = forecast[0].numpy()` on the Chronos v1 path will raise "can't convert cuda tensor to numpy" when the pipeline loaded on CUDA (it loads with `device_map=device`, device="cuda"). The broad `except` in `forecast_chronos` (line 136) swallows it and returns None, silently disabling the v1 fallback whenever Chronos-2 is unavailable. → Use `forecast[0].cpu().numpy()`.

portfolio/forecast_accuracy.py:142,159: P2: `actual_change = outcome.get("change_pct", 0)` then `actual_up = actual_change > 0` with no try-guard. If a backfilled entry carries `"change_pct": null` (key present, value None — possible after a partial/edited write), `None > 0` raises TypeError and aborts the entire forecast-accuracy computation. Also no `_MIN_CHANGE_PCT` neutral band, so an exact-zero move counts every BUY wrong / SELL right — inconsistent with `accuracy_stats._vote_correct`. → None-guard change_pct (skip) and apply the same neutral-band filter used elsewhere.

portfolio/meta_learner.py:289-310: P2: the calibrated decision threshold is selected by maximizing accuracy ON THE TEST SET (`for t in np.arange(...): cal_acc = accuracy_score(y_test, ...)`), then reported as `calibrated_accuracy` and persisted to drive production `predict()`. This is in-sample threshold optimization on the only OOS slice, so `calibrated_accuracy` is optimistic and the deployed threshold is overfit to the test window. → Tune the threshold on a separate validation split (e.g. the tail of train), not on the test set used to report OOS accuracy.

portfolio/train_signal_weights.py:89-98 + signal_weight_optimizer.py:75-77: P2: `_load_signal_history` interleaves all tickers into one frame and `set_index("ts")` produces duplicate timestamps; `walk_forward_optimize` then does `signals_df.index.intersection(returns.index)` and `.loc[common]`. With duplicate index labels `.loc` can return a Cartesian-expanded / misaligned result, and positional `iloc` train/test slicing mixes different tickers sharing a ts across the train/test boundary. Offline trainer only (not the live loop), but weights/OOS metrics it produces are unreliable. → Use a RangeIndex (keep ts as a column) or a (ts,ticker) MultiIndex and split by time, not row position.

portfolio/signal_db.py:262-301: P2: `SignalDB.signal_accuracy()` / `consensus_accuracy()` count `change_pct > 0` directly with NO `_MIN_CHANGE_PCT` neutral threshold, diverging from the canonical `accuracy_stats._vote_correct` used by the live path. Any caller that reaches the SQL methods (vs the Python `load_entries` path) gets systematically different accuracy numbers (near-zero moves counted as directional). → Apply the same `abs(change_pct) >= _MIN_CHANGE_PCT` filter in the SQL aggregation, or route all accuracy through the Python path.

## P3
portfolio/signal_registry.py:48-71: P3: `load_signal_func` mutates the shared `entry` dict (`entry["func"]=...`, `entry["_fail_ts"]=...`) from 8 ThreadPoolExecutor workers without a lock (entries are shared — `get_enhanced_signals` shallow-copies the outer dict only). Races are idempotent today (same func object / same sentinel) so effectively benign, but it's an unsynchronized read-modify-write on hot shared state. → Guard with a module lock or pre-warm all funcs single-threaded at startup.

portfolio/shadow_registry.py:287-295: P3: `_PROMOTED_CACHE` (dict) is read-modified by `is_promoted()` from multiple ticker threads without a lock. Benign (last-writer-wins on a frozenset swap) but technically a data race on module state. → Minor; add a lock if it ever holds richer state.

portfolio/forecast_signal.py:97: P3: `except (ImportError, Exception) as e` — `Exception` subsumes `ImportError`; redundant tuple. Harmless. → Drop `ImportError`.

portfolio/ic_computation.py:235-262 vs signal_engine.py:2305-2329: P3: `compute_and_cache_ic` writes a single `ic_cache.json` keyed by one `horizon`; `load_cached_ic` rejects on `cache.get("horizon") != horizon`. Under 8 threads requesting different horizons the file is overwritten and most reads miss, forcing repeated recompute (perf, not correctness — IC mult just falls back to 1.0x cleanly). → Key the cache file/dict by horizon.

## Subsystem risk summary
The live `generate_signal` → `_weighted_consensus` path is unusually well-defended against the cardinal sins (NaN/None coercion, poisoned-cache pruning, fail-closed accuracy gate, no silent-default votes that mask outages), so no P0/P1 wrong-trade bug was found in the hot path — the one real P1 is a post-cap confidence inflation (seasonal mult) that overstates conviction to downstream consumers. The larger latent risk is drift between documented and actual behavior (accuracy-tier boost is dead code) and inconsistent accuracy math across the SQL/Python/forecast accuracy paths, which can make offline tuning and dashboards disagree with what the live gate actually does.
