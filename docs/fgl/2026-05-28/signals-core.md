# Signals-Core Adversarial Review — finance-analyzer

Scope: `portfolio/signal_engine.py`, `signal_registry.py`, `accuracy_stats.py`,
`accuracy_degradation.py`, `outcome_tracker.py`, `ticker_accuracy.py`,
`tickers.py`, `signal_utils.py`, `signal_weights.py`,
`signal_weight_optimizer.py`, `ic_computation.py`, `forecast_accuracy.py`.
Reviewed against worktree `Q:\fa-fgl\signals-core` (`git diff fgl-baseline HEAD`)
with full-codebase cross-reference at `Q:\finance-analyzer`.

## Count summary

| Severity | Count |
|----------|-------|
| P0 (money loss / crash / data-corruption / silent-exit-0) | 0 |
| P1 (wrong behavior under realistic conditions) | 4 |
| P2 (latent / fragile) | 9 |
| P3 (minor) | 4 |

Headline: no P0 found. The look-ahead / future-data question is **clean** —
outcomes are measured at `entry_ts + horizon` with a strict `now_ts < target_ts`
guard, base price is the snapshot price at decision time, and the accuracy gate
only ever force-HOLDs (never inverts). The empty-recent-window → 0% → force-HOLD
fear is **defended**: `blend_accuracy_data` falls back to all-time on
`rc_samples < min_recent_samples` and uses neutral 0.5 for immature signals.
The accuracy_degradation "fires many times/day" is **not over-alerting on
Telegram**: 24h per-key cooldown + ViolationTracker identity hashing on the
sorted alert-key set (`loop_contract.violation_identity_payload`) dedups
re-fires; the per-cycle WARNING log is recompute noise, not re-alerts. The main
real defects are a documented horizon/accuracy mismatch and several
methodology inconsistencies that quietly bias accuracy numbers.

---

## P1 — wrong behavior under realistic conditions

portfolio/signal_engine.py:4119-4121: P1 correctness: `acc_horizon = horizon if horizon in ("3h","4h","12h") else "1d"` collapses the 3d/5d/10d consensus horizons onto **1d** accuracy stats for the accuracy gate, weighting, IC, regime, and utility overlays. A signal good at 1d but poor at 10d (or vice versa) is gated/weighted with the wrong horizon's edge on every 3d/5d/10d vote. Acknowledged by an inline `TODO: MANUAL REVIEW — P1.12`. Fix: build per-horizon accuracy caches for 3d/5d/10d (the cache infra already keys by horizon string) and pass the true horizon, or explicitly document that 3d/5d/10d consensus is advisory-only and exclude it from any execution path.

portfolio/forecast_accuracy.py:142,159: P1 methodology: forecast direction-correctness uses `actual_up = actual_change > 0` with **no neutral-move filter**, unlike every other accuracy path in the codebase (which skips `|change_pct| < _MIN_CHANGE_PCT = 0.05`). A 0.001% drift counts a BUY as "correct". Forecast accuracy is therefore computed on a different rule than `signals_recent`/`per_ticker_recent`/consensus, so the accuracy_degradation diff compares forecast-side numbers that are systematically inflated/biased vs the signal-side baseline. Fix: route forecast outcomes through `accuracy_stats._vote_correct` (or replicate the `_MIN_CHANGE_PCT` neutral skip) so all four degradation scopes use one definition of "correct".

portfolio/ticker_accuracy.py:131-137: P1 sizing impact: `direction_probability` (Mode-B Telegram + Kelly sizing) hard-drops any signal with per-ticker recent accuracy `< ACCURACY_GATE_THRESHOLD (0.47)`, then for surviving SELL votes uses `p_up = 1 - accuracy`. Because `days=7` recent windows on per-ticker per-signal data routinely have tiny N, the surviving set is small and noise-dominated; combined with the `min_samples=5` floor (vs the engine's 30), the published P(up) can swing hard on 5-sample signals — and the user trades real money off these probabilities. Fix: raise `min_samples` to match the engine (≥30 for sizing-grade probabilities) and blend recent with all-time the way `blend_accuracy_data` does rather than a raw 7d window.

portfolio/ic_computation.py:130-134: P1 metric correctness: ICIR is `ic_mean / ic_std` where the std is taken over `_rolling_ic` windows that **overlap by step=1** (line 155: `range(len(votes)-window+1)`) and are pooled **across all tickers** (votes/returns concatenated in `compute_signal_ic`, line 106). Overlapping windows are massively autocorrelated, so `ic_std` is understated and ICIR is inflated; the `_IC_STABILITY_MIN` "stable ICIR" gate (`signal_engine._compute_ic_mult`) then admits IC-based weight boosts that aren't statistically stable. Fix: compute rolling IC on non-overlapping windows (step=window) and segment by ticker before pooling, or drop ICIR-gated boosts until the std is computed on independent blocks.

---

## P2 — latent / fragile

portfolio/forecast_accuracy.py:142: P2 latent-crash: `actual_change = outcome.get("change_pct", 0)` returns `None` (not 0) when the key exists with a null value; line 159 `actual_change > 0` then raises TypeError, aborting the whole `compute_forecast_accuracy` call for that horizon. Currently masked because `backfill_forecast_outcomes` always writes a float (line 315) and `_diff_against_baseline`/`check_degradation` wrap the call in `except`, so it degrades to "forecast degradation silently disabled" rather than a crash. Fix: guard `if actual_change is None: continue` like the signal-side `_vote_correct`.

portfolio/accuracy_stats.py:1017-1027 + outcome_tracker.py:567-573: P2 cross-process cache staleness: `write_accuracy_cache` / `load_cached_accuracy` use a per-process `_accuracy_write_lock` and a 1h TTL with no mtime check against `signal_log.db`. `invalidate_signal_utility_cache` clears the L2 utility disk file but does NOT touch `accuracy_cache.json`, and the in-memory `_signal_utility_cache` clear is process-local only. After the daily backfill, satellite loops (crypto/oil/metals) keep serving up-to-1h-stale accuracy → the gate can keep a freshly-degraded signal voting for up to an hour post-backfill. Fix: add an mtime/`snapshot_count` check to the accuracy cache freshness test, or have the backfill bump a shared version stamp the loaders compare.

portfolio/signal_engine.py:2766-2775: P2 single-signal dominance: final direction is purely `buy_weight/total_weight >= 0.5`. Beyond `_ACTIVITY_RATE_PENALTY` and correlation penalties there is no per-signal cap on a single signal's contribution to `buy_weight`/`sell_weight`. With `MIN_VOTERS_METALS = 2` (line 1074) and direction-specific weight = `buy_accuracy` up to ~0.95 plus utility boost (×1.5) plus IC boost, one high-accuracy signal can supply the entire winning weight while a second weak voter merely satisfies the count gate. Fix: cap any single signal's share of `total_weight` (e.g. ≤0.6) before the `>= 0.5` decision, or require ≥2 *weighted* contributors on the winning side.

portfolio/ticker_accuracy.py:38,112,298: P2 performance / cost: `accuracy_by_ticker_signal` calls `load_entries()` (full 50k-row scan) on EVERY call with no cache, while `accuracy_stats.accuracy_by_ticker_signal_cached` exists. `get_focus_probabilities` calls `direction_probability` per ticker × per horizon (3) plus another `accuracy_by_ticker_signal` per ticker — i.e. N×4 full scans per Mode-B notification build. On the notification path, not the 60s loop, but still a multi-second stall that can overlap the loop. Fix: route through the cached variant or pass a single pre-loaded `entries` list.

portfolio/ic_computation.py:235-239,253-262 + signal_engine.py:2179-2203: P2 double-cache divergence: `compute_and_cache_ic` writes a single `IC_CACHE_FILE` whose `horizon` field is the LAST-computed horizon; `load_cached_ic` returns None when `cache["horizon"] != horizon`, so two horizons requested in the same hour repeatedly invalidate each other's disk cache and recompute (the in-memory `_ic_data_cache` in signal_engine partly masks this per-process). The disk file holds only one horizon at a time despite being TTL-shared. Fix: key the IC cache by horizon (nested dict or per-horizon file) like `accuracy_cache.json` does.

portfolio/forecast_accuracy.py:332-369: P2 loose outcome alignment: `_lookup_price_at_time` accepts a snapshot up to `tolerance_hours` (default 2h, but callers pass up to 72h for stocks per the docstring) away from `target_time` as the "actual price at horizon". A 24h-horizon outcome matched to a snapshot 72h later measures a 4-day move, not a 1-day move, silently corrupting forecast accuracy for stocks. Fix: cap tolerance to a small fraction of the horizon (e.g. ≤ horizon/4) and drop outcomes that can't be matched tightly rather than widening to 72h.

portfolio/signal_engine.py:3227-3282: P2 fragility: core signals rsi/macd/ema/bb read `ind["rsi"]`, `ind["macd_hist"]`, `ind["ema9"]`, `ind["ema21"]` with direct indexing and NO try/except (unlike the enhanced-signal dispatch loop at 3764-3797 which force-HOLDs on exception). Production is safe because `indicators.compute_indicators._safe()` coerces NaN→defaults and returns None for bad bars, but any caller (test harness, future refactor, partial `ind`) that passes a dict missing these keys or with None values raises TypeError and kills the entire voter for that ticker. Fix: wrap the four core-vote blocks in the same defensive guard the enhanced loop uses, or `ind.get(..)` with finite defaults.

portfolio/accuracy_degradation.py:629-648: P2 baseline asymmetry: `_maybe_alert` requires BOTH `old` and `new`; a signal that existed in the 14-day-ago baseline but has **zero** recent samples produces `new=None` and is silently skipped — i.e. a signal that went completely dark (the most severe degradation possible) never alerts. Conversely a brand-new signal with no baseline never alerts even at 30% accuracy. Fix: emit a distinct "signal went silent" alert when `old` has samples ≥ floor and `new` is absent/zero-sample.

portfolio/signal_weights.py (whole file) + signal_weight_optimizer.py (whole file): P2 dead code on a reliability-critical path: both modules are confirmed unread by `_weighted_consensus` (MWU removed per outcome_tracker.py:497-500; walk-forward results "NEVER consumed" per scripts/write_quant_research.py:44). They still import `linear_factor`/`file_utils` and `signal_weights` is still constructed in `train_signal_weights.py`, burning maintenance surface and risking a future contributor wiring stale weights back in. Fix: delete or quarantine behind a clearly-labeled `experimental/` path so no live import can resurrect them.

portfolio/signal_weight_optimizer.py:75-97: P2 walk-forward leak risk: `walk_forward_optimize` slices train/test by positional `.iloc[start:train_end]` / `.iloc[train_end:test_end]` and assumes `signals_df`/`returns` are chronologically sorted on entry; it does `index.intersection` but never sorts. An unsorted index produces train/test sets that interleave in time = look-ahead in the (offline) optimizer. Currently offline-only so impact is bounded, but if results are ever consumed it taints them. Fix: `signals_df = signals_df.sort_index()` (and `returns`) at the top.

---

## P3 — minor

portfolio/ic_computation.py:127-128: P3 naming: `ic_buy`/`ic_sell` are labeled "IC" but are computed as mean directional return (`sum(buy_returns)/len`), not a rank correlation. Misleading field name in the cache consumed by reports. Rename to `mean_ret_buy`/`mean_ret_sell`.

portfolio/accuracy_stats.py:434-441: P3 inconsistency: `signal_accuracy_cost_adjusted` re-implements its own neutral filter (`abs(change_pct) < _MIN_CHANGE_PCT`) and direction check inline instead of reusing `_vote_correct(vote, change_pct, min_change_pct=cost_pct)`. Two copies of the same logic drift over time. Reuse the helper.

portfolio/outcome_tracker.py:211-231: P3 outcome-late bias: Binance kline backfill uses `interval=1h, startTime=target_ts, limit=1` and reads `data[0][4]` (close). The returned candle's openTime ≥ target_ts, so the close can be up to ~1h AFTER the intended horizon timestamp. Never look-ahead, but systematically measures the horizon ~0-60min late. Use a tighter interval (1m/5m) for the boundary fetch if precision matters.

portfolio/signal_engine.py:2596-2604,4347: P3 stale-field hazard: `extra_info["_voters"]` is set to the PRE-persistence `active_voters` while `_weighted_consensus` runs on post-persistence `consensus_votes`; the circuit-breaker Guard A intentionally consumes the raw count, but the dual meaning of "_voters" (raw) vs "_voters_post_filter" (effective) is a foot-gun for any new consumer that grabs `_voters`. Document the distinction at the assignment site or rename to `_voters_raw`.
