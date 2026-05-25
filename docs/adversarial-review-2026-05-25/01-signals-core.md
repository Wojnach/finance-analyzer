# Signals-Core Adversarial Review

Scope: 15 files under `portfolio/` totalling ~10,000 LOC. Worktree
`Q:/finance-analyzer-worktrees/review-signals-core` reviewed against the
voting / accuracy / consensus pipeline of `signal_engine.generate_signal()`.

## P0 findings

None. The hot live-trading path has been reinforced extensively across the
last 6+ months of reviews (fail-closed accuracy, atomic I/O, thread locks,
WAL on SQLite, circuit breakers); no remaining issues found that I can
demonstrate will lose money on the next cycle.

## P1 findings

`portfolio/signal_engine.py:3834-3843`: 🟠 **`btc_proxy` is in `DISABLED_SIGNALS` but its vote is injected after the dispatch loop and reaches the unweighted consensus counts.** `tickers.py:242` adds `btc_proxy` to `DISABLED_SIGNALS` (44.6% 1d / 24.1% 10d — formally disabled 2026-05-24). The enhanced-signal dispatch loop at `signal_engine.py:3664` correctly skips disabled signals — but `btc_proxy` is NOT a registered enhanced signal. It is set directly into `votes` at line 3840 for MSTR after the dispatch loop, with no `if "btc_proxy" not in DISABLED_SIGNALS:` guard. Downstream: `buy`/`sell` at line 4050-4051, `core_active` math (btc_proxy is not core, OK), `active_voters` (line 4063, used as MIN_VOTERS quorum gate at line 4076), `_buy_count`/`_sell_count` written to `extra_info` (lines 4331-4332, consumed by unanimity penalty stage 5 and ensemble-entropy stage 5b). `_weighted_consensus` does gate it via the accuracy gate (44.6% < 47% with 139 samples) so the WEIGHTED action is unaffected; but the unweighted action computed at line 4082-4090 IS affected, and `extra_info["_raw_action"]` (line 4324) — surfaced to dashboard and Layer 2 — is built from a count that includes a disabled signal. The bigger downstream impact is that `_total_applicable` (line 4328) computed by `_compute_applicable_count()` correctly excludes `btc_proxy`, so `_ent_hold = max(0, _ent_total - _ent_buy - _ent_sell)` (line 3081) can return 0 incorrectly when `_buy_count + _sell_count > _total_applicable` because of btc_proxy, warping the entropy calculation. Fix: at line 3840 add `if "btc_proxy" not in DISABLED_SIGNALS: votes["btc_proxy"] = btc_action` (or add btc_proxy to `_DISABLED_SIGNAL_OVERRIDES` for MSTR if it should still vote there).

`portfolio/main.py:1085-1091` + `portfolio/ic_computation.py:235-262`: 🟠 **`ic_cache.json` is single-horizon; refresh of both `3h` and `1d` every 60 cycles guarantees one of them is always missing.** `compute_and_cache_ic(h)` overwrites the entire cache file with one horizon's data. `load_cached_ic(h)` returns `None` if `cache["horizon"] != h`. main.py refreshes `("3h", "1d")` sequentially, so the second call wipes the first; the in-memory `_ic_data_cache` at `signal_engine.py:2154` does keep both horizons in-process, but a process restart (after task scheduler bounce) cold-starts with only the last-written horizon on disk, and the loop's own refresh keeps overwriting. Result: the 3h horizon IC multiplier is unreliable for the first several cycles after a restart and any non-`1d` horizon that's not in the refresh tuple is always cold. Fix: change `ic_cache.json` to a per-horizon nested dict (`{"3h": {...}, "1d": {...}, "time_3h": ..., "time_1d": ...}`), or write per-horizon files (`ic_cache_3h.json`, `ic_cache_1d.json`).

`portfolio/accuracy_stats.py:280-289, 1641-1643`: 🟠 **`signal_accuracy_recent` / `check_accuracy_changes` use UTC-aware `now` but `datetime.fromisoformat(ts_str)` returns whatever TZ the string carries; mixed naive/aware comparisons silently fail.** The `_load_accuracy_snapshots()` JSONL contains ISO timestamps written via `datetime.now(UTC).isoformat()` (TZ-aware), so the current code happens to work today. But `signal_accuracy_ewma` at line 327-332 catches `(ValueError, TypeError)` and silently `continue`s on a naive-vs-aware subtraction, which means any future writer that drops the `+00:00` suffix will silently zero out EWMA weighting for those entries with no error log. Fix: in both `signal_accuracy_ewma` and `_find_snapshot_near` (line 1609), normalise naive timestamps to UTC explicitly: `if entry_dt.tzinfo is None: entry_dt = entry_dt.replace(tzinfo=UTC)`.

`portfolio/signal_engine.py:2398, 4395`: 🟠 **Weighted-vs-unweighted action inconsistency reported in `_raw_*` extra_info fields.** Line 4324-4325 stamps `_raw_action` and `_raw_confidence` from the pre-weighted counts (lines 4076-4090 — derived from `votes` after macro + horizon disable mutations but before `_weighted_consensus`'s deeper gating). Reports / Layer 2 contexts that read `_raw_action` get a value computed from a different gate cascade than the live `action`, so a UI showing "raw consensus = SELL, weighted = HOLD" can be a true gate discrepancy or just gate skew. Not a money loss, but adds noise to Layer 2's reasoning chain. Fix: either align the raw-action gate cascade with the weighted one (apply the same regime_gated_effective + horizon_disabled + macro mutations only — already done) and document the residual difference (top-N exclusion, correlation leader gating, IC, bias penalty), or rename `_raw_*` to `_unweighted_*` to make the asymmetry explicit.

## P2 findings

`portfolio/signal_weights.py:1-121`: 🟡 **`SignalWeightManager` class is dead code.** `outcome_tracker.py:497-500` explicitly comments out the call: "MWU weight update removed — `SignalWeightManager.batch_update()` wrote to data/signal_weights.json but signal_engine.py never read it." Only test code (`tests/test_signal_weights.py`) imports the class. The entire 120-LOC module + `data/signal_weights.json` are zero-value disk-and-CPU surface. Fix: delete `portfolio/signal_weights.py`, `tests/test_signal_weights.py`, and remove the `signal_weights.json` write path; recover ~150 LOC of unmaintained complexity.

`portfolio/signal_db.py:262-405`: 🟡 **`SignalDB.signal_accuracy()` / `consensus_accuracy()` / `per_ticker_accuracy()` / `ticker_signal_accuracy()` skip the `_MIN_CHANGE_PCT` neutral-outcome filter that the JSONL pipeline applies.** `accuracy_stats._vote_correct()` returns `None` (skip the outcome) for `|change_pct| < 0.05%` — these are noise moves that should not score for or against a signal. The SQL versions at lines 288, 319, 347, 387 count ALL non-zero outcomes. If anything ever calls these methods, the accuracy numbers will diverge from the JSONL pipeline. Currently called only from tests (`grep -rn "db.signal_accuracy\|db.consensus_accuracy\|db.per_ticker_accuracy\|db.ticker_signal_accuracy" portfolio/ dashboard/ scripts/` returns no production hits) so impact is limited to test correctness — but the methods exist in the production-imported class and could be called any time. Fix: either delete the SQL methods (use the JSONL path consistently) or port the `_vote_correct()` threshold into the WHERE clauses (`WHERE ABS(o.change_pct) >= 0.05`).

`portfolio/signal_engine.py:1666-1758`: 🟡 **`_compute_dynamic_correlation_groups` flattens vote rows across tickers, conflating cross-asset agreement with intra-signal redundancy.** Lines 1690-1694 iterate `for entry in recent: for _tk, tdata in entry.get("tickers", {}).items(): rows.append(row)`. If both BTC and ETH have RSI BUY at the same cycle, they count as "agreement" — which they almost always do because they're correlated assets, not because their RSI implementations are redundant. The output drives `_get_correlation_groups()` which controls leader gating (signals downgraded to 0.15-0.30x weight). Worst case: legitimately independent signals get clustered as "agreeing" because they agree about market direction across tickers. Fix: build per-(ticker, signal) rows then compute per-ticker agreement rates and average them, or drop crypto/metals/stock tickers into separate matrices and union the resulting clusters.

`portfolio/signal_engine.py:2415, 2430-2435`: 🟡 **Group-leader selection uses raw `accuracy_data.get(s, {}).get("accuracy", 0.5)` rather than the macro-adjusted key everywhere — Codex Fix 2026-04-28 was applied to `_leader_accuracy_key` but `_topn_accuracy_key` and several other branches default a missing accuracy entry to 0.5.** That neutral default lets a brand-new signal with zero samples *out-rank* a 0.44-accuracy mature signal for Top-N inclusion and group leadership. The other gates eventually catch it (accuracy gate forces HOLD when samples ≥ 30 and acc < 0.47), but during the warmup window where a new signal has < 30 samples, its 0.5 default beats a mature 0.44 signal that would have been gated out anyway — non-trivial waste but not a money bug. Fix: when `samples < MIN_SAMPLES`, treat the signal as 0.0 accuracy for ranking (so it doesn't pre-empt gate-eligible peers) rather than 0.5.

`portfolio/accuracy_degradation.py:298-302`: 🟡 **`_find_baseline_snapshot` filters by exact `window_days==BASELINE_TARGET_DAYS` match.** If `BASELINE_TARGET_DAYS` is ever tuned (e.g., to 21d), the entire snapshot history becomes invisible because old snapshots stamped `window_days=14` will never match. The 13-day transition window is documented but a value change has no migration path. Fix: accept any snapshot whose `window_days` is within ±1 of the target, OR record an explicit set of acceptable historical windows.

`portfolio/signal_engine.py:2898-2905`: 🟡 **`_compute_adx` cache key does not include `period`.** `df_id` is `(n, close_first, close_mid, close_last, high_max, low_min)`. If any caller ever passes `period != 14`, they'd collide with the default-period cached value. Currently the only caller is `apply_confidence_penalties` which uses the default — but the function signature accepts period, advertising contractually a cache key that doesn't honour it. Fix: include `period` in `df_id`.

`portfolio/signal_weights.py:119-121`: 🟡 **`_load` has a trailing comment about honoring stored `eta` with no implementation.** Lines 119-120 read: "Honour stored eta only if caller did not override it (caller passes None → _DEFAULT_ETA, so we preserve stored value)". Nothing reads `data["eta"]`. The comment is misleading documentation for behaviour that does not exist. (Caught with the dead-code note above — fold this into the deletion or implement the eta restore.)

`portfolio/signal_engine.py:3520-3537, 3573-3589`: 🟡 **Confidence-gated LLM votes leave a stale `extra_info["<llm>_action"]` value.** Line 3527 sets `extra_info["ministral_action"] = gated_action` BEFORE the confidence gate at line 3534 flips `gated_action` to HOLD. `votes["ministral"]` (line 3537) gets the correct value but `extra_info["ministral_action"]` is the pre-confidence-gate value. Reporting and dashboard endpoints display the stale value. Fix: move the `extra_info["<llm>_action"] = gated_action` assignment to AFTER the confidence gate.

`portfolio/signal_decay_alert.py:12`: 🟡 **Unused `import json`.**

`portfolio/signal_engine.py:2940-2942`: 🟡 **`_compute_adx` exception path writes `None` to cache without checking the size cap.** If `_adx_cache` is at the cap and an exception is hit, the cap is silently exceeded. Inconsequential (caches are bounded by realistic input shapes), but the asymmetry vs. the success-path eviction at line 2937-2941 is invisible. Fix: factor the eviction into a helper and call it from both branches.

`portfolio/accuracy_stats.py:1644-1646`: 🟡 **`check_accuracy_changes` and `_find_snapshot_near` use a 36-hour max delta to find "7 days ago" — but if the daily snapshot fails to write for 2+ days, the alert silently picks the most-recent-within-36h snapshot, which may be 5 or 9 days old. Difference vs the documented 7d is then claimed in the alert.** The new `accuracy_degradation` pipeline addresses this with `MIN_SNAPSHOT_AGE_DAYS=13`, but the older `check_accuracy_changes` path is still wired into `signal_decay_alert` indirectly and surfaces in the dashboard. Fix: when picking the baseline, log the actual chosen `(target - chosen).days` so operators see when the comparison is skewed.

## P3 findings

`portfolio/signal_engine.py:1716`: ⚪ `df[s1].values` / `df[s2].values` use the deprecated `.values` accessor — pandas suggests `.to_numpy()`.

`portfolio/ic_computation.py:225-232`: ⚪ `_load_entries(days=N)` does `entries = load_entries()` (loads ALL entries) then filters. For multi-month logs this loads the whole 50K-entry SQLite each call; the recent-N-days path could use a SQL WHERE clause on `snapshots.ts`. Performance not correctness.

`portfolio/signal_engine.py:1115-1118`: ⚪ `_prev_sentiment_loaded` is a module-level bool guarded by `_sentiment_lock`. The lazy-load pattern is fine but the `_sentiment_dirty` global also lives outside the lock — only the assignments are guarded. No bug because all reads/writes are inside the lock, but the `global` declaration in two places is fragile.

`portfolio/signal_engine.py:4461`: ⚪ Global confidence cap `conf = min(conf, 0.80)` is hardcoded; the surrounding code uses named constants for every other threshold. Move to a module-level constant for discoverability.

## Cross-cutting observations

- The signal engine has accreted 8+ orthogonal gate/penalty cascades: accuracy gate, directional gate, regime gate, horizon disable, macro window force-HOLD, per-ticker blacklist, correlation cluster leader gate, top-N gate, persistence filter, IC multiplier, bias penalty, activity-rate cap, crisis mode, market-health penalty, earnings gate, per-ticker consensus gate, calibration compression, time-of-day factor, market-health, linear-factor confirmation. The interactions are documented but the order matters and there is no single integration test that exercises all gates in sequence. Recommend a property-style test that exhaustively perturbs each gate's inputs and asserts the consensus output is monotone w.r.t. confidence-improving changes.

- The `_raw_votes` / `_weighted_action` / `_raw_action` triple exposed in `extra_info` has three different vote-counting contracts. Layer 2 reads all three and a discrepancy between them is treated by the playbook as informative ("system disagrees with itself"), but in practice they reflect different gate-cascade vantage points, not genuine disagreement. Recommend either consolidating to one canonical action + audit trail, or renaming so the asymmetry is explicit (e.g., `_pre_correlation_action`, `_post_top_n_action`).

- Several files have heavy comment-as-changelog ("2026-04-15 (BUG-178)…"). Useful for audit but `signal_engine.py` is now 4476 lines, ~30% of which are dated rationale comments. A move to `docs/decisions/` ADRs referenced from short inline comments would make the live code easier to navigate.

- 2026-05-24 disabled-signal list bloat: `momentum_factors`, `btc_proxy`, `cryptotrader_lm`, `finance_llama`, `meta_trader` are all in `DISABLED_SIGNALS` per tickers.py with explanatory comments. The disable list now has 50+ entries. Consider periodic pruning of signals that have been disabled > 60 days and whose code is unmaintained — pending code may be subject to bit-rot (e.g., the `btc_proxy` injection issue above only matters because btc_proxy still has a code path that fires).

- `_compute_dynamic_correlation_groups` and `compute_signal_ic_per_ticker` each walk the full 50K-entry signal log per call with no shared cross-pipeline cache. The L1+L2 cache pattern installed for `signal_utility` / `regime_accuracy` should be replicated for IC and dynamic correlation groups — the dogpile risk is the same on a cold-start cycle.

## Files reviewed

- `portfolio/signal_engine.py` — 4476 lines
- `portfolio/signal_registry.py` — 377 lines
- `portfolio/signal_weights.py` — 120 lines (dead in production)
- `portfolio/signal_weight_optimizer.py` — 170 lines
- `portfolio/signal_utils.py` — 132 lines
- `portfolio/accuracy_stats.py` — 2077 lines
- `portfolio/ic_computation.py` — 296 lines
- `portfolio/signal_state_since.py` — 67 lines
- `portfolio/signal_history.py` — 215 lines
- `portfolio/signal_decay_alert.py` — 163 lines
- `portfolio/signal_db.py` — 405 lines
- `portfolio/signal_postmortem.py` — 266 lines
- `portfolio/cusum_accuracy_monitor.py` — 170 lines
- `portfolio/accuracy_degradation.py` — 1062 lines
- `portfolio/correlation_priors.py` — 31 lines

Total: ~10,027 lines.
