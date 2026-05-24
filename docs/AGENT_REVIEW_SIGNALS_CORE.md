# Adversarial Review: signals-core (Agent Findings)

Reviewer: code-reviewer subagent (fgl whole-codebase audit, empty-baseline)
Date: 2026-05-24
Worktree: `Q:\finance-analyzer\finance-analyzer-reviews\2026-05-24`
Branch: `review/fgl-2026-05-24` @ `289e5030`
Scope: 18 files, ~11,000 lines.

---

## Top 5 must-fix

1. **P0 — Bias penalty STILL double-applied** in `_weighted_consensus`. The
   `normalized_weight` factor multiplies in a symmetric `bias_penalty =
   max(1 - bias, 0.1)` for every vote (line 2677), then the directional
   `_resolve_bias_penalty(signal_bias)` multiplies on top of that only for
   bias-direction votes (line 2698). For a calendar-style 100%-BUY signal:
   contrarian SELL gets `1.0 * 0.1 = 0.1x` (not "full weight" as the docstring
   claims at lines 2685-2690); bias-direction BUY gets `0.1 * 0.2 = 0.02x`
   (way past the intended 0.2x). The prior review (2026-05-19) reported this
   identical bug; the May-19 fix to `_resolve_bias_penalty` did not remove the
   legacy `bias_penalty` term from `signal_activation_rates()` —
   `portfolio/accuracy_stats.py:840`.
2. **P0 — `blend_accuracy_data` double-counts directional samples.**
   `portfolio/accuracy_stats.py:970-974`: `total_buy/total_sell` are summed
   `at + rc`, but `recent` is a strict subset of `alltime`. Downstream
   directional gates (`_DIRECTIONAL_GATE_MIN_SAMPLES = 30`,
   `_DIR_WEIGHT_MIN_SAMPLES = 20`) and directional rescue thresholds use these
   inflated counts to decide whether to trust per-direction accuracy. The
   accuracy gate itself uses `total = max(at_samples, rc_samples)` which is
   the correct upper bound — but the buy/sell pair is summed and is therefore
   incorrect when recent overlaps alltime.
3. **P1 — `direction_probability_with_forecast` does 2 + N full signal-log
   loads per cycle, uncached.** `portfolio/ticker_accuracy.py:294` and `:298`
   each call `accuracy_by_ticker_signal(...)` which calls `load_entries()`
   (50K SQLite rows). For 3 horizons × 5 tickers per Mode-B refresh = 20
   full scans every call. No L1/L2 cache exists for this path while
   `signal_utility` and `regime_accuracy` have aggressive caches.
4. **P1 — Macro-window force-HOLD is computed but `claude_fundamental` is
   `DISABLED_SIGNALS`-listed elsewhere.** `MACRO_WINDOW_FORCE_HOLD_SIGNALS =
   {"claude_fundamental"}` (`portfolio/signal_engine.py:1027`). claude_fundamental
   is already disabled and force-HOLD anyway; this set is dead code unless
   another signal is added, in which case the comment about it being
   "the single direct mutation point" silently lies (regime gate and horizon
   blacklist mutate `votes` too).
5. **P2 — `compute_vote_correlation` (postmortem) is O(entries × ticker ×
   signals²).** `portfolio/signal_postmortem.py:148-163`: 5K entries × 5
   tickers × ~30 active signals → ~435 pairs per inner-loop iteration =
   ~32M dict writes per call. No caching. Called from `generate_postmortem()`
   which is invoked at session-end and from agent_summary build. Burns
   seconds on every Layer 2 report.

---

## P0 — Money / correctness

`portfolio/signal_engine.py:2677,2698` | P0 | bias double-application | `_weighted_consensus` multiplies `norm_weight` (which already contains `bias_penalty = max(1-bias, 0.1)`) AND then `_resolve_bias_penalty(signal_bias)` on bias-direction votes. Contrarian votes lose their advertised "full weight" — they get the legacy symmetric 0.1x. Bias-direction votes get squared (0.02x for extreme). Same bug the 2026-05-19 review flagged, NOT FIXED. | Remove `bias_penalty` and `normalized_weight` columns from `signal_activation_rates` output (or set `bias_penalty=1.0`) so the directional tier at line 2698 is the sole application. Verify all callers of `activation_rates` reference only `activation_rate`, `bias`, `buy_rate`, `sell_rate`, and `rarity_weight`.

`portfolio/accuracy_stats.py:970-974` | P0 | directional sample double-count | `total_buy / total_sell` summed across `at + rc` while `recent` is a strict subset of `alltime`. A signal with `at.total_buy=180, rc.total_buy=40` gets `result.total_buy=220` — inflating both `_DIRECTIONAL_GATE_MIN_SAMPLES` and `_DIR_WEIGHT_MIN_SAMPLES` boundary decisions. | Either pick the larger via `max(at_v, rc_v)` (matches the `total = max(at_samples, rc_samples)` pattern at line 937) or treat recent as an overlay (use `rc_v` if `rc_v >= min_recent_samples` else `at_v`).

`portfolio/signal_engine.py:2705-2710` | P0 | soft-confidence dampening is multiplicative below 1.0 but `weight` may already be tiny | After the bias+activation+correlation+regime+macro+horizon+IC stack, a soft vote at conf=0.15 with bias=0.91 and accuracy=0.5 has `weight = 0.5 * 1.0 * 0.1 * 0.15 = 0.0075`. The total_weight floor at line 2721-2723 (`if total_weight == 0`) doesn't catch values like 0.0075 — they vote, but their fractional contribution is rounding-error against any normal-weight signal. Net effect: soft votes can never tip a consensus they are theoretically supposed to. | Either skip soft votes when stacked penalties drop weight below an explicit floor (e.g. 0.05), or apply soft_conf BEFORE the bias/regime/horizon stack so it composes from the same baseline as strong votes.

---

## P1 — Silent failure / loop crash

`portfolio/ticker_accuracy.py:294,298` | P1 | uncached repeated signal-log scans | `direction_probability` → `accuracy_by_ticker_signal` → `load_entries()` (full SQLite scan). Called per ticker per horizon. No L1 cache. For 3 horizons × 5 focus tickers = 15 cold scans per get_focus_probabilities call. | Wrap `accuracy_by_ticker_signal` with the same double-checked-locking pattern as `get_or_compute_accuracy` / `get_or_compute_regime_accuracy`. TTL of 600-1800s acceptable given Mode-B refresh cadence.

`portfolio/signal_engine.py:1017-1020` | P1 | macro-window force-HOLD list is single-membership; promised mutation invariant violated by neighbors | `MACRO_WINDOW_FORCE_HOLD_SIGNALS` contains only `claude_fundamental`, which is already in `DISABLED_SIGNALS`. Comment at lines 2353-2356 says "force-HOLD pre-pass" is the macro mutation point — but regime_gated, horizon_disabled, and shadow throttle all mutate `votes` independently. Adding a second signal would expose hidden ordering bugs (each pre-pass reads `votes` after the prior mutated it). | Either delete the set + the four spots that consult it (lines 2358-2363, 4016-4019, and the two `_topn_accuracy_key`/`_leader_accuracy_key` overlays), or convert to a centralised `apply_macro_window_mutations(votes)` helper.

`portfolio/signal_engine.py:2278` | P1 | `import math as _math` shadowed by global `math` import elsewhere | `_math` is locally imported inside `_weighted_consensus` then dropped after `accuracy_data` sanitization block. Subsequent `_math.log2` / `_math.sqrt` usage in `apply_confidence_penalties` (line 3053 imports `import math as _math_ent` separately) shows inconsistent import naming. Not a correctness bug, but a future cleanup-attempt collapses one and breaks the other. | Move `import math` to module level; remove the local aliases.

`portfolio/signal_db.py:25-37` | P1 | SQLite connection has no `check_same_thread=False` and no lock | `_get_conn()` caches `self._conn`. If a future caller reuses one `SignalDB` across worker threads (the docstring example doesn't warn against it), sqlite3 raises `ProgrammingError("SQLite objects created in a thread can only be used in that same thread")`. Production callers all construct per-call instances, so the bug is latent — but the API silently invites the trap. | Either pass `check_same_thread=False` + a `threading.Lock`, or document the per-thread-instance contract in the class docstring.

`portfolio/meta_learner.py:402-408` | P1 | `_model_cache` mutated from multiple threads without lock | Module-level dict, read-modify-write on cache miss. ThreadPoolExecutor with 8 workers and 5 tickers can race: two threads simultaneously hit `predict()` post-retrain, both call `joblib.load()` (~50-200ms), both write to `_model_cache[horizon]`. Wasted I/O, but worse: if Python's dict rehashes mid-write the lookup can return a partially-constructed tuple in CPython 3.12+. | Wrap with `threading.Lock` (same pattern as `_adx_lock` at `signal_engine.py:36`).

`portfolio/signal_engine.py:2700-2710` | P1 | `soft_confidences = soft_confidences or {}` masks misuse | Callers can pass `extra_info` dict (4278) which contains keys like `"_soft_conf_ema"` AND `"_buy_count"` AND ~70 other fields. The `f"_soft_conf_{signal_name}"` lookup is correct, but nothing prevents a future contributor from changing key prefix on one side. Silent miss — soft votes get full weight. | Add explicit set of legal soft-conf keys in a module constant, assert membership at boundary.

---

## P2 — Perf / test gap

`portfolio/signal_postmortem.py:148-163` | P2 | O(entries × tickers × signals²) per call | 5K entries × 5 tickers × 435 signal pairs ≈ 11M dict ops per `compute_vote_correlation()` call. No caching. | Cache the pairwise agreement matrix with `_signal_utility`-style L1+L2 TTL. Outcomes update daily, so 1h TTL is safe.

`portfolio/ic_computation.py:130-136` | P2 | ICIR uses rolling window 50 but n<50 silently returns 0 | When samples<50, `_rolling_ic` returns `[]`, `icir=0`, `_compute_ic_mult` falls back to a less-confident multiplier. No log. Operators see `ICIR=0` reported for new signals and assume they have zero stability — when in fact the metric isn't computable. | Surface "samples < 50" as a sentinel (None or NaN) and have `_compute_ic_mult` route that distinctly from genuine ICIR=0.

`portfolio/ic_computation.py:127-128` | P2 | `ic_buy` / `ic_sell` mislabeled — they are mean returns, not ICs | Variable names suggest rank correlations; values are arithmetic means of directional returns. Not used downstream as IC, but the JSON output uses the name `ic_buy` which leaks the misleading label into ic_cache.json. | Rename to `avg_buy_return` / `avg_sell_return` (or drop entirely; nothing reads them).

`portfolio/signal_engine.py:1649-1740` | P2 | `_compute_dynamic_correlation_groups` walks entire 30-day signal log on every TTL miss | Iterates `recent` entries (~5K snapshots × 5 tickers × ~30 active signals = ~750K dataframe rows) building a pandas DF then computing pairwise agreement over `n*(n-1)/2 ≈ 435` pairs. Each pair calls `_compute_agreement_rate` which walks the column twice. TTL hides cost but a cold restart pays it once per horizon per cycle. | Cache the dataframe in L1; recompute only when row count grows past a delta threshold rather than on TTL.

`portfolio/accuracy_stats.py:937` | P2 | `total = max(at_samples, rc_samples)` is correct for overall but yields a confusing pair with the buggy total_buy sum | The `total` and `total_buy/sell` use different combining rules — that asymmetry is silent. A consumer reading `total=10000` and `total_buy=220` (which exceeds total because of the sum bug) sees an obvious inconsistency. | After fixing the P0 above, document the combining rule (max vs sum) in the function docstring.

`portfolio/signal_engine.py:36-37` | P2 | `_adx_cache` content-keyed by `(len, first_close, last_close)` floats | Float keys are exact-equality compared. Two distinct dataframes with identical first/last closes (common after market-hours rounding) collide. ADX caches the WRONG ADX value. Probability low but observable when prices reach round numbers. | Include a hash of more bars or a monotonic call counter as part of the key.

`portfolio/signal_engine.py:2453-2476` | P2 | meta-cluster dedup picks "best" leader by accuracy without min-sample floor | `_leader_accuracy_key` falls back to `0.5` for missing data. Two leaders with no samples both score 0.5; ties broken by `max()` insertion order. The leader that "wins" gets full weight, the other gets `_META_CLUSTER_PENALTY`. Effectively arbitrary. | Add `accuracy_data.get(s,{}).get("total",0) >= ACCURACY_GATE_MIN_SAMPLES` filter to leader selection.

`portfolio/cusum_accuracy_monitor.py:104` | P2 | Alert throttle "n > last_alert_n + 10" couples to update frequency, not time | If outcome_tracker fires 10 outcomes per backfill, the throttle resets on the very next outcome. No wall-clock floor. | Add `COOLDOWN_S` time gate analogous to `accuracy_degradation.COOLDOWN_PER_SIGNAL_S`.

`portfolio/accuracy_degradation.py:285-302` | P2 | `_find_baseline_snapshot` filters then asks `_find_snapshot_near` to pick closest | Two-step filter; if no 14d-window snapshot exists in the search radius, returns None silently. Caller logs "no matching baseline" (line 408-413) but does not surface a single Telegram alert saying the detector is dark. Two weeks of regression would go unnoticed. | Emit a one-shot Telegram WARNING on entry to the "no baseline" branch when 14d have elapsed since the last successful check.

---

## P3 — Style / dead code

`portfolio/signal_weights.py` (whole file) | P3 | dead code | `SignalWeightManager` is not constructed anywhere live. `portfolio/outcome_tracker.py:497-500` explicitly says "MWU weight update removed — SignalWeightManager.batch_update() wrote to data/signal_weights.json but signal_engine.py never read it". File is 120 lines of zero-effect code. | Delete the file (or keep behind a clear "unused, see outcome_tracker C6" comment).

`portfolio/ic_computation.py:80` | P3 | docstring promises `ic_buy`/`ic_sell` fields are ICs | They are returns, see P2 above. | Update docstring to match values.

`portfolio/signal_decay_alert.py:12` | P3 | `import json` unused | json module imported but never referenced; all JSON I/O via `file_utils`. | Remove import.

`portfolio/signal_engine.py:1` | P3 | module docstring says "32-signal voting system" | Codebase has 65 registered modules (17 active, 49 disabled). Doc drift. | Update to "65-module / 17-active signal voting".

`portfolio/signal_engine.py:1017-1020` | P3 | `MACRO_WINDOW_FORCE_HOLD_SIGNALS` is a one-element frozenset of an already-disabled signal | See P1 above; if the set is intentional scaffolding for future signals, mark with a `# scaffold:` comment so the next reviewer doesn't try to "clean it up". | Document intent or delete.

`portfolio/signal_engine.py:1825-1840` | P3 | `_CLUSTER_CORRELATION_PENALTIES` references groups that no longer exist | The 2026-04-30 split of trend_direction into 3 sub-clusters is documented inline. The 2026-05-07 disable of `trend` / `macd` / `futures_flow` leaves several "kept-as-group-in-case-re-enabled" entries with one effective member. A one-member cluster has no follower penalty. | Either prune to ≥2-member clusters or comment-document "future re-enable target".

`portfolio/signal_history.py:30` | P3 | `_history_lock` is module-global but file is per-call | Lock prevents concurrent threads from racing, but two PROCESSES (main loop, dashboard, crypto_loop) all import this and write the same file. Inter-process race remains. | Either accept (current state) and document, or migrate to SQLite/sqlite WAL for true cross-process safety.

`portfolio/correlation_priors.py` | P3 | 31-line file with hardcoded BTC/ETH and XAG/XAU priors | New ticker addition silently gets prior=0.0. No mechanism to derive priors from market data. | Add a TODO or a derivation script reference.

---

## Files verified clean (no findings ≥ P2)

- `portfolio/signal_state_since.py` — pure helper, no I/O, well-bounded.
- `portfolio/signal_utils.py` — math primitives, no surprises.
- `portfolio/signal_registry.py` — clean lazy-loader with failure cooldown.
- `portfolio/train_signal_weights.py` — change_pct conversion arithmetic checked OK.
- `portfolio/cusum_accuracy_monitor.py` — CUSUM formula matches Page's test.

---

## Prior P0 status

1. **Bias-penalty double-application (~line 2646 prior, ~line 2698 current)** —
   **NOT FIXED**. Re-flagged as P0 above.
2. **Utility-boost lets failed signals bypass accuracy gate (~line 4140 prior,
   ~line 4188-4217 current)** — **FIXED**. The gate check at line 4210
   (`if raw_samples < MIN or raw_acc >= eff_gate`) correctly limits boost to
   already-passing signals.
3. **Dynamic correlation groups dead code / HOLD dilution of Pearson** —
   **FIXED**. `_compute_dynamic_correlation_groups` now uses non-HOLD pair
   agreement rate (line 1638-1646), not Pearson over a HOLD-inflated vector.

---

## Cross-cutting observations (not findings)

- The bias-penalty story is one bug; the fact that the 2026-05-19 reviewer
  flagged it, the fix landed at one site, and the second site stayed
  untouched, suggests the testing harness for `_weighted_consensus` doesn't
  exercise the activation-rate / bias / weight composition end-to-end. A
  unit test that passes `activation_rates = {"calendar": {"bias": 1.0,
  "normalized_weight": 0.1, ...}}` and asserts contrarian-vote weight == 1.0
  would catch this on the next attempt.
- The persistence filter, per-asset cycle counts, soft-confidence dampening,
  IC multiplier, regime mults, horizon mults, macro downweight, crisis
  mode, activation rate normalization, correlation penalty, directional
  bias penalty, and rescue weight stack to ~9 multiplicative factors on a
  single signal weight. A signal at acc=0.5, bias=0.91, IC=0.0 in a ranging
  regime during a macro window with a soft vote ends up with weight ≈
  `0.5 * 0.85 (IC zero penalty) * 0.75 (regime) * 0.5 (macro) * 0.1 (norm
  weight) * 0.2 (bias direction) * 0.15 (soft conf) ≈ 0.00006`. That's
  effectively zero — fine — but it means the test surface area for "did
  this signal vote get counted" is impossible to reason about by inspection.
  A property-based test asserting that the final weight is monotonic in each
  factor would help.
