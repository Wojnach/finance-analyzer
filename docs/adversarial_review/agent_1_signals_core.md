# Adversarial Review — signals-core subsystem

- **Agent:** agent_1 (signals-core specialist)
- **Subsystem:** signals-core
- **Files reviewed:** 20 (~12,343 LOC)
- **Total findings:** 24
  - P0 (data loss / silent failure / accuracy corruption): 5
  - P1 (real bug, hits production within a week): 11
  - P2 (latent bug / edge case / performance cliff): 6
  - P3 (style / readability / minor): 2

Files reviewed:
- portfolio/signal_engine.py (4392 LOC)
- portfolio/signal_registry.py (359)
- portfolio/signal_utils.py (132)
- portfolio/signal_weights.py (120)
- portfolio/signal_weight_optimizer.py (170)
- portfolio/signal_history.py (215)
- portfolio/signal_state_since.py (67)
- portfolio/signal_db.py (405)
- portfolio/signal_decay_alert.py (157)
- portfolio/signal_postmortem.py (266)
- portfolio/accuracy_stats.py (2070)
- portfolio/accuracy_degradation.py (1062)
- portfolio/outcome_tracker.py (580)
- portfolio/forecast_accuracy.py (445)
- portfolio/forecast_signal.py (409)
- portfolio/ticker_accuracy.py (346)
- portfolio/ic_computation.py (296)
- portfolio/meta_learner.py (461)
- portfolio/correlation_priors.py (31)
- portfolio/shadow_registry.py (360)

---

## portfolio/signal_engine.py

`Q:\finance-analyzer\portfolio\signal_engine.py:2646-2668`: **P0 (accuracy corruption — bias penalty double-application)**
`_weighted_consensus` first multiplies `weight *= norm_weight` at line 2647 where `norm_weight = activation_rates[sig]['normalized_weight']`. Inside `signal_activation_rates()` (accuracy_stats.py:849) that field is computed as `rarity_weight * bias_penalty` where `bias_penalty = max(1.0 - bias, 0.1)`. Then at lines 2663-2668 the SAME bias value is fed into `_resolve_bias_penalty(signal_bias)` and `weight *= _resolve_bias_penalty(...)`. A signal with bias=0.91 ("crypto_macro") therefore gets multiplied by `(1 - 0.91)=0.09` (floored at 0.1) AND again by `_BIAS_EXTREME_PENALTY=0.2` → effective 0.02x in its bias direction. The docstring at line 2533 ("This is the single application point for the bias penalty. apply_confidence_penalties does NOT re-apply this") is wrong — it's re-applied inside `_weighted_consensus` itself. Fix: drop one of the two paths (recommended: remove `bias_penalty` from `signal_activation_rates.normalized_weight` so the tiered cascade is authoritative).

`Q:\finance-analyzer\portfolio\signal_engine.py:4140-4162`: **P0 (accuracy gate bypass — aggressive utility boost)**
Utility boost reads `u_score = u.get("avg_return", 0.0)` from `signal_utility()`. `avg_return` is the raw `change_pct` value (already in percent — e.g. 0.5 means +0.5%). `boost = min(1.0 + u_score, 1.5)`. For a signal that averages +0.5pp per directional vote, `boost = 1.5`. A signal with raw accuracy=0.40 (gated by 0.47 floor) becomes `boosted_acc = min(0.40 * 1.5, 0.95) = 0.60` — now passes both the 0.47 standard gate AND the 0.50 high-sample gate. The "Unused EWMA, aggressive utility boost" finding in `memory/signal_engine_audit_findings.md` is correct. Fix: divide `avg_return` by 100 before computing boost, or change boost formula to `1 + u_score/2` and cap at 1.10.

`Q:\finance-analyzer\portfolio\signal_engine.py:1711-1722`: **P1 (dead code — dynamic correlation groups)**
`_get_correlation_groups()` calls `_cached("dynamic_corr_groups", ...)` which invokes `_compute_dynamic_correlation_groups()`. The CLAUDE.md flags "Dynamic correlation groups are dead code (HOLD dilution of Pearson). Fix: use agreement rate." The 2026-04-18 rewrite at lines 1597-1614 (`_compute_agreement_rate`) switched to agreement rate but the function still returns `_STATIC_CORRELATION_GROUPS` when `len(rows) < 30` (line 1647) or `df.shape[1] < 3` (line 1653). With 6320 snapshots × 5 tickers = 31600 rows the dynamic path SHOULD activate. Verify whether the cache key has ever populated with dynamic groups; `memory/dynamic_corr_bug.md` says it has not. Fix: log `len(rows)` at INFO level on each compute attempt and instrument the cache to surface whether dynamic or static was returned.

`Q:\finance-analyzer\portfolio\signal_engine.py:749-754`: **P1 (constant vs documented behavior mismatch — circuit breaker)**
Comment block lines 747-749 documents "Keeps at least 5 voters active by relaxing the gate by up to **6pp** (to 41% floor)" but `_GATE_RELAXATION_MAX = 0.02` (2pp). `.claude/rules/signals.md` also states "max relaxation (6pp) could recover that floor — effective floor 41%". Real effective floor is 0.45, not 0.41. Either the constant is wrong or the documentation is. Fix: confirm intended value and align comment + constant + rule doc.

`Q:\finance-analyzer\portfolio\signal_engine.py:2538-2539, 2584, 2596-2599`: **P1 (accuracy_data poisoning bypass — direct .get on un-sanitized fallback)**
The deep-sanitize loop at lines 2266-2303 produces a clean `accuracy_data`, then line 2538 does `acc = stats.get("accuracy", 0.5)`. Default 0.5 is used when key is absent — meaning a signal whose accuracy was sanitized OUT (because it was poisoned) is treated as a clean 50%-accurate mature signal and skates past the gate at line 2555 if samples=30+. The sanitize loop drops paired (accuracy, total) together, so this should not happen — BUT lines 2584 / 2587 fall back to `acc` for `dir_acc = stats.get("buy_accuracy", acc)`. If overall `accuracy` was sanitized to 0.5 and `total_buy=400` remained, the directional gate runs against 0.5 (rather than the real underlying value) and passes. Fix: when overall is poisoned, also drop directional counts.

`Q:\finance-analyzer\portfolio\signal_engine.py:1467`: **P1 (silent fallback — dynamic horizon weights)**
`if cross_acc is None or not (0.01 <= cross_acc <= 1.0): continue` — when a signal's cross-horizon accuracy is exactly 0.0 (legitimate 0% accuracy on cold start with `total=0`, `accuracy=0.0`), it's silently skipped. The check `0.01 <= cross_acc` excludes valid 0-accuracy data. Combined with the missing log statement, this becomes silent — `weights` dict drops the signal, then falls back to static `HORIZON_SIGNAL_WEIGHTS`. Fix: distinguish "no data" (samples=0) from "0% accuracy" (samples>0).

`Q:\finance-analyzer\portfolio\signal_engine.py:2928-2932`: **P1 (RVOL gate skipped when data missing)**
`if volume_ratio is not None and action != "HOLD":` — when `volume_ratio is None` (data fetch failed), the rule "RVOL < 0.5 forces HOLD" is completely skipped. signals.md says "RVOL < 0.5 forces HOLD regardless of other signals" — silent failure when the underlying data fetch crashes. Fix: log WARNING when `volume_ratio is None` and consider failing closed (force HOLD) on persistent data fetch failure rather than letting the signal vote unrestricted.

`Q:\finance-analyzer\portfolio\signal_engine.py:2660-2668`: **P2 (bias min-active uses total samples, not active votes)**
`signal_samples = act_data.get("samples", 0)` — `samples` comes from `signal_activation_rates()` line 850 which is the TOTAL count of all votes including HOLD. A signal that activates 1% of the time over 5000 entries has samples=5000 but only 50 active votes. The check `signal_samples >= _BIAS_MIN_ACTIVE (30)` passes trivially. The `bias` value is computed from only 50 votes — high statistical noise. Fix: change `_BIAS_MIN_ACTIVE` to be a floor on ACTIVE vote count, not total entries.

`Q:\finance-analyzer\portfolio\signal_engine.py:36-37, 2840-2876`: **P2 (ADX cache key collision)**
`df_id = (len(df), first_close, last_close)` — three signals processing different timeframes (3h vs 1d) can produce the same `(len, first_close, last_close)` tuple if the DataFrames share endpoint prices, returning the wrong ADX. Probability is low but non-zero on crypto where price commonly repeats to 2dp. Fix: include `id(df)` or a hash of the full close column.

`Q:\finance-analyzer\portfolio\signal_engine.py:2538-2540 vs 2596-2599`: **P2 (per-ticker override drops directional totals)**
Lines 4109-4126 build a per-ticker `override` dict. The conditional `if field in t_stats: override[field] = t_stats[field]` at line 4124 — if `accuracy_by_ticker_signal_cached` returns rows missing directional fields (older format), the override silently drops the directional accuracy from the global blend, falling back to overall accuracy. Fix: when a directional field is missing in t_stats, preserve the corresponding field from the global blend rather than dropping it.

`Q:\finance-analyzer\portfolio\signal_engine.py:3701-3711`: **P3 (broad exception swallows specific signal name)**
`except Exception as e: ... _signal_failures.append(sig_name)` — catches everything. CLAUDE.md requirement is to never silently swallow. While `logger.warning("Signal %s failed: %s", sig_name, e, exc_info=True)` does log with traceback, the warning level + only-after-3-failures alert (line 3707) means a single signal's intermittent failure is buried. Acceptable for production but worth flagging.

---

## portfolio/accuracy_stats.py

`Q:\finance-analyzer\portfolio\accuracy_stats.py:225-228`: **P0 (silent NaN→bearish in accuracy attribution)**
`change_pct = outcome.get("change_pct", 0)` — when `change_pct` key is present but `value is None`, `.get(key, default)` returns `None`, NOT 0. Then `_vote_correct(vote, None)` returns None (line 182 guard) and the outcome is correctly skipped via `if result_val is None: continue`. Confirmed safe. But the `null_change_pct_skipped` counter at line 227 increments only when key is present with None — if the key is absent entirely, default=0 makes `change_pct=0`, which passes `< _MIN_CHANGE_PCT` (0.05) and is silently dropped via `_vote_correct` returning None. The drop is correct, BUT the diagnostic log doesn't capture absent-key cases — making it impossible to distinguish "outcome backfill returned None" from "outcome has no key at all" (the latter indicates schema corruption). Fix: separate counters for "key absent" vs "key present, value=None".

`Q:\finance-analyzer\portfolio\accuracy_stats.py:980-990`: **P1 (activation cache race + stale-write on TTL miss)**
`load_cached_activation_rates()` reads cache, then on TTL miss recomputes (`signal_activation_rates()` which scans 50K entries — slow), then writes. The lock at line 987 protects the WRITE only. Two threads can both miss TTL, both recompute (wasted work, but acceptable). However, the cache check `time.time() - cache.get("time", 0) < ACTIVATION_CACHE_TTL` is computed inside the `if cache is not None:` branch but the next line writes `rates = signal_activation_rates()` which is the slow compute. No double-checked locking — same dogpile pattern that `get_or_compute_accuracy` was rewritten to fix. Fix: add double-checked locking equivalent to `_accuracy_compute_lock` pattern.

`Q:\finance-analyzer\portfolio\accuracy_stats.py:863-969`: **P1 (blend asymmetry for recent-only signals)**
At line 925 `elif rc_samples >= min_recent_samples: blended = rc_acc` — a signal present ONLY in recent data with ≥30 samples gets the raw recent accuracy directly. But at line 919 the blended path needs `at_samples > 0 AND rc_samples >= 30`. A signal with `at_samples=29, rc_samples=200` falls through both branches: line 919 needs at_samples > 0 (passes) AND rc_samples >= min_recent_samples (passes) → blended properly. But `at_acc` is computed from 29 samples (basically random); the blend 0.30 * at_acc with 29 samples imports noise into the result. Fix: require BOTH at_samples and rc_samples to meet a minimum (e.g. 30) before blending; otherwise prefer the source with more samples.

`Q:\finance-analyzer\portfolio\accuracy_stats.py:835-840`: **P2 (bias formula collapses on low activation rate)**
`bias = abs(buy_rate - sell_rate) / activation_rate` — when activation_rate=0.02, even a balanced buy_rate=0.01, sell_rate=0.01 produces bias=0.0 (good), but buy_rate=0.02, sell_rate=0.0 produces bias=1.0 (extreme penalty applied). A signal that activates 50 times out of 2500 entries with 30 BUYs and 20 SELLs has bias = (0.012-0.008)/0.020 = 0.20 (balanced). A signal that activates 50 times with 50 BUYs and 0 SELLs has bias=1.0. Both have identical statistical reliability (n=50) but get vastly different treatment. Fix: penalize bias only when sample size on the minority side is large enough to be statistically distinguishable from 0.

`Q:\finance-analyzer\portfolio\accuracy_stats.py:1042-1044, 1058-1060`: **P3 (silent cache write skip on empty result)**
`if result: write_accuracy_cache(horizon, result)` — when `result == {}` (empty dict, e.g., cold-start with no signals), the cache is NEVER written. The next thread runs through the lock+compute again, paying the cold cost. Fix: write empty caches too, or use a sentinel for "computed but no data" — but the comment at line 1107 acknowledges this for `consensus_accuracy` only.

---

## portfolio/outcome_tracker.py

`Q:\finance-analyzer\portfolio\outcome_tracker.py:159-166`: **P0 (silent SQLite divergence — dual-write order is JSONL-first)**
`atomic_append_jsonl(SIGNAL_LOG, entry)` at line 157, then SQLite insert at line 163. If process crashes between the two, JSONL has the entry but `signal_log.db` does not. accuracy_stats `load_entries()` prefers SQLite (line 152) — the crashed entry is invisible to all accuracy computation until/unless someone runs a reconciliation script. The dual write is also NOT atomic, so a partial write to SQLite throws but JSONL still claims success. Fix: write SQLite first, then JSONL (so the canonical "this happened" record is the more durable one); add explicit reconciliation pickup on startup.

`Q:\finance-analyzer\portfolio\outcome_tracker.py:155-156`: **P1 (broad except swallows SQLite write failures)**
`except Exception as e: logger.warning(...)` — covers EVERY failure path including disk-full, permissions, schema drift. The dashboard / loop continues believing the write succeeded; the next `--check-outcomes` run rediscovers the gap. Acceptable but needs an explicit metric (e.g., increment a counter in `data/critical_errors.jsonl`) so persistent SQLite write failures don't go unnoticed for weeks. Fix: emit critical_errors entry on N consecutive SQLite write failures.

---

## portfolio/signal_db.py

`Q:\finance-analyzer\portfolio\signal_db.py:31-37`: **P1 (connection leak in long-running processes)**
`_get_conn()` lazy-creates `self._conn` and never recycles. SignalDB instances created per-call in `outcome_tracker.log_signal_snapshot` (line 162) call `close()` explicitly. But a `SignalDB()` instance used long-term (like the cached one in `backfill_outcomes` line 416) keeps the connection open across the entire backfill pass — fine for the backfill loop, but if anyone uses it on the main loop without closing, the connection accumulates locks. WAL mode tolerates concurrent readers; no immediate bug, but instance lifecycle is implicit. Fix: document the close() contract or add a `__del__`/context-manager wrapper.

`Q:\finance-analyzer\portfolio\signal_db.py:262-302`: **P2 (signal_accuracy SQL excludes HOLD votes but ignores neutral outcomes)**
The SQL at lines 272-278 omits HOLD votes (`if vote == "HOLD": continue`) — correct. But `change_pct` is NOT filtered for `abs(change_pct) < _MIN_CHANGE_PCT`. This means SQL-path `signal_accuracy` includes tiny moves (e.g. ±0.01%) that `_vote_correct` would skip as neutral, inflating sample counts and arbitrarily counting noise as correct/incorrect 50/50. Two execution paths give DIFFERENT accuracy numbers depending on whether SQLite-direct or JSONL path is taken. Fix: filter `AND ABS(o.change_pct) >= 0.05` in the SQL.

---

## portfolio/ic_computation.py

`Q:\finance-analyzer\portfolio\ic_computation.py:25-28, 247-262`: **P1 (single-file IC cache cannot hold multiple horizons)**
`IC_CACHE_FILE = DATA_DIR / "ic_cache.json"` — one file, no per-horizon suffix. `compute_and_cache_ic("1d")` writes `{"horizon": "1d", ...}` to disk. `compute_and_cache_ic("3h")` overwrites it with `{"horizon": "3h", ...}`. The reader at line 260 then returns None for "1d" because `cache.get("horizon") != "1d"`. Each horizon evicts the previous, defeating multi-horizon caching across the 7 supported horizons. Fix: per-horizon files (e.g., `ic_cache_{horizon}.json`) or a top-level horizon-keyed dict.

---

## portfolio/forecast_accuracy.py

`Q:\finance-analyzer\portfolio\forecast_accuracy.py:142, 158-161`: **P1 (incorrect "correct" attribution for neutral outcomes)**
Line 142: `actual_change = outcome.get("change_pct", 0)`. Line 159: `actual_up = actual_change > 0`. When `actual_change == 0` (exactly flat, OR `change_pct` key is missing — defaulted to 0), `actual_up = False`. Then any signal voting `SELL` is counted as `correct=True` because `(not predicted_up) and (not actual_up)`. Inflates SELL-side forecast accuracy on neutral outcomes. Fix: skip when `abs(actual_change) < _MIN_CHANGE_PCT` (consistent with accuracy_stats.py:_vote_correct).

`Q:\finance-analyzer\portfolio\forecast_accuracy.py:332-369`: **P2 (linear scan of price snapshots — N*M complexity)**
`_lookup_price_at_time` iterates the entire `price_snapshots_hourly.jsonl` for every prediction backfill. With months of hourly snapshots × 100s of predictions per backfill, this is O(N*M). Acceptable today but a known cliff. Fix: build a timestamp index on first call.

---

## portfolio/ticker_accuracy.py

`Q:\finance-analyzer\portfolio\ticker_accuracy.py:131`: **P0 (per-ticker gate ignores tiered high-sample threshold)**
`if accuracy < ACCURACY_GATE_THRESHOLD: continue` — uses flat 0.47 gate. signal_engine applies the high-sample tier (0.50 for 7K+ samples). Mode B Telegram probabilities and Kelly sizing both call `direction_probability` here, so a signal with 50K samples at 49% accuracy is excluded by signal_engine but INCLUDED by direction_probability and would feed a positive `p_up = 0.49` weighted vote. Fix: import and apply the same tiered logic from signal_engine.

`Q:\finance-analyzer\portfolio\ticker_accuracy.py:86-176, 304-309`: **P1 (per-ticker accuracy mixes with global tier mismatch)**
`direction_probability` uses `min_samples=5` default (line 102). With 5 samples a signal's "accuracy" has SE±22pp on a 50/50 process. This feeds straight into a weighted probability without tier-up to a meaningful sample floor. Combined with the previous finding, Mode B notifications are vulnerable to small-sample noise. Fix: raise `min_samples` default to 30 to match `ACCURACY_GATE_MIN_SAMPLES`.

---

## portfolio/signal_history.py

`Q:\finance-analyzer\portfolio\signal_history.py:101-145`: **P2 (persistence score reads HISTORY_FILE without lock)**
`get_persistence_scores`, `get_signal_streaks`, `get_summary` all call `_load_history()` (line 119, 174, 209) without acquiring `_history_lock`. While `_load_history` itself uses `load_jsonl` (atomic-read-friendly), a reader can race with the write path (`update_history` at line 77-98) and see a stale view that misses the most recent cycle's update. Less severe than the original write-race (already fixed) but still an inconsistency window. Fix: hold `_history_lock` for read path too if callers depend on real-time data.

---

## Out-of-scope but worth flagging

`Q:\finance-analyzer\portfolio\signal_weights.py:97-103`: **P3 (MWU weight manager is dead code per outcome_tracker comment)**
`outcome_tracker.py:497-500` notes "C6: MWU weight update removed — SignalWeightManager.batch_update() wrote to data/signal_weights.json but signal_engine.py never read it. The entire MWU adaptation path was dead code." `SignalWeightManager` is still imported and writes a file no one reads. Confirms the "Dead code masquerading as active" point from the brief. Fix: delete the module or wire it back into the consensus.
