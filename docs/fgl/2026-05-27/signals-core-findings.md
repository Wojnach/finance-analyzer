# Signals-Core Adversarial Review — 2026-05-27

## Summary

| Severity | Count |
|----------|-------|
| P0       | 3     |
| P1       | 9     |
| P2       | 6     |
| P3       | 4     |
| **Total**| **22**|

**Top 3 themes:**
1. **Policy contradictions between code and rules**: MIN_VOTERS for metals is hardcoded at 2 in `signal_engine.py` but the project rule (`.claude/rules/signals.md`) requires 3 for all asset classes. Worse, even where 2 is allowed it is silently re-gated by `_dynamic_min_voters_for_regime` to 5 for ranging/unknown regimes — making the metals override an illusion in the most common regime.
2. **In-memory state with no persistence**: Persistence filter, sentiment hysteresis, IC cache, macro-window cache, promoted-signal cache, and ADX cache are all process-local. The system runs under `PF-DataLoop` with auto-restart-on-crash; every restart silently bypasses the 2-cycle stocks confirmation requirement and resets sentiment hysteresis.
3. **Activation-rate / bias arithmetic uses denominator that includes HOLD**: `signal_activation_rates` records `samples = total` (including HOLD). `_weighted_consensus` then compares `signal_samples >= _BIAS_MIN_ACTIVE (30)` to decide if bias penalty applies. A signal with 1% activation rate and 30 total samples has 0.3 active votes — the bias penalty fires on essentially no evidence, producing 0.2x weight on the directional vote.

**Biggest risk:** The Stage 4 dynamic-MIN_VOTERS gate in `apply_confidence_penalties` (`signal_engine.py:3067`) silently overrides the metals MIN_VOTERS=2 setting in ranging/unknown regimes, forcing HOLD whenever active_voters < 5. The codebase comment at line 4086-4089 claims "metals lowered from 3 to 2 because old 3-voter floor produced 0 trades in 20 days" — but the dynamic-min downstream gate makes it ≥5 in ranging, so the change does not have the intended effect and metals will continue producing zero trades during ranging regimes. Real-money impact: the metals subsystem (the user's primary focus per CLAUDE.md) is dead in the most common regime.

---

## P0 — Critical

### [P0] `MIN_VOTERS_METALS = 2` directly violates `.claude/rules/signals.md`
**File:** `portfolio/signal_engine.py:1072` and `4090`
**Issue:** Module-level constant `MIN_VOTERS_METALS = 2` and the dispatch at line 4085-4090 elects this value for XAU/XAG. The project rule explicitly states "MIN_VOTERS = 3 for all asset classes." CLAUDE.md echoes the same: "MIN_VOTERS = 3 (all asset classes)."
**Impact:** Either (a) the rule is wrong and the code is right, or (b) the rule is binding and the code is shipping a 2-voter consensus for the user's highest-focus instruments. Either way the divergence is uncommunicated and any reviewer (human or LLM) consulting the rules file will draw the wrong conclusion. Combined with the next finding (Stage-4 dynamic-min override), the change also fails to achieve its stated goal of "enable metals trading in low-voter regimes."
**Fix:** Either revert `MIN_VOTERS_METALS` to 3 to match the rule, or update `.claude/rules/signals.md` and CLAUDE.md to document and justify the 2-voter exception for metals. Add an assertion at module load that mirrors whichever choice you make.
**Confidence:** high

### [P0] Stage-4 `_dynamic_min_voters_for_regime` silently overrides metals MIN_VOTERS=2 in ranging/unknown
**File:** `portfolio/signal_engine.py:2008-2027` and `3067-3073`
**Issue:** `_dynamic_min_voters_for_regime` returns 5 for `ranging`, `unknown`, and `None` regimes. Stage 4 of `apply_confidence_penalties` reads `active_voters = extra_info["_voters_post_filter"]` (the post-persistence count) and force-HOLDs when `active_voters < dynamic_min`. For metals in ranging regime (the dominant regime per the codebase commentary), `min_voters=2` lets the consensus pass the line-4095 gate — only to be force-HOLD'd at Stage 4 because 2 < 5. The whole point of the metals carve-out (line 4086-4089: "metals lowered from 3 to 2 because the 3-voter floor produced 0 trades in 20 days") is defeated.
**Impact:** Metals subsystem stays HOLD in ranging/unknown regimes regardless of the per-asset min_voters override. Given that metals are described in CLAUDE.md/MEMORY.md as the primary focus and "user expects 0 trades in 20 days as evidence the gate was too strict," the override is intended to fix a known bug but does not. Real-money consequence: missed metals trade signals across the most common regime.
**Fix:** Either (a) extend `_dynamic_min_voters_for_regime` to be asset-class aware (return `MIN_VOTERS_METALS` instead of 5 for metals tickers in ranging), or (b) drop Stage 4 dynamic-min entirely for tickers whose own `min_voters` is below it. Add a test that asserts a 2-voter metals consensus in ranging regime emits BUY/SELL, not HOLD.
**Confidence:** high

### [P0] `_apply_persistence_filter` loses all state on process restart, bypassing 2-cycle stocks confirmation
**File:** `portfolio/signal_engine.py:598` (`_persistence_state: dict[...] = {}`) and `629-688`
**Issue:** Persistence state is stored in a module-level dict with no disk persistence. The `PF-DataLoop` scheduled task is configured "logon + auto-restart (30s delay)" per CLAUDE.md, and crashes / merges trigger restarts via `schtasks /run`. After every restart the persistence dict is empty; the first cycle for each ticker hits the cold-start branch at line 649 and returns `votes` unfiltered — even for stocks where `min_cycles=2`. The comment at line 658 ("first cycle — trust all signals") admits this is by design at cold start, but the policy "stocks require 2 consecutive same-direction votes" is silently violated every time the loop restarts.
**Impact:** A flickering single-cycle signal on a stock that would normally be filtered out can drive a non-HOLD consensus immediately after restart. Combined with auto-restart-on-crash, this can chain: every crash window emits one "trusted" cycle of unfiltered votes before the filter re-engages. Trades emitted in that window violate the documented 2-cycle policy and would not have been emitted in steady state.
**Fix:** Persist `_persistence_state` via `atomic_write_json` after every update (or batched once per cycle like sentiment state). Load on first call. Alternatively, change cold-start semantics to treat first-cycle votes as `cycles=1` for stocks so they still need a confirmation cycle before voting.
**Confidence:** high

---

## P1 — Important

### [P1] `_BIAS_MIN_ACTIVE = 30` is compared against `total` (incl. HOLD), not active vote count
**File:** `portfolio/signal_engine.py:538, 2734-2741`; `portfolio/accuracy_stats.py:850`
**Issue:** `signal_activation_rates` records `"samples": total` where `total` is the count of all votes including HOLD (line 807 of `accuracy_stats.py`). `_weighted_consensus` reads `signal_samples = act_data.get("samples", 0)` and uses it as `signal_samples >= _BIAS_MIN_ACTIVE` to decide whether to apply the bias penalty cascade. The comment on `_BIAS_MIN_ACTIVE = 30` explicitly says "need enough active votes to judge bias" — but the value being compared is total-including-HOLD, not active votes.
**Impact:** Rare-activation signals (e.g., a 1% activation signal with 3000 total samples = 30 active) cross the threshold at the same time as a high-activation signal (e.g., a 80% signal with 38 total samples = ~30 active). For low-activation signals, the bias penalty fires on essentially no directional evidence — emitting 0.2x weight on the few directional votes the signal does produce. Practical consequence: contrarian rare signals are over-penalized.
**Fix:** Either record `samples = buy + sell` in `signal_activation_rates`, or change `_weighted_consensus` to compute `signal_samples = int(act_data.get("activation_rate", 0) * act_data.get("samples", 0))`. Add a unit test asserting bias penalty is not applied for activation_rate=0.01 with 30 total samples.
**Confidence:** high

### [P1] `_dynamic_min_voters_for_regime` returns the strict 5-voter floor for `None`/`"unknown"` regime — the default state at cold start
**File:** `portfolio/signal_engine.py:2008-2027` and the call sites at `3067`, `2085`
**Issue:** When `regime` is `None` (e.g., on first cycle before `detect_regime` resolves) or `"unknown"` (the default fallback for `detect_regime` failures), the dynamic-min returns 5. Stage 4 then force-HOLDs any consensus with fewer than 5 voters. On freshly-started loops or during regime-detection failures, this produces HOLD for every ticker except crypto (which routinely has 5+ voters); metals and stocks will be HOLD regardless of signal strength until regime is resolved.
**Impact:** Every restart produces a HOLD-only window for metals/stocks. Combined with the persistence-filter cold-start (P0 above), the system behavior immediately after restart is a confusing mix of "trust everything (persistence)" + "force HOLD everything (dynamic_min)" that effectively means metals/stocks emit no signals.
**Fix:** Default to `min_voters` of the asset class when regime is unknown, rather than the strictest value. Or short-circuit Stage 4 when regime is unknown / first-cycle.
**Confidence:** high

### [P1] `signal_history.py` lock is process-local but multiple processes (main, metals, crypto, oil, MSTR loops) write the same `signal_history.jsonl`
**File:** `portfolio/signal_history.py:30, 77-98`
**Issue:** The lock `_history_lock` serializes the read-modify-write of `signal_history.jsonl` within one Python process. CLAUDE.md describes at least five long-running processes (PF-DataLoop, PF-MetalsLoop, PF-CryptoLoop, PF-MstrLoop, PF-OilLoop) plus the dashboard, all of which may import `portfolio.signal_history`. There is no file-level lock (cf. `outcome_tracker.py:281-339` which uses an msvcrt/fcntl sidecar lock for the signal_log JSONL). The atomic write at line 48 makes the FINAL write atomic, but two concurrent writers each loading 50 entries, both appending an entry, both trimming, and both writing back will lose one entry to last-writer-wins.
**Impact:** Persistence scores, noisy-signal lists, and signal streaks computed from `signal_history.jsonl` use a corrupted history during overlap windows. The corruption is silent — no log line indicates the lost write.
**Fix:** Replace `_history_lock` with the cross-process sidecar lock pattern used in `outcome_tracker._hold_signal_log_lock`. Or move signal-history persistence into the SQLite signal_db where SQLite's WAL handles concurrent writers.
**Confidence:** high

### [P1] IC computation `_rolling_ic` uses overlapping windows, ICIR std drastically underestimated
**File:** `portfolio/ic_computation.py:130-160`
**Issue:** `_rolling_ic` slides a 50-sample window by 1 sample at a time. Each consecutive window shares 49/50 samples; the resulting IC values are extremely autocorrelated. `compute_signal_ic` then computes `icir = ic_mean / ic_std` from this list. Because std is computed on a highly autocorrelated series, ic_std is artificially small, making ICIR artificially large.
**Impact:** Inflated ICIR values fool the `_IC_STABILITY_MIN = 0.10` gate in `signal_engine._compute_ic_mult`. Signals with no real predictive stability get full IC-boost (up to 1.5x weight). The "stability check" intended to gate noisy signals is broken.
**Fix:** Either slide the window by `window // 2` (non-overlapping or minimally-overlapping) for std purposes, or compute ICIR as `ic / SE(ic)` using the Spearman SE formula on the full sample. Add a regression test verifying ICIR for a random-vote signal is < 0.5 (currently can pass 1.0+).
**Confidence:** high

### [P1] `train_signal_weights._load_signal_history` reads JSONL only, never the SQLite signal_db that production writes
**File:** `portfolio/train_signal_weights.py:30-98`
**Issue:** Loads `_SIGNAL_LOG = data/signal_log.jsonl` via `load_jsonl`. But the production write path in `outcome_tracker.log_signal_snapshot` dual-writes to both JSONL and SignalDB (line 159-166), and `accuracy_stats.load_entries` prefers SQLite when available (line 144-164). If JSONL is rotated/truncated/missing while SQLite has data, training will silently fail with "Insufficient entries" even though the canonical store has plenty of history.
**Impact:** Linear-factor model never retrains; recommended_weights become stale; the system stops adapting weights over time. Detected only by manual inspection of the training script's output.
**Fix:** Call `accuracy_stats.load_entries()` instead of `load_jsonl(_SIGNAL_LOG)` so the same SQLite-first fallback applies. Or assert at training start that SQLite count matches JSONL count.
**Confidence:** high

### [P1] `cusum_accuracy_monitor` docstring claims "3-7 observations" detection latency but MIN_OBSERVATIONS=20 + 10-obs cooldown enforce ~30 observations
**File:** `portfolio/cusum_accuracy_monitor.py:1-22, 42, 104`
**Issue:** Module docstring: "online detection that can catch accuracy shifts within 3-7 observations." Actual gate at line 104: `if sig["n"] >= MIN_OBSERVATIONS and sig["n"] > sig.get("last_alert_n", 0) + 10`. First alert needs n=20 (so 20 outcomes); subsequent alerts need 10 more observations on top of the previous alert.
**Impact:** At horizon=1d, 20 outcomes means 20 calendar days for the first alert and 10 calendar days between alerts. This is no faster than the batch `accuracy_degradation` daily check, defeating the entire point of online CUSUM. Operators relying on this for "fast detection" are misinformed.
**Fix:** Either reduce `MIN_OBSERVATIONS` to 5-7 (matching the docstring) with a stricter `CONTROL_LIMIT_H` to maintain false-positive rate, or update the docstring to honestly describe the ~20-30 observation detection latency.
**Confidence:** high

### [P1] `signal_accuracy` uses `outcome.get("change_pct", 0)` default — silently treats missing key as flat outcome
**File:** `portfolio/accuracy_stats.py:225`
**Issue:** `change_pct = outcome.get("change_pct", 0)` — if a corrupt outcome dict is missing the `change_pct` key, default is 0, which is < `_MIN_CHANGE_PCT = 0.05`, so `_vote_correct` returns None and the outcome is silently dropped. The defensive `if change_pct is None: null_change_pct_skipped += 1` logging at line 226-227 only fires for explicit None, NOT for missing keys.
**Impact:** A bad outcome backfill that writes outcomes without `change_pct` (e.g., partial schema migration, network failure mid-write) is invisible to the data-quality monitor. The accuracy denominator silently shrinks; alerts based on accuracy drift fire spuriously.
**Fix:** Change default sentinel to `None` and let the existing None-handling apply (`outcome.get("change_pct")` without default). Also surface "missing change_pct key" as a distinct counter from "None change_pct."
**Confidence:** medium

### [P1] `direction_probability_with_forecast` accepts `chronos_24h_pct=None` silently and continues with pct_move=0
**File:** `portfolio/ticker_accuracy.py:219-228`
**Issue:** `pct_move = forecast_data.get(pct_key, 0) or 0`. When Chronos returns explicit `None` (model failed/abstained), `or 0` collapses to 0, then `abs(pct_move) > 0.1` is False, so forecast is silently not blended. No telemetry distinguishes "Chronos abstained" from "Chronos predicted ~0% move."
**Impact:** When Chronos is misconfigured / OOM / silently failing, the forecast simply stops contributing to consensus with no alert. Combined with the silent retries elsewhere, a multi-day Chronos outage is undetectable from forecast_blended counts alone.
**Fix:** When `pct_move is None`, set `base["forecast_blended"] = False` with a distinct reason field (e.g., `forecast_skip_reason = "chronos_abstain"`) instead of falling through to the `abs(pct_move) > 0.1` check.
**Confidence:** medium

### [P1] `shadow_registry.is_promoted` cache races between threads — possible duplicate disk reads
**File:** `portfolio/shadow_registry.py:251-273`
**Issue:** `_PROMOTED_CACHE` is a module-level mutable dict with no lock. `is_promoted` performs `if now - _PROMOTED_CACHE["loaded_at"] > _PROMOTED_TTL_S` then `load_registry()` then writes both fields. Two threads racing through this can both miss, both read, both write. While the writes are idempotent (both produce the same frozenset), the disk I/O is duplicated under contention and adds latency to the hot dispatch path.
**Impact:** Hot path `signal_engine` calls `is_promoted` per signal per ticker per cycle — under ThreadPoolExecutor with 8 workers, 5 tickers × 80 signals = 400 calls/cycle, each potentially triggering a duplicate disk read every 60s. Cumulative I/O cost grows with signal count.
**Fix:** Add a `threading.Lock` around the TTL-check-and-write path (same pattern used in `_macro_window_cache`).
**Confidence:** medium

---

## P2 — Correctness gaps requiring rare conditions

### [P2] Sentinel `_FAILED_IMPORT_COOLDOWN = 300` per-entry but `signal_registry.load_signal_func` mutates the registry dict shared across threads without a lock
**File:** `portfolio/signal_registry.py:42-69`
**Issue:** `load_signal_func` reads `entry.get("func")`, may set `entry["func"] = _FAILED_IMPORT_SENTINEL`, `entry["_fail_ts"] = ...`, etc. The `_ENHANCED_SIGNALS` dict and its entries are mutated from multiple ThreadPoolExecutor threads without a lock. Two threads racing the first call to a freshly-registered signal can both attempt the import and both write to the entry dict, with last-writer-wins on the (`func`, `_fail_ts`) pair.
**Impact:** On a transient ImportError (e.g., file IO interruption), one thread may write the sentinel while the other writes the successfully-loaded function — non-deterministic which wins. In practice modules are loaded once at startup before threads fan out, so this rarely manifests, but the dict-mutation contract is unsafe.
**Fix:** Add a per-module `threading.Lock` around the import block, or do an upfront synchronous import of all registered signals at process start.
**Confidence:** medium

### [P2] Per-ticker override `_DISABLED_SIGNAL_OVERRIDES` set is correctly typed, but no validation that the (signal, ticker) tuple actually has a ticker in the active universe
**File:** `portfolio/signal_engine.py:698-713`
**Issue:** The override set hardcodes pairs like `("williams_vix_fix", "XAU-USD")`. If a ticker is removed from the active set (e.g., the way AMD/GOOGL were removed per CLAUDE.md), the override silently keeps applying to a dead ticker. No load-time validation cross-references against `ALL_TICKERS`.
**Impact:** Drift over time — overrides accumulate for tickers that no longer exist, polluting future audits. Currently small but grows with each ticker rotation.
**Fix:** Add a module-load assertion: `assert all(t in ALL_TICKERS for _, t in _DISABLED_SIGNAL_OVERRIDES)`.
**Confidence:** medium

### [P2] `meta_learner._build_features` silently skips rows with JSONDecodeError, no telemetry on how many
**File:** `portfolio/meta_learner.py:127-131`
**Issue:** `except (json.JSONDecodeError, TypeError): continue`. If schema changes break the JSON in many rows, training silently proceeds on the subset of parseable rows without warning. The "Insufficient data" guard at line 211 may still pass, producing a model trained on a biased subset.
**Impact:** Silent training-data degradation. Operator sees "model trained successfully" but model was trained on, e.g., only the last week's data because schema migration broke older rows.
**Fix:** Count `skipped_rows`; log a warning if > 1% of total. Add to metrics dict.
**Confidence:** medium

### [P2] `accuracy_degradation._is_econ_blackout` returns False on import failure (line 322), masking a missing `portfolio.econ_dates`
**File:** `portfolio/accuracy_degradation.py:317-323`
**Issue:** `except Exception: return False`. If `portfolio.econ_dates` is missing or has an import error, the degradation check proceeds as if there's no blackout — potentially firing false-positive degradation alerts during a real macro event. Failure mode is silent.
**Impact:** During an econ event week, a real and expected accuracy drop from macro noise produces an alert that should have been suppressed by the blackout. Telegram spam and operator desensitization.
**Fix:** Log the import failure at WARNING with `exc_info=True`; consider fail-closed (treat unknown blackout state as "blackout active" to suppress alerts).
**Confidence:** medium

### [P2] Macro-window force-HOLD applied twice — once in `generate_signal` (line 4058) and once in `_weighted_consensus` (line 2402)
**File:** `portfolio/signal_engine.py:2402-2407` and `4058-4062`
**Issue:** Both mutations are idempotent (HOLD → HOLD is a no-op), but the double-application is a maintenance hazard: a future contributor changing one site has to remember the other. The comment at line 4055 acknowledges the duplication but offers no enforcement.
**Impact:** Minor; harmless today. But if `MACRO_WINDOW_FORCE_HOLD_SIGNALS` ever changes to "downweight, not HOLD," one site might be updated and the other forgotten, producing inconsistent behavior. Detection would require carefully reading two locations.
**Fix:** Extract the mutation into a helper `_apply_macro_window_force_hold(votes)` called from both sites, or leave a single canonical site and remove the defense-in-depth duplicate.
**Confidence:** medium

### [P2] `signal_weight_optimizer.WalkForwardResult` round-trips lists where the dataclass declares `list[tuple[str, float]]`
**File:** `portfolio/signal_weight_optimizer.py:39, 164-170`
**Issue:** `signal_rankings: list[tuple[str, float]] = field(default_factory=list)`. `to_dict()` returns this as-is; JSON serialization converts tuples to lists; `load_results(**data)` rebinds them as lists. Downstream consumers expecting tuple unpacking (`for name, score in signal_rankings`) work because list unpacks too, but the type annotation lies.
**Impact:** Cosmetic; future tooling that pattern-matches on tuple-vs-list types will misfire.
**Fix:** Either change annotation to `list[list]` or apply `[tuple(x) for x in data['signal_rankings']]` in `load_results`.
**Confidence:** medium

---

## P3 — Code smell / future hazard

### [P3] `signal_engine._weighted_consensus` returns `("HOLD", round(max(buy_conf, sell_conf), 4))` at line 2773 — a non-zero confidence on a HOLD outcome
**File:** `portfolio/signal_engine.py:2773`
**Issue:** When buy_conf == sell_conf or both < 0.5, the function returns HOLD with confidence = max(buy_conf, sell_conf). HOLD-with-confidence semantically conflates "no signal" with "signal but tied/inconclusive." Other code paths (e.g., `_confluence_score`) document HOLD as the absence of a signal (confidence 0.0). The `extra_info["_weighted_confidence"]` then carries a non-zero number that downstream loggers may display as if it were a directional confidence.
**Impact:** Cosmetic confusion in dashboards / journals; semantic mismatch with `signal_utils.majority_vote` (which returns 0.0 for HOLD ties at line 124-126).
**Fix:** Return `("HOLD", 0.0)` for ties, matching the `majority_vote` convention.
**Confidence:** medium

### [P3] `_compute_dynamic_correlation_groups` walks the entire signal_log every 7200s — no incremental update
**File:** `portfolio/signal_engine.py:1685-1776`
**Issue:** Every 2 hours, this function loads all signal_log entries (`load_entries()` reads SQLite), builds a DataFrame of `recent` (30 days), iterates `len(active_signals)²/2 ≈ 1300` pairs, and recomputes groups from scratch. Acceptable today but scales poorly as signal log grows.
**Impact:** Sub-second today but the 30-day window will eventually contain 30 × 1440 × 5 = 216K entries; the iteration cost grows linearly. Not urgent.
**Fix:** Optionally cache the agreement-rate matrix incrementally, evicting only the oldest 24h on each refresh. Track the recompute wall time and alert if it exceeds a budget.
**Confidence:** medium

### [P3] `_validate_signal_result` clamps confidence to [0, max_confidence] but does not clamp the `sub_signals` dict — a poisoned `sub_signals` value can pollute `extra_info` and downstream consumers
**File:** `portfolio/signal_engine.py:1615-1650`
**Issue:** Validation only checks `action`, `confidence`, and that `sub_signals` is a dict. Values inside `sub_signals` are not validated. A buggy signal that puts NaN, inf, or None into sub_signals will pass through to `extra_info[f"{sig_name}_sub_signals"]` and into the JSON snapshot.
**Impact:** Snapshot writes may JSON-serialize NaN as "NaN" (invalid JSON in some loaders), or downstream consumers reading sub_signals values may crash on arithmetic.
**Fix:** Deep-coerce sub_signals values via `_safe_accuracy`-style normalization, or document that sub_signals is "informational only, not for arithmetic."
**Confidence:** low

### [P3] `correlation_priors.CORRELATION_PRIORS` is a `dict[tuple[str, str], float]` but the lookup is order-independent — easy to silently duplicate
**File:** `portfolio/correlation_priors.py:8-12`
**Issue:** Both `("BTC-USD", "ETH-USD"): 0.75` and a hypothetical `("ETH-USD", "BTC-USD"): 0.70` would be considered valid dict entries (different keys), but `get_prior` checks both orderings. Conflicting pairs would silently use whichever ordering the caller passes.
**Impact:** Future maintainer adding pairs may accidentally insert both orderings with different values; the actual prior used depends on caller arg order, which is non-obvious from reading the file.
**Fix:** Add a load-time assertion that no `(b, a)` key exists when `(a, b)` is present, or canonicalize to alphabetical ordering on insert.
**Confidence:** medium
