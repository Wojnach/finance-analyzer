# Claude adversarial review: signals-core

## Summary

The subsystem is functionally rich but **leaky** — defense layers stack up to cover earlier
regressions while several silent-failure exception swallowers, a few atomic-I/O violations,
and at least one race in the JSONL hot path remain. Math is broadly correct on accuracy
gating but the IC pipeline embeds look-ahead bias and IC-buy/IC-sell are mislabeled.
SQLite handling is the worst hotspot: one connection shared across threads.

## P0 — Blockers (production breakage / data loss / silent wrong trades)

- `portfolio/signal_db.py:31-37` — single `sqlite3.connect()` cached on `self._conn`, then
  shared across all threads. **Why it bites:** `sqlite3.Connection` is not safe for
  multi-threaded use by default (Python raises `ProgrammingError: SQLite objects created
  in a thread can only be used in that same thread.`). The 8-worker ThreadPoolExecutor in
  `main.py` plus the dashboard's per-request `SignalDB()` instances (each calling
  `_get_conn()`) means the *first* signal_db.py method invoked by a non-owning thread will
  raise, but the surrounding `try/except Exception` in `load_entries` and
  `log_signal_snapshot` swallows it (line 157, line 165). The result is silent fallback to
  JSONL plus a lost-write on the SQLite leg — exactly the dual-write divergence the
  comment at outcome_tracker.py:166 (`"SQLite may lag"`) admits. **Fix:** open per-call
  with `check_same_thread=False` and use a `threading.local()` connection cache, or open
  a fresh connection per public method and close in `finally`.

- `portfolio/ic_computation.py:73-147` — IC is computed over `entries` that **already
  include each signal's vote AND the realized outcome at horizon H** drawn from
  `outcomes[ticker][horizon]`. `compute_signal_ic` then feeds (vote_sign, change_pct) pairs
  through `_spearman_rank_correlation` and applies a **rolling IC window of 50 samples
  taken from the same list in input order** (line 130, `_rolling_ic`). The list is sorted
  *neither* chronologically *nor* per-ticker — pairs from different tickers and different
  cycles interleave. ICIR is then computed as `mean/std` of that rolling list and used in
  `signal_engine._compute_ic_mult` to multiplicatively boost weights up to 1.5x.
  **Why it bites:** The "rolling" window is meaningless (no time order) and the IC is
  computed on the same population the signal is later weighted against — classical
  look-ahead. A 1.5x weight ramp from a Spearman ρ ≈ 0.10 derived this way is structural
  noise, not signal. Worse, `ic_buy` / `ic_sell` at lines 125-128 are **average returns
  for BUY/SELL votes**, NOT information coefficients — yet the field name is `ic_buy`,
  and downstream code (signal_engine.py uses `_ic_info.get("ic")` so the mislabel is at
  least scoped, but the per-ticker dict (line 215-218) writes the average return into a
  field called `ic`). **Fix:** sort entries by `ts` per ticker, hold out test set, compute
  IC on chronologically-prior data only. Rename `ic_buy`/`ic_sell` to `avg_return_buy`/
  `avg_return_sell` (they are not IC values).

- `portfolio/signal_decay_alert.py:34-39` — **atomic-I/O violation**:
  ```python
  with open(accuracy_cache_path, encoding="utf-8") as f:
      cache = json.load(f)
  ```
  This is a raw read of `data/accuracy_cache.json`, written concurrently by
  `accuracy_stats.write_accuracy_cache` (which uses `_atomic_write_json`). On the daily
  rename race, the reader can hit an empty path or a half-written file and either
  blow up with `FileNotFoundError` (caught, returns `[]` — *silent* miss) or read a torn
  JSON (caught as `JSONDecodeError` — *silent* miss). **Why it bites:** the decay
  detector is the watchdog for accuracy regressions; if it silently noops because of a
  read race against the writer it is meant to monitor, you have a single-point-of-failure
  monitoring loop. Violates CLAUDE.md rule 4. **Fix:** use `file_utils.load_json`.

- `portfolio/signal_history.py:64-98` — `update_history` is called from the per-cycle
  ThreadPoolExecutor (5 tickers × 8 workers) with `_history_lock`, but `_load_history()`
  inside the lock calls `load_jsonl(HISTORY_FILE)` which streams from disk **while the
  satellite loops (`crypto_loop.py`, `oil_loop.py`, `metals_loop.py`) and outcome_tracker
  also touch `signal_history.jsonl` from different processes**. The `threading.Lock`
  serializes *in-process* writers; it does NOT coordinate with other processes. The
  `_save_history()` call rewrites the entire file every call (no atomic_append) so a
  cross-process race produces a last-writer-wins truncation. **Why it bites:** persistence
  scores feed `get_noisy_signals` which gates downstream weighting; corruption is invisible
  unless someone diffs disk size. **Fix:** add the same sidecar-lockfile dance used by
  `outcome_tracker._hold_signal_log_lock`, or switch to `atomic_append_jsonl` + bounded
  trimming as a separate maintenance pass.

## P1 — High (will cause incidents)

- `portfolio/signal_engine.py:566-625` — `_apply_persistence_filter` mutates
  `_persistence_state[ticker]` in place under lock and also calls
  `_persistence_state[ticker] = {...}` on first entry. On cold start (line 591-595) the
  initialization writes `cycles: min_cycles if vote != "HOLD" else 0` — but the *current*
  cycle's votes are also returned unfiltered. That's documented intent. The subtle bug:
  the seed counts a HOLD vote's `cycles` as 0 but a directional vote as `min_cycles`,
  i.e. **the first cycle's directional vote is implicitly treated as 1 or 2 cycles old**.
  Then on cycle 2, the SAME-direction branch at line 619 increments cycles → 3 (or 2),
  but the FLIPPED branch resets to 1. The net effect is asymmetric: a same-direction
  vote on cycle 2 passes immediately even though the policy says "2 consecutive
  cycles". **Why it bites:** the doc claims persistence filters single-check noise, but
  the seed defeats it for the very first cycle that matters after a process restart.
  Each restart re-trusts every signal's first directional vote, which the system itself
  treats as suspicious noise per memory/proven-signal-patterns. **Fix:** seed at
  `cycles: 1 if vote != "HOLD" else 0` so the policy's "2 consecutive" semantics hold
  identically across restart vs steady state.

- `portfolio/signal_engine.py:3700-3724` — `regime_gated` is computed with
  `_get_regime_gated(regime, horizon)` then **mutated to `regime_gated_effective`**
  by removing per-ticker exempt signals AND by removing signals with strong recent
  accuracy. Then at line 4023 the function passes `regime_gated_effective` to
  `_weighted_consensus` as `regime_gated_override`. Good. **But** the per-ticker
  exemption at line 3732 only requires `t_samples >= 50` and `t_acc >= 0.60`. The
  `_ticker_acc_data` is `accuracy_by_ticker_signal_cached(acc_horizon)` (line 3705)
  whose horizon defaults to "1d" when `horizon` is None or 12h/1d/etc. — *not* "3h".
  So a 3h prediction will exempt a regime-gated signal based on the signal's 1d accuracy
  on this ticker. **Why it bites:** the comment a few lines below (`acc_horizon = horizon
  if horizon in ("3h", "4h", "12h") else "1d"`) is correct *for the cache pick*, but the
  exemption is a directional bet for a different horizon than the one being decided.
  fear_greed at 93.8% on XAG-USD 1d may be 30% at 3h. **Fix:** lookup the actual horizon
  bucket. Mirror what the dual-bucket `_recent_acc_data` does and merge.

- `portfolio/accuracy_stats.py:316-334` — `signal_accuracy_ewma` parses
  `entry.get("ts", "")` via `datetime.fromisoformat(ts_str)` and then computes
  `age_days = (now - entry_dt).total_seconds() / 86400.0` where `now =
  datetime.now(UTC)`. If `entry_dt` is naive (any caller that writes a non-UTC ISO string)
  the subtraction raises `TypeError: can't subtract offset-naive and offset-aware
  datetimes`, **caught by `(ValueError, TypeError)` at line 331 → `continue`**, silently
  dropping that entry from the EWMA weight. **Why it bites:** the JSONL is supposed to
  carry UTC, but if even one writer (e.g. backtester, replay, a test harness, or a fixed
  bug from an older version) ever wrote naive timestamps, those entries are deleted from
  EWMA — biasing the weighting toward whichever subset is tz-aware. **Fix:** assume UTC
  when tz info missing (`entry_dt.replace(tzinfo=UTC)`); log at WARNING if you fall into
  that branch.

- `portfolio/signal_history.py:81-83` — `update_history` writes only signal names from
  `SIGNAL_NAMES`, defaulting missing keys to "HOLD". **Why it bites:** any
  dynamically-registered signal that isn't in the hardcoded `SIGNAL_NAMES` set (e.g.,
  the `btc_proxy` synthetic injected at signal_engine.py:3615, or newly added shadow
  signals) **never gets recorded**. Their persistence scores stay at the default 0.5
  forever, so `get_noisy_signals` can't ever flag them. The dynamic registry exists,
  but this consumer hardcodes against a frozen list. **Fix:** record every key in
  `votes_dict`, not just `SIGNAL_NAMES`.

- `portfolio/signal_weight_optimizer.py:111-114` — `np.corrcoef(predictions, test_y.values)`
  is called with raw test predictions that may have zero variance (e.g., all signals
  HOLD → all predictions equal `intercept`). `numpy` returns `nan` and the `not
  np.isnan(corr)` guard at line 113 silently drops the window. **Why it bites:** combined
  with `test_y.std() > 1e-10` (only guards the y side), entire prediction-degenerate
  windows are dropped from `oos_correlations`, then `avg_oos_corr` is computed only over
  surviving windows. The result is **survivorship-biased optimism** in walk-forward
  reporting. **Fix:** check `predictions` std too, and count dropped windows in
  `WalkForwardResult` for transparency.

- `portfolio/outcome_tracker.py:478-482` — `outcomes[ticker][h_key]` is set inside the
  loop, **but the lock around the JSONL rewrite (Phase 3) doesn't include the SQLite
  dual-write at line 488**. If the rewrite phase fails (rename error, disk full) the
  SQLite outcome rows have ALREADY been committed at line 491. **Why it bites:** the
  JSONL and SQLite legs diverge: SQLite says "outcome filled for this snapshot ts" while
  JSONL says "outcome empty, retry next cycle". `entries_missing_outcomes` at
  signal_db.py:231 then skips re-backfill for that snapshot, but `signal_accuracy()`
  reads from `load_entries()` which prefers SQLite and serves the orphaned outcomes ←
  this works in this direction but reverses on next backfill which re-fetches the
  historical price (Binance) and may get a different number on retry, producing
  duplicate-but-different outcome data. **Fix:** dual-write should happen *after* the
  successful `os.replace`, or be rolled into a single transactional unit.

- `portfolio/accuracy_stats.py:1145-1237` — `maybe_prewarm_dashboard_accuracy` uses
  `acquire_lock_file` from `portfolio.process_lock`, but on `except Exception` the
  fallback at line 1199-1200 sets the helpers to `None` and then proceeds with
  `fh = ... if acquire_lock_file else "noop"` → `fh == "noop"` → the `if fh is None`
  test passes (it's the string "noop") and the prewarm fires *without cross-process
  exclusion*. **Why it bites:** if `process_lock` is missing or partially-imported,
  the cross-process guarantee disappears silently. Two main loops (during a botched
  restart, per the comment) would both fire the 12-cache-compute fanout in parallel.
  **Fix:** treat the missing import as "must not fire" or at least log loudly.

## P2 — Medium (correctness / robustness)

- `portfolio/signal_engine.py:1352-1404` — `_compute_dynamic_horizon_weights` loads
  `accuracy_cache.json` via `load_json` (good) but then keys cross-horizons as
  `f"{ch}_recent"` (line 1369). The keys actually written by `write_accuracy_cache`
  for the recent variant use the same `_recent` suffix (`get_or_compute_recent_accuracy`
  at line 1050). So the read works, but the **denominator-ratio formula at line 1394
  `ratio = this_acc / cross_acc`** divides by zero when `cross_acc` is in the half-open
  range `[0.01, 1.0]` (the gate at line 1390) and `this_acc` is large. No actual ZeroDiv
  because the range is strictly positive, but a near-zero `cross_acc` (0.011) blows the
  ratio to ~90× and is then clamped at 1.5. The clamp masks the issue; the resulting
  multipliers carry no signal. **Fix:** require `cross_acc >= 0.30` or compute ratio in
  log space.

- `portfolio/signal_engine.py:1402-1404` — `except Exception: logger.debug(...,
  exc_info=True)` — DEBUG level swallow. If `accuracy_cache.json` is corrupted, the
  dynamic horizon weights silently fall back to the static dict; you only see it if you
  set DEBUG logging. Same pattern at lines 1629-1631, 2078, 2046-2048, 3673, 3707,
  3724, 3907, 3969, 3989, 4097, 4112, 4144, 4177. **Why it bites:** the codebase
  systematically demotes failure logs to DEBUG, which means production logs (INFO+)
  never surface them. Several of these mask correctness issues, not noise. **Fix:** at
  minimum, log at WARNING on first failure per process via a once-per-process flag,
  then DEBUG thereafter.

- `portfolio/accuracy_stats.py:863-969` — `blend_accuracy_data`'s directional-key merge
  (lines 953-967) picks the *larger-sample source per key independently*. That means
  for a signal with `at = {buy_accuracy: 0.30, total_buy: 1000}` and
  `rc = {buy_accuracy: 0.62, total_buy: 50}`, the result uses `at`'s 30% — sensible.
  But for `sell_accuracy` it would independently pick the larger-sample source there.
  **Why it bites:** the blended `accuracy` field at line 922 mixes all-time and recent,
  while the directional fields are NOT blended — they're a winner-take-all by
  sample size. A signal with 70% recent BUY accuracy (300 samples) but 35% all-time
  (5000 samples) ends up with `accuracy ≈ 0.51` (blended) but `buy_accuracy = 0.35`
  (all-time wins). Downstream `_weighted_consensus` at line 2517-2520 uses
  `buy_accuracy` as the actual weight — so the signal contributes 0.51 to gate decisions
  but 0.35 to vote weight. Inconsistent. **Fix:** blend directional accuracy with the
  same EWMA formula used for overall, or document the asymmetry.

- `portfolio/signal_engine.py:2095-2110` — `_weighted_consensus` accepts
  `soft_confidences=extra_info` at the call site (line 4027). `extra_info` is a dict
  with `_soft_conf_*` keys, but `extra_info` *also* contains every other key the engine
  has stuffed in (`fear_greed`, `volume_ratio`, `_voters`, `_buy_count`, etc.). The
  function only reads `soft_confidences.get(f"_soft_conf_{signal_name}")` so the noise
  is harmless — but the type hint and docstring imply a focused soft-conf dict. **Why
  it bites:** future contributor passes the same `extra_info` dict and finds a signal
  named "fear_greed" with `extra_info["fear_greed"] = 23` (an int), `float()` succeeds
  for `_soft_conf_fear_greed` would not be set, but anyone reading the function might
  accidentally key off `soft_confidences.get(signal_name)` (no prefix) and pick up
  `extra_info["fear_greed"] = 23` as a 23× weight multiplier. **Fix:** filter to
  `_soft_conf_*` keys at the call site, or pass a typed dict.

- `portfolio/signal_engine.py:3879-3881` — fail-closed branch on accuracy load failure:
  ```python
  accuracy_data = {sig: {"accuracy": 0.0, "total": 999} for sig in SIGNAL_NAMES}
  ```
  This gates EVERY signal (total=999 ≥ 30 and 0% < 0.47). Good. **But** the
  `_per_ticker_consensus_gate` later (line 4154) and the `apply_confidence_penalties`
  cascade run *after* this fail-closed setup — and they read `_ptc_data`/`_ticker_ptc`
  from `extra_info`, which is populated *inside* the `try` block at line 3858-3862.
  When the `try` block throws, `_ptc_accuracy` is never set, but `_compute_gate_relaxation`
  and the per-ticker consensus stages keep running with stale extra_info data from a
  prior generate_signal call. **Why it bites:** Python doesn't reset `extra_info` between
  ticker calls in the loop, but `extra_info` is local to the function. Still: the
  fail-closed branch returns `weighted_action = "HOLD", weighted_conf = 0.0` correctly,
  but the surrounding code paths assume `extra_info["_ptc_*"]` keys exist with valid
  data when present. Today they're absent, which `.get(...)` returns None for — but
  a future refactor that defaults `_ptc_accuracy = 0.5` would re-introduce the bug.
  **Fix:** explicit reset of `extra_info["_ptc_accuracy"] = None` at function entry.

- `portfolio/signal_engine.py:4182` — `conf = min(conf, 0.80)` applied **after**
  `apply_confidence_penalties` and **before** the cross-ticker cache write at line 4194.
  The cache stores `confidence: conf` capped at 0.80. **Why it bites:** the `btc_proxy`
  consumer at line 3617 reads back this confidence; it always sees ≤ 0.80 which is
  fine, but the `_raw_confidence` stored at 4050 is *also* the pre-cap value from the
  initial naive consensus path — the dashboard and any downstream consumer that
  compares `_raw_confidence` to `confidence` will see misaligned semantics. The
  `_weighted_confidence` field at 4065 is also stored BEFORE the cap. Three different
  "confidence" fields with three different cap states is a footgun. **Fix:** document
  which field is the post-cap source of truth.

- `portfolio/signal_db.py:33` — `sqlite3.connect(..., timeout=10)` is the only
  contention guard. No `isolation_level=None` (so implicit BEGIN; commit needed); no
  `WAL` autocheckpoint setting beyond the default. **Why it bites:** at 5 tickers ×
  8 workers + dashboard + outcome_tracker, the 10s timeout is the only thing standing
  between the loop and `OperationalError: database is locked`. The exception is caught
  by `try/except Exception` at outcome_tracker.py:492 and demoted to debug. **Fix:**
  per-thread connection (see P0) plus explicit `synchronous=NORMAL` for the WAL.

- `portfolio/accuracy_stats.py:1605` — indentation typo:
  ```python
              if delta <= max_delta_hours and (best_delta is None or delta < best_delta):
                      best = snap
                      best_delta = delta
  ```
  The body is indented an extra level — valid Python, just visually misleading. (Not a
  bug, P3-ish.)

- `portfolio/forecast_accuracy.py:152-156` — `if "_" in sub_name: sub_horizon =
  sub_name.split("_", 1)[1]`. A sub-signal named `chronos_1h` produces `sub_horizon =
  "1h"`. But `compute_forecast_accuracy(horizon="1h")` then matches it. The issue: any
  sub-signal name with an underscore for non-horizon reasons (e.g., `lstm_dropout_1h`)
  would split incorrectly. **Why it bites:** today's models are named consistently
  (`chronos_1h`, `chronos_24h`, etc.) but the registry has no guard against future
  models with multi-token names. **Fix:** use `rsplit("_", 1)` and validate against
  a known horizon allowlist.

- `portfolio/ic_computation.py:125-128` — variable name disaster:
  ```python
  ic_buy = sum(buy_returns) / len(buy_returns) if buy_returns else 0.0
  ic_sell = -sum(sell_returns) / len(sell_returns) if sell_returns else 0.0
  ```
  These are **average returns**, not information coefficients. The fields are then
  written to the cache as `ic_buy` / `ic_sell` at line 142. (See P0 above.)

- `portfolio/outcome_tracker.py:160-166` — `log_signal_snapshot` opens `SignalDB()` per
  call and closes it. Each call walks the SQLite schema migration check at line 81-84,
  which is harmless but wasted I/O. Then catches `Exception` at line 165 and logs
  warning. **Why it bites:** if SignalDB is broken, every cycle pays the migration cost
  plus the warning log. **Fix:** open-once at module level via a lazy property guarded
  by a lock.

## P3 — Low (style / dead code / minor)

- `portfolio/signal_weights.py:25-77` — `SignalWeightManager.batch_update` is referenced
  by the comment at outcome_tracker.py:497-500 as **dead code** ("C6: MWU weight update
  removed"). The file persists and is loadable but nothing in the engine reads
  `signal_weights.json`. Tag for removal.

- `portfolio/signal_decay_alert.py:13` — `import json` is present but only used inside
  the raw-read block (P0 above). Once the load is moved to `load_json`, this import
  can go.

- `portfolio/signal_postmortem.py:121-182` — `compute_vote_correlation` iterates pairs
  in nested `for i, s1` / `for s2 in active_names[i+1:]` — O(N²) per entry across a
  potentially 50K-entry signal log. With ~30 active signals that's ~435 pairs × 50K
  entries × 5 tickers = 100M pair-counts per call. Not a P2 because the function is
  only called from the postmortem path (not per-cycle), but at scale this is a memory
  pressure point. Use `itertools.combinations` and consider streaming.

- `portfolio/signal_engine.py:3038-3039` — `_set_last_signal(ticker, "__pre_dispatch__")`
  followed by `_reset_phase_log(ticker)`. Both are called only when `ticker` is truthy
  (line 3037 guard), but the WARNING-on-empty at line 3006-3011 still uses the empty
  ticker as a key. The warning log spam is bounded by caller behavior but not by code.

- `portfolio/feature_normalizer.py:32` — module-level `_buffers: dict` is shared across
  threads with no lock. The `deque(maxlen=N)` writes are atomic for single-append but
  the `update` + `normalize` pair is not. **Today this is unused in production hot path**
  (grep yields no callers in signal_engine), but if wired up via main.py per ticker
  thread without a lock it would corrupt the per-(ticker,indicator) distribution.

- `portfolio/signal_history.py:14` — `from portfolio.tickers import SIGNAL_NAMES` —
  reinforces the P1 hardcode bug above.

- `portfolio/short_horizon.py:13` — `SLOW_SIGNALS_3H = frozenset({"trend", "fibonacci",
  "macro_regime"})` — `fibonacci` was disabled globally (`DISABLED_SIGNALS` per the
  CLAUDE.md comment about 2026-04-29). The membership is redundant: a disabled signal
  doesn't vote anyway. Cleanup.

- `portfolio/signal_engine.py:108-114` — phase-log eviction strategy: "drop the first
  half by insertion order". Comment claims "LRU by last-reset is fine" — but insertion
  order in CPython 3.7+ is FIFO, not LRU. The eviction is actually FIFO. Minor mislabel.

## Tests missing

- No test verifies the SignalDB thread-safety claim. Add a `pytest -n auto` test that
  spawns 4 threads each calling `db.load_entries()` and `db.insert_snapshot()` and
  asserts no `ProgrammingError`.
- No test verifies the `accuracy_cache.json` reader race. Add a fixture that has one
  thread writing via `_atomic_write_json` in a loop while another calls
  `check_signal_decay` and asserts the latter never raises.
- No test verifies the persistence-filter seed (P1 above) behaves identically on cold
  start vs post-restart with state. Add: pre-populate `_persistence_state`, take one
  cycle, then `clear()` and rerun — same input should yield same filtered output.
- No test verifies IC look-ahead. Compute IC on a leaked (vote=outcome) synthetic dataset
  and verify it doesn't produce 1.5x boost; or split train/test by ts and verify the
  IC computed on train matches a separate train-only computation.
- No test exercises `blend_accuracy_data` when `at.total_buy` is large but
  `rc.buy_accuracy` is wildly different — verify the asymmetric merge documented in P2.
- No test for `_apply_persistence_filter`'s seed asymmetry across `min_cycles` 1 vs 2.
- No test for cross-process race on `signal_history.jsonl` (CryptoLoop + MetalsLoop +
  PF-DataLoop all touching the same file).

## Cross-cut observations

- **Silent-failure-as-design**. `try/except Exception: logger.debug(...)` is the
  dominant error-handling pattern (counted 30+ in signal_engine.py alone). The
  defensive intent is sound (don't crash the loop), but DEBUG hides failures from
  ops. The codebase needs a "log-once-WARNING-then-DEBUG" helper.

- **Cache TTL inconsistency**. Every cache picks its own TTL: 300s (signal_utility L1),
  3600s (signal_utility L2, accuracy, IC, dynamic_horizon, regime accuracy disk),
  7200s (dynamic_corr), 1800s (local model accuracy). No invalidation manifest. When
  outcome_tracker backfills, it invalidates only `signal_utility_cache`. IC,
  regime-accuracy, dynamic horizon, dynamic correlation, best_horizon caches are all
  stale for 1-2h after a backfill — meaning gates and weights run on yesterday's data
  for the rest of the day.

- **Dual-write divergence risk** between JSONL and SQLite. The SQLite leg is supposed
  to be authoritative for reads (`load_entries` prefers it) but the JSONL leg is the
  one protected by the sidecar lock in `outcome_tracker`. P0 SignalDB threading + P1
  ordering of dual-write commits compound to make the two stores drift, with the
  preference reversed depending on which read path is used.

- **MIN_VOTERS arithmetic is correct**. `active_voters = buy + sell` (line 3798), the
  3-voter floor is enforced (line 3811), and metals are explicitly relaxed to 2 with
  comment justification. The `_compute_gate_relaxation` circuit breaker correctly
  excludes the high-sample tier from relaxation (SC-P1-2 audit referenced in code).
  This is the most carefully-reviewed code path in the file.

- **No inversion bug found**. Per CLAUDE.md "NEVER invert sub-50% signals — gate them
  as HOLD." Searched the engine for any path that flips `BUY`↔`SELL` based on
  accuracy and found none. Force-HOLD via `gated_signals.append(signal_name); continue`
  is the consistent pattern. The "directional rescue" at line 2477-2495 is also
  correct (reduces weight, doesn't invert).
