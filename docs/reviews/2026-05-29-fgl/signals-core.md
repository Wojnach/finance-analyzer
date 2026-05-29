# Signals-Core Adversarial Review — 2026-05-29-fgl

Subsystem: signal voting engine + accuracy/weighting infrastructure.
Files reviewed at HEAD: signal_engine.py, signal_registry.py, signal_utils.py,
signal_weights.py, signal_weight_optimizer.py, train_signal_weights.py,
accuracy_stats.py, accuracy_degradation.py, ticker_accuracy.py, ic_computation.py,
correlation_priors.py, meta_learner.py, signal_decay_alert.py, signal_state_since.py,
signal_history.py, signal_db.py, shadow_registry.py, signal_postmortem.py,
cusum_accuracy_monitor.py, perception_gate.py, escalation_gate.py, tickers.py.

## Counts
- P0: 0
- P1: 4
- P2: 8
- P3: 5
- Total: 17

No loop-crash, data-corruption, or wrong-direction P0 was found. The consensus hot
path is heavily defended (deep accuracy_data sanitization, _safe_accuracy/_safe_sample_count
coercion, fail-closed accuracy gate, atomic I/O, per-cache locks). Findings below are
correctness/consistency drift and a few silent-failure / resource issues.

---

## P1 — incorrect math / silent divergence / missing guard

- portfolio/signal_engine.py:448: P1: `_ACCURACY_GATE_HIGH_SAMPLE_MIN = 7000` contradicts the
  high-sample gate documented as 10,000 in `.claude/rules/signals.md`, in CLAUDE.md ("Tiered:
  50% for 7K+ sample signals" vs "10,000+"), AND in this file's OWN comment at lines 442–446
  ("raised high-sample min 5000 -> 10000"). The code uses 7000 everywhere (lines 1993, 2650,
  4288), so signals with 7,000–9,999 samples are silently gated at 50% even though the rules
  doc says they gate at 47%. This is a live decision-driver divergence — operators reasoning
  from the rules file will mispredict which signals vote. → Reconcile: either set the constant
  to 10000 to match the documented contract, or fix the comment + `.claude/rules/signals.md` +
  CLAUDE.md to say 7,000. Pick one source of truth.

- portfolio/signal_db.py:288: P1: SQL `signal_accuracy()` counts `change_pct > 0` (BUY) /
  `change_pct < 0` (SELL) as correct with NO `_MIN_CHANGE_PCT` (±0.05%) neutral filter and NO
  None handling on the JOIN's `change_pct` (guarded only by `o.change_pct IS NOT NULL`). The
  Python `accuracy_stats.signal_accuracy()` (line 234 → `_vote_correct`) DOES skip ±0.05% as
  neutral. The two paths therefore report different accuracy for the same data. Same divergence
  in `consensus_accuracy()` (line 319) and `per_ticker_accuracy()` (line 347) and
  `ticker_signal_accuracy()` (line 387). Currently only reached from tests, but it is a latent
  trap: any future caller that uses the DB SQL accuracy will get systematically inflated
  numbers (every micro-move scored as a hit). Flagged in prior reviews (SYNTHESIS-2026-04-23,
  04-30) and still unfixed. → Route the SQL methods through the same neutral threshold
  (`ABS(change_pct) >= 0.0005`) or delete them and force callers to the Python path.

- portfolio/train_signal_weights.py:54: P1: `_load_signal_history` reads
  `load_jsonl(signal_log.jsonl)` directly, bypassing the SQLite-preferred `load_entries()` used
  everywhere else. If `signal_log.jsonl` is stale relative to `signal_log.db` (the canonical
  store per signal_db.py docstring — JSONL is the fallback), `train_weights()` trains the
  LinearFactorModel on stale/partial data and silently writes worse weights that then feed the
  live `_linear_factor` confidence boost/dampen in signal_engine.py:4456. No error surfaces. →
  Use `accuracy_stats.load_entries()` (SQLite-first) instead of raw `load_jsonl`.

- portfolio/signal_engine.py:4216: P1: the regime-accuracy overlay replaces
  `accuracy_data[sig_name] = rdata` with a dict that omits directional fields
  (`buy_accuracy`/`total_buy`/`sell_accuracy`/`total_sell`). The per-ticker override block
  immediately below (lines 4252–4255) deliberately copies those fields precisely to avoid this,
  but the regime overlay does not. Consequence: when a regime override fires for a signal, the
  directional accuracy gate and BUG-182 direction-specific weighting in `_weighted_consensus`
  silently fall back to overall accuracy, re-introducing the exact over-weighting BUG-182 was
  written to prevent (e.g. a signal 30% BUY / 75% SELL votes BUY at the regime overall weight).
  → Merge directional fields into `rdata` the same way the per-ticker block does, or only
  override the `accuracy`/`total` keys in-place rather than replacing the whole dict.

---

## P2 — robustness / consistency with correctness impact

- portfolio/escalation_gate.py:203: P2: on runner timeout the code calls `_fut.cancel()` but a
  future already running cannot be cancelled — the ministral thread keeps executing on the
  single-worker `_RUNNER_EXECUTOR` (max_workers=1). A genuinely hung llama-server call pins that
  worker, so every subsequent `should_escalate()` waits the full 10s timeout before failing
  open. Fails open (no missed trigger), but adds 10s latency per call during an outage. → Use a
  per-call executor or a larger pool, and treat a still-running future as poisoned (recreate the
  executor) rather than relying on `.cancel()`.

- portfolio/signal_db.py:31: P2: `SignalDB` caches a single `sqlite3.connect(...)` connection in
  `self._conn` with default `check_same_thread=True`. `load_entries()` constructs a fresh
  SignalDB per call and closes it, so the common path is safe, but any caller that holds a
  SignalDB instance and reads from multiple ThreadPoolExecutor workers (8 in the loop) will hit
  "SQLite objects created in a thread can only be used in that same thread". → Either pass
  `check_same_thread=False` with an explicit lock, or document/enforce per-thread instances.

- portfolio/accuracy_stats.py:939: P2: `blend_accuracy_data` sets `total = max(at_samples,
  rc_samples)` and derives `correct = int(round(blended*total))`. The downstream accuracy gate
  keys off this `total` as the min-samples check. A signal with e.g. 5 all-time + 35 recent
  reports total=35 and is treated as a mature, gateable signal at the blended accuracy even
  though the all-time leg is statistically empty. Combined with the catastrophic-floor / blend
  branches this is mostly intended, but `max()` (not sum or the leg that actually drove the
  blend) can overstate confidence in the leg that was discarded. → Use the sample count of the
  leg that actually determined `blended` (recent when recent-weighted, all-time otherwise), or
  document why `max` is correct.

- portfolio/signal_engine.py:2796: P2: final consensus emits a direction only when
  `buy_conf >= 0.5` (or sell). Because `total_weight = buy_weight + sell_weight`, buy_conf and
  sell_conf always sum to 1.0, so a 50/50 weight tie returns HOLD — correct — but a 0.5000…
  vs 0.4999… split from float rounding can flip on `round(buy_conf,4)` reporting vs the
  unrounded comparison, producing a BUY with reported confidence 0.5000 that the >=0.5 branch
  accepted on the unrounded value. Cosmetic at worst, but the reported confidence can read below
  the branch it claims to have passed. → Compare and report on the same rounded value.

- portfolio/ic_computation.py:123: P2: IC is Spearman rank correlation between a binary vote
  vector (only +1/-1; HOLD already excluded) and returns. Spearman on a two-valued x collapses
  to a point-biserial-like statistic where the rank structure of x carries almost no
  information — the resulting "IC" is a noisy proxy, and `_compute_ic_mult` (signal_engine.py
  2193) turns it into a ±50% weight swing (`1 + 2.0*ic`, clamped 0.6–1.5). A spuriously large
  IC on a thin sample can meaningfully move live weights. → Consider using the mean directional
  return (already computed as ic_buy/ic_sell) or a proper biserial formula, and/or raise
  `_IC_MIN_SAMPLES`/`_IC_STABILITY_MIN`.

- portfolio/signal_engine.py:4146: P2: `acc_horizon = horizon if horizon in ("3h","4h","12h")
  else "1d"` collapses 3d / 5d / 10d consensus onto 1d accuracy stats (the inline TODO at
  4144 acknowledges this). The per-horizon blacklist (`_TICKER_DISABLED_BY_HORIZON`) and
  HORIZON_SIGNAL_WEIGHTS DO have 3d/5d entries, so weighting and accuracy gating disagree about
  which horizon they're scoring — a signal good at 1d but bad at 5d votes with 1d accuracy at
  the 5d horizon. → Build per-horizon accuracy caches for 3d/5d/10d, or document that those
  horizons intentionally borrow 1d accuracy.

- portfolio/signal_engine.py:4279: P2: utility boost multiplies blended accuracy by up to 1.5x
  (`min(raw_acc*boost, 0.95)`) using `signal_utility(acc_horizon)` whose cache has a pure-TTL
  (300s L1 / 3600s L2) invalidation with NO mtime check vs signal_log (documented at
  accuracy_stats.py:49). After an outcome backfill the boost can lag reality by up to an hour,
  inflating a signal whose return profile just degraded. The gate at 4292 protects against
  boosting an already-gated signal, but a barely-passing signal (e.g. 0.48 acc at a relaxed
  gate) can be boosted to 0.72 on stale utility. → Acceptable per the documented daily-backfill
  cadence, but consider mtime-gating the utility cache or calling
  `invalidate_signal_utility_cache()` from the backfill path (it already exists for exactly
  this).

- portfolio/meta_learner.py:104: P2: `_load_data`/`predict` open raw `sqlite3.connect(SIGNAL_DB)`
  and `joblib.load` the model file with a `mtime`-keyed module cache `_model_cache` that is read
  and written without a lock; concurrent ticker threads calling `predict()` for different
  horizons race the dict (benign in CPython but a torn (model, mtime) tuple read is possible if
  a future contributor mutates in place). Read-only DB access is fine. → Guard `_model_cache`
  with a lock if predict() is ever called from the 8-worker pool.

---

## P3 — nits with minor or no correctness impact

- portfolio/ic_computation.py:127: P3: `ic_buy`/`ic_sell` are labeled "ic" but are mean
  directional returns, not information coefficients. They are unused by `_compute_ic_mult` so
  there's no live impact, but the naming is misleading for anyone wiring them in later. → Rename
  to `mean_buy_return`/`mean_sell_return`.

- portfolio/signal_engine.py:566: P3: `BIAS_POLICY_VERSION = "2026-05-19"` is stamped manually
  and the docstring says "Bump on any change to the constants above," but nothing enforces it;
  the bias thresholds can drift without the version moving. → Derive the version from a hash of
  the bias constants, or add a test asserting they move together.

- portfolio/signal_weights.py:1: P3: `SignalWeightManager` MWU weights are persisted and
  `get_normalized_weights` is public, but a grep shows no live consensus caller — the class is
  documented as "only called from the single-threaded outcome backfill path." If that's no
  longer true it's an unguarded read-modify-write; if it IS dead in consensus, it's drift. →
  Confirm the caller set and either remove or wire it in deliberately.

- portfolio/accuracy_stats.py:344: P3: `signal_accuracy_ewma` reads `change_pct =
  outcome.get("change_pct", 0)` (default 0, not None) before `_vote_correct`. A genuinely
  missing field becomes 0 → treated as neutral/skip rather than "unknown", which is the right
  outcome but for the wrong reason (a real 0.0% move and a missing field are conflated). →
  Default to None to match `_compute_signal_utility` (line 695) and `compute_signal_ic`.

- portfolio/signal_registry.py:62: P3: `load_signal_func` catches bare `Exception` on import and
  caches `_FAILED_IMPORT_SENTINEL` for 300s — correct for resilience, but a syntax error in a
  newly added signal module is logged once at WARNING and then silently force-HOLD for 5 minutes
  with no escalation to critical_errors. A signal that never imports is invisible after the
  first log line. → Consider routing repeated import failures (same name, N retries) to
  `record_critical_error` so a broken plugin surfaces.
