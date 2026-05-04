# Claude Review — signals-core

## P0 (money-losing or data-corrupting)

- `portfolio/signal_decay_alert.py:35-36` — raw `open() + json.load()` on `accuracy_cache.json` violates atomic I/O rule
  ```python
  with open(accuracy_cache_path, encoding="utf-8") as f:
      cache = json.load(f)
  ```
  CLAUDE.md rule #4 forbids this. `accuracy_cache.json` is written by 60s loop atomic write — concurrent rename can produce torn read on Windows. `JSONDecodeError` is caught and silently returns `[]` — real accuracy degradation goes unreported. Decay monitoring is the last line of defense before manual audit. Fix: `load_json(accuracy_cache_path, default={})`. Confidence 95.

- `portfolio/signal_db.py:271, 302-303, 330-331, 370` — SQL accuracy methods omit the neutral-outcome filter (`_MIN_CHANGE_PCT`)
  ```python
  # signal_db.py:271 — signal_accuracy()
  if (vote == "BUY" and change_pct > 0) or (vote == "SELL" and change_pct < 0):
      stats[sig_name]["correct"] += 1
  ```
  The Python path (`accuracy_stats._vote_correct`) skips moves where `|change_pct| < 0.05%` as neutral. All four SQL methods count near-zero moves as wins/losses, **overstating accuracy**. `per_ticker_accuracy()` feeds `apply_confidence_penalties()` Stage 6 which applies a per-ticker penalty when accuracy < 52% — overstated 53% reads as "exempt" when true accuracy < threshold. Affects MSTR (47.8%) and XAG most. Fix: `WHERE ABS(o.change_pct) >= 0.0005`. Confidence 92.

- `portfolio/signal_engine.py:3132-3139` — `btc_proxy` injected into MSTR votes outside `SIGNAL_NAMES`, bypasses accuracy gate
  ```python
  if ticker == "MSTR" and "BTC-USD" in _cross_ticker_consensus:
      btc_cons = _cross_ticker_consensus["BTC-USD"]
      btc_action = btc_cons.get("action", "HOLD")
      if btc_action in ("BUY", "SELL", "HOLD"):
          votes["btc_proxy"] = btc_action
  ```
  `accuracy_data.get("btc_proxy")` returns None — no accuracy validation. Counts toward MIN_VOTERS=3 with no accountability. Comment claims "the vote goes through all normal gates (accuracy, regime, persistence)" — factually incorrect. Permanent gap in signal_log accuracy tracking. Fix: register in SIGNAL_NAMES, OR inject as weighted contribution not synthetic vote, OR gate participation by BTC consensus accuracy. Confidence 90.

## P1 (high-confidence bugs)

- `portfolio/signal_decay_alert.py:27` — relative path default fails under PF-OutcomeCheck CWD
  ```python
  def check_signal_decay(accuracy_cache_path="data/accuracy_cache.json"):
  ```
  Scheduled task may run from `C:\Windows\System32`. `data/accuracy_cache.json` resolves nowhere. `FileNotFoundError` caught, returns `[]`. Same class of bug fixed in `ic_computation.py:22-25` (per the comment there). Fix: default `None`, compute `Path(__file__).resolve().parent.parent / "data" / "accuracy_cache.json"`. Confidence 88.

- `portfolio/signal_engine.py:3475-3484` — utility boost can silently promote sub-47% signals above accuracy gate
  ```python
  if samples >= 30 and u_score > 0:
      boost = min(1.0 + u_score, 1.5)
      if sig_name in accuracy_data:
          boosted_acc = min(accuracy_data[sig_name]["accuracy"] * boost, 0.95)
          accuracy_data[sig_name] = {**accuracy_data[sig_name], "accuracy": boosted_acc}
  ```
  Signal at 44% raw with utility 0.08 → boost 1.08 → 47.5% (above gate). Project guideline: "signals below 47% accuracy (30+ samples) are force-HOLD." Boost creates silent bypass at the boundary. Fix: apply boost to weight multiplier in `_weighted_consensus()`, not to the gating accuracy figure. Confidence 85.

- `portfolio/accuracy_stats.py:1918-1929` — `write_ticker_accuracy_cache()` uses single shared `"time"` key for all horizons
  ```python
  cache[horizon] = data
  cache["time"] = time.time()  # resets TTL for ALL horizons
  ```
  `load_cached_ticker_accuracy()` checks `time.time() - cache.get("time", 0) < TTL` — single staleness clock. Fresh "3h" write resets `"time"`, making "1d" and "3d" appear fresh. Main accuracy cache uses per-horizon `f"time_{horizon}"`. Mirror that pattern. Confidence 83.

- `portfolio/signal_db.py:302-303, 330-331` — same neutral-filter bug in `consensus_accuracy()` and `per_ticker_accuracy()` — separate impact vector beyond gate bypass; suppresses confidence penalty for sub-52% per-ticker (MSTR at 47.8%). Confidence 82.

## P2 (concerns / smells)

- `portfolio/signal_engine.py:3225, 3350` — `acc_horizon = horizon if horizon in ("3h","4h","12h") else "1d"` defined twice with identical expression
  Latent divergence risk: future change to either copy creates silent inconsistency. Extract to local helper.

- `portfolio/signal_history.py` — `get_persistence_scores()` / `get_signal_streaks()` read history without `_history_lock`
  `update_history()` holds lock for read-modify-write. Readers don't. Concurrent rename can cause readers to miss latest write — persistence scores one cycle stale. Persistence filter (2+ consecutive votes) may admit a just-flipped signal. Either lock readers or document the staleness window.

- `portfolio/signal_weights.py` — `SignalWeightManager.batch_update()` writes `signal_weights.json` but `signal_engine.py` never reads it
  Documented as C6 dead code. Risk: future dev adds a reader without knowing MWU updates use post-gate votes (not pre-gate raw) — subtle accuracy attribution error.

## Did NOT find

1. Double-counting / sign errors in `_weighted_consensus()` — verified correct.
2. Naive datetimes — `datetime.now(UTC)` used throughout.
3. Training-on-test-set leakage in `meta_learner.py` — `_PURGE_DAYS = 2`, temporal split correct.
4. Non-atomic writes in hot paths — atomic_write_json/atomic_append_jsonl used.
5. Dict mutation during iteration in dispatch loop.
6. Sub-50% inversion — code correctly gates as HOLD; no inversion.
7. MIN_VOTERS=3 violation — all paths use the constant.
8. Race in `outcome_tracker.py` three-phase lock pattern — sidecar lockfile correct.
9. ML walk-forward leakage — methodology concern (no purge gap) but ML signal globally disabled at 28.2% so production impact nil.
