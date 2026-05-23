# signals-core review — 2026-05-23

Reviewer: adversarial code reviewer.
Scope: the 16 files listed in the prompt.
Method: read all files (long ones in slices), cross-checked against
`.claude/rules/signals.md`, CLAUDE.md, and the project's history of
silent-failure incidents.

I will only call out things I can quote a line for. Where I am
uncertain, I downgrade to P2/P3 rather than overstate.

## P0 (production incident risk)

### P0-1 — `signal_decay_alert.py:27,148` use relative file paths
`check_signal_decay(accuracy_cache_path="data/accuracy_cache.json")` and
`atomic_append_jsonl("data/signal_decay_alerts.jsonl", entry)` both pass
relative paths. This is *the exact* failure pattern that
`ic_computation.py:19-26` documents as a 2026-05-02 P0 fix (relative
`Path("data")` silently routed to a phantom directory when the
scheduled task CWD differed from repo root). The "silent" part is
critical: `load_json(...)` returns `None` for a missing file →
`check_signal_decay` returns `[]` → the operator sees "no decay
detected" instead of "decay detector is broken". The `--check-outcomes`
caller in main.py invokes `run_decay_check()` from a working dir we do
not control (PF-OutcomeCheck Task Scheduler).

Fix: mirror `ic_computation.py:25`:
```py
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ACCURACY_CACHE_FILE = DATA_DIR / "accuracy_cache.json"
DECAY_ALERTS_FILE = DATA_DIR / "signal_decay_alerts.jsonl"
```

This is the same class of bug as the March-April 2026 Layer 2 auth
outage — code that runs and "succeeds" while producing nothing.

### P0-2 — `signal_engine.py:4205` config can lower the accuracy gate below the 47% rule
```py
accuracy_gate = sig_cfg.get("accuracy_gate_threshold", ACCURACY_GATE_THRESHOLD)
```
`.claude/rules/signals.md` explicitly states the floor: 47% with 30+
samples, raised to 50% for 7000+ samples. This config knob has NO floor
check — a config-file typo or a research experiment can set the gate to
0.30 and the system will happily count 30% signals as voters. The
high-sample tier (line 2557-2561) clamps the relaxed gate via `max(...,
0.50)` but **only after** the user-supplied value is used; the standard
tier's effective_gate becomes whatever the YAML says.

Fix: clamp at function boundary:
```py
accuracy_gate = max(ACCURACY_GATE_THRESHOLD,
                    float(sig_cfg.get("accuracy_gate_threshold",
                                      ACCURACY_GATE_THRESHOLD)))
```
Or surface it as a hard error if config sets it below the rule.

### P0-3 — `signal_engine.py:3601-3685` "promoted" signal + shadow status = silently HOLD
The promoted-override path at 3601-3607 lets a signal that's still in
`DISABLED_SIGNALS` vote when `shadow_registry.is_promoted(sig_name)`
returns True (intentional — promotion path documented in commit
history). HOWEVER the very next gate at 3663-3685 reads
`get_status(sig_name)` and force-HOLDs anything whose status is
`"shadow"` (per cycle_modulo throttle).

If a signal's `shadow_registry.json` has both `status="shadow"` AND
`promoted=true` — which the promotion workflow doesn't atomically
prevent — every cycle the dispatch loop:
1. Sets `_promoted_override = True`, skips the DISABLED_SIGNALS branch.
2. Falls through to the throttle, status == "shadow", `_throttle_skip
   = True`, sets `votes[sig_name] = "HOLD"`, continues.

Result: a "promoted" signal silently never votes. There is no log line
explaining why — it just produces HOLD forever. The `_throttled = True`
extra_info flag is the only trace.

Fix: in the throttle block, treat `_promoted_override` as "exit shadow
mode regardless of status":
```py
if _promoted_override:
    _throttle_skip = False
else:
    ... existing logic ...
```
Or reset `status` to `"active"` atomically in the promotion script and
add an assertion `not (status == "shadow" and promoted)`.

### P0-4 — `signal_engine.py:4019,4252` core-gate uses pre-persistence votes
```py
# Line 3998-4000:
core_buy = sum(1 for s in CORE_SIGNAL_NAMES if votes.get(s) == "BUY")
core_sell = sum(1 for s in CORE_SIGNAL_NAMES if votes.get(s) == "SELL")
core_active = core_buy + core_sell

# Line 4252:
if core_active == 0 or post_persistence_voters < min_voters:
    weighted_action = "HOLD"
```
`core_active` is computed from `votes` (the **pre-persistence** dict).
`consensus_votes = _apply_persistence_filter(votes, ticker)` mutes
votes that haven't held for 2 cycles. If the persistence filter mutes
**every** core signal (e.g. all of rsi/macd/ema/bb/fear_greed/sentiment/
volume/ministral/qwen3/claude_fundamental flipped this cycle), the
weighted consensus runs on enhanced-only votes. The check at 4252 still
sees `core_active >= 1` (pre-filter) and the action goes through.

The documented contract is "enhanced signals can strengthen/weaken but
never create consensus alone." This contract is breached whenever
persistence flushes the core slate. The cold-start case is the worst:
on cycle 1 after restart, the cold-start branch at 624-634 returns
votes unfiltered, but on cycle 2 any flipped core signal becomes HOLD
in `consensus_votes` and the core gate doesn't notice.

Fix: compute a `core_active_post_persistence` from `consensus_votes`
and use it at 4252.

### P0-5 — `signal_db.py:33` SQLite connection lifecycle + threading
```py
self._conn = sqlite3.connect(str(self.db_path), timeout=10)
```
`sqlite3.connect()` defaults to `check_same_thread=True`. Today each
caller creates its own `SignalDB()` instance (one per thread), so the
problem is latent. But:

* `outcome_tracker.py:413-416` opens `_db = SignalDB()` once, holds it
  through a loop, calls `_db.update_outcome(...)` for many entries.
  Single thread — OK.
* `accuracy_stats.py:148` opens a fresh DB per call to `load_entries`.
  Fine — closes via `try/finally`.
* `outcome_tracker.py:162` opens-writes-closes per snapshot. Fine.

The `signal_log.db-shm` and `signal_log.db-wal` files in the current
git status confirm WAL is live. WAL handles multi-connection writes
**from different connections** correctly, but a refactor that caches
a module-level `SignalDB()` and shares it across the 8-worker
ThreadPoolExecutor in `main.py` would immediately crash.

Recommend: add `check_same_thread=False` + a per-instance write lock,
or document loudly that `SignalDB` is per-thread, not shared.

This is P0 not because anything is currently broken but because the
documented "always reuse helpers" rule (CLAUDE.md #2) makes this a
foot-gun for the next developer who tries to consolidate.

## P1 (correctness / money-loss)

### P1-1 — `signal_engine.py:2181-2229` `regime_mults` lookup not normalized
```py
regime_mults = REGIME_WEIGHTS.get(regime, {})
```
`regime` is the caller-supplied string. `_normalize_regime` exists
(line 1919) and is used inside `_dynamic_min_voters_for_regime`, but
`_weighted_consensus` doesn't call it. If a caller (live or test)
passes `"trending_up"` (underscore) or `"TRENDING-UP"`, the lookup
silently returns `{}` and ALL regime weight adjustments are skipped.
`detect_regime` (`portfolio.indicators`) is presumably canonical, but
the function accepts the param as-is — there is no contract
enforcement. The replace-semantics design of REGIME_GATED_SIGNALS
makes this even nastier: a typo silently removes gating.

Fix: at the top of `_weighted_consensus`, `regime = _normalize_regime(regime)`.
Apply same to `_get_regime_gated` (already accepts strings).

### P1-2 — `signal_engine.py:1086-1099` sentiment state loaded once per process, never reloaded
```py
def _load_prev_sentiments():
    global _prev_sentiment, _prev_sentiment_loaded
    with _sentiment_lock:
        if _prev_sentiment_loaded:
            return
```
After the first call, the in-memory `_prev_sentiment` is authoritative
for the lifetime of the process. If a second process (crypto_loop,
oil_loop, metals_loop, dashboard prewarmer) writes to
`sentiment_state.json` concurrently, the main loop's view is permanently
stale. The hysteresis threshold (0.55 vs 0.40) then triggers off
incorrect history — a sentiment flip that *another process* already
recorded gets treated as a "first flip" by this process and fights it.

Five tasks routinely run independently per CLAUDE.md (PF-DataLoop,
PF-CryptoLoop, PF-OilLoop, PF-MstrLoop, PF-MetalsLoop). Verify each
is or isn't a writer of sentiment_state — if any is, this is a
correctness bug.

### P1-3 — `signal_engine.py:4344` linear factor uses pre-persistence votes
```py
numeric_votes = {k: _vote_map.get(v, 0.0) for k, v in votes.items()}
lf_score = _lf_model.predict(numeric_votes)
```
This uses `votes` not `consensus_votes`. The trained linear model is
fed signals that the persistence filter would have HOLD-suppressed.
Two failure modes:
* The model was trained on pre-filter votes (likely, since
  `train_signal_weights.py:78` just reads `signals` from the log) and
  is being fed pre-filter inputs at inference — consistent. OK.
* But the *boost/dampen* logic at 4352-4360 compares `lf_action` to
  `action` (the weighted-consensus action) which IS post-persistence.
  A flip-flop signal that the persistence filter rightfully muted can
  swing the linear model into agreeing with consensus when, against a
  steady-state, it wouldn't have. Confidence boost from a coincidence.

Fix: pass `consensus_votes` to `_lf_model.predict`, or feed both and
require agreement to boost.

### P1-4 — `signal_engine.py:2699-2703` `_weighted_consensus` returns HOLD with nonzero confidence on tie
```py
if buy_conf > sell_conf and buy_conf >= 0.5:
    return "BUY", round(buy_conf, 4)
if sell_conf > buy_conf and sell_conf >= 0.5:
    return "SELL", round(sell_conf, 4)
return "HOLD", round(max(buy_conf, sell_conf), 4)
```
The final return ships a HOLD with confidence up to ~0.49. Downstream
`apply_confidence_penalties` (line 2917) does `if cfg.get("enabled") is
False: return action, conf, []` — confidence preserved. Later
`extra_info["_weighted_confidence"] = weighted_conf` ships the HOLD-
with-nonzero-conf to journal and the dashboard. `majority_vote` in
`signal_utils.py:122-126` correctly returns `("HOLD", 0.0)`. The
weighted path's exception is unusual and confusing for downstream
consumers expecting `HOLD ⇒ conf == 0` semantics. Not directly
money-losing but it confuses calibration / postmortem tooling.

### P1-5 — `accuracy_stats.py:937,943` blended `correct` count is fabricated
```py
total = max(at_samples, rc_samples)
result = {
    "accuracy": blended,
    "total": total,
    ...
    "correct": int(round(blended * total)),  # BUG-186
```
`correct` is rebuilt from the blended accuracy × the larger sample
count. This is a fictional integer that doesn't correspond to any
actual count. The dashboard's `/api/accuracy` and the `--accuracy`
report present this number. If a signal had `at_samples=10000,
rc_samples=200`, `total=10000`, and `correct = round(blended * 10000)`.
The reader believes the signal has 10,000 trials when in fact the
blended accuracy is dominated by the 200-sample recent window. This
also poisons any downstream code that does
`correct/total` expecting integer cancellation.

Fix: don't synthesize `correct` for blended rows. Store `at_correct`
and `rc_correct` separately if needed, or drop the field.

### P1-6 — `signal_engine.py:_apply_persistence_filter` cold-start trusts all votes
```py
# Line 624-634:
if ticker not in _persistence_state:
    ...
    _persistence_state[ticker] = {
        sig: {"vote": vote, "cycles": min_cycles if vote != "HOLD" else 0}
        for sig, vote in votes.items()
    }
    return votes  # first cycle — trust all signals
```
First-cycle-after-restart: every vote passes through. CLAUDE.md notes
the rule "Single-check MACD/RSI/volume improvements are noise — need
3+ sustained" yet a process restart resets this counter. The W14-W16
consensus analysis cites persistence as the noise-control mechanism;
on restart we open with the noise.

Compounding: `_PERSISTENCE_MAX_TICKERS = 32` with eviction logic at
626-629 — if more than 32 tickers cycle through (which happens in
test/backtest), eviction drops state and the evicted ticker re-cold-
starts on its next call. The eviction `len(_persistence_state) // 2`
drops the **oldest half** by insertion order, which roughly correlates
with least-recent-use, but `time.monotonic()` is not used so an active
ticker that was just added is at risk on a churn.

Fix: persist `_persistence_state` to disk (atomic JSON) so restarts
preserve the counter. Add a per-entry LRU touch on read.

### P1-7 — `signal_engine.py:3138` `should_skip_gpu(ticker, config=config) if ticker else False`
Empty ticker → `skip_gpu=False` → GPU signals run even on the
diagnostic "ticker=None" path that BUG-178 (the documented silent-hang
class) added explicitly to surface. Combined with the warning at 3102
("tracker/phase updates will be skipped"), an empty-ticker dispatch
runs all GPU signals with no diagnostic instrumentation. If a future
caller misuses generate_signal, the system is silent.

Fix: `if not ticker:` should also `skip_gpu = True` to defend the
diagnostic exit.

### P1-8 — `signal_engine.py:1734-1796` "merged" correlation group accounting
```py
elif g1 != g2:
    assert g1 is not None and g2 is not None
    merged = groups[g1] | groups[g2]
    groups[g1] = merged
    del groups[g2]
    for s in merged:
        signal_to_group[s] = g1
```
The `groups` dict is then iterated to produce named frozensets, but
the now-deleted `g2` slot is silently skipped. The order of iteration
through the upper-triangular signal-pair loop determines which group
gets merged into which — so the same data can produce different group
ids on rerun. Not a correctness bug today (groups are anonymized as
`dynamic_<gid>`), but it's a debugging hazard: two consecutive cycles
can label the same membership with different names, breaking
log-grep-by-cluster-name.

Fix: deterministic naming (e.g. hash of sorted members) or sorted
iteration by member count.

### P1-9 — `ticker_accuracy.py:38,46` no caching of `load_entries()`
`accuracy_by_ticker_signal` calls `load_entries()` on every invocation.
`get_focus_probabilities` iterates `tickers × horizons` and each call
re-scans up to 50,000 entries. The function is called from
`reporting.py` and the Mode B notification path. Without caching this
costs 5 × 3 = 15 full DB walks per probability cycle.
`accuracy_stats.load_entries()` does NOT cache. The wall-time cost
could blow trigger response time.

The signal_log SQLite path is fast but `load_entries()` still walks
the entire table.

Fix: thread one `entries=` through `get_focus_probabilities` →
`accuracy_by_ticker_signal` so a single load is reused across all
ticker × horizon calls.

### P1-10 — `signal_postmortem.py:122-182` correlation analysis is unbounded
```py
def compute_vote_correlation(entries: list[dict] | None = None) -> list[dict]:
    if entries is None:
        try:
            from portfolio.accuracy_stats import load_entries
            entries = load_entries()
```
Loads the full signal log, then iterates all (entry, ticker) pairs and
computes pairwise agreement for all sorted active signal pairs. With
N=50000 entries × 5 tickers × ~17 active signals × ~136 pairs, this is
billions of dict lookups. The `compute_vote_correlation()` call inside
`generate_postmortem()` has no timeout. The caller in
`signal_engine.py` doesn't invoke this, but a misconfigured cron or
manual `--postmortem` call could time out the cycle.

Fix: timeout-bound the call, or downsample entries (e.g. last 7d), or
move to SQL.

### P1-11 — `accuracy_degradation.py:753` daily snapshot is hour-gated and silent on miss
```py
if now.hour < target_hour:
    return False
```
Default `target_hour=6` UTC. If the loop is down between 06:00 UTC and
midnight (process restart, crash window), `state["last_snapshot_date_utc"]`
never advances. The Tuesday silent failure pattern: the loop's first
post-06:00 cycle does write a snapshot, but if no cycle runs at all in
that window (loop down → recovered after 22:00 → only cycles after
midnight before next 06:00) the date string never matches `today_str`
because `now.date()` changed before the next 06:00 check. The
post-2026-04-28 silent-failure journal helps but only triggers when
the writer DOES run.

Acceptable mitigation already in place (lines 786-787 size check). Not
P0, just brittle.

## P2 (reliability / observability)

### P2-1 — `signal_engine.py:1097,1487,1714,2132,2744,2893,2983,3736,3880,3914,3931,4085,4116,4183,4203,4315,4330,4362,4395` broad `except Exception`
Many of these are correctly logged with `logger.warning(... exc_info=True)`.
Some are downgraded to `logger.debug(... exc_info=True)`:
* 3736 — signal health tracking
* 3915 — per-ticker accuracy for regime gating
* 3932 — recent accuracy for regime override
* 4117 — regime-conditional accuracy overlay
* 4184 — utility weighting
* 4204 — best-horizon accuracy
* 4316 — market health penalty
* 4331 — earnings gate
* 4363 — linear factor
* 4396 — per-ticker consensus gate

The earnings gate failing silently is the highest-risk — if the
`earnings_calendar` module is broken on a Friday, stocks are still
traded into Monday earnings reports. `logger.debug` means it doesn't
even appear at INFO level.

Fix: promote at minimum the earnings-gate exception to WARNING, since
its failure mode is "no earnings filter" → silent loss of risk
control.

### P2-2 — `signal_engine.py:3739-3741 SLOW-DISPATCH log` is the only observability for dispatch loop
```py
if _dispatch_dt > 5.0:
    logger.warning("[SLOW-DISPATCH] %s: enhanced signals took %.1fs", ...)
```
5-second threshold. Single signal threshold at 3706 is 1s. With 30+
enhanced signals registered, a "death-by-a-thousand-cuts" cycle of
30 × 0.9s each (= 27s) prints NO warning. The phase log helps but
only if BUG-178 fires.

Fix: lower the dispatch threshold (3.0s?) and also log when
`sum_of_known_signal_times << _dispatch_dt` (i.e. unaccounted time
inside the dispatch loop).

### P2-3 — `signal_engine.py:3680` fail-closed shadow-LLM list is statically duplicated
```py
_KNOWN_SHADOW_LLMS = frozenset({
    "forecast", "claude_fundamental", "finance_llama",
    "cryptotrader_lm", "meta_trader",
})
```
This is a fallback when `shadow_registry.get_status` errors. The
comment at 698 says "keep in sync with shadow_registry.json". The
"keep in sync" rule is fragile — a new shadow LLM added to the
registry without updating this set runs every cycle on a fault and
blows the 60s budget. Project history shows this exact rule keeps
getting violated.

Fix: at import, parse `data/shadow_registry.json` once and build the
set dynamically. Fall back to the hardcoded frozenset only if the
file itself can't be read.

### P2-4 — `signal_engine.py:1502-1508` `_get_horizon_weights` returns empty dict via cast
```py
weights = _cached(cache_key, _DYNAMIC_HORIZON_WEIGHT_TTL,
                  lambda: _compute_dynamic_horizon_weights(horizon))
return cast(dict[str, float], weights) if weights else {}
```
When dynamic weights are unavailable, the static fallback inside
`_compute_dynamic_horizon_weights` (which returns `HORIZON_SIGNAL_WEIGHTS.get(horizon, {})`)
goes through `_cached` and is cached for an hour. If the accuracy
cache is fixed in that hour, dynamic recompute is suppressed. Acceptable
trade-off but worth noting — a "loaded broken accuracy cache once →
pinned to static for 60min" pattern.

### P2-5 — `signal_engine.py:2329-2339` macro-window pre-pass mutates votes copy
```py
votes = {k: ("HOLD" if k in MACRO_WINDOW_FORCE_HOLD_SIGNALS else v)
         for k, v in votes.items()}
```
The caller-passed `votes` dict is replaced with a local copy. Fine.
But at the outer scope (line 3886) `raw_votes = dict(votes)` happens
BEFORE this re-rewrite, so `raw_votes` reflects the pre-macro-mutation
state. Yet the macro mutation in `generate_signal:3982-3986` happens
inline on `votes` (not a copy), so `raw_votes` and `votes` diverge
after that point. The accuracy backfill against `raw_votes` includes
the macro-suppressed signal's true vote — which is correct, but the
operator reading the journal will see `raw_votes[claude_fundamental] =
"BUY"` and `votes[claude_fundamental] = "HOLD"` and have to guess why.

Fix: add `extra_info["_macro_suppressed"] = list(MACRO_WINDOW_FORCE_HOLD_SIGNALS)
if macro_active_effective else []`.

### P2-6 — `accuracy_stats.py:147-156 SignalDB closure path
```py
try:
    from portfolio.signal_db import SignalDB
    db = SignalDB()
    try:
        count = db.snapshot_count()
        if count > 0:
            entries = db.load_entries()
            return entries
    finally:
        db.close()
except Exception as e:
    logger.debug("SQLite signal_db unavailable, falling back to JSONL: %s", e)
```
The outer `except` swallows DB import errors AND any exception
inside the inner try at `logger.debug` level. A corrupt DB (the
`.db-wal` files are live, so corruption is possible) silently falls
back to JSONL. This is by design but the LOG LEVEL hides the failure
from operators. Recurring corruption would be invisible.

Fix: count consecutive failures and escalate to WARNING after N
fallbacks.

### P2-7 — `signal_engine.py:4089-4090` H3 fail-closed gate value uses arbitrary 999
```py
accuracy_data = {sig: {"accuracy": 0.0, "total": 999} for sig in SIGNAL_NAMES}
```
"999" is below the high-sample threshold (7000) so the 50% gate
doesn't trigger; the standard 47% gate applies. accuracy=0.0
guarantees gating. OK. But the choice of 999 is fragile — if
`_ACCURACY_GATE_HIGH_SAMPLE_MIN` is ever lowered, this code silently
flips from the standard gate to the high-sample gate. Should be
`ACCURACY_GATE_MIN_SAMPLES` for clarity (and add a comment).

### P2-8 — `signal_history.py:64-98` `update_history` does full RMW of the JSONL file
```py
with _history_lock:
    entries = _load_history()           # load EVERY history entry
    ...
    entries.append(new_entry)
    by_ticker = defaultdict(list)
    for e in entries:
        by_ticker[e.get("ticker", "unknown")].append(e)
    trimmed = []
    for _t, t_entries in by_ticker.items():
        trimmed.extend(t_entries[-MAX_ENTRIES_PER_TICKER:])
    trimmed.sort(key=lambda e: e.get("ts", ""))
    _save_history(trimmed)
```
Every signal cycle for every ticker reads the entire history file,
rewrites it. With 5 tickers × 60s cycles × 7TF and a 50-entry-per-
ticker trim, the file stays small. But the lock is global — main loop
parallelism is serialized through `_history_lock`. With 5 ticker
threads, contention will produce serialized writes. Not P0 (small
file) but the entire purpose of the file (history) could be served
via SQLite append.

### P2-9 — `accuracy_stats.py:1232-1244 maybe_prewarm_dashboard_accuracy` lock fall-back path
```py
fh = acquire_lock_file(...) if acquire_lock_file else "noop"
if fh is None: return False
...
finally:
    if release_lock_file and fh != "noop":
        release_lock_file(fh)
```
The `"noop"` sentinel path means: if `process_lock` import fails, the
process bypasses cross-process exclusion entirely. Two processes both
fan out the 12 cache reads, potentially writing accuracy_cache.json
concurrently. The atomic_write_json prevents torn writes but the
double work is the exact dogpile this code exists to prevent.

Fix: if process_lock is unavailable, log WARNING and refuse to prewarm
(safer than dogpile).

## P3 (style / nit)

### P3-1 — `signal_weights.py:121` file truncates mid-comment
The `_load` method ends:
```py
def _load(self) -> None:
    """Load weights from disk.  No-ops silently if the file is missing."""
    data = load_json(self._path, default=None)
    if data is None:
        return
    if isinstance(data, dict):
        self._weights = {
            k: float(v)
            for k, v in data.get("weights", {}).items()
        }
        # Honour stored eta only if caller did not override it
        # (caller passes None → _DEFAULT_ETA, so we preserve stored value)
```
The file is exactly 120 lines and ends on the trailing comment with
no `pass`, no `return`, no actual eta-handling code. The comment
promises "honour stored eta" logic that is not present — eta is read
from disk implicitly via `data.get("eta")` nowhere. The `__init__`
sets `self._eta = eta if eta is not None else _DEFAULT_ETA` and
ignores whatever the JSON contains.

This is technically valid Python (the if-block produces side effects,
then implicit None return). But it documents behavior that doesn't
exist.

Less importantly: `outcome_tracker.py:497-499` says MWU was removed as
dead code (C6). If true, the entire `signal_weights.py` file might be
dead — verify with grep.

```
$ grep -r "SignalWeightManager\|signal_weights" Q:/finance-analyzer/portfolio
portfolio\signal_weights.py:25:class SignalWeightManager:
```
Confirmed: nothing imports `SignalWeightManager`. The file is dead
code. Delete it or fold into a meta-readme.

### P3-2 — `signal_decay_alert.py:73,80-82` `recent_acc/recent_total` use `.get("accuracy", 0)` not `0.0`
Style nit. The integer 0 will compare equal to 0.0 fine but is
inconsistent with `_safe_accuracy` semantics elsewhere.

### P3-3 — `ic_computation.py:67-70` `_spearman_rank_correlation` returns 0.0 on zero variance
A signal that always votes BUY (returns all positive) produces
`den_x == 0` (no rank variance) and reports IC=0.0. The 30-sample gate
prevents this from poisoning weights, but the IC report at line 286
will list this as "phantom performer" (IC<0.01 ≥ 500 samples → 0.85x
penalty). Mathematically correct — a constant signal does have zero
IC — but the message would be clearer if it logged a warning when
denominator is zero.

### P3-4 — `cusum_accuracy_monitor.py:122-142` 100-alert circular buffer in JSON file
```py
state["alerts"] = alerts_list[-100:]
```
The state file grows indefinitely until 100 alerts then truncates.
For a system with 30+ signals × multiple horizons, alerts can churn
quickly. Keeping the last 100 only is fine, but combined with the
`_save_state` after each update_cusum call, the WHOLE state JSON is
rewritten on each outcome. With 30 signals × n_outcomes/day, this is a
lot of `atomic_write_json` calls. Use append-only JSONL for alerts.

### P3-5 — `accuracy_degradation.py:1647` `_find_snapshot_near` no longer enforces the `BASELINE_MAX_DELTA_HOURS` window via the wrapper
`_find_snapshot_near` is called by `check_accuracy_changes` (line
1643) with the default `max_delta_hours=36`. This is the legacy 7-day
checker. The newer code at `_find_baseline_snapshot` filters by
`window_days==14` first. The two coexisting paths can disagree on
"which snapshot is the baseline" — if `check_accuracy_changes` is
called for the dashboard, it can find a snapshot the newer
`check_degradation` deliberately ignores.

## Coverage notes

* Files read fully: signal_utils.py, signal_weights.py,
  signal_state_since.py, signal_history.py, signal_decay_alert.py,
  signal_postmortem.py, signal_weight_optimizer.py,
  train_signal_weights.py, signal_db.py, ticker_accuracy.py,
  ic_computation.py, cusum_accuracy_monitor.py, signal_registry.py.
* Files read in slices (large): signal_engine.py (4416 lines — read
  in 5 chunks covering: top constants, persistence/dispatch helpers,
  gate-cascade math, `_weighted_consensus`, `apply_confidence_penalties`,
  `generate_signal` outer loop + post-dispatch consensus + linear-factor +
  per-ticker gate); accuracy_stats.py (2077 lines — read in 4 chunks
  covering: load_entries/_vote_correct/signal_accuracy/ewma/cost_adjusted/
  consensus/per_ticker, blend_accuracy_data + cache wrappers,
  signal_accuracy_by_regime + caches, best_horizon + ticker_signal +
  calibration); accuracy_degradation.py (1062 lines — read in 4 chunks
  covering: header constants + snapshot writer + diff engine,
  alert / cooldown / daily-summary / Telegram path, summary diffs).
* Files skimmed only: none.
* Files skipped: none.

## What I did not find (worth saying out loud)

* No bare `except:` (no-argument). All exception handlers are
  `except Exception` or specific types.
* No `json.loads(open(...).read())` — all reads route through
  `file_utils.load_json` / `load_jsonl` / `load_jsonl_tail`. The atomic
  I/O rule is honored across signals-core.
* No `try/except: pass` that silently throws away exceptions. The
  closest is `signal_decay_alert.py:39-41` which logs a warning then
  returns `[]` — degraded but not silent.
* No off-by-one in vote counting that I can substantiate. The math
  at `_weighted_consensus` (2697-2703) and `majority_vote`
  (signal_utils.py:118-126) is correct.
* No NaN propagation in the hot path. `_safe_accuracy` /
  `_safe_sample_count` guard the consensus loop; np.isfinite checks
  in `_validate_signal_result` guard signal outputs.
* No accuracy-gate bypass via the weight optimizer — the optimizer
  writes `data/models/walkforward_results.json` which signal_engine
  does NOT read (confirmed by grep: `walkforward_results` only appears
  in signal_weight_optimizer.py). Like signal_weights.py, the
  optimizer pipeline produces output that no consumer reads — likely
  dead code worth a follow-up audit.
* No disabled-signal "leak" past the dispatch gate at line 3607.
  The dispatch loop respects DISABLED_SIGNALS except via the
  documented `_DISABLED_SIGNAL_OVERRIDES` set and the shadow-registry
  promotion path. Both are intentional.
