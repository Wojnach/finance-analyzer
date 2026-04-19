# Independent Adversarial Review — 2026-04-19

Author: Opus (independent review, written in parallel with 8 subsystem agents)

## Methodology

Deep read of all critical code paths across all 8 subsystems, with particular
focus on the signal engine consensus pipeline (the most complex and highest-impact
code path), portfolio state management, and cross-subsystem interaction bugs.

---

## P1 — Critical Findings

### [P1-I-01] ic_computation.py uses relative Path("data") — IC weighting silently disabled

**File**: `portfolio/ic_computation.py:19`
**Description**: `DATA_DIR = Path("data")` is a relative path. Every other module
uses `Path(__file__).resolve().parent.parent / "data"`. When the process CWD is
not the project root (pytest-xdist workers, subprocess, notebook), IC cache lookup
silently fails, `_get_ic_data()` returns None, and all IC-based weight multipliers
become 1.0. The IC feature was added 2026-04-18 and may have never been functional
in certain execution contexts.
**Impact**: IC-based signal weighting silently absent — signals with zero IC
(phantom performers) don't get the 0.85x penalty, and signals with high IC don't
get boosted. Trading decisions are suboptimal without operator awareness.
**Fix**: Change to `Path(__file__).resolve().parent.parent / "data"`.

### [P1-I-02] Utility boost can amplify per-ticker accuracy above gate threshold

**File**: `portfolio/signal_engine.py:2822-2836`
**Description**: The utility boost (`boost = min(1.0 + u_score, 1.5)`) is applied
to `accuracy_data[sig_name]["accuracy"]` AFTER per-ticker accuracy override. A
signal with 48% per-ticker accuracy and positive utility (avg_return=0.5) gets
boosted to `48% * 1.5 = 72%`. This signal then passes the accuracy gate (47%)
inside `_weighted_consensus` with a weight of 0.72, even though its actual
directional accuracy on this ticker is below coin-flip.
**Impact**: Signals that are demonstrably worse than random on a specific ticker
can vote with high weight if their utility score (average return magnitude) is
positive. The utility score measures return magnitude, not directional correctness
— a signal that's wrong 52% of the time but catches big moves when it's right
will have high utility but still generates net-negative trading decisions.
**Fix**: Apply utility boost to the weight WITHIN `_weighted_consensus`, not to
the accuracy_data used for gating. The gate should use raw per-ticker accuracy;
the weight can incorporate utility as a multiplicative factor.

### [P1-I-03] ticker_accuracy neutral-outcome filter missing — per-ticker accuracy inflated

**File**: `portfolio/ticker_accuracy.py:60-61`
**Description**: `accuracy_by_ticker_signal()` counts a vote as correct when
`(vote == "BUY" and change_pct > 0)` — any positive change, even 0.001%.
`accuracy_stats.py:_vote_correct()` applies a 0.05% neutral-outcome filter.
The per-ticker accuracy is systematically higher than the global accuracy for
the same signal because neutral micro-moves inflate the correct count.
**Impact**: Per-ticker accuracy overrides in `generate_signal:2782-2799` use
inflated numbers, potentially preventing the accuracy gate from catching bad
signals on specific tickers. Mode B probability notifications show inflated
confidence to the user.
**Fix**: Add `if abs(change_pct) < 0.05: continue` before the correctness check.

### [P1-I-04] Directional accuracy not blended — stale all-time overrides recent degradation

**File**: `portfolio/accuracy_stats.py:818-828`
**Description**: `blend_accuracy_data` blends overall accuracy using adaptive
recency weighting, but for directional keys (`buy_accuracy`, `sell_accuracy`),
it takes the raw value from whichever source has more samples (all-time usually).
If BUY accuracy degraded from 55% to 30% in the last 7 days, but all-time has
more samples, the blended data uses 55% for the directional gate. The directional
gate (40%) doesn't fire, and the signal keeps voting BUY at the stale 55% weight.
**Impact**: Recently degraded directional accuracy is invisible to the directional
gate, allowing harmful BUY/SELL votes through.
**Fix**: Apply the same blending math to directional keys as overall accuracy.

---

## P2 — High Findings

### [P2-I-01] Dashboard CORS wildcard exposes trading data to any origin

**File**: `dashboard/app.py:44`
**Description**: `Access-Control-Allow-Origin: *` allows any website to read
portfolio state, trade history, signal data, and configuration. Combined with
optional (not mandatory) token auth, a malicious website visited by the user
could exfiltrate all trading data and portfolio positions.
**Impact**: Information leak of financial data and trading strategy.
**Fix**: Restrict CORS to localhost/LAN origins, or require auth token for all
data endpoints.

### [P2-I-02] _streaming_max in risk_management.py uses raw open() without file locking

**File**: `portfolio/risk_management.py:37-49`
**Description**: `_streaming_max` opens the portfolio value history JSONL directly
to stream the max value. Concurrent writes from the main loop (via
`atomic_append_jsonl`) can produce partially-visible lines. The
`try/except json.JSONDecodeError: continue` handles corrupt partial lines, but
on Windows, a concurrent exclusive lock by `atomic_append_jsonl` can cause
`PermissionError` on the read, which propagates as an `OSError` and is caught
by the `except OSError` at line 50, returning `floor`. During a drawdown, if
the floor is INITIAL_VALUE_DEFAULT (500K), the peak appears to be 500K, and
drawdown calculation shows 0% — the circuit breaker becomes blind.
**Impact**: Transient file lock contention can make drawdown appear zero,
preventing the circuit breaker from firing during actual drawdowns.
**Fix**: Read the history file using `load_jsonl_tail` which already handles
locking, or catch OSError and return a sentinel that triggers the breaker.

### [P2-I-03] outcome_tracker backfill TOCTOU — concurrent append + replace can lose entries

**File**: `portfolio/outcome_tracker.py:430-446`
**Description**: `backfill_outcomes` reads signal_log.jsonl, modifies entries,
and writes the full file back via `os.replace()`. If `log_signal_snapshot()`
appends new entries between the read and the replace, those entries are silently
lost. PF-OutcomeCheck runs daily at 18:00 when the main loop is active.
**Impact**: Signal log entries from the last cycle before backfill are deleted.
Accuracy statistics are permanently degraded.
**Fix**: Use a process-level lock file, or redesign backfill to use a separate
"outcomes" file rather than mutating signal_log.jsonl in place.

### [P2-I-04] main.py re-exports 60+ private APIs from other modules

**File**: `portfolio/main.py:119-235`
**Description**: main.py re-exports `_log_trigger`, `_atomic_write_json`,
`_cached`, `_prev_sentiment`, `_cross_asset_signals`, `_RateLimiter`, etc.
These are internal APIs that tests and trigger.py depend on via main.py imports.
Any rename or removal of these private functions in their home module will
silently break main.py imports, which will crash the production loop on startup.
**Impact**: Tight coupling through private API re-exports creates maintenance
hazard. A routine refactor of any source module can crash the production loop.
**Fix**: Tests should import directly from source modules. Remove re-exports
after updating all importers.

### [P2-I-05] apply_confidence_penalties Stage 7 calibration can compress confidence below action threshold

**File**: `portfolio/signal_engine.py:2044-2053`
**Description**: Stage 7 compresses confidence above 0.55 with factor 0.3:
`conf = 0.55 + (conf - 0.55) * 0.3`. A signal at 60% becomes 56.5%, at 70%
becomes 59.5%. Combined with earlier stages (regime 0.75x, unanimity 0.6x),
a signal that starts at 80% weighted confidence could end up at:
`80% * 0.75 (ranging) * 0.6 (unanimity) = 36%`. The calibration compression
doesn't fire because 36% < 55%. But without unanimity: `80% * 0.75 = 60%`,
compression gives `55% + (60%-55%)*0.3 = 56.5%`. The minimum confidence for
trading recommendations is documented as 60% — so compressed signals at 56.5%
may never trigger Layer 2 trade decisions, even with strong consensus.
**Impact**: Good signals in ranging markets are systematically compressed below
the trading threshold. The system becomes overly conservative in exactly the
conditions where contrarian signals have proven value.
**Fix**: Either lower the compression threshold, or apply calibration BEFORE
regime penalties so the interaction is better controlled.

### [P2-I-06] Concurrent accuracy cache writes — `_atomic_write_json` is atomic but load-modify-write is not

**File**: `portfolio/accuracy_stats.py:165-180`
**Description**: `write_accuracy_cache` calls `_atomic_write_json` which is
atomic at the file level, but `get_or_compute_accuracy` does
load→compute→write in sequence. `_accuracy_compute_lock` protects this for
accuracy computation, but `write_regime_accuracy_cache` (line 293) and
`write_accuracy_cache` (line 165) don't share this lock. Two different
cache writes can interleave, with one overwriting the other's data.
**Impact**: Stale accuracy cache data used for gating decisions.
**Fix**: Use the same lock for all accuracy cache writes, or accept that
stale-for-one-cycle is tolerable (document this).

---

## P3 — Medium Findings

### [P3-I-01] generate_signal is 969 lines — unmaintainable

**File**: `portfolio/signal_engine.py:2061-3030`
**Description**: A single function that handles RSI, MACD, EMA, BB, Fear&Greed,
sentiment, ML, funding, on-chain, volume, LLM signals, enhanced signals, regime
gating, accuracy loading, per-ticker override, utility boost, best-horizon,
weighted consensus, confidence penalties, market health, earnings gate, linear
factor, per-ticker consensus gate, and horizon confidence cap.
**Impact**: High cognitive load, difficult to test individual stages, easy to
introduce cross-stage bugs (as evidenced by the 50+ BUG-XXX comments).
**Fix**: Extract numbered stages into separate functions that each take and
return a well-typed context object.

### [P3-I-02] _TICKER_DISABLED_BY_HORIZON hardcoded — should be data-driven

**File**: `portfolio/signal_engine.py:286-330`
**Description**: Per-ticker per-horizon blacklists are hardcoded in source code,
requiring a code deployment to update. The system has accuracy data that could
drive these blacklists dynamically.
**Impact**: Blacklists become stale as signal performance changes. Requires
manual audit and code changes to update.
**Fix**: Compute blacklists from accuracy_cache.json with configurable
thresholds, using the hardcoded list as a fallback.

### [P3-I-03] Dead code: votes["ml"] = "HOLD" unconditionally

**File**: `portfolio/signal_engine.py:2269`
**Description**: The ML signal is permanently disabled. It still occupies space
in the votes dict, accuracy tracking, and reporting. It's in DISABLED_SIGNALS
which means it's already handled by the dispatch loop skip at line 2513.
**Impact**: Noise in code and data. The unconditional HOLD assignment at line
2269 is redundant with the DISABLED_SIGNALS check.
**Fix**: Remove the explicit votes["ml"] = "HOLD" line and the surrounding
comments.

### [P3-I-04] ministral applicable count mismatch for non-crypto tickers

**File**: `portfolio/signal_engine.py:829-831 vs 2370`
**Description**: `_compute_applicable_count` excludes ministral for non-crypto
tickers, but `generate_signal` actually runs ministral for all tickers. The
`_total_applicable` field in extra_info is wrong for metals and stocks.
**Impact**: Incorrect signal utilization reporting. Doesn't affect consensus.
**Fix**: Align the two code paths — either exclude ministral from non-crypto
in generate_signal, or include it in the count.

### [P3-I-05] _compute_agreement_rate uses zip without length check

**File**: `portfolio/signal_engine.py:901`
**Description**: `zip(votes_a, votes_b)` silently truncates to the shorter array
if lengths differ. If the signal log has entries with inconsistent vote counts
(e.g., a signal was added mid-period), the agreement rate is computed on a
truncated window without warning.
**Impact**: Correlation groups may be computed on fewer pairs than expected,
potentially missing or creating spurious groups.
**Fix**: Add `assert len(votes_a) == len(votes_b)` or use `itertools.zip_longest`.

### [P3-I-06] ic_buy/ic_sell in ic_computation.py are average returns, not ICs

**File**: `portfolio/ic_computation.py:119-122`
**Description**: Named as `ic_buy` and `ic_sell` but computed as mean returns.
Misleading for any downstream consumer expecting Spearman rank correlations.
**Impact**: Diagnostic confusion. Not used in consensus path currently.
**Fix**: Rename to `avg_buy_return` / `avg_sell_return`.

---

## Cross-Subsystem Interaction Findings

### [P2-X-01] IC computation relative path + metals_loop os.chdir creates fragile dependency

The metals loop does `os.chdir(BASE_DIR)` at startup, which happens to make
ic_computation.py's `Path("data")` resolve correctly — but only for the metals
process. The main loop in portfolio/main.py does NOT chdir. If IC computation
is ever called from the main loop (which it is, via signal_engine.py →
_get_ic_data → ic_computation), the relative path fails. This means IC
weighting works in metals_loop context but NOT in the main loop context.

### [P2-X-02] accuracy_data mutation chain crosses abstraction boundaries

In generate_signal, accuracy_data is built from blend (accuracy_stats.py),
overlaid with regime accuracy (accuracy_stats.py), overridden with per-ticker
data (ticker_accuracy.py), then boosted by utility (accuracy_stats.py). Each
step can modify entries that the previous step set. The utility boost (step 4)
can undo the per-ticker override (step 3) by multiplying a 48% accuracy above
the gate threshold. There's no invariant enforcement between steps — the gate
threshold is checked only inside _weighted_consensus, after all modifications.

### [P2-X-03] Per-ticker consensus gate uses different cache key than writer

In generate_signal line 2996: `load_cached_accuracy(f"per_ticker_consensus_{acc_horizon}")`.
The writer in accuracy_stats uses `write_accuracy_cache(f"per_ticker_consensus_{horizon}", data)`.
If acc_horizon and horizon ever diverge (they shouldn't, since acc_horizon is
derived from horizon at line 2700), the reader would miss the cache. Currently
these are always equal, but the variable aliasing is fragile.

---

## Summary by Severity

| Severity | Count | Key Themes |
|----------|-------|------------|
| P1 | 4 | IC relative path, utility gate bypass, ticker accuracy inflation, directional blend inconsistency |
| P2 | 6+3 | CORS, streaming max, backfill TOCTOU, re-exports, calibration compression, cache writes, cross-subsystem |
| P3 | 6 | Function complexity, hardcoded blacklists, dead code, agreement rate, naming |
