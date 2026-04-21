# Track B: Independent Adversarial Review

**Reviewer:** Primary analyst (manual deep-read)
**Focus:** Cross-cutting concerns, state consistency, financial safety invariants, security
**Date:** 2026-04-21

## Methodology

Deep-read of 15 critical files, focusing on:
- Cross-subsystem state consistency (things no single-subsystem agent would catch)
- Financial safety invariants (money at risk)
- Concurrency between main loop, metals loop, and Layer 2
- Security boundaries (dashboard, telegram, Avanza)
- Silent failure patterns

---

## P0 (Critical) Findings

### IND-001: Utility boost inflates accuracy data BEFORE the accuracy gate
**File:** `portfolio/signal_engine.py:2974-2985` + `signal_engine.py:1605-1620`
**Severity:** P0

The utility boost (lines 2974-2985) modifies `accuracy_data[sig_name]["accuracy"]` by
multiplying up to 1.5x BEFORE the data is passed to `_weighted_consensus`. Inside
`_weighted_consensus`, the accuracy gate at line ~1605 reads `accuracy_data.get(sig)["accuracy"]`
to decide whether to gate the signal.

A signal with true 45% accuracy (below the 47% gate) but positive average return gets
boosted to `45% * 1.5 = 67.5%`, sailing past the gate. The gate's purpose — filtering
noise signals — is undermined.

**Concrete scenario:** `forecast` has 40.3% accuracy on XAG-USD at 3h but occasionally
catches big moves (positive avg_return). The utility boost inflates it past 47%, it
enters the consensus, and its 40% directional accuracy actively degrades the ensemble.

**Impact:** The accuracy gate, the system's primary quality filter, has a systematic bypass
via utility boost. Any signal with low accuracy but occasional large correct calls (which
is the profile of a noise signal that sometimes gets lucky) can pass the gate.

**Fix:** Apply utility boost to the WEIGHT multiplier inside `_weighted_consensus`, not to
`accuracy_data["accuracy"]`. The gate should check raw accuracy; the consensus weight
should reflect utility-adjusted quality:
```python
# In _weighted_consensus, after accuracy gate:
utility_adjusted_weight = base_weight * utility_boost
```

### IND-002: Drawdown circuit breaker exception is swallowed — check_drawdown crash = no protection
**File:** `portfolio/agent_invocation.py:396-397`
**Severity:** P0

```python
except Exception as e:
    logger.warning("drawdown check failed (proceeding): %s", e)
```

If `check_drawdown` raises ANY exception (import error, missing file, NaN in calculation,
KeyError in portfolio state), the entire circuit breaker is bypassed and trading continues.
This is a fail-OPEN design on the single most important safety gate in the system.

The existing risk_management.py code has multiple paths that can throw:
- `_streaming_max` opens the history file without catching `PermissionError`
- `_compute_portfolio_value` does arithmetic that can NaN if fx_rate is corrupt
- `load_json` returns None on corruption, which then crashes `.get("holdings")`

**Impact:** Any single exception in the drawdown calculation path means the 50% hard-block
never fires. During exactly the conditions when drawdown is likely (volatile markets,
system stress, data issues), exceptions become more probable.

**Fix:** Fail CLOSED:
```python
except Exception as e:
    logger.critical("drawdown check FAILED — blocking invocation for safety: %s", e)
    _log_trigger(reasons, "blocked_drawdown_error", tier=tier)
    return False
```

### IND-003: Dashboard CORS allows any origin + API token in URL query string
**File:** `dashboard/app.py:43-49` + `dashboard/app.py:675-676`
**Severity:** P0

```python
response.headers["Access-Control-Allow-Origin"] = "*"
```
Combined with:
```python
token = request.args.get("token")
```

The dashboard exposes portfolio state, holdings, trade history, and signal data. With
`Access-Control-Allow-Origin: *`, any JavaScript on any website can make authenticated
requests to the dashboard if it knows the token. And the token can appear in URL query
strings, which are logged in:
- Browser history
- HTTP access logs
- Proxy/CDN logs
- Referer headers when navigating away

**Impact:** Portfolio holdings, trade signals, and strategy data are accessible to any
website visited by the user if the token leaks via URL. Since the dashboard runs on port
5055 on localhost/LAN, the blast radius is limited to same-network attackers, but
XSS on any site the user visits could exfiltrate all portfolio data.

**Fix:**
1. Restrict CORS to specific origins (or disable for localhost-only operation)
2. Remove query-string token support; require Authorization header only
3. Add `SameSite=Strict` cookie option if cookies are used

---

## P1 (High) Findings

### IND-004: Persistence filter creates stale phantom signals after ticker removal
**File:** `portfolio/signal_engine.py:238-287`
**Severity:** P1

`_persistence_state` is an unbounded module-level dict keyed by ticker. When a ticker
is removed from the system (e.g., AMD, GOOGL removed Mar 15), its persistence state
stays in memory forever. If the ticker is later re-added, the stale persistence state
from weeks ago becomes the baseline — signals that flip to BUY/SELL will have stale
cycle counts, potentially auto-passing the persistence filter on first cycle.

More critically: the dict has no maximum size bound. Tests that create ephemeral tickers
will leak entries. The `_PHASE_LOG_MAX_TICKERS` eviction pattern exists for phase logs
but NOT for persistence state.

**Fix:** Add LRU eviction (matching phase_log pattern) and clear state when tickers change:
```python
if len(_persistence_state) > 64 and ticker not in _persistence_state:
    # evict oldest half
```

### IND-005: Regime gating exemption check uses accuracy data from a different source than the gate
**File:** `portfolio/signal_engine.py:2762-2778` vs `signal_engine.py:2880-2886`
**Severity:** P1

At line 2766, `_ticker_acc_data` is loaded from `accuracy_by_ticker_signal_cached(acc_horizon)`.
At line 2880, `accuracy_data` is loaded from `blend_accuracy_data(alltime, recent, ...)`.
Then at line 2930-2947, `_ticker_acc_data` overrides entries in `accuracy_data`.

The regime exemption check at line 2776 uses `_ticker_acc_data` directly (per-ticker accuracy),
but the gate inside `_weighted_consensus` uses `accuracy_data` (which may be the BLENDED
global accuracy for that signal, not per-ticker). This means:

- A signal can be EXEMPT from regime gating because its per-ticker accuracy is 60%+
- But then GATED by the accuracy gate because its blended global accuracy is 45%

Or vice versa — exempt from regime gating (per-ticker 60%) but the accuracy data used
in consensus is the blended value, not per-ticker, because the override at line 2930
hasn't run yet (it runs AFTER the regime exemption check at line 2772).

The ordering is: regime exemption (uses _ticker_acc_data) → accuracy_data load (global)
→ per-ticker override into accuracy_data. So the regime exemption and the consensus use
different accuracy sources for the same signal.

**Impact:** A signal can be exempted from regime gating based on per-ticker performance
but then get accuracy-gated based on global performance, creating an inconsistent state
where the signal is un-gated for regime purposes but still blocked by accuracy.

**Fix:** Move the per-ticker accuracy override (lines 2930-2947) BEFORE the regime
exemption check, so both use the same source.

### IND-006: `_agent_start` module-level global is shared across all invocations — no process isolation
**File:** `portfolio/agent_invocation.py:36-48`
**Severity:** P1

All agent state (`_agent_proc`, `_agent_start`, `_agent_timeout`, `_agent_tier`) is in
module-level globals. There is no locking around the global state transitions in
`invoke_agent` (line 297). If `main.py`'s 8-worker ThreadPoolExecutor triggers two
tickers simultaneously and both call trigger detection → `invoke_agent`, the second
call will check `_agent_proc.poll()` on the process spawned by the first, and the
globals will be in an inconsistent state.

The code does check `_agent_proc.poll() is None` at line 322, which prevents spawning
a second agent while one is running. But the `_agent_start` and `_agent_timeout` globals
are NOT protected by a lock, so two threads entering `invoke_agent` simultaneously can
race:
- Thread A sets `_agent_proc = subprocess.Popen(...)` at line ~470
- Thread B enters, sees `_agent_proc.poll() is None` → returns False (correct)
- But if Thread A hasn't yet set `_agent_start` (line ~474), Thread B's elapsed check
  uses the STALE `_agent_start` from the previous invocation

**Impact:** Rare but can cause incorrect timeout calculations. The `_safe_elapsed_s`
fallback mitigates most damage, but the root cause is unguarded global state.

**Fix:** Add a `threading.Lock` around the entire invoke_agent function, or use a
dataclass to bundle agent state atomically.

### IND-007: `_compute_dynamic_correlation_groups` single-link clustering can produce transitive mega-clusters
**File:** `portfolio/signal_engine.py:976-1064`
**Severity:** P1

The clustering algorithm is single-link: if A agrees with B at 86% and B agrees with C
at 86%, A and C are merged into the same cluster even if their direct agreement is only
50%. With 30+ active signals, transitive chains can merge independent information sources
into a single mega-cluster, suppressing all but the leader to 0.12x weight.

The static groups carefully separate trend_direction (8 members at 0.12x) from
momentum_cluster (4 members at 0.15x). But the dynamic computation has no such
separation — it can merge them if any bridge signal (e.g., `momentum_factors`) agrees
with both `trend` and `mean_reversion` at >85%.

**Impact:** A transitive chain can create a 15+ member cluster where 14 signals get
0.12x weight. Total effective weight: 1 + 14*0.12 = 2.68. Compare to 15 independent
signals at 1.0x each = 15.0. The consensus becomes dominated by whichever single signal
has the highest accuracy in the mega-cluster.

**Fix:** Use complete-link clustering (require ALL pairwise agreements > threshold) or
cap cluster size. Or add a cluster-size limit:
```python
if len(merged) > 10:
    # Don't merge — cluster too large, likely transitive chain
    continue
```

### IND-008: `check_triggers` stores `set()` in state dict passed to `atomic_write_json` — set is not JSON-serializable
**File:** `portfolio/trigger.py:148`
**Severity:** P1

```python
state["_current_tickers"] = set(signals.keys())
```
Then `_save_state(state)` at line 111 pops this key before saving. But if an exception
occurs between line 148 and the `_save_state` call, or if any code path saves the state
without calling `_save_state` (which does the pop), `json.dump` will raise `TypeError:
Object of type set is not JSON serializable`, and the trigger state file will not be
updated. The next cycle will use stale trigger baselines, potentially causing missed
triggers.

**Impact:** An exception in `check_triggers` between state assignment and save corrupts
the trigger flow for all subsequent cycles until restart.

**Fix:** Use a list instead of set: `state["_current_tickers"] = list(signals.keys())`

---

## P2 (Medium) Findings

### IND-009: Signal log entries grow unbounded — no pruning scheduled
**File:** `data/signal_log.jsonl` (written by `portfolio/signal_db.py`)
**Severity:** P2

`signal_log.jsonl` receives one entry per ticker per cycle (5 tickers × ~1440 cycles/day
= 7,200 entries/day). At ~2KB per entry, that's ~14MB/day, ~5GB/year. There is a
`prune_jsonl(path, max_entries=5000)` utility in `file_utils.py` but no evidence it's
called on `signal_log.jsonl` from any scheduled task or loop iteration.

`load_entries()` in `accuracy_stats.py` reads the ENTIRE file to compute accuracy. As
the file grows, accuracy computation time grows linearly, directly contributing to
BUG-178 slow cycles.

**Impact:** Performance degradation over time. The 7s+ accuracy load time documented in
BUG-178 will worsen as the file grows. Disk usage is moderate but unbounded.

**Fix:** Add periodic pruning in the main loop (e.g., once daily):
```python
from portfolio.file_utils import prune_jsonl
prune_jsonl(DATA_DIR / "signal_log.jsonl", max_entries=50000)  # ~7 days
```

### IND-010: `_DISABLED_SIGNAL_OVERRIDES` allows re-enabling ML for ETH-USD but ML is globally force-HOLD'd separately
**File:** `portfolio/signal_engine.py:297-301` vs `signal_engine.py:2376`
**Severity:** P2

The override `("ml", "ETH-USD")` at line 299 is in `_DISABLED_SIGNAL_OVERRIDES`, which
is checked at line 2620 (`if sig_name in DISABLED_SIGNALS and (sig_name, ticker) not in
_DISABLED_SIGNAL_OVERRIDES`). This would un-skip ML for ETH-USD in the enhanced signal
dispatch loop.

However, ML is also hardcoded at line 2376: `votes["ml"] = "HOLD"` — this runs BEFORE
the enhanced signal loop and is NOT gated by `_DISABLED_SIGNAL_OVERRIDES`. So ML gets
force-HOLD'd at line 2376 regardless of the override. If ML were ever moved to the
enhanced signal registry (instead of being a core signal), the override would work. As-is,
it's dead code — the override exists but has no effect.

**Impact:** Dead code. Not harmful but misleading — suggests ML is active for ETH-USD
when it isn't.

**Fix:** Remove the override or add a comment explaining it's pending ML migration to
the plugin system.

### IND-011: Stale `now` variable in `_cached` after func() call
**File:** `portfolio/shared_state.py:94`
**Severity:** P2

```python
_tool_cache[key] = {"data": data, "time": now, "ttl": ttl}
```
`now` was captured at line 48 (before the `_cache_lock` acquisition and `func()` call).
If `func()` takes 30 seconds (e.g., LLM inference), the cache entry's timestamp is 30s
in the past. The next caller within 30s of `func()` completing will see the entry as
still fresh (TTL not expired), but after the TTL the entry appears to have expired 30s
early, causing a premature refresh.

For a TTL of 900s (15 min), being off by 30s is 3.3% — minor. But for short TTLs like
60s, being off by 30s means the entry appears to expire at 30s instead of 60s, doubling
the refresh rate and the associated API/LLM load.

**Fix:** Use `time.time()` at the point of cache insertion:
```python
_tool_cache[key] = {"data": data, "time": time.time(), "ttl": ttl}
```

### IND-012: Dashboard TTL cache has no eviction — memory grows with unique API patterns
**File:** `dashboard/app.py:66-81`
**Severity:** P2

The `_cache` dict in the dashboard grows by one entry per unique cache key and is never
evicted. Each API endpoint creates a key (e.g., `"summary"`, `"portfolio"`, etc.). The
number of keys is bounded by the number of endpoints (~20), so in practice this is a
fixed-size cache. However, the `load_jsonl` calls for time-filtered views create
timestamp-parameterized keys, and if the frontend sends unique timestamps, the cache
grows without bound.

**Impact:** Minimal in practice unless the frontend generates unique cache keys per
request. The 5-second TTL means stale entries dominate quickly.

**Fix:** Add simple eviction when cache exceeds a threshold (e.g., 100 entries).

### IND-013: `_compute_applicable_count` doesn't account for regime gating or persistence filtering
**File:** `portfolio/signal_engine.py:871-900`
**Severity:** P2

`_compute_applicable_count` returns the theoretical maximum number of signals that COULD
vote for a ticker. But it doesn't subtract:
- Regime-gated signals (which are force-HOLD'd and don't vote)
- Persistence-filtered signals (which are force-HOLD'd on first cycle)
- Accuracy-gated signals

This count is stored as `total_applicable` at line 2815 and used in extra_info for
reporting. It's not used in any gate decision (those use `active_voters` from actual
BUY+SELL counts), but it creates a misleading "12/28 signals voting" display when the
real denominator should be 18 (after regime gating removes 10).

**Impact:** Misleading reporting only. No trading logic depends on this count.

---

## P3 (Low) Findings

### IND-014: Telegram poller doesn't validate sender identity for mode commands
**File:** `portfolio/telegram_poller.py` (based on avanza_orders.py pattern)
**Severity:** P3

The telegram poller checks `chat_id` matches the configured chat, which provides
sender validation. However, Telegram group chats allow any member to send messages.
If the configured chat_id is a group (not a private chat), any group member could
send `/mode` commands to change the system's notification mode.

**Impact:** Low — the system likely uses a private chat. But if it's ever configured
with a group chat ID, mode commands from unauthorized users would be accepted.

### IND-015: Hardcoded constant proliferation in signal_engine.py
**File:** `portfolio/signal_engine.py` (entire file)
**Severity:** P3

The file has 50+ hardcoded threshold constants (accuracy gates, correlation penalties,
regime weights, bias thresholds, etc.) scattered across ~400 lines of module-level
declarations. While each has good documentation, the interaction effects are
unpredictable — changing one constant can silently break assumptions in another.

**Impact:** Maintenance risk. The calibration compression at line 2147 assumes upstream
penalties produce confidence in a certain range. Changing regime penalties or unanimity
multipliers shifts that range without updating the compression curve.

---

## Cross-Cutting Observations

1. **Exception swallowing pattern is systemic.** The codebase has ~30 instances of
   `except Exception: ... logger.warning(...)` on safety-critical paths. The philosophy
   is "keep the loop running," but this creates a blind spot where multiple independently
   "minor" failures compose into a major failure (no accuracy data + no drawdown check +
   no regime detection = flying blind with no safety nets).

2. **Two Avanza implementations.** `avanza_session.py` (Playwright-based) and
   `portfolio/avanza/` (TOTP-based) coexist. Both have `place_stop_loss` functions.
   The canonical implementation is unclear, and a future developer might call the wrong
   one (regular order API instead of stoploss API, recreating the Mar 3 incident).

3. **Signal engine complexity.** `signal_engine.py` is ~3,000 lines with 7+ cascading
   gate layers. The interaction between regime gating, per-ticker gating, horizon gating,
   persistence filtering, accuracy gating, directional gating, and correlation group
   gating creates a combinatorial state space that is extremely difficult to reason about.
   A signal can be simultaneously: exempted from regime gating (per-ticker accuracy 60%),
   horizon-disabled (3h blacklist), accuracy-gated (global accuracy 45%), and correlation-
   penalized (0.12x in trend_direction group). The effective treatment depends on which
   gate runs first, and the ordering matters.

4. **No integration test for the full signal → trigger → invoke → trade path.**
   Each subsystem is well-tested in isolation, but the end-to-end path from signal
   computation through trigger detection, agent invocation, trade decision, and portfolio
   state update has no integration test. The RISK-001 (portfolio state overwrite) bug
   exists because the path was never tested end-to-end with concurrent writers.
