# Improvement Plan — Auto-Session 2026-04-17

Updated: 2026-04-17
Branch: `improve/auto-session-2026-04-17`
Worktree: `Q:/finance-analyzer-improve`

## Methodology

Five parallel exploration agents examined: core loop/orchestration, signal engine/accuracy,
data layer/file I/O, metals subsystem/Avanza, and dashboard/test infrastructure. Findings
were verified against source code. False positives from agents were filtered out.

## 1. Bugs & Problems Found

### P1 — trigger.py: wall-clock in sustained gate causes NTP-jump false negatives

**File:** `portfolio/trigger.py:57-81`
**Issue:** `_update_sustained()` stores and compares `started_ts` via `time.time()` (wall clock).
If NTP daemon adjusts the clock backward, `now_ts - entry["started_ts"]` goes negative and
the duration gate fails even if the signal has been sustained for minutes.
**Impact:** Sustained signal flips (consensus change triggers) can silently fail to fire.
**Fix:** Use `time.monotonic()` for duration tracking. Store a separate `_monotonic_ts` in
the state entry. The `started_ts` wall-clock field is preserved for persistence/debugging;
a new `_mono_start` field handles duration math. On process restart, `_mono_start` is
re-initialized (unknown — treat as "just started"), which is the correct behavior since
a restart already resets the sustained counter.

### P2 — agent_invocation.py: stack overflow counter not persisted

**File:** `portfolio/agent_invocation.py:39-40,617-643`
**Issue:** `_consecutive_stack_overflows` is an in-memory global that counts consecutive
Claude CLI stack overflow crashes. After 5, Layer 2 is auto-disabled. But on loop restart
(code deploy, crash, schtasks restart), the counter resets to 0, re-enabling Layer 2 even
if the underlying cause (large file in project root, etc.) persists.
**Impact:** Crash loop: restart -> 5 stack overflows -> disable -> restart -> repeat.
**Fix:** Persist counter to `data/stack_overflow_counter.json` via `atomic_write_json()`.
Load on module import. Clear on successful non-stack-overflow completion.

### P3 — microstructure_state.py: unprotected concurrent buffer access

**File:** `portfolio/microstructure_state.py:36-38`
**Issue:** `_snapshot_buffers`, `_spread_buffers`, `_ofi_history` are plain dicts accessed by
both the metals_loop 10s fast-tick thread and the main 60s cycle (via `accumulate_snapshot`
and `get_state`). Python dicts are thread-safe for individual operations but deque
append + iterate patterns can raise RuntimeError.
**Impact:** Potential `RuntimeError: deque mutated during iteration` in OFI computation.
**Fix:** Add `threading.Lock()` around buffer mutations and reads in public functions.

### P4 — trend.py: Supertrend direction inferred from float equality

**File:** `portfolio/signals/trend.py:164`
**Issue:** `supertrend[i - 1] == upper_band[i - 1]` compares numpy floats with `==`.
Floating-point arithmetic can produce epsilon differences, causing the equality check
to fail and the direction to be incorrectly determined.
**Impact:** Supertrend direction flip can be missed, producing wrong trend signal.
**Fix:** Use the `direction` array (already allocated) for state transitions instead of
inferring from float equality. Replace `supertrend[i-1] == upper_band[i-1]` with
`direction[i-1] == -1` (was downtrend).

### P5 — health.py: unguarded fromisoformat in agent silence check

**File:** `portfolio/health.py:103`
**Issue:** `datetime.fromisoformat(last_ts)` can raise `ValueError` on corrupt state file.
**Impact:** Agent silence detection crashes, propagating exception to caller.
**Fix:** Wrap in try-except, return `silent=True` on parse error (fail-closed).

### P6 — outcome_tracker.py: backfill doesn't invalidate signal utility cache

**File:** `portfolio/outcome_tracker.py` (end of `backfill_outcomes`)
**Issue:** After writing new outcomes to signal_log, the signal_utility TTL cache in
accuracy_stats.py (300s TTL) isn't invalidated. Next cycle sees stale utility scores.
**Impact:** Signal utility weights lag by up to 5 minutes after outcome backfill.
**Fix:** Call `invalidate_signal_utility_cache()` at end of `backfill_outcomes()`.

## 2. Performance Improvements

### PERF-1 — market_timing.py: cache holiday sets per year

**File:** `portfolio/market_timing.py:158-216`
**Issue:** `us_market_holidays()` and `swedish_market_holidays()` recompute the full
holiday set including Easter calculation on every call. Called every 60s cycle.
**Impact:** Unnecessary CPU: divmod + date arithmetic 1440 times/day.
**Fix:** Module-level `_holiday_cache` dict keyed by `(country, year)`. Auto-invalidates
on year boundary.

## 3. Code Quality

### CQ-1 — Ruff auto-fixable lint violations

4 auto-fixable violations + manual SIM103/UP017:
- 2 x I001 (unsorted imports): `accuracy_degradation.py:431`, `metals_loop.py:298`
- 1 x F401 (unused import): identified during exploration
- 1 x SIM103 (needless bool): portfolio module
- 1 x UP017 (datetime-timezone-utc): portfolio module

E402 (54 occurrences) are intentional lazy imports — leave as-is.

### CQ-2 — analyze.py: use monotonic clock for elapsed timing

**File:** `portfolio/analyze.py:270,280,314`
**Issue:** Uses `time.time()` for elapsed measurement. Should use `time.monotonic()`.
**Fix:** Replace 3 call sites.

## 4. Deferred (Not This Session)

- **metals_loop.py decomposition** — 7295 lines, 94 functions. Needs careful multi-session
  refactoring with live trading path testing. Too risky for autonomous session.
- **Dashboard CORS restriction** — Intentional LAN-access design. Requires user input on
  whether to restrict.
- **Dashboard auth default** — "No token = open access" is backwards-compatible design.
  Changing default requires user approval.

## 5. Batch Plan

### Batch 1: Bug Fixes (5 files, 5 new test files)
| File | Change | Risk |
|------|--------|------|
| `portfolio/trigger.py` | P1: monotonic clock in `_update_sustained()` | LOW |
| `portfolio/agent_invocation.py` | P2: persist stack overflow counter | LOW |
| `portfolio/microstructure_state.py` | P3: thread safety for buffers | LOW |
| `portfolio/signals/trend.py` | P4: fix Supertrend float equality | MEDIUM |
| `portfolio/health.py` | P5: guard fromisoformat parse | LOW |

### Batch 2: Perf + Cache + Quality (3 files)
| File | Change | Risk |
|------|--------|------|
| `portfolio/market_timing.py` | PERF-1: cache holiday sets | LOW |
| `portfolio/outcome_tracker.py` | P6: invalidate utility cache | LOW |
| `portfolio/analyze.py` | CQ-2: monotonic clock | LOW |

### Batch 3: Lint Cleanup
| File | Change | Risk |
|------|--------|------|
| Various | CQ-1: ruff auto-fix | LOW |
