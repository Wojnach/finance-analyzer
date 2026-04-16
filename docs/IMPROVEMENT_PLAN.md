# Improvement Plan — Auto-Session 2026-04-16

Updated: 2026-04-16
Branch: `improve/auto-session-2026-04-16`
Worktree: `Q:/fa-improve-20260416`

## Session Context

Three consecutive overnight Layer 2 outages (2026-04-14 through 16) — each ~4-5h
silent. Pattern: a trigger fires ~01:00 UTC, the claude CLI subprocess returns
exit 0 while printing "Not logged in", and the system doesn't detect the failure
until ~05:00 UTC when the `layer2_journal_activity` contract finally fires (60 min
grace + cycle latency + trigger age > grace).

Existing detection infrastructure is correctly built (`detect_auth_failure` in
`claude_gate.py`, `check_layer2_journal_activity` in `loop_contract.py`,
`scripts/check_critical_errors.py` surface at session start), but it has two
**gaps** that keep the detection slow:

1. **Bypass sites.** `bigbet.py`, `iskbets.py`, and `analyze.py` invoke
   `claude -p` with direct `subprocess.run()` — skipping `claude_gate` entirely.
   Auth failures from these paths are never recorded, so they never reach
   `critical_errors.jsonl` to kick off the startup-check surfacing chain.
2. **Slow grace window.** `LAYER2_JOURNAL_GRACE_S = 3600` (1 hour) gives too
   much room. Valid T3 invocations complete in <=900s (15 min); anything past
   that is definitionally a stall. The grace period should match the real
   upper bound on a healthy invocation, not the observed worst case.

This session fixes both. It also picks up three small quality fixes that
agents surfaced while exploring.

---

## 1. Bugs & Problems Found

### BUG-200: bigbet.py bypasses detect_auth_failure (P1)
- **File**: `portfolio/bigbet.py:168-192`
- **Issue**: `invoke_layer2_eval()` calls `subprocess.run(["claude", "-p", ...])`
  directly and only checks `result.returncode == 0 and output`. When auth
  expires, claude exits 0 with stdout = `"Not logged in -- Please run /login"`.
  The response parser can't extract a number, so the function degrades to
  returning `(None, "")` -- but **the auth failure is never recorded** to
  `critical_errors.jsonl`, so `check_critical_errors.py` never surfaces it.
  Bigbet runs BEFORE Layer 2 on some code paths, so its silent failure can
  precede the slower contract detection by hours.
- **Fix**: After subprocess completes, call
  `detect_auth_failure(result.stdout + result.stderr, caller="bigbet_layer2")`
  before parsing the output. If True, return the safe default and exit.
- **Impact**: Auth failures from the bigbet path now trigger the same
  `critical_errors.jsonl` escalation as the main Layer 2 path, bringing
  detection latency from "never" to "next cycle".

### BUG-201: iskbets.py bypasses detect_auth_failure (P1)
- **File**: `portfolio/iskbets.py:304-353`
- **Issue**: Identical pattern to BUG-200. Worse consequence: the function
  **defaults to `approved=True`** when parsing fails, meaning an auth-failure
  output could be interpreted as gate-approved for a warrant trade. This is
  a real safety gap, not just a detection gap.
- **Fix**: Same as BUG-200 -- call `detect_auth_failure` after subprocess.
  On auth failure, return `(False, "auth failure")` -- the default-approve
  policy should NEVER pass an auth-failed output.
- **Impact**: Prevents auth-failed outputs from becoming approved gate
  decisions. Surfaces auth_failure to the critical journal immediately.

### BUG-202: LAYER2_JOURNAL_GRACE_S is 1h (P1 contributing)
- **File**: `portfolio/loop_contract.py:42`
- **Issue**: Grace period is 3600s. Layer 2 T3 invocation timeout is 900s
  (15 min). Any invocation that takes longer than 900s has already been
  killed by the timeout. So 900s is the tightest-meaningful grace window.
  1h was a safety margin but empirically produces 4-5h detection windows
  (trigger at 01:00 + 60m grace = earliest detection 02:00, but we've seen
  05:00 consistently -- the extra gap is cycle cadence + stale state reads).
  Tightening to 15 min (900s) brings worst-case detection to ~15-20 min
  after the trigger instead of 60-300 min.
- **Fix**: `LAYER2_JOURNAL_GRACE_S = 15 * 60` with an inline comment
  explaining the 900s = T3 timeout justification.
- **Impact**: Overnight silent failures detected in 15-20 min instead of
  hours. No false positives expected: T3 invocations genuinely complete
  within 900s or they're killed.

### BUG-203: agent_invocation.py elapsed uses time.time() (P3)
- **File**: `portfolio/agent_invocation.py`
- **Issue**: Wall-clock `time.time()` is susceptible to NTP jumps. For
  measuring "how long has the agent been running", `time.monotonic()` is
  the right primitive. Current bug severity is low on Windows (NTP jumps
  small), but it's the correct fix and trivial to apply.
- **Fix**: Replace `time.time()` with `time.monotonic()` for elapsed-
  duration calculations only. Keep `time.time()` where a real wall-clock
  timestamp is needed (log entries, journal records).
- **Impact**: Small robustness improvement for timeout correctness.

### BUG-204: qwen3_signal.py silent exception in GPU reaper (P3)
- **File**: `portfolio/qwen3_signal.py` (~line 153)
- **Issue**: `except Exception: pass` around GPU process reaper. Reaper
  failures are invisible -- if the reaper itself is broken, VRAM leaks
  silently.
- **Fix**: Replace with `logger.debug("GPU reaper failed", exc_info=True)`.
- **Impact**: Diagnostic improvement only. Leaks would still happen but
  now observable.

### BUG-205: dashboard/app.py silent exception in market_health (P3)
- **File**: `dashboard/app.py` (~line 1182)
- **Issue**: `except Exception: pass` around optional market_health
  enrichment. If the enrichment source is broken, API silently omits the
  field without any trace.
- **Fix**: Replace with `logger.debug` form. Same as BUG-204.
- **Impact**: Diagnostic only.

---

## 2. Architecture Notes

All proposed changes preserve the existing modularity. The auth-failure
detector in `claude_gate.py` is already well-factored; the fix is simply
routing the three bypass sites through it. No new abstractions needed.

The `LAYER2_JOURNAL_GRACE_S` constant is the only single point of truth for
the detection window; tightening it here automatically tightens all callers.
Keep the value as a module-level constant, not in config, to prevent
accidental loosening during an incident.

---

## 3. Test Coverage Additions

### TEST-A: Auth failure detection at bigbet + iskbets
- `tests/test_bigbet_auth_failure.py` (new) -- mock `subprocess.run` to
  return `"Not logged in"` with exit 0; verify the function returns the
  safe default AND that `detect_auth_failure` was invoked (critical_errors
  entry written).
- `tests/test_iskbets_auth_failure.py` (new) -- same pattern. Additionally,
  verify that the default-approve policy is overridden to `approved=False`
  on auth failure.

### TEST-B: Layer 2 journal grace window
- Extend `tests/test_loop_contract.py` (or new test file) with 2-3 tests
  covering:
  - Trigger age < 15 min (grace) -> no violation
  - Trigger age = 15 min + 1s, no journal -> violation recorded
  - Trigger with recent journal (post-trigger) -> no violation

### TEST-C: Monotonic time in agent_invocation
- Deferred -- the change is internal to functions that use `time.time()` as
  a local epoch, not exposed via API. Existing tests of invocation timeout
  still pass.

---

## 4. Ordering -- Batches

### Batch 1: Bigbet + iskbets auth routing (BUG-200, BUG-201)
- `portfolio/bigbet.py` + `portfolio/iskbets.py` (+ `portfolio/analyze.py` if
  the same pattern)
- New tests: `tests/test_bigbet_auth_failure.py`, `tests/test_iskbets_auth_failure.py`

### Batch 2: Journal grace window tightening (BUG-202)
- `portfolio/loop_contract.py`
- Tests: extend `tests/test_loop_contract.py`

### Batch 3: Monotonic time + silent exception logging (BUG-203-205)
- `portfolio/agent_invocation.py`
- `portfolio/qwen3_signal.py`
- `dashboard/app.py`

### Batch 4 (optional): Ruff F541 in scripts/verify_kronos.py
- `scripts/verify_kronos.py` -- 7 f-strings without placeholders (cosmetic).

---

## 5. Deferred

The following surfaced during exploration but are out of scope:

- **fin_fish.py / metals_precompute.py / oil_precompute.py / reporting.py test
  coverage.** All >1000 lines with zero unit tests. Adding meaningful tests
  requires fixture work for Avanza sessions and Binance FAPI responses.
  Deferred -- track as TEST-4 for a future session.
- **Direct subprocess in `analyze.py:273`.** Same pattern as bigbet/iskbets
  but `analyze.py` is a manual CLI tool invoked by the user, not an
  autonomous signal path. Lower priority; bundle with Batch 1 if time permits.
- **Scheduled `check_critical_errors.py` notifier.** Could fire a Telegram
  alert every hour if unresolved errors exist, independent of Claude session
  starts. Deferred -- the existing session-start surfacing chain is adequate
  once Batches 1-2 land.

---

## 6. Impact Assessment

- **BUG-200/201 fix:** Additive. The code path already has the safe default
  fallback; we're just adding the escalation. No behavior change for healthy
  invocations.
- **BUG-202 fix:** Tightens the detection window only. No effect on healthy
  cycles. False-positive risk is low because T3 timeout already caps
  healthy invocation duration at 900s.
- **BUG-203-205:** Diagnostic only. No behavior change.

Each batch is independently shippable and independently revertable.
