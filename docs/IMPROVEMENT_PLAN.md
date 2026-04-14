# Improvement Plan — Auto-Session 2026-04-12

Updated: 2026-04-12
Branch: improve/auto-session-2026-04-12
Status: **COMPLETE**

Previous session (2026-04-11): shipped directional accuracy weights (BUG-182),
per-ticker throttle, trade guards locking. All verified in codebase.

## Session Context

Deep exploration by 6 parallel agents found 22+ potential issues. After manual verification
against the actual code, many were false positives (see "Rejected" below). This plan covers
only verified, confirmed bugs and improvements.

### False Positives Rejected (agents were wrong)
- `structure.py` highlow breakout: CORRECT breakout/momentum logic (not mean-reversion)
- `crypto_macro.py` OPTIONS_TTL: Valid Python (module-level vars resolved at call time)
- `oscillators.py` Aroon: Off-by-one claim was wrong (argmax math checks out)
- `equity_curve.py` Sortino: Standard formula divides by total observations (correct)
- `hurst_regime.py` fill_method=None: Works fine in pandas 2.3.3
- `kelly_sizing.py` edge cases: Documented intentional behavior

---

## 1. Bugs & Problems Found

### BUG-185: Directional accuracy KeyError risk (MEDIUM)
- **File**: `portfolio/signal_engine.py:861,863`
- **Issue**: `stats["buy_accuracy"]` accessed without `.get()`. If the key is missing (cache
  corruption or version mismatch), raises KeyError. The `total_buy >= 20` guard passes but
  `buy_accuracy` could be absent in a corrupt cache entry.
- **Fix**: Use `stats.get("buy_accuracy", acc)` with fallback to overall accuracy.
- **Impact**: Theoretical crash in weighted consensus. Unlikely but zero-cost to fix.

### BUG-186: Blended accuracy `correct` field inconsistency (LOW)
- **File**: `portfolio/accuracy_stats.py:650`
- **Issue**: `correct` uses all-time count while `accuracy` is a blended value. The ratio
  `correct/total` ≠ `accuracy`. Downstream reporting uses this value.
- **Fix**: Compute `correct` from blended accuracy: `int(round(blended * total))`.
- **Impact**: Reporting inconsistency only. Gating decisions use `accuracy` field directly.

### BUG-187: Circuit breaker HALF_OPEN dead code (LOW)
- **File**: `portfolio/circuit_breaker.py:89-92`
- **Issue**: The HALF_OPEN probe branch (`if not self._half_open_probe_sent`) never executes.
  The OPEN→HALF_OPEN transition at line 84-86 sets the flag AND returns True. Lines 89-92
  are unreachable.
- **Fix**: Remove dead code, add comment explaining probe is handled in OPEN→HALF_OPEN.
- **Impact**: Code clarity only.

### BUG-188: Redundant acc_horizon assignment (LOW)
- **File**: `portfolio/signal_engine.py:1826`
- **Issue**: Same computation at line 1813 and 1826. Line 1826 inside try block is redundant.
- **Fix**: Remove line 1826, add comment referencing line 1813.
- **Impact**: Code clarity only.

### BUG-189: Agent invocation orphaned process risk (MEDIUM)
- **File**: `portfolio/agent_invocation.py:198-201`
- **Issue**: When taskkill fails, `_agent_proc` is set to None. Next cycle can spawn a
  new agent while old process may still be running. Two concurrent agents writing to
  same journal/Telegram. Also: taskkill rc=128 (process already exited) incorrectly
  treated as failure.
- **Fix**: Treat rc=128 as success. Track orphaned PIDs for cleanup.
- **Impact**: Prevents duplicate decisions on taskkill failure.

### BUG-190: Digest loads entire invocations.jsonl (LOW)
- **File**: `portfolio/digest.py`
- **Issue**: Uses `load_jsonl()` which reads entire file, unlike signal_log which uses
  `load_jsonl_tail()`. Performance degrades over time.
- **Fix**: Replace with `load_jsonl_tail()`.

---

## 2. Architecture Improvements

None needed this session — previous sessions addressed the major ones (directional weights,
per-ticker throttle, trade guards locking).

---

## 3. Useful Features

### FEAT-A: Edge case test coverage for signal engine
Add tests for:
- Directional accuracy gating with missing keys (covers BUG-185)
- Blended accuracy edge cases (covers BUG-186)
- Circuit breaker probe lifecycle (covers BUG-187)
- Per-ticker directional gating verification

---

## 4. Refactoring TODOs

- Remove circuit breaker dead code
- Remove redundant signal_engine assignment
- Use `.get()` for safety in directional accuracy access

---

## 5. Ordering — Batches

### Batch 1: Safety Fixes (3 files, ~15 lines changed)
1. `portfolio/signal_engine.py` — `.get()` safety (BUG-185) + remove redundant line (BUG-188)
2. `portfolio/accuracy_stats.py` — Fix `correct` field in blend (BUG-186)
3. `portfolio/circuit_breaker.py` — Remove dead code (BUG-187)

### Batch 2: Agent Invocation Reliability (1 file, ~15 lines changed)
4. `portfolio/agent_invocation.py` — Graceful taskkill handling (BUG-189)

### Batch 3: Performance (1 file, ~3 lines changed)
5. `portfolio/digest.py` — Use tail read for invocations (BUG-190)

### Batch 4: Test Coverage (~100 lines added)
6. Tests for directional gating edge cases
7. Tests for circuit breaker probe lifecycle
8. Tests for blended accuracy edge cases

---

## 6. Risk Assessment

- **Batch 1** (safety fixes): Zero risk — defensive `.get()`, dead code removal, math fix.
  All changes are backwards-compatible.
- **Batch 2** (agent invocation): Low risk — improves recovery behavior. No change to
  happy path.
- **Batch 3** (digest performance): Zero risk — same data, faster access.
- **Batch 4** (tests): Zero risk — additive only.
