# Improvement Plan — Auto Session 2026-04-20

Based on deep exploration by 5 parallel agents covering: signal system (signal_engine.py,
accuracy_stats.py, 36 signal modules), core loop (main.py, trigger.py, agent_invocation.py,
data_collector.py, health.py), portfolio & risk (portfolio_mgr.py, risk_management.py,
trade_guards.py, equity_curve.py), infrastructure (file_utils.py, shared_state.py,
loop_contract.py, dashboard, telegram), and test suite (242 test files, ~5994 tests).

## Prior Plan Status (2026-04-19)

All 5 batches from the prior plan are **COMPLETED** (implemented between sessions):
- Batch 1 (xdist hygiene) — autouse fixtures in conftest.py ✓
- Batch 2 (crash recovery) — persisted counter, jitter, periodic summary ✓
- Batch 3 (JSONL prune isolation) — per-file try/except ✓
- Batch 4 (dead code) — trigger.py started_ts removed, health.py write-back added ✓
- Batch 5 (tests) — covered ✓

## New Findings — Assessment

**Overall**: Production-grade system. All previously documented bugs (BUG-15 through BUG-186)
have targeted fixes or mitigations. The main improvement opportunities are: 3 real bugs
found during exploration, plus I/O modernization in loop_contract.py.

---

## Batch 1: Bug Fixes (HIGHEST PRIORITY)

### 1.1 risk_management.py — False positive regime mismatch on missing volume

**File:** `portfolio/risk_management.py` (lines 670-679)

**Problem:** `check_regime_mismatch()` treats `volume_ratio is None` as a failing condition.
When volume data is unavailable (e.g., off-hours, API timeout), the function flags a regime
mismatch even though there's no evidence of one. The `None` case should be "unknown, don't
flag" rather than "confirmed mismatch".

**Fix:** Add explicit `volume_ratio is not None` guard before the comparison. When volume
data is missing, skip the regime mismatch flag (fail-open for missing data, fail-closed only
for confirmed low-volume counter-trend trades).

**Impact:** Eliminates false positive risk flags during data gaps. These flags could cause
Layer 2 to avoid valid trades or reduce confidence unnecessarily.

**Risk:** LOW — only affects the regime mismatch advisory flag, not trade execution.

### 1.2 signal_engine.py — Silent exception handlers need debug logging

**File:** `portfolio/signal_engine.py` (lines 2527, 2972, 2987, 3019, 3052)

**Problem:** 4 bare `except Exception: pass` handlers silently swallow ALL exceptions,
including unexpected bugs (import errors, type errors, data corruption). These are labeled
"graceful degradation" but provide zero diagnostic information when something goes wrong.
The graceful-degradation intent is correct — these are optional enhancement stages — but
they should log at debug level so failures are diagnosable from logs.

**Fix:** Replace `pass` with `logger.debug("...", exc_info=True)` in all 4+1 handlers.
Preserves graceful degradation (no crash, no visible noise) while making failures
diagnosable.

**Impact:** Faster debugging when optional components fail silently.

**Risk:** VERY LOW — debug-level logging only, no behavior change.

### 1.3 loop_contract.py — I/O functions bypass file_utils

**File:** `portfolio/loop_contract.py` (lines 145-181)

**Problem:** Two local I/O functions duplicate existing file_utils functionality:
- `_read_json()` uses raw `json.load()` instead of `file_utils.load_json()`
- `_last_jsonl_entry()` reads ENTIRE file O(N) instead of `file_utils.last_jsonl_entry()`
  which reads only the last 4KB O(1)

The module already imports from file_utils in `ViolationTracker._save()` (line 898) and
`_log_violations()` (line 951), so the "self-contained" design argument doesn't hold.

**Fix:** Replace `_read_json()` calls with `file_utils.load_json()` and `_last_jsonl_entry()`
calls with `file_utils.last_jsonl_entry()`. Remove the two local helper functions.

**Impact:** Consistent I/O (atomic reads, better error handling), O(1) JSONL tail reads
instead of O(N) full-file scans. As journal files grow, the O(N) scan will become
increasingly expensive since it runs every 60s cycle.

**Risk:** LOW — file_utils functions are battle-tested and used everywhere else.

---

## Batch 2: Tests for Batch 1

**Files:** New/modified test files

### 2.1 test_risk_management.py — regime mismatch with None volume_ratio

Test that `check_regime_mismatch()` returns None (no flag) when volume_ratio is missing
from agent_summary, for both BUY-in-downtrend and SELL-in-uptrend scenarios.

### 2.2 test_signal_engine.py — graceful degradation logging

Test that the signal_engine's optional stages (seasonality, market health, earnings gate,
linear factor, per-ticker consensus) still produce valid results when their imports fail,
and that failures are logged at debug level.

### 2.3 test_loop_contract.py — file_utils integration

Test that contract checks produce correct results using the file_utils implementations.
Verify behavior for JSONL files.

---

## Dependency Order

Batch 1 (bug fixes) → Batch 2 (tests)

Batch 1 items are independent of each other and could be implemented in parallel.

## What NOT to Implement (Deferred)

- **BUG-186 (accuracy_stats.py:802):** `correct = int(round(blended * total))` rounding
  inconsistency. Cosmetic — no downstream code recomputes accuracy from `correct/total`
  after blending. The `accuracy` field is authoritative.
- **Disabled signal registry cleanup:** 12 signals registered but force-HOLD'd in dispatch.
  Registry consumers all respect DISABLED_SIGNALS. No real risk.
- **Dynamic correlation groups performance:** O(n²) pairwise computation on 2h TTL cache.
  Acceptable for current signal count (~36).
- **Metals loop split (7634 lines):** Too large for auto-improvement. Manual session needed.
- **Per-ticker signal filtering:** Research priority, not a bug fix.
