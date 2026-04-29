# Improvement Plan — Auto-Improve Session 2026-04-29

## Summary

Continuation of auto-improve. Prior session (2026-04-28) fixed BUG-230 (CORS),
BUG-231 (heartbeat atomicity), BUG-232 (NaN fx_rate). This session addresses
circuit breaker reliability, risk management performance, and log rotation coverage.

---

## 1. Bugs & Improvements Found

### BUG-245 (P2): Circuit breaker uses fixed recovery timeout — no backoff
- **File**: `portfolio/circuit_breaker.py`
- **Issue**: `recovery_timeout` is fixed (default 60s). During extended API outages
  (e.g., Binance maintenance), the breaker cycles OPEN→HALF_OPEN→OPEN every 60s
  indefinitely, generating probe requests and log noise.
- **Fix**: Add exponential backoff on failed recovery. Double timeout on each
  HALF_OPEN→OPEN transition, cap at 5 minutes. Reset on successful recovery.
- **Impact**: Reduces retry pressure during extended outages. No behavioral change
  during normal operation (first recovery attempt still at original timeout).

### FEAT-3 (P3): `_streaming_max` scans entire JSONL every call
- **File**: `portfolio/risk_management.py:21-51`
- **Issue**: `_streaming_max()` reads the full portfolio history file line-by-line
  on every invocation. At 60s cycle cadence, after 30 days that's ~43K lines per
  call. The peak value can only increase (append-only file), so re-scanning old
  entries is redundant.
- **Fix**: Cache peak value + byte offset. On subsequent calls, seek to last offset
  and only scan new entries. Invalidate cache if file shrinks (rotation).
- **Impact**: O(new_entries) instead of O(all_entries) per call. ~43K→~1 line
  in steady state.

### FEAT-5 (P3): Log rotation coverage gap
- **File**: `portfolio/log_rotation.py`
- **Issue**: 59 JSONL files in data/, but `ROTATION_POLICIES` covers only 8.
  Most others are self-managed by modules, but several high-growth files are
  uncovered: `accuracy_snapshots.jsonl`, `contract_violations.jsonl`,
  `sentiment_ab_log.jsonl`, `forecast_health.jsonl`.
- **Fix**: Add rotation policies for uncovered high-growth files.
- **Impact**: Prevents unbounded disk growth from files not covered by
  module-level pruning.

---

## 2. Implementation Batches

### Batch 1: Circuit breaker backoff + peak cache (2 files)
1. `portfolio/circuit_breaker.py` — exponential backoff on recovery timeout
2. `portfolio/risk_management.py` — cached peak value with byte-offset seek

### Batch 2: Log rotation + tests (2 files)
1. `portfolio/log_rotation.py` — add missing JSONL rotation policies
2. `tests/test_circuit_breaker.py` — backoff behavior tests
3. `tests/test_risk_management.py` — peak cache tests

---

## 3. Risk Assessment

- **Circuit breaker backoff**: Low risk. Only changes retry cadence during
  extended failures. First probe is at the original timeout. Adds no new
  failure modes. Fully backward-compatible constructor API.
- **Peak cache**: Low risk. Cache invalidation on file-shrink handles rotation.
  Falls back to full scan if seek fails. No change to peak value computation.
- **Log rotation**: Very low risk. Adding policies for files that currently
  have none. No change to existing policies.
