# Improvement Plan

Updated: 2026-03-12
Branch: worktree-auto-session-2026-03-12

Previous sessions: 2026-03-05, 2026-03-06, 2026-03-07, 2026-03-08, 2026-03-09, 2026-03-10, 2026-03-11.

## Session Plan (2026-03-12)

### 1) Bugs & Problems Found

#### BUG-39 (P1): `check_agent_completion()` is never called from the main loop

- **File**: `portfolio/agent_invocation.py:245`, `portfolio/main.py`
- **Issue**: `check_agent_completion()` was implemented (with 30+ tests in
  `test_agent_completion.py`) but is never called from `main.py` or anywhere
  in production code. This means:
  1. Agent completion status (success/incomplete/failed) is never recorded
  2. `get_completion_stats()` always returns zero counts
  3. The digest message's "Succeeded/Failed" counts are computed as
     `invoked - journal_entries`, a fragile heuristic instead of actual tracking
  4. Running agents are never checked for timeout between invocations —
     timeout only fires when a new trigger tries to invoke a new agent
- **Fix**: Add `check_agent_completion()` call at the start of each `run()` cycle
  and after the post-cycle housekeeping. This ensures completed agents are detected
  promptly and their status is logged.
- **Impact**: High. Enables accurate agent completion tracking, which feeds into
  digest messages and health monitoring.

#### BUG-40 (P2): `digest.py` reads/writes `trigger_state.json` — race with `trigger.py`

- **File**: `portfolio/digest.py:33,40-46`
- **Issue**: `_get_last_digest_time()` and `_set_last_digest_time()` both operate
  on `trigger_state.json`, which is also read/written by `trigger.py` every cycle.
  The pattern is: load → modify → atomic_write. If `trigger.py._save_state()` runs
  between digest's load and write, the trigger state changes are overwritten.
  This is a classic read-modify-write race condition. While single-threaded in
  practice (both run in the same loop iteration), the execution order within
  `_run_post_cycle` → `_maybe_send_digest` means trigger state was just saved,
  then digest reads it, modifies one key, and saves — potentially dropping any
  concurrent writes from other modules.
- **Fix**: Move digest state to its own file `data/digest_state.json`. The only
  key needed is `last_digest_time`. This eliminates the shared-file contention.
- **Impact**: Medium. Prevents silent data loss in trigger_state.json.

#### BUG-41 (P2): `daily_digest.py` also reads `trigger_state.json`

- **File**: `portfolio/daily_digest.py:25`
- **Issue**: Same contention pattern as BUG-40. `daily_digest.py` reads
  `trigger_state.json` for `last_daily_digest_time`. Should use its own file.
- **Fix**: Move daily digest state to `data/daily_digest_state.json`.
- **Impact**: Medium. Same race elimination as BUG-40.

#### BUG-42 (P2): `reporting.py` reads portfolio files with raw `json.loads()` — TOCTOU risk

- **File**: `portfolio/reporting.py:331-338`
- **Issue**: Portfolio state files are loaded with `json.loads(path.read_text())`
  wrapped in try/except. This bypasses `load_json()` which was specifically
  designed to handle the TOCTOU race (BUG-30, fixed in session 2026-03-10).
  If the file is being atomically rewritten when read, `read_text()` could return
  partial content.
- **Fix**: Use `load_json(path, default={})` instead.
- **Impact**: Low. The try/except catches the exception, but inconsistent with
  the established pattern.

#### BUG-43 (P2): `trigger.py` `_check_recent_trade()` uses raw `json.loads()` — same TOCTOU

- **File**: `portfolio/trigger.py:78-79`
- **Issue**: Same pattern as BUG-42. `json.loads(pf_file.read_text())` instead
  of `load_json()`.
- **Fix**: Use `load_json(path, default={})`.
- **Impact**: Low. Consistency fix.

#### BUG-44 (P3): `_last_jsonl_ts()` scans entire file — O(n) for last entry

- **File**: `portfolio/agent_invocation.py:90-110`
- **Issue**: `_last_jsonl_ts()` reads the entire JSONL file line-by-line to find
  the last `ts` value. The signal_log.jsonl can be 5000+ entries. This is called
  twice at agent invocation start (once for journal, once for telegram). Meanwhile,
  `check_agent_silence()` in `health.py:99-105` already has the efficient
  tail-read implementation (seek to last 4KB, parse backwards).
- **Fix**: Refactor `_last_jsonl_ts()` to read from the end of the file (last 4KB),
  matching the pattern in `health.py`.
- **Impact**: Low. Performance improvement for agent invocation path.

#### BUG-45 (P3): `digest.py` loads entire JSONL files without limit

- **File**: `portfolio/digest.py:59,93,114`
- **Issue**: `_build_digest_message()` loads three JSONL files entirely into memory
  (invocations, journal, signal_log) then filters by 4-hour cutoff. Signal_log can
  have 5000+ entries. Most entries are discarded.
- **Fix**: Use `load_jsonl(path, limit=500)` to cap memory usage. 500 entries
  covers ~8 hours of data at 1-per-minute, well beyond the 4-hour window.
- **Impact**: Low. Memory optimization.

#### BUG-46 (P3): `reporting.py` calls `load_config()` 3+ times per cycle

- **File**: `portfolio/reporting.py:362,389,454`
- **Issue**: Each section (monte_carlo, price_targets, focus_probabilities) imports
  and calls `load_config()` independently, reading and parsing `config.json` from
  disk 3+ times per reporting cycle.
- **Fix**: Load config once at the top of `write_agent_summary()` and pass it
  into sections that need it.
- **Impact**: Low. Minor I/O reduction.

### 2) Architecture Improvements

#### ARCH-15: Centralize JSONL tail-read utility

- **Files**: `portfolio/file_utils.py`, `portfolio/agent_invocation.py`, `portfolio/health.py`
- **Issue**: The "read last entry from JSONL" pattern is duplicated in
  `agent_invocation._last_jsonl_ts()` (full scan) and `health.check_agent_silence()`
  (efficient tail read). Both should use a shared utility.
- **Fix**: Add `last_jsonl_entry(path, field=None)` to `file_utils.py`. Returns
  the last parsed JSON entry, or the value of a specific field if `field` is set.
  Migrate both callers.
- **Impact**: DRY + performance. One implementation to test and maintain.

### 3) Test Coverage Gaps

#### TEST-9: Test `check_agent_completion()` integration in main loop

- **File**: `tests/test_main_agent_completion.py`
- **Issue**: Tests exist for `check_agent_completion()` in isolation but there
  are no tests verifying it is called from the main loop. Need an integration
  test that confirms the main loop calls completion checking.
- **Fix**: Add a test that patches `check_agent_completion` and verifies it's
  called during a `run()` cycle.

#### TEST-10: Test digest state isolation

- **File**: `tests/test_digest.py` (or existing file)
- **Issue**: After moving digest state to its own file, need tests verifying
  that the new file is used and trigger_state.json is not modified.

### 4) Refactoring TODOs

#### REF-7: Remove legacy trigger_state.json migration in signal_engine.py

- **File**: `portfolio/signal_engine.py:57-61`
- **Issue**: Migration code reads `trigger_state.json` for `prev_sentiment` as
  fallback when `sentiment_state.json` doesn't exist. This migration was added
  months ago and sentiment_state.json has been the primary store since. The
  migration path is dead code.
- **Fix**: Remove the fallback branch. Keep only the `sentiment_state.json` read.

### 5) Dependency/Ordering

#### Batch 1: Agent Completion Tracking (BUG-39, TEST-9)
- Files: `portfolio/main.py`, `portfolio/agent_invocation.py`
- Tests: Add integration test
- Zero-risk addition — no existing behavior changed

#### Batch 2: Eliminate trigger_state.json Contention (BUG-40, BUG-41, REF-7, TEST-10)
- Files: `portfolio/digest.py`, `portfolio/daily_digest.py`, `portfolio/signal_engine.py`
- Tests: Verify new state files, no trigger_state.json modification
- Must verify that existing digest behavior is preserved

#### Batch 3: I/O Safety & Performance (BUG-42, BUG-43, BUG-44, BUG-45, BUG-46, ARCH-15)
- Files: `portfolio/reporting.py`, `portfolio/trigger.py`, `portfolio/agent_invocation.py`,
  `portfolio/file_utils.py`
- Independent of Batches 1 and 2

## Summary

| Category | Count | Items |
|----------|-------|-------|
| Bugs | 8 | BUG-39 through BUG-46 |
| Architecture | 1 | ARCH-15 |
| Test Coverage | 2 | TEST-9, TEST-10 |
| Refactoring | 1 | REF-7 |
| Batches | 3 | Batch 1 critical → Batch 2 safety → Batch 3 performance |
