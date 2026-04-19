# Improvement Plan — Auto Session 2026-04-19

Based on deep exploration of the full codebase by 4 parallel agents covering:
- Signal system (signal_engine.py, accuracy_stats.py, 33 signal modules)
- Core loop (main.py, trigger.py, agent_invocation.py, data_collector.py, health.py)
- Portfolio & risk (portfolio_mgr.py, risk_management.py, trade_guards.py, equity_curve.py)
- Infrastructure (file_utils.py, shared_state.py, claude_gate.py, dashboard, telegram)

## Assessment

**Overall**: Production-grade system with excellent atomic I/O, thread safety, and error
handling. All previously identified bugs (BUG-XXX series) have targeted fixes. The main
improvement opportunities are in test reliability and minor resilience gaps.

**Infrastructure**: 4-5 star quality across all modules. No unresolved bugs.
**Signal system**: Well-designed weighted consensus with multiple gating layers.
**Portfolio/Risk**: Solid atomic state management, correct metrics calculations.
**Core loop**: Good crash recovery, but minor resilience gaps.

---

## Priority Order

### Batch 1: xdist Test Hygiene — Module State Reset Fixtures (HIGHEST IMPACT)

**Problem:** `pytest -n auto` produces 5-10 different failures per run. Root cause is
module-level mutable state that leaks across xdist workers. This is the #1 item in
`docs/IMPROVEMENT_BACKLOG.md` (TEST-HYGIENE-1) and affects every CI/merge-verification run.

**Files to create/modify:**
- `tests/conftest.py` — Add autouse fixtures for the most-affected modules
- Individual test files that need per-test state resets

**Modules with mutable global state (catalogued during exploration):**

| Module | Mutable State | Leak Risk |
|--------|--------------|-----------|
| `agent_invocation` | `_agent_proc`, `_agent_log`, `_agent_start`, `_agent_start_wall`, `_agent_timeout`, `_agent_tier`, `_agent_reasons`, `_journal_ts_before`, `_telegram_ts_before`, `_agent_log_start_offset`, `_consecutive_stack_overflows` | HIGH |
| `signal_engine` | `_adx_cache`, `_last_signal_per_ticker`, `_phase_log_per_ticker`, `_prev_sentiment`, `_prev_sentiment_loaded`, `_sentiment_dirty` | HIGH |
| `shared_state` | `_tool_cache`, `_run_cycle_id`, `_regime_cache`, `_regime_cache_cycle`, `_full_llm_cycle_count`, `_current_market_state`, `_newsapi_daily_count` | HIGH |
| `forecast (signal)` | `_FORECAST_MODELS_DISABLED`, `_KRONOS_ENABLED`, `_kronos_tripped_until`, `_chronos_tripped_until`, `_predictions_dedup_cache` | MEDIUM |
| `accuracy_stats` | `_accuracy_cache` (module-level TTL cache) | MEDIUM |
| `llama_server` | `_local_proc`, `_local_model` | MEDIUM |
| `logging_config` | `_configured` | LOW |
| `api_utils` | `_config_cache`, `_config_mtime` | LOW |
| `trigger` | `_startup_grace_active` | LOW |

**Changes:**
1. Create `tests/_state_reset.py` with helper functions to reset each module's state
2. Add autouse session-scoped fixture in `conftest.py` that resets HIGH-risk modules
3. For MEDIUM-risk modules, add per-test resets in affected test files
4. Verify: run `pytest -n auto -q` 3x, confirm 0 new failures each run

**Impact:** Eliminates 5-10 random test failures per CI run. Makes test results trustworthy.
**Risk:** LOW — only adds reset fixtures, no production code changes.

---

### Batch 2: Crash Recovery Resilience (MEDIUM IMPACT)

**Problem 1:** Crash counter (`_consecutive_crashes` in main.py) resets to 0 on process
restart. If a wrapper script immediately restarts the loop after a crash, the counter resets
and alerts re-fire from 1. Pattern already solved in agent_invocation.py (stack_overflow_counter).

**Problem 2:** Backoff delay is exponential but not jittered. Two simultaneously crashing
loops (main + metals) retry in lockstep, causing synchronized load spikes.

**Problem 3:** After 5 consecutive crashes, Telegram alerts are fully suppressed with no
periodic summary — operators lose visibility into ongoing failures.

**Files:** `portfolio/main.py` (lines 929-976)

**Changes:**
1. Persist crash counter to `data/crash_counter.json` (like stack_overflow_counter pattern)
2. Add jitter: `delay = delay * (0.5 + random.random())`
3. After suppression threshold, send one summary alert every 100 crashes

**Impact:** More robust crash recovery, no alert blindness during extended outages.
**Risk:** LOW — crash recovery path only, doesn't affect normal operation.

---

### Batch 3: JSONL Prune Per-File Failure Isolation (MEDIUM IMPACT)

**Problem:** `_run_post_cycle()` prunes 3 JSONL files in a single try/except. If any file
is locked (e.g., by antivirus), ALL pruning fails silently. Over time, unpruned files can
grow to hundreds of MB.

**File:** `portfolio/main.py` (lines 346-354)

**Changes:**
1. Move to per-file try/except with individual error logging
2. Log which specific file(s) failed

**Impact:** Prevents unbounded file growth when a single file is locked.
**Risk:** VERY LOW — error handling path only.

---

### Batch 4: Dead Code Cleanup (LOW IMPACT)

**Problem 1:** `trigger.py` persists `started_ts` (wall-clock) in sustained flip state but
never reads it back. The `_mono_start` field is used for duration checks. The wall-clock
field is dead code that confuses readers.

**Problem 2:** `health.py` `check_agent_silence()` reads `last_invocation_ts` from fallback
(parsing invocations.jsonl) but never writes it back to the cache. This means the cache
stays stale forever for concurrent readers.

**Files:** `portfolio/trigger.py`, `portfolio/health.py`

**Changes:**
1. Remove `started_ts` persistence from trigger.py sustained state (keep `_mono_start` only)
2. In health.py, write back `last_invocation_ts` to health_state after fallback read

**Impact:** Cleaner code, correct health cache.
**Risk:** VERY LOW — dead code removal and cache consistency fix.

---

### Batch 5: Test Coverage for New Improvements (LOW IMPACT)

**Files:** New test files for batch 2-4 changes

**Changes:**
1. Test crash counter persistence (load/save/reset across restarts)
2. Test jitter is within expected range
3. Test per-file JSONL prune failure isolation
4. Test health cache write-back

**Risk:** NONE — test-only changes.

---

## Dependency Order

Batch 1 (xdist hygiene) → Batch 2 (crash recovery) → Batch 3 (prune isolation) →
Batch 4 (dead code) → Batch 5 (tests for 2-4)

Batches 2-4 are independent and could run in parallel, but sequential is safer for
commit clarity.

## What NOT to Implement (Deferred)

- **Metals loop split (7634→3 files):** Too large for this session. Risk of breaking live system.
- **reporting.py split (1188 lines):** Cosmetic, doesn't affect correctness.
- **Thread pool→Process pool:** Python limitation, can't forcibly kill threads. Accepted design.
- **Signal module NaN handling standardization:** Low impact, all modules already handle NaN safely.
- **Auth failure detection hardening:** Already has 3-layer validation, low marginal value.
- **Cross-process Claude lock:** Currently single-process, not needed yet.
