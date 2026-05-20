# Improvement Plan — Auto-Session 2026-05-20

Created: 2026-05-20
Branch: `improve/auto-session-2026-05-20`
Prior session (2026-05-19): All 7 P1 fixes implemented and merged.

## Exploration Summary

4 parallel exploration agents + manual review of 30+ modules.
System is mature (244 bugs fixed, ~6000 tests, 67 signal modules).
Security strong (hmac auth, CF JWT, symlink-safe writes, URL redaction).
Focus this session: test reliability, documentation accuracy, minor code fixes.

---

## 1. Bugs & Problems Found

### B0: CRITICAL — _journal_count_before NameError in completion detection (P0)
**File:** `portfolio/agent_invocation.py`
**Impact:** PRODUCTION BUG since commit 28af2f73 (May 17). `_journal_count_before` and
`_telegram_count_before` are set as local vars in `invoke_agent()` (line 1065-1066) but
referenced as globals in `_check_agent_completion_locked()` (line 1401-1402). Missing from:
(a) module-level declarations, (b) `global` statement in `invoke_agent()` (line 680-681),
(c) `global` statement in `_check_agent_completion_locked()` (line 1345-1346).
Result: every agent completion raises NameError. The watchdog swallows it via `except
Exception` (line 107). Completion status, invocation logging, and new-trade detection
(BUG-219 record_trade wiring) all silently fail.
**Fix:** Add module-level declarations and update both `global` statements.
**Risk:** Very low — restoring intended behavior.

### B1: Consensus test flakes from production accuracy_cache.json (HIGH)
**Files:** `tests/test_consensus.py`, `tests/conftest.py`
**Impact:** 7 tests fail nondeterministically. The `_null_cached` mock doesn't intercept
`accuracy_stats.get_or_compute_accuracy()` (imported inside function body, bypasses
`signal_engine._cached`). Production accuracy data gates signals below 47%, changing
voter counts depending on what the production loop last wrote.
**Fix:** (a) Add session-scoped conftest fixture redirecting ACCURACY_CACHE_FILE,
REGIME_ACCURACY_CACHE_FILE, and TICKER_ACCURACY_CACHE_FILE to tmp paths. (b) Add
`_bypass_accuracy_gates` fixture for consensus tests that mocks all 5 accuracy_stats
functions imported inside generate_signal.
**Risk:** Low. Test-only changes.

### B2: data_collector.py rate limiter retries lack jitter (MEDIUM)
**File:** `portfolio/data_collector.py`
**Impact:** On Binance 429, all 8 ThreadPoolExecutor workers retry at identical intervals.
**Fix:** Add `random.uniform(0, base_delay)` jitter to the retry delays.
**Risk:** Very low — only affects retry timing.

### B3: SYSTEM_OVERVIEW.md stale (signal counts, features) (MEDIUM)
**File:** `docs/SYSTEM_OVERVIEW.md`
**Impact:** Says 65 modules / 49 disabled. Reality: 67 modules / 51 disabled.
Missing: ConnorsRSI, ADX regime switch, trigger density gate, May 19 P1 fixes.
**Fix:** Update to current state.
**Risk:** None — documentation only.

### B4: shared_state.py newsapi_ttl_for_ticker DST edge (LOW)
**File:** `portfolio/shared_state.py`
**Impact:** During CET/CEST transitions (2x/year), NewsAPI cache TTL boundary shifts 1h.
**Fix:** Use timezone-aware hour check.
**Risk:** Very low.

### B5: Documented but deferred issues (NOT fixing)
- ARCH-17: main.py re-exports (breaks 10+ tests, needs migration)
- ARCH-18: metals_loop.py monolith (7800+ lines, dedicated session)
- P1.6: Backtester look-ahead bias (structural redesign)
- P1.12: Horizon mismatch 3d/5d/10d→1d (needs multi-horizon accuracy infra)
- BUG-149: meta_learner.py orphaned (documented, no production impact)
- trade_guards TOCTOU (theoretical — Layer 2 is single-threaded by design)

---

## 2. Implementation Batches

### Batch 1: Fix consensus test flakes (B1)
**Files:** `tests/conftest.py`, `tests/test_consensus.py`
**Tests affected:** 7 known flakes + any other accuracy-coupled tests
**Impact check:** Only test infrastructure changes. No production code.

### Batch 2: data_collector jitter + shared_state DST (B2, B4)
**Files:** `portfolio/data_collector.py`, `portfolio/shared_state.py`
**Impact check:** Both used by main loop. Full test suite after.

### Batch 3: SYSTEM_OVERVIEW update (B3)
**File:** `docs/SYSTEM_OVERVIEW.md`
**Impact check:** None — documentation.

---

## 3. Success Criteria

- [ ] Full test suite green (no new failures)
- [ ] Consensus test flakes eliminated
- [ ] SYSTEM_OVERVIEW reflects current system state
- [ ] Adversarial review passes
- [ ] Merged to main and pushed
