# Improvement Plan

Updated: 2026-03-22
Branch: improve/auto-session-2026-03-22

Previous sessions: 2026-03-05 through 2026-03-21.

## Session Plan (2026-03-22)

### Theme: Digest Safety, Budget Tracking, Reporting Tests

Previous session (2026-03-21) fixed crash safety (BUG-101), thread safety (BUG-102),
zero-division (BUG-103), silent failures (BUG-104), alert routing (BUG-105), and
memory leak (BUG-106).

This session addresses five verified issues found by deep code audit:

1. **Zero-division** — digest P&L calculation crashes on corrupt portfolio state (same
   class as BUG-103, but in digest.py and daily_digest.py — missed in prior fix)
2. **Thread safety** — Alpha Vantage `_daily_budget_used` incremented without lock
3. **Performance** — digest reads entire 68MB signal_log.jsonl to get last 500 entries
4. **Test coverage** — reporting.py (1,109 lines) has ZERO tests despite being critical
   Layer 2 input builder
5. **Stale import** — digest.py imports `load_jsonl` from `portfolio.stats` instead of
   canonical `portfolio.file_utils`

---

### 1) Bugs & Problems Found

#### BUG-107 (P2): Zero-division in digest P&L calculations

- **Files**: `portfolio/digest.py:149`, `portfolio/daily_digest.py:204,213`
- **Issue**: `state["initial_value_sek"]` used as divisor without zero guard. Same class
  as BUG-103 (fixed in main.py) and BUG-99 (fixed in reporting.py), but NOT fixed in the
  two digest modules. If portfolio state is corrupt (`initial_value_sek: 0` or missing),
  the digest crashes with ZeroDivisionError, preventing the user from receiving their 4h
  and daily digest messages.
- **Fix**: Guard: `initial = state.get("initial_value_sek") or INITIAL_CASH_SEK`
- **Impact**: Digest crash blocks user notifications for 4+ hours until next attempt.

#### BUG-108 (P3): Alpha Vantage budget counter not thread-safe

- **File**: `portfolio/alpha_vantage.py:159-164,277`
- **Issue**: `_daily_budget_used` is a module-level int incremented (`+= 1`) and read
  (`_check_budget()`) without any lock protection. While `refresh_fundamentals_batch()`
  is currently called single-threaded from the main loop post-cycle, the lack of
  synchronization is inconsistent with the rest of the module (which uses `_cache_lock`
  for all `_cache` operations) and creates a latent bug if the function is ever called
  from ThreadPoolExecutor.
- **Fix**: Protect budget reads/writes with `_cache_lock` (reuse existing lock — budget
  operations are infrequent and short).
- **Impact**: Currently LOW (single-threaded caller), but inconsistent pattern.

#### BUG-109 (P3): Digest reads entire signal_log.jsonl (68MB+)

- **File**: `portfolio/digest.py:120`
- **Issue**: `load_jsonl(SIGNAL_LOG_FILE, limit=500)` reads the entire 68MB+ file line
  by line (deque keeps last 500). The digest only needs entries from the last 4 hours,
  but there's no efficient way to seek to recent entries. This causes a ~2-3 second
  I/O spike every 4 hours.
- **Fix**: Use `last_jsonl_entries()` helper that reads from the end of the file, or
  add a `tail_bytes` parameter to `load_jsonl` that seeks to the last N bytes before
  parsing. 500 entries × ~500 bytes each ≈ 250KB tail read instead of 68MB.
- **Impact**: Performance — 2-3s I/O pause every 4 hours. Not critical but wasteful.

#### BUG-110 (P3): Stale import path in digest.py

- **File**: `portfolio/digest.py:59`
- **Issue**: `from portfolio.stats import load_jsonl` — imports `load_jsonl` via the
  `stats.py` re-export rather than the canonical `portfolio.file_utils`. This creates
  an unnecessary dependency on `stats.py` and masks the actual data source.
- **Fix**: Change to `from portfolio.file_utils import load_jsonl`.
- **Impact**: Code clarity only. No behavioral change.

#### COVERAGE-1 (P2): reporting.py has ZERO test coverage

- **File**: `portfolio/reporting.py` (1,109 lines, ~20 functions)
- **Issue**: The reporting module builds `agent_summary.json` and all tiered context
  files that Layer 2 reads for every invocation. It is the most critical untested module
  in the system. Functions like `write_agent_summary()`, `_write_compact_summary()`, and
  `_write_tier2_summary()` process all 30 signals, macro context, accuracy data, risk
  flags, and Monte Carlo results. A regression here silently corrupts Layer 2 input.
- **Fix**: Write targeted tests for core functions: `write_agent_summary()` output
  structure, `_write_compact_summary()` three-tier compaction, `_cross_asset_signals()`,
  and `_module_warnings` propagation.
- **Impact**: HIGH — regressions in reporting silently degrade all Layer 2 decisions.

---

### 2) Implementation Batches

#### Batch 1: Zero-division & Import Fixes (3 files)

| Bug | File | Change |
|-----|------|--------|
| BUG-107 | digest.py | Guard `state["initial_value_sek"]` division with `or INITIAL_CASH_SEK` |
| BUG-107 | daily_digest.py | Same guard for patient and bold P&L calculations |
| BUG-110 | digest.py | Change `from portfolio.stats import load_jsonl` to `from portfolio.file_utils import load_jsonl` |

**Risk**: LOW — additive guards, import path change has identical behavior.

#### Batch 2: Thread Safety & Performance (2 files)

| Bug | File | Change |
|-----|------|--------|
| BUG-108 | alpha_vantage.py | Wrap `_daily_budget_used` reads/writes with `_cache_lock` |
| BUG-109 | file_utils.py | Add `load_jsonl_tail()` helper that reads from file end |
| BUG-109 | digest.py | Use `load_jsonl_tail()` for signal_log reads |

**Risk**: LOW — lock addition is additive. File tail-read is new code but isolated.

#### Batch 3: Reporting Tests

| Test | Covers |
|------|--------|
| test_reporting_core.py | write_agent_summary output structure, _write_compact_summary three-tier compaction, _cross_asset_signals, _module_warnings propagation |

**Risk**: NONE — tests only, no production code changes.

---

### 3) What Was NOT Changed (and Why)

- **sentiment.py test coverage**: 608 lines, 0 tests. Large module with many external
  dependencies (subprocess calls, HTTP). Would need extensive mocking. Deferred.
- **main.py test coverage**: 851 lines, 0 tests. Loop orchestration is hard to unit test.
  Integration testing would be more valuable. Deferred.
- **NewsAPI quota persistence**: `_newsapi_daily_count` resets on restart. LOW risk since
  the daily budget (90) is generous and restarts are rare during active hours.
- **Circuit breaker state persistence**: State is in-memory only. On restart, breakers
  reset to CLOSED. This is acceptable — the recovery timeout (60s) means a brief burst
  of retries, then the breaker re-trips if the API is still down.
- **Stale data files in data/**: ~50 experimental scripts and state files. Cleanup would
  be valuable but is housekeeping, not a bug fix. Deferred to a dedicated cleanup session.

---

### 4) Results

| ID | Type | Status | Details |
|----|------|--------|---------|
| BUG-107 | Zero-division | FIXED | `digest.py:150`, `daily_digest.py:204,214` — added `or INITIAL_CASH_SEK` guard |
| BUG-108 | Thread safety | FIXED | `alpha_vantage.py:164,281` — budget ops wrapped in `_cache_lock` |
| BUG-109 | Performance | FIXED | `file_utils.py:74-125` — new `load_jsonl_tail()`, `digest.py:121` uses it |
| BUG-110 | Stale import | FIXED | `digest.py:59` — changed to `from portfolio.file_utils import load_jsonl` |
| COVERAGE-1 | Tests | DONE | `test_reporting_core.py` — 50 tests for reporting.py (was 0) |

**New test files:** 2 (`test_bug_fixes_session_mar22.py` — 11 tests, `test_reporting_core.py` — 50 tests)
**Total new tests:** 61
**Regressions:** 0 (4528 passed, 139 failed pre-existing, 10 errors pre-existing)
