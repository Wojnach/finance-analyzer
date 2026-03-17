# Improvement Plan

Updated: 2026-03-17
Branch: improve/auto-session-20260317

Previous sessions: 2026-03-05 through 2026-03-16.

## Session Plan (2026-03-17)

### Theme: Silent Exception Elimination & Error Visibility

Previous sessions completed signal health tracking (BUG-51), dynamic applicable count
(BUG-52), atomic JSONL appends (BUG-53), and dead IO fallback removal (BUG-55). This
session addresses the largest remaining class of hidden failures: silent `except Exception: pass`
blocks that swallow errors without logging. These are the single biggest source of invisible
bugs in production.

### 1) Bugs & Problems Found

#### BUG-56 (P1): 15+ silent `except Exception: pass` blocks swallow errors

- **Files** (highest risk first):
  - `portfolio/message_store.py:148` — send failure silenced
  - `portfolio/avanza_session.py:135-148` — 3 separate silent catches in session management
  - `portfolio/focus_analysis.py:111` — analysis error silenced
  - `portfolio/agent_invocation.py:310` — agent process cleanup error silenced
  - `portfolio/analyze.py:233,393` — 2 analysis errors silenced
  - `portfolio/golddigger/runner.py:291,296,376,381` — 4 golddigger errors silenced
  - `portfolio/metals_precompute.py:596` — precompute error silenced (has comment "Non-critical")
  - `portfolio/fin_snipe_manager.py:1363` — snipe manager error silenced
  - `portfolio/health.py:244` — health status circuit breaker import error silenced
- **Issue**: When these code paths fail, the system silently continues with potentially
  stale or missing data. Debugging production issues requires correlating timestamps across
  multiple log files because the actual failure point is invisible.
- **Fix**: Replace `except Exception: pass` with `except Exception: logger.debug(...)` at
  minimum for non-critical paths, or `logger.warning(...)` for paths where failure affects
  data quality. Keep the pass/continue behavior — just add logging.
- **Impact**: High. This is the #1 source of invisible production bugs. Every silent failure
  that gets logged becomes a diagnosable issue instead of a mystery.

#### BUG-57 (P2): `analyze.py:392` uses non-atomic JSONL append

- **File**: `portfolio/analyze.py:390-392`
- **Issue**: Uses raw `open("a") + write()` instead of `atomic_append_jsonl()` from file_utils.
  On crash during write, the file could end up with a partial JSON line.
- **Fix**: Replace with `atomic_append_jsonl(WATCH_LOG_FILE, event)`.
- **Impact**: Low-medium. `analyze.py` watch log is non-critical, but consistency matters.

#### BUG-58 (P2): `message_store.py` silently fails on send

- **File**: `portfolio/message_store.py:148`
- **Issue**: The `except Exception: pass` in message delivery means Telegram send failures
  are completely invisible. The message is logged to JSONL (good) but delivery failure
  is never reported.
- **Fix**: Add `logger.warning("Telegram delivery failed: %s", e)` before the pass.
  Do NOT raise — message logging should still succeed even if delivery fails.
- **Impact**: Medium. Users won't know they missed a notification.

#### BUG-59 (P3): `avanza_session.py` has 3 silent cleanup catches

- **File**: `portfolio/avanza_session.py:135,141,147`
- **Issue**: Session cleanup (browser close, page close, etc.) silently catches all
  exceptions. While cleanup failures are indeed non-critical, they can indicate
  resource leaks (zombie browser processes, file handle leaks).
- **Fix**: Add `logger.debug("session cleanup: %s", e)` to each catch block.
- **Impact**: Low. Helps diagnose resource leaks.

### 2) Architecture Improvements

#### ARCH-15: Structured exception logging helper

- **Files**: `portfolio/file_utils.py` (add helper), multiple consumers
- **What**: Add a `log_exception(logger, msg, exc, level="debug")` helper that provides
  a consistent pattern for logging exceptions that shouldn't be raised. This makes the
  "catch and log" pattern a one-liner instead of repeating `logger.X("...: %s", e)`.
- **Why**: Currently each silent catch needs manual conversion. A helper makes it trivial
  to convert `except Exception: pass` to `except Exception as e: log_exception(...)`.
- **Decision**: Skip this — it's over-engineering. The standard `logger.debug/warning` is
  sufficient and more readable. Just do the manual conversion.

### 3) Refactoring TODOs

#### REF-11: Convert silent exception handlers to logged ones

- **Scope**: All 15+ `except Exception: pass` blocks identified in BUG-56.
- **Pattern**: For each block:
  1. Add `as e` to capture the exception
  2. Add appropriate log level:
     - `logger.debug(...)` for truly non-critical cleanup (session close, temp file removal)
     - `logger.warning(...)` for data-affecting failures (message send, analysis, session mgmt)
  3. Keep the original control flow (pass/continue/return)
- **Impact assessment**: Zero behavioral change — only adds logging. No tests should break.

#### REF-12: Replace raw JSONL append in analyze.py

- **Scope**: `portfolio/analyze.py:390-392`
- **Pattern**: Replace `open("a") + write()` with `atomic_append_jsonl()`.

### 4) Dependency/Ordering

**Batch 1** (REF-11 + BUG-56): Silent exception elimination
- Files: ~10 portfolio modules
- Risk: Zero — only adds logging, no behavior change
- Tests: Run full suite to verify no regressions

**Batch 2** (REF-12 + BUG-57): Atomic JSONL in analyze.py
- Files: `portfolio/analyze.py`
- Risk: Near-zero — same behavior, just atomic
- Tests: Run test_analyze.py if it exists

### 5) What We're NOT Doing

- **Not refactoring reporting.py exception handlers**: Those 26 `except Exception:` blocks
  already have proper `logger.warning()` + `_module_warnings.append()`. They're correct.
- **Not touching signal modules**: The `except Exception:` blocks in signal modules (calendar,
  fibonacci, momentum, etc.) are the sub-indicator isolation pattern — each sub-indicator is
  independently wrapped so one failure doesn't kill the whole signal. These are correct.
- **Not adding new features**: Focus is purely on error visibility.
