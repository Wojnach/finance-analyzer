# Improvement Plan — Auto-Session #9 (2026-03-05)

## Status: IN PROGRESS

## Priority: Critical Bugs > Architecture > Tests > Features > Polish

Previous sessions fixed BUG-1 through BUG-60, ARCH-1 through ARCH-13, REF-1 through REF-13.
This session continues from BUG-61 onward.

---

## 1. Bugs & Problems Found

### BUG-61: autonomous.py silent pass on Telegram send (line 604)
- **File:** `portfolio/autonomous.py:604`
- **Severity:** LOW (notification lost silently)
- **Issue:** `except Exception: pass` when sending Telegram notification. If Telegram is down,
  the notification is lost without any log entry — user never knows Layer 2 spoke.
- **Fix:** Replace with `except Exception as e: logger.warning("Telegram send failed: %s", e)`.

### BUG-62: fx_rates.py silent pass on FX fetch (line 50)
- **File:** `portfolio/fx_rates.py:50`
- **Severity:** LOW (stale FX rate used without visibility)
- **Issue:** `except Exception: pass` when fetching USD/SEK rate from Frankfurter API. On
  failure, the cache returns stale data — but there's no log that the fetch failed or how
  stale the cached value is.
- **Fix:** Replace with `except Exception as e: logger.warning("FX rate fetch failed: %s", e)`.

### BUG-63: outcome_tracker.py silent pass on outcome processing (line ~78)
- **File:** `portfolio/outcome_tracker.py`
- **Severity:** LOW (outcome records silently dropped)
- **Issue:** Bare `except Exception: pass` in outcome processing loop. If one record fails
  to process, it's silently skipped — corrupting accuracy statistics without any trace.
- **Fix:** Replace with `except Exception as e: logger.warning("Outcome processing failed for entry: %s", e)`.

### BUG-64: journal.py silent passes (multiple locations)
- **File:** `portfolio/journal.py`
- **Severity:** LOW (journal load/save failures invisible)
- **Issue:** Multiple `except Exception: pass` blocks when loading/saving journal state.
  If the journal file is corrupt, entries are silently lost.
- **Fix:** Replace each with appropriate `logger.warning()` calls.

### BUG-65: message_store.py silent pass on save (line ~40)
- **File:** `portfolio/message_store.py`
- **Severity:** LOW (message log entries silently dropped)
- **Issue:** `except Exception: pass` when saving message to JSONL. If disk is full or
  file is locked, the message record is lost.
- **Fix:** Replace with `except Exception as e: logger.warning("Message store save failed: %s", e)`.

### BUG-66: telegram_notifications.py silent pass on send (line ~60)
- **File:** `portfolio/telegram_notifications.py`
- **Severity:** LOW (Telegram send failure invisible)
- **Issue:** `except Exception: pass` when sending Telegram message. This is the primary
  notification path — silent failure means the user gets no alerts at all.
- **Fix:** Replace with `except Exception as e: logger.warning("Telegram send failed: %s", e)`.

### BUG-67: forecast.py silent pass on Kronos init (line 59)
- **File:** `portfolio/signals/forecast.py:59`
- **Severity:** LOW (Kronos silently disabled without logging why)
- **Issue:** `except Exception: pass` in `_init_kronos_enabled()`. If config is corrupt or
  missing the kronos key, Kronos is silently disabled. The operator has no way to know
  whether Kronos is off by config or by error.
- **Fix:** Replace with `except Exception as e: logger.debug("Kronos init: %s", e)`.

### BUG-69: main.py DATA_DIR inconsistency in _run_post_cycle
- **File:** `portfolio/main.py:161`
- **Severity:** LOW (fragile path derivation)
- **Issue:** `_run_post_cycle()` re-derives `DATA_DIR` via
  `Path(__file__).resolve().parent.parent / "data"` instead of using the module-level
  `DATA_DIR` constant defined at line 30. If the module is ever relocated, the
  re-derivation would silently break while the constant would be updated in one place.
- **Fix:** Use the existing `DATA_DIR` constant.

### BUG-70: main.py in-function imports that could be module-level
- **File:** `portfolio/main.py:306-410`
- **Severity:** LOW (code clarity, minor startup cost on each call)
- **Issue:** Several safe, always-available imports (pathlib, json, datetime) are done
  inside functions instead of at module level. These add minor overhead per call and
  reduce readability. Note: imports of optional/heavy modules (like torch, chromadb)
  should stay in-function to avoid import-time failures.
- **Fix:** Move safe stdlib imports to module level. Keep optional/conditional imports in-function.

---

## 2. Architecture Improvements

### ARCH-14: Observability for silent exception handlers
- **Files:** 7 files across portfolio/
- **Why:** BUG-61 through BUG-67 fix. Replace bare `except: pass` with logged warnings.
- **Impact:** Pure observability improvement. No behavior change for callers. All exceptions
  still caught — they're just logged now instead of silently swallowed.

### ARCH-15: main.py path and import cleanup
- **File:** `portfolio/main.py`
- **Why:** BUG-69 + BUG-70 fix. Use DATA_DIR constant consistently, move safe imports to
  module level.
- **Impact:** Pure cleanup. No behavior change.

---

## 3. Test Improvements

### TEST-16: Verify logged warnings for previously-silent exceptions
- **File:** `tests/test_silent_exceptions.py` (new)
- **Why:** BUG-61 through BUG-67. Verify that each fixed exception handler actually logs
  a warning when an error occurs, rather than silently passing.
- **Approach:** For each fixed handler, monkeypatch the failing dependency, trigger the
  code path, and assert that `logger.warning` (or `logger.debug`) was called.

---

## 4. Items NOT Planned (Justified)

1. **30+ additional silent except:pass in reporting.py** — Already fixed in session #7
   (BUG-13 through BUG-17). The remaining ones in reporting.py are `logger.warning` calls.
2. **Type hints across codebase** — Valuable but enormous scope. Not justified for one session.
3. **Refactor signal_engine.py** (727 lines) — Complex, high-risk module touching live trading.
   Not safe to refactor autonomously without extensive integration testing.
4. **Upgrade Kronos reliability** — Kronos has 0.5% success rate. Fixing requires GPU/model
   debugging, not code changes. Out of scope.
5. **config.json credentials migration** — Requires manual migration to env vars. Too risky
   to automate.

---

## 5. Dependency/Ordering — Implementation Batches

### Batch 1: Silent exception logging (7 files, BUG-61 through BUG-67 + ARCH-14 + TEST-16)
**Files changed:**
- `portfolio/autonomous.py` — Replace `except Exception: pass` with logged warning
- `portfolio/fx_rates.py` — Replace `except Exception: pass` with logged warning
- `portfolio/outcome_tracker.py` — Replace `except Exception: pass` with logged warning
- `portfolio/journal.py` — Replace multiple `except Exception: pass` with logged warnings
- `portfolio/message_store.py` — Replace `except Exception: pass` with logged warning
- `portfolio/telegram_notifications.py` — Replace `except Exception: pass` with logged warning
- `portfolio/signals/forecast.py` — Replace `except Exception: pass` with `logger.debug`
**Tests:** `tests/test_silent_exceptions.py` (new)
**Test impact:** New test file only. No existing tests affected.

### Batch 2: main.py cleanup (1 file, BUG-69 + BUG-70 + ARCH-15)
**Files changed:**
- `portfolio/main.py` — Use DATA_DIR constant in `_run_post_cycle()`, move safe stdlib
  imports to module level
**Tests:** Existing tests should still pass. No new test file needed (pure refactor).
**Test impact:** None. Pure cleanup.
