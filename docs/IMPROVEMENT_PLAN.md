# Improvement Plan — Auto-Session #6 (2026-03-03)

## Status: COMPLETE

## Priority: Critical Bugs > Architecture > Tests > Features > Polish

Previous sessions fixed BUG-1 through BUG-41, ARCH-1 through ARCH-9, REF-1 through REF-12.
This session continues from BUG-42 onward.

---

## 1. Bugs & Problems Found

### BUG-42: signal_engine.py trap detection silently swallows exceptions
- **File:** `portfolio/signal_engine.py:273-274`
- **Severity:** HIGH (affects trading decisions)
- **Issue:** Bull/bear trap detection at line 257-274 catches all exceptions with bare `pass`.
  If the DataFrame has unexpected column types, NaN values, or missing "volume" column,
  the trap detection silently fails — meaning a bull/bear trap penalty is never applied.
  This could cause the system to enter a trapped position.
- **Fix:** Replace `except Exception: pass` with `except Exception: logger.warning(...)`.
- **Impact:** `signal_engine.py` only. No behavior change when trap detection succeeds.

### BUG-43: main.py signal error log missing stack trace
- **File:** `portfolio/main.py:252`
- **Severity:** MEDIUM (hinders debugging)
- **Issue:** `logger.error("%s: %s", name, e)` logs only the exception message, not the
  full stack trace. When a ticker fails (e.g., API returns unexpected JSON), the log shows
  "BTC-USD: KeyError: 'close'" with no trace of which function or line caused it.
- **Fix:** Add `exc_info=True` to the logger.error call.
- **Impact:** `main.py` only. Improves debuggability.

### BUG-44: bigbet.py has 3 silent exception swallowers
- **File:** `portfolio/bigbet.py:100-101, 214-215`
- **Severity:** MEDIUM (loses macro context and trade alert data)
- **Issue:** Lines 100-101 silently fail when fetching macro context (DXY, FOMC) for big bet
  evaluation. Lines 214-215 silently fail when appending to JSONL gate log, losing trade
  alert data. Line 41-42 (state load) is acceptable — returns default empty state.
- **Fix:** Add `logger.warning(...)` to lines 100-101 and 214-215.
- **Impact:** `bigbet.py` only. No behavior change on success path.

### BUG-45: iskbets.py has 3 silent exception swallowers (same pattern as bigbet)
- **File:** `portfolio/iskbets.py:268-269, 365-366`
- **Severity:** MEDIUM (loses macro context and gate log data)
- **Issue:** Same pattern as BUG-44. Macro context fetch and JSONL append fail silently.
  Line 76-77 (state load) is acceptable — returns default empty state.
- **Fix:** Add `logger.warning(...)` to lines 268-269 and 365-366.
- **Impact:** `iskbets.py` only.

### BUG-46: macro_context.py volume signal fetch fails silently
- **File:** `portfolio/macro_context.py:210-211`
- **Severity:** MEDIUM (volume signal returns None, HOLD vote)
- **Issue:** `get_volume_signal()` fetches Binance klines to compute volume ratio. If the
  fetch fails (network error, rate limit), the exception is swallowed and the function
  returns None. Caller in signal_engine.py gets no volume data → HOLD vote.
  Problem: no log trail — indistinguishable from "no volume spike detected".
- **Fix:** Add `logger.warning("Volume signal fetch failed for %s", ticker, exc_info=True)`.
- **Impact:** `macro_context.py` only.

### BUG-47: sentiment.py dissemination score fails silently
- **File:** `portfolio/sentiment.py:412-413`
- **Severity:** LOW (dissemination is a multiplier, default 1.0)
- **Issue:** If `dissemination_score()` raises, the multiplier defaults to 1.0 (no effect).
  But there's no log to indicate the feature is broken.
- **Fix:** Add `logger.debug(...)` (debug level — non-critical feature).
- **Impact:** `sentiment.py` only.

### BUG-48: main.py _re.compile() called inside function body every invocation
- **File:** `portfolio/main.py:130`
- **Severity:** LOW (performance — pattern compiled on every trigger)
- **Issue:** `_re.compile(r'^([A-Z][A-Z0-9]*(?:-[A-Z]+)?)\s+...')` is inside
  `_extract_triggered_tickers()`. Since this function is called on every trigger (~20-50/day),
  the pattern is recompiled each time.
- **Fix:** Move `_re.compile()` to module level as `_TICKER_PAT`.
- **Impact:** `main.py` only. Minor perf improvement.

---

## 2. Architecture Improvements

### ARCH-10: Deduplicate post-cycle helpers in main.py loop
- **File:** `portfolio/main.py:448-462` and `portfolio/main.py:486-497`
- **Why:** The daily digest + message throttle flush code is duplicated between the first
  `run(force_report=True)` block and the main while-loop. Both have identical try/except
  patterns for `maybe_send_daily_digest()` and `flush_and_send()`.
- **Change:** Extract a `_run_post_cycle(config)` helper that calls both. Call it from
  both locations. Also move the Alpha Vantage refresh into the helper.
- **Impact:** `main.py` only. Reduces ~30 lines of duplication to ~5.

---

## 3. Test Coverage Improvements

### TEST-8: signal_engine trap detection regression test
- **File:** `tests/test_signal_engine_core.py` (augment existing)
- **Why:** BUG-42. Verify that trap detection logs warnings instead of silently failing.
  Test: DataFrame with missing volume column, DataFrame with NaN values, normal trap detection.

### TEST-9: main.py _extract_triggered_tickers unit tests
- **File:** `tests/test_main_helpers.py` (new)
- **Why:** BUG-48 area. The regex extraction function is untested. Test: various trigger
  reason strings (consensus, moved, flipped, unknown format, empty list).

---

## 4. Refactoring

### REF-13: Extract _run_post_cycle helper
- **File:** `portfolio/main.py`
- **Why:** See ARCH-10. DRY principle.
- **Impact:** `main.py` only. Deduplicates ~30 lines.

---

## 5. Items NOT Planned (Justified)

1. **Silent `except: pass` in crash handlers** (main.py:392, 418, 515, 520) — These are
   last-resort error handlers. Logging inside them could cause infinite recursion if the
   logging system is broken. Keeping them silent is intentional defensive programming.

2. **Silent `except: pass` in Playwright cleanup** (avanza_session.py:132-145) — Standard
   cleanup pattern for browser automation. Playwright objects may be in an invalid state;
   swallowing cleanup exceptions is correct.

3. **Silent `except: pass` in SQLite best-effort writes** (outcome_tracker.py:150, 281, 353,
   363) — JSONL is the primary data store; SQLite is secondary. Documented as best-effort.

4. **Silent `except: pass` in forecast config load** (forecast.py:59) — Import-time config
   read with a safe default. Logging at import time is unreliable.

5. **Thread safety for `_prev_sentiment`** — The main loop is single-threaded. `_set_prev_sentiment`
   is only called from `generate_signal()` which runs in the main loop. No race condition.

6. **Freqtrade legacy config in config.json** — ~40 lines of unused Freqtrade config exist but
   are inert (never read by Layer 1/2). Removing them risks breaking any external tooling that
   might reference them. Low ROI cleanup.

---

## 6. Dependency/Ordering — Implementation Batches

### Batch 1: Silent exception fixes (6 files, BUG-42 through BUG-47)
**Files:** `portfolio/signal_engine.py`, `portfolio/main.py`, `portfolio/bigbet.py`,
`portfolio/iskbets.py`, `portfolio/macro_context.py`, `portfolio/sentiment.py`
**Changes:** Replace `except Exception: pass` with `logger.warning(...)` in 8 locations.
**Test impact:** None — logging changes don't affect behavior.

### Batch 2: Code cleanup + refactoring (1 file, BUG-48 + ARCH-10 + REF-13)
**Files:** `portfolio/main.py`
**Changes:** Module-level regex compile + _run_post_cycle helper extraction.
**Test impact:** New test file for _extract_triggered_tickers.

### Batch 3: Tests (2 files, TEST-8 + TEST-9)
**Files:** `tests/test_signal_engine_core.py` (augment), `tests/test_main_helpers.py` (new)
**Changes:** Add trap detection logging test + trigger ticker extraction tests.
**Prerequisites:** Batches 1-2 must complete first.
