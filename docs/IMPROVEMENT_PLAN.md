# Improvement Plan — Auto-Session #7 (2026-03-04)

## Status: IN PROGRESS

## Priority: Critical Bugs > Architecture > Tests > Features > Polish

Previous sessions fixed BUG-1 through BUG-48, ARCH-1 through ARCH-10, REF-1 through REF-13.
This session continues from BUG-49 onward.

---

## 1. Bugs & Problems Found

### BUG-49: forecast.py resource leak — unclosed file handles
- **File:** `portfolio/signals/forecast.py:57, 241`
- **Severity:** HIGH (file descriptor exhaustion)
- **Issue:** `json.load(open(...))` without context manager. File objects are never explicitly
  closed. Over many invocations (every minute for each ticker), file descriptors accumulate.
  Python's GC eventually closes them, but under load this causes "too many open files" errors.
- **Fix:** Replace with `with open(...) as f: json.load(f)` or use existing `file_utils.load_json()`.
- **Impact:** `portfolio/signals/forecast.py` only. No behavior change.

### BUG-50: smart_money.py hardcoded divisor in supply/demand zone margin
- **File:** `portfolio/signals/smart_money.py:375, 385`
- **Severity:** MEDIUM (logic correctness)
- **Issue:** `margin = (z_high - z_low) * proximity_pct / 0.005` — divides by hardcoded `0.005`
  instead of the module constant `_ZONE_PROXIMITY_PCT`. When `proximity_pct` equals the default
  (0.005), this simplifies to `(z_high - z_low)`, which may be intended. But if `proximity_pct`
  is ever changed, the margin scaling breaks.
- **Fix:** Replace `/ 0.005` with `/ _ZONE_PROXIMITY_PCT` on both lines.
- **Impact:** `portfolio/signals/smart_money.py` only.

### BUG-51: digest.py crashes on corrupted trigger_state.json
- **File:** `portfolio/digest.py:42`
- **Severity:** HIGH (digest silently breaks)
- **Issue:** `_set_last_digest_time()` calls `json.loads(path.read_text())` without try/except
  when `path.exists()` is True. If `trigger_state.json` exists but is corrupted (which has
  happened before — Mar 2-3 incident), `json.loads()` raises and the digest time never gets set,
  causing duplicate digests on every cycle.
- **Fix:** Wrap in try/except, default to empty dict on failure.
- **Impact:** `portfolio/digest.py` only. Prevents crash-loop on corrupt state.

### BUG-52: avanza_orders.py silent pass on file write failures
- **File:** `portfolio/avanza_orders.py:162-163, 202-203`
- **Severity:** MEDIUM (order state lost)
- **Issue:** Two `except: pass` blocks silently swallow file write errors when saving pending
  order state. If disk is full or file is locked, orders have no persistent record.
- **Fix:** Add `logger.warning(...)` to both locations.
- **Impact:** `portfolio/avanza_orders.py` only.

### BUG-53: avanza_orders.py silent pass on order error notification
- **File:** `portfolio/avanza_orders.py:260-261`
- **Severity:** MEDIUM (double failure invisible)
- **Issue:** After an order fails, code tries to notify via Telegram. If that also fails, the
  exception is silently swallowed — user never knows order failed AND notification failed.
- **Fix:** Add `logger.warning("Order error notification failed: %s", e)`.
- **Impact:** `portfolio/avanza_orders.py` only.

### BUG-54: health.py silent pass on state load and agent silence check
- **File:** `portfolio/health.py:42, 113`
- **Severity:** MEDIUM (health monitoring blind spots)
- **Issue:** `_load_health_state()` swallows corruption silently. `check_agent_silence()` at
  line 113 swallows `OSError` — if invocations.jsonl is locked, silence detection fails.
- **Fix:** Add `logger.warning(...)` to both locations.
- **Impact:** `portfolio/health.py` only.

---

## 2. Architecture Improvements

### ARCH-11: Use file_utils.load_json() consistently in forecast.py
- **File:** `portfolio/signals/forecast.py`
- **Why:** BUG-49 fix. Use existing `file_utils.load_json()` instead of raw `json.load(open(...))`.
- **Impact:** Fixes resource leak and standardizes config loading pattern.

---

## 3. Test Coverage Improvements

### TEST-10: digest.py corrupt state handling
- **File:** `tests/test_digest_state.py` (new)
- **Why:** BUG-51. Test that `_set_last_digest_time()` handles corrupt JSON gracefully.

### TEST-11: avanza_orders.py error handling paths
- **File:** `tests/test_avanza_orders_errors.py` (new)
- **Why:** BUG-52/53. Test that write failures and notification failures are logged.

---

## 4. Items NOT Planned (Justified)

1. **forecast.py `_last_prediction_ts` dict** — Max 19 tickers, bounded by design. Skip.
2. **SQLite best-effort writes in outcome_tracker.py** — Reviewed in session #6. Skip.
3. **Playwright cleanup in avanza_session.py** — Reviewed in session #6. Skip.
4. **avanza_tracker.py empty-dict returns** — Acceptable fallback by design. Skip.

---

## 5. Dependency/Ordering — Implementation Batches

### Batch 1: Resource leak + crash fix (2 files, BUG-49 + BUG-51 + ARCH-11)
**Files:** `portfolio/signals/forecast.py`, `portfolio/digest.py`
**Changes:** Replace `json.load(open(...))` with `load_json()`. Wrap digest JSON read in try/except.
**Test impact:** TEST-10 for digest.

### Batch 2: Silent exception logging + logic fix (3 files, BUG-50 + BUG-52 + BUG-53 + BUG-54)
**Files:** `portfolio/avanza_orders.py`, `portfolio/health.py`, `portfolio/signals/smart_money.py`
**Changes:** Add `logger.warning(...)` to 4 silent exception blocks. Fix smart_money margin divisor.
**Test impact:** TEST-11 for avanza_orders.

### Batch 3: Tests (2 files, TEST-10 + TEST-11)
**Files:** `tests/test_digest_state.py` (new), `tests/test_avanza_orders_errors.py` (new)
**Prerequisites:** Batches 1-2 must complete first.
