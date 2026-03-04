# Improvement Plan — Auto-Session #8 (2026-03-04)

## Status: IN PROGRESS

## Priority: Critical Bugs > Architecture > Tests > Features > Polish

Previous sessions fixed BUG-1 through BUG-54, ARCH-1 through ARCH-11, REF-1 through REF-13.
This session continues from BUG-55 onward.

---

## 1. Bugs & Problems Found

### BUG-55: shared_state.py cache eviction fails when all entries are fresh
- **File:** `portfolio/shared_state.py:38-41`
- **Severity:** MEDIUM (memory growth under load)
- **Issue:** Cache eviction only removes entries older than 1 hour. If all 256+ entries are < 1h old
  (common during high-activity market hours), eviction does nothing and cache grows unbounded.
  With ~19 tickers × ~15 cache keys each = 285 potential entries, this threshold is easily exceeded.
- **Fix:** Add LRU fallback: after clearing stale entries, if still over limit, evict oldest 25%.
- **Impact:** `portfolio/shared_state.py` only. Pure improvement, no behavior change for callers.

### BUG-56: Forecast circuit breaker never auto-resets on success
- **File:** `portfolio/signals/forecast.py:131-155`
- **Severity:** MEDIUM (delays recovery from transient GPU failures)
- **Issue:** `_trip_kronos()` sets a 300s circuit breaker timeout. But `_log_health()` at line 158
  doesn't reset the breaker on successful execution. If Kronos fails once (GPU OOM), recovers at
  150s, the signal stays disabled until 300s. This wastes ~150s of available signal data.
- **Fix:** In `_log_health()`, reset the relevant breaker on `success=True`.
- **Impact:** `portfolio/signals/forecast.py` only. Faster recovery after transient failures.

### BUG-57: agent_invocation.py silent pass on config load (line 94)
- **File:** `portfolio/agent_invocation.py:94-95`
- **Severity:** LOW (silent failure on config read)
- **Issue:** `except Exception: pass` when loading config to check `layer2.enabled`. If config.json
  is corrupt, the error is silently swallowed and Layer 2 defaults to enabled — which may be the
  wrong behavior if the user explicitly disabled it and the config got corrupted.
- **Fix:** Add `logger.warning("Failed to load config for layer2 check: %s", e)`.
- **Impact:** `portfolio/agent_invocation.py` only. Visibility improvement.

### BUG-58: agent_invocation.py silent pass on notification send (line 201-202)
- **File:** `portfolio/agent_invocation.py:201-202`
- **Severity:** LOW (invocation notification lost silently)
- **Issue:** `except Exception: pass` when sending Telegram notification about Layer 2 invocation.
  If Telegram is down, the notification is lost without any log entry.
- **Fix:** Add `logger.debug("invocation notification failed: %s", e)`.
- **Impact:** `portfolio/agent_invocation.py` only.

### BUG-59: JSONL files grow unbounded (invocations, journal, telegram_messages)
- **Files:** `data/invocations.jsonl`, `data/layer2_journal.jsonl`, `data/telegram_messages.jsonl`
- **Severity:** MEDIUM (disk exhaustion over months of operation)
- **Issue:** These JSONL files are append-only with no rotation. At ~50 triggers/day × 365 days =
  18K entries/year for invocations alone. Layer 2 journal grows faster. No pruning mechanism exists.
- **Fix:** Add `prune_jsonl(path, max_entries)` to `file_utils.py`. Call from post-cycle or digest.
- **Impact:** `portfolio/file_utils.py`, `portfolio/main.py`. Adds rotation; no data loss (keeps
  most recent N entries). N=5000 default keeps ~3 months of history.

### BUG-60: bigbet.py silent pass swallows state load failure
- **File:** `portfolio/bigbet.py:41-42`
- **Severity:** LOW (bigbet state lost silently)
- **Issue:** `except Exception: pass` when loading bigbet gate state. If the state file is corrupt,
  the error is invisible and bigbet cooldowns are silently reset.
- **Fix:** Add `logger.warning("bigbet state load failed: %s", e)`.
- **Impact:** `portfolio/bigbet.py` only.

---

## 2. Architecture Improvements

### ARCH-12: LRU cache eviction in shared_state.py
- **File:** `portfolio/shared_state.py`
- **Why:** BUG-55 fix. Replace fragile age-based eviction with proper LRU fallback.
- **Impact:** Pure improvement, no API change. All callers work identically.

### ARCH-13: JSONL file rotation utility
- **File:** `portfolio/file_utils.py`
- **Why:** BUG-59 fix. Reusable `prune_jsonl()` function for all JSONL files.
- **Impact:** Adds new function. Called from post-cycle housekeeping.

---

## 3. Test Improvements

### TEST-12: Cache eviction edge cases
- **File:** `tests/test_shared_state_cache.py` (new)
- **Why:** BUG-55. Test that eviction works when cache is full of fresh entries.

### TEST-13: Forecast circuit breaker auto-reset
- **File:** `tests/test_forecast_circuit_reset.py` (new)
- **Why:** BUG-56. Test that successful forecast execution resets the circuit breaker.

### TEST-14: JSONL pruning
- **File:** `tests/test_file_utils_prune.py` (new)
- **Why:** BUG-59. Test that pruning keeps the most recent N entries and handles edge cases.

### TEST-15: Signal registry test isolation for xdist
- **File:** `tests/test_signal_registry.py` (modify)
- **Why:** Tests directly mutate global `_ENHANCED_SIGNALS` dict. Under `pytest -n auto`,
  concurrent test processes share state and can race. Fix: use monkeypatch autouse fixture.

---

## 4. Items NOT Planned (Justified)

1. **Credentials in config.json** — Security concern noted. Requires manual migration to env vars
   or secret manager. Too risky to automate — could break running loop. Marked TODO: MANUAL REVIEW.
2. **Hardcoded Windows paths in data/metals_*.py** — These are user scripts in data/, not the main
   portfolio package. Out of scope for this session.
3. **Type hints** — Valuable but enormous scope. Not justified for a single session.
4. **Forecast accuracy threshold (55%)** — Changing this affects live signal behavior. Requires
   backtesting to validate. Not safe to change autonomously.
5. **smart_money.py BUG-50 from session #7** — Re-examined: the `proximity_pct / _ZONE_PROXIMITY_PCT`
   ratio is intentional normalization, not a hardcoded divisor bug. When `proximity_pct` equals the
   default, it simplifies to `(z_high - z_low)`. When it changes, the margin scales proportionally.
   The previous plan was incorrect — this is correct behavior.

---

## 5. Dependency/Ordering — Implementation Batches

### Batch 1: Cache eviction fix + tests (1 file, BUG-55 + ARCH-12 + TEST-12)
**Files:** `portfolio/shared_state.py`, `tests/test_shared_state_cache.py` (new)
**Changes:** Add LRU fallback eviction. Write tests for full-cache and stale-cache scenarios.
**Test impact:** New test file only. No existing tests affected.

### Batch 2: Forecast circuit breaker auto-reset + tests (1 file, BUG-56 + TEST-13)
**Files:** `portfolio/signals/forecast.py`, `tests/test_forecast_circuit_reset.py` (new)
**Changes:** Reset breaker on successful `_log_health()`. Write tests for reset behavior.
**Test impact:** New test file. Existing forecast tests should still pass.

### Batch 3: JSONL pruning utility + integration + tests (2 files, BUG-59 + ARCH-13 + TEST-14)
**Files:** `portfolio/file_utils.py`, `portfolio/main.py`, `tests/test_file_utils_prune.py` (new)
**Changes:** Add `prune_jsonl()` to file_utils. Call from `_run_post_cycle()` in main.py.
**Test impact:** New test file. No existing tests affected.

### Batch 4: Silent exception logging + test isolation (3 files, BUG-57 + BUG-58 + BUG-60 + TEST-15)
**Files:** `portfolio/agent_invocation.py`, `portfolio/bigbet.py`, `tests/test_signal_registry.py`
**Changes:** Add logger.warning to 3 silent exception handlers. Fix signal registry test isolation.
**Test impact:** Modify existing test file for xdist safety.
