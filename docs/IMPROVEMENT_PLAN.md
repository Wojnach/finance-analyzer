# Improvement Plan — Auto-Session 2026-03-26

Updated: 2026-03-26 (COMPLETED)
Branch: improve/auto-session-2026-03-26

## 1. Bugs & Problems Found

### P2 — Important (could cause incorrect behavior)

#### BUG-128: avanza_orders.py non-atomic offset file write
- **File**: `portfolio/avanza_orders.py` (offset_file write in `_check_telegram_confirm`)
- **Problem**: Uses `write_text()` to persist Telegram update offset. If process crashes mid-write, the offset file is corrupt or empty. Next read falls back to 0, reprocessing old updates and potentially confirming expired orders.
- **Impact**: Duplicate order confirmations or missed confirmations after crash during offset write.
- **Fix**: Use `atomic_write_json()` or atomic temp+rename pattern for offset persistence.

#### BUG-129: avanza_session.py global Playwright state not thread-safe
- **File**: `portfolio/avanza_session.py` (global `_pw_instance`, `_pw_browser`, `_pw_context`)
- **Problem**: Three module-level variables hold Playwright browser state with no lock protection. If `api_get()` is called from multiple threads (e.g., metals_loop + dashboard), browser context could be corrupted.
- **Impact**: Unlikely in current single-threaded metals_loop usage, but a latent defect if concurrency increases. Could cause silent API failures or browser crashes.
- **Fix**: Add `threading.Lock` around context access, or document single-thread constraint.

### P3 — Minor (observability, performance, code quality)

#### BUG-130: Dashboard reads files on every API request
- **File**: `dashboard/app.py`
- **Problem**: Every API endpoint reads JSON/JSONL files from disk with no caching layer. Under load (multiple browser tabs, monitoring scripts), this causes redundant I/O.
- **Impact**: Slow API responses during high query volume. Not critical since dashboard is LAN-only.
- **Fix**: Add TTL-based in-memory caching (e.g., 5-10s TTL for hot endpoints like `/api/portfolio`).

#### BUG-131: message_store.py Telegram truncation breaks Markdown
- **File**: `portfolio/message_store.py` (4096 char limit)
- **Problem**: Messages exceeding Telegram's 4096 char limit are truncated at the character boundary. This can split Markdown formatting mid-tag (e.g., `*bold text` without closing `*`), causing parse errors.
- **Impact**: Telegram sends fail with "Can't parse entities" error, message lost. Fallback sends without formatting but user misses structured data.
- **Fix**: Truncate at last complete line before limit, or strip trailing incomplete Markdown tags.

#### BUG-132: orb_predictor.py fetches 5000 candles every call
- **File**: `portfolio/orb_predictor.py` (`fetch_klines()`)
- **Problem**: Fetches 5 batches of 1000 candles from Binance FAPI on every prediction call with no caching. This is ~375 days of 15m data, re-fetched each time.
- **Impact**: Unnecessary API calls, ~2-3 second latency per call. Not breaking but wasteful.
- **Fix**: Cache klines with TTL (e.g., 1 hour for historical data, 5 min for recent).

---

## 2. Architecture Improvements

### ARCH-21: autonomous.py function decomposition
- **File**: `portfolio/autonomous.py`
- **Problem**: `_build_telegram_mode_a()` and `_build_telegram_mode_b()` are 200-300+ lines each. `_autonomous_decision_inner()` is the orchestrator at ~500 lines. Hard to read, test, and maintain.
- **Impact**: Refactoring risk is moderate (functions are stable), but maintainability suffers.
- **Action**: Deferred — stable code, not worth the risk in an autonomous session.

### ARCH-22: agent_invocation.py class extraction
- **File**: `portfolio/agent_invocation.py`
- **Problem**: Module-level globals (`_agent_proc`, `_agent_start`, `_agent_timeout`, etc.) manage process lifecycle implicitly. A class would make the state machine explicit.
- **Impact**: Readability improvement, no behavioral change.
- **Action**: Deferred — would touch every caller in main.py, risk of regression.

---

## 3. Improvements to Implement

### Batch 1: Atomic I/O fixes (2 files)
**Priority**: High — prevents data corruption on crash.

| # | Change | File | Bug |
|---|--------|------|-----|
| 1 | Use atomic write for Telegram offset file | `portfolio/avanza_orders.py` | BUG-128 |
| 2 | Add threading.Lock to Playwright global state | `portfolio/avanza_session.py` | BUG-129 |

**Impact**: avanza_orders.py change is write-path only. avanza_session.py adds lock around existing code.

### Batch 2: Dashboard caching + message safety (2 files)
**Priority**: Medium — performance and reliability.

| # | Change | File | Bug |
|---|--------|------|-----|
| 1 | Add TTL cache layer to hot API endpoints | `dashboard/app.py` | BUG-130 |
| 2 | Fix Telegram message truncation to preserve Markdown | `portfolio/message_store.py` | BUG-131 |

**Impact**: Dashboard caching is additive. Message truncation fix changes truncation boundary logic.

### Batch 3: Test coverage expansion
**Priority**: Medium — improves confidence for future changes.

| # | Change | File | Coverage |
|---|--------|------|----------|
| 1 | Add tests for avanza_orders order lifecycle | `tests/test_avanza_orders_lifecycle.py` | BUG-128 verification |
| 2 | Add tests for message_store truncation | `tests/test_message_store_truncation.py` | BUG-131 verification |
| 3 | Add tests for dashboard caching behavior | `tests/test_dashboard_caching.py` | BUG-130 verification |

**Impact**: Test-only additions. No production code changes.

---

## 4. Deferred Items (from prior sessions, still valid)

- **ARCH-17**: main.py re-exports 100+ symbols (breaking change risk too high)
- **ARCH-18**: metals_loop.py monolith split (risks destabilizing live metals trading)
- **ARCH-19**: No CI/CD (out of scope — needs GitHub Actions + Windows runner)
- **ARCH-20**: No type checking/mypy (incremental adoption not worth session time)
- **ARCH-16**: Golddigger/elongir duplicated config loading (localized, may diverge)
- **BUG-121**: news_event.py sector mapping hardcoded (low value, ticker list stable)
- **BUG-132**: orb_predictor.py no caching (low priority, not on critical path)
- **TEST-1**: gpu_gate.py zero test coverage (requires GPU/CUDA mocking)
- **TEST-3**: 26 pre-existing test failures (integration, config, state isolation)

---

## 5. Dependency & Ordering

```
Batch 1 (atomic I/O) → independent, do first for safety
Batch 2 (dashboard + messages) → independent of Batch 1
Batch 3 (tests) → depends on Batch 1 + 2 code changes

Run tests after each batch.
```

### Risk Summary

| Batch | Files Changed | Production Risk | Test Risk |
|-------|--------------|-----------------|-----------|
| 1 | 2 files (modify) | Low — atomic write swap + lock addition | Low — isolated paths |
| 2 | 2 files (modify) | Low — additive caching + truncation fix | Low — existing tests + new |
| 3 | 3 files (add) | None — test files only | None — new tests |
