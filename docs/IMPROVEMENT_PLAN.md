# Improvement Plan — Auto Session 2026-02-23

> Based on deep exploration of all ~76 Python files, 1332 passing tests.
> Previous session (Feb 22) completed: signal_utils extraction, thread-safe cache,
> DB connection reuse, cached accuracy, kline dedup, API URL centralization,
> atomic JSONL appends, encoding fixes, accuracy param optimization.

## Priority 1: Bugs & Stale Tests

### B1. Stale cooldown test assertion
**File:** `tests/test_trigger_edge_cases.py:526`
**Bug:** `assert COOLDOWN_SECONDS == 60` but actual value is `600` (changed to 10min).
**Fix:** Update assertion to `assert COOLDOWN_SECONDS == 600` and rename test.
**Impact:** Eliminates 1 of 2 failing tests.

### B2. Sentiment reversal test doesn't respect SUSTAINED_CHECKS
**File:** `tests/test_portfolio.py:534-542`
**Bug:** Test calls `check_triggers` only twice, but sentiment reversal requires
`SUSTAINED_CHECKS = 3` consecutive cycles to fire. Test was written before the
sustained-check requirement was added.
**Fix:** Call check_triggers 3+ times with consistent sentiment to satisfy sustained
check, THEN flip to trigger the reversal.
**Impact:** Eliminates the other pre-existing test failure.

### B3. Silent exception swallowing in accuracy_stats.py
**File:** `accuracy_stats.py` lines 28-29, 251-252, 256-257
**Bug:** Three `except: pass` patterns silently swallow errors. SQLite failures,
cache corruption, and cache write failures go unlogged.
**Fix:** Add `logger.debug()` or `logger.warning()` to each catch.
**Impact:** Makes debugging accuracy issues possible.

## Priority 2: Code Quality & Deduplication

### A1. Duplicate _load_json/_load_jsonl helpers (4+2 copies)
**Files:** `kelly_sizing.py:20`, `regime_alerts.py:30,34`, `risk_management.py:20`,
`weekly_digest.py:28,32`
**Bug:** Each reimplements a wrapper around `file_utils.load_json()`.
**Fix:** Replace all with direct import from `file_utils.py`. For `weekly_digest._load_jsonl`
which adds a `since` time filter, inline the filtering at the call site.
**Impact:** ~40 lines of duplicated code removed.

### A2. f-string formatting in logger calls (25+ instances)
**Files:** `http_retry.py`, `shared_state.py`, `signal_engine.py`, `main.py`,
`forecast_signal.py`, `agent_invocation.py`, `telegram_notifications.py`, `news_event.py`
**Bug:** `logger.warning(f"...")` does eager string formatting even when the log
level is filtered out. Should use `logger.warning("...", arg1, arg2)`.
**Fix:** Convert to %-style formatting in all logger calls.
**Impact:** Minor performance improvement, follows Python logging best practices.

## Priority 3: Robustness

### R1. HTTP retry without jitter
**File:** `http_retry.py:38-41, 48-51`
**Bug:** Pure exponential backoff (1s, 2s, 4s) with no randomization. Multiple
clients retrying simultaneously will all hit the server at the same moment.
**Fix:** Add `random.uniform(0, wait * 0.1)` jitter (10% of wait time).
**Impact:** Prevents thundering herd on API failures.

### R2. Architecture doc drift
**File:** `docs/architecture-plan.md`
**Bug:** Says 25 signals, 1min cooldown, MIN_VOTERS=2 for stocks. All stale.
**Fix:** Update to reflect current reality: 27 signals, 10min cooldown, MIN_VOTERS=3.
**Impact:** Keeps documentation truthful.

## Execution Order

### Batch 1: Fix failing tests (B1, B2) — 2 files
- `tests/test_trigger_edge_cases.py` — Update cooldown assertion
- `tests/test_portfolio.py` — Fix sentiment reversal test

### Batch 2: Silent exceptions + deduplication (B3, A1) — 6 files
- `accuracy_stats.py` — Add logging to silent catches
- `kelly_sizing.py` — Replace _load_json with file_utils.load_json
- `regime_alerts.py` — Replace _load_json/_load_jsonl
- `risk_management.py` — Replace _load_json
- `weekly_digest.py` — Replace _load_json/_load_jsonl

### Batch 3: Logger formatting + jitter (A2, R1) — 9 files
- `http_retry.py` — Add jitter + fix f-string loggers
- `shared_state.py` — Fix f-string logger
- `signal_engine.py` — Fix f-string logger
- `main.py` — Fix f-string loggers (18+ instances)
- `forecast_signal.py` — Fix f-string loggers
- `agent_invocation.py` — Fix f-string logger
- `telegram_notifications.py` — Fix f-string logger
- `signals/news_event.py` — Fix f-string logger

### Batch 4: Documentation (R2) — 1 file
- `docs/architecture-plan.md` — Update stale values
