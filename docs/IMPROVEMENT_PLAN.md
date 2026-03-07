# Improvement Plan

Updated: 2026-03-07
Branch: worktree-auto-session-2026-03-07

Previous sessions: 2026-03-05 (dashboard hardening), 2026-03-06 (CircuitBreaker, TTL cache, prune fix).

## 1) Bugs & Problems Found

### BUG-10 (P1): `digest._build_digest_message()` crashes on missing keys

- **File**: `portfolio/digest.py:62,87`
- **Issue**: Uses `e["ts"]` (direct key access) when iterating JSONL entries. If any
  entry in `invocations.jsonl` or `layer2_journal.jsonl` is missing the `"ts"` key
  (e.g., due to partial write or schema change), this raises `KeyError` and the entire
  digest fails silently (caught by outer `except Exception` in `_maybe_send_digest`).
- **Compare**: Line 99 correctly uses `e.get("ts", "2000-01-01")` for signal_log entries.
- **Fix**: Replace `e["ts"]` with `e.get("ts", "")` and skip entries with empty/invalid ts.
- **Impact**: Low risk — only affects digest generation, not trading. But digest is the
  primary monitoring tool; silent failure means no 4h updates.

### BUG-11 (P1): `digest._build_digest_message()` raw file read without error handling

- **File**: `portfolio/digest.py:107`
- **Issue**: `json.loads(AGENT_SUMMARY_FILE.read_text())` — if `agent_summary.json`
  doesn't exist or is corrupted, this crashes. Should use `load_json()` with fallback.
- **Fix**: Use `load_json(AGENT_SUMMARY_FILE, default={})` from file_utils.
- **Impact**: Digest crashes if agent_summary is stale/missing. Same silent failure
  as BUG-10 — no 4h monitoring updates.

### BUG-12 (P2): `outcome_tracker.backfill_outcomes()` loads entire signal_log into memory

- **File**: `portfolio/outcome_tracker.py:262-267`
- **Issue**: Reads all JSONL entries into a list. With ~40-100 entries/day × 19 tickers,
  after 6 months this could be 10K+ entries with nested outcomes dicts. Each entry can
  be 10-50KB. Total memory: 100MB-500MB.
- **Fix**: Process file in streaming chunks, write updates incrementally, or add a
  max-entries limit (e.g., only process last 2000 entries — older ones are fully filled).
- **Impact**: Memory pressure on the trading host. Not urgent at current data volumes
  (~2 months of data) but will become a problem.

### BUG-13 (P2): `digest._build_digest_message()` doesn't handle missing bold state

- **File**: `portfolio/digest.py:179`
- **Issue**: `json.loads(BOLD_STATE_FILE.read_text())` inside an `if BOLD_STATE_FILE.exists()`
  check. But the file could be corrupted even if it exists. No try/except around this read.
- **Fix**: Use `load_json()` with try/except, or skip bold section on error.
- **Impact**: Digest crash if bold state file is corrupted.

### BUG-14 (P3): `_get_last_digest_time()` uses bare `except Exception`

- **File**: `portfolio/digest.py:36`
- **Issue**: Catches all exceptions including `FileNotFoundError`, `PermissionError`, etc.
  Should narrow to `(json.JSONDecodeError, FileNotFoundError, OSError)`.
- **Fix**: Narrow exception types.
- **Impact**: Cosmetic — function returns 0 (safe fallback) on any error.

## 2) Architecture Improvements

### ARCH-6: Harden digest.py resilience (Batch 1)

Bundle BUG-10, BUG-11, BUG-13, BUG-14 into a single digest hardening commit.

- Replace all raw `json.loads(file.read_text())` with `load_json()` from file_utils
- Replace all `e["ts"]` with safe `.get()` + skip on invalid
- Narrow exception types in `_get_last_digest_time`
- Add tests for: missing files, corrupted JSON, missing "ts" key in JSONL entries

**Files changed**: `portfolio/digest.py`
**Impact assessment**: No behavioral change on happy path. Prevents silent digest failures.

### ARCH-7: Add streaming to outcome_tracker backfill (Batch 2)

Fix BUG-12 by limiting backfill to recent entries only.

- Add `max_entries` parameter to `backfill_outcomes()` (default 2000)
- Only process the last N entries, skip fully-filled entries early
- This is safe because older entries are already fully backfilled

**Files changed**: `portfolio/outcome_tracker.py`
**Impact assessment**: Reduces memory usage from O(all_entries) to O(max_entries).

### ARCH-8: Clean up disabled signal references (Batch 3)

Three signals are disabled (ml, funding, custom_lora) but still referenced in code paths:
- `signal_engine.py` still has import paths for ml_signal, funding_rate
- `outcome_tracker.py` still derives votes for ml, funding signals
- `tickers.py` SIGNAL_NAMES still includes disabled names

Remove dead code paths. Keep SIGNAL_NAMES entries (needed for accuracy tracking).

**Files changed**: `portfolio/signal_engine.py`, `portfolio/outcome_tracker.py`
**Impact assessment**: No behavioral change — disabled signals already return HOLD.

## 3) Useful Features

### FEAT-2: Add digest health indicator

When digest runs, include a brief "system health" line showing:
- Loop uptime (from health.py heartbeat)
- Signal failures in last 4h
- Agent completion rate (invoked vs succeeded)

This is already partially computed but not surfaced clearly.

**Files changed**: `portfolio/digest.py`
**Impact**: Better monitoring visibility without additional infrastructure.

## 4) Refactoring TODOs

### REF-1: DRY `outcome_tracker._fetch_historical_price()` and `_fetch_current_price()`

Both functions duplicate Binance spot/FAPI/yfinance dispatch logic. Could extract a
`_fetch_price(ticker, timestamp=None)` helper. Low priority — code is stable.

### REF-2: Align reflection.py and equity_curve.py trade matching

reflection.py uses simple avg-cost matching; equity_curve.py uses FIFO. Results diverge
for multiple partial entries/exits. Low priority — reflection is supplementary context.

## 5) Dependency/Ordering

### Batch 1: Digest hardening (BUG-10, BUG-11, BUG-13, BUG-14, ARCH-6)
- Files: `portfolio/digest.py`, `tests/test_digest.py`
- No dependencies on other batches

### Batch 2: Outcome tracker memory optimization (BUG-12, ARCH-7)
- Files: `portfolio/outcome_tracker.py`, `tests/test_outcome_tracker.py` (new)
- No dependencies on other batches

### Batch 3: Disabled signal cleanup (ARCH-8)
- Files: `portfolio/signal_engine.py`, `portfolio/outcome_tracker.py`
- Depends on Batch 2 (outcome_tracker.py modified in both)

### Batch 4: Digest health indicator (FEAT-2)
- Files: `portfolio/digest.py`
- Depends on Batch 1 (digest.py modified in both)

## 6) Summary

| Category | Count | Items |
|----------|-------|-------|
| Bugs | 5 | BUG-10 through BUG-14 |
| Architecture | 3 | ARCH-6 through ARCH-8 |
| Features | 1 | FEAT-2 |
| Refactoring | 2 | REF-1, REF-2 (deferred) |
| Batches | 4 | Ordered by dependency |
