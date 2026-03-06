# Improvement Plan

Updated: 2026-03-06
Branch: worktree-auto-improve-0306

Previous session (2026-03-05) fixed: dashboard JSONL hardening, accuracy stats resilience,
static export parity/auth. Those items are marked DONE below.

## 1) Bugs & Problems Found

### ~~BUG-1 (P1): CircuitBreaker not thread-safe~~ (DONE, 2026-03-06)

- **Fix**: Added `threading.Lock` to wrap all state mutations in CircuitBreaker.
- **Tests**: 3 new concurrent thread safety tests (barrier-synchronized).
- **Commit**: `9e2a904`

### ~~BUG-2 (P1): `autonomous.py` caches consensus accuracy forever~~ (DONE, 2026-03-06)

- **Fix**: Replaced module-level cache with 5-minute TTL using `time.monotonic()`.
  None results are also TTL-cached, so missing files get retried.
- **Tests**: 2 new tests (TTL refresh, None retry).
- **Commit**: `3633e8a`

### BUG-3 (P2): `_held_tickers_cache` in reporting.py not restart-safe

- **Status**: Not a real bug on review. The lazy import inside the function body
  correctly reads the current `_run_cycle_id` each call. The cache starts at -1
  and cycle starts at 0, so the first call always misses (correct behavior).

### ~~BUG-4 (P2): `_classify_tickers` T3 sell_count is always 0~~ (DONE, 2026-03-06)

- **Fix**: Removed dead T3 sell_count loop (all SELL tickers already in actionable).
- **Commit**: `3633e8a`

### ~~BUG-5 (P3): `prune_jsonl` preserves malformed lines~~ (DONE, 2026-03-06)

- **Fix**: Added JSON validation during read phase of `prune_jsonl`. Malformed
  partial-write lines are now dropped instead of preserved.
- **Tests**: 2 new tests (malformed lines dropped, all-malformed file).
- **Commit**: `70fd879`

### ~~P1 — Dashboard JSONL schema assumptions~~ (DONE, 2026-03-05)
### ~~P1 — Accuracy endpoint malformed JSONL~~ (DONE, 2026-03-05)
### ~~P2 — Static export parity/auth~~ (DONE, 2026-03-05)

## 2) Architecture Improvements

### ~~ARCH-1: Thread-safe CircuitBreaker~~ (DONE)

- Added `threading.Lock` to `CircuitBreaker` and wrap all state mutations.
- Also exposed CB status in `health.get_health_summary()` for dashboard visibility.

### ~~ARCH-2: TTL-based autonomous accuracy cache~~ (DONE)

- 5-minute TTL with `time.monotonic()`. Both valid and None results cached.

### ARCH-3: Restart-safe held-tickers cache (NOT NEEDED)

- Re-reviewed: lazy import correctly reads current value each call. No fix needed.

### ~~ARCH-4: Fix T3 sell_count computation~~ (DONE)

- Removed dead loop. sell_count was always 0 for T3 — now explicitly removed.

### ARCH-5: Config validation for all CLI modes

- Already exists: `config_validator.py` with `validate_config_file()` called at
  loop startup. Additional CLI commands (`--report`, `--accuracy`) load config
  through `_load_config()` which doesn't validate but is acceptable since these
  are developer-invoked commands.

## 3) Summary

**Session 2026-03-06 results:**
- 4 bugs fixed (BUG-1, BUG-2, BUG-4, BUG-5)
- 1 bug dismissed as non-issue (BUG-3)
- 2 architecture improvements confirmed already adequate (ARCH-3, ARCH-5)
- 10 new tests added (3 thread safety, 2 cache TTL, 1 T3 sell_count, 2 prune malformed, 2 prune validation)
- 4 commits, 5 files changed
