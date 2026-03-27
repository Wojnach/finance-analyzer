# Improvement Plan — Auto-Session 2026-03-27

Updated: 2026-03-27
Branch: improve/auto-session-2026-03-27

## 1. Bugs & Problems Found

### P1 — Critical (affects accuracy or causes incorrect behavior)

#### BUG-133: Accuracy cache shared timestamp causes cross-horizon staleness
- **File**: `portfolio/accuracy_stats.py:529-548`
- **Problem**: `write_accuracy_cache()` stores all horizons in a single JSON file (`accuracy_cache.json`) but uses ONE `"time"` key for TTL checks. When `write_accuracy_cache("3h", data)` is called, it updates `cache["time"]`, making stale "1d" data appear fresh. This means after a 3h cache refresh, stale 1d data (potentially hours old) is served as if just computed.
- **Impact**: Signal weighting uses wrong accuracy data for whichever horizon was not most recently refreshed. Could cause over/under-weighting of signals by 5-15% accuracy points.
- **Fix**: Use per-horizon timestamps: `cache["time_1d"]`, `cache["time_3h"]`, etc. Check `cache.get(f"time_{horizon}", 0)` in `load_cached_accuracy()`.

#### BUG-134: Regime accuracy hardcoded to "1d" regardless of prediction horizon
- **File**: `portfolio/signal_engine.py:1118`
- **Problem**: `load_cached_regime_accuracy("1d")` is always called with "1d", even when `acc_horizon` is "3h", "4h", or "12h". The regime-conditional accuracy for a 3h prediction should use 3h outcome data, not 1d.
- **Impact**: 3h/4h/12h predictions use 1d regime accuracy, which may have different signal rankings. A signal could be 70% accurate at 3h-ranging but only 45% at 1d-ranging — using the wrong one corrupts weighting.
- **Fix**: Replace `load_cached_regime_accuracy("1d")` with `load_cached_regime_accuracy(acc_horizon)` on line 1118. Repeat for `signal_accuracy_by_regime("1d")` on line 1120.

#### BUG-135: Signal utility always evaluated at "1d" horizon
- **File**: `portfolio/signal_engine.py:1149`
- **Problem**: `signal_utility("1d")` is called regardless of the actual prediction horizon. A signal that catches big 1d moves might be noise at 3h. The utility boost (up to 1.5x accuracy multiplier) is applied based on wrong-horizon returns.
- **Impact**: 3h predictions get utility-boosted with 1d return data. Signals that are profitable at 1d but not at 3h get boosted when they shouldn't be.
- **Fix**: Replace `signal_utility("1d")` with `signal_utility(acc_horizon)`.

### P2 — Important (could cause incorrect behavior in edge cases)

#### BUG-136: Utility boost mutates accuracy_data dict in-place
- **File**: `portfolio/signal_engine.py:1157-1160`
- **Problem**: `accuracy_data[sig_name]["accuracy"] *= boost` directly mutates the accuracy value. When `accuracy_data = alltime` (line 1103, elif branch), this mutates the dict returned by `load_json()`. While not persistent across calls (load_json reads from disk), it means the dict is modified for all subsequent uses within the same `generate_signal()` call, including the weighted consensus. Later, `signal_best_horizon_accuracy` may overwrite some entries but not others, creating inconsistency.
- **Impact**: The accuracy data passed to `_weighted_consensus()` has some signals boosted by utility (up to 1.5x) while others were replaced by best-horizon data. Not deterministically reproducible.
- **Fix**: Build a new dict for the boosted data instead of mutating: `accuracy_data[sig_name] = {**accuracy_data[sig_name], "accuracy": min(accuracy_data[sig_name]["accuracy"] * boost, 0.95)}`.

#### BUG-137: SQLite DB resource leak in load_entries()
- **File**: `portfolio/accuracy_stats.py:28-35`
- **Problem**: `db = SignalDB()` opens a connection, but if `db.load_entries()` throws an exception, `db.close()` is never called (falls through to the `except` block).
- **Impact**: On error, SQLite connection stays open until garbage collection. Unlikely to cause issues on Windows (single-threaded access) but is a resource leak.
- **Fix**: Use try/finally or a context manager: `try: db = SignalDB(); ...; finally: db.close()`.

#### BUG-138: Backtester duplicates accuracy blending logic
- **File**: `portfolio/backtester.py:32-85` vs `portfolio/signal_engine.py:1077-1105`
- **Problem**: `_build_accuracy_data()` in backtester.py manually replicates the EWMA blending logic (recency divergence threshold, weight constants). If signal_engine.py is updated, the backtester silently uses stale logic.
- **Impact**: Backtest results may not reflect actual signal engine behavior, giving misleading accuracy comparisons.
- **Fix**: Extract the blending logic into a shared function in `accuracy_stats.py` and call it from both modules.

#### BUG-139: `load_json()` silently returns default on permission errors
- **File**: `portfolio/file_utils.py:36-42`
- **Problem**: `load_json()` catches all `ValueError` but does NOT catch `PermissionError` or `OSError` (only `FileNotFoundError`). On Windows, if a file is locked by another process, `path.read_text()` raises `PermissionError` which propagates as an uncaught exception. This is inconsistent: `FileNotFoundError` returns default silently, but `PermissionError` crashes. Both are transient filesystem conditions.
- **Impact**: If antivirus or another process briefly locks a data file, the caller crashes instead of gracefully degrading.
- **Fix**: Add `OSError` to the first except clause: `except (FileNotFoundError, OSError):`.

### P3 — Minor (code quality, performance, observability)

#### BUG-140: `_cached()` holds lock during cache eviction sort
- **File**: `portfolio/shared_state.py:40-52`
- **Problem**: When cache exceeds `_CACHE_MAX_SIZE` (256), the LRU fallback eviction sorts all cache entries while holding `_cache_lock`. This blocks all other threads from cache reads for the duration of the sort.
- **Impact**: Negligible — 256 entries sorts in microseconds. But the lock is also held during the expired-entry scan which iterates all entries.
- **Fix**: Accept risk. Document that eviction happens under lock.

#### BUG-141: `print_accuracy_report()` calls `load_entries()` 3× per horizon
- **File**: `portfolio/accuracy_stats.py:562-618`
- **Problem**: For each horizon, `signal_accuracy(h)`, `consensus_accuracy(h)`, and `per_ticker_accuracy(h)` each call `load_entries()`. With 7 horizons, that's up to 21 full file reads of a 68MB+ file for a single report.
- **Impact**: Report takes 30-60 seconds on a 68MB file. Only called by CLI `--accuracy` command, not in the loop.
- **Fix**: Load entries once and pass to each function. Add an `entries=None` parameter to `signal_accuracy()`, `consensus_accuracy()`, `per_ticker_accuracy()`.

#### BUG-142: `signal_best_horizon_accuracy()` inner loop is O(signals × horizons × entries × tickers)
- **File**: `portfolio/accuracy_stats.py:916-937`
- **Problem**: For each entry, for each ticker, for each horizon, iterates all SIGNAL_NAMES. This is O(E × T × H × S) where E=entries, T≈20 tickers, H=7 horizons, S=30 signals. With 150K entries and 20 tickers, that's 150K × 20 × 7 × 30 = 630M iterations.
- **Impact**: Slow computation (10-30 seconds) but cached with 1h TTL. Only runs once per hour.
- **Fix**: Restructure to single pass: iterate entries × tickers × horizons, then for each signal vote accumulate stats. Same total work but better cache locality.

---

## 2. Architecture Improvements

### ARCH-23: Extract accuracy blending into reusable function
- **File**: `portfolio/accuracy_stats.py` (new function)
- **Problem**: Signal engine and backtester independently implement the recency-divergence EWMA blend. Three constants (`_RECENCY_DIVERGENCE_THRESHOLD`, `_RECENCY_WEIGHT_NORMAL`, `_RECENCY_WEIGHT_FAST`) are duplicated.
- **Fix**: Add `blend_accuracy_data(alltime, recent, divergence_threshold=0.15, normal_weight=0.7, fast_weight=0.9)` to `accuracy_stats.py`. Both callers use it.
- **Impact**: Removes ~30 lines of duplication, prevents future divergence.

### ARCH-24: Parameterize accuracy functions with pre-loaded entries
- **File**: `portfolio/accuracy_stats.py`
- **Problem**: Functions like `signal_accuracy()`, `consensus_accuracy()`, `per_ticker_accuracy()`, `signal_utility()`, `signal_activation_rates()`, `signal_accuracy_by_regime()`, `signal_best_horizon_accuracy()` each independently call `load_entries()`. When multiple are called in sequence (e.g., `print_accuracy_report()`, or during cache refresh), the file is read multiple times.
- **Fix**: Add optional `entries=None` parameter to each function. If None, load from disk. If provided, use the pre-loaded data. This is backwards-compatible.
- **Impact**: Eliminates redundant I/O during report generation and cache refresh. No behavioral change.

---

## 3. Improvements to Implement

### Batch 1: Accuracy system correctness (3 files, P1)
**Priority**: High — fixes signal weighting bugs that affect trade decisions.

| # | Change | File | Bug |
|---|--------|------|-----|
| 1 | Per-horizon timestamps in accuracy cache | `portfolio/accuracy_stats.py` | BUG-133 |
| 2 | Use `acc_horizon` for regime accuracy | `portfolio/signal_engine.py` | BUG-134 |
| 3 | Use `acc_horizon` for signal utility | `portfolio/signal_engine.py` | BUG-135 |

**Impact**: Changes accuracy data flow. Test coverage exists for accuracy_stats (test_best_horizon.py, test_accuracy_stats.py). Need to update cache-related tests.

### Batch 2: Data integrity fixes (3 files, P2)
**Priority**: Medium — prevents data corruption and resource leaks.

| # | Change | File | Bug |
|---|--------|------|-----|
| 1 | Don't mutate accuracy_data in-place for utility boost | `portfolio/signal_engine.py` | BUG-136 |
| 2 | Fix SQLite resource leak with try/finally | `portfolio/accuracy_stats.py` | BUG-137 |
| 3 | Add OSError handling to load_json | `portfolio/file_utils.py` | BUG-139 |

**Impact**: Low risk — all are defensive improvements. file_utils change touches every load_json caller but only adds a new exception to the catch clause (no behavioral change for existing callers).

### Batch 3: Architecture — extract blending + entries param (2 files)
**Priority**: Medium — deduplication and performance.

| # | Change | File | Bug/Arch |
|---|--------|------|----------|
| 1 | Extract `blend_accuracy_data()` function | `portfolio/accuracy_stats.py` | ARCH-23 |
| 2 | Use shared blending in signal_engine | `portfolio/signal_engine.py` | ARCH-23 |
| 3 | Use shared blending in backtester | `portfolio/backtester.py` | BUG-138 |
| 4 | Add `entries=` parameter to accuracy functions | `portfolio/accuracy_stats.py` | ARCH-24 |
| 5 | Pass pre-loaded entries in print_accuracy_report | `portfolio/accuracy_stats.py` | BUG-141 |

**Impact**: `blend_accuracy_data()` is a new function. Existing callers are updated to use it. All callers are already tested.

### Batch 4: Test coverage for new changes
**Priority**: Medium — verifies all fixes.

| # | Change | File | Coverage |
|---|--------|------|----------|
| 1 | Tests for per-horizon cache timestamps | `tests/test_accuracy_cache_timestamps.py` | BUG-133 |
| 2 | Tests for blend_accuracy_data function | `tests/test_blend_accuracy.py` | ARCH-23 |
| 3 | Tests for load_json OSError handling | `tests/test_file_utils_oserror.py` | BUG-139 |

**Impact**: Test-only additions. No production code changes.

---

## 4. Deferred Items (from prior sessions + this session)

- **ARCH-17**: main.py re-exports 100+ symbols (breaking change risk)
- **ARCH-18**: metals_loop.py monolith (risks live trading)
- **ARCH-19**: No CI/CD pipeline (needs GitHub Actions + Windows runner)
- **ARCH-20**: No type checking/mypy (incremental adoption)
- **ARCH-21**: autonomous.py function decomposition (stable, low ROI)
- **ARCH-22**: agent_invocation.py class extraction (touches every caller)
- **BUG-121**: news_event.py sector mapping hardcoded (low value)
- **BUG-132**: orb_predictor.py no caching (low priority)
- **BUG-140**: `_cached()` eviction under lock (negligible impact)
- **BUG-142**: `signal_best_horizon_accuracy()` O(E×T×H×S) (cached, 1h TTL)
- **TEST-1**: gpu_gate.py zero test coverage (requires GPU mocking)
- **TEST-3**: 26 pre-existing test failures (integration, config)

---

## 5. Dependency & Ordering

```
Batch 1 (accuracy correctness) → independent, highest priority
Batch 2 (data integrity) → independent of Batch 1
Batch 3 (architecture) → depends on Batch 1 (cache timestamp change)
Batch 4 (tests) → depends on Batch 1 + 2 + 3

Run tests after each batch.
```

### Risk Summary

| Batch | Files Changed | Production Risk | Test Risk |
|-------|--------------|-----------------|-----------|
| 1 | 2 files (modify) | Medium — changes accuracy data used for weighting | Medium — cache format change |
| 2 | 3 files (modify) | Low — defensive fixes | Low — no behavioral change |
| 3 | 3 files (modify) | Low — extract + reuse | Low — same logic, shared |
| 4 | 3 files (add) | None — test files only | None — new tests |
