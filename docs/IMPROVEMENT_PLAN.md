# Improvement Plan — Auto-Session 2026-03-25

Updated: 2026-03-25 (COMPLETED)
Branch: improve/auto-session-2026-03-25

## 1. Bugs & Problems Found

### P1 — Critical (affects system reliability or data correctness)

#### BUG-122: health.py reads entire 68MB signal_log.jsonl (x2)
- **Files**: `portfolio/health.py:265-266` and `portfolio/health.py:314-315`
- **Problem**: Both `check_outcome_staleness()` and `check_dead_signals()` use `f.readlines()` on the 68MB signal_log.jsonl file. This loads ~75MB into memory to check only 20-50 entries. The exact same pattern was fixed in BUG-109 (digest.py).
- **Impact**: ~150MB total memory spike per health check cycle (both functions called back-to-back in main.py). On memory-constrained systems, could trigger OOM or slow GC pauses.
- **Fix**: Replace `open()/readlines()` with `load_jsonl_tail()` from `file_utils.py`.

#### BUG-123: Untracked files break worktrees and CI
- **Files**: `portfolio/metals_ladder.py`, `portfolio/process_lock.py`, `portfolio/subprocess_utils.py`, `portfolio/notification_text.py` + 5 test files
- **Problem**: These files are imported by tracked modules (e.g., `fin_snipe.py` → `metals_ladder.py`, `fin_snipe_manager.py` → `process_lock.py`) but are not committed to git. Any git worktree or fresh clone fails with `ModuleNotFoundError`.
- **Impact**: Every worktree creation requires manual file copying. CI would fail. Current test suite fails on `test_fin_snipe_manager.py` immediately.
- **Fix**: Track and commit all required files.

### P2 — Important (could cause incorrect behavior)

#### BUG-124: fin_snipe_manager.py raw config.json read
- **File**: `portfolio/fin_snipe_manager.py:97-98`
- **Problem**: Uses `open(BASE_DIR / "config.json")` + `_json.load(fh)` instead of `load_json()` from `file_utils`. Violates the project's "Atomic I/O only" rule (CLAUDE.md rule 4).
- **Impact**: On corrupt/partial config.json, crashes with unhandled JSONDecodeError instead of graceful fallback. Race condition with config writes.
- **Fix**: Use `load_json()` or `api_utils.load_config()`.

#### BUG-125: onchain_data.py non-atomic cache write
- **File**: `portfolio/onchain_data.py:57`
- **Problem**: `CACHE_FILE.write_text(json.dumps(...))` is not atomic. If process crashes during write, the cache file is corrupt and subsequent reads fail.
- **Impact**: On-chain data cache corruption on crash → stale MVRV/SOPR data → degraded BTC analysis.
- **Fix**: Replace with `atomic_write_json(CACHE_FILE, data, ensure_ascii=False)`.

### P3 — Minor (observability, code quality)

#### BUG-126: main.py silent Telegram exception handlers
- **File**: `portfolio/main.py:573-574` and `portfolio/main.py:585-586`
- **Problem**: Two `except Exception: pass` blocks swallow Telegram send failures without logging. These are inside safeguard checks (outcome staleness, dead signals).
- **Impact**: If Telegram is down, no visibility into failed safeguard alerts.
- **Fix**: Add `logger.debug("Failed to send safeguard alert", exc_info=True)`.

#### BUG-127: crypto_scheduler.py silent exception handler
- **File**: `portfolio/crypto_scheduler.py:286-287`
- **Problem**: `except Exception: pass` swallows fundamentals cache read failure without logging.
- **Impact**: Low — just a display issue in crypto scheduler output.
- **Fix**: Add `logger.debug()`.

---

## 2. Refactoring

### REF-10: fin_evolve.py aliased imports (previously flagged, still open)
- **File**: `portfolio/fin_evolve.py:23-37`
- **Problem**: Imports from `file_utils` use underscore-prefixed aliases (`atomic_append_jsonl as _atomic_append_jsonl_single`, etc.). Legacy from when these were local fallback wrappers that have since been removed.
- **Impact**: Confusing code, appears to be private functions but are just renames.
- **Fix**: Replace 5 aliased imports with direct imports, update all 13 call sites.

### REF-20: Consolidate remaining raw JSONL reads in health.py
- **File**: `portfolio/health.py:259-267` and `portfolio/health.py:312-317`
- **Problem**: Uses raw `import json` + manual file parsing instead of `load_jsonl_tail()`.
- **Impact**: Code duplication, missing malformed-line resilience that `load_jsonl_tail()` provides.
- **Fix**: Part of BUG-122 fix.

---

## 3. Improvements to Implement

### Batch 1: Track untracked files + fix import chain (9 files)
**Priority**: Critical — blocks all worktree/CI operations.

| # | Change | File |
|---|--------|------|
| 1 | Track in git | `portfolio/metals_ladder.py` |
| 2 | Track in git | `portfolio/process_lock.py` |
| 3 | Track in git | `portfolio/subprocess_utils.py` |
| 4 | Track in git | `portfolio/notification_text.py` |
| 5 | Track in git | `tests/test_avanza_control.py` |
| 6 | Track in git | `tests/test_metals_ladder.py` |
| 7 | Track in git | `tests/test_metals_swing_trader_notifications.py` |
| 8 | Track in git | `tests/test_notification_text.py` |
| 9 | Track in git | `tests/test_silver_monitor.py` |

**Impact**: No production code changes — only adding files that already exist but aren't tracked.

### Batch 2: health.py memory optimization + I/O safety (3 files)
**Priority**: High — eliminates 150MB memory spikes per cycle.

| # | Change | File | Bug |
|---|--------|------|-----|
| 1 | Replace readlines() with load_jsonl_tail() in check_outcome_staleness() | `portfolio/health.py` | BUG-122 |
| 2 | Replace readlines() with load_jsonl_tail() in check_dead_signals() | `portfolio/health.py` | BUG-122 |
| 3 | Replace write_text() with atomic_write_json() | `portfolio/onchain_data.py` | BUG-125 |

**Impact**: health.py is called every cycle. Changes are isolated to data reading — no behavioral change. onchain_data.py change is a write-path safety improvement.

### Batch 3: Silent exception cleanup + config I/O (3 files)
**Priority**: Medium — improves observability and crash safety.

| # | Change | File | Bug |
|---|--------|------|-----|
| 1 | Use load_json() for config read in _notify_critical() | `portfolio/fin_snipe_manager.py` | BUG-124 |
| 2 | Add logger.debug to Telegram exception handlers | `portfolio/main.py` | BUG-126 |
| 3 | Add logger.debug to fundamentals cache handler | `portfolio/crypto_scheduler.py` | BUG-127 |

**Impact**: fin_snipe_manager.py touches notification path only. main.py/crypto_scheduler.py changes are logging-only additions.

### Batch 4: Import cleanup (1 file)
**Priority**: Low — code readability only.

| # | Change | File | Bug |
|---|--------|------|-----|
| 1 | Remove aliased imports, use direct names | `portfolio/fin_evolve.py` | REF-10 |

**Impact**: No behavioral change. 13 call sites renamed from `_load_json` → `load_json`, etc.

---

## 4. Deferred Items (from prior sessions, still valid)

- **ARCH-17**: main.py re-exports 100+ symbols (breaking change risk too high)
- **ARCH-18**: metals_loop.py monolith split (risks destabilizing live metals trading)
- **ARCH-19**: No CI/CD (out of scope — needs GitHub Actions + Windows runner)
- **ARCH-20**: No type checking/mypy (incremental adoption not worth session time)
- **ARCH-16**: Golddigger/elongir duplicated config loading (localized, may diverge)
- **BUG-121**: news_event.py sector mapping hardcoded (low value, ticker list stable)
- **TEST-1**: gpu_gate.py zero test coverage (requires GPU/CUDA mocking)
- **TEST-3**: 26 pre-existing test failures (integration, config, state isolation)

---

## 5. Dependency & Ordering

```
Batch 1 (track files) → required for test suite to pass
Batch 2 (health/onchain) → independent
Batch 3 (exceptions/config) → independent
Batch 4 (imports) → independent

All batches are independent after Batch 1. Can be parallelized.
```

### Risk Summary

| Batch | Files Changed | Production Risk | Test Risk |
|-------|--------------|-----------------|-----------|
| 1 | 9 files (add) | None — existing files, just tracking | Low — adds passing tests |
| 2 | 2 files (modify) | Low — isolated data reading | Low — existing tests cover behavior |
| 3 | 3 files (modify) | Low — logging + load_json swap | None — no behavioral change |
| 4 | 1 file (modify) | None — import rename only | None — no behavioral change |
