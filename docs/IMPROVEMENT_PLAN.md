# Improvement Plan — Auto-Session 2026-05-26

**Branch:** `improve/auto-session-2026-05-26`
**Created:** 2026-05-26 10:00 CET
**Status:** COMPLETED

---

## 1. Bugs & Problems Found

### BUG-A: JSONL rotation ignores `max_size_mb` — golddigger_log at 136MB (HIGH) ✓
**File:** `portfolio/log_rotation.py:240-367` (rotate_jsonl)
**Impact:** `golddigger_log.jsonl` is 136MB with 293,780 entries despite a
50MB policy cap. Age-based archival works (2026-03/04 months archived), but
the size cap is never enforced. `rotate_jsonl()` reads `max_size_mb` from
the policy dict but never checks it after age pruning. Compare with
`rotate_text()` which correctly gates on size.
**Root cause:** High-frequency writers (~10K entries/day) accumulate 30 days
of data within the age window, far exceeding the size cap.
**Fix:** After age-based archival, drop oldest kept entries until under cap.
Size-pruned entries go to monthly archives (not lost). Result dict includes
`size_pruned` count.
**Commit:** `b8e1fe46`

### BUG-B: 9 high-growth JSONL files lack rotation policies (HIGH) ✓
**File:** `portfolio/log_rotation.py` ROTATION_POLICIES dict
**Fix:** Added rotation policies for all 9 files with appropriate age/size caps.
**Commit:** `b8e1fe46`

### BUG-C: `shell=True` subprocess call in subprocess_utils.py (P1) ✓
**File:** `portfolio/subprocess_utils.py:285-291`
**Fix:** Converted PowerShell invocation from `shell=True` string to list args.
**Commit:** `7c651946`

### BUG-D: Resource leak in silver_monitor.py (P1) ✓
**File:** `data/silver_monitor.py:796`
**Fix:** Wrapped raw `open()` in context manager.
**Commit:** `7c651946`

### BUG-E: BUG-97 test drift — tests pass on stale heuristic (P2) ✓
**File:** `tests/test_batch2_fixes.py`
**Fix:** BUG-97 tests updated for 2026-05-17 count-delta heuristic. Added
`tmp_telegram`/`tmp_journal` fixtures and `_count_before` baselines.
Fixes 1 pre-existing test failure.
**Commit:** `7c651946`

### BUG-F: Deprecated `datetime.utcnow()` in escalation_gate (P2) ✓
**File:** `portfolio/escalation_gate.py:160`
**Fix:** Replaced with `datetime.now(timezone.utc)`. Eliminates DeprecationWarning.
**Commit:** `fe54c891`

---

## 2. Architecture Improvements

### FEAT-A: Auto-discovery of unmanaged log files ✓
**File:** `portfolio/log_rotation.py`
**What:** `find_unmanaged_files()` scans `data/` for JSONL/log/txt files >1MB
without rotation policies. `rotate_all()` calls this every run and prints
warnings. Prevents future BUG-B scenarios.
**Breaking:** `rotate_all()` returns `dict {results, unmanaged}` instead of list.
**Commit:** `10642459`

---

## 3. Documentation Updates ✓

- `docs/SYSTEM_OVERVIEW.md`: Removed log_rotation from untested modules list.
  Now has 36 tests covering age archival, size pruning, text rotation,
  unmanaged file detection.
- **Commit:** `4fbfc640`

---

## 4. Test Results

- **10,309 passed**, 26 failed (all pre-existing), 4 skipped
- **1 pre-existing failure fixed** (`test_telegram_read_failure_yields_not_sent`)
- **8 new tests added** (4 size-cap tests + 4 unmanaged-files tests)
- Runtime: 133s parallel (8 workers)

---

## 5. Commits (chronological)

1. `263b2c27` — docs: improvement plan for 2026-05-26 auto-session
2. `b8e1fe46` — fix(log-rotation): enforce max_size_mb for JSONL files + add 9 missing policies
3. `10642459` — feat(log-rotation): auto-discover unmanaged JSONL/log files >1MB
4. `7c651946` — fix: shell=True removal, resource leak, BUG-97 test drift
5. `fe54c891` — fix: replace deprecated datetime.utcnow() in escalation_gate
6. `4fbfc640` — docs: update SYSTEM_OVERVIEW test coverage for log_rotation
