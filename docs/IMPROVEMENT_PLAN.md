# Improvement Plan — Auto-Session 2026-05-26

**Branch:** `improve/auto-session-2026-05-26`
**Created:** 2026-05-26 10:00 CET

---

## 1. Bugs & Problems Found

### BUG-A: JSONL rotation ignores `max_size_mb` — golddigger_log at 136MB (HIGH)
**File:** `portfolio/log_rotation.py:240-367` (rotate_jsonl)
**Impact:** `golddigger_log.jsonl` is 136MB with 293,780 entries despite a
50MB policy cap. Age-based archival works (2026-03/04 months archived), but
the size cap is never enforced. `rotate_jsonl()` reads `max_size_mb` from
the policy dict but never checks it after age pruning. Compare with
`rotate_text()` which correctly gates on size.
**Root cause:** High-frequency writers (~10K entries/day) accumulate 30 days
of data within the age window, far exceeding the size cap.
**Fix:** After age-based archival, if `keep_lines` still produce a file
exceeding `max_size_mb`, drop oldest kept entries until under the cap.

### BUG-B: 9 high-growth JSONL files lack rotation policies (HIGH)
**File:** `portfolio/log_rotation.py` ROTATION_POLICIES dict
**Impact:** These files grow unbounded:

| File | Size | Est. growth |
|------|------|-------------|
| llm_probability_outcomes.jsonl | 23.9 MB | ~5 MB/week |
| metals_llm_outcomes.jsonl | 14.2 MB | ~3 MB/week |
| metals_llm_predictions.jsonl | 13.6 MB | ~3 MB/week |
| mstr_loop_poll.jsonl | 3.4 MB | ~1 MB/week |
| grid_fisher_decisions.jsonl | 2.8 MB | ~1 MB/week |
| fin_snipe_manager_log.jsonl | 2.6 MB | ~0.5 MB/week |
| crypto_value_history.jsonl | 3.0 MB | ~0.5 MB/week |
| oil_value_history.jsonl | 2.5 MB | ~0.5 MB/week |
| sentiment_ab_log.jsonl | 2.2 MB | ~0.5 MB/week |

**Fix:** Add rotation policies with appropriate age/size caps per file.

---

## 2. Architecture Improvements

None proposed this session. Known backlog items (ARCH-17 through ARCH-20
in IMPROVEMENT_BACKLOG.md) are correctly deferred — high-risk refactors
with no functional impact. This session focuses on concrete reliability fixes.

---

## 3. Useful Features

None proposed. Deferred: `/api/log-rotation` dashboard endpoint for
monitoring disk growth (existing `get_file_stats()` not yet exposed).

---

## 4. Refactoring & Cleanup

### REF-A: Test coverage for JSONL size enforcement
No tests exist for size-based JSONL pruning. Need:
- Test that `rotate_jsonl` prunes to `max_size_mb` after age archival
- Test new policy entries rotate correctly

### REF-B: Update SYSTEM_OVERVIEW.md
Add: rotation policy coverage map, disk growth observations.

---

## 5. Dependency & Ordering

### Batch 1: Fix BUG-A + BUG-B (log_rotation.py)
**Files:** `portfolio/log_rotation.py`
**Risk:** Low — additive logic after existing age rotation. Existing behavior
unchanged unless file exceeds size cap after age pruning.
**Tests first:** Write failing test for size enforcement → implement → pass.

### Batch 2: Tests for rotation changes
**Files:** `tests/test_log_rotation.py` (new or extend existing)
**Risk:** None — test-only changes.

### Batch 3: Documentation
**Files:** `docs/SYSTEM_OVERVIEW.md`
**Risk:** None.
