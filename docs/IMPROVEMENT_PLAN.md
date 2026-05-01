# Improvement Plan — Auto-Session 2026-05-01

## Methodology

5 parallel exploration agents + manual code review across 152 portfolio modules,
46 signal modules, 10 bot modules, dashboard, and 242 test files. Agent claims
verified against actual code before inclusion.

## 1. Bugs & Problems Found

### P1 (Critical — production impact)

**BUG-243: `llama_server.py` — `open()` without encoding on 5 call sites**
- Lines 149, 164, 187, 199, 395: `open(path)` without `encoding="utf-8"`.
- On Windows, uses system default encoding (cp1252), which can corrupt non-ASCII
  PID file content or crash on UTF-8 log paths.
- **Fix:** Add `encoding="utf-8"` to all 5 `open()` calls.

**BUG-244: `signal_decay_alert.py:35` — `open()` without encoding**
- `with open(accuracy_cache_path) as f:` misses encoding.
- **Fix:** Add `encoding="utf-8"`.

### P2 (Important — correctness/maintainability)

**BUG-245 through BUG-248:** 458 auto-fixable ruff violations across portfolio/ and data/*.py.
- F541 (80): f-string without placeholders
- F401 (45): unused imports
- UP015 (15): redundant open modes
- UP017 (44): datetime.timezone.utc → datetime.UTC
- I001 (183): unsorted imports
- E401 (82): multiple imports on one line
- UP032/W292 (misc): f-string simplification, missing newline

All are style/syntax with zero behavior change.

## 2. Architecture Improvements

None attempted this session. The codebase is architecturally solid after 70+
prior improvement sessions. The metals_loop.py monolith (7,667 lines) is the
main tech debt target but too risky for autonomous refactoring.

## 3. Useful Features

None proposed. Recent crypto/MSTR/oil subsystems still in DRY_RUN validation.

## 4. Batch Plan

### Batch 1: Production code lint + encoding fixes (portfolio/ + data/*.py)
- Ruff auto-fix for F401, F541, I001, E401, UP015, UP017, UP032, W292
- Manual encoding fixes for BUG-243, BUG-244
- Verify no side-effect imports broken
- Run full test suite

### Batch 2: Test file lint (tests/*.py)
- Same ruff auto-fix on test files
- Run full test suite

### Batch 3: Documentation update
- Update docs/SYSTEM_OVERVIEW.md with current module/signal counts
- Write session progress
