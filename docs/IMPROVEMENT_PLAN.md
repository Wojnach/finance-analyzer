# Improvement Plan — Auto-Session 2026-04-09

Updated: 2026-04-09
Branch: improve/auto-session-2026-04-09
Status: **IN PROGRESS**

## Summary

382 ruff violations remain across the codebase, 2 actual bugs (F821 undefined name,
F811 shadowed test), and broad lint cleanup opportunities. The system is mature and
well-maintained — 182+ bugs fixed in previous sessions. This session focuses on:
fixing remaining bugs, comprehensive lint cleanup, dead code removal, and test correctness.

---

## 1. Bugs & Problems Found

### BUG-183 (P2): Dead code after return in `metals_swing_trader.py:322-324`
- **File**: `data/metals_swing_trader.py:322-324`
- **Issue**: Three lines after `return all(...)` in `_regime_confirmed()` are unreachable.
  They reference `signal_data` which is undefined in that scope (ruff F821).
- **Fix**: Remove dead code block (lines 322-324).
- **Impact**: None (unreachable). Eliminates F821 lint error.

### BUG-184 (P2): Duplicate test method shadows BUY test case
- **File**: `tests/test_signal_improvements.py:420,431`
- **Issue**: Two methods named `test_btc_leads_eth` in `TestCrossAssetSignals`. The second
  (SELL case) shadows the first (BUY case), so the BUY test never runs (ruff F811).
- **Fix**: Rename second to `test_btc_leads_eth_sell`.
- **Impact**: Restores a silently-skipped test.

---

## 2. Lint Cleanup

### Auto-fixable (63 violations, `ruff --fix`):
| Rule | Count | Description |
|------|-------|-------------|
| I001 | 28 | Unsorted imports |
| F401 | 23 | Unused imports |
| F541 | 6 | f-strings without placeholders |
| E401 | 3 | Multiple imports on one line |
| SIM114 | 2 | If-with-same-arms |
| UP017 | 2 | datetime.timezone.utc → UTC |
| UP015 | 1 | Redundant open modes |

### Manual fixes — `data/metals_loop.py` (49 violations):
| Rule | Count | Description |
|------|-------|-------------|
| SIM105 | 22 | try/except/pass → contextlib.suppress |
| F841 | 5 | Unused variables |
| F401 | 4 | Unused imports |
| I001 | 3 | Unsorted imports |
| SIM102 | 3 | Collapsible if |
| Others | 12 | E402, SIM115, E741, etc. |

### Remaining (intentional, not fixing):
- **52 E402**: Module-import-not-at-top — intentional lazy imports for startup speed
- **5 SIM115**: open-file-with-context-handler — file handles that need longer scope
- **56 SIM117**: Multiple-with-statements — cosmetic, not fixing
- **40 E741**: Ambiguous variable names — mostly in tests (`l` for list)

---

## 3. Implementation Batches

### Batch 1: Bug Fixes (2 files)
1. `data/metals_swing_trader.py` — Remove dead code (lines 322-324)
2. `tests/test_signal_improvements.py` — Rename duplicate test method

### Batch 2: Auto-fix Ruff Violations (all files)
1. Run `ruff check --fix` on portfolio/, data/, dashboard/, tests/
2. Review changes for false positives (re-exports, etc.)
3. Run tests

### Batch 3: Manual Ruff Fixes — `data/metals_loop.py` (1 file)
1. Fix 5 F841 unused variables
2. Convert SIM105 try/except/pass → contextlib.suppress
3. Fix remaining safe violations (SIM102, F401)

### Batch 4: Documentation Update
1. Update SYSTEM_OVERVIEW.md known issues section
2. Update ruff violation counts

---

## 4. Ordering & Dependencies

- Batch 1 first (bug fixes before cosmetic changes)
- Batch 2 after Batch 1 (auto-fixes are safe, broad)
- Batch 3 after Batch 2 (manual fixes on top of auto-fixes)
- Batch 4 last (reflects final state)
- Full test suite after each batch

## 5. Risk Assessment

- **Very low risk**: All changes are lint fixes, dead code removal, or test renames
- **No behavioral changes** to production trading logic
- **No new features** — pure quality improvement
- **Full test suite** after each batch verifies no regressions
