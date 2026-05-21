# Improvement Plan — Auto-Session 2026-05-21

Created: 2026-05-21
Branch: `improve/auto-session-2026-05-21`
Prior session (2026-05-20): B0 critical fix (agent_invocation globals), B1-B4 test/infra fixes.

## Exploration Summary

Explored all core modules (signal_engine, main, agent_invocation, trigger, market_timing,
autonomous), data/infra layer (file_utils, shared_state, health, portfolio_mgr, risk_management,
trade_guards, data_collector, reporting, journal, telegram_notifications), metals subsystem
(metals_loop, grid_fisher, avanza_session/orders, exit_optimizer, iskbets, fin_snipe),
dashboard (app.py), additional loops (crypto, oil, mstr), test suite (430 files), and all
existing documentation.

**Key finding**: Codebase is mature with strong fundamentals — atomic I/O, thread-safe caching,
robust error recovery. Most prior auto-sessions (2026-05-04, 2026-05-12, 2026-05-15,
2026-05-19, 2026-05-20) have already fixed the obvious concurrency bugs. What remains is:
documentation drift, a minor API safety bug, dashboard code quality, and test coverage gaps.

---

## 1. Bugs & Problems Found

### BUG-A: avanza_orders.py — Missing None guard (P2)
**File**: `portfolio/avanza_orders.py:367`
**Issue**: `place_buy_order()` / `place_sell_order()` can return `None` on Playwright errors.
Line 367 calls `result.get("orderRequestStatus", "UNKNOWN")` without checking if `result` is
None first. If None, raises `AttributeError`.
**Impact**: Caught by the broad `except Exception` on line 393, but produces a generic error
message instead of a specific diagnostic. The order is marked as "error" with `str(e)` =
`'NoneType' object has no attribute 'get'` — misleading for operators.
**Fix**: Add explicit None check before `.get()`. Set status to "error" with clear message
"API returned no response".

### BUG-B: dashboard/app.py — Raw JSONL reads in mstr_loop endpoint (P3)
**File**: `dashboard/app.py:896-918`
**Issue**: The `/api/mstr_loop` endpoint reads `mstr_loop_poll.jsonl` and `mstr_loop_trades.jsonl`
using raw `open()` + manual JSON parsing instead of `file_utils.last_jsonl_entry()` which
handles corruption, encoding errors, and seeks from end efficiently.
**Impact**: Inconsistent with all other JSONL reads in the dashboard which use `_read_jsonl()`.
On very large files, reads the entire file line-by-line to find the last entry.
**Fix**: Replace with `last_jsonl_entry()` calls.

---

## 2. Documentation Drift (High Priority)

### DOC-A: SYSTEM_OVERVIEW.md — Line counts wildly outdated
Many module line counts in Section 3 (Module Map) are wrong by 30-200%:
- `main.py`: listed 909, actual **1532**
- `agent_invocation.py`: listed 489, actual **1644**
- `trigger.py`: listed 330, actual **651**
- `market_timing.py`: listed 141, actual **342**
- `signal_engine.py`: listed ~4,280, actual **4399**
- `signal_registry.py`: listed ~300, actual **380**
- `reporting.py`: listed 962, actual **1330**
- `shared_state.py`: listed 206, actual **388**
- `data_collector.py`: listed 299, actual **344**
- `file_utils.py`: listed ~250, actual **423**
- `health.py`: listed ~340, actual **452**
- `trade_guards.py`: listed 267, actual **406**
- `accuracy_stats.py`: not listed, actual **2070**
- `risk_management.py`: listed 710, actual **988**

### DOC-B: SYSTEM_OVERVIEW.md — Signal counts wrong
- Header says "69 modules (18 active, 51 disabled)"
- Actual: **70 signals in SIGNAL_NAMES** (7 core + 63 enhanced), **18 active, 52 disabled**
- Signal file count: 64 files in `portfolio/signals/`, 20,798 total lines

### DOC-C: SYSTEM_OVERVIEW.md — Module/test counts wrong
- Says "~152 portfolio modules" — actual **283**
- Test stats reference "~5,994 tests across 242 files" — actual **430 test files**

### DOC-D: SYSTEM_OVERVIEW.md — Updated date stale
- Says "Updated: 2026-05-20" — will update to 2026-05-21

---

## 3. Test Coverage Gaps (Medium Priority)

### TEST-A: trigger.py has no dedicated unit tests
**Module**: `portfolio/trigger.py` (651 lines)
**Risk**: Core change detection logic. Controls when Layer 2 fires. Has 5 trigger sections,
tier classification, flip cooldowns, sustained debounce, ranging dampening.
**Note**: Some trigger logic is indirectly tested via integration flows, but no unit tests
for individual trigger sections, tier classification, or edge cases.
**Scope**: ~200 lines of tests covering: consensus trigger, flip detection, price move,
F&G crossing, tier classification, startup grace, flip cooldown.

---

## 4. Code Quality

### QUAL-A: IMPROVEMENT_BACKLOG.md — Missing recent findings
The backlog should document architecture issues found during exploration that are too
risky for autonomous implementation:
- ARCH-17: main.py re-exports 100+ symbols (module boundary issue)
- ARCH-18: metals_loop.py 7,880-line monolith
- ARCH-19: No CI/CD pipeline
- Signal schema validation missing (sub_signal keys vary per module)

---

## 5. Batch Execution Order

### Batch 1: Bug fixes (2 files)
- `portfolio/avanza_orders.py` — None guard
- `dashboard/app.py` — file_utils for JSONL reads

### Batch 2: SYSTEM_OVERVIEW.md update (1 file)
- Full accuracy pass on line counts, signal counts, module counts

### Batch 3: trigger.py tests (1 new file)
- `tests/test_trigger_unit.py` — unit tests for trigger sections

### Batch 4: Backlog + cleanup (1 file)
- `docs/IMPROVEMENT_BACKLOG.md` — add new findings

### Batch 5: Verify & ship
- Full test suite
- Review, merge, push

---

## Impact Assessment

| Change | Files Modified | Risk | Other Systems Affected |
|--------|---------------|------|----------------------|
| BUG-A: avanza_orders None guard | 1 | Low — additive check | None |
| BUG-B: dashboard JSONL reads | 1 | Low — uses existing utility | None |
| DOC-A/B/C/D: SYSTEM_OVERVIEW | 1 | Zero — documentation only | None |
| TEST-A: trigger tests | 1 new | Zero — tests only | None |
| QUAL-A: Backlog update | 1 | Zero — documentation only | None |

Total: 4 files modified, 1 new file. All changes low-risk or zero-risk.
