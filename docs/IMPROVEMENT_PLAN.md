# Improvement Plan — Auto-Session 2026-05-28

**Branch:** `improve/auto-session-2026-05-28`
**Created:** 2026-05-28 10:00 CET
**Status:** IN PROGRESS

## Methodology

4 parallel exploration agents mapped the entire codebase (~65,000+ lines):
- Core orchestration (main.py, agent_invocation.py, trigger.py, market_timing.py)
- Signal engine & pipeline (signal_engine.py, signal_registry.py, accuracy_stats.py, outcome_tracker.py, 6 signal modules)
- Infrastructure (file_utils.py, health.py, data_collector.py, shared_state.py, subprocess_utils.py, circuit_breaker.py, process_lock.py, llama_server.py, risk_management.py, portfolio_mgr.py)
- Dashboard & reporting (app.py, reporting.py, journal.py, telegram_notifications.py, digest.py, metals_loop.py)

Infrastructure layer: no critical bugs. Several medium-priority bugs and systemic robustness issues found.

---

## 1. Bugs & Problems Found

### BUG-A: `digest.py` falsy-zero initial_value_sek (P1)
**File:** `portfolio/digest.py:151`
**Issue:** `p_initial = state.get("initial_value_sek") or INITIAL_CASH_SEK` — if `initial_value_sek` is `0` (valid after full liquidation), falls back to constant. Should use `is None` check.
**Impact:** PnL% calculation wrong after full liquidation.
**Fix:** `p_initial = state.get("initial_value_sek"); if p_initial is None: p_initial = INITIAL_CASH_SEK`

### BUG-B: `telegram_notifications.py` escape_markdown_v1 None crash (P2)
**File:** `portfolio/telegram_notifications.py:29`
**Issue:** `escape_markdown_v1()` doesn't guard against None/non-string input.
**Impact:** Telegram send fails on None message.
**Fix:** Add string type guard.

### BUG-C: `journal.py` unbounded levels array access (P2)
**File:** `portfolio/journal.py:100-104`
**Issue:** `levels[0], levels[1]` without bounds check.
**Impact:** Journal context build fails for entries with incomplete levels data.
**Fix:** Guard with `if len(levels) >= 2`.

### BUG-D: `llama_server.py` zombie process on startup timeout (P2)
**File:** `portfolio/llama_server.py:471`
**Issue:** `proc.kill()` without subsequent `proc.wait()`. Zombie holds VRAM.
**Fix:** Add `proc.wait(timeout=5)` after `proc.kill()`.

### BUG-E: `journal.py` prices.get() None formatting (P2)
**File:** `portfolio/journal.py:379`
**Issue:** `prices.get(t)` could return None, formatted as `${p:,.2f}` crashes.
**Fix:** Guard with `if p is not None`.

---

## 2. Silent Exception Swallowers (P1 — systemic)

110+ bare `except Exception` blocks. Previous session added logging to 10.

### Targets for this session:
- `accuracy_stats.py`: 18 handlers in hot paths — add `logger.debug(..., exc_info=True)` to those that currently swallow silently
- `reporting.py`: 15 untracked handlers — add debug logging

### Approach: Add visibility only. No control flow changes.

---

## 3. Architecture Improvements (Deferred)

- ARCH-1: signal_engine.generate_signal() 1,333-line function → split into 5 functions. Too risky for autonomous session.
- ARCH-2: agent_invocation.py 14 globals → dataclass. Requires 20+ test file updates.
- ARCH-3: reporting.py 738-line function → section builders. Too many downstream consumers.

---

## 4. Execution Batches

### Batch 1: Bug fixes (5 files, ~30 lines changed)
- `portfolio/digest.py` — BUG-A
- `portfolio/telegram_notifications.py` — BUG-B
- `portfolio/journal.py` — BUG-C, BUG-E
- `portfolio/llama_server.py` — BUG-D
Tests: Write targeted tests first, then fix.

### Batch 2: Silent exception logging (2 files, ~40 lines changed)
- `portfolio/accuracy_stats.py` — add debug logging to 8 highest-risk silent handlers
- `portfolio/reporting.py` — add debug logging to 5 untracked handlers
No test changes needed (logging-only).

### Batch 3: Documentation update (2 files)
- `docs/SYSTEM_OVERVIEW.md` — refresh signal counts, module line counts
- `docs/IMPROVEMENT_BACKLOG.md` — add deferred ARCH items
