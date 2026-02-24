# Improvement Plan — Auto Session 2026-02-24

> Based on deep exploration of all ~76 Python files, 47 test files.
> Previous sessions completed: signal_utils extraction, thread-safe cache,
> DB connection reuse, cached accuracy, kline dedup, API URL centralization,
> atomic JSONL appends, encoding fixes, accuracy param optimization,
> f-string logger cleanup (13 modules), HTTP jitter, silent exception logging,
> helper deduplication, test fixes, architecture doc update.

## Priority 1: Bugs

### B1. Dashboard heatmap missing news_event and econ_calendar signals
**File:** `dashboard/app.py` lines 261-266
**Bug:** `enhanced_signals` list has 14 entries but system has 16 enhanced signals. Missing: `news_event`, `econ_calendar`.
**Impact:** Heatmap grid shows incomplete data for 2 of the newest signals.
**Fix:** Add `"news_event"` and `"econ_calendar"` to the `enhanced_signals` list.

### B2. Agent tier-specific timeout is computed but never used
**File:** `agent_invocation.py` line 160
**Bug:** `AGENT_TIMEOUT_DYNAMIC = timeout` is assigned but the timeout check (line 91) always reads the global `AGENT_TIMEOUT = 900`. T1 agents (120s timeout) run for up to 900s before being killed.
**Impact:** Quick-check agents monopolize resources 7.5x longer than intended.
**Fix:** Store tier timeout in module-level `_agent_timeout` alongside `_agent_start` and use it in the kill check.

### B3. FX rate hardcoded fallback is stale
**File:** `fx_rates.py` line 46
**Bug:** Fallback rate `10.50` is outdated — actual USD/SEK is ~10.8-11.0.
**Impact:** Portfolio valuation off by ~3-5% if API is down and no cached rate exists.
**Fix:** Update to `10.85` (closer to current rate).

### B4. Stale comments referencing wrong signal count and cooldown
**File:** `signal_engine.py` line 20, `trigger.py` line 9
**Bug:** Comments say "25-signal" (should be 27) and "1 min cooldown" (should be 10 min).
**Impact:** Developer confusion; misleading documentation.
**Fix:** Update comments to match code.

### B5. f-string logger calls in agent_invocation.py
**File:** `agent_invocation.py` lines 109, 162-164
**Bug:** Prior session converted all f-string loggers to %-style across 13 modules but missed these 2 in agent_invocation.py.
**Impact:** Inconsistency with project convention; string formatting happens even when log level is disabled.
**Fix:** Convert to %-style logging.

## Priority 2: Architecture Improvements

### A1. Telegram message truncation guard
**File:** `telegram_notifications.py`
**Why:** Telegram API rejects messages over 4096 characters with HTTP 400. Currently no enforcement — long messages silently fail or trigger the Markdown-fallback retry path.
**Fix:** Add truncation before sending, with a warning log if truncated.
**Impact assessment:** Low risk. Only affects edge-case long messages.

## Priority 3: Features

*None proposed this session. The system is mature and well-functioning.*

## Priority 4: Refactoring TODOs

*Covered by B2 fix (remove unused `AGENT_TIMEOUT_DYNAMIC` variable).*

## Execution Order

### Batch 1 (5 files — bug fixes, low risk, no cross-dependencies)

| File | Change | Risk |
|------|--------|------|
| `dashboard/app.py` | B1: Add missing signal names to heatmap | None — additive only |
| `agent_invocation.py` | B2: Use tier timeout; B5: fix f-string loggers | Low — affects agent kill timing |
| `fx_rates.py` | B3: Update fallback rate | None — only affects fallback path |
| `signal_engine.py` | B4: Fix stale comment | None — comment only |
| `trigger.py` | B4: Fix stale comment | None — comment only |

### Batch 2 (1 file — defensive improvement)

| File | Change | Risk |
|------|--------|------|
| `telegram_notifications.py` | A1: Add message length truncation | Very low — only truncates >4096 chars |

**Total files modified:** 6
**Estimated risk:** Low. All changes are additive, defensive, or comment-only. No logic flow changes to the core signal engine or portfolio management.
