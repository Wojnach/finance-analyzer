# Improvement Plan — Auto Session 2026-02-24 (Session 2)

> Deep dive following Session 1's bug fixes. Focus: breaking import bug,
> test coverage for critical untested modules, stale data safety, voting
> consistency across signal modules.
>
> **Status: ALL ITEMS COMPLETED** — see commits on `improve/auto-session-2026-02-24`.

## Priority 1: Critical Bugs — DONE

### B1. AGENT_TIMEOUT import breaks main.py ✅
**Commit:** `c98a0a3`
Removed stale `AGENT_TIMEOUT` re-export from `main.py`. Fixed docstring "25-signal" → "27-signal".

### B2. Stale data returned indefinitely on cache errors ✅
**Commit:** `c98a0a3`
Added `_MAX_STALE_FACTOR = 5` guard to `_cached()`. Returns `None` with warning when cached data exceeds 5x TTL.

### B3. Regime detection uses 1.0% EMA gap vs signal engine's 0.5% ✅
**Commit:** `c98a0a3`
Aligned `detect_regime()` to use `>= 0.5%` (was `> 1.0%`), matching signal_engine's EMA deadband.

## Priority 2: Architecture & Safety — DONE

### A1. heikin_ashi.py duplicate majority_vote ✅
**Commit:** `5e9c659`
Replaced local `_majority_vote()` with `signal_utils.majority_vote(count_hold=True)`.

### A2. Stale data missing timestamps ✅
**Commit:** `5e9c659`
Added `stale_since` ISO timestamp to preserved data in `reporting.py`.

### A3. triggered_consensus unbounded growth ✅
**Commit:** `5e9c659`
Added pruning in `_save_state()` for tickers no longer in current signals.

## Priority 3: Test Coverage — DONE

### T1. Trigger core tests ✅
**Commit:** `07f76ff` — 64 tests in `tests/test_trigger_core.py`

### T2. Market timing DST tests ✅
**Commit:** `07f76ff` — 100 tests in `tests/test_market_timing.py`

### T3. Shared state cache tests ✅
**Commit:** `07f76ff` — 49 tests in `tests/test_shared_state.py`

## Summary

| Metric | Before | After |
|--------|--------|-------|
| Tests passing | 1333 | 1546 |
| New test files | 0 | 3 |
| Bugs fixed | 0 | 3 (1 critical) |
| Files modified | 0 | 6 production + 3 test |

---

# Improvement Plan — Telegram Message Routing & Dashboard Integration

**Session:** 2026-02-24 (telegram routing)
**Branch:** `improve/auto-session-2026-02-24-telegram`
**Status: COMPLETED**

## Goal

Disable most Telegram sending while preserving message generation. Route messages by category:
- **Always send to Telegram:** ISKBETS, BIG BET, simulated trades (Patient/Bold BUY/SELL), 4-hourly digest
- **Save only (no Telegram):** Analysis/HOLD messages, Layer 2 invocation notifications, regime alerts, FX warnings, errors

All messages saved to `data/telegram_messages.jsonl` with category metadata for dashboard viewing.

## Architecture

### Message Categories

| Category     | Source                    | Send to Telegram | Description                          |
|-------------|--------------------------|-----------------|--------------------------------------|
| `trade`     | Layer 2 agent (CLAUDE.md)| YES             | Simulated BUY/SELL executions        |
| `iskbets`   | iskbets.py               | YES             | Intraday entry/exit alerts           |
| `bigbet`    | bigbet.py                | YES             | Mean-reversion BIG BET alerts        |
| `digest`    | digest.py                | YES             | 4-hourly activity report             |
| `analysis`  | Layer 2 agent (CLAUDE.md)| NO              | HOLD analysis, market commentary     |
| `invocation`| agent_invocation.py      | NO              | "Layer 2 T2 invoked" notifications   |
| `regime`    | regime_alerts.py         | NO              | Regime shift alerts                  |
| `fx_alert`  | fx_rates.py              | NO              | FX rate staleness warnings           |
| `error`     | main.py                  | NO              | Loop crash notifications             |

### Files Modified

- `portfolio/message_store.py` (NEW) — central message routing
- `portfolio/bigbet.py` — category "bigbet"
- `portfolio/iskbets.py` — category "iskbets"
- `portfolio/agent_invocation.py` — category "invocation"
- `portfolio/regime_alerts.py` — category "regime"
- `portfolio/fx_rates.py` — category "fx_alert"
- `portfolio/main.py` — category "error"
- `portfolio/digest.py` — category "digest" + enhanced stats
- `CLAUDE.md` — trade/analysis conditional sending
- `data/layer2_invoke.py`, `layer2_action.py`, `layer2_exec.py` — updated examples
- `dashboard/app.py` — enhanced /api/telegrams with filtering
- `dashboard/static/index.html` — Messages tab with category chips
