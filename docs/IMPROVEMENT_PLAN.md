# Improvement Plan — Auto-Session 2026-02-26

## 1. Bugs & Problems Found

### BUG-5: Unclosed file handle in `_crash_alert()`
- **File:** portfolio/main.py:346
- **Problem:** `json.load(open(config_path, encoding="utf-8"))` creates an fd that is never
  closed. Should use `with` statement.
- **Impact:** File descriptor leak on every crash. Not frequent, but a real resource leak.
- **Fix:** Replace with `Path.read_text()` + `json.loads()`.

### BUG-6: `save_accuracy_snapshot()` uses raw write instead of `atomic_append_jsonl()`
- **File:** portfolio/accuracy_stats.py:426-428
- **Problem:** Uses `f.write()` without fsync. All other JSONL appends in the codebase use
  `atomic_append_jsonl()` which includes flush+fsync for durability. This was missed in
  the Feb 25 session's BUG-1 fix.
- **Impact:** Accuracy snapshot data could be lost on crash. Low frequency (daily snapshot).
- **Fix:** Replace with `atomic_append_jsonl()`.

### BUG-7: `_portfolio_snapshot()` computes total_sek from cash only, ignores holdings
- **File:** portfolio/reporting.py:450-455
- **Problem:** Tier 1/2 context files show `total_sek = cash_sek` and `pnl_pct` calculated
  from `cash - initial`. When holdings exist, this is wildly inaccurate. Example: Patient
  holds MU worth ~8K SEK, but snapshot shows 425K as total (should be ~433K).
- **Impact:** Layer 2 receives incorrect portfolio values for Tier 1/2 invocations.
- **Fix:** Load prices from agent_summary.json and compute `portfolio_value()` properly,
  or at minimum include holdings count as context.

### BUG-8: Division by zero possible in `indicators.py:78` for `atr_pct`
- **File:** portfolio/indicators.py:78
- **Problem:** `atr_pct = atr14 / close.iloc[-1] * 100` — if the last close price is 0
  (corrupted data, delisted stock, test edge case), this raises ZeroDivisionError.
- **Impact:** Would crash the entire signal pipeline for that ticker. Unlikely with real data
  but testable edge case.
- **Fix:** Add guard: `atr_pct = (atr14 / close.iloc[-1] * 100) if close.iloc[-1] != 0 else 0.0`

### BUG-9: Redundant exception catch in reporting.py
- **File:** portfolio/reporting.py:143
- **Problem:** `except (ImportError, Exception)` is equivalent to `except Exception` since
  ImportError is a subclass of Exception. Not a bug per se, but misleading.
- **Fix:** Simplify to `except Exception`.

## 2. Architecture Improvements

### ARCH-4: Test coverage for indicators.py (CRITICAL)
- **Why:** Zero tests for the module computing RSI, MACD, EMA, BB, ATR, regime detection.
  These calculations underpin all 29 signals. A regression here breaks everything silently.
- **What:** Add tests/test_indicators.py with ~40 tests covering compute_indicators(),
  detect_regime(), technical_signal(), edge cases.
- **Enables:** Safe refactoring of indicator logic, regression protection.

### ARCH-5: Test coverage for signal_engine.py (CRITICAL)
- **Why:** Zero tests for the 29-signal voting system, weighted consensus, confidence
  penalties. This is the most complex module in the codebase.
- **What:** Add tests/test_signal_engine_core.py with ~30 tests covering generate_signal(),
  _weighted_consensus(), apply_confidence_penalties(), core signal gate, MIN_VOTERS.
- **Enables:** Safe evolution of consensus algorithm.

### ARCH-6: DRY up accuracy_stats.py duplicate functions
- **Why:** `signal_accuracy()` (lines 53-87) and `signal_accuracy_recent()` (lines 90-133)
  are 90% identical. The only difference is a time cutoff filter.
- **What:** Merge into single function with optional `since` datetime parameter.
- **Enables:** Reduced maintenance burden, single point of logic for accuracy calculation.

### ARCH-7: Test coverage for portfolio_mgr.py and trigger.py
- **Why:** Both are critical modules with zero dedicated tests.
- **What:** Add basic unit tests for load_state/save_state/portfolio_value and
  check_triggers/classify_tier.
- **Enables:** Regression protection for portfolio state management and trigger logic.

## 3. Useful Features

### FEAT-3: Log message when yfinance fallback is activated
- **File:** portfolio/data_collector.py:247-249
- **Why:** Silent Alpaca→yfinance switch when market closed makes debugging data quality
  issues difficult. A log.info() message would help trace data source changes.
- **Impact:** Logging only, zero code risk.

## 4. Refactoring TODOs

### REF-3: Simplify exception handling in reporting.py
- **File:** portfolio/reporting.py:143
- **What:** Replace `except (ImportError, Exception)` with `except Exception`.
- **Why:** Clearer intent, same behavior.

## 5. Implementation Batches (ordered)

### Batch 1: Critical Bug Fixes
**Files:** portfolio/main.py, portfolio/accuracy_stats.py, portfolio/indicators.py,
portfolio/reporting.py, portfolio/data_collector.py
- BUG-5: Fix unclosed file in _crash_alert()
- BUG-6: Use atomic_append_jsonl() in save_accuracy_snapshot()
- BUG-7: Fix _portfolio_snapshot() to include holdings context
- BUG-8: Add zero-division guard in indicators.py
- BUG-9 / REF-3: Simplify exception in reporting.py
- FEAT-3: Add yfinance fallback log message

### Batch 2: Test Coverage for indicators.py + signal_engine.py
**Files:** tests/test_indicators.py (NEW), tests/test_signal_engine_core.py (NEW)
- ARCH-4: Add ~40 indicator tests
- ARCH-5: Add ~30 signal engine tests

### Batch 3: Code Quality — DRY accuracy_stats.py
**Files:** portfolio/accuracy_stats.py
- ARCH-6: Merge signal_accuracy() and signal_accuracy_recent()

### Batch 4: Test Coverage for portfolio_mgr.py + trigger.py
**Files:** tests/test_portfolio_mgr.py (NEW), tests/test_trigger_full.py (NEW)
- ARCH-7: Add basic tests for portfolio state management and trigger logic

## Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| BUG-5 (crash_alert fd) | Very low: error path only | Code review |
| BUG-6 (accuracy fsync) | Very low: additive | Test snapshot append |
| BUG-7 (snapshot pnl) | Low: reporting only | Verify T1/T2 output |
| BUG-8 (atr zero div) | Very low: guard only | Add test with zero price |
| BUG-9 (except cleanup) | None: same behavior | Trivial |
| ARCH-4 (indicator tests) | None: additive tests | Run full suite |
| ARCH-5 (signal tests) | None: additive tests | Run full suite |
| ARCH-6 (DRY accuracy) | Low: refactor | Run accuracy tests |
| ARCH-7 (trigger/portfolio tests) | None: additive tests | Run full suite |
| FEAT-3 (log message) | None: logging only | Visual inspection |

---

# Improvement Plan — Auto-Session 2026-02-25

> Previous session results preserved below. See Session 2026-02-26 for current work.

## Previous Session Results (Feb 25)

| Metric | Before | After |
|--------|--------|-------|
| Tests passing | 1546 | 1594 |
| Bugs fixed | 0 | 4 |
| Files modified | 0 | 11 production + 1 test |
| New files | 0 | 3 (pyproject.toml, SYSTEM_OVERVIEW.md, progress.json) |

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

### JSONL Format

```json
{"ts": "ISO-8601", "text": "message", "category": "trade", "sent": true}
```

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
