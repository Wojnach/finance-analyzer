# Improvement Plan — Auto-Session 2026-02-27

## Session Results

| Metric | Before | After |
|--------|--------|-------|
| Tests passing | 2127 | 2235 (+108) |
| Test files added | 0 | 2 |
| Bugs fixed | 0 | 2 (BUG-11 doc drift, BUG-12 silent skip) |
| Refactors | 0 | 1 (REF-4 Binance dedup) |
| Pre-existing fixes | 0 | 1 (FX fallback test) |

### Commits
1. `0a3daff` — docs: SYSTEM_OVERVIEW.md and IMPROVEMENT_PLAN.md
2. `9e43158` — fix: Batch 1 — Binance dedup, skip logging, signal count 30
3. `26fd715` — test: Batch 2 — 48 tests for data_collector.py
4. `65069fd` — test: Batch 3 — 60 tests for agent_invocation.py

---

## 1. Bugs & Problems Found

### BUG-10: Sentiment hysteresis doesn't persist neutral direction
- **File:** portfolio/signal_engine.py:392-414
- **Problem:** `_set_prev_sentiment()` is only called when sentiment is "positive" or "negative".
  When sentiment becomes "neutral", the previous direction is retained. Next cycle, if sentiment
  returns to the same direction (e.g., positive→neutral→positive), the code sees a "flip" and
  raises the threshold from 0.40 to 0.55 — even though the sentiment didn't actually reverse.
- **Impact:** False threshold elevation causing missed sentiment signals.
- **Fix:** After the voting block, always update prev_sentiment for non-neutral sentiments.
  This is already the behavior, but we should also clear prev_sentiment on sustained neutral
  to prevent stale direction from causing threshold elevation later.

### BUG-11: Architecture doc says 29 signals, code has 30
- **File:** docs/architecture-plan.md, CLAUDE.md
- **Problem:** futures_flow signal #30 was added Feb 26 but architecture doc still says 29.
  Applicable counts also need updating: crypto=27, stocks/metals=25.
- **Impact:** Documentation drift — Layer 2 reads CLAUDE.md which correctly says 27/30 but
  architecture doc is the "source of truth" and it's wrong.
- **Fix:** Update architecture-plan.md signal count and applicable counts.

### BUG-12: `collect_timeframes()` silently skips timeframes with insufficient data
- **File:** portfolio/data_collector.py:274-276
- **Problem:** When `compute_indicators(df)` returns None (insufficient data), the timeframe
  is silently skipped with `continue`. No logging, no indication to the caller.
- **Impact:** If a data source consistently returns too few rows for a timeframe, the signal
  pipeline silently degrades. Debugging data quality issues becomes harder.
- **Fix:** Add `logger.debug()` when skipping a timeframe.

### BUG-13: `_fetch_klines` Alpaca fallback doesn't check yfinance failures
- **File:** portfolio/data_collector.py:247-250
- **Problem:** When market is closed and yfinance is used as fallback, if yfinance raises an
  exception, it's caught by the outer try/except in `collect_timeframes()` line 288. But
  the circuit breaker `alpaca_cb` doesn't record a failure since we never tried Alpaca.
  Conversely, no yfinance circuit breaker exists to prevent repeated failed calls.
- **Impact:** Low — yfinance failures are rare and the error is logged. But repeated failures
  would cause unnecessary API calls without circuit breaker protection.
- **Fix:** Document as TODO. Adding a yfinance circuit breaker is low priority since failures
  are rare and the fallback itself is a backup path.

## 2. Architecture Improvements

### ARCH-8: Test coverage for data_collector.py
- **Why:** The data fetching module has zero dedicated tests. It's the foundation of the
  entire pipeline — bad data silently corrupts all downstream signals.
- **What:** Add tests/test_data_collector.py with mocked API responses for binance_klines(),
  alpaca_klines(), yfinance_klines(), _fetch_klines() dispatch, and collect_timeframes().
- **Enables:** Safe evolution of data fetching logic, regression protection.

### ARCH-9: Test coverage for agent_invocation.py
- **Why:** Layer 2 subprocess management has no tests. This is the most critical integration
  point — if invocation fails, all Layer 2 analysis stops (as seen in the 34h outage).
- **What:** Add tests/test_agent_invocation.py testing invoke_agent(), _log_trigger(),
  subprocess timeout handling, tier-specific context.
- **Enables:** Faster debugging of Layer 2 invocation failures.

### ARCH-10: Config example missing feature keys
- **Why:** Five features (perception_gate, reflection, trade_guards, risk_audit,
  confidence_penalties) are referenced in code but not documented in config.example.json.
  New users/deployments won't discover these features.
- **What:** Add all missing config keys to config.example.json with defaults and comments.
- **Enables:** Self-documenting configuration.

## 3. Refactoring TODOs

### REF-4: Deduplicate binance_klines and binance_fapi_klines
- **File:** portfolio/data_collector.py:64-135
- **Problem:** These two functions are ~90% identical. The only difference is the base URL
  and circuit breaker instance. Both could be a single function parameterized by source.
- **Impact:** Low — the duplication is harmless but increases maintenance burden.
- **Fix:** Extract shared logic into `_binance_fetch(base_url, cb, symbol, interval, limit)`.

## 4. Implementation Batches (ordered)

### Batch 1: Bug Fixes + Doc Updates
**Files:** portfolio/signal_engine.py, portfolio/data_collector.py, docs/architecture-plan.md
- BUG-10: Fix sentiment hysteresis neutral gap
- BUG-12: Add logging when timeframe is skipped
- BUG-11: Update architecture doc signal count (29→30)

### Batch 2: Test Coverage — data_collector.py
**Files:** tests/test_data_collector.py (NEW)
- ARCH-8: ~30 tests for data fetching with mocked API responses

### Batch 3: Test Coverage — agent_invocation.py
**Files:** tests/test_agent_invocation.py (NEW)
- ARCH-9: ~20 tests for Layer 2 subprocess management

### Batch 4: Refactoring + Config Docs
**Files:** portfolio/data_collector.py, config.example.json
- REF-4: Deduplicate Binance kline functions
- ARCH-10: Add missing config keys to example

## 5. Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| BUG-10 (sentiment neutral) | Very low: only affects threshold logic | Test with mock sentiments |
| BUG-12 (log on skip) | None: logging only | Visual inspection |
| BUG-11 (doc update) | None: documentation only | Review |
| ARCH-8 (data collector tests) | None: additive tests | Run full suite |
| ARCH-9 (agent invocation tests) | None: additive tests | Run full suite |
| REF-4 (Binance dedup) | Low: refactoring tested paths | Run existing + new tests |
| ARCH-10 (config example) | None: documentation only | Review |

---

# Previous Session Results

## Auto-Session 2026-02-26
| Refactors | 0 | 1 (DRY accuracy_stats) |
| Features | 0 | 1 (yfinance fallback logging) |

### Commits
1. `2ebc4a2` — fix: Batch 1 — 5 bug fixes + 1 feature
2. `e4a57d0` — test: Batch 2 — 138 tests for indicators.py + signal_engine.py
3. `9139e4e` — refactor: DRY accuracy_stats (ARCH-6)
4. `6519114` — test: Batch 4 — 68 tests for portfolio_mgr.py + trigger.py

---

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
