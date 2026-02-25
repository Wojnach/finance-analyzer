# Improvement Plan — Auto-Session 2026-02-25

## 1. Bugs & Problems Found

### BUG-1: atomic_append_jsonl missing fsync (data corruption risk)
- **File:** portfolio/file_utils.py:66-76
- **Problem:** f.write(line) without f.flush() + os.fsync(). If OS crashes mid-write,
  JSONL file may have partial line. All JSONL consumers handle parse errors gracefully
  via continue, but partial writes still corrupt data integrity.
- **Impact:** Low probability but high severity: corrupted signal_log.jsonl loses accuracy data.
- **Fix:** Add flush+fsync after write.
- **Affected:** signal_log.jsonl, layer2_journal.jsonl, invocations.jsonl, telegram_messages.jsonl

### BUG-2: Sentiment state write races with trigger state
- **File:** portfolio/signal_engine.py (_set_prev_sentiment) and portfolio/trigger.py (_save_state)
- **Problem:** Both write to data/trigger_state.json independently via atomic_write_json.
  One write can overwrite the other's changes.
- **Impact:** Sentiment state may be lost, causing false triggers or missed flips.
- **Fix:** Move prev_sentiment out of trigger_state.json into signal_engine's own state file.

### BUG-3: Agent log file descriptor leak on timeout
- **File:** portfolio/agent_invocation.py
- **Problem:** When agent times out, log file handle may not be properly closed in all paths.
- **Impact:** File descriptor leak over many invocations.
- **Fix:** Use context manager for log file handle.

### BUG-4: Stale tickers accumulate in agent_summary.json
- **File:** portfolio/reporting.py (stale data preservation logic)
- **Problem:** Off-hours stock data preserved with stale=True but never pruned.
- **Impact:** Gradually increases file size and Layer 2 context consumption.
- **Fix:** Prune stale entries older than 24 hours.

## 2. Architecture Improvements

### ARCH-1: Consolidate trigger_state writes (resolves BUG-2)
- **Why:** Three different writers to the same JSON file creates race conditions.
- **What:** Extract sentiment persistence into data/sentiment_state.json.
  Keep trigger_state.json for trigger-specific state only.
- **Enables:** Clean separation of concerns, eliminates BUG-2.
- **Impact:** signal_engine.py, trigger.py: state path changes.

### ARCH-2: Add pyproject.toml for Python packaging and tooling
- **Why:** No linter, formatter, or standardized build commands. requirements.txt stale.
- **What:** Create pyproject.toml with dependencies, dev deps (pytest, ruff), and tool config.
- **Enables:** Automated code quality, CI/CD readiness, reproducible environments.
- **Impact:** Additive only, no code changes needed.

### ARCH-3: Expand conftest.py with shared test fixtures
- **Why:** 45 test files reimplement helpers (~500 lines of duplication).
- **What:** Add shared fixtures: sample_indicators, sample_ohlcv_df, sample_config, tmp_data_dir.
- **Enables:** Faster test writing, reduced maintenance, consistent test data.
- **Impact:** Test files only.

## 3. Useful Features

### FEAT-1: Update architecture-plan.md to reflect actual state (29 signals)
- **Why:** Doc says 27 signals, actual is 29. Signal counts and file layout need updating.
- **Impact:** Documentation only, zero code risk.

### FEAT-2: Add ruff linting configuration
- **Why:** No automated code quality checking exists.
- **What:** Add ruff config to pyproject.toml, run initial pass.
- **Impact:** Config only, optional fixes.

## 4. Refactoring TODOs

### REF-1: Remove disabled signals from accuracy tracking
- **Files:** portfolio/signal_engine.py, portfolio/accuracy_stats.py
- **What:** ML, Funding Rate, Custom LoRA disabled but still in accuracy stats.
- **Why:** Confuses Layer 2 reading reports.

### REF-2: Clean up stale requirements.txt
- **File:** requirements.txt
- **What:** References Freqtrade and other outdated deps. Replace with actual.

## 5. Implementation Batches (ordered)

### Batch 1: Foundation fixes (file I/O, state management)
**Files:** portfolio/file_utils.py, portfolio/signal_engine.py
- BUG-1: Add fsync to atomic_append_jsonl
- ARCH-1 + BUG-2: Extract sentiment persistence from trigger_state.json

### Batch 2: Reporting and invocation fixes
**Files:** portfolio/reporting.py, portfolio/agent_invocation.py
- BUG-3: Fix agent log file descriptor with context manager
- BUG-4: Add stale ticker pruning in reporting.py

### Batch 3: Documentation and tooling
**Files:** docs/architecture-plan.md, pyproject.toml, tests/conftest.py, requirements.txt
- FEAT-1: Update architecture doc
- ARCH-2: Add pyproject.toml
- ARCH-3: Expand conftest.py
- REF-2: Update requirements.txt

### Batch 4: Code quality and cleanup
**Files:** pyproject.toml, portfolio/accuracy_stats.py, portfolio/signal_engine.py
- FEAT-2: Add ruff config
- REF-1: Clean up disabled signal references

## Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| BUG-1 (fsync) | Very low: additive | Test that write still works |
| ARCH-1 (sentiment state) | Medium: changes read/write paths | Test sentiment hysteresis |
| BUG-3 (log fd) | Low: cleanup only | Test agent timeout path |
| BUG-4 (stale pruning) | Low: additive logic | Verify threshold |
| FEAT-1 (doc update) | None: documentation | Review accuracy |
| ARCH-2 (pyproject.toml) | None: new file | Verify pytest runs |
| ARCH-3 (conftest.py) | Low: additive fixtures | Run test suite |
| FEAT-2 (ruff) | None: config only | No auto-fix |
| REF-1 (disabled signals) | Low: signals already HOLD | Test accuracy reporting |

## 6. Results

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
