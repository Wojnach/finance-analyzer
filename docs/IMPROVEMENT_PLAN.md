# Improvement Plan

Updated: 2026-03-18
Branch: improve/auto-session-2026-03-18

Previous sessions: 2026-03-05 through 2026-03-17.

## Session Plan (2026-03-18)

### Theme: Lint Cleanup & Subsystem IO Hardening

Previous sessions completed silent exception elimination (BUG-56 to BUG-70) and IO safety
sweeps across the core 23 portfolio modules. This session addresses two remaining gaps:

1. **229 ruff lint violations** across `portfolio/` — 94 unused imports, 15 unused variables,
   15 empty f-strings, and more. These are auto-fixable in bulk plus a handful of manual fixes.
2. **Golddigger & elongir subsystems** merged recently but bypass the IO safety patterns
   established in the March sessions (raw `json.load(open(...))`, direct Telegram API calls,
   duplicated config loading).

### 1) Bugs & Problems Found

#### BUG-71 (P2): Golddigger runner uses raw `json.load(open(...))` for config

- **File**: `portfolio/golddigger/runner.py:59-61`
- **Issue**: Uses `with open(config_path) as f: return json.load(f)` instead of
  `load_json()` from file_utils. If config.json is corrupt or partially written, this
  crashes instead of returning a default.
- **Fix**: Use `load_json()` from `portfolio.file_utils`, raising explicitly if None.
- **Impact**: Low. Consistency with the rest of the codebase.

#### BUG-72 (P2): Golddigger sends Telegram directly instead of via message_store

- **File**: `portfolio/golddigger/runner.py:67-79`
- **Issue**: `_send_telegram()` calls `requests.post()` to the Telegram API directly.
  This bypasses `message_store.send_or_store()` which provides: (a) JSONL message logging,
  (b) Markdown escaping, (c) 4096 char limit handling. Elongir correctly uses `send_or_store`.
- **Fix**: Replace direct API call with `from portfolio.message_store import send_or_store`.
- **Impact**: Medium. Golddigger notifications not logged to dashboard.

#### BUG-73 (P2): Elongir runner uses raw `json.load(open(...))` for config

- **File**: `portfolio/elongir/runner.py:59-60`
- **Issue**: Same as BUG-71 but in the elongir subsystem.
- **Fix**: Same pattern.
- **Impact**: Low.

#### BUG-74 (P2): Golddigger data_provider uses raw `json.load(open(...))`

- **File**: `portfolio/golddigger/data_provider.py:376-377`
- **Issue**: Raw file read for cached data.
- **Fix**: Use `load_json()`.
- **Impact**: Low.

#### BUG-75 (P3): Dead variables in signal_engine.py confidence penalties

- **File**: `portfolio/signal_engine.py:408-409`
- **Issue**: `buy_count` and `sell_count` assigned but never used.
- **Fix**: Remove the assignments.
- **Impact**: Zero.

#### BUG-76 (P3): Dead variable in trigger.py

- **File**: `portfolio/trigger.py:69`
- **Issue**: `last_trigger` assigned but never used.
- **Fix**: Remove the assignment.
- **Impact**: Zero.

#### BUG-77 (P3): Dead variable and reimport in telegram_poller.py

- **File**: `portfolio/telegram_poller.py:113,136`
- **Issue**: `text_lower` unused. `json` reimported at line 136.
- **Fix**: Remove both.
- **Impact**: Zero.

#### BUG-78 (P3): claude_gate.py silent exception without logging

- **File**: `portfolio/claude_gate.py:68`
- **Issue**: `except Exception:` returns True without logging.
- **Fix**: Add `as e` and `logger.debug(...)`.
- **Impact**: Low.

#### BUG-79 (P3): avanza_tracker.py silent exception without logging

- **File**: `portfolio/avanza_tracker.py:55`
- **Issue**: `except Exception:` returns {} without logging.
- **Fix**: Add `as e` and `logger.debug(...)`.
- **Impact**: Low.

### 2) Architecture Improvements

#### ARCH-16: Golddigger/elongir duplicated config loading (DEFERRED)

- Both have nearly identical `_load_config()` functions.
- Skip for this session — duplication is localized, subsystems may diverge.

### 3) Refactoring TODOs

#### REF-13: ruff --fix auto-cleanup (111 auto-fixable issues)

- Run `ruff check --fix` to auto-remove 94 unused imports, 15 empty f-strings, 2 reimports.
- Zero behavioral change.

#### REF-14: Manual dead variable removal (15 F841 violations)

- Remove unused variable assignments across ~8 modules.
- Must verify RHS has no side effects before removing.

#### REF-15: Golddigger Telegram via message_store

- Replace `_send_telegram()` with `send_or_store()` matching elongir's pattern.

### 4) Dependency/Ordering

**Batch 1** (REF-13): ruff auto-fix
- Files: ~40 portfolio modules
- Risk: Near-zero
- Commit: `refactor: ruff auto-fix unused imports and f-strings`

**Batch 2** (REF-14 + BUG-75/76/77): Manual dead variable removal
- Files: signal_engine.py, trigger.py, telegram_poller.py, smart_money.py, portfolio_validator.py
- Risk: Near-zero
- Commit: `fix: remove dead variable assignments`

**Batch 3** (BUG-71/72/73/74 + REF-15 + BUG-78/79): Subsystem IO hardening
- Files: golddigger/runner.py, golddigger/data_provider.py, elongir/runner.py, claude_gate.py, avanza_tracker.py
- Risk: Low
- Commit: `fix: harden golddigger/elongir IO + add missing exception logging`

### 5) What We're NOT Doing

- **Not fixing E501 line-too-long** (73 issues): Ignored in ruff config.
- **Not fixing E402 module-import-not-at-top** (20 issues): Intentional conditional imports.
- **Not fixing E741 ambiguous-variable-name** (6 issues): Mathematical variables.
- **Not refactoring config loading duplication** (ARCH-16): Localized, may diverge.
- **Not adding new features**: Focus is purely on cleanup and hardening.
