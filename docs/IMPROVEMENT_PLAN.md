# Improvement Plan

Updated: 2026-03-19
Branch: improve/auto-session-2026-03-19

Previous sessions: 2026-03-05 through 2026-03-18.

## Session Plan (2026-03-19)

### Theme: Python Modernization & Final Bug Sweep

Previous sessions completed IO safety hardening (BUG-47 through BUG-74), silent exception
elimination (BUG-56 through BUG-70), and initial lint cleanup (REF-13/14). This session
addresses two remaining gaps:

1. **494 auto-fixable ruff violations** — Python 3.11+ modernization (datetime.UTC, PEP 604
   union types, PEP 585 generics, import sorting) and style cleanups (redundant open modes,
   deprecated imports, extraneous parentheses).
2. **6 real bugs** found by ruff B/SIM rules + 5 remaining silent exception handlers.

### 1) Bugs & Problems Found

#### BUG-80 (P3): Duplicate value in sentiment.py stopwords set

- **File**: `portfolio/sentiment.py:336,340`
- **Issue**: `"could"` appears twice in the `_STOPWORDS` set literal. No functional impact
  (sets deduplicate), but indicates a copy-paste error.
- **Fix**: Remove the duplicate entry.
- **Impact**: Zero functional. Code cleanliness.

#### BUG-81 (P3): Missing `raise ... from` in avanza_client.py

- **File**: `portfolio/avanza_client.py:89`
- **Issue**: `raise ImportError(...)` inside `except ImportError:` without `from err` or
  `from None`. This obscures the original traceback.
- **Fix**: Change to `raise ImportError(...) from None`.
- **Impact**: Low. Better debugging experience on import failures.

#### BUG-82 (P3): Unused imports in claude_gate.py

- **File**: `portfolio/claude_gate.py:27,31`
- **Issue**: `platform` and `timedelta` imported but never used.
- **Fix**: Remove both imports.
- **Impact**: Zero.

#### BUG-83 (P3): 5 remaining silent `except Exception: pass` handlers

- **Files**:
  - `portfolio/gpu_gate.py:40-41` — lock release failure
  - `portfolio/telegram_notifications.py:69-70` — fallback send failure
  - `portfolio/signal_engine.py:816-817` — health tracking best-effort
  - `portfolio/reporting.py:834-835` — weekly digest failure
  - `portfolio/reporting.py:849-850` — daily digest failure
- **Issue**: Exceptions swallowed without any logging. While these are all best-effort
  code paths, silent failures make debugging harder.
- **Fix**: Add `logger.debug()` or `logger.warning()` to each handler.
- **Impact**: Low. Better observability.

#### BUG-84 (P3): ADX not cached in indicator cache (was BUG-54)

- **File**: `portfolio/signal_engine.py:361`
- **Issue**: `_compute_adx(df)` is called fresh every time `apply_confidence_penalties()`
  runs. ADX computation involves EMA smoothing over the full DataFrame. While not expensive
  per-call (~1ms), it adds up across 20 tickers × 7 timeframes = 140 calls/cycle.
- **Fix**: Cache ADX result per (ticker, cycle) using `_cached()`.
- **Impact**: Low. Performance improvement, ~140ms saved per cycle.

### 2) Architecture Improvements

None planned. The codebase architecture is solid after 14 sessions of hardening.
Remaining open items (ARCH-12 signal failure tracking, ARCH-13 flat-market accuracy)
are feature work, not cleanup — they belong in dedicated feature branches.

### 3) Refactoring TODOs

#### REF-16: ruff auto-fix (494 auto-fixable violations)

Run `ruff check --fix` to auto-modernize:
- **UP017** (199): `datetime.timezone.utc` → `datetime.UTC` (Python 3.11+)
- **UP045** (149): `Optional[X]` → `X | None` (PEP 604)
- **I001** (75): unsorted imports
- **UP006** (44): `Dict`/`List`/`Tuple` → `dict`/`list`/`tuple` (PEP 585)
- **UP015** (10): redundant open modes (`open(f, "r")` → `open(f)`)
- **UP035** (8): deprecated typing imports
- **SIM114** (6): if-with-same-arms
- **UP024** (2): `OSError` alias cleanup
- **UP032** (2): unnecessary f-string
- **UP034** (1): extraneous parentheses
- **E401** (1): multiple imports on one line
- **SIM117** (1): collapsible with-statements
- **B033** (1): duplicate set value

Zero behavioral change. All are syntactic modernization.

#### REF-17: Manual ruff fixes (non-auto-fixable)

- **B904** (1): Add `from None` to re-raised ImportError
- **SIM103** (1): Simplify needless bool return
- **B007** (9): Prefix unused loop variables with `_`

### 4) What We're NOT Doing

- **Not fixing E402** (20 issues): Intentional conditional imports at module level.
- **Not fixing E741** (6 issues): Mathematical variable names (l, I, O).
- **Not fixing SIM105** (11 issues): `try/except: pass` → `contextlib.suppress()`.
  These are all best-effort cleanup handlers where the existing pattern is clearer.
- **Not fixing SIM102** (8 issues): Collapsible-if. These are intentionally split for
  readability (e.g., separate early-return checks).
- **Not fixing SIM115** (3 issues): `file_utils.py` open-without-context-manager.
  The try/except FileNotFoundError pattern requires this structure.
- **Not fixing B023** (1 issue): False positive — `_vote_str()` in analyze.py is called
  immediately within the same loop iteration, not deferred.

### 5) Dependency/Ordering

**Batch 1** (REF-16): ruff auto-fix
- Files: ~80+ portfolio modules
- Risk: Near-zero (all syntactic, tested by existing suite)
- Commit: `refactor: ruff auto-fix Python 3.11 modernization (494 fixes)`

**Batch 2** (BUG-80/81/82 + REF-17): Manual bug fixes
- Files: sentiment.py, avanza_client.py, claude_gate.py, + ~5 modules for B007/SIM103
- Risk: Near-zero
- Commit: `fix: manual ruff fixes — duplicate set value, raise-from, unused imports`

**Batch 3** (BUG-83): Silent exception logging
- Files: gpu_gate.py, telegram_notifications.py, signal_engine.py, reporting.py
- Risk: Near-zero
- Commit: `fix: add logging to 5 remaining silent exception handlers`

**Batch 4** (BUG-84): ADX caching
- Files: signal_engine.py
- Risk: Low (cache key must include ticker to avoid cross-contamination)
- Commit: `perf: cache ADX computation per ticker per cycle`
