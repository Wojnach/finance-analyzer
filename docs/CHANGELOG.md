# Changelog

## 2026-03-05
- **BUG-61 through BUG-67**: Replaced 15 silent `except Exception: pass` handlers with logged warnings/debug messages across 6 modules: `autonomous.py`, `fx_rates.py`, `outcome_tracker.py`, `journal.py`, `forecast.py`, `main.py`.
- **BUG-69**: Fixed `_run_post_cycle()` in `main.py` to use module-level `DATA_DIR` constant instead of re-deriving path.
- **BUG-70**: Removed redundant `import time as _time` from `run()` in `main.py` — `time` already imported at module level.
- Added `tests/test_silent_exceptions.py` with 9 tests verifying each previously-silent handler now logs.

## 2026-03-04
- Aligned `tests/test_shared_state.py` with current LRU fallback cache eviction semantics in `portfolio/shared_state.py`.
- Verified targeted batch tests for shared-state eviction, forecast circuit reset, JSONL pruning, and signal registry isolation.
- Confirmed full-suite failures are mostly pre-existing integration/runtime-environment issues (Freqtrade strategy path, metals autonomous expectations, trigger/report timing).
- Removed transient working document `docs/SESSION_PROGRESS.md` per auto-improve workflow.
