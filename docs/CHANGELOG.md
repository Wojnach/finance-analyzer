# Changelog

## 2026-03-11
- **NewsAPI configured**: Added API key to `config.json → newsapi_key` (free tier, 100 req/day). Enhances stock sentiment headlines in `sentiment.py` and news_event signal #26 alongside Yahoo Finance fallback.
- **Config validator updated**: Added `newsapi_key`, `alpha_vantage.api_key`, `golddigger.fred_api_key`, `bgeometrics.api_token` to `OPTIONAL_KEYS` in `config_validator.py` — warns at startup if missing.
- **API inventory documented**: Full external API integration table added to `docs/SYSTEM_OVERVIEW.md` section 6 (12 services, all configured). Avanza manual auth status documented.
- **TODO.md updated**: Alpha Vantage and NewsAPI marked as done. Avanza credential automation added as pending item.

## 2026-03-05 (autonomous improvement session)
- Hardened dashboard JSONL consumers:
  - `/api/telegrams` now ignores non-object JSONL entries instead of propagating malformed shapes.
  - `/api/decisions` now ignores non-object JSONL entries instead of assuming dict records.
- Added dashboard API test coverage for:
  - malformed JSONL resilience in `/api/telegrams` and `/api/decisions`,
  - `/api/metals-accuracy` success/missing/auth behavior.
- Improved `portfolio.accuracy_stats.load_entries()` JSONL fallback to skip malformed lines instead of failing the entire accuracy read.
- Upgraded static dashboard export tool:
  - supports token-protected dashboards (reads `dashboard_token` from `config.json`),
  - exports frontend-required routes `/api/metals-accuracy` and `/api/lora-status`.

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
