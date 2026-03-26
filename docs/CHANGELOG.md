# Changelog

## 2026-03-26 (autonomous improvement session)
- **BUG-128: Avanza offset file atomicity**: `avanza_orders.py` Telegram offset file now uses `atomic_write_json()` instead of `write_text()`, preventing corruption on crash. Read path handles both legacy plain-text and new JSON format for backwards compatibility.
- **BUG-129: Playwright thread safety**: `avanza_session.py` global Playwright state (`_pw_instance`, `_pw_browser`, `_pw_context`) now protected by `threading.Lock` to prevent concurrent access corruption.
- **BUG-130: Dashboard TTL cache**: Added thread-safe in-memory TTL cache to `dashboard/app.py` (5s default, 60s for config). Eliminates redundant disk I/O on concurrent API requests.
- **BUG-131: Safe Telegram truncation**: `message_store.py` now truncates at the last newline boundary before the 4096-char limit instead of at an arbitrary character position. Prevents splitting Markdown tags mid-formatting.
- **SYSTEM_OVERVIEW.md**: Updated test count (5,994 across 159 files), added fixture documentation, tracked BUG-128 through BUG-132, ARCH-21/22.
- **New tests**: 9 new tests — avanza offset format compatibility (3), message truncation safety (3), dashboard cache behavior (3). All pass.
- Theme: Crash Safety, Thread Safety, Performance Caching, Markdown Integrity.

## 2026-03-25 (autonomous improvement session)
- **BUG-122: Health module 68MB memory spike (x2)**: `check_outcome_staleness()` and `check_dead_signals()` in `health.py` both used `f.readlines()` on the 68MB signal_log.jsonl to check 20-50 entries. Replaced with `load_jsonl_tail()` — reads ~512KB instead of 68MB. Eliminates ~150MB memory spike per health cycle.
- **BUG-123: Untracked files break worktrees**: `portfolio/metals_ladder.py`, `portfolio/process_lock.py`, `portfolio/subprocess_utils.py`, `portfolio/notification_text.py` were imported by tracked modules but never committed. Any worktree or fresh clone hit `ModuleNotFoundError`. Now tracked in git along with 5 test files.
- **BUG-124: fin_snipe_manager raw config read**: `_notify_critical()` used raw `open()/json.load()` for config.json. Replaced with `load_json()` for crash-safe fallback.
- **BUG-125: onchain_data non-atomic cache write**: `_save_onchain_cache()` used `write_text()`. Replaced with `atomic_write_json()` to prevent corrupt cache on crash.
- **BUG-126: main.py silent exception handlers**: Two `except Exception: pass` in safeguard Telegram alerts. Added `logger.debug()` for visibility.
- **BUG-127: crypto_scheduler silent exception**: Fundamentals cache read failure silently swallowed. Added `logger.debug()`.
- **REF-9: Raw JSONL append consolidation**: Replaced 5 remaining raw `open("a")/f.write(json.dumps())` patterns with `atomic_append_jsonl()` in `crypto_macro_data.py`, `analyze.py`, `bigbet.py`, `iskbets.py`. Provides fsync durability.
- **REF-10: fin_evolve.py import cleanup**: Removed 5 underscore-prefixed import aliases (`_load_json`, `_atomic_write_json`, etc.) — legacy from removed fallback wrappers. Updated 13 call sites and fixed 2 test assertions.
- Also synced 5 modified portfolio files from main that were never committed (ministral/qwen3 signal/trader, signal_engine).
- Theme: Memory Optimization, I/O Safety, Git Hygiene, Observability.

## 2026-03-23 (autonomous improvement session)
- **BUG-111: Accuracy tracking corruption**: `outcome_tracker._derive_signal_vote("rsi")` used hardcoded 30/70 thresholds while `signal_engine` uses adaptive `rsi_p20`/`rsi_p80` percentiles. Accuracy backfill recorded different RSI votes than actually cast, corrupting signal accuracy tracking. Fixed to use adaptive thresholds with [15,85] clamp, matching signal_engine exactly.
- **BUG-112: Backfill memory optimization**: `backfill_outcomes()` loaded entire 68MB signal_log.jsonl (~150K entries, ~75MB parsed JSON) into memory to process only 2,000 entries. Refactored to streaming approach: count lines (binary scan), skip head without parsing, parse only tail, stream head bytes verbatim on rewrite. Memory: 75MB → 2MB.
- **BUG-113: majority_vote HOLD confidence**: When HOLD won (neither BUY nor SELL majority), `majority_vote()` returned misleading non-zero confidence with `count_hold=True`. HOLD is the absence of a signal — confidence is now always 0.0.
- **BUG-114: Forecast extraction observability**: `_extract_json_from_stdout()` had 3 fallback strategies for parsing JSON from contaminated subprocess stdout but never logged which succeeded. Added debug-level logging for each fallback path.
- **COVERAGE-2: outcome_tracker tests**: Added 85 new tests (81 in `test_outcome_tracker_core.py` + 4 streaming tests in `test_outcome_tracker_backfill.py`). Coverage for `_derive_signal_vote` (all 11 signal branches) and `log_signal_snapshot`.
- Theme: Accuracy Tracking Correctness, Memory Optimization, Signal Robustness. See `docs/IMPROVEMENT_PLAN.md` for full details.

## 2026-03-22 (autonomous improvement session)
- **BUG-107: Digest zero-division**: `digest.py` and `daily_digest.py` P&L calculations crashed when `initial_value_sek` was 0 or missing. Added `or INITIAL_CASH_SEK` fallback (same fix as BUG-103, missed in these two modules).
- **BUG-108: Alpha Vantage budget thread safety**: `_daily_budget_used` counter was read/incremented without lock protection. Wrapped in existing `_cache_lock`.
- **BUG-109: Signal log performance**: `digest.py` read entire 68MB `signal_log.jsonl` to get last 500 entries. Added `load_jsonl_tail()` to `file_utils.py` — seeks to last 512KB instead of reading entire file.
- **BUG-110: Stale import path**: `digest.py` imported `load_jsonl` from `portfolio.stats` re-export instead of canonical `portfolio.file_utils`.
- **COVERAGE-1: reporting.py tests**: Added 50 tests for `reporting.py` (1,109 lines, previously ZERO coverage). Covers `write_agent_summary`, `_write_compact_summary`, `_cross_asset_signals`, `_macro_headline`, `_portfolio_snapshot`, `write_tiered_summary`, `_get_held_tickers`.
- Theme: Digest Safety, Budget Tracking, Reporting Tests. See `docs/IMPROVEMENT_PLAN.md` for full details.

## 2026-03-19 (autonomous improvement session)
- **REF-16: Python 3.11 modernization**: ruff auto-fix applied 1,910 fixes across 268 files. Key changes: `datetime.timezone.utc` → `datetime.UTC` (199), `Optional[X]` → `X | None` (149), unsorted imports (75), `Dict`/`List`/`Tuple` → builtins (44), redundant open modes (10), deprecated typing imports (8), duplicate set value (1). Zero behavioral change.
- **REF-17: Manual ruff fixes**: 28 fixes across 20 files. `raise ImportError(...)` → `raise ... from None` (B904), 17 unused loop variables prefixed with `_` (B007), 2 needless bool returns simplified (SIM103).
- **BUG-81**: `avanza_client.py` `raise ImportError` now chains with `from None` for clean tracebacks.
- **BUG-83: Silent exception logging**: Added `logger.debug()` to 5 remaining `except Exception: pass` handlers in gpu_gate.py, telegram_notifications.py, signal_engine.py, reporting.py (x2).
- **BUG-84: ADX caching**: `_compute_adx()` now cached per DataFrame identity (`id(df)`), eliminating ~140 redundant computations per loop cycle. Cache auto-clears on overflow (200 entries max).
- Theme: Python Modernization & Final Bug Sweep. See `docs/IMPROVEMENT_PLAN.md` for full details.

## 2026-03-18 (autonomous improvement session)
- **REF-13: ruff lint cleanup**: Auto-fixed 112 violations (94 unused imports, 15 empty f-strings, 2 reimports) across 59 files. Manually fixed 3 Python 3.11 f-string backslash compatibility issues in `autonomous.py` and 1 unused import in `risk_management.py`.
- **REF-14 + BUG-75/76/77: Dead variable removal**: Removed 15 unused variable assignments across 13 modules: `signal_engine.py`, `trigger.py`, `telegram_poller.py`, `smart_money.py`, `autonomous.py`, `alpha_vantage.py`, `avanza_session.py`, `bigbet.py`, `daily_digest.py`, `equity_curve.py`, `http_retry.py`, `portfolio_validator.py`.
- **BUG-71/73: Config IO hardening**: Replaced raw `json.load(open(...))` in golddigger and elongir config loading with `load_json()` from `file_utils`. Corrupt config now raises `ValueError` instead of cryptic `JSONDecodeError`.
- **BUG-72: Golddigger Telegram routing**: Replaced direct `requests.post()` Telegram call with `send_or_store()` from `message_store`. Gains JSONL message logging, Markdown escaping, and 4096 char handling.
- **BUG-74: Golddigger data cache IO**: Replaced local `_load_json_safe()` body with `load_json()` from `file_utils`.
- **BUG-79: Silent exception logging**: Added `logger.debug()` to `avanza_tracker.py`'s silent import exception handler.
- Theme: Lint Cleanup & Subsystem IO Hardening. See `docs/IMPROVEMENT_PLAN.md` for full details.

## 2026-03-17
- **Model/runtime hardening session**: landed the `feat/model-upgrades` work on `main`, moving both local trading LLMs onto the native CUDA llama.cpp path and tightening the Windows loop launcher.
- **Qwen3 upgrade**: `qwen3_trader.py` now runs through `llama-completion` on CUDA 13.1, gained batch-mode support for multi-ticker runs, and exposes explicit native asset validation via `load_model()` so missing GGUF/binary paths fail clearly.
- **Ministral-3 upgrade**: `ministral_trader.py` now prefers native `llama-completion` inference for Ministral-3-8B and falls back cleanly to the legacy Ministral-8B path when native arch/load errors occur.
- **GPU gate**: added exclusive GPU/VRAM coordination with VRAM usage logging across all four GPU-backed models (Ministral, Qwen3, Chronos, Kronos), then tightened the wait timeout from 60s to 15s after measuring real lock hold times.
- **Loop crash fix**: `scripts/win/pf-loop.bat` now sets `PYTHONPATH=Q:\finance-analyzer` before launching the loop to prevent `ModuleNotFoundError` crash loops in detached Windows contexts, and `scripts/restart_loop.py` mirrors the same safeguard for manual restarts.

## 2026-03-14 (autonomous improvement session)
- **IO safety sweep complete**: Replaced all 37+ raw `json.loads(path.read_text())` calls across 23 portfolio modules with `load_json()` from `file_utils` — eliminates TOCTOU race conditions and crash-on-corrupt-file.
- **REF-8**: Added `atomic_write_jsonl()` helper to `file_utils.py` for safe full-file JSONL rewrites.
- **BUG-48**: Replaced 3 non-atomic writes: `prophecy.py` (→ `atomic_write_json`), `signal_history.py` and `forecast_accuracy.py` (→ `atomic_write_jsonl`).
- **BUG-49**: Replaced manual JSONL parse loops with `load_jsonl()` in `analyze.py`, `signal_history.py`, `focus_analysis.py`, `equity_curve.py`.
- **TEST-11**: Added `tests/test_io_safety_sweep.py` (34 tests) — static analysis scan verifying no raw file reads remain, plus functional tests for `atomic_write_jsonl` and `load_json` edge cases.
- Modules touched: accuracy_stats, alpha_vantage, analyze, autonomous, avanza_client, avanza_orders, avanza_session, avanza_tracker, bigbet, daily_digest, equity_curve, focus_analysis, forecast_accuracy, forecast_signal, iskbets, journal, local_llm_report, main, onchain_data, perception_gate, prophecy, signal_history, telegram_notifications, signals/claude_fundamental.

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
