# System Overview

Updated: 2026-03-05
Branch: improve/auto-session-2026-03-05

## Session Delta (This Branch)

Implemented improvements in this branch:

1. Dashboard endpoint hardening for malformed/non-object JSONL lines in `/api/telegrams` and `/api/decisions`.
2. Accuracy log ingestion hardening in `portfolio/accuracy_stats.py` JSONL fallback (skip malformed lines).
3. Static dashboard export parity/auth updates in `dashboard/export_static.py`:
   - token-aware export requests when `dashboard_token` is configured,
   - inclusion of `/api/metals-accuracy` and `/api/lora-status` in exported static data.

## 1) Architecture Summary

This repository is a two-layer autonomous trading system:

1. Layer 1 (`portfolio/main.py`) runs a continuous loop, fetches market data, computes signals, detects meaningful changes, writes summary/context files, and decides whether Layer 2 should run.
2. Layer 2 (Claude subprocess via `portfolio/agent_invocation.py`) consumes Layer 1 output, makes strategy decisions, writes journals/portfolio updates, and sends notifications.

A separate Flask dashboard (`dashboard/app.py`) serves current state and historical telemetry from `data/`.

## 2) Entry Points and Runtime Surfaces

### Trading runtime

- Primary continuous entry: `portfolio/main.py --loop`
- One-shot/reporting and maintenance entrypoints: `portfolio/main.py` flags (`--report`, `--accuracy`, `--check-outcomes`, `--forecast-accuracy`, `--retrain`, etc.)
- Launcher scripts: `scripts/win/pf-loop.bat`, `scripts/win/pf-agent.bat`, `start-loop.bat`

### Dashboard runtime

- Flask app: `dashboard/app.py` on port `5055`
- Static export tooling: `dashboard/export_static.py`
- External sync tooling: `scripts/sync_dashboard.py`

### Auxiliary runtime

- Metals subsystem in `data/metals_*.py` + `data/silver_monitor.py`
- LoRA training pipeline in `training/lora/`

## 3) Module Responsibilities

### Orchestration and control

- `portfolio/main.py`: lifecycle, scheduling, crash backoff, health heartbeat, trigger routing, module orchestration.
- `portfolio/market_timing.py`: market-hour/off-hour scheduling and agent invocation window logic.
- `portfolio/agent_invocation.py`: Layer 2 subprocess lifecycle, timeout handling, invocation logging.
- `portfolio/trigger.py`: change detection (consensus transitions, sustained flips, price threshold, F&G threshold crossing, sustained sentiment reversal, post-trade reassessment).

### Data collection and signal generation

- `portfolio/data_collector.py`: Binance/Alpaca/yfinance collection and timeframe assembly.
- `portfolio/indicators.py`: indicator calculation primitives.
- `portfolio/signal_engine.py`: core and enhanced signal aggregation, consensus/weighted consensus.
- `portfolio/signals/*.py`: enhanced signal modules.
- `portfolio/accuracy_stats.py`, `portfolio/outcome_tracker.py`, `portfolio/signal_db.py`: signal/outcome logging and accuracy analytics.

### Portfolio, risk, and reporting

- `portfolio/portfolio_mgr.py`: portfolio state persistence and value calculation.
- `portfolio/risk_management.py`, `portfolio/trade_guards.py`, `portfolio/equity_curve.py`, `portfolio/monte_carlo*.py`: risk/equity analytics.
- `portfolio/reporting.py`: full/compact/tiered summary file generation for Layer 2 and dashboard.
- `portfolio/journal.py`, `portfolio/journal_index.py`: Layer 2 memory context generation.

### Shared infrastructure

- `portfolio/file_utils.py`: safe JSON/JSONL load and atomic write/append helpers.
- `portfolio/http_retry.py`: resilient HTTP retry policy.
- `portfolio/shared_state.py`: in-process cache and rate-limiter shared state.
- `portfolio/config_validator.py`: startup config validation.
- `portfolio/health.py`, `portfolio/message_store.py`, `portfolio/telegram_notifications.py`: health and messaging.

### Dashboard

- `dashboard/app.py`: authenticated API over `data/` files.
- `dashboard/static/index.html`: client UI consuming API and static fallback data.
- `dashboard/export_static.py`: endpoint snapshots to `dashboard/static/api-data/`.

## 4) End-to-End Data Flow

1. `main.loop()` selects active instruments based on market timing.
2. `run()` collects OHLCV/timeframes per symbol, computes indicators, generates action/confidence + extra signal metadata.
3. Trigger state is evaluated in `trigger.check_triggers(...)` against persisted baseline.
4. On trigger/forced run:
   - `reporting.write_agent_summary(...)` writes full summary.
   - Tier classification/context is generated (`classify_tier`, `write_tiered_summary`).
   - Signal snapshot/outcome tracking writes to JSONL/SQLite paths.
   - `agent_invocation.invoke_agent(...)` optionally launches Layer 2.
5. Health/risk/post-cycle jobs run (digest, throttled messaging, AV refresh, JSONL pruning).
6. Dashboard endpoints read `data/` JSON/JSONL files and expose a read API.

## 5) External Dependencies

- Market and macro data: Binance spot/futures, Alpaca, yfinance, Frankfurter, Alpha Vantage, (plus optional sentiment/on-chain sources).
- Local/LLM model stack: ministral/trader inference scripts and forecast components.
- Notification and agent orchestration: Telegram Bot API, Claude CLI subprocess.
- Python stack: Flask, pandas, numpy, scikit-learn, joblib (+ optional training dependencies).

## 6) Configuration Model

Primary runtime config is `config.json` (template: `config.example.json`). Key domains include:

- `telegram.*`
- `alpaca.*`
- optional feature domains (`layer2`, `iskbets`, dashboard token, etc.)

`portfolio/config_validator.py` enforces required keys at loop startup. Authentication for dashboard APIs is conditional on `dashboard_token`.

## 7) Observed Discrepancies vs Existing Docs

1. `docs/architecture-plan.md` still contains outdated assumptions (instrument counts/history, cooldown wording, and module inventory drift vs current code).
2. Runtime trigger behavior has no periodic cooldown trigger in `trigger.py`, but several docs still describe cooldown/check-in semantics.
3. Documentation frequently describes legacy ticker counts and old signal totals that do not align with current symbols/signal module set.
4. Existing overview docs understate dashboard/api surface changes and metals subsystem coupling into the same `data/` namespace.

## 8) Concrete Risk Areas Found During Exploration

1. Dashboard endpoints `/api/telegrams` and `/api/decisions` assume each JSONL entry is a dict; valid non-dict JSON lines can cause 500s.
2. `accuracy_stats.load_entries()` JSONL fallback has no per-line decode guard; one malformed line can break `/api/accuracy`.
3. `dashboard/export_static.py` does not include all frontend-used endpoints and can fail when dashboard token auth is enabled.
4. `main.py` validates config in loop mode, but non-loop command paths can bypass strict startup validation.
5. Trigger trade-check path silently swallows malformed portfolio files, reducing observability of state corruption.

## 9) Test Surface Snapshot

- Strong coverage exists for trigger logic and dashboard routes.
- Targeted gaps remain around:
  - endpoint robustness against non-dict JSONL lines,
  - `/api/metals-accuracy` route behavior,
  - static export script behavior under dashboard auth and endpoint parity expectations.

These findings inform the improvement plan in `docs/IMPROVEMENT_PLAN.md`.
