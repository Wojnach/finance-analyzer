# Improvement Plan — Auto-Session 2026-05-11

## Date: 2026-05-11 (Sunday)
## Branch: improve/auto-session-2026-05-11

## 1. Bugs & Problems Found

### CRITICAL — Silent Failure Path

| # | Issue | File:Line | Impact |
|---|-------|-----------|--------|
| C1 | `status="incomplete"` (exit 0, no journal) sends NO Telegram alert — operator sees nothing after "invoked" notification | `agent_invocation.py:1272` | Root cause of 43 unresolved critical errors about Layer 2 silent failures |
| C2 | `_journal_ts_before` captured AFTER subprocess spawned — race where fast agent writes before baseline is read | `agent_invocation.py:871 vs 856` | False "incomplete" detection (low probability but trivial fix) |
| C3 | `reporting.py` macro/market_health/earnings submodules not covered by `_track_module_outcome` — failures silently swallowed | `reporting.py:248-270` | Submodule failures don't surface in `critical_errors.jsonl` |
| C4 | `process_lock.py:_lock_file` returns without locking when neither msvcrt nor fcntl available — caller thinks lock is held | `process_lock.py:60` | Silent no-op mutual exclusion on exotic platforms |

### HIGH — Data Integrity & Code Quality

| # | Issue | File:Line | Impact |
|---|-------|-----------|--------|
| H1 | `signal_db.py:insert_snapshot` no explicit `conn.rollback()` on `ticker_signals` INSERT failure — orphaned snapshot rows inflate accuracy denominators | `signal_db.py` | Analytically incorrect accuracy stats |
| H2 | `config_validator.py` uses raw `json.load` instead of `file_utils.load_json` — race with concurrent config write | `config_validator.py:59` | Startup crash on concurrent write |
| H3 | `config_validator.py` doesn't validate Binance keys — missing key causes silent data failure at runtime | `config_validator.py` | BTC/ETH data silently missing |
| H4 | `health.py:check_staleness` — `datetime.fromisoformat(hb)` crashes on corrupt `health_state.json` | `health.py:152` | Dashboard `/api/health` crash |
| H5 | `metals_loop.py:_load_json_state` uses raw `json.load(open(...))` violating CLAUDE.md rule 4 | `metals_loop.py:559` | Partial read on concurrent write |
| H6 | `_CORE_SIGNALS` in signal_registry.py is dead code — no signal registers as "core" type | `signal_registry.py:14` | Confusing dead API |
| H7 | `subprocess_utils.py:kill_orphaned_by_cmdline` uses deprecated WMIC — removed in some Win11 builds | `subprocess_utils.py:213` | Orphan detection fails silently |

### HIGH — Signal System

| # | Issue | File:Line | Impact |
|---|-------|-----------|--------|
| S1 | Disabled core signals (macd, sentiment, claude_fundamental) not force-HOLD'd at generation time — utility boost can circumvent accuracy gate | `signal_engine.py:2725` | Disabled signals can still influence consensus |
| S2 | Dead `oscillator_trend` correlation group — oscillators always HOLD, group is permanent no-op | `signal_engine.py:1356` | Misleading config, stale meta-cluster comment |
| S3 | `fibonacci` in `_SHADOW_SAFE_SIGNALS` wastes ~50ms/cycle computing a signal confirmed dead at 43.6% (17K samples) | `signal_engine.py:339` | Unnecessary CPU cost |

### MEDIUM — Performance & Robustness

| # | Issue | File:Line | Impact |
|---|-------|-----------|--------|
| M1 | `signal_db.load_entries()` O(n²) — individual queries per snapshot, degrades over months | `signal_db.py` | Cycle budget blocker after ~2 months |
| M2 | `kelly_sizing.py` min_samples=5 for per-ticker accuracy — too few for meaningful Kelly sizing | `kelly_sizing.py` | Volatile sizing input |
| M3 | CF Access JWT not cryptographically verified — presence-only check | `dashboard/auth.py:134` | Auth bypass if port becomes internet-reachable |

## 2. Implementation Batches

### Batch 1: Silent Failure Alerting (2 files)
**Goal:** Fix the #1 operational issue — Layer 2 silent failures go undetected.

1. `portfolio/agent_invocation.py`:
   - Add Telegram alert on `status="incomplete"` (matching "failed" alert pattern)
   - Move `_journal_ts_before` capture to BEFORE `subprocess.Popen` call
2. `portfolio/reporting.py`:
   - Add `_track_module_outcome` calls to macro_context, market_health, earnings_calendar exception handlers

### Batch 2: Data Integrity Fixes (3 files)
**Goal:** Prevent data corruption paths.

1. `portfolio/signal_db.py`:
   - Add explicit `conn.rollback()` in except handler for `insert_snapshot`
   - Wrap ticker_signals INSERT in try/except with proper rollback
2. `portfolio/health.py`:
   - Guard `datetime.fromisoformat(hb)` with try/except ValueError
3. `portfolio/process_lock.py`:
   - Raise RuntimeError when no locking mechanism available (match main.py pattern)

### Batch 3: Config & Convention Fixes (3 files)
**Goal:** Fix CLAUDE.md rule violations and startup validation gaps.

1. `portfolio/config_validator.py`:
   - Use `file_utils.load_json` instead of raw `json.load`
   - Add Binance API key validation to `REQUIRED_KEYS`
2. `data/metals_loop.py`:
   - Replace `_load_json_state` raw `json.load` with `file_utils.load_json`
3. `portfolio/signal_registry.py`:
   - Remove dead `_CORE_SIGNALS` dict and all references
   - Simplify `get_signal_names()` to return enhanced-only

### Batch 4: Infrastructure Hardening (2 files)
**Goal:** Fix platform-specific issues.

1. `portfolio/subprocess_utils.py`:
   - Replace WMIC-based orphan detection with PowerShell `Get-CimInstance`
2. `portfolio/signal_db.py`:
   - Optimize `load_entries()` to use JOINs instead of per-snapshot queries

## 3. Impact Assessment

| Batch | Risk | Testing Impact | Live System Impact |
|-------|------|----------------|-------------------|
| 1 | LOW — adds alerting, doesn't change data flow | Existing agent_invocation tests cover the path | Telegram gets alerts that were previously silent |
| 2 | LOW — adds guards, doesn't change happy path | Need tests for new rollback + fromisoformat guard | More resilient to corrupt data |
| 3 | LOW — uses existing atomic I/O instead of raw | Existing config_validator tests need update | Startup validation catches more issues |
| 4 | MEDIUM — changes query patterns and orphan detection | Need tests for JOIN-based queries | Performance improvement for signal_db |
