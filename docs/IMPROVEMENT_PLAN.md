# Improvement Plan — Auto-Session 2026-05-12

## Date: 2026-05-12 (Tuesday)
## Branch: improve/auto-session-2026-05-12

## Exploration Summary

4 parallel agents explored orchestration, signals, portfolio/risk, and infrastructure.
Direct analysis covered: signal_db, config_validator, auth, file_utils, grid_fisher.

**Codebase health**: Solid. 244 bugs fixed historically, thread-safe caching, atomic I/O,
proper auth, no security vulnerabilities. Previous auto-session (2026-05-11) addressed
many items. Several agent "P0" findings were false positives on manual verification.

## 1. Bugs & Problems Found

### HIGH — Race Condition

| # | Issue | File:Line | Impact |
|---|-------|-----------|--------|
| B1 | Agent spawn sets `_agent_proc` BEFORE `_agent_start`/`_agent_timeout` — watchdog can see new process with stale start time and kill it instantly | `agent_invocation.py:858-871` | Rare but real: freshly spawned agent killed by watchdog |

### HIGH — Platform Deprecation

| # | Issue | File:Line | Impact |
|---|-------|-----------|--------|
| B2 | `kill_orphaned_by_cmdline()` uses WMIC which is removed in some Win11 builds | `subprocess_utils.py:216` | Orphan detection fails silently on newer Windows |

### MEDIUM — Performance

| # | Issue | File:Line | Impact |
|---|-------|-----------|--------|
| B3 | `signal_db.load_entries()` runs per-snapshot SELECT for ticker_signals + outcomes — O(n²) | `signal_db.py:186-222` | Degrades over months as snapshot count grows |

### MEDIUM — Data Integrity

| # | Issue | File:Line | Impact |
|---|-------|-----------|--------|
| B4 | `health.py:check_staleness()` calls `datetime.fromisoformat()` on untrusted health_state data without guard | `health.py:~160` | Dashboard `/api/health` crash on corrupt timestamp |

### LOW — Code Quality

| # | Issue | File:Line | Impact |
|---|-------|-----------|--------|
| B5 | Dead `oscillator_trend` correlation group in signal_engine — oscillators always HOLD | `signal_engine.py` | Misleading config |
| B6 | `__import__("json")` inline in metals_cross_asset.py | `signals/metals_cross_asset.py:139` | Code smell |
| B7 | Equity curve annualization uses 365.25 in one place, 365 in another (0.07% diff) | `equity_curve.py:185 vs :228` | Inconsistent but negligible |

## 2. Implementation Batches

### Batch 1: Agent Spawn Race Condition (1 file, ~10 lines)
**Goal:** Fix watchdog-vs-spawn race in agent_invocation.py

**Change:** Move `_agent_start`, `_agent_start_wall`, `_agent_timeout`, `_agent_tier`,
`_agent_reasons` assignments to BEFORE `subprocess.Popen()` call. If Popen fails,
`_agent_proc` stays None and stale state is harmless (watchdog skips None proc).

**Impact:** Zero behavior change on happy path. Eliminates rare kill-on-spawn race.
**Risk:** LOW — only reorders assignments within the same function.

### Batch 2: WMIC → PowerShell Migration (1 file, ~20 lines)
**Goal:** Replace deprecated WMIC with PowerShell Get-CimInstance in subprocess_utils.py

**Change:** Rewrite `kill_orphaned_by_cmdline()` to use
`powershell -NoProfile -Command "Get-CimInstance Win32_Process ..."` instead of WMIC.

**Impact:** Orphan detection works on all Win11 builds.
**Risk:** LOW — PowerShell Get-CimInstance is the documented replacement.

### Batch 3: signal_db JOIN Optimization (1 file, ~40 lines)
**Goal:** Replace O(n²) per-snapshot queries with JOINs in load_entries()

**Change:** Single query with LEFT JOINs across snapshots, ticker_signals, and outcomes.
Post-process into the same dict structure.

**Impact:** Load time goes from O(n²) to O(n). Matters after months of data.
**Risk:** MEDIUM — query structure change, needs careful testing.

### Batch 4: Health Data Guard (1 file, ~5 lines)
**Goal:** Guard datetime.fromisoformat() against corrupt health_state.json

**Change:** Wrap the fromisoformat call in try/except ValueError, return (True, inf, state)
on failure (treat corrupt timestamp as stale — safe behavior).

**Impact:** Dashboard no longer crashes on corrupt health data.
**Risk:** LOW — adds a guard, doesn't change happy path.

### Batch 5: Dead Code & Quality (2 files, ~10 lines)
**Goal:** Remove dead oscillator_trend correlation group, fix inline __import__

**Changes:**
1. `signal_engine.py`: Remove `oscillator_trend` from correlation groups
2. `signals/metals_cross_asset.py`: Replace `__import__("json")` with module-level import

**Impact:** Cleaner code, no behavioral change.
**Risk:** LOW — dead code removal + import style fix.

## 3. Impact Assessment

| Batch | Files | Risk | Testing |
|-------|-------|------|---------|
| 1 | agent_invocation.py | LOW | Existing agent tests + new ordering test |
| 2 | subprocess_utils.py | LOW | Manual verification on Win11 |
| 3 | signal_db.py | MEDIUM | Existing signal_db tests + new JOIN test |
| 4 | health.py | LOW | Existing health tests + new corrupt guard test |
| 5 | signal_engine.py, metals_cross_asset.py | LOW | Existing signal tests |

## 4. Dependency Order

Batches are independent — no ordering constraints.
Implement in numbered order (critical → performance → polish).

## 5. Implementation Results

| Batch | Status | Commit | Notes |
|-------|--------|--------|-------|
| 1 | DONE | `339daf15` | Reordered 5 assignments before Popen. 89/89 agent tests pass. |
| 2 | DONE | `40429468` | PowerShell Get-CimInstance replaces WMIC. Verified on Win11. |
| 3 | DONE | `228f7cd8` | 3 bulk queries + dict reassembly. 18/18 signal_db tests pass. |
| 4 | SKIPPED | — | All 3 fromisoformat calls already have try/except guards. |
| 5 | PARTIAL | `1221ff02` | Fixed `__import__("json")` → module-level import. oscillator_trend verified NOT dead (holds momentum_factors). |

### False Positives Rejected

- **B4**: health.py fromisoformat already guarded at lines 161, 202, 401.
- **B5a**: oscillator_trend correlation group holds active `momentum_factors` signal and participates in meta-cluster dedup. NOT dead code.
- **Agent P0**: risk_management concentration `min(total*pct, cash)` correctly caps allocation at available cash.
- **Agent P1**: grid_fisher `record_fill()` sets `ORDER_FILLED` status, preventing double-count on subsequent iterations.
