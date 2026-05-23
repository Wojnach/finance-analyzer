# Improvement Plan — Auto-Session 2026-05-23

Created: 2026-05-23
Branch: `improve/auto-session-2026-05-23`
Prior sessions: 7 auto-sessions (05-04 through 05-22), adversarial reviews (05-21, 05-22).

## Context

6 exploration/audit agents ran in parallel covering: orchestration (main.py, agent_invocation.py, trigger.py, market_timing.py, health.py, autonomous.py), signal system (signal_engine.py, signal_registry.py, accuracy_stats.py, outcome_tracker.py), data layer (file_utils.py, shared_state.py, http_retry.py, circuit_breaker.py, data_collector.py, portfolio_mgr.py, trade_guards.py, risk_management.py, equity_curve.py, fx_rates.py), dashboard (app.py, auth.py, cf_access.py), metals subsystem (metals_loop.py, grid_fisher.py, avanza_session.py, avanza_orders.py, exit_optimizer.py).

Baseline: 10,210 passing, 36 pre-existing failures (all pre-existing on main).

## Bugs & Fixes

### Batch 1: Signal Accuracy (affects trading decisions directly)

| ID | File(s) | Bug | Impact | Fix |
|----|---------|-----|--------|-----|
| B1 | signal_engine.py:4162-4173 | Utility boost can rescue signals below accuracy gate. A signal at 44% acc with positive avg_return gets boosted to 48.4%, passing the 47% gate. | Noise signals trade. E.g. ministral on XAG-USD (33.8% recent) could be rescued. | Only apply utility boost to signals already passing the raw gate. |
| B2 | signal_engine.py:3989-3990,3004-3037 | Unanimity penalty (Stage 5) uses pre-persistence voter counts but consensus uses post-persistence. 8 pre-persistence BUYs → 0.6x penalty, but only 2 actually voted → overcorrection. | Systematically suppresses legitimate metals signals. | Compute `_buy_count`/`_sell_count` from `consensus_votes` (post-persistence), not `votes`. |
| B3 | signal_engine.py:2848 | `_compute_adx` crashes with KeyError on DataFrames without 'high' column. Tests provide DataFrames with only 'close'. | Test failures (3 pre-existing). Production safe because `generate_signal` always has full OHLCV. | Guard `_compute_adx`: return None if required columns missing. |

### Batch 2: Agent Invocation Safety

| ID | File(s) | Bug | Impact | Fix |
|----|---------|-----|--------|-----|
| B4 | multi_agent_layer2.py:97-119 | `cleanup_reports()` defined but never called. Specialist report files persist across invocations — synthesis agent reads stale specialist data from previous runs. | Stale analysis feeds trading decisions. | Call `cleanup_reports()` at start of `launch_specialists`. |
| B5 | multi_agent_layer2.py:178-184 | Specialist subprocesses don't pass `stdin=subprocess.DEVNULL`. Three specialists share parent stdin handle. | Potential deadlock on interactive sessions. | Add `stdin=subprocess.DEVNULL` to specialist Popen. |
| B6 | agent_invocation.py:1533 | Stack overflow auto-disable uses `==` instead of `>=`. If counter exceeds threshold, alert never fires but invocations still blocked. | Silent Layer 2 disable with no alert. | Change `==` to `>=`. |
| B7 | agent_invocation.py:429 | `_build_decision_feedback` loads entire journal unboundedly via `load_jsonl(JOURNAL_FILE)`. | O(N) memory/disk on hot path, growing indefinitely. | Replace with `load_jsonl_tail(JOURNAL_FILE, max_entries=200)`. |

### Batch 3: Trigger & Data Hygiene

| ID | File(s) | Bug | Impact | Fix |
|----|---------|-----|--------|-----|
| B8 | trigger.py | `flip_cooldowns` and `sustained_counts` never pruned for removed tickers — unbounded growth in trigger_state.json. | State file grows forever, slow reads. | Add pruning in `_save_state` matching existing `triggered_consensus` pattern. |
| B9 | futures_data.py | No circuit breaker for Binance FAPI calls. API outage → 18 retry calls per 5-min cache window (6 metrics × 3 retries). | Wastes rate limit budget during outage, delays loop. | Add shared `CircuitBreaker("binance_fapi")` gating all futures API calls. |

### Batch 4: Test Fixes & Correctness

| ID | File(s) | Bug | Impact | Fix |
|----|---------|-----|--------|-----|
| B10 | tests/test_signal_engine_circuit_breaker.py | Tests pass DataFrames without 'high'/'low' columns to `apply_confidence_penalties` → KeyError in `_compute_adx`. | 3 pre-existing test failures. | Fix tests to include full OHLCV columns, AND add guard in `_compute_adx`. |
| B11 | agent_invocation.py:737-741,947-950 | Config loaded multiple times in `invoke_agent` — TOCTOU if config file changes mid-invocation. | Config inconsistency between tier check and multi-agent check. | Load config once at top of `invoke_agent`, pass through. |

## Skipped / Deferred

- **P0 #0 (prior session): Barrier-blind stops** — 10+ files, real-money paths. Still deferred: MANUAL REVIEW needed.
- **P0 #1 (prior session): Layer 2 Edit/Write/Bash tools** — security design session needed.
- **Signal engine unanimity pre/post persistence (B2)** — marked for this session but if too risky to change at 4400-line scale, defer with TODO.
- **exit_optimizer.py trading minutes mismatch** (820 vs canonical 810) — minor impact on MC simulation, defer.
- **IC data cache mutation outside lock** (signal_engine.py:2127-2131) — GIL-protected on CPython, low risk.
- **Regime gate REPLACE semantics validation assertion** — design improvement, not a bug.
- **Dashboard rate limiting on auth** — security enhancement, not a bug.
- **load_equity_curve full JSONL scan** — performance only, defer to SQLite migration.

## Implementation Order

### Batch 1 (B1, B2, B3): signal_engine.py — Signal accuracy fixes
Files: `portfolio/signal_engine.py`
Impact: Directly affects trading decisions. B1 lets noise signals through. B2 suppresses metals signals.

### Batch 2 (B4, B5, B6, B7): agent_invocation.py + multi_agent_layer2.py — Invocation safety
Files: `portfolio/agent_invocation.py`, `portfolio/multi_agent_layer2.py`
Impact: Stale data in decisions (B4), deadlock risk (B5), silent disable (B6), memory growth (B7).

### Batch 3 (B8, B9): trigger.py + futures_data.py — Data hygiene
Files: `portfolio/trigger.py`, `portfolio/futures_data.py`
Impact: State growth (B8), API waste (B9).

### Batch 4 (B10, B11): Test fixes + config consistency
Files: `tests/test_signal_engine_circuit_breaker.py`, `portfolio/agent_invocation.py`
Impact: Fix 3+ pre-existing test failures.
