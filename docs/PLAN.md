# System-Wide Silent Failure Audit — Fix Plan

## Date: 2026-04-04

## Problem

System-wide audit found 198 silent exception handling patterns, 2 critical
data integrity bugs, and 32 untested modules. Changes to the system silently
break operations because exceptions are swallowed and callers don't know
operations failed.

## Findings Summary

- 16 HIGH-severity silent returns (trading/signal degradation)
- 42 logger.debug instances (invisible in production)
- 140 logger.warning without exc_info (no stack traces)
- 2 CRITICAL data integrity bugs (division by zero, price ≤ 0 accepted)
- 4 HIGH data integrity risks (stale context, PID race, NaN propagation)
- 32 untested modules (coverage gaps)

## Fix Strategy

Focus on the highest-impact fixes that prevent real damage. Don't touch
every warning/debug instance — prioritize money, signals, and data integrity.

### Batch 1: P0 Data Integrity (signal_engine.py, portfolio_mgr.py)
- Guard division by zero in cross-accuracy weighting
- Add price > 0 validation in portfolio value calculation
- Add NaN/Inf guard in accuracy bounds check
- Guard empty signal dict returns

### Batch 2: P0 Silent Returns in Trading Modules
- avanza_control.py: Add logging before returning defaults
- fish_monitor_smart.py: Log Z-score/half-life computation failures
- onchain_data.py: Log token/cache load failures
- journal.py: Log journal load failures
- crypto_macro_data.py: Log history load failures
- llama_server.py: Log health check and start failures
- claude_fundamental.py: Log fundamentals fetch failures
- fish_instrument_finder.py: Log instrument find failures
- iskbets.py: Log parse failures

### Batch 3: P1 Staleness & Data Guards
- Deep context freshness check (silver, gold, oil precompute files)
- PID file fsync in llama_server.py
- NaN guard in signal accuracy weighting

### Batch 4: P1 Logger Upgrades (critical paths only)
- Upgrade logger.debug → logger.warning(exc_info=True) in:
  - agent_invocation.py (4 instances)
  - avanza_session.py (6 instances)
  - health.py (1 instance)
  - main.py heartbeat/health write failures (2 instances)

### Batch 5: Test + Ship
- Run full test suite
- Fix any regressions
- Merge, push, clean up
