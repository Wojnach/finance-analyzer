# Round 3 Dual Adversarial Review
**Date**: 2026-04-08

## Goal

Full codebase adversarial review — Round 3. Find new bugs, verify Round 2 fixes held.
8 parallel Claude code-reviewer subagents (one per subsystem) + synthesis.

## Fixed Since Round 2

- CM1: config_data NameError → FIXED (d534fc1)
- CM2: order["price"] KeyError → FIXED (d534fc1)
- CM3: raw open() in read_signal_data → FIXED (d534fc1)
- CM4: velocity alert double-fire epoch boundary → FIXED (d534fc1)
- C1: ADX cache id(df) → FIXED (c1f5bb6)
- Drawdown all-time peak false -93.6% → FIXED (bfc716f)
- DRY_RUN=False, INITIAL_BUDGET_SEK cash fallback → FIXED (af0ed78)
- KNOWN_WARRANT ob_id 1650161 missing → FIXED (d534fc1)

## Subsystems

| Agent | Files |
|-------|-------|
| metals-core | data/metals_loop.py, data/metals_risk.py, data/metals_swing_trader.py |
| signals-core | portfolio/signal_engine.py, signal_registry.py, accuracy_stats.py |
| orchestration | portfolio/main.py, agent_invocation.py, trigger.py |
| portfolio-risk | portfolio/risk_management.py, trade_guards.py, portfolio_mgr.py |
| avanza-api | portfolio/avanza_session.py, avanza_orders.py |
| data-external | portfolio/data_collector.py, fear_greed.py, sentiment.py, alpha_vantage.py |
| infrastructure | portfolio/file_utils.py, health.py, shared_state.py |
| signals-modules | portfolio/signals/*.py (21 modules) |

## Output

`docs/ADVERSARIAL_REVIEW_3_SYNTHESIS.md`

## Status

- [x] 8 agents running
- [x] Synthesis → docs/ADVERSARIAL_REVIEW_3_SYNTHESIS.md
- [ ] Fix top issues (Tier 1 next)
