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
- [ ] Fix Tier 1 issues (branch: fix/tier1-adversarial-r3)

## Tier 1 Fixes (trivial, high impact)

1. MIN_TRADE_SEK 500→1000 — `data/metals_swing_config.py:59`
2. NFP Good Friday date(2026,4,3)→date(2026,4,2) — `portfolio/econ_dates.py:61`
3. _silver_reset_session() never called — `data/metals_loop.py:6063`
4. metals_context.json raw open() → atomic_write_json — `data/metals_loop.py:5078`
5. SwingTrader _save_state raw open("w") → atomic_write_json — `data/metals_swing_trader.py:110`
6. get_buying_power() wrong JSON keys — `portfolio/avanza_session.py:302`
7. StopLossResult.from_api: stoplossOrderId key missing — `portfolio/avanza/types.py:212`
8. cvar_99_sek key missing from compute_portfolio_var — `portfolio/monte_carlo_risk.py:503`
