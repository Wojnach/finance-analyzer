# PLAN — Adversarial Review (2026-04-19)

## Current Active Plan

Full dual adversarial review of the finance-analyzer codebase.
Partition into 8 subsystems, run independent reviews in parallel,
then cross-critique and synthesize findings.

### 8 Subsystems

| # | Subsystem | Key Files | Focus |
|---|-----------|-----------|-------|
| 1 | signals-core | signal_engine.py, signal_registry.py, accuracy_stats.py, outcome_tracker.py, signal_utils.py, signal_weights.py, ic_computation.py | Consensus logic, gating, weighting, accuracy tracking |
| 2 | orchestration | main.py, agent_invocation.py, trigger.py, market_timing.py, autonomous.py, process_lock.py | Loop lifecycle, agent subprocess, trigger detection |
| 3 | portfolio-risk | portfolio_mgr.py, trade_guards.py, risk_management.py, equity_curve.py, monte_carlo.py, monte_carlo_risk.py, circuit_breaker.py | State management, drawdown, position sizing, guards |
| 4 | metals-core | data/metals_loop.py, exit_optimizer.py, price_targets.py, orb_predictor.py, iskbets.py, fin_snipe.py, fin_snipe_manager.py | Metals trading loop, exit optimization, warrant logic |
| 5 | avanza-api | avanza_session.py, avanza_orders.py, avanza_client.py, avanza_control.py, avanza_order_lock.py, avanza_resilient_page.py, avanza_tracker.py | Session auth, order placement, Playwright integration |
| 6 | signals-modules | portfolio/signals/*.py (34 modules) | Individual signal implementations, edge cases |
| 7 | data-external | data_collector.py, fear_greed.py, sentiment.py, alpha_vantage.py, futures_data.py, onchain_data.py, fx_rates.py, crypto_macro_data.py | Data fetching, API integration, error handling |
| 8 | infrastructure | file_utils.py, http_retry.py, health.py, shared_state.py, claude_gate.py, gpu_gate.py, telegram_notifications.py, message_store.py, journal.py, dashboard/app.py | I/O, caching, notifications, health monitoring |

### Execution Order

1. Launch 8 adversarial review agents in parallel (one per subsystem)
2. Write independent adversarial review (my own analysis)
3. Collect all 8 agent results
4. Cross-critique: compare agent findings vs my findings, both directions
5. Write synthesis document with severity-ranked findings
6. Commit all docs, merge to main, push, clean up worktree

### Review Criteria

Each review evaluates:
- **P1 (Critical)**: Bugs causing data loss, incorrect trades, financial harm
- **P2 (High)**: Race conditions, silent failures, security issues, data corruption
- **P3 (Medium)**: Dead code, inefficiencies, maintainability concerns
- **P4 (Low)**: Style, naming, minor improvements

---

## Previous plans (archived)

- `docs/plans/2026-04-16-gemma4-loop-plan.md` — Gemma 4 E4B integration
- `docs/plans/2026-04-16-accuracy-gating-plan.md` — accuracy gating
  reconfiguration shipped in merge `a739a56`
