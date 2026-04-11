# Adversarial Review Round 5 — Plan (2026-04-11)

## Context
Round 4 (2026-04-09) found 67 findings. Round 1-4 cumulative: 148 findings.
11 P0/P1 fixes shipped today in `fix/queue-2026-04-11` branch (merged to main).

## Still-Open from Prior Rounds
- C6: `check_drawdown()` never called from production code
- C12: `log_portfolio_value` raw `open("a")` in metals_loop.py
- C14: Naked position on stop-loss failure — no retry or fallback
- H17: VWAP cumulative from bar 0 (should reset per session)
- H31: POSITIONS dict shared without lock in metals_loop.py

## Round 5 Objectives
1. Verify all 11 fixes from today's fix queue are correct and complete
2. Check for regressions introduced by the fixes
3. Find NEW bugs that prior rounds missed (deeper analysis)
4. Focus on money-losing bugs and data corruption
5. Dual review: 8 parallel agents + 1 independent author review
6. Cross-critique in both directions
7. Synthesize all findings

## 8 Subsystem Partition (same as Rounds 1-4 for consistency)

1. **signals-core** — signal_engine, signal_registry, signal_utils, signal_weights, accuracy_stats, outcome_tracker
2. **orchestration** — main.py, agent_invocation, trigger, market_timing, autonomous, health, claude_gate, gpu_gate
3. **portfolio-risk** — portfolio_mgr, risk_management, trade_guards, equity_curve, monte_carlo, circuit_breaker, kelly
4. **metals-core** — metals_loop, metals_swing_trader, metals_execution_engine, exit_optimizer, price_targets, orb_predictor, fin_fish, fin_snipe
5. **avanza-api** — avanza_session, avanza_orders, avanza_client, avanza_control, avanza_tracker
6. **signals-modules** — all 26 modules in portfolio/signals/
7. **data-external** — data_collector, alpha_vantage, fear_greed, sentiment, onchain_data, futures_data, forecast_signal, ministral, qwen3, ml, microstructure
8. **infrastructure** — file_utils, http_retry, shared_state, telegram, journal, dashboard, reporting, digest

## Execution
1. Commit this plan
2. Launch 8 adversarial review agents in parallel
3. Write independent author review concurrently
4. Cross-critique
5. Synthesize
6. Commit, merge, push
