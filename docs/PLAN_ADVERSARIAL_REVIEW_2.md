# Dual Adversarial Review Plan — 2026-04-08 (Round 2)

## Objective
Full dual adversarial review of the finance-analyzer codebase. Two independent reviewers
(Claude direct + 8 parallel subagent reviewers) analyze 8 subsystems, then cross-critique
each other's findings for a final synthesis.

## Subsystem Partition

| # | Subsystem | Key Files | Focus Areas |
|---|-----------|-----------|-------------|
| 1 | **signals-core** | `signal_engine.py`, `signal_registry.py`, `signal_utils.py`, `signal_weights.py`, `signal_weight_optimizer.py`, `signal_db.py`, `accuracy_stats.py`, `outcome_tracker.py`, `meta_learner.py`, `ticker_accuracy.py` | Voting logic, accuracy gates, weight drift, consensus math |
| 2 | **orchestration** | `main.py`, `agent_invocation.py`, `trigger.py`, `market_timing.py`, `autonomous.py`, `loop_contract.py`, `process_lock.py`, `digest.py`, `daily_digest.py` | Loop reliability, crash recovery, trigger logic, race conditions |
| 3 | **portfolio-risk** | `portfolio_mgr.py`, `risk_management.py`, `trade_guards.py`, `trade_validation.py`, `equity_curve.py`, `monte_carlo.py`, `monte_carlo_risk.py`, `kelly_sizing.py`, `circuit_breaker.py`, `exposure_coach.py` | Position sizing, drawdown limits, atomic state, guard bypass |
| 4 | **metals-core** | `data/metals_loop.py`, `metals_cross_assets.py`, `metals_orderbook.py`, `metals_precompute.py`, `exit_optimizer.py`, `price_targets.py`, `orb_predictor.py`, `fin_fish.py`, `fin_snipe.py`, `microstructure.py` | 6500-line god file, fast-tick reliability, exit logic |
| 5 | **avanza-api** | `avanza_session.py`, `avanza_orders.py`, `avanza_client.py`, `portfolio/avanza/` (auth, client, trading, market_data, streaming, types) | Auth lifecycle, order safety, stop-loss API, session expiry |
| 6 | **signals-modules** | `portfolio/signals/` (22 modules: trend, momentum, volatility, smart_money, fibonacci, etc.) | Indicator math, NaN handling, edge cases, signal contract compliance |
| 7 | **data-external** | `data_collector.py`, `fear_greed.py`, `sentiment.py`, `alpha_vantage.py`, `futures_data.py`, `onchain_data.py`, `fx_rates.py`, `forecast_signal.py`, `ministral_signal.py`, `qwen3_signal.py` | API failure modes, rate limits, stale data, fallback logic |
| 8 | **infrastructure** | `file_utils.py`, `http_retry.py`, `health.py`, `claude_gate.py`, `gpu_gate.py`, `shared_state.py`, `telegram_notifications.py`, `message_store.py`, `journal.py`, `reporting.py` | Atomic I/O, lock contention, message delivery, health tracking |

## Review Focus Areas (Adversarial)

For each subsystem, both reviewers look for:
1. **CRITICAL**: Bugs that can lose money, corrupt state, or crash the loop
2. **HIGH**: Silent failures, swallowed exceptions, stale data used for decisions
3. **MEDIUM**: Race conditions, resource leaks, missing validation
4. **LOW**: Code quality, dead code, unclear logic, missing tests

## Execution Order
1. Commit this plan
2. Launch 8 parallel review agents (background)
3. Read key files, write independent Claude review
4. Collect agent results, cross-critique
5. Write synthesis doc
6. Commit all docs to worktree branch, merge to main
7. Push via Windows git, clean up worktree
