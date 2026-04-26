# Adversarial Review #8 — Plan (2026-04-26)

## Objective
Full dual adversarial code review: 8 subsystems, parallel agent reviewers + independent manual review, cross-critique, synthesis.

## 8 Subsystems

| # | Subsystem | Key files | ~LOC |
|---|-----------|-----------|------|
| 1 | signals-core | signal_engine, signal_registry, accuracy_stats, outcome_tracker, signal_weights, ic_computation, ticker_accuracy | ~3K |
| 2 | orchestration | main, agent_invocation, trigger, market_timing, autonomous, multi_agent_layer2, loop_contract | ~3K |
| 3 | portfolio-risk | portfolio_mgr, trade_guards, risk_management, equity_curve, monte_carlo*, kelly_*, circuit_breaker | ~3K |
| 4 | metals-core | metals_loop, exit_optimizer, price_targets, orb_predictor, iskbets, fin_snipe*, fin_fish, microstructure* | ~4K |
| 5 | avanza-api | avanza_session, avanza_orders, avanza_client, avanza_control, avanza_resilient_page, avanza_tracker | ~2K |
| 6 | signals-modules | portfolio/signals/*.py (40 modules) | ~6K |
| 7 | data-external | data_collector, fear_greed, sentiment, alpha_vantage, futures_data, onchain_data, fx_rates, funding_rate | ~3K |
| 8 | infrastructure | file_utils, http_retry, health, shared_state, telegram_*, journal, dashboard, golddigger, elongir | ~5K |

## Review Criteria
1. Bugs — logic errors, race conditions, off-by-one, dead code paths
2. Security — credential leaks, injection, unsafe deserialization
3. Reliability — silent failures, missing retries, crash paths
4. Data integrity — non-atomic writes, stale reads, corruption
5. Performance — unnecessary I/O, O(n²), memory leaks, thread contention
6. Architecture — coupling, god functions, circular deps
7. Correctness — signal math, wrong formulas, timezone bugs

## Execution
1. ✅ Commit this plan
2. Launch 8 parallel code-reviewer agents
3. Simultaneously write independent manual review (read key files)
4. Collect agent results → docs/adversarial-review/agent-{subsystem}.md
5. Cross-critique both directions
6. Write SYNTHESIS.md
7. Commit all, merge main, push, clean up
