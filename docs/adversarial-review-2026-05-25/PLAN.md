# Adversarial Review — 2026-05-25

Goal: full adversarial sweep of finance-analyzer. 8 subsystem partition. 8 background
reviewer subagents. Independent self-review. Cross-critique. Synthesis.

## Subsystem partition

| # | Name | Scope | Reviewer |
|---|------|-------|----------|
| 1 | signals-core | signal_engine, signal_registry, signal_weights, signal_weight_optimizer, accuracy_stats, ic_computation, signal_state_since, signal_history, signal_decay_alert, signal_db, signal_utils, signal_postmortem | pr-review-toolkit:code-reviewer |
| 2 | orchestration | main.py, agent_invocation, autonomous, trigger, trigger_buffer, claude_gate, loop_contract, loop_health, loop_processes, market_timing, shared_state, multi_agent_layer2, perception_gate | pr-review-toolkit:code-reviewer |
| 3 | portfolio-risk | portfolio_mgr, portfolio_validator, risk_management, trade_guards, trade_validation, kelly_sizing, kelly_metals, equity_curve, circuit_breaker, monte_carlo, monte_carlo_risk, cost_model, exit_optimizer, exposure_coach, escalation_gate, escalation_router | pr-review-toolkit:code-reviewer |
| 4 | metals-core | data/metals_loop, data/metals_execution_engine, data/crypto_swing_trader, portfolio/metals_ladder, portfolio/metals_orderbook, portfolio/metals_precompute, portfolio/fin_fish, portfolio/fin_snipe*, portfolio/grid_fisher*, portfolio/fish_*, portfolio/iskbets, portfolio/oil_grid_signal | pr-review-toolkit:code-reviewer |
| 5 | avanza-api | portfolio/avanza/*.py, portfolio/avanza_*.py | pr-review-toolkit:code-reviewer |
| 6 | signals-modules | portfolio/signals/*.py (68 modules) | pr-review-toolkit:code-reviewer |
| 7 | data-external | data_collector, alpha_vantage, fear_greed, futures_data, fx_rates, onchain_data, sentiment, social_sentiment, news_keywords, earnings_calendar, crypto_macro_data, funding_rate, http_retry, api_utils, metals_cross_assets, bert_sentiment, econ_dates, fomc_dates, macro_context | pr-review-toolkit:code-reviewer |
| 8 | infrastructure | file_utils, journal, journal_index, health, logging_config, log_rotation, gpu_gate, process_lock, message_store, message_throttle, telegram_notifications, telegram_poller, alert_budget, subprocess_utils, dashboard/*.py | pr-review-toolkit:code-reviewer |

## Severity scheme
- **P0**: live trading risk, money loss, silent corruption, auth/data leak
- **P1**: persistent bug, accuracy regression, loop crash, race condition
- **P2**: maintainability, dead code that misleads, missing test where it matters
- **P3**: style/cosmetic; documented but not necessarily fixed

## Workflow
1. Create 8 worktrees: `worktrees/review-<sub>` on branch `review/<sub>` from `main`
2. Spawn 8 background reviewer subagents — each scoped to its subsystem file list, writes to `docs/adversarial-review-2026-05-25/<sub>.md`
3. Self-review in parallel: cross-cutting concerns (atomic I/O, race conditions, secrets, signal-flow integrity, headless-vs-interactive)
4. Collect agent reports. Cross-critique each.
5. Write `SYNTHESIS.md` with top P0/P1 findings, agreed-by-multiple-agents items, contested findings, action backlog
6. Commit docs to main, push, clean up worktrees
