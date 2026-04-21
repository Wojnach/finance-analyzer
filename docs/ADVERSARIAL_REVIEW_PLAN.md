# Dual Adversarial Review — 2026-04-21

## Approach

Two independent review tracks run concurrently:
- **Track A (Agents):** 8 specialized code-reviewer agents, one per subsystem
- **Track B (Independent):** Manual deep-read of critical files, focusing on different angles

After both tracks complete, cross-critique in both directions:
- Agent findings reviewed for false positives, missed context
- Independent findings stress-tested against agent coverage gaps

## 8 Subsystem Partition

| # | Subsystem | Scope (key files) |
|---|-----------|-------------------|
| 1 | **signals-core** | signal_engine.py, signal_registry.py, signal_utils.py, signal_weights.py, signal_weight_optimizer.py, signal_db.py, signal_history.py, accuracy_stats.py, accuracy_degradation.py, ticker_accuracy.py, forecast_accuracy.py |
| 2 | **orchestration** | main.py, agent_invocation.py, trigger.py, market_timing.py, autonomous.py, claude_gate.py, config_validator.py, tickers.py, session_calendar.py |
| 3 | **portfolio-risk** | portfolio_mgr.py, risk_management.py, trade_guards.py, trade_validation.py, trade_risk_classifier.py, equity_curve.py, monte_carlo.py, monte_carlo_risk.py, cost_model.py, circuit_breaker.py |
| 4 | **metals-core** | data/metals_loop.py, exit_optimizer.py, price_targets.py, orb_predictor.py, fin_fish.py, fin_snipe.py, fin_snipe_manager.py, fish_instrument_finder.py, silver_precompute.py |
| 5 | **avanza-api** | avanza_session.py, avanza_orders.py, avanza_order_lock.py, avanza_resilient_page.py, avanza_tracker.py, avanza_control.py, avanza_client.py, portfolio/avanza/ |
| 6 | **signals-modules** | portfolio/signals/*.py (36 modules) |
| 7 | **data-external** | data_collector.py, fear_greed.py, sentiment.py, alpha_vantage.py, futures_data.py, fx_rates.py, crypto_macro_data.py, bert_sentiment.py, social_sentiment.py, earnings_calendar.py |
| 8 | **infrastructure** | file_utils.py, http_retry.py, health.py, shared_state.py, telegram_notifications.py, telegram_poller.py, message_store.py, reporting.py, journal.py, digest.py, dashboard/app.py, subprocess_utils.py |

## Severity Scale

- **P0 (Critical):** Data loss, money loss, security vulnerability, silent failure hiding real problems
- **P1 (High):** Logic errors causing wrong trades, race conditions, unhandled edge cases in production paths
- **P2 (Medium):** Code quality, missing validation, dead code, performance issues
- **P3 (Low):** Style, minor improvements, documentation gaps

## Deliverables

1. `docs/adversarial-review/track-a-agents.md` — Agent findings per subsystem
2. `docs/adversarial-review/track-b-independent.md` — Independent manual findings
3. `docs/adversarial-review/cross-critique.md` — Bidirectional critique
4. `docs/adversarial-review/synthesis.md` — Final ranked finding list
