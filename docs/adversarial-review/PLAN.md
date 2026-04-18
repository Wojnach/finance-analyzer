# Dual Adversarial Review Plan — 2026-04-18

## Methodology
Two independent reviewers examine each subsystem:
1. **Reviewer A (Claude agents)** — 8 parallel subagents, each doing deep adversarial analysis
2. **Reviewer B (Codex CLI)** — OpenAI Codex `review` command with adversarial prompts

After both complete, cross-critique in both directions:
- Claude critiques Codex findings (false positives? missed issues?)
- Codex critiques Claude findings (same)

Final synthesis merges all validated findings.

## 8 Subsystems

### 1. signals-core
Signal engine, registry, voting, accuracy, weights, optimization.
Files: `signal_engine.py`, `signal_registry.py`, `signal_utils.py`, `signal_weights.py`,
`signal_weight_optimizer.py`, `signal_db.py`, `signal_history.py`, `signal_postmortem.py`,
`accuracy_stats.py`, `accuracy_degradation.py`, `ticker_accuracy.py`, `outcome_tracker.py`,
`train_signal_weights.py`

### 2. orchestration
Main loop, Layer 2 invocation, triggers, scheduling, process management.
Files: `main.py`, `agent_invocation.py`, `trigger.py`, `market_timing.py`, `autonomous.py`,
`multi_agent_layer2.py`, `process_lock.py`, `claude_gate.py`, `session_calendar.py`,
`crypto_scheduler.py`

### 3. portfolio-risk
Portfolio state, risk limits, trade guards, equity curve, Monte Carlo, circuit breakers.
Files: `portfolio_mgr.py`, `portfolio_validator.py`, `risk_management.py`, `trade_guards.py`,
`trade_risk_classifier.py`, `trade_validation.py`, `equity_curve.py`, `circuit_breaker.py`,
`cost_model.py`, `monte_carlo.py`, `monte_carlo_risk.py`, `warrant_portfolio.py`, `exposure_coach.py`

### 4. metals-core
Metals loop, swing trader, execution engine, fish/snipe, ORB, exit optimizer.
Files: `data/metals_loop.py`, `data/metals_swing_trader.py`, `data/metals_swing_config.py`,
`data/metals_shared.py`, `data/metals_signal_tracker.py`, `data/metals_risk.py`,
`data/metals_execution_engine.py`, `data/metals_avanza_helpers.py`, `data/metals_llm.py`,
`data/metals_history_fetch.py`, `data/metals_warrant_refresh.py`,
`portfolio/fin_fish.py`, `portfolio/fin_snipe.py`, `portfolio/fin_snipe_manager.py`,
`portfolio/fish_instrument_finder.py`, `portfolio/fish_monitor_smart.py`,
`portfolio/exit_optimizer.py`, `portfolio/price_targets.py`, `portfolio/orb_predictor.py`

### 5. avanza-api
Avanza client, auth, orders, session, tracking, control.
Files: `portfolio/avanza/*.py`, `portfolio/avanza_client.py`, `portfolio/avanza_control.py`,
`portfolio/avanza_orders.py`, `portfolio/avanza_order_lock.py`, `portfolio/avanza_resilient_page.py`,
`portfolio/avanza_session.py`, `portfolio/avanza_tracker.py`

### 6. signals-modules
All 30+ signal plugin modules in `portfolio/signals/`, plus standalone signal providers.
Files: `portfolio/signals/*.py`, `portfolio/forecast_signal.py`, `portfolio/funding_rate.py`,
`portfolio/qwen3_signal.py`, `portfolio/bert_sentiment.py`, `portfolio/sentiment.py`,
`portfolio/fear_greed.py`, `portfolio/onchain_data.py`, `portfolio/forecast_accuracy.py`

### 7. data-external
Data collection, external APIs, price sources.
Files: `portfolio/data_collector.py`, `portfolio/data_refresh.py`, `portfolio/alpha_vantage.py`,
`portfolio/crypto_macro_data.py`, `portfolio/futures_data.py`, `portfolio/fx_rates.py`,
`portfolio/earnings_calendar.py`, `portfolio/econ_dates.py`, `portfolio/fomc_dates.py`,
`portfolio/social_sentiment.py`, `portfolio/news_keywords.py`, `portfolio/price_source.py`,
`portfolio/oil_precompute.py`, `portfolio/silver_precompute.py`

### 8. infrastructure
File I/O, HTTP, health, shared state, Telegram, reporting, journals, dashboard.
Files: `portfolio/file_utils.py`, `portfolio/http_retry.py`, `portfolio/health.py`,
`portfolio/shared_state.py`, `portfolio/gpu_gate.py`, `portfolio/telegram_notifications.py`,
`portfolio/telegram_poller.py`, `portfolio/message_store.py`, `portfolio/reporting.py`,
`portfolio/journal.py`, `portfolio/prophecy.py`, `portfolio/digest.py`, `portfolio/daily_digest.py`,
`portfolio/weekly_digest.py`, `portfolio/config_validator.py`, `portfolio/api_utils.py`,
`portfolio/subprocess_utils.py`, `portfolio/tickers.py`, `portfolio/notification_text.py`,
`dashboard/app.py`
