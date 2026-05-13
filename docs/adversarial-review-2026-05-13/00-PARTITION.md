# Adversarial Review 2026-05-13 — Subsystem Partition

Second dual adversarial review of finance-analyzer (previous: 2026-05-11). Codebase split into 8 disjoint subsystems for parallel dual review (Codex + Claude). Each subsystem is reviewed against an empty baseline so the entire current contents are surfaced to the reviewer.

Baseline: `main@59ad394e` (Merge fix/grid-fisher-buying-power).

## Coverage map

| # | Subsystem        | Files (canonical)                                                                                                                                                                                                                  |
|---|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | signals-core     | `portfolio/signal_engine.py`, `portfolio/signal_registry.py`, `portfolio/signal_utils.py`, `portfolio/signal_weights.py`, `portfolio/signal_weight_optimizer.py`, `portfolio/signal_history.py`, `portfolio/signal_state_since.py`, `portfolio/signal_decay_alert.py`, `portfolio/signal_postmortem.py`, `portfolio/signal_db.py`, `portfolio/accuracy_stats.py`, `portfolio/accuracy_degradation.py`, `portfolio/ticker_accuracy.py`, `portfolio/outcome_tracker.py`, `portfolio/forecast_accuracy.py`, `portfolio/ic_computation.py`, `portfolio/train_signal_weights.py`, `portfolio/linear_factor.py`, `portfolio/feature_normalizer.py`, `portfolio/short_horizon.py` |
| 2 | orchestration    | `portfolio/main.py`, `portfolio/agent_invocation.py`, `portfolio/autonomous.py`, `portfolio/trigger.py`, `portfolio/market_timing.py`, `portfolio/claude_gate.py`, `portfolio/gpu_gate.py`, `portfolio/health.py`, `portfolio/alert_budget.py`, `portfolio/llm_prewarmer.py`, `portfolio/llm_calibration.py`, `portfolio/llm_batch.py`, `portfolio/llm_outcome_backfill.py`, `portfolio/llm_probability_log.py`, `portfolio/llama_server.py`, `portfolio/multi_agent_layer2.py`, `portfolio/perception_gate.py`, `portfolio/focus_analysis.py`, `portfolio/reporting.py`, `portfolio/journal.py`, `portfolio/journal_index.py`, `portfolio/telegram_notifications.py`, `portfolio/telegram_poller.py`, `portfolio/digest.py`, `portfolio/daily_digest.py`, `portfolio/weekly_digest.py`, `portfolio/reflection.py`, `portfolio/regime_alerts.py`, `portfolio/analyze.py`, `portfolio/bigbet.py`, `portfolio/prophecy.py`, `portfolio/qwen3_signal.py`, `portfolio/circuit_breaker.py`, `portfolio/cumulative_tracker.py`, `portfolio/decision_outcome_tracker.py`, `portfolio/loop_contract.py` |
| 3 | portfolio-risk   | `portfolio/portfolio_mgr.py`, `portfolio/portfolio_validator.py`, `portfolio/trade_guards.py`, `portfolio/trade_validation.py`, `portfolio/trade_risk_classifier.py`, `portfolio/risk_management.py`, `portfolio/monte_carlo.py`, `portfolio/monte_carlo_risk.py`, `portfolio/equity_curve.py`, `portfolio/exit_optimizer.py`, `portfolio/kelly_sizing.py`, `portfolio/kelly_metals.py`, `portfolio/exposure_coach.py`, `portfolio/warrant_portfolio.py`, `portfolio/cost_model.py`, `portfolio/instrument_profile.py`, `portfolio/stats.py`, `portfolio/iskbets.py`, `portfolio/strategies/` |
| 4 | metals-core      | `data/metals_loop.py`, `data/crypto_loop.py`, `data/oil_loop.py`, `portfolio/grid_fisher.py`, `portfolio/grid_fisher_config.py`, `portfolio/grid_tiers.py`, `portfolio/oil_grid_signal.py`, `portfolio/fin_fish.py`, `portfolio/fin_fish_manager.py`, `portfolio/fin_snipe.py`, `portfolio/fin_snipe_manager.py`, `portfolio/fish_instrument_finder.py`, `portfolio/fish_monitor_smart.py`, `portfolio/gold_precompute.py`, `portfolio/silver_precompute.py`, `portfolio/oil_precompute.py`, `portfolio/crypto_precompute.py`, `portfolio/price_targets.py`, `portfolio/orb_predictor.py`, `portfolio/orb_backtest.py`, `portfolio/orb_postmortem.py`, `portfolio/mstr_loop/`, `portfolio/elongir/`, `portfolio/golddigger/` |
| 5 | avanza-api       | `portfolio/avanza_session.py`, `portfolio/avanza_client.py`, `portfolio/avanza_orders.py`, `portfolio/avanza_tracker.py`, `portfolio/avanza_control.py`, `portfolio/avanza_account_check.py`, `portfolio/avanza_order_lock.py`, `portfolio/avanza_resilient_page.py`, `portfolio/avanza/` |
| 6 | signals-modules  | `portfolio/signals/` (all .py files, ~50)                                                                                                                                                                                          |
| 7 | data-external    | `portfolio/data_collector.py`, `portfolio/fear_greed.py`, `portfolio/sentiment.py`, `portfolio/bert_sentiment.py`, `portfolio/alpha_vantage.py`, `portfolio/futures_data.py`, `portfolio/funding_rate.py`, `portfolio/fx_rates.py`, `portfolio/onchain_data.py`, `portfolio/news_keywords.py`, `portfolio/social_sentiment.py`, `portfolio/crypto_macro_data.py`, `portfolio/crypto_scheduler.py`, `portfolio/earnings_calendar.py`, `portfolio/econ_dates.py`, `portfolio/fomc_dates.py`, `portfolio/seasonality.py`, `portfolio/seasonality_updater.py`, `portfolio/session_calendar.py`, `portfolio/price_source.py`, `portfolio/http_retry.py`, `portfolio/api_utils.py`, `portfolio/data_refresh.py`, `portfolio/forecast_signal.py`, `portfolio/indicators.py`, `portfolio/metals_orderbook.py`, `portfolio/microstructure.py`, `portfolio/microstructure_state.py`, `portfolio/metals_cross_assets.py`, `portfolio/tickers.py` |
| 8 | infrastructure   | `portfolio/file_utils.py`, `portfolio/shared_state.py`, `portfolio/process_lock.py`, `portfolio/subprocess_utils.py`, `portfolio/config_validator.py`, `portfolio/notification_text.py`, `portfolio/message_store.py`, `portfolio/shadow_registry.py`, `portfolio/vector_memory.py`, `portfolio/backtester.py`, `dashboard/`, `scripts/check_critical_errors.py`, `scripts/fix_agent_dispatcher.py`, `conftest.py` |

## Empty-baseline branch convention

Per subsystem `S` an orphan branch `adv-2026-05-13/baseline-<S>` is created with the **only commit** being a snapshot of `main@59ad394e` with every file listed for subsystem `S` removed. Running:

```
codex review --base adv-2026-05-13/baseline-<S>
```

from `adv-review-2026-05-13` (sibling of main@59ad394e) then surfaces every file in subsystem `S` as a clean "added" diff, scoping the reviewer to that subsystem only.

## Workflow

1. Worktree: `Q:/fa-adv-2026-05-13` checked out at current `main` HEAD (`59ad394e`), on branch `adv-review-2026-05-13`.
2. For each subsystem, build orphan branch with subsystem files removed (one commit).
3. Launch `codex review --base <branch>` as background process per subsystem; redirect output to `_codex_out/<n>-<S>.txt`.
4. In parallel, Claude general-purpose agents write independent adversarial reviews to `<n>-<S>-claude.md`.
5. After both complete, extract codex final-report (or prose) to `<n>-<S>-codex.md`; cross-critique to `<n>-<S>-cross.md`.
6. Final synthesis to `99-SYNTHESIS.md`.
