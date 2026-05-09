# /fgl 2026-05-09 — Subsystem Partition Map

Authoritative file list for each of the eight review subsystems. Both
codex and claude reviewers operate on these exact lists. The partition is
mutually exclusive: every reviewed file appears in exactly one subsystem.

## signals-core (13)
```
portfolio/signal_engine.py
portfolio/signal_registry.py
portfolio/signal_utils.py
portfolio/signal_weights.py
portfolio/signal_db.py
portfolio/signal_history.py
portfolio/signal_postmortem.py
portfolio/signal_weight_optimizer.py
portfolio/accuracy_stats.py
portfolio/accuracy_degradation.py
portfolio/outcome_tracker.py
portfolio/forecast_accuracy.py
portfolio/ticker_accuracy.py
```

## orchestration (15)
```
portfolio/main.py
portfolio/agent_invocation.py
portfolio/trigger.py
portfolio/market_timing.py
portfolio/multi_agent_layer2.py
portfolio/loop_contract.py
portfolio/autonomous.py
portfolio/claude_gate.py
portfolio/config_validator.py
portfolio/health.py
portfolio/perception_gate.py
portfolio/reflection.py
portfolio/reporting.py
portfolio/session_calendar.py
portfolio/tickers.py
```

## portfolio-risk (13)
```
portfolio/portfolio_mgr.py
portfolio/portfolio_validator.py
portfolio/risk_management.py
portfolio/equity_curve.py
portfolio/monte_carlo.py
portfolio/monte_carlo_risk.py
portfolio/kelly_sizing.py
portfolio/circuit_breaker.py
portfolio/trade_guards.py
portfolio/trade_validation.py
portfolio/trade_risk_classifier.py
portfolio/cost_model.py
portfolio/warrant_portfolio.py
```

## metals-core (16)
```
data/metals_loop.py
portfolio/metals_orderbook.py
portfolio/metals_cross_assets.py
portfolio/metals_ladder.py
portfolio/metals_precompute.py
portfolio/exit_optimizer.py
portfolio/price_targets.py
portfolio/orb_predictor.py
portfolio/orb_postmortem.py
portfolio/silver_precompute.py
portfolio/iskbets.py
portfolio/fin_fish.py
portfolio/fin_snipe.py
portfolio/fin_snipe_manager.py
portfolio/microstructure.py
portfolio/microstructure_state.py
```

## avanza-api (18)
```
portfolio/avanza/__init__.py
portfolio/avanza/account.py
portfolio/avanza/auth.py
portfolio/avanza/client.py
portfolio/avanza/market_data.py
portfolio/avanza/scanner.py
portfolio/avanza/search.py
portfolio/avanza/streaming.py
portfolio/avanza/tick_rules.py
portfolio/avanza/trading.py
portfolio/avanza/types.py
portfolio/avanza_session.py
portfolio/avanza_orders.py
portfolio/avanza_client.py
portfolio/avanza_control.py
portfolio/avanza_order_lock.py
portfolio/avanza_resilient_page.py
portfolio/avanza_tracker.py
```

## signals-modules (49)
```
portfolio/signals/__init__.py
portfolio/signals/calendar_seasonal.py
portfolio/signals/candlestick.py
portfolio/signals/claude_fundamental.py
portfolio/signals/complexity_gap_regime.py
portfolio/signals/copper_gold_ratio.py
portfolio/signals/cot_positioning.py
portfolio/signals/credit_spread.py
portfolio/signals/cross_asset_tsmom.py
portfolio/signals/crypto_cross_asset.py
portfolio/signals/crypto_evrp.py
portfolio/signals/crypto_macro.py
portfolio/signals/drift_regime_gate.py
portfolio/signals/dxy_cross_asset.py
portfolio/signals/econ_calendar.py
portfolio/signals/fibonacci.py
portfolio/signals/forecast.py
portfolio/signals/futures_basis.py
portfolio/signals/futures_flow.py
portfolio/signals/gold_real_yield_paradox.py
portfolio/signals/hash_ribbons.py
portfolio/signals/heikin_ashi.py
portfolio/signals/hurst_regime.py
portfolio/signals/intraday_seasonality.py
portfolio/signals/macro_regime.py
portfolio/signals/mahalanobis_turbulence.py
portfolio/signals/mean_reversion.py
portfolio/signals/metals_cross_asset.py
portfolio/signals/momentum.py
portfolio/signals/momentum_factors.py
portfolio/signals/network_momentum.py
portfolio/signals/news_event.py
portfolio/signals/orderbook_flow.py
portfolio/signals/oscillators.py
portfolio/signals/ovx_metals_spillover.py
portfolio/signals/realized_skewness.py
portfolio/signals/residual_pair_reversion.py
portfolio/signals/shannon_entropy.py
portfolio/signals/smart_money.py
portfolio/signals/statistical_jump_regime.py
portfolio/signals/structure.py
portfolio/signals/treasury_risk_rotation.py
portfolio/signals/trend.py
portfolio/signals/vix_term_structure.py
portfolio/signals/vol_ratio_regime.py
portfolio/signals/volatility.py
portfolio/signals/volume_flow.py
portfolio/signals/williams_vix_fix.py
portfolio/signals/xtrend_equity_spillover.py
```

## data-external (14)
```
portfolio/data_collector.py
portfolio/futures_data.py
portfolio/onchain_data.py
portfolio/fx_rates.py
portfolio/fear_greed.py
portfolio/alpha_vantage.py
portfolio/news_keywords.py
portfolio/sentiment.py
portfolio/social_sentiment.py
portfolio/bert_sentiment.py
portfolio/earnings_calendar.py
portfolio/macro_context.py
portfolio/crypto_macro_data.py
portfolio/market_health.py
```

## infrastructure (18)
```
portfolio/file_utils.py
portfolio/shared_state.py
portfolio/log_rotation.py
portfolio/logging_config.py
portfolio/telegram_notifications.py
portfolio/telegram_poller.py
portfolio/message_store.py
portfolio/message_throttle.py
portfolio/alert_budget.py
portfolio/prophecy.py
portfolio/journal.py
portfolio/journal_index.py
portfolio/gpu_gate.py
portfolio/process_lock.py
portfolio/subprocess_utils.py
portfolio/api_utils.py
portfolio/http_retry.py
dashboard/app.py
```
